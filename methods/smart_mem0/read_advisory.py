"""Question-owned retrieval plus one grounded answer-or-search controller."""

import json
import re
from itertools import zip_longest


ANSWER_OR_SEARCH_PROMPT = """You are the only semantic controller in a grounded memory system.

Use QUESTION, BASE EVIDENCE, STORED RELATIONS, and optional CALLER INSTRUCTIONS.
BASE EVIDENCE and STORED RELATIONS are factual evidence. CALLER INSTRUCTIONS may
control style or output format but are not factual evidence.

You may interpret paraphrases, resolve references, compare evidence, and combine
multiple displayed premises. Do not introduce entity-specific factual premises from
general knowledge. If every material factual premise needed for a complete answer is
present, return ANSWER. If any material premise is missing, uncertain, or needs more
evidence, return SEARCH. When uncertain, prefer SEARCH.

Return exactly one JSON object in one of these shapes:

ANSWER:
{
  "decision": "ANSWER",
  "answer": "<complete final answer in the requested format>",
  "supports": [
    {"memory_id": "<displayed id>", "quote": "<exact contiguous quote from that memory>"}
  ]
}

SEARCH:
{
  "decision": "SEARCH",
  "queries": ["<short search probe>", "..."]
}

Rules:
- ANSWER must rely only on displayed memories/relations for factual premises.
- ANSWER must include at least one support. Cite only displayed memory ids.
- Each support quote must be copied exactly from a displayed memory field.
- SEARCH may contain at most 4 concise, distinct probes aimed at missing evidence.
- Search probes are retrieval hints, never facts or proposed proof.
- Do not emit benchmark types, projections, option-set types, retrieval budgets,
  confidence scores, proof labels, support verdicts, or hidden chain-of-thought.
"""


class AdvisoryReadMixin:
    """Compatibility name for the active grounded answer-or-search READ controller."""

    BASE_WORLD_LIMIT = 10
    CANDIDATE_WORLD_LIMIT = 16
    FINAL_CONTEXT_LIMIT = 16

    @staticmethod
    def _ad_round_robin(groups, limit):
        output, seen = [], set()
        for row in zip_longest(*groups):
            for memory in row:
                if memory and memory["id"] not in seen:
                    output.append(memory)
                    seen.add(memory["id"])
                    if len(output) >= limit:
                        return output
        return output

    def _ad_search(self, question, top_k=16):
        # Historical versions remain eligible; semantic selection belongs to the LLM.
        return self._hybrid_search(question, top_k=top_k)

    def _ad_exact_surfaces(self, question):
        """Extract only literal/high-information retrieval surfaces; never proof."""
        surfaces = re.findall(r'["“「]([^"”」]+)["”」]', question)
        surfaces += re.findall(
            r"(?<!\w)\d+(?:[.,:/-]\d+)*(?:\s*[%\w]+(?:/\w+)?)?", question
        )
        texts = [self._memory_text(m).casefold() for m in self._memories]
        candidates = set()
        for memory in self._memories:
            for value in [memory.get("object_anchor"), *(memory.get("entities") or [])]:
                if isinstance(value, str) and len(value.strip()) >= 3:
                    surface = value.replace("_", " ").strip()
                    if re.search(
                        r"(?<!\w)" + re.escape(surface) + r"(?!\w)", question, re.I
                    ):
                        candidates.add(surface)
        for surface in sorted(candidates):
            pattern = re.compile(
                r"(?<!\w)" + re.escape(surface.casefold()) + r"(?!\w)"
            )
            df = sum(bool(pattern.search(text)) for text in texts)
            if df / max(1, len(texts)) <= 0.2:
                surfaces.append(surface)
        return list(dict.fromkeys(s for s in surfaces if s.strip()))[:16]

    def _ad_base_world(self, question):
        """Question-owned lexical+dense+literal acquisition before any LLM output."""
        self._refresh_index()
        ranked = self._ad_search(question, max(16, len(self._memories)))
        lexical = sorted(
            (m for m in ranked if m.get("_bm25_rank")),
            key=lambda m: (m["_bm25_rank"], m["id"]),
        )[:4]
        dense = sorted(
            (m for m in ranked if m.get("_dense_rank")),
            key=lambda m: (m["_dense_rank"], m["id"]),
        )[:4]

        surfaces = self._ad_exact_surfaces(question)
        exact = []
        for memory in self._memories:
            text = self._memory_text(memory)
            count = sum(
                bool(re.search(r"(?<!\w)" + re.escape(s) + r"(?!\w)", text, re.I))
                for s in surfaces
            )
            if count:
                exact.append((count, memory["id"], self._snapshot(memory)))
        exact = [m for _, _, m in sorted(exact, key=lambda x: (-x[0], x[1]))[:2]]

        base = self._ad_round_robin(
            [lexical, dense, exact],
            self.BASE_WORLD_LIMIT,
        )
        return base, {
            "base_world_ids": [m["id"] for m in base],
            "base_world_limit": self.BASE_WORLD_LIMIT,
            "base_lexical_ids": [m["id"] for m in lexical],
            "base_dense_ids": [m["id"] for m in dense],
            "base_exact_ids": [m["id"] for m in exact],
            "question_exact_surfaces": surfaces,
        }

    @staticmethod
    def _ad_clean_texts(values, limit):
        if not isinstance(values, list):
            return []
        return list(
            dict.fromkeys(
                value.strip()
                for value in values
                if isinstance(value, str) and value.strip() and len(value) <= 320
            )
        )[:limit]

    def _ad_normalize(self, raw, question=None):
        """Normalize controller output with a safe SEARCH default."""
        raw = raw if isinstance(raw, dict) else {}
        decision = str(raw.get("decision") or "").strip().upper()
        if decision not in {"ANSWER", "SEARCH"}:
            decision = "SEARCH"

        answer = raw.get("answer")
        answer = (
            answer.strip()
            if isinstance(answer, str) and answer.strip() and len(answer) <= 8000
            else ""
        )

        supports = []
        values = raw.get("supports")
        if isinstance(values, list):
            for value in values[:6]:
                if not isinstance(value, dict):
                    continue
                memory_id = value.get("memory_id")
                quote = value.get("quote")
                if (
                    isinstance(memory_id, str)
                    and memory_id.strip()
                    and isinstance(quote, str)
                    and quote.strip()
                    and len(quote) <= 800
                ):
                    supports.append(
                        {"memory_id": memory_id.strip(), "quote": quote.strip()}
                    )

        queries = self._ad_clean_texts(raw.get("queries"), 4)

        # An incomplete ANSWER is never terminal. Downgrade instead of guessing.
        if decision == "ANSWER" and (not answer or not supports):
            decision = "SEARCH"

        return {
            "decision": decision,
            "answer": answer if decision == "ANSWER" else "",
            "supports": supports if decision == "ANSWER" else [],
            "queries": queries if decision == "SEARCH" else [],
        }

    def _ad_relation_payload(self, memories):
        by_id = {m["id"]: m for m in memories}
        output = []
        for relation in self._relations:
            source, target = relation.get("source_id"), relation.get("target_id")
            kind = relation.get("type")
            if source not in by_id or target not in by_id:
                continue
            if kind not in {"REFINE", "SUPERSEDE", "CONFLICT", "CAUSES"}:
                continue
            if kind == "CAUSES" and not self._valid_causal_relation(relation, by_id):
                continue
            output.append(
                {
                    "source_id": source,
                    "type": kind,
                    "target_id": target,
                }
            )
        return output[:24]

    def _ad_controller_memory(self, memory):
        status = memory.get(
            "_status", self._belief_status.get(memory.get("id"), "active")
        )
        return {
            "id": memory["id"],
            "kind": memory.get("kind", "FACT"),
            "status": status,
            "claim": memory.get("claim", ""),
            "owner_id": memory.get("owner_id") or memory.get("subject_id") or "",
            "subject": memory.get("subject") or "",
            "scope": memory.get("scope") or "",
            "predicate": memory.get("state_key") or memory.get("predicate") or "",
            "object": memory.get("object_anchor") or "",
            "value": self._memory_value(memory),
            "verbatim_value": memory.get("verbatim_value") or "",
            "stance": memory.get("stance", "AFFIRM"),
            "assertion_mode": memory.get("assertion_mode", "DIRECT"),
            "event_time": memory.get("event_time") or "",
            "document_time": memory.get("document_time") or "",
            "origin_document_time": memory.get("origin_document_time") or "",
            "effective_event_time": self._date_for(memory, "effective_event_time") or "",
        }

    def _ad_advise(self, question, base, *, hard_metadata=None, caller_instructions=None):
        payload = {
            "memories": [self._ad_controller_memory(m) for m in base],
            "relations": self._ad_relation_payload(base),
        }
        prompt = (
            ANSWER_OR_SEARCH_PROMPT
            + "\nQUESTION:\n"
            + str(question)
            + "\nHARD CALLER METADATA:\n"
            + json.dumps(hard_metadata or {}, ensure_ascii=False)
            + "\nCALLER INSTRUCTIONS:\n"
            + str(caller_instructions or "")
            + "\nBASE EVIDENCE:\n"
            + json.dumps(payload, ensure_ascii=False)
        )
        usage, error = {}, ""
        try:
            response = self._llm_client.chat(
                [{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=1024,
                response_format={"type": "json_object"},
            )
            usage = self._response_usage(response, prompt)
            raw = self._parse_json(response.content)
        except Exception as exc:
            raw, error = {}, str(exc)
        return self._ad_normalize(raw, question), usage, error

    @staticmethod
    def _ad_normalize_space(value):
        return re.sub(r"\s+", " ", str(value or "")).strip()

    def _ad_grounding_guard(self, base, controller, *, hard_owner_id=None):
        """Mechanical integrity checks only; never semantic entailment."""
        if controller.get("decision") != "ANSWER":
            return {
                "accepted": False,
                "reason": "SEARCH_REQUESTED",
                "support_ids": [],
                "failures": [],
            }

        by_id = {m["id"]: m for m in base}
        evidence_ids = {str(e.get("id")) for e in self._evidence if e.get("id")}
        failures, support_ids = [], []

        for support in controller.get("supports") or []:
            memory_id = support["memory_id"]
            quote = self._ad_normalize_space(support["quote"])
            memory = by_id.get(memory_id)
            if memory is None:
                failures.append({"memory_id": memory_id, "reason": "OUTSIDE_BASE_WORLD"})
                continue

            if hard_owner_id:
                owner = memory.get("owner_id") or memory.get("subject_id") or ""
                if not owner or str(owner) != str(hard_owner_id):
                    failures.append(
                        {"memory_id": memory_id, "reason": "HARD_OWNER_MISMATCH"}
                    )
                    continue

            linked = [str(eid) for eid in (memory.get("evidence_ids") or [])]
            if not linked or not any(eid in evidence_ids for eid in linked):
                failures.append(
                    {"memory_id": memory_id, "reason": "MISSING_PROVENANCE"}
                )
                continue

            exposed = self._ad_controller_memory(memory)
            exact_fields = [
                self._ad_normalize_space(value)
                for key, value in exposed.items()
                if key != "id" and isinstance(value, (str, int, float))
            ]
            if not quote or not any(quote in field for field in exact_fields if field):
                failures.append(
                    {"memory_id": memory_id, "reason": "QUOTE_NOT_IN_MEMORY"}
                )
                continue

            if memory_id not in support_ids:
                support_ids.append(memory_id)

        if failures:
            return {
                "accepted": False,
                "reason": "GROUNDING_INTEGRITY_FAILED",
                "support_ids": support_ids,
                "failures": failures,
            }
        if not support_ids:
            return {
                "accepted": False,
                "reason": "NO_VALID_SUPPORT",
                "support_ids": [],
                "failures": [],
            }
        return {
            "accepted": True,
            "reason": "GROUNDED_ANSWER",
            "support_ids": support_ids,
            "failures": [],
        }

    def _ad_expand(self, question, base, controller):
        """Add only controller-requested search evidence; BaseWorld is immutable."""
        world = list(base)
        seen = {m["id"] for m in world}
        candidates, novel, trace = [], [], []
        searches = list(controller.get("queries") or [])[:4]

        for index, query in enumerate(searches):
            remaining = self.CANDIDATE_WORLD_LIMIT - len(world)
            if remaining <= 0:
                break
            remaining_probes = len(searches) - index - 1
            quota = min(2, max(1, remaining - remaining_probes))
            outputs = self._ad_search(query, 8)
            candidates.extend(m["id"] for m in outputs)
            added = []
            for memory in outputs:
                if memory["id"] in seen:
                    continue
                world.append(memory)
                seen.add(memory["id"])
                novel.append(memory["id"])
                added.append(memory)
                if len(added) >= quota or len(world) >= self.CANDIDATE_WORLD_LIMIT:
                    break
            trace.append(
                {
                    "operation": "CONTROLLER_SEARCH",
                    "input": query,
                    "output_ids": [m["id"] for m in added],
                }
            )

        recovery = not world and self.enable_zero_result_recovery
        recovery_ids = []
        if recovery:
            outputs = self._ad_search(question, 4)
            for memory in outputs:
                if memory["id"] not in seen and len(world) < self.CANDIDATE_WORLD_LIMIT:
                    world.append(memory)
                    seen.add(memory["id"])
                    recovery_ids.append(memory["id"])
            trace.append(
                {"operation": "ZERO_HIT_RECOVERY", "output_ids": recovery_ids}
            )

        return world, {
            "hint_candidate_ids": list(dict.fromkeys(candidates)),
            "hint_novel_ids": novel,
            "recovery_called": recovery,
            "recovery_novel_ids": recovery_ids,
            "zero_hit_requirements": ["question"] if recovery else [],
            "retrieval_trace": trace,
        }
