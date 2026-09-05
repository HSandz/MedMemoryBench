"""Minimal seed-conditioned semantic IR for SmartMem0 reads.

The question owns evidence obligations. Seeds may suggest a candidate answer or
soft retrieval neighborhood, but durable identities and physical operations are
resolved deterministically after this layer.
"""

import json
import re
import unicodedata
from typing import Any, Dict, List, Sequence

from .canonicalization import state_identity
from .contracts import (
    GENERIC_OBJECT_ANCHORS,
    GENERIC_STATE_KEYS,
    STATE_KEY_NOISE,
    STOPWORDS,
    VALID_TEMPORAL_AXES,
)

SEMANTIC_IR_POLICY = """You produce the minimal semantic IR for an evidence-grounded memory system.

QUESTION defines what must be answered. First derive requirements from QUESTION. Then use SEEDS only to suggest where relevant evidence may live or to propose one directly supported atomic candidate. A seed must never redefine the question predicate, subject, comparison, causal request, or temporal constraint.

Return only answer_type, requirement nodes, relations between those nodes, and an optional candidate. Do not emit a route, query class, operator, difficulty score, budget, retrieval operation, evidence role, durable memory key, or internal subject ID.

For every requirement:
- focus_span is the shortest useful contiguous span copied from QUESTION. It is the immutable question-owned anchor.
- For a comparison, include each side's local temporal selector in its focus_span when present in QUESTION.
- retrieval_hint is a soft semantic description informed by QUESTION and, when useful, SEEDS. It may broaden the evidence neighborhood but is never a fact, hard filter, answer, or durable key.
- time_constraint is optional. document_time is when something was documented or mentioned; event_time is when it happened; origin_document_time is the original source date; effective_event_time is the explicitly combined event/source chronology. LOCATE means the requested time is unknown. EXACT/BEFORE/AFTER/BETWEEN constrain a known time. EARLIEST/LATEST request an extremum.

Relations are generic requirement-graph edges:
- COMPARE compares two requirement nodes.
- CAUSES requests an explicit stored causal path.
- POSSIBLE_CAUSE asks the answer model to connect grounded participant endpoints using general domain knowledge.
- DEPENDS_ON declares evidence dependency without asserting causality.
- TEMPORAL_ORDER orders two requirement nodes by event time. Its relation is BEFORE, AFTER, or OVERLAPS; temporal order never implies causality.
- INFER authorizes a general-domain bridge after all referenced participant requirements are grounded.
- CURRENT marks one requirement as the current state.
- VERIFY_SOURCE asks for exact linked source evidence when wording, source attribution, or what was said in a particular source matters. A document_time filter or a question asking only for a documentation date does not by itself require raw evidence.

Use to="ANSWER" for an INFER edge whose result is the answer rather than another requirement. Temporal order alone is not CAUSES.
Use CAUSES only when the requested relation is explicitly stated in participant evidence. For "likely explains", clinical deduction, or mechanism questions where the endpoints may be observed but the connecting rule is not stored, use POSSIBLE_CAUSE or INFER instead; do not demand a nonexistent stored CAUSES edge.

INFER is exceptional. Do not emit INFER for extraction, paraphrase, entity selection, ordinary synthesis, choosing a supported option, or when one retrieved participant memory can state the answer directly. Emit it only when grounded participant facts must be combined with a general-domain rule to produce an answer that is not explicit in memory.

Keep requirements atomic but do not split one answer-bearing phrase into a noun and its qualifier when one memory can satisfy both. A single fact such as a medication with its dose, a symptom with its severity, or an event with its date is normally one requirement.

Visible answer options are propositions to evaluate, not memory facts. Requirements must describe shared participant-specific evidence needed to evaluate them. Do not create one requirement per option merely because the option is visible.

candidate is optional and is allowed only when exactly one seed directly and atomically contains the answer to one requirement without comparison, temporal localization, source verification, or multi-step inference. The candidate may name only $seed0, $seed1, or $seed2. Code will independently authorize it.
"""

SEMANTIC_IR_SCHEMA = """Return JSON only:
{{"answer_type":"ENTITY|VALUE|DATE|RELATIVE_TIME|OPTION_SET|TEXT","subject_span":"exact contiguous subject span from QUESTION or empty","requirements":[{{"id":"r1","focus_span":"exact contiguous span from QUESTION","retrieval_hint":"soft semantic retrieval description","time_constraint":{{"axis":"event_time|document_time|origin_document_time|effective_event_time|","relation":"LOCATE|EXACT|EARLIEST|LATEST|BEFORE|AFTER|BETWEEN|","anchor":"","end":""}}}}],"relations":[{{"type":"COMPARE|CAUSES|POSSIBLE_CAUSE|DEPENDS_ON|TEMPORAL_ORDER|INFER|CURRENT|VERIFY_SOURCE","from":"r1","to":"r2|ANSWER|","relation":"BEFORE|AFTER|OVERLAPS|"}}],"candidate":null}}
When candidate exists use: {{"candidate":{{"answer":"exact answer value","support_ref":"$seed0"}}}}
QUESTION:
{question}
VISIBLE OPTIONS:
{options}
STRUCTURAL HINTS (constraints only; never evidence):
{hints}
SEEDS:
{seeds}"""

VALID_ANSWER_TYPES = {
    "ENTITY",
    "VALUE",
    "DATE",
    "RELATIVE_TIME",
    "OPTION_SET",
    "TEXT",
}
VALID_IR_RELATIONS = {
    "COMPARE",
    "CAUSES",
    "POSSIBLE_CAUSE",
    "DEPENDS_ON",
    "TEMPORAL_ORDER",
    "INFER",
    "CURRENT",
    "VERIFY_SOURCE",
}
VALID_TIME_RELATIONS = {
    "LOCATE",
    "EXACT",
    "EARLIEST",
    "LATEST",
    "BEFORE",
    "AFTER",
    "BETWEEN",
    "",
}


class ReadContractMixin:
    @staticmethod
    def _rc_text(value: Any) -> str:
        return " ".join(
            unicodedata.normalize("NFKC", str(value or "")).casefold().split()
        )

    @classmethod
    def _rc_terms(cls, value: Any) -> List[str]:
        normalized = cls._rc_text(value).replace("_", " ").replace("-", " ")
        return [
            token
            for token in re.findall(r"\w+", normalized, flags=re.UNICODE)
            if len(token) > 1
        ]

    @classmethod
    def _rc_content_terms(cls, value: Any) -> List[str]:
        # This vocabulary is a ranking aid only. Structural validity never
        # depends on a stopword-normalized score.
        noise = (
            set(STOPWORDS)
            | set(STATE_KEY_NOISE)
            | set(GENERIC_STATE_KEYS)
            | set(GENERIC_OBJECT_ANCHORS)
            | {"primary", "primary_user"}
        )
        return [term for term in cls._rc_terms(value) if term not in noise]

    def _rc_owner(self, value: Any) -> str:
        raw = "_".join(self._rc_terms(value))
        aliases = {
            "_".join(self._rc_terms(key)): "_".join(self._rc_terms(mapped))
            for key, mapped in (getattr(self, "subject_aliases", {}) or {}).items()
        }
        return aliases.get(raw, raw)

    def _rc_owner_match(self, slot: Dict[str, Any], memory: Dict[str, Any]) -> bool:
        wanted = self._rc_owner(slot.get("subject_id") or slot.get("subject") or "")
        actual = self._rc_owner(memory.get("subject_id") or memory.get("subject") or "")
        return not wanted or bool(actual and actual == wanted)

    def _rc_known_subject(self, value: Any) -> str:
        candidate = self._rc_owner(value)
        if not candidate:
            return ""
        known = {
            self._rc_owner(memory.get("subject_id") or memory.get("subject") or "")
            for memory in getattr(self, "_memories", [])
        }
        known.discard("")
        return candidate if candidate in known else ""

    def _rc_question_span(self, value: Any, question: str) -> str:
        raw = " ".join(str(value or "").strip().split())
        if not raw:
            return ""
        return raw if self._rc_token_sequence_present(raw, question) else ""

    def _rc_target_from_question(self, target: Any, question: str) -> str:
        """Compatibility alias for the question-owned target firewall."""
        return self._rc_question_span(target, question)

    def _rc_memory_concept_keys(self, memory: Dict[str, Any]) -> List[str]:
        values: List[Any] = [
            memory.get("scope"),
            memory.get("state_key"),
            memory.get("object_anchor"),
        ]
        values.extend(memory.get("entities") or [])
        values.extend(memory.get("scope_entities") or [])
        output: List[str] = []
        for value in values:
            terms = self._rc_content_terms(value)
            if not terms:
                continue
            key = " ".join(terms)
            if key not in output:
                output.append(key)
        return output

    def _rc_resolve_target_keys(self, target: str, subject_id: str = "") -> List[str]:
        """Resolve a question span to durable metadata as a ranking hint."""
        target_terms = set(self._rc_content_terms(target))
        if not target_terms:
            return []
        wanted_owner = self._rc_owner(subject_id)
        scores: Dict[str, float] = {}
        for memory in self._memories:
            owner = self._rc_owner(
                memory.get("subject_id") or memory.get("subject") or ""
            )
            if wanted_owner and owner != wanted_owner:
                continue
            for key in self._rc_memory_concept_keys(memory):
                key_terms = set(self._rc_content_terms(key))
                if not key_terms:
                    continue
                overlap = len(target_terms & key_terms)
                if not overlap:
                    continue
                precision = overlap / len(key_terms)
                coverage = overlap / len(target_terms)
                if key_terms.issubset(target_terms) or precision >= 0.67:
                    scores[key] = max(
                        scores.get(key, 0.0), 0.75 * precision + 0.25 * coverage
                    )
        return [
            key
            for key, _ in sorted(
                scores.items(),
                key=lambda item: (
                    -item[1],
                    -len(self._rc_content_terms(item[0])),
                    item[0],
                ),
            )[:4]
        ]

    def _rc_seed_payload(self, seeds: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        output = []
        for index, memory in enumerate(seeds[:3]):
            identity = state_identity(memory) or ""
            output.append(
                {
                    "ref": f"$seed{index}",
                    "kind": memory.get("kind"),
                    "subject": memory.get("subject_id") or memory.get("subject"),
                    "scope": memory.get("scope"),
                    "object": memory.get("object_anchor"),
                    "value": self._memory_value(memory),
                    "claim": str(memory.get("claim") or "")[:320],
                    "event_time": memory.get("event_time"),
                    "document_time": memory.get("document_time"),
                    "origin_document_time": memory.get("origin_document_time"),
                    "state_identity": identity,
                    "is_state_head": bool(identity and self._is_state_head(memory)),
                    "status": memory.get(
                        "_status",
                        self._belief_status.get(memory.get("id"), "active"),
                    ),
                }
            )
        return output

    def _rc_normalize_time_constraint(
        self, raw: Any, frame: Any, question: str
    ) -> Dict[str, str]:
        value = raw if isinstance(raw, dict) else {}
        axis = str(value.get("axis") or "").lower()
        axis = axis if axis in VALID_TEMPORAL_AXES else ""
        relation = str(value.get("relation") or "").upper()
        relation = relation if relation in VALID_TIME_RELATIONS else ""
        anchor = str(value.get("anchor") or "").strip()
        end = str(value.get("end") or "").strip()
        dates = [str(item) for item in (getattr(frame, "dates", ()) or ()) if item]
        question_text = self._rc_text(question)
        if (
            anchor
            and anchor not in dates
            and self._rc_text(anchor) not in question_text
        ):
            anchor = ""
        if end and end not in dates and self._rc_text(end) not in question_text:
            end = ""
        if relation in {"EXACT", "BEFORE", "AFTER"} and not anchor and len(dates) == 1:
            anchor = dates[0]
        if relation == "BETWEEN" and len(dates) >= 2:
            anchor = anchor or dates[0]
            end = end or dates[1]
        if relation == "EXACT" and not anchor:
            relation = "LOCATE"
        if relation == "BETWEEN" and (not anchor or not end):
            relation = "LOCATE"
        if relation in {"BEFORE", "AFTER"} and not anchor:
            relation = "LOCATE"
        # A relation without an explicit axis is not executable. Silently
        # choosing event_time would turn a malformed semantic request into a
        # different one, so retain neither the relation nor its anchors.
        if not axis:
            relation = ""
            anchor = ""
            end = ""
        return {
            "axis": axis,
            "relation": relation,
            "anchor": anchor,
            "end": end,
        }

    def _rc_normalize_requirement(
        self, raw: Dict[str, Any], index: int, question: str, frame: Any
    ) -> Dict[str, Any]:
        focus = self._rc_question_span(raw.get("focus_span"), question)
        if not focus:
            focus = self._question_stem(question).strip()
        time_constraint = self._rc_normalize_time_constraint(
            raw.get("time_constraint"), frame, question
        )
        focus_binder = getattr(self, "_rc_bind_focus_time_constraint", None)
        if callable(focus_binder):
            time_constraint = focus_binder(focus, time_constraint)
        return {
            "id": str(raw.get("id") or f"r{index + 1}"),
            "focus_span": focus,
            "retrieval_hint": " ".join(str(raw.get("retrieval_hint") or "").split())[
                :320
            ],
            "time_constraint": time_constraint,
        }

    def _rc_normalize_ir(
        self, parsed: Dict[str, Any], question: str, frame: Any
    ) -> Dict[str, Any]:
        options = self._question_options(question) or {}
        answer_type = str(parsed.get("answer_type") or "TEXT").upper()
        answer_type = (
            "OPTION_SET"
            if options
            else answer_type if answer_type in VALID_ANSWER_TYPES else "TEXT"
        )
        subject_span = self._rc_question_span(parsed.get("subject_span"), question)
        raw_requirements = (
            parsed.get("requirements")
            if isinstance(parsed.get("requirements"), list)
            else []
        )
        requirements = [
            self._rc_normalize_requirement(raw, index, question, frame)
            for index, raw in enumerate(raw_requirements[:4])
            if isinstance(raw, dict)
        ]
        if not requirements:
            stem = self._question_stem(question).strip()
            requirements = [
                {
                    "id": "r1",
                    "focus_span": stem,
                    "retrieval_hint": stem,
                    "time_constraint": {
                        "axis": "",
                        "relation": "",
                        "anchor": "",
                        "end": "",
                    },
                }
            ]

        seen = set()
        for index, requirement in enumerate(requirements):
            candidate_id = requirement["id"]
            if not re.fullmatch(r"r[\w-]{0,31}", candidate_id) or candidate_id in seen:
                candidate_id = f"r{index + 1}"
            while candidate_id in seen:
                candidate_id += "x"
            requirement["id"] = candidate_id
            seen.add(candidate_id)

        valid_nodes = {requirement["id"] for requirement in requirements}
        relations = []
        raw_relations = (
            parsed.get("relations") if isinstance(parsed.get("relations"), list) else []
        )
        for raw in raw_relations[:6]:
            if not isinstance(raw, dict):
                continue
            relation_type = str(raw.get("type") or "").upper()
            source = str(raw.get("from") or "")
            target = str(raw.get("to") or "")
            if relation_type not in VALID_IR_RELATIONS or source not in valid_nodes:
                continue
            if relation_type in {"CURRENT", "VERIFY_SOURCE"}:
                target = ""
            elif target != "ANSWER" and target not in valid_nodes:
                continue
            if (
                relation_type
                in {
                    "COMPARE",
                    "CAUSES",
                    "POSSIBLE_CAUSE",
                    "DEPENDS_ON",
                    "TEMPORAL_ORDER",
                }
                and target not in valid_nodes
            ):
                continue
            relation = {"type": relation_type, "from": source, "to": target}
            if relation_type == "TEMPORAL_ORDER":
                order = str(raw.get("relation") or "").upper()
                if order not in {"BEFORE", "AFTER", "OVERLAPS"}:
                    continue
                relation["relation"] = order
            if relation not in relations:
                relations.append(relation)

        # Keep intermediate temporal obligations referenced by the requirement
        # graph, even when the final answer is an entity or value.
        temporal_dependencies = {
            endpoint
            for relation in relations
            if relation.get("type") == "TEMPORAL_ORDER"
            for endpoint in (relation.get("from"), relation.get("to"))
        }
        temporal_dependencies.update(
            relation.get("to")
            for relation in relations
            if relation.get("type") == "DEPENDS_ON"
        )
        if answer_type not in {"DATE", "RELATIVE_TIME"}:
            for requirement in requirements:
                constraint = requirement.get("time_constraint") or {}
                if (
                    constraint.get("relation") == "LOCATE"
                    and requirement["id"] not in temporal_dependencies
                ):
                    requirement["time_constraint"] = {
                        "axis": "",
                        "relation": "",
                        "anchor": "",
                        "end": "",
                    }

        candidate = parsed.get("candidate")
        if not isinstance(candidate, dict):
            candidate = None
        if candidate is not None:
            answer = str(candidate.get("answer") or "").strip()
            support_ref = str(candidate.get("support_ref") or "")
            if not answer or not re.fullmatch(r"\$seed[0-2]", support_ref):
                candidate = None
            else:
                candidate = {"answer": answer, "support_ref": support_ref}

        # A one-requirement atomic candidate does not need an INFER edge merely
        # to authorize extraction of a value already present in its seed.
        if candidate and len(requirements) == 1 and relations and all(
            relation.get("type") == "INFER"
            and relation.get("from") == requirements[0]["id"]
            and relation.get("to") == "ANSWER"
            for relation in relations
        ):
            relations = []

        # Repair only a narrow, observable over-decomposition pattern: an
        # atomic candidate plus noun/qualifier requirements connected solely by
        # dependency/inference edges. Keep any one compatible temporal selector
        # on the merged evidence obligation; do not collapse real comparisons,
        # causal endpoints, or independently answerable requirements.
        relation_types = {relation.get("type") for relation in relations}
        if (
            candidate
            and len(requirements) > 1
            and answer_type in {"ENTITY", "VALUE", "DATE", "RELATIVE_TIME"}
            and relation_types
            and relation_types.issubset({"DEPENDS_ON", "INFER"})
        ):
            constraints = [
                requirement.get("time_constraint") or {}
                for requirement in requirements
                if (requirement.get("time_constraint") or {}).get("axis")
            ]
            signatures = {
                (
                    item.get("axis", ""),
                    item.get("relation", ""),
                    item.get("anchor", ""),
                    item.get("end", ""),
                )
                for item in constraints
            }
            if len(signatures) <= 1:
                merged_constraint = constraints[0] if constraints else {
                    "axis": "",
                    "relation": "",
                    "anchor": "",
                    "end": "",
                }
                requirements = [
                    {
                        "id": "r1",
                        "focus_span": self._question_stem(question).strip(),
                        "retrieval_hint": " | ".join(
                            dict.fromkeys(
                                str(item.get("retrieval_hint") or "").strip()
                                for item in requirements
                                if str(item.get("retrieval_hint") or "").strip()
                            )
                        )[:320],
                        "time_constraint": dict(merged_constraint),
                    }
                ]
                relations = []

        subject_id = self._rc_known_subject(subject_span)

        return {
            "answer_type": answer_type,
            "subject_span": subject_span,
            "_resolved_subject_id": subject_id,
            "requirements": requirements,
            "relations": relations,
            "candidate": candidate,
            "visible_options": dict(options),
        }

    def _rc_answer_grounded(
        self, answer: str, memories: Sequence[Dict[str, Any]]
    ) -> bool:
        answer_norm = self._rc_text(answer)
        evidence = self._rc_text(
            " ".join(
                str(value or "")
                for memory in memories
                for value in (
                    memory.get("claim"),
                    self._memory_value(memory),
                    memory.get("verbatim_value"),
                    memory.get("object_anchor"),
                    " ".join(memory.get("entities", [])),
                )
            )
        )
        if not answer_norm:
            return False
        if self._rc_token_sequence_present(answer_norm, evidence):
            return True
        numbers = set(re.findall(r"\d+(?:\.\d+)?", answer_norm))
        evidence_numbers = set(re.findall(r"\d+(?:\.\d+)?", evidence))
        if numbers and not numbers.issubset(evidence_numbers):
            return False
        answer_terms = self._rc_content_terms(answer_norm)
        evidence_terms = set(self._rc_content_terms(evidence))
        return bool(
            answer_terms
            and len(answer_terms) <= 12
            and sum(term in evidence_terms for term in answer_terms) / len(answer_terms)
            >= 0.8
        )

    @classmethod
    def _rc_token_sequence_present(cls, needle: Any, haystack: Any) -> bool:
        """Match complete Unicode-aware tokens, never arbitrary substrings."""
        wanted = cls._rc_terms(needle)
        available = cls._rc_terms(haystack)
        width = len(wanted)
        return bool(
            wanted
            and any(
                available[index : index + width] == wanted
                for index in range(len(available) - width + 1)
            )
        )

    @classmethod
    def _rc_candidate_has_atomic_surface(
        cls, answer: str, memory: Dict[str, Any]
    ) -> bool:
        """Require the answer in an atomic value field, not incidental seed text."""
        surfaces = []
        for value in (
            memory.get("value"),
            memory.get("verbatim_value"),
            str(memory.get("object_anchor") or "").replace("_", " "),
        ):
            normalized = cls._rc_text(value)
            if normalized and normalized not in surfaces:
                surfaces.append(normalized)
        return any(
            cls._rc_token_sequence_present(answer, surface)
            or cls._rc_token_sequence_present(surface, answer)
            for surface in surfaces
        )

    def _authorize_controller_candidate(self, ir, seeds, frame):
        candidate = ir.get("candidate")
        if not candidate or len(ir.get("requirements") or []) != 1:
            return None, "NO_ATOMIC_CANDIDATE"
        relations = ir.get("relations") or []
        if ir.get("visible_options") or any(
            relation.get("type") != "CURRENT" for relation in relations
        ):
            return None, "RELATIONAL_QUERY_REQUIRES_RETRIEVAL"
        requirement = ir["requirements"][0]
        time_constraint = requirement.get("time_constraint") or {}
        if ir.get("answer_type") in {"DATE", "RELATIVE_TIME"}:
            return None, "TEMPORAL_ANSWER_REQUIRES_RETRIEVAL"
        if getattr(frame, "dates", ()) and not time_constraint.get("axis"):
            return None, "TEMPORAL_CONSTRAINT_REQUIRES_PLAN"
        reference = candidate["support_ref"]
        match = re.fullmatch(r"\$seed(\d+)", reference)
        if not match or int(match.group(1)) >= min(3, len(seeds)):
            return None, "INVALID_SUPPORT_REF"
        valid = self._validate_fast_support(reference, seeds, frame)
        if not valid:
            return None, "STRUCTURAL_SUPPORT_REJECTED"
        if time_constraint.get("axis") or time_constraint.get("relation"):
            axis = str(time_constraint.get("axis") or "")
            relation = str(time_constraint.get("relation") or "")
            anchor = str(time_constraint.get("anchor") or "")
            if (
                relation != "EXACT"
                or not axis
                or not anchor
                or not self._date_matches(self._date_for(valid[0], axis), anchor)
            ):
                return None, "TEMPORAL_QUERY_REQUIRES_RETRIEVAL"
        # focus_span protects the semantic obligation in the IR. Candidate
        # authorization itself is deliberately structural: requiring lexical
        # overlap here would reject valid paraphrases and make English
        # stopwords part of the correctness path.
        if relations and not self._is_state_head(valid[0]):
            return None, "CURRENT_CANDIDATE_IS_NOT_STATE_HEAD"
        if not self._rc_candidate_has_atomic_surface(candidate["answer"], valid[0]):
            return None, "ANSWER_NOT_IN_ATOMIC_SURFACE"
        if not self._rc_answer_grounded(candidate["answer"], valid):
            return None, "ANSWER_NOT_GROUNDED"
        return valid, "AUTHORIZED"

    def _authorize_controller_answer(self, ir, seeds, frame):
        """Authorize an atomic candidate, including an exact-date filter match."""
        return self._authorize_controller_candidate(ir, seeds, frame)

    @staticmethod
    def _rc_public_ir(ir):
        return {
            "answer_type": ir.get("answer_type") or "TEXT",
            "subject_span": ir.get("subject_span") or "",
            "requirements": [dict(item) for item in ir.get("requirements") or []],
            "relations": [dict(item) for item in ir.get("relations") or []],
            "candidate": dict(ir["candidate"]) if ir.get("candidate") else None,
        }

    def _semantic_controller(self, question, seeds, frame, context_map=None):
        del context_map
        self._last_option_probe_coverage = {}
        self._active_controller_seeds = list(seeds[:3])
        options = self._question_options(question) or {}
        hints = {
            "dates": list(getattr(frame, "dates", ()) or ()),
            "source_speaker": getattr(frame, "speaker_role", ""),
            "explicit_entities": list(getattr(frame, "entities", ()) or ()),
        }
        prompt = (
            SEMANTIC_IR_POLICY
            + "\n"
            + SEMANTIC_IR_SCHEMA.format(
                question=question,
                options=json.dumps(options, ensure_ascii=False),
                hints=json.dumps(hints, ensure_ascii=False),
                seeds=json.dumps(self._rc_seed_payload(seeds), ensure_ascii=False),
            )
        )
        try:
            response = self._llm_client.chat(
                [{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=512,
                response_format={"type": "json_object"},
            )
            usage = self._response_usage(response, prompt)
            ir = self._rc_normalize_ir(
                self._parse_json(response.content), question, frame
            )
            error = ""
        except Exception as exc:
            usage = {}
            ir = self._rc_normalize_ir({}, question, frame)
            error = str(exc)

        supports, authorization = self._authorize_controller_answer(ir, seeds, frame)
        if supports is not None:
            candidate = ir["candidate"]
            telemetry = {
                "called": True,
                "route": "DIRECT",
                "route_source": "derived_from_authorized_candidate",
                "answer": candidate["answer"],
                "support_ref": candidate["support_ref"],
                "support_refs": [candidate["support_ref"]],
                "fallback_reason": "",
                "usage": usage,
                "answer_type": ir["answer_type"],
                "requirement_count": 1,
                "relation_count": 0,
                "semantic_ir": self._rc_public_ir(ir),
            }
            return supports, {}, telemetry

        plan = self._controller_plan(ir, question, frame)
        telemetry = {
            "called": True,
            "route": "PLAN",
            "route_source": "derived_from_candidate_authorization",
            "answer": "",
            "support_ref": "",
            "support_refs": [],
            "fallback_reason": error or (authorization if ir.get("candidate") else ""),
            "error": error,
            "usage": usage,
            "answer_type": ir["answer_type"],
            "requirement_count": len(ir["requirements"]),
            "relation_count": len(ir["relations"]),
            "semantic_ir": self._rc_public_ir(ir),
        }
        return None, plan, telemetry
