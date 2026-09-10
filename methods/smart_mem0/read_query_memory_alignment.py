"""Query/memory alignment for the locked SmartMem0 READ architecture.

This mixin is intentionally benchmark-, dataset-, domain-, and language-agnostic.
LLM #1 owns only semantic interpretation: projection, participant-memory obligations,
selectors, relations, and optional retrieval-only bridges.  Runtime code owns physical
recall, candidate admission, ranking, proof, context sufficiency, and terminal rendering.

The same durable semantic substrate written into memories (subject/state/family/scope,
object/value/stance, temporal axes and provenance) is reused at READ time as structural
ranking/certification capability.  Retrieval-only text can broaden recall but can never
become proof.
"""

import json
import re
from copy import deepcopy
from typing import Any, Dict, Iterable, List, Sequence

from .contracts import VALID_TEMPORAL_AXES


SEMANTIC_CONTROLLER_POLICY_V2 = """
You are the single semantic interpreter for an evidence-grounded memory system.
Understand the QUESTION in its own language and context. Resolve paraphrase, coreference,
implicit participant references, and temporal/relation meaning when needed.

Return only semantic information that is actually present or required. Omit semantic
fields you cannot justify from QUESTION. Missing means UNKNOWN / not asserted; never turn
missing into CURRENT, LATEST, a subject, or another semantic claim.

You may emit:
- requested_projection: ENTITY, VALUE, DATE, RELATIVE_TIME, OPTION_SET, or TEXT.
- subject_span: only an explicit contiguous subject span from QUESTION.
- requirements: the smallest sufficient participant-memory evidence obligations.
- bridges: relations needed between grounded obligations.
- retrieval_bridges: optional paraphrases/contextual rewrites used ONLY to retrieve memory.
- terminal_candidate: only when one Top-3 seed already contains the complete atomic answer;
  point to a support_ref plus a canonical field, or an exact answer_span contained in it.

Do NOT emit benchmark/query-class labels, route, difficulty, budget, retrieval operations,
proof status, final-context decisions, confidence-as-truth, or a free-generated final answer.

REQUIREMENTS
- Usually 1-3; never more than 4.
- grounding_kind=QUESTION when answer_obligation is the shortest useful contiguous span
  copied from QUESTION.
- grounding_kind=DERIVED only for an additional participant-memory variable whose stored
  value can change the answer; never invent a general-domain mechanism as memory.
- evidence_family is selector-neutral recall language, not evidence.
- selector is optional. Emit it only when temporal selection changes which memory is needed.
- constraints are short semantic restrictions. They narrow recall but are never proof.

RETRIEVAL BRIDGES
A retrieval_bridge may help bridge language, paraphrase, context or terminology. It is
retrieval_only by definition and must never be treated as a stored fact or proof.

TERMINAL CANDIDATE
Use only for one atomic QUESTION requirement grounded by one of $seed0..$seed2.
Prefer {support_ref, projection:{field}} where field is claim, value, verbatim_value,
object_anchor, event_time, document_time, origin_document_time, or effective_event_time.
Alternatively use {support_ref, answer_span} only when answer_span occurs in that seed.
Do not use terminal_candidate for comparison, option/candidate selection, advice/action,
source verification, temporal extrema/ranges, multi-memory synthesis, or world-knowledge
inference. Runtime will independently reject any candidate that is not structurally grounded.
"""


SEMANTIC_CONTROLLER_SCHEMA_V2 = """
Return one sparse JSON object. Omit unused optional fields. Example shape (not a template
that must be filled):
{{
  "requested_projection": "VALUE",
  "requirements": [
    {{
      "id": "r1",
      "grounding_kind": "QUESTION",
      "answer_obligation": "exact useful QUESTION span",
      "evidence_family": "selector-neutral recall phrase"
    }}
  ],
  "bridges": [{{"type":"TEMPORAL_ORDER","from":"r1","to":"r2","relation":"AFTER"}}],
  "retrieval_bridges": [{{"for":"r1","text":"contextual paraphrase","retrieval_only":true}}],
  "terminal_candidate": {{"support_ref":"$seed0","projection":{{"field":"value"}}}}
}}

QUESTION:
{question}
CANDIDATE SET (structural alternatives, not memory facts):
{candidate_set}
STRUCTURAL HINTS (constraints only, never evidence):
{hints}
TOP-3 SEEDS (planning views, never proof by themselves):
{seeds}
"""


class ReadQueryMemoryAlignmentMixin:
    """Unify query semantics, durable memory semantics and deterministic selection."""

    QUERY_MEMORY_ALIGNMENT_VERSION = "query-memory-alignment-v2"
    CONTROLLER_SCHEMA_VERSION = "sparse-semantic-ir-v2"
    CONTROLLER_MAX_OUTPUT_TOKENS = 512

    # Candidate acquisition and answer-context packing are deliberately independent.
    CANDIDATE_WORLD_HARD_CAP = 16
    PER_SEARCH_ACQUISITION_FLOOR = 8
    ANSWER_CONTEXT_MIN_CAP = 4
    ANSWER_CONTEXT_HARD_CAP = 8

    _TERMINAL_FIELDS = frozenset(
        {
            "claim",
            "value",
            "verbatim_value",
            "object_anchor",
            *VALID_TEMPORAL_AXES,
        }
    )

    @staticmethod
    def _alignment_unique(values: Iterable[Any]) -> List[str]:
        output, seen = [], set()
        for value in values or []:
            text = str(value or "")
            if text and text not in seen:
                output.append(text)
                seen.add(text)
        return output

    @classmethod
    def _aop_raw_candidate(cls, parsed: Any):
        """Parse a structural terminal pointer; never accept a generated answer as authority."""
        if not isinstance(parsed, dict):
            return None
        raw = parsed.get("terminal_candidate")
        # Read old manifests safely, but intentionally ignore legacy free-form answer text.
        if not isinstance(raw, dict):
            raw = parsed.get("candidate") if isinstance(parsed.get("candidate"), dict) else None
        if not isinstance(raw, dict):
            return None
        support_ref = str(raw.get("support_ref") or "")
        if not re.fullmatch(r"\$seed[0-2]", support_ref):
            return None
        projection = raw.get("projection") if isinstance(raw.get("projection"), dict) else {}
        field = str(projection.get("field") or raw.get("field") or "")
        answer_span = " ".join(str(raw.get("answer_span") or "").split())
        if field and field not in cls._TERMINAL_FIELDS:
            field = ""
        if not field and not answer_span:
            return None
        candidate = {"support_ref": support_ref}
        if field:
            candidate["projection"] = {"field": field}
        if answer_span:
            candidate["answer_span"] = answer_span
        return candidate

    def _controller_seed_payload(self, seeds):
        """Expose only compact canonical WRITE fields that help semantic interpretation."""
        output = []
        for index, memory in enumerate((seeds or [])[:3]):
            output.append(
                {
                    "ref": f"$seed{index}",
                    "claim": str(memory.get("claim") or "")[:240],
                    "kind": str(memory.get("kind") or "FACT"),
                    "semantic_role": str(memory.get("semantic_role") or ""),
                    "subject": str(memory.get("subject_id") or memory.get("subject") or "")[:80],
                    "evidence_family": str(memory.get("evidence_family") or "")[:120],
                    "state_key": str(memory.get("state_key") or "")[:120],
                    "scope": str(memory.get("scope") or "")[:120],
                    "object_anchor": str(memory.get("object_anchor") or "")[:140],
                    "value": str(memory.get("value") or "")[:180],
                    "verbatim_value": str(memory.get("verbatim_value") or "")[:180],
                    "stance": str(memory.get("stance") or ""),
                    "entities": list(memory.get("entities") or [])[:8],
                    "event_time": memory.get("event_time") or "",
                    "document_time": memory.get("document_time") or "",
                    "origin_document_time": memory.get("origin_document_time") or "",
                }
            )
        return output

    def _rc_public_ir(self, ir):
        public = super()._rc_public_ir(ir)
        public.pop("candidate", None)
        if ir.get("candidate"):
            public["terminal_candidate"] = deepcopy(ir["candidate"])
        shape = ir.get("query_shape") or self._derive_query_shape(ir)
        public["query_shape"] = deepcopy(shape)
        bridges = list(ir.get("retrieval_bridges") or [])
        if bridges:
            public["retrieval_bridges"] = deepcopy(bridges)
        return public

    def _rc_normalize_ir(self, parsed: Dict[str, Any], question: str, frame: Any):
        parsed = deepcopy(parsed) if isinstance(parsed, dict) else {}
        # Retrieval bridges are semantic search helpers only. Fold them into the matching
        # requirement's retrieval family while retaining an explicit retrieval_only trace.
        raw_bridges = parsed.get("retrieval_bridges") if isinstance(parsed.get("retrieval_bridges"), list) else []
        bridge_trace = []
        by_requirement: Dict[str, List[str]] = {}
        for raw in raw_bridges[:6]:
            if not isinstance(raw, dict):
                continue
            rid = str(raw.get("for") or raw.get("requirement_id") or "")
            text = " ".join(str(raw.get("text") or raw.get("query") or "").split())[:240]
            if rid and text:
                by_requirement.setdefault(rid, []).append(text)
                bridge_trace.append({"for": rid, "text": text, "retrieval_only": True})
        requirements = parsed.get("requirements") if isinstance(parsed.get("requirements"), list) else []
        for index, raw in enumerate(requirements):
            if not isinstance(raw, dict):
                continue
            rid = str(raw.get("id") or f"r{index + 1}")
            bridge_text = by_requirement.get(rid) or []
            if bridge_text:
                family = " ".join(str(raw.get("evidence_family") or "").split())
                raw["evidence_family"] = " | ".join(self._alignment_unique([family, *bridge_text]))
        ir = super()._rc_normalize_ir(parsed, question, frame)
        ir["retrieval_bridges"] = bridge_trace
        ir["query_shape"] = self._derive_query_shape(ir)
        return ir

    @staticmethod
    def _derive_query_shape(ir: Dict[str, Any]) -> Dict[str, Any]:
        requirements = list(ir.get("requirements") or [])
        relation_types = sorted(
            {
                str(edge.get("type") or "").upper()
                for edge in (ir.get("relations") or [])
                if str(edge.get("type") or "").strip()
            }
        )
        projection = str(ir.get("answer_type") or "TEXT").upper()
        has_candidates = bool(ir.get("candidate_propositions") or ir.get("visible_options"))
        selectors = [
            requirement.get("selector") or requirement.get("time_constraint") or {}
            for requirement in requirements
        ]
        temporal_selection = any(
            str(selector.get("relation") or "").upper()
            in {"EARLIEST", "LATEST", "BEFORE", "AFTER", "BETWEEN"}
            for selector in selectors
            if isinstance(selector, dict)
        )
        if has_candidates:
            reasoning_form = "CANDIDATE_EVALUATION"
        elif "COMPARE" in relation_types:
            reasoning_form = "COMPARISON"
        elif temporal_selection or "TEMPORAL_ORDER" in relation_types:
            reasoning_form = "TEMPORAL_SELECTION"
        elif len(relation_types) > 1:
            reasoning_form = "RELATION_CHAIN"
        elif len(requirements) > 1 or relation_types:
            reasoning_form = "MULTI_PREMISE"
        else:
            reasoning_form = "ATOMIC_EXTRACTIVE"
        terminal_eligible = (
            reasoning_form == "ATOMIC_EXTRACTIVE"
            and len(requirements) == 1
            and projection not in {"OPTION_SET", "RELATIVE_TIME"}
        )
        return {
            "reasoning_form": reasoning_form,
            "requested_projection": projection,
            "requirement_count": len(requirements),
            "relation_types": relation_types,
            "has_explicit_subject": bool(str(ir.get("subject_span") or "").strip()),
            "has_temporal_selector": any(bool(selector) for selector in selectors),
            "terminal_eligible": terminal_eligible,
            "routing_semantics": "telemetry_and_sufficiency_only",
        }

    def _answer_context_cap(self, ir: Dict[str, Any]) -> int:
        """Compute only an upper packing cap from evidence topology; never a retrieval route."""
        shape = ir.get("query_shape") or self._derive_query_shape(ir)
        requirements = max(1, int(shape.get("requirement_count") or 1))
        relations = len(shape.get("relation_types") or [])
        # The actual context may stop before this cap.  Extra obligations/relations merely
        # preserve room for independent premises or competing states.
        return min(
            self.ANSWER_CONTEXT_HARD_CAP,
            max(self.ANSWER_CONTEXT_MIN_CAP, 3 + requirements + min(2, relations)),
        )

    def _controller_plan(self, ir, question, frame):
        plan = super()._controller_plan(ir, question, frame)
        shape = ir.get("query_shape") or self._derive_query_shape(ir)
        cap = self._answer_context_cap(ir)
        plan["max_memories"] = cap
        plan["answer_context_cap"] = cap
        plan["candidate_world_cap"] = self.CANDIDATE_WORLD_HARD_CAP
        plan.setdefault("query_spec", {})["query_shape"] = deepcopy(shape)
        plan.setdefault("query_spec", {})["answer_context_cap"] = cap
        plan.setdefault("semantic_ir", {})["query_shape"] = deepcopy(shape)
        return plan

    def _terminal_projection_value(self, candidate: Dict[str, Any], memory: Dict[str, Any]) -> str:
        projection = candidate.get("projection") if isinstance(candidate.get("projection"), dict) else {}
        field = str(projection.get("field") or "")
        if field:
            if field in VALID_TEMPORAL_AXES:
                return " ".join(str(self._date_for(memory, field) or "").split())
            return " ".join(str(memory.get(field) or "").replace("_", " ").split())
        span = " ".join(str(candidate.get("answer_span") or "").split())
        if not span:
            return ""
        # A span is authorized only by literal containment in one durable answer-bearing
        # field.  Similarity is deliberately not a truth predicate.
        for field_name in ("value", "verbatim_value", "object_anchor", "claim"):
            surface = " ".join(str(memory.get(field_name) or "").replace("_", " ").split())
            if surface and span.casefold() in surface.casefold():
                return span
        return ""

    def _authorize_controller_answer(self, ir, seeds, frame):
        candidate = ir.get("candidate") or {}
        requirements = list(ir.get("requirements") or [])
        if not candidate or len(requirements) != 1 or not self._aop_direct_surface_allowed("", ir):
            return None, "NO_STRUCTURAL_TERMINAL_CANDIDATE"
        reference = str(candidate.get("support_ref") or "")
        match = re.fullmatch(r"\$seed(\d+)", reference)
        if not match or int(match.group(1)) >= min(3, len(seeds or [])):
            return None, "INVALID_SUPPORT_REF"
        valid = self._validate_fast_support(reference, seeds, frame)
        if not valid:
            return None, "STRUCTURAL_SUPPORT_REJECTED"
        memory = valid[0]
        if not self._memory_satisfies_frame(memory, frame):
            return None, "EXPLICIT_FRAME_MISMATCH"

        selector = requirements[0].get("selector") or requirements[0].get("time_constraint") or {}
        relation = str(selector.get("relation") or "").upper()
        axis = str(selector.get("axis") or "")
        anchor = str(selector.get("anchor") or "")
        if relation in {"EARLIEST", "LATEST", "BEFORE", "AFTER", "BETWEEN"}:
            return None, "TEMPORAL_ARBITRATION_REQUIRES_RETRIEVAL"
        if relation == "EXACT":
            if axis not in VALID_TEMPORAL_AXES or not anchor or not self._date_matches(self._date_for(memory, axis), anchor):
                return None, "TEMPORAL_FILTER_MISMATCH"
        if ir.get("relations"):
            if any(str(edge.get("type") or "").upper() != "CURRENT" for edge in ir.get("relations") or []):
                return None, "RELATIONAL_QUERY_REQUIRES_RETRIEVAL"
            if not self._is_state_head(memory):
                return None, "CURRENT_CANDIDATE_IS_NOT_STATE_HEAD"

        answer = self._terminal_projection_value(candidate, memory)
        if not answer:
            return None, "TERMINAL_PROJECTION_NOT_GROUNDED"
        candidate["answer"] = answer  # deterministic renderer compatibility; not LLM authority
        return valid, "AUTHORIZED_STRUCTURAL_TERMINAL"

    def _semantic_controller(self, question, seeds, frame, context_map=None):
        """One semantic LLM call: certified terminal pointer or general evidence IR."""
        reset = getattr(self, "_reset_candidate_set_state", None)
        if callable(reset):
            reset()
        terminal_reset = getattr(self, "_terminal_reset_state", None)
        if callable(terminal_reset):
            terminal_reset()
        self._active_controller_seeds = list((seeds or [])[:3])
        self._alignment_seed_ids = [str(memory.get("id") or "") for memory in (seeds or [])[:3] if memory.get("id")]

        options = self._question_options(question) or {}
        propositions = {}
        normalize = getattr(self, "_normalize_candidate_propositions", None)
        if callable(normalize):
            propositions = normalize(options)
            if not propositions and isinstance(context_map, dict):
                propositions = normalize(context_map.get("candidate_set") or context_map.get("candidate_propositions"))
        self._last_candidate_propositions = dict(propositions)
        self._last_proposition_probe_coverage = {pid: [] for pid in propositions}

        hints = {
            "dates": list(getattr(frame, "dates", ()) or ()),
            "source_speaker": getattr(frame, "speaker_role", ""),
            "explicit_entities": list(getattr(frame, "entities", ()) or ()),
        }
        prompt = SEMANTIC_CONTROLLER_POLICY_V2 + "\n" + SEMANTIC_CONTROLLER_SCHEMA_V2.format(
            question=question,
            candidate_set=json.dumps(propositions, ensure_ascii=False),
            hints=json.dumps(hints, ensure_ascii=False),
            seeds=json.dumps(self._controller_seed_payload(seeds), ensure_ascii=False),
        )
        raw_ir = {}
        try:
            response = self._llm_client.chat(
                [{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=self.CONTROLLER_MAX_OUTPUT_TOKENS,
                response_format={"type": "json_object"},
            )
            usage = self._response_usage(response, prompt)
            raw_ir = self._parse_json(response.content)
            ir = self._rc_normalize_ir(raw_ir, question, frame)
            error = ""
        except Exception as exc:
            usage = {}
            ir = self._rc_normalize_ir({}, question, frame)
            error = str(exc)
        ir["candidate_propositions"] = dict(propositions)
        ir["query_shape"] = self._derive_query_shape(ir)

        pack = {}
        pack_builder = getattr(self, "_build_candidate_proposition_pack", None)
        if callable(pack_builder) and propositions:
            pack = pack_builder(propositions, frame=frame, seeds=(seeds or [])[:3], question=question)

        projection = self._aop_direct_projection(ir, question)
        supports, authorization, active_ir = None, "NO_STRUCTURAL_TERMINAL_CANDIDATE", ir
        if projection is not None:
            supports, authorization = self._authorize_controller_answer(projection, seeds, frame)
            if supports is not None:
                active_ir = projection
        shape = active_ir.get("query_shape") or self._derive_query_shape(active_ir)
        common = {
            "called": True,
            "error": error,
            "usage": usage,
            "requested_projection": active_ir.get("answer_type", "TEXT"),
            "answer_type": active_ir.get("answer_type", "TEXT"),
            "answer_mode": shape.get("reasoning_form"),
            "answer_mode_source": "derived_query_shape_telemetry_only",
            "query_shape": deepcopy(shape),
            "requirement_count": len(active_ir.get("requirements") or []),
            "bridge_count": len(active_ir.get("relations") or []),
            "relation_count": len(active_ir.get("relations") or []),
            "semantic_ir": self._rc_public_ir(active_ir),
            "controller_raw_ir": raw_ir,
            "normalized_ir": self._rc_public_ir(active_ir),
            "normalization_status": active_ir.get("normalization_status", "VALID"),
            "graph_validation": active_ir.get("graph_validation") or {},
            "normalization_actions": list(active_ir.get("normalization_actions") or []),
            "graph_warnings": [item for item in (active_ir.get("normalization_actions") or []) if item.get("action") == "GRAPH_WARNING"],
            "candidate_authorization": authorization,
            "candidate_set_count": len(propositions),
            "candidate_proposition_count": len(propositions),
            "candidate_proposition_pack_size": len(pack.get("retrieval_views") or []),
            "controller_schema_version": self.CONTROLLER_SCHEMA_VERSION,
            "fallback_reason": error or (authorization if ir.get("candidate") and supports is None else ""),
        }
        if supports is not None:
            answer = str(active_ir["candidate"].get("answer") or "")
            telemetry = dict(common)
            telemetry.update(
                {
                    "route": "DIRECT",
                    "route_source": "structurally_certified_seed_projection",
                    "answer": answer,
                    "support_ref": active_ir["candidate"]["support_ref"],
                    "support_refs": [active_ir["candidate"]["support_ref"]],
                    "fallback_reason": "",
                    "terminal_rendered": True,
                }
            )
            return supports, {}, telemetry

        plan = self._controller_plan(ir, question, frame)
        telemetry = dict(common)
        telemetry.update(
            {
                "route": "PLAN",
                "route_source": "semantic_ir_requires_retrieval_or_synthesis",
                "answer": "",
                "support_ref": "",
                "support_refs": [],
                "terminal_rendered": False,
            }
        )
        return None, plan, telemetry

    @staticmethod
    def _raw_channel_heads(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Reserve one raw lexical and one raw dense representative when available."""
        rows = list(rows or [])
        output: List[Dict[str, Any]] = []

        def add(memory):
            if memory and all(item.get("id") != memory.get("id") for item in output):
                output.append(memory)

        add(
            min((m for m in rows if m.get("_bm25_rank") is not None), key=lambda m: m.get("_bm25_rank"), default=None)
        )
        add(
            min((m for m in rows if m.get("_dense_rank") is not None), key=lambda m: m.get("_dense_rank"), default=None)
        )
        return output

    def _fusion_multiview(self, operation, outputs, seeds, frame):
        """Give raw lexical/dense recall independent bounded admission rights."""
        semantic_rows, relations, evidence_refs = super()._fusion_multiview(operation, outputs, seeds, frame)
        views = self._fusion_views(operation.get("retrieval_views") or [])
        raw_view = views.get("question")
        if not raw_view:
            return semantic_rows, relations, evidence_refs

        include_history = str(operation.get("strategy") or "FOCAL").upper() == "TRAJECTORY"
        eligible_ids = {
            str(memory.get("id") or "")
            for memory in getattr(self, "_memories", []) or []
            if memory.get("id")
            and self._memory_satisfies_frame(memory, frame, include_entities=bool(getattr(frame, "hard_entities", ())))
            and self._query_visible_memory(memory, include_history=include_history)
        }
        raw_rows = list(
            self._hybrid_search(
                raw_view["query"],
                top_k=min(max(self.PER_SEARCH_ACQUISITION_FLOOR, int(operation.get("top_k", 6) or 6)), len(eligible_ids)),
                candidate_ids=eligible_ids,
            )
            or []
        ) if eligible_ids else []
        raw_heads = self._raw_channel_heads(raw_rows)

        # One semantic head plus lexical/dense heads are protected before normal fill.
        # No channel has semantic truth authority; these are candidate admission rights only.
        ordered: List[Dict[str, Any]] = []
        seen = set()
        def add(memory, source):
            if not memory or not memory.get("id") or memory["id"] in seen:
                return
            item = deepcopy(memory)
            tags = list(item.get("_alignment_recall_sources") or [])
            if source not in tags:
                tags.append(source)
            item["_alignment_recall_sources"] = tags
            ordered.append(item)
            seen.add(item["id"])

        add((semantic_rows or [None])[0], "semantic_ir")
        if raw_heads:
            add(raw_heads[0], "raw_lexical")
        if len(raw_heads) > 1:
            add(raw_heads[1], "raw_dense")
        for memory in semantic_rows or []:
            add(memory, "semantic_ir")
        for memory in raw_rows:
            add(memory, "raw_question")
        limit = max(1, int(operation.get("top_k", len(ordered)) or len(ordered)))
        return ordered[:limit], relations or [], evidence_refs or []

    def _execute_plan(self, plan, seeds, round_offset=0, frame=None, question=""):
        """Widen physical acquisition without widening the original answer-context cap."""
        physical = deepcopy(plan)
        answer_cap = max(1, int(plan.get("answer_context_cap") or plan.get("max_memories") or self.ANSWER_CONTEXT_MIN_CAP))
        physical["answer_context_cap"] = answer_cap
        physical["max_memories"] = self.CANDIDATE_WORLD_HARD_CAP
        physical["candidate_world_cap"] = self.CANDIDATE_WORLD_HARD_CAP
        for operation in physical.get("operations") or []:
            if str(operation.get("op") or "").upper() == "SEMANTIC_SEARCH" or str(operation.get("_lean_op") or "").upper() == "SEARCH_FAMILY":
                try:
                    requested = int(operation.get("top_k", 0) or 0)
                except (TypeError, ValueError):
                    requested = 0
                operation["top_k"] = min(
                    self.CANDIDATE_WORLD_HARD_CAP,
                    max(self.PER_SEARCH_ACQUISITION_FLOOR, requested),
                )
        result = super()._execute_plan(
            physical,
            seeds,
            round_offset=round_offset,
            frame=frame,
            question=question,
        )
        self._alignment_candidate_meta = {
            str(memory.get("id") or ""): deepcopy(memory)
            for output in (result.get("operation_outputs") or [])
            for memory in output
            if memory.get("id")
        }
        result["candidate_world_limit"] = self.CANDIDATE_WORLD_HARD_CAP
        result["answer_context_cap"] = answer_cap
        result["acquisition_separated_from_answer_context"] = True
        return result

    def _alignment_memory(self, memory_id: str) -> Dict[str, Any]:
        meta = (getattr(self, "_alignment_candidate_meta", {}) or {}).get(str(memory_id))
        if meta:
            return meta
        return next((memory for memory in getattr(self, "_memories", []) or [] if str(memory.get("id") or "") == str(memory_id)), {})

    def _alignment_projection_capability(self, memory: Dict[str, Any]) -> int:
        shape = getattr(self, "_active_query_shape", None) or {}
        projection = str(shape.get("requested_projection") or "TEXT").upper()
        if projection == "DATE":
            return int(any(self._date_for(memory, axis) for axis in VALID_TEMPORAL_AXES))
        if projection == "VALUE":
            return int(bool(memory.get("value") or memory.get("verbatim_value")))
        if projection == "ENTITY":
            return int(bool(memory.get("object_anchor") or memory.get("entities") or memory.get("value")))
        return int(bool(memory.get("claim") or self._memory_value(memory)))

    def _alignment_signature(self, memory: Dict[str, Any]):
        """Structural substitutability signature; similarity alone never defines redundancy."""
        norm = getattr(self, "_rc_text", lambda value: " ".join(str(value or "").lower().split()))
        return (
            norm(memory.get("subject_id") or memory.get("subject")),
            norm(memory.get("state_key") or memory.get("evidence_family") or memory.get("scope")),
            norm(memory.get("object_anchor")),
            norm(memory.get("value") or memory.get("verbatim_value")),
            norm(memory.get("stance")),
            str(memory.get("event_time") or ""),
            str(memory.get("document_time") or ""),
        )

    def _role_aware_support_ids(self, slots, slot_support, candidate_order, limit):
        """Coverage/scarcity first; fused relevance only ranks substitutes within coverage."""
        baseline = list(super()._role_aware_support_ids(slots, slot_support, candidate_order, limit))
        allowed = self._alignment_unique(candidate_order)
        allowed_set = set(allowed)
        bounded_limit = min(self.ANSWER_CONTEXT_HARD_CAP, max(0, int(limit)))
        if not bounded_limit:
            return []
        original_index = {memory_id: index for index, memory_id in enumerate(allowed)}
        context = getattr(self, "_last_requirement_context_candidates", {}) or {}
        proof = getattr(self, "_last_requirement_proof_support", {}) or {}

        lanes: Dict[str, List[str]] = {}
        for slot in slots or []:
            rid = str(slot.get("id") or "")
            if not rid:
                continue
            values = self._alignment_unique([
                *(proof.get(rid) or []),
                *(context.get(rid) or []),
                *((slot_support or {}).get(rid) or []),
            ])
            values = [memory_id for memory_id in values if memory_id in allowed_set]
            if values:
                lanes[f"requirement:{rid}"] = values
        for proposition_id, values in (getattr(self, "_last_proposition_probe_coverage", {}) or {}).items():
            filtered = [memory_id for memory_id in self._alignment_unique(values) if memory_id in allowed_set]
            if filtered:
                lanes[f"proposition:{proposition_id}"] = filtered

        membership: Dict[str, set] = {memory_id: set() for memory_id in allowed}
        for lane, values in lanes.items():
            for memory_id in values:
                membership.setdefault(memory_id, set()).add(lane)
        proof_ids = {memory_id for values in proof.values() for memory_id in (values or [])}
        seed_ids = set(getattr(self, "_alignment_seed_ids", []) or [])
        selected: List[str] = []
        uncovered = set(lanes)

        while uncovered and len(selected) < bounded_limit:
            choices = [memory_id for memory_id in allowed if memory_id not in selected and membership.get(memory_id, set()) & uncovered]
            if not choices:
                break
            def choice_key(memory_id):
                new_lanes = membership.get(memory_id, set()) & uncovered
                scarcity = min((len(lanes[lane]) for lane in new_lanes), default=10**6)
                memory = self._alignment_memory(memory_id)
                views = len(set(memory.get("_binding_views") or memory.get("_alignment_recall_sources") or []))
                specificity = sum(bool(memory.get(field)) for field in ("state_key", "evidence_family", "scope", "object_anchor", "value", "verbatim_value"))
                return (
                    -len(new_lanes),
                    scarcity,
                    -int(memory_id in proof_ids),
                    -self._alignment_projection_capability(memory),
                    -views,
                    -specificity,
                    -int(memory_id in seed_ids),
                    original_index.get(memory_id, 10**9),
                    memory_id,
                )
            chosen = min(choices, key=choice_key)
            selected.append(chosen)
            uncovered -= membership.get(chosen, set())

        # Redundancy-aware fill. Exact structural substitutes can be dropped only after all
        # material lanes they contribute are already covered. Unique evidence remains cheap
        # to keep even when its global similarity rank is lower.
        signatures = {
            self._alignment_signature(self._alignment_memory(memory_id))
            for memory_id in selected
            if self._alignment_memory(memory_id)
        }
        for memory_id in [*baseline, *allowed]:
            if memory_id in selected or memory_id not in allowed_set:
                continue
            memory = self._alignment_memory(memory_id)
            signature = self._alignment_signature(memory) if memory else None
            contributes_uncovered = bool(membership.get(memory_id, set()) & uncovered)
            if signature and signature in signatures and not contributes_uncovered:
                continue
            selected.append(memory_id)
            if signature:
                signatures.add(signature)
            uncovered -= membership.get(memory_id, set())
            if len(selected) >= bounded_limit:
                break

        self._last_query_memory_alignment = {
            "version": self.QUERY_MEMORY_ALIGNMENT_VERSION,
            "candidate_world_cap": self.CANDIDATE_WORLD_HARD_CAP,
            "answer_context_cap": bounded_limit,
            "lane_count": len(lanes),
            "uncovered_lanes": sorted(uncovered),
            "selected_ids": list(selected),
            "selection_semantics": "coverage_scarcity_then_redundancy_aware_fill",
            "proof_semantics": "selection_is_not_truth",
        }
        return selected[:bounded_limit]

    def prepare_batch_query(self, question, system_message=None, **kwargs):
        prepared = super().prepare_batch_query(question, system_message=system_message, **kwargs)
        extra = prepared.setdefault("extra", {})
        controller = extra.get("semantic_controller") or {}
        self._active_query_shape = deepcopy(controller.get("query_shape") or {})
        extra["query_memory_alignment_version"] = self.QUERY_MEMORY_ALIGNMENT_VERSION
        extra["query_shape"] = deepcopy(controller.get("query_shape") or {})
        extra["query_memory_alignment"] = deepcopy(getattr(self, "_last_query_memory_alignment", {}) or {})
        extra["candidate_world_hard_cap"] = self.CANDIDATE_WORLD_HARD_CAP
        return prepared
