"""Stable, language-neutral semantic control plane for SmartMem0 READ.

The controller owns intent only. Deterministic code owns retrieval, selectors, recovery,
proof, context packing and terminality.

Invariants:
- LLM #1 never sees retrieved memory/seeds.
- The public IR contains only projection, minimal memory requirements, selectors and genuine
  relations. It contains no route, budget, retrieval operation, proof result, alias expansion,
  terminal candidate or answer.
- Full-question BM25/dense recall remains an independent acquisition rail.
- Recovery is allowed only for structurally empty requirements.
- Proof is one post-retrieval, non-destructive checkpoint. It may promote/certify but may
  not erase candidates, downgrade structural coverage, or trigger acquisition.
"""

import json
from copy import deepcopy
from typing import Any, Dict, List

from .read_execution_contract import ReadExecutionContractMixin


STABLE_SEMANTIC_POLICY = """
You are a semantic parser for a general evidence-memory system.
Interpret the QUESTION in its own language. Return only the smallest semantic structure
needed to retrieve participant-specific evidence. Do not answer the question.

Your output must be domain-, dataset-, benchmark- and language-neutral.

Allowed semantics:
- projection: ENTITY, VALUE, DATE, RELATIVE_TIME, OPTION_SET, or TEXT.
- optional subject_span: only an exact contiguous span copied from QUESTION.
- requirements: usually 1-3, maximum 4. Each requirement is one independent
  participant-specific fact/state/event/constraint whose stored value could change the answer.
- optional selector on a requirement:
  CURRENT, LOCATE, EARLIEST, LATEST, EXACT, BEFORE, AFTER, BETWEEN.
  CURRENT means the active durable state, not merely "recent"; use LATEST for a temporal
  extremum. Selectors choose among memories and must not be embedded into retrieval meaning.
- relations only when the answer truly depends on a relation between independent requirements:
  DEPENDS_ON, COMPARE, TEMPORAL_ORDER, CAUSES, INFER, VERIFY_SOURCE.
  INFER is allowed only when a general-domain rule/mechanism not stored as participant memory
  is genuinely necessary. It must include a short bridge_goal. Never emit INFER just to mean
  "use this requirement to answer".

Requirement rules:
- need: selector-neutral description of what memory family is needed, in the QUESTION's
  language. Do not include words equivalent to latest/earliest/current unless they are part
  of the underlying concept itself.
- question_span: optional exact contiguous QUESTION span that names the concept.
- Do not invent canonical database keys, ontology labels, aliases, synonyms or memory facts.
  Runtime resolves durable addresses independently.
- Do not emit proof/certificates, confidence, routes, difficulty, budgets, retrieval
  operations, candidate memory IDs, terminal candidates, or final answers.
- Missing information means unknown; never manufacture a selector, subject or relation.

Relations refer to requirement positions using 1-based integers. Use "ANSWER" only as the
target of a genuine INFER relation.
"""


STABLE_SEMANTIC_SCHEMA = """
Return one sparse JSON object:
{{
  "projection": "ENTITY|VALUE|DATE|RELATIVE_TIME|OPTION_SET|TEXT",
  "subject_span": "optional exact QUESTION span",
  "requirements": [
    {{
      "need": "selector-neutral participant-memory variable",
      "question_span": "optional exact contiguous QUESTION span",
      "selector": {{
        "relation": "CURRENT|LOCATE|EARLIEST|LATEST|EXACT|BEFORE|AFTER|BETWEEN",
        "axis": "event_time|document_time|origin_document_time|effective_event_time",
        "anchor": "only when required",
        "end": "only for BETWEEN"
      }}
    }}
  ],
  "relations": [
    {{
      "type": "DEPENDS_ON|COMPARE|TEMPORAL_ORDER|CAUSES|INFER|VERIFY_SOURCE",
      "from": 1,
      "to": 2,
      "order": "BEFORE|AFTER|OVERLAPS",
      "bridge_goal": "required only for INFER"
    }}
  ]
}}

QUESTION:
{question}
CANDIDATE SET (structural answer alternatives, not memory evidence):
{candidate_set}
STRUCTURAL HINTS (question-derived constraints only, not memory evidence):
{hints}
"""


class ReadStableSemanticRuntimeMixin:
    """Own the stable semantic IR and one-shot non-destructive proof boundary."""

    STABLE_READ_ARCHITECTURE_VERSION = "stable-semantic-read-v1"
    STABLE_SEMANTIC_IR_VERSION = "semantic-intent-ir-v2"
    CONTROLLER_SCHEMA_VERSION = "stable-semantic-intent-v2"
    CONTROLLER_MAX_OUTPUT_TOKENS = 420
    PROOF_CHECKPOINT_VERSION = "one-shot-nondestructive-proof-v1"

    _STABLE_PROJECTIONS = frozenset(
        {"ENTITY", "VALUE", "DATE", "RELATIVE_TIME", "OPTION_SET", "TEXT"}
    )
    _STABLE_RELATIONS = frozenset(
        {"DEPENDS_ON", "COMPARE", "TEMPORAL_ORDER", "CAUSES", "INFER", "VERIFY_SOURCE"}
    )

    @classmethod
    def _stable_projection(cls, value: Any) -> str:
        value = str(value or "TEXT").upper().strip()
        return value if value in cls._STABLE_PROJECTIONS else "TEXT"

    def _stable_exact_question_span(self, value: Any, question: str) -> str:
        value = " ".join(str(value or "").split()).strip()
        if not value:
            return ""
        resolver = getattr(self, "_rc_question_span", None)
        if callable(resolver):
            return str(resolver(value, question) or "").strip()
        return value if value.casefold() in str(question or "").casefold() else ""

    def _stable_selector(self, raw: Any) -> Dict[str, str]:
        sparse = getattr(self, "_rg_sparse_selector", None)
        return dict(sparse(raw) or {}) if callable(sparse) else {}

    def _stable_sanitize_controller_ir(
        self, parsed: Any, question: str, candidate_set: Dict[str, str]
    ) -> Dict[str, Any]:
        parsed = parsed if isinstance(parsed, dict) else {}
        projection = self._stable_projection(
            parsed.get("projection")
            or parsed.get("requested_projection")
            or (parsed.get("goal") or {}).get("projection")
        )
        if candidate_set:
            projection = "OPTION_SET"

        subject_span = self._stable_exact_question_span(
            parsed.get("subject_span"), question
        )

        raw_requirements = (
            parsed.get("requirements")
            if isinstance(parsed.get("requirements"), list)
            else []
        )
        requirements: List[Dict[str, Any]] = []
        old_to_new: Dict[int, str] = {}
        seen = {}
        for raw_index, raw in enumerate(raw_requirements[:4], start=1):
            if not isinstance(raw, dict):
                continue
            need = " ".join(
                str(
                    raw.get("need")
                    or raw.get("evidence_family")
                    or raw.get("target")
                    or ""
                ).split()
            ).strip()[:280]
            question_span = self._stable_exact_question_span(
                raw.get("question_span") or raw.get("focus_span"), question
            )
            if not need:
                need = question_span
            if not need:
                continue
            selector = self._stable_selector(
                raw.get("selector") or raw.get("time_constraint")
            )
            key = (
                self._rc_text(need),
                json.dumps(selector, sort_keys=True, ensure_ascii=False),
            )
            if key in seen:
                old_to_new[raw_index] = seen[key]
                continue
            rid = f"r{len(requirements) + 1}"
            seen[key] = rid
            old_to_new[raw_index] = rid
            item: Dict[str, Any] = {"id": rid, "need": need}
            if question_span:
                item["question_span"] = question_span
            if selector:
                item["selector"] = selector
            requirements.append(item)

        if not requirements:
            fallback = " ".join(
                str(self._question_stem(question) or question or "").split()
            ).strip()[:280]
            if fallback:
                requirements = [
                    {
                        "id": "r1",
                        "need": fallback,
                        "question_span": self._stable_exact_question_span(
                            fallback, question
                        )
                        or fallback,
                    }
                ]
                old_to_new[1] = "r1"

        canonical_ids = {item["id"] for item in requirements}

        def resolve_ref(value: Any) -> str:
            if isinstance(value, int):
                return old_to_new.get(value, "")
            text = str(value or "").strip()
            if text.upper() == "ANSWER":
                return "ANSWER"
            if text.lower().startswith("r") and text[1:].isdigit():
                return old_to_new.get(int(text[1:]), "")
            if text.isdigit():
                return old_to_new.get(int(text), "")
            return text if text in canonical_ids else ""

        raw_relations = (
            parsed.get("relations")
            if isinstance(parsed.get("relations"), list)
            else parsed.get("links")
            if isinstance(parsed.get("links"), list)
            else []
        )
        relations: List[Dict[str, Any]] = []
        for raw in raw_relations[:8]:
            if not isinstance(raw, dict):
                continue
            relation_type = str(raw.get("type") or "").upper().strip()
            if relation_type not in self._STABLE_RELATIONS:
                continue
            source = resolve_ref(raw.get("from"))
            target = resolve_ref(raw.get("to"))
            if not source:
                continue
            if relation_type == "VERIFY_SOURCE":
                target = ""
            elif relation_type == "INFER":
                bridge_goal = " ".join(
                    str(raw.get("bridge_goal") or "").split()
                ).strip()[:220]
                if not bridge_goal or target != "ANSWER":
                    continue
            elif target == "ANSWER" or not target:
                continue
            if target and target == source:
                continue

            item: Dict[str, Any] = {
                "type": relation_type,
                "from": source,
                "to": target,
            }
            if relation_type == "TEMPORAL_ORDER":
                order = str(
                    raw.get("order") or raw.get("relation") or ""
                ).upper().strip()
                if order not in {"BEFORE", "AFTER", "OVERLAPS"}:
                    continue
                item["order"] = order
            if relation_type == "INFER":
                item["bridge_goal"] = bridge_goal
            if item not in relations:
                relations.append(item)

        return {
            "goal": {
                "projection": projection,
                "directive": " ".join(
                    str(self._question_stem(question) or question or "").split()
                ).strip()[:260],
            },
            "subject_span": subject_span,
            "requirements": requirements,
            "links": relations,
        }

    def _stable_normalize_internal_ir(
        self, ir: Dict[str, Any], candidate_set: Dict[str, str]
    ) -> Dict[str, Any]:
        ir = deepcopy(ir or {})
        ir["answer_type"] = self._stable_projection(
            (ir.get("goal") or {}).get("projection") or ir.get("answer_type")
        )
        if candidate_set:
            ir["answer_type"] = "OPTION_SET"
            ir.setdefault("goal", {})["projection"] = "OPTION_SET"
        ir["candidate_propositions"] = dict(candidate_set)
        for requirement in ir.get("requirements") or []:
            requirement["material_anchors"] = []
            requirement["search_aliases"] = []
            requirement["constraints"] = []
        ir["retrieval_bridges"] = []
        ir.pop("candidate", None)
        ir["query_shape"] = self._derive_query_shape(ir)
        ir["stable_semantic_ir_version"] = self.STABLE_SEMANTIC_IR_VERSION
        return ir

    @staticmethod
    def _derive_query_shape(ir: Dict[str, Any]) -> Dict[str, Any]:
        requirements = list(ir.get("requirements") or [])
        relations = [
            item
            for item in (ir.get("relations") or ir.get("query_links") or [])
            if str(item.get("type") or "").upper() != "CURRENT"
        ]
        relation_types = sorted(
            {
                str(item.get("type") or "").upper()
                for item in relations
                if str(item.get("type") or "").strip()
            }
        )
        projection = str(
            (ir.get("goal") or {}).get("projection")
            or ir.get("answer_type")
            or "TEXT"
        ).upper()
        selectors = [
            requirement.get("selector") or requirement.get("time_constraint") or {}
            for requirement in requirements
        ]
        has_candidates = bool(
            ir.get("candidate_propositions") or ir.get("visible_options")
        )
        if has_candidates:
            reasoning_form = "CANDIDATE_EVALUATION"
        elif len(requirements) == 1 and not relation_types:
            reasoning_form = "ATOMIC_EXTRACTIVE"
        else:
            reasoning_form = "COMPOSED"
        return {
            "reasoning_form": reasoning_form,
            "requested_projection": projection,
            "requirement_count": len(requirements),
            "relation_types": relation_types,
            "has_explicit_subject": bool(str(ir.get("subject_span") or "").strip()),
            "has_temporal_selector": any(bool(selector) for selector in selectors),
            "terminal_eligible": bool(
                reasoning_form == "ATOMIC_EXTRACTIVE"
                and len(requirements) == 1
                and projection not in {"OPTION_SET", "RELATIVE_TIME"}
            ),
            "routing_semantics": "deterministic_from_stable_semantic_ir",
        }

    def _semantic_controller(self, question, seeds, frame, context_map=None):
        """Exactly one question-only semantic call; retrieved memory never enters the prompt."""
        del seeds, context_map
        reset = getattr(self, "_reset_candidate_set_state", None)
        if callable(reset):
            reset()
        terminal_reset = getattr(self, "_terminal_reset_state", None)
        if callable(terminal_reset):
            terminal_reset()

        self._active_controller_seeds = []
        self._alignment_seed_ids = []
        self._stable_raw_question = str(question or "")

        options = self._question_options(question) or {}
        propositions = {}
        normalize = getattr(self, "_normalize_candidate_propositions", None)
        if callable(normalize):
            propositions = normalize(options) or {}
        self._last_candidate_propositions = dict(propositions)
        self._last_proposition_probe_coverage = {
            str(pid): [] for pid in propositions
        }

        hints = {
            "dates": list(getattr(frame, "dates", ()) or ()),
            "source_speaker": getattr(frame, "speaker_role", ""),
            "explicit_entities": list(getattr(frame, "entities", ()) or ()),
        }
        prompt = STABLE_SEMANTIC_POLICY + "\n" + STABLE_SEMANTIC_SCHEMA.format(
            question=question,
            candidate_set=json.dumps(propositions, ensure_ascii=False),
            hints=json.dumps(hints, ensure_ascii=False),
        )
        raw_ir: Dict[str, Any] = {}
        error = ""
        try:
            response = self._llm_client.chat(
                [{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=self.CONTROLLER_MAX_OUTPUT_TOKENS,
                response_format={"type": "json_object"},
            )
            usage = self._response_usage(response, prompt)
            raw_ir = self._parse_json(response.content)
        except Exception as exc:
            usage = {}
            error = str(exc)

        sanitized = self._stable_sanitize_controller_ir(
            raw_ir, question, propositions
        )
        ir = self._rg_normalize_controller_ir(sanitized, question, frame)
        ir = self._stable_normalize_internal_ir(ir, propositions)

        self._active_requirement_graph = {
            "version": self.STABLE_SEMANTIC_IR_VERSION,
            "goal": deepcopy(ir.get("goal") or {}),
            "requirements": [
                deepcopy(item) for item in (ir.get("requirements") or [])
            ],
            "links": deepcopy(ir.get("query_links") or []),
        }
        self._active_query_links = deepcopy(ir.get("query_links") or [])

        plan = self._controller_plan(ir, question, frame)
        shape = ir.get("query_shape") or self._derive_query_shape(ir)
        telemetry = {
            "called": True,
            "error": error,
            "usage": usage,
            "requested_projection": ir.get("answer_type", "TEXT"),
            "answer_type": ir.get("answer_type", "TEXT"),
            "answer_mode": shape.get("reasoning_form"),
            "answer_mode_source": "deterministic_stable_semantic_ir",
            "query_shape": deepcopy(shape),
            "requirement_count": len(ir.get("requirements") or []),
            "bridge_count": len(ir.get("query_links") or []),
            "relation_count": len(ir.get("query_links") or []),
            "semantic_ir": self._rc_public_ir(ir),
            "requirement_graph": deepcopy(self._active_requirement_graph),
            "controller_raw_ir": raw_ir,
            "controller_sanitized_ir": sanitized,
            "normalized_ir": self._rc_public_ir(ir),
            "normalization_status": ir.get("normalization_status", "VALID"),
            "normalization_actions": list(ir.get("normalization_actions") or []),
            "graph_validation": ir.get("graph_validation") or {},
            "candidate_authorization": "POST_RETRIEVAL_CERTIFICATION_ONLY",
            "candidate_set_count": len(propositions),
            "candidate_proposition_count": len(propositions),
            "controller_schema_version": self.CONTROLLER_SCHEMA_VERSION,
            "stable_semantic_ir_version": self.STABLE_SEMANTIC_IR_VERSION,
            "memory_context_used": False,
            "route": "PLAN",
            "route_source": "stable_semantic_intent_requires_evidence",
            "answer": "",
            "support_ref": "",
            "support_refs": [],
            "terminal_rendered": False,
            "fallback_reason": error,
        }
        return None, plan, telemetry

    def _er_requirement_views(
        self, slot: Dict[str, Any], question: str
    ) -> List[Dict[str, Any]]:
        """Minimal recall rails: need first; runtime keys second; exact question span last."""
        del question
        need = str(
            slot.get("need")
            or slot.get("evidence_family")
            or slot.get("retrieval_target")
            or ""
        ).strip()
        keys = " ".join(
            self._rg_unique_text(
                [
                    *(slot.get("resolved_keys") or []),
                    *(slot.get("material_anchors") or []),
                ],
                limit=6,
            )
        )
        span = str(
            slot.get("question_span")
            or slot.get("proof_question_span")
            or slot.get("target_surface")
            or ""
        ).strip()
        raw = [
            ("need", need, 1.00),
            ("keys", keys, 0.90),
            ("question_span", span, 0.65),
        ]
        output, seen = [], set()
        for kind, text, weight in raw:
            normalized = self._rc_text(text)
            if normalized and normalized not in seen:
                output.append({"kind": kind, "query": text, "weight": weight})
                seen.add(normalized)
        return output

    def _rc_search_query(self, slot: Dict[str, Any], question: str) -> str:
        """Selector-neutral search query. Full question is handled by independent raw rails."""
        del question
        need = str(
            slot.get("need")
            or slot.get("evidence_family")
            or slot.get("retrieval_target")
            or slot.get("target_surface")
            or ""
        ).strip()
        keys = " ".join(
            str(key)
            for key in (slot.get("resolved_keys") or [])[:2]
            if str(key or "").strip()
        )
        parts, seen = [], set()
        for value in (need, keys):
            key = self._rc_text(value)
            if value and key and key not in seen:
                parts.append(value)
                seen.add(key)
        return " | ".join(parts)

    def _rc_bundle_query(self, slots: List[Dict[str, Any]], question: str) -> str:
        del question
        return " | ".join(
            value
            for value in dict.fromkeys(
                self._rc_search_query(slot, "") for slot in (slots or [])
            )
            if value
        )

    def _compile_gap_operations(
        self, slots, question, budget_tier="MEDIUM", plan=None
    ):
        """Compile stable IR directly: acquisition always precedes deterministic selection."""
        del budget_tier
        plan = plan or {}
        slots = list(slots or [])
        if not slots:
            return []

        options = plan.get("visible_options") or self._question_options(question) or {}
        if options:
            return [
                {
                    "op": "SEARCH_FAMILY",
                    "query": self._rc_bundle_query(slots, ""),
                    "top_k": 8,
                    "family_mode": "candidate_set",
                    "candidate_queries": [
                        {"label": str(label), "query": str(text)}
                        for label, text in options.items()
                    ],
                    "produces": [slot["id"] for slot in slots],
                }
            ]

        operations: List[Dict[str, Any]] = []
        producer_for: Dict[str, int] = {}
        for slot in slots:
            sid = str(slot.get("id") or "")
            if not sid:
                continue
            slot_type = str(slot.get("type") or "DIRECT").upper()
            query = self._rc_search_query(slot, "")
            if slot_type == "CURRENT_STATE":
                producer_for[sid] = len(operations)
                operations.append(
                    {"op": "RESOLVE_STATE", "query": query, "produces": [sid]}
                )
                continue

            if slot_type == "TEMPORAL":
                relation = str(
                    slot.get("temporal_relation")
                    or slot.get("time_relation")
                    or (slot.get("selector") or {}).get("relation")
                    or "LOCATE"
                ).upper()
                axis = str(
                    slot.get("time_axis")
                    or (slot.get("selector") or {}).get("axis")
                    or "event_time"
                ).lower()
                family_mode = (
                    "temporal_extremum"
                    if relation in {"EARLIEST", "LATEST"}
                    else "anchor"
                )
                producer_for[sid] = len(operations)
                family = {
                    "op": "SEARCH_FAMILY",
                    "query": query,
                    "top_k": 8,
                    "family_mode": family_mode,
                    "produces": [sid],
                    "requirement_id": sid,
                    "retrieval_views": self._er_requirement_views(slot, ""),
                    "retrieval_view_semantics": "stable_need_first_views",
                }
                if family_mode == "temporal_extremum":
                    family["axis"] = axis
                operations.append(family)
                operations.append(
                    {
                        "op": "SELECT",
                        "query": query,
                        "relation": relation,
                        "axis": axis,
                        "fallback_axis": "",
                        "anchor": str(slot.get("time_anchor") or ""),
                        "end": str(slot.get("time_end") or ""),
                        "candidate_refs": [f"${producer_for[sid]}"],
                        "produces": [sid],
                        "deterministic_transform": True,
                    }
                )
                continue

            producer_for[sid] = len(operations)
            operations.append(
                {
                    "op": "SEARCH_FAMILY",
                    "query": query,
                    "top_k": 6,
                    "family_mode": "semantic",
                    "produces": [sid],
                    "requirement_id": sid,
                    "retrieval_views": self._er_requirement_views(slot, ""),
                    "retrieval_view_semantics": "stable_need_first_views",
                }
            )

        for relation in plan.get("semantic_relations") or []:
            if str(relation.get("type") or "").upper() != "CAUSES":
                continue
            source = str(relation.get("from") or "")
            target = str(relation.get("to") or "")
            if source not in producer_for or not target:
                continue
            operations.append(
                {
                    "op": "EXPAND_RELATION",
                    "relation": "CAUSES",
                    "start": [f"${producer_for[source]}"],
                    "direction": "OUT",
                    "max_hops": 1,
                    "produces": [source, target],
                }
            )

        verify_ids = {
            str(relation.get("from") or "")
            for relation in plan.get("semantic_relations") or []
            if str(relation.get("type") or "").upper() == "VERIFY_SOURCE"
        }
        for sid in verify_ids:
            producer = producer_for.get(sid)
            if producer is not None:
                operations.append(
                    {
                        "op": "VERIFY_SOURCE",
                        "memory_refs": [f"${producer}"],
                        "produces": [sid],
                        "deterministic_transform": True,
                    }
                )
        return operations

    def _fusion_multiview(self, operation, outputs, seeds, frame):
        """Fuse semantic need views with independently acquired raw BM25/dense heads."""
        rows, relations, evidence_refs = super()._fusion_multiview(
            operation, outputs, seeds, frame
        )
        question = str(getattr(self, "_stable_raw_question", "") or "").strip()
        if not question:
            return rows, relations, evidence_refs
        include_history = str(operation.get("family_mode") or "").lower() in {
            "anchor",
            "temporal_extremum",
        }
        eligible_ids = {
            str(memory.get("id") or "")
            for memory in getattr(self, "_memories", []) or []
            if memory.get("id")
            and self._memory_satisfies_frame(
                memory,
                frame,
                include_entities=bool(getattr(frame, "hard_entities", ()) or ()),
            )
            and self._query_visible_memory(memory, include_history=include_history)
        }
        raw_rows = self._rg_raw_question_channel_candidates(question, eligible_ids)
        ordered, seen = [], set()

        def add(memory):
            if not memory:
                return
            mid = str(memory.get("id") or "")
            if mid and mid not in seen:
                ordered.append(deepcopy(memory))
                seen.add(mid)

        if rows:
            add(rows[0])
        for memory in raw_rows:
            add(memory)
        for memory in rows or []:
            add(memory)
        limit = max(1, int(operation.get("top_k", 6) or 6))
        return ordered[:limit], relations or [], evidence_refs or []

    def _slot_covered(self, slot, support_ids, selected, relations):
        return self._slot_structure_covered(slot, support_ids, selected, relations)

    def _retrieval_status(self, plan, slot_support, selected, relations):
        """Retrieval completeness is structural viability, never semantic proof."""
        requirement_status = {}
        for slot in (plan or {}).get("required_slots") or []:
            sid = str(slot.get("id") or "")
            supports = list((slot_support or {}).get(sid) or [])
            requirement_status[sid] = (
                "FOUND"
                if supports
                and self._slot_structure_covered(
                    slot, supports, selected or [], relations or []
                )
                else "EMPTY"
            )
        relation_status = ReadExecutionContractMixin._relation_status_map(
            self, plan or {}, slot_support or {}, selected or [], relations or []
        )
        complete = bool(requirement_status) and all(
            value == "FOUND" for value in requirement_status.values()
        )
        self._stable_last_requirement_status = dict(requirement_status)
        self._last_retrieval_viability = {
            "requirements": dict(requirement_status),
            "relations": dict(relation_status),
            "complete": bool(complete),
            "semantics": "structural_candidate_viability_not_proof",
        }
        return requirement_status, relation_status, bool(complete)

    def _make_deterministic_recovery_plan(
        self, missing_slots, question, existing_plan
    ):
        """Recover only structurally empty requirements; relation/proof misses never retry."""
        statuses = getattr(self, "_stable_last_requirement_status", {}) or {}
        structural_missing = [
            slot
            for slot in (missing_slots or [])
            if statuses.get(str(slot.get("id") or "")) == "EMPTY"
        ]
        if not structural_missing:
            return None
        return super()._make_deterministic_recovery_plan(
            structural_missing, question, existing_plan
        )

    def _requirement_target_proof(self, slot, memory):
        """Memoize one proof decision per requirement-memory pair for this query."""
        cache = getattr(self, "_stable_proof_cache", None)
        if cache is None:
            cache = self._stable_proof_cache = {}
        key = (str(slot.get("id") or ""), str(memory.get("id") or ""))
        if key in cache:
            self._stable_proof_cache_hits = (
                int(getattr(self, "_stable_proof_cache_hits", 0) or 0) + 1
            )
            return cache[key]
        value = bool(super()._requirement_target_proof(slot, memory))
        cache[key] = value
        return value

    def _prepare_requirement_context_state(self, run, initial_seeds):
        """Run the proof checkpoint once without replacing structural support."""
        slot_support = run.setdefault("slot_support", {})
        original_support = {
            str(key): list(value or [])
            for key, value in slot_support.items()
            if str(key) != "__answer_context_candidates__"
        }
        before_ids = {
            str(memory.get("id") or "")
            for memory in (
                *(run.get("operation_candidates") or []),
                *(run.get("beliefs") or []),
                *(run.get("planning_seeds") or []),
                *(initial_seeds or []),
            )
            if memory and memory.get("id")
        }
        prepared = super()._prepare_requirement_context_state(run, initial_seeds)

        candidate_by_id = {}
        for memory in (
            *(prepared.get("operation_candidates") or []),
            *(prepared.get("beliefs") or []),
            *(prepared.get("planning_seeds") or []),
            *(initial_seeds or []),
        ):
            if memory and memory.get("id"):
                candidate_by_id[str(memory["id"])] = memory
        proof_support = prepared.setdefault("requirement_proof_support", {})
        context_candidates = prepared.get("requirement_context_candidates") or {}
        relations = prepared.get("relations") or []
        slots = {
            str(slot.get("id") or ""): slot
            for slot in (prepared.get("plan") or {}).get("required_slots") or []
            if slot.get("id")
        }
        for sid, slot in slots.items():
            certified = []
            for memory_id in context_candidates.get(sid, []) or []:
                memory = candidate_by_id.get(str(memory_id))
                if not memory:
                    continue
                if not self._requirement_target_proof(slot, memory):
                    continue
                if self._slot_structure_covered(
                    slot, [str(memory_id)], [memory], relations
                ):
                    certified.append(str(memory_id))
            proof_support[sid] = list(dict.fromkeys(certified))

        support_map = prepared.setdefault("slot_support", {})
        for key, values in original_support.items():
            support_map[key] = list(values)

        after_ids = {
            str(memory.get("id") or "")
            for memory in (
                *(prepared.get("operation_candidates") or []),
                *(prepared.get("beliefs") or []),
                *(prepared.get("planning_seeds") or []),
                *(initial_seeds or []),
            )
            if memory and memory.get("id")
        }
        checkpoint = {
            "version": self.PROOF_CHECKPOINT_VERSION,
            "checks": len(getattr(self, "_stable_proof_cache", {}) or {}),
            "cache_hits": int(getattr(self, "_stable_proof_cache_hits", 0) or 0),
            "proven_pairs": sum(len(values or []) for values in proof_support.values()),
            "candidate_world_unchanged": before_ids == after_ids,
            "structural_support_restored": True,
            "retrieval_authority": False,
            "pruning_authority": False,
            "expansion_authority": False,
        }
        self._last_stable_proof_checkpoint = deepcopy(checkpoint)
        prepared["proof_checkpoint"] = checkpoint
        return prepared

    def _role_aware_support_ids(
        self, slots, slot_support, candidate_order, limit
    ):
        """Coverage-first context selection; proof can promote elsewhere but never evict."""
        bounded = max(0, int(limit))
        if not bounded:
            return []
        allowed = [str(mid) for mid in candidate_order or [] if str(mid)]
        allowed_set = set(allowed)
        selected: List[str] = []

        def add(mid: str):
            mid = str(mid or "")
            if (
                mid
                and mid in allowed_set
                and mid not in selected
                and len(selected) < bounded
            ):
                selected.append(mid)

        seen_slots = set()
        contexts = getattr(self, "_last_requirement_context_candidates", {}) or {}
        for slot in slots or []:
            sid = str(slot.get("id") or "")
            if not sid or sid in seen_slots:
                continue
            seen_slots.add(sid)
            candidates = [
                *((slot_support or {}).get(sid, []) or []),
                *((contexts.get(sid) or [])),
            ]
            for mid in candidates:
                if str(mid) in allowed_set:
                    add(str(mid))
                    break

        local = getattr(self, "_last_candidate_local_coverage", {}) or {}
        propositions = getattr(self, "_last_candidate_propositions", {}) or {}
        for pid in propositions:
            for mid in local.get(str(pid), []) or []:
                if str(mid) in allowed_set:
                    add(str(mid))
                    break

        for mid in allowed:
            add(mid)
        return selected

    def _post_retrieval_closure(self, plan, candidates, frame, relations):
        """Disable per-operation proof/terminal checks; final closure is invoked once later."""
        return None

    def _run_query_retrieval(
        self,
        question,
        initial_seeds,
        frame,
        fast_supports,
        gate,
        planning_seeds=None,
        planning_context=None,
    ):
        self._stable_proof_cache = {}
        self._stable_proof_cache_hits = 0
        self._last_stable_proof_checkpoint = {}
        self._stable_last_requirement_status = {}
        run = super()._run_query_retrieval(
            question,
            initial_seeds,
            frame,
            fast_supports,
            gate,
            planning_seeds=planning_seeds,
            planning_context=planning_context,
        )

        candidate_by_id = {}
        for memory in (
            *(run.get("operation_candidates") or []),
            *(run.get("beliefs") or []),
        ):
            if memory and memory.get("id"):
                candidate_by_id[str(memory["id"])] = memory
        final_closure = super()._post_retrieval_closure(
            run.get("plan") or {},
            list(candidate_by_id.values()),
            frame,
            run.get("relations") or [],
        )
        checkpoint = dict(
            getattr(self, "_last_stable_proof_checkpoint", {}) or {}
        )
        checkpoint["terminal_checked_once"] = True
        checkpoint["terminal_closed"] = bool(final_closure)
        self._last_stable_proof_checkpoint = checkpoint
        run["proof_checkpoint"] = deepcopy(checkpoint)
        if final_closure:
            run["atomic_closure"] = deepcopy(final_closure)
            run["precomputed_answer"] = str(final_closure.get("answer") or "")
        return run

    def prepare_batch_query(self, question, system_message=None, **kwargs):
        prepared = super().prepare_batch_query(
            question, system_message=system_message, **kwargs
        )
        extra = prepared.setdefault("extra", {})
        extra["stable_read_architecture_version"] = (
            self.STABLE_READ_ARCHITECTURE_VERSION
        )
        extra["stable_semantic_ir_version"] = self.STABLE_SEMANTIC_IR_VERSION
        extra["controller_memory_context_used"] = False
        extra["proof_checkpoint"] = deepcopy(
            getattr(self, "_last_stable_proof_checkpoint", {}) or {}
        )
        extra["proof_checkpoint_semantics"] = {
            "version": self.PROOF_CHECKPOINT_VERSION,
            "post_retrieval_only": True,
            "non_destructive": True,
            "can_trigger_retrieval": False,
            "can_prune_candidate_world": False,
        }
        extra["active_read_version_stack"] = {
            "architecture": self.STABLE_READ_ARCHITECTURE_VERSION,
            "semantic_ir": self.STABLE_SEMANTIC_IR_VERSION,
            "controller_schema": self.CONTROLLER_SCHEMA_VERSION,
            "requirement_identity": getattr(
                self, "REQUIREMENT_IDENTITY_VERSION", ""
            ),
            "proof": getattr(self, "PROJECTION_PROOF_VERSION", ""),
            "retrieval_executor": getattr(self, "RETRIEVAL_EXECUTOR_VERSION", ""),
            "proof_context": getattr(self, "PROOF_CONTEXT_VERSION", ""),
        }
        return prepared
