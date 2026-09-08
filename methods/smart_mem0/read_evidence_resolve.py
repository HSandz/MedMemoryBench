"""Evidence Resolve integrity and binding for SmartMem0's two-stage READ architecture.

This component replaces the old conceptual "Direct B" with one evidence-resolution path.
It preserves semantically useful controller work monotonically and binds each semantic
requirement to evidence through bounded independent recall views. Graph quality is reported
as VALID/PARTIAL/DEGRADED; unresolved semantics are left for deterministic retrieval plus the
optional LLM #2 synthesis stage. No model call is added.
"""

from copy import deepcopy
from typing import Any, Dict, Iterable, List


class ReadEvidenceResolveMixin:
    """Keep Certified Direct narrow and make Evidence Resolve requirement-local."""

    EVIDENCE_RESOLVE_VERSION = "evidence-resolve-v2-binding"
    REQUIREMENT_BINDING_VERSION = "requirement-binding-v1"
    CERTIFIED_DIRECT_PATH = "CERTIFIED_DIRECT"
    EVIDENCE_RESOLVE_PATH = "EVIDENCE_RESOLVE"
    REQUIREMENT_VIEW_RRF_K = 60.0
    REQUIREMENT_VIEW_WEIGHTS = {
        "obligation": 1.00,
        "family": 0.85,
        "keys": 0.75,
        "question": 0.45,
    }
    REQUIREMENT_LOCAL_QUOTA = 2
    CANDIDATE_LOCAL_QUOTA = 2

    def _er_exact_surface(self, value: Any, question: str) -> str:
        text = " ".join(str(value or "").split()).strip()
        if not text:
            return ""
        return str(self._rc_question_span(text, question) or "").strip()

    @classmethod
    def _er_unique_text(cls, values: Iterable[Any]) -> List[str]:
        output: List[str] = []
        seen = set()
        for value in values:
            text = " ".join(str(value or "").split()).strip()
            if not text:
                continue
            key = cls._rc_text(text)
            if key and key not in seen:
                output.append(text)
                seen.add(key)
        return output

    def _aop_vnext_to_legacy(self, parsed: Dict[str, Any], question: str) -> Dict[str, Any]:
        """Bind QUESTION nodes conservatively without changing their semantic obligation."""
        lean = super()._aop_vnext_to_legacy(parsed, question)
        if not isinstance(parsed, dict):
            return lean
        raw_requirements = parsed.get("requirements") if isinstance(parsed.get("requirements"), list) else []
        raw_by_id = {
            str(raw.get("id") or f"r{index + 1}"): raw
            for index, raw in enumerate(raw_requirements[:4]) if isinstance(raw, dict)
        }
        subject_surface = self._er_exact_surface(parsed.get("subject_span"), question)
        stem = self._question_stem(question).strip()
        stem_surface = self._er_exact_surface(stem, question) or stem
        for requirement in lean.get("requirements") or []:
            if str(requirement.get("grounding_kind") or "").upper() != "QUESTION":
                continue
            requirement_id = str(requirement.get("id") or "")
            raw = raw_by_id.get(requirement_id, {})
            obligation = str(raw.get("answer_obligation") or raw.get("focus_span") or requirement.get("focus_span") or "").strip()
            exact = self._er_exact_surface(obligation, question)
            if exact:
                requirement["focus_span"] = exact
                continue
            binder = subject_surface or stem_surface
            if binder:
                requirement["focus_span"] = binder
                requirement["_semantic_focus_binding"] = "PARAPHRASED_QUESTION_OBLIGATION"
        return lean

    @staticmethod
    def _er_graph_state(ir: Dict[str, Any]) -> str:
        status = str(ir.get("normalization_status") or "").upper()
        validation = ir.get("graph_validation") or {}
        requirements = list(ir.get("requirements") or [])
        if status == "DEGRADED" or any(req.get("degraded") for req in requirements):
            return "DEGRADED"
        if not requirements:
            return "DEGRADED"
        if validation.get("valid") is True:
            return "VALID"
        return "PARTIAL"

    def _rc_normalize_ir(self, parsed: Dict[str, Any], question: str, frame: Any):
        ir = super()._rc_normalize_ir(parsed, question, frame)
        graph_state = self._er_graph_state(ir)
        ir["graph_state"] = graph_state
        actions = list(ir.get("normalization_actions") or [])
        normalized_ids = {str(req.get("id") or "") for req in ir.get("requirements") or []}
        raw_requirements = parsed.get("requirements") if isinstance(parsed, dict) and isinstance(parsed.get("requirements"), list) else []
        for index, raw in enumerate(raw_requirements[:4]):
            if not isinstance(raw, dict):
                continue
            requirement_id = str(raw.get("id") or f"r{index + 1}")
            if requirement_id not in normalized_ids or str(raw.get("grounding_kind") or "QUESTION").upper() != "QUESTION":
                continue
            obligation = str(raw.get("answer_obligation") or raw.get("focus_span") or "").strip()
            if obligation and not self._er_exact_surface(obligation, question):
                marker = {"action": "PRESERVE_SEMANTIC_REQUIREMENT", "reason": "PARAPHRASED_QUESTION_OBLIGATION", "requirement_id": requirement_id}
                if marker not in actions:
                    actions.append(marker)
        ir["normalization_actions"] = actions
        ir["graph_integrity_semantics"] = "VALID_KEEP;PARTIAL_KEEP_VALID_COMPONENTS_FOR_SYNTHESIS;DEGRADED_USE_DETERMINISTIC_EVIDENCE_RECOVERY"
        return ir

    def _er_requirement_views(self, slot: Dict[str, Any], question: str) -> List[Dict[str, Any]]:
        obligation = str(slot.get("answer_obligation") or slot.get("proof_anchor") or slot.get("target_surface") or "").strip()
        family = str(slot.get("evidence_family") or slot.get("retrieval_hint") or slot.get("retrieval_target") or "").strip()
        keys = " ".join(str(value).strip() for value in (slot.get("resolved_keys") or [])[:2] if str(value).strip())
        question_surface = self._question_stem(question).strip() if question else ""
        raw = [("obligation", obligation), ("family", family), ("keys", keys), ("question", question_surface)]
        views, seen = [], set()
        for kind, text in raw:
            normalized = self._rc_text(text)
            if not normalized or normalized in seen:
                continue
            views.append({"kind": kind, "query": text, "weight": float(self.REQUIREMENT_VIEW_WEIGHTS[kind])})
            seen.add(normalized)
        return views

    def _compile_gap_operations(self, slots, question, budget_tier="MEDIUM", plan=None):
        operations = list(super()._compile_gap_operations(slots, question, budget_tier, plan=plan))
        slot_by_id = {str(slot.get("id") or ""): slot for slot in (slots or []) if slot.get("id")}
        for operation in operations:
            if str(operation.get("op") or "").upper() != "SEARCH_FAMILY":
                continue
            mode = str(operation.get("family_mode") or "semantic").lower()
            if mode not in {"semantic", "anchor"}:
                continue
            produced = [str(value) for value in (operation.get("produces") or []) if value]
            if len(produced) != 1 or produced[0] not in slot_by_id:
                continue
            views = self._er_requirement_views(slot_by_id[produced[0]], question)
            if len(views) <= 1:
                continue
            operation["requirement_id"] = produced[0]
            operation["retrieval_views"] = views
            operation["retrieval_view_semantics"] = "independent_views_fused_inside_one_search_family_operation"
        return operations

    def _execute_operation(self, operation, outputs, seeds, frame=None):
        views = list(operation.get("retrieval_views") or [])
        if str(operation.get("op") or "").upper() != "SEARCH_FAMILY" or not views:
            return super()._execute_operation(operation, outputs, seeds, frame)
        requirement_id = str(operation.get("requirement_id") or "")
        top_k = max(1, int(operation.get("top_k", 6) or 6))
        fused, view_coverage, relations, evidence_refs = {}, {}, [], []
        for view in views:
            query = str(view.get("query") or "").strip()
            kind = str(view.get("kind") or "view")
            if not query:
                continue
            weight = float(view.get("weight", 1.0) or 1.0)
            child = deepcopy(operation)
            child.pop("retrieval_views", None)
            child.pop("retrieval_view_semantics", None)
            child.pop("requirement_id", None)
            child["query"] = query
            result, child_relations, child_evidence = super()._execute_operation(child, outputs, seeds, frame)
            ids = []
            for rank, memory in enumerate(result or [], start=1):
                memory_id = str(memory.get("id") or "")
                if not memory_id:
                    continue
                ids.append(memory_id)
                entry = fused.setdefault(memory_id, {"memory": deepcopy(memory), "score": 0.0, "views": [], "best_rank": rank})
                entry["score"] += weight / (self.REQUIREMENT_VIEW_RRF_K + rank)
                entry["best_rank"] = min(int(entry["best_rank"]), rank)
                if kind not in entry["views"]:
                    entry["views"].append(kind)
            view_coverage[kind] = ids
            for relation in child_relations or []:
                if relation not in relations:
                    relations.append(relation)
            for reference in child_evidence or []:
                if reference not in evidence_refs:
                    evidence_refs.append(reference)
        ordered = sorted(fused.values(), key=lambda item: (-float(item["score"]), int(item["best_rank"]), str(item["memory"].get("id") or "")))
        result, binding_scores, binding_views = [], {}, {}
        for item in ordered[:top_k]:
            memory = deepcopy(item["memory"])
            memory["_binding_score"] = round(float(item["score"]), 8)
            memory["_binding_views"] = list(item["views"])
            result.append(memory)
            memory_id = str(memory.get("id") or "")
            binding_scores[memory_id] = float(item["score"])
            binding_views[memory_id] = list(item["views"])
        if requirement_id:
            self._last_requirement_view_coverage[requirement_id] = deepcopy(view_coverage)
            self._last_requirement_binding_scores[requirement_id] = binding_scores
            self._last_requirement_binding_views[requirement_id] = binding_views
        return result, relations, evidence_refs

    def _run_query_retrieval(self, question, initial_seeds, frame, fast_supports, gate, planning_seeds=None, planning_context=None):
        self._last_requirement_view_coverage = {}
        self._last_requirement_binding_scores = {}
        self._last_requirement_binding_views = {}
        run = super()._run_query_retrieval(question, initial_seeds, frame, fast_supports, gate, planning_seeds=planning_seeds, planning_context=planning_context)
        run["requirement_view_coverage"] = deepcopy(self._last_requirement_view_coverage)
        run["requirement_binding_scores"] = deepcopy(self._last_requirement_binding_scores)
        run["requirement_binding_version"] = self.REQUIREMENT_BINDING_VERSION
        return run

    @staticmethod
    def _er_has_selector(slot: Dict[str, Any]) -> bool:
        relation = str(slot.get("time_relation") or slot.get("temporal_relation") or (slot.get("selector") or {}).get("relation") or "").upper()
        return bool(relation) or str(slot.get("type") or "").upper() in {"TEMPORAL", "CURRENT_STATE"}

    def _role_aware_support_ids(self, slots, slot_support, candidate_order, limit):
        selected = list(super()._role_aware_support_ids(slots, slot_support, candidate_order, limit))
        bounded_limit = max(0, int(limit))
        if not bounded_limit:
            return []
        allowed, ordered = set(candidate_order), []
        def add(memory_id: str):
            memory_id = str(memory_id or "")
            if memory_id and memory_id in allowed and memory_id not in ordered and len(ordered) < bounded_limit:
                ordered.append(memory_id)
        eligible_slots = [slot for slot in (slots or []) if not self._er_has_selector(slot)]
        binding_scores = getattr(self, "_last_requirement_binding_scores", {}) or {}
        for depth in range(self.REQUIREMENT_LOCAL_QUOTA):
            for slot in eligible_slots:
                rid = str(slot.get("id") or "")
                ranked = sorted(((str(memory_id), float(score)) for memory_id, score in (binding_scores.get(rid) or {}).items() if str(memory_id) in allowed), key=lambda pair: (-pair[1], pair[0]))
                if depth < len(ranked):
                    add(ranked[depth][0])
        local_map = getattr(self, "_last_candidate_local_coverage", {}) or {}
        proposition_order = list((getattr(self, "_last_candidate_propositions", {}) or {}).keys())
        for depth in range(self.CANDIDATE_LOCAL_QUOTA):
            for candidate_id in proposition_order:
                local = [str(memory_id) for memory_id in (local_map.get(str(candidate_id)) or []) if str(memory_id) in allowed]
                if depth < len(local):
                    add(local[depth])
        for memory_id in selected:
            add(memory_id)
        for memory_id in candidate_order:
            add(memory_id)
        return ordered[:bounded_limit]

    def _terminal_render_answer(self, question: str, answer_type: str, memory: Dict[str, Any], *, candidate_answer: str = "", answer_field: str = "") -> str:
        rendered = super()._terminal_render_answer(question, answer_type, memory, candidate_answer=candidate_answer, answer_field=answer_field)
        answer_type = str(answer_type or "TEXT").upper()
        candidate_answer = " ".join(str(candidate_answer or "").split())
        if answer_type != "VALUE" or not candidate_answer or answer_field or any(character.isdigit() for character in candidate_answer):
            return rendered
        claim = " ".join(str(memory.get("claim") or "").split())
        if not claim:
            return rendered
        scalar_match = getattr(self, "_terminal_candidate_is_scalar_surface", None)
        if not callable(scalar_match) or not scalar_match(candidate_answer, memory):
            return rendered
        char_counter = getattr(self, "_rq_surface_chars", None)
        candidate_len = len(char_counter(candidate_answer)) if callable(char_counter) else len(candidate_answer)
        claim_len = len(char_counter(claim)) if callable(char_counter) else len(claim)
        if claim_len > max(18, int(1.35 * max(1, candidate_len))):
            return claim
        return rendered

    def _controller_plan(self, ir, question, frame):
        plan = super()._controller_plan(ir, question, frame)
        legacy_shape = str(plan.get("compiled_mode") or plan.get("query_mode") or "")
        program_shape = "ATOMIC" if legacy_shape == "DIRECT" else legacy_shape or "UNKNOWN"
        plan["resolution_path"] = self.EVIDENCE_RESOLVE_PATH
        plan["program_shape"] = program_shape
        plan["legacy_compiled_mode"] = legacy_shape
        plan["evidence_resolve_version"] = self.EVIDENCE_RESOLVE_VERSION
        plan.setdefault("query_spec", {})["resolution_path"] = self.EVIDENCE_RESOLVE_PATH
        plan.setdefault("semantic_ir", {})["graph_state"] = ir.get("graph_state")
        plan["semantic_ir"]["graph_integrity_semantics"] = ir.get("graph_integrity_semantics")
        return plan

    def _semantic_controller(self, question, seeds, frame, context_map=None):
        supports, plan, telemetry = super()._semantic_controller(question, seeds, frame, context_map=context_map)
        telemetry = dict(telemetry or {})
        if supports is not None:
            resolution_path, program_shape = self.CERTIFIED_DIRECT_PATH, "CERTIFIED_ATOMIC"
        else:
            resolution_path = self.EVIDENCE_RESOLVE_PATH
            program_shape = str((plan or {}).get("program_shape") or "UNKNOWN")
        telemetry["resolution_path"] = resolution_path
        telemetry["program_shape"] = program_shape
        telemetry["evidence_resolve_version"] = self.EVIDENCE_RESOLVE_VERSION
        telemetry["graph_state"] = (plan or {}).get("semantic_ir", {}).get("graph_state") or telemetry.get("graph_validation", {}).get("state") or ""
        return supports, plan, telemetry

    def _er_binding_block(self, prepared: Dict[str, Any]) -> str:
        extra = prepared.get("extra") or {}
        plan = extra.get("replan") or extra.get("plan") or {}
        requirements = list((plan.get("semantic_ir") or {}).get("requirements") or [])
        if not requirements:
            return ""
        final = {str(memory.get("id") or ""): memory for memory in (prepared.get("retrieved_memories") or []) if memory.get("id")}
        scores = getattr(self, "_last_requirement_binding_scores", {}) or {}
        views = getattr(self, "_last_requirement_binding_views", {}) or {}
        lines = ["=== REQUIREMENT BINDING MAP ===", "Evidence retrieved directly through an obligation/family/key view is stronger endpoint context than evidence found only through the whole-question view. Binding is retrieval provenance, not proof or truth."]
        for requirement in requirements:
            rid = str(requirement.get("id") or "")
            label = " ".join(str(requirement.get("answer_obligation") or requirement.get("focus_span") or requirement.get("target") or rid).split())[:220]
            lines.append(f"- {rid}: {label}")
            ranked = sorted(((str(memory_id), float(score)) for memory_id, score in (scores.get(rid) or {}).items() if str(memory_id) in final), key=lambda pair: (-pair[1], pair[0]))
            for memory_id, _score in ranked[:2]:
                claim = " ".join(str(final[memory_id].get("claim") or "").split())[:240]
                labels = ",".join(views.get(rid, {}).get(memory_id, [])) or "unknown"
                lines.append(f"  bound[{labels}]: {claim}")
        if (extra.get("candidate_set") or {}).get("candidates"):
            lines.append("CandidateSet: inspect every candidate independently and combine up to two candidate-local evidence items when needed for a multi-clause proposition. Do not stop after finding the first plausible label; UNKNOWN is not false.")
        return "\n".join(lines)

    def prepare_batch_query(self, question, system_message=None, **kwargs):
        prepared = super().prepare_batch_query(question, system_message=system_message, **kwargs)
        extra = prepared.setdefault("extra", {})
        controller = extra.get("semantic_controller") or {}
        plan = extra.get("replan") or extra.get("plan") or {}
        path = str(controller.get("resolution_path") or plan.get("resolution_path") or (self.CERTIFIED_DIRECT_PATH if prepared.get("precomputed_answer") not in (None, "") else self.EVIDENCE_RESOLVE_PATH))
        extra["resolution_path"] = path
        program_shape = str(controller.get("program_shape") or plan.get("program_shape") or "")
        if not program_shape:
            program_shape = "CERTIFIED_ATOMIC" if path == self.CERTIFIED_DIRECT_PATH else "UNKNOWN"
        extra["program_shape"] = program_shape
        extra["evidence_resolve_version"] = self.EVIDENCE_RESOLVE_VERSION
        extra["requirement_binding_version"] = self.REQUIREMENT_BINDING_VERSION
        extra["requirement_view_coverage"] = deepcopy(getattr(self, "_last_requirement_view_coverage", {}) or {})
        extra["requirement_binding_scores"] = deepcopy(getattr(self, "_last_requirement_binding_scores", {}) or {})
        if path == self.EVIDENCE_RESOLVE_PATH:
            semantic_ir = plan.get("semantic_ir") or {}
            graph_state = str(semantic_ir.get("graph_state") or controller.get("graph_state") or "VALID").upper()
            extra["graph_state"] = graph_state
            block = (
                "EVIDENCE RESOLVE CONTRACT:\n"
                "- Controller requirements/bridges are a query program, never proof.\n"
                "- Preserve grounded participant-specific endpoints and structurally valid bridge components from LLM #1.\n"
                "- A DERIVED node is only a retrieval hypothesis. If retrieved evidence does not ground it as a participant fact, do not pretend it was remembered.\n"
                "- General-domain mechanisms belong in authorized POSSIBLE_CAUSE/INFER reasoning, not in participant memory.\n"
                "- For an action/decision, a supplied participant-specific constraint, prior policy, contraindication, permission, or veto condition can override generic preference/advice and must be considered before recommending the action.\n"
            )
            if graph_state == "PARTIAL":
                block += "- Graph state is PARTIAL: complete missing reasoning only from grounded evidence plus authorized general knowledge; do not erase valid graph components and do not invent participant facts.\n"
            elif graph_state == "DEGRADED":
                block += "- Graph state is DEGRADED: rely on deterministic retrieved evidence and answer conservatively; do not infer missing participant history.\n"
            else:
                block += "- Graph state is VALID: follow the graph as the reasoning obligation, while still requiring retrieved evidence for participant facts.\n"
            binding_block = self._er_binding_block(prepared)
            if binding_block:
                block += "\n" + binding_block + "\n"
            for message in prepared.get("messages") or []:
                if str(message.get("role") or "").lower() == "system":
                    message["content"] = str(message.get("content") or "").rstrip() + "\n\n" + block
                    extra["evidence_resolve_materialized"] = True
                    extra["requirement_binding_materialized"] = bool(binding_block)
                    break
            else:
                extra["evidence_resolve_materialized"] = False
                extra["requirement_binding_materialized"] = False
        else:
            extra["evidence_resolve_materialized"] = False
            extra["requirement_binding_materialized"] = False
        return prepared
