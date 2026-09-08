"""Evidence Resolve integrity for SmartMem0's two-stage READ architecture.

This component replaces the old conceptual "Direct B" with one evidence-resolution path.
It preserves semantically useful controller work monotonically: exact lexical binding is
helpful provenance, but a paraphrased QUESTION obligation is not deleted. Graph quality is
reported as VALID/PARTIAL/DEGRADED and unresolved semantics are left for the existing
retrieval + optional LLM #2 synthesis stage. No model call is added.
"""

from copy import deepcopy
from typing import Any, Dict


class ReadEvidenceResolveMixin:
    """Keep Certified Direct narrow and route everything else through Evidence Resolve."""

    EVIDENCE_RESOLVE_VERSION = "evidence-resolve-v1"
    CERTIFIED_DIRECT_PATH = "CERTIFIED_DIRECT"
    EVIDENCE_RESOLVE_PATH = "EVIDENCE_RESOLVE"

    def _er_exact_surface(self, value: Any, question: str) -> str:
        text = " ".join(str(value or "").split()).strip()
        if not text:
            return ""
        return str(self._rc_question_span(text, question) or "").strip()

    def _aop_vnext_to_legacy(self, parsed: Dict[str, Any], question: str) -> Dict[str, Any]:
        """Bind QUESTION nodes conservatively without changing their semantic obligation.

        The legacy compiler historically required an exact contiguous focus span. LLM #1
        may emit a faithful paraphrase (especially across languages), so use an exact
        question-owned surface only as a structural binder while preserving the original
        answer_obligation in Requirement-vNext metadata.
        """
        lean = super()._aop_vnext_to_legacy(parsed, question)
        if not isinstance(parsed, dict):
            return lean

        raw_requirements = (
            parsed.get("requirements")
            if isinstance(parsed.get("requirements"), list)
            else []
        )
        raw_by_id = {
            str(raw.get("id") or f"r{index + 1}"): raw
            for index, raw in enumerate(raw_requirements[:4])
            if isinstance(raw, dict)
        }
        subject_surface = self._er_exact_surface(parsed.get("subject_span"), question)
        stem = self._question_stem(question).strip()
        stem_surface = self._er_exact_surface(stem, question) or stem

        for requirement in lean.get("requirements") or []:
            if str(requirement.get("grounding_kind") or "").upper() != "QUESTION":
                continue
            requirement_id = str(requirement.get("id") or "")
            raw = raw_by_id.get(requirement_id, {})
            obligation = str(
                raw.get("answer_obligation")
                or raw.get("focus_span")
                or requirement.get("focus_span")
                or ""
            ).strip()
            exact = self._er_exact_surface(obligation, question)
            if exact:
                requirement["focus_span"] = exact
                continue

            # Preserve semantics; only substitute a question-owned lexical binder.
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
        raw_requirements = (
            parsed.get("requirements")
            if isinstance(parsed, dict) and isinstance(parsed.get("requirements"), list)
            else []
        )
        for index, raw in enumerate(raw_requirements[:4]):
            if not isinstance(raw, dict):
                continue
            requirement_id = str(raw.get("id") or f"r{index + 1}")
            if requirement_id not in normalized_ids:
                continue
            if str(raw.get("grounding_kind") or "QUESTION").upper() != "QUESTION":
                continue
            obligation = str(
                raw.get("answer_obligation") or raw.get("focus_span") or ""
            ).strip()
            if obligation and not self._er_exact_surface(obligation, question):
                marker = {
                    "action": "PRESERVE_SEMANTIC_REQUIREMENT",
                    "reason": "PARAPHRASED_QUESTION_OBLIGATION",
                    "requirement_id": requirement_id,
                }
                if marker not in actions:
                    actions.append(marker)

        ir["normalization_actions"] = actions
        ir["graph_integrity_semantics"] = (
            "VALID_KEEP;PARTIAL_KEEP_VALID_COMPONENTS_FOR_SYNTHESIS;"
            "DEGRADED_USE_DETERMINISTIC_EVIDENCE_RECOVERY"
        )
        return ir

    def _controller_plan(self, ir, question, frame):
        plan = super()._controller_plan(ir, question, frame)
        legacy_shape = str(plan.get("compiled_mode") or plan.get("query_mode") or "")
        # "DIRECT" here historically meant one unresolved atomic requirement, not a
        # certified answer. Keep it only as compatibility telemetry and expose the
        # correct semantic path explicitly.
        program_shape = "ATOMIC" if legacy_shape == "DIRECT" else legacy_shape or "UNKNOWN"
        plan["resolution_path"] = self.EVIDENCE_RESOLVE_PATH
        plan["program_shape"] = program_shape
        plan["legacy_compiled_mode"] = legacy_shape
        plan["evidence_resolve_version"] = self.EVIDENCE_RESOLVE_VERSION
        plan.setdefault("query_spec", {})["resolution_path"] = self.EVIDENCE_RESOLVE_PATH
        plan.setdefault("semantic_ir", {})["graph_state"] = ir.get("graph_state")
        plan["semantic_ir"]["graph_integrity_semantics"] = ir.get(
            "graph_integrity_semantics"
        )
        return plan

    def _semantic_controller(self, question, seeds, frame, context_map=None):
        supports, plan, telemetry = super()._semantic_controller(
            question, seeds, frame, context_map=context_map
        )
        telemetry = dict(telemetry or {})
        if supports is not None:
            resolution_path = self.CERTIFIED_DIRECT_PATH
            program_shape = "CERTIFIED_ATOMIC"
        else:
            resolution_path = self.EVIDENCE_RESOLVE_PATH
            program_shape = str((plan or {}).get("program_shape") or "UNKNOWN")

        telemetry["resolution_path"] = resolution_path
        telemetry["program_shape"] = program_shape
        telemetry["evidence_resolve_version"] = self.EVIDENCE_RESOLVE_VERSION
        telemetry["graph_state"] = (
            (plan or {}).get("semantic_ir", {}).get("graph_state")
            or telemetry.get("graph_validation", {}).get("state")
            or ""
        )
        return supports, plan, telemetry

    def prepare_batch_query(self, question, system_message=None, **kwargs):
        prepared = super().prepare_batch_query(
            question, system_message=system_message, **kwargs
        )
        extra = prepared.setdefault("extra", {})
        controller = extra.get("semantic_controller") or {}
        plan = extra.get("replan") or extra.get("plan") or {}
        path = str(
            controller.get("resolution_path")
            or plan.get("resolution_path")
            or (
                self.CERTIFIED_DIRECT_PATH
                if prepared.get("precomputed_answer") not in (None, "")
                else self.EVIDENCE_RESOLVE_PATH
            )
        )
        extra["resolution_path"] = path
        program_shape = str(
            controller.get("program_shape") or plan.get("program_shape") or ""
        )
        if not program_shape:
            program_shape = (
                "CERTIFIED_ATOMIC"
                if path == self.CERTIFIED_DIRECT_PATH
                else "UNKNOWN"
            )
        extra["program_shape"] = program_shape
        extra["evidence_resolve_version"] = self.EVIDENCE_RESOLVE_VERSION

        if path == self.EVIDENCE_RESOLVE_PATH:
            semantic_ir = plan.get("semantic_ir") or {}
            graph_state = str(
                semantic_ir.get("graph_state")
                or controller.get("graph_state")
                or "VALID"
            ).upper()
            extra["graph_state"] = graph_state
            block = (
                "EVIDENCE RESOLVE CONTRACT:\n"
                "- Controller requirements/bridges are a query program, never proof.\n"
                "- Preserve grounded participant-specific endpoints and structurally valid "
                "bridge components from LLM #1.\n"
                "- A DERIVED node is only a retrieval hypothesis. If retrieved evidence does "
                "not ground it as a participant fact, do not pretend it was remembered.\n"
                "- General-domain mechanisms belong in authorized POSSIBLE_CAUSE/INFER "
                "reasoning, not in participant memory.\n"
            )
            if graph_state == "PARTIAL":
                block += (
                    "- Graph state is PARTIAL: complete missing reasoning only from grounded "
                    "evidence plus authorized general knowledge; do not erase valid graph "
                    "components and do not invent participant facts.\n"
                )
            elif graph_state == "DEGRADED":
                block += (
                    "- Graph state is DEGRADED: rely on deterministic retrieved evidence and "
                    "answer conservatively; do not infer missing participant history.\n"
                )
            else:
                block += (
                    "- Graph state is VALID: follow the graph as the reasoning obligation, "
                    "while still requiring retrieved evidence for participant facts.\n"
                )
            for message in prepared.get("messages") or []:
                if str(message.get("role") or "").lower() == "system":
                    message["content"] = (
                        str(message.get("content") or "").rstrip() + "\n\n" + block
                    )
                    extra["evidence_resolve_materialized"] = True
                    break
            else:
                extra["evidence_resolve_materialized"] = False
        else:
            extra["evidence_resolve_materialized"] = False
        return prepared
