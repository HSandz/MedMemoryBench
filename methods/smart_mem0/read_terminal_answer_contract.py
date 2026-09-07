"""Terminal authorization and deterministic rendering for SmartMem0 reads.

DIRECT means semantic completeness, not brevity. This layer never searches, proves by
retrieval similarity, repairs a graph, ranks context, or calls an LLM. Terminal grounding
and rendering are script-agnostic: no English interrogative patterns, stemmers, polarity
lists, or benchmark labels participate in the decision.
"""

import re
from typing import Any, Dict

from .contracts import VALID_TEMPORAL_AXES


class ReadTerminalAnswerContractMixin:
    """Own terminal authorization plus one renderer shared by DIRECT-A/B."""

    TERMINAL_ANSWER_CONTRACT_VERSION = "complete-grounded-answer-v4-language-neutral"

    def _terminal_reset_state(self):
        self._last_terminal_closure_diagnostic = {}

    def _terminal_memory_text(self, memory: Dict[str, Any]) -> str:
        return " ".join(str(value or "") for value in (memory.get("claim"), memory.get("value"), memory.get("verbatim_value"), str(memory.get("object_anchor") or "").replace("_", " "), " ".join(memory.get("entities") or []), " ".join(memory.get("scope_entities") or [])))

    def _terminal_surface_similarity(self, target: Any, surface: Any) -> float:
        similarity = getattr(self, "_rq_surface_similarity", None)
        if callable(similarity):
            return float(similarity(target, surface))
        return 1.0 if self._rc_text(target) == self._rc_text(surface) else 0.0

    def _terminal_answer_grounded(self, answer: str, memory: Dict[str, Any]) -> bool:
        answer = str(answer or "").strip()
        if not answer:
            return False
        answer_norm = self._rc_text(answer)
        for surface in (memory.get("value"), memory.get("verbatim_value"), memory.get("object_anchor")):
            surface_norm = self._rc_text(str(surface or "").replace("_", " "))
            if answer_norm and surface_norm and (answer_norm == surface_norm or answer_norm in surface_norm or surface_norm in answer_norm):
                return True
        evidence = self._terminal_memory_text(memory)
        if self._rc_token_sequence_present(answer, evidence):
            return True
        answer_numbers = set(re.findall(r"\d+(?:\.\d+)?", self._rc_text(answer)))
        evidence_numbers = set(re.findall(r"\d+(?:\.\d+)?", self._rc_text(evidence)))
        if answer_numbers and not answer_numbers.issubset(evidence_numbers):
            return False
        return self._terminal_surface_similarity(answer, evidence) >= 0.92

    def _terminal_candidate_is_scalar_surface(self, candidate_answer: str, memory: Dict[str, Any]) -> bool:
        candidate = self._rc_text(candidate_answer)
        if not candidate:
            return False
        for value in (memory.get("value"), memory.get("verbatim_value"), memory.get("object_anchor")):
            surface = self._rc_text(str(value or "").replace("_", " "))
            if surface and (candidate == surface or candidate in surface or surface in candidate):
                return True
        return False

    def _terminal_render_answer(self, question: str, answer_type: str, memory: Dict[str, Any], *, candidate_answer: str = "", answer_field: str = "") -> str:
        del question
        answer_type = str(answer_type or "TEXT").upper()
        candidate_answer = " ".join(str(candidate_answer or "").split())
        field = str(answer_field or "")
        if field in VALID_TEMPORAL_AXES:
            return " ".join(str(self._date_for(memory, field) or "").split())
        if field in {"value", "verbatim_value", "object_anchor"}:
            return " ".join(str(memory.get(field) or "").replace("_", " ").split())
        if answer_type in {"ENTITY", "VALUE", "DATE"}:
            return candidate_answer
        if answer_type != "TEXT":
            return candidate_answer
        claim = " ".join(str(memory.get("claim") or "").split())
        value = " ".join(str(self._memory_value(memory) or "").split())
        if candidate_answer:
            if claim and self._terminal_candidate_is_scalar_surface(candidate_answer, memory) and len(self._rq_surface_chars(claim)) > max(12, int(1.35 * len(self._rq_surface_chars(candidate_answer)))):
                return claim
            return candidate_answer
        return claim or value

    def _terminal_render_seed_answer(self, question: str, ir: Dict[str, Any], candidate_answer: str, memory: Dict[str, Any]) -> str:
        return self._terminal_render_answer(question, str(ir.get("answer_type") or "TEXT"), memory, candidate_answer=candidate_answer)

    def _authorize_controller_answer(self, ir, seeds, frame):
        supports, reason = super()._authorize_controller_answer(ir, seeds, frame)
        if supports is not None:
            return supports, reason
        candidate = ir.get("candidate")
        requirements = ir.get("requirements") or []
        if str(ir.get("answer_type") or "TEXT").upper() != "TEXT" or not candidate or len(requirements) != 1 or ir.get("visible_options"):
            return None, reason
        relations = ir.get("relations") or []
        if any(str(edge.get("type") or "").upper() != "CURRENT" for edge in relations):
            return None, "RELATIONAL_QUERY_REQUIRES_RETRIEVAL"
        reference = str(candidate.get("support_ref") or "")
        match = re.fullmatch(r"\$seed(\d+)", reference)
        if not match or int(match.group(1)) >= min(3, len(seeds)):
            return None, "INVALID_SUPPORT_REF"
        valid = self._validate_fast_support(reference, seeds, frame)
        if not valid:
            return None, "STRUCTURAL_SUPPORT_REJECTED"
        memory = valid[0]
        constraint = requirements[0].get("time_constraint") or {}
        axis = str(constraint.get("axis") or "")
        relation = str(constraint.get("relation") or "").upper()
        anchor = str(constraint.get("anchor") or "")
        if axis or relation:
            if relation == "EXACT" and axis in VALID_TEMPORAL_AXES and anchor and self._date_matches(self._date_for(memory, axis), anchor):
                pass
            else:
                return None, "TEMPORAL_SELECTOR_REQUIRES_RETRIEVAL"
        elif getattr(frame, "dates", ()) and not self._memory_satisfies_frame(memory, frame):
            return None, "TEMPORAL_FILTER_MISMATCH"
        if relations and not self._is_state_head(memory):
            return None, "CURRENT_CANDIDATE_IS_NOT_STATE_HEAD"
        if not self._terminal_answer_grounded(str(candidate.get("answer") or ""), memory):
            return None, "ANSWER_PROPOSITION_NOT_GROUNDED"
        return valid, "AUTHORIZED_COMPLETE_PROPOSITION"

    def _post_retrieval_closure(self, plan, candidates, frame, relations):
        slots = plan.get("required_slots") or []
        ir = plan.get("semantic_ir") or {}
        diagnostic = {"eligible": False, "reason": "", "candidate_count": 0}
        self._last_terminal_closure_diagnostic = diagnostic
        if len(slots) != 1 or len(ir.get("requirements") or []) != 1 or plan.get("visible_options") or plan.get("need_evidence") or plan.get("query_spec", {}).get("world_knowledge_bridge_allowed"):
            diagnostic["reason"] = "NON_SINGLE_MEMORY_OBLIGATION"
            return None
        if any(str(relation.get("type") or "").upper() != "CURRENT" for relation in plan.get("semantic_relations") or []):
            diagnostic["reason"] = "RELATIONAL_SYNTHESIS_REQUIRED"
            return None
        slot = slots[0]
        if str(slot.get("grounding_kind") or "").upper() != "QUESTION" or str(slot.get("evidence_role") or "").upper() != "REQUIREMENT":
            diagnostic["reason"] = "NON_QUESTION_REQUIREMENT"
            return None
        spec = slot.get("proof_spec") or {}
        if spec.get("status") != "VALID":
            diagnostic["reason"] = "NO_VALID_STRUCTURED_CERTIFICATE"
            return None
        answer_type = str(ir.get("answer_type") or plan.get("query_spec", {}).get("answer_type") or "TEXT").upper()
        if answer_type in {"OPTION_SET", "RELATIVE_TIME"}:
            diagnostic["reason"] = "ANSWER_TYPE_REQUIRES_SYNTHESIS"
            return None
        field = str(spec.get("answer_field") or "")
        if answer_type in {"ENTITY", "VALUE", "DATE"} and not field:
            diagnostic["reason"] = "NO_CERTIFIED_ANSWER_FIELD"
            return None
        if answer_type == "DATE":
            if field not in VALID_TEMPORAL_AXES:
                diagnostic["reason"] = "INVALID_DATE_ANSWER_FIELD"
                return None
            relation = str(slot.get("temporal_relation") or slot.get("time_relation") or "LOCATE").upper()
            if relation not in {"LOCATE", "EXACT"}:
                diagnostic["reason"] = "DATE_SELECTOR_REQUIRES_ARBITRATION"
                return None
        question = str(getattr(self, "_active_answer_question", "") or "")
        matches = []
        for memory in candidates:
            certified, reason = self._certificate_result(slot, memory)
            if not certified:
                continue
            if not self._slot_structure_covered(slot, [memory["id"]], [memory], relations):
                continue
            if not self._memory_satisfies_frame(memory, frame):
                continue
            if self._has_competing_active_value(memory):
                continue
            answer = self._terminal_render_answer(question, answer_type, memory, answer_field=field)
            if answer_type == "TEXT" and not answer:
                answer = self._terminal_render_answer(question, answer_type, memory)
            if answer:
                matches.append((memory, answer, reason))
        diagnostic["eligible"] = True
        diagnostic["candidate_count"] = len(matches)
        if not matches:
            diagnostic["reason"] = "NO_CERTIFIED_ANSWER_BEARING_CANDIDATE"
            return None
        normalized_answers = {self._rc_text(answer) for _, answer, _ in matches}
        if len(normalized_answers) != 1:
            diagnostic["reason"] = "AMBIGUOUS_CERTIFIED_VALUES"
            return None
        if self._has_unresolved_conflict([memory for memory, _, _ in matches]):
            diagnostic["reason"] = "UNRESOLVED_CONFLICT"
            return None
        memory, answer, _ = matches[0]
        diagnostic["reason"] = "STRUCTURED_TERMINAL_CERTIFICATE"
        return {"answer": answer, "support_ids": [memory["id"]], "slot_id": slot["id"], "reason": "STRUCTURED_TERMINAL_CERTIFICATE", "answer_precomputed": True, "answer_type": answer_type}
