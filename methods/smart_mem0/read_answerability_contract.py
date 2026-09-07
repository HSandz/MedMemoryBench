"""Candidate-viability answerability for SmartMem0 READ.

Answerability never performs retrieval, semantic normalization, or proof. It chooses only:
RECOVER when an evidence obligation has zero viable candidates,
TERMINAL when upstream strict certification already produced a precomputed answer,
SYNTHESIZE otherwise.
"""

from copy import deepcopy


class ReadAnswerabilityContractMixin:
    ANSWERABILITY_CONTRACT_VERSION = "viable-evidence-v3"

    @staticmethod
    def _answerability_action(
        requirement_status,
        relation_status=None,
        *,
        viable_candidate_counts=None,
        terminal=False,
    ):
        del relation_status
        if terminal:
            return "TERMINAL"
        statuses = dict(requirement_status or {})
        if not statuses:
            return "RECOVER"
        if any(status != "FOUND" for status in statuses.values()):
            return "RECOVER"
        counts = dict(viable_candidate_counts or {})
        if counts and any(int(value or 0) <= 0 for value in counts.values()):
            return "RECOVER"
        return "SYNTHESIZE"

    @staticmethod
    def _viable_candidate_counts(run):
        plan = run.get("plan") or {}
        context_candidates = run.get("requirement_context_candidates") or {}
        slot_support = run.get("slot_support") or {}
        counts = {}
        if run.get("fast_supports") is not None:
            return {
                "fast_atomic_answer": len(run.get("fast_supports") or [])
            }
        for slot in plan.get("required_slots") or []:
            slot_id = str(slot.get("id") or "")
            role = str(slot.get("evidence_role") or "").upper()
            values = (
                context_candidates.get(slot_id)
                if role in {"REQUIREMENT", "COMPARAND"}
                and slot_id in context_candidates
                else slot_support.get(slot_id)
            )
            counts[slot_id] = len(list(values or []))
        return counts

    def _retrieval_status(self, plan, slot_support, selected, relations):
        statuses, relation_status, complete = super()._retrieval_status(
            plan, slot_support, selected, relations
        )
        self._last_requirement_answerability = dict(statuses)
        return statuses, relation_status, complete

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
        self._last_requirement_answerability = {}
        self._last_answerability_state = ""
        self._last_viable_candidate_counts = {}
        run = super()._run_query_retrieval(
            question,
            initial_seeds,
            frame,
            fast_supports,
            gate,
            planning_seeds=planning_seeds,
            planning_context=planning_context,
        )
        statuses = dict(run.get("requirement_status") or {})
        relation_status = dict(run.get("relation_status") or {})
        counts = self._viable_candidate_counts(run)
        action = self._answerability_action(
            statuses,
            relation_status,
            viable_candidate_counts=counts,
            terminal=bool(run.get("precomputed_answer")),
        )
        self._last_requirement_answerability = statuses
        self._last_viable_candidate_counts = counts
        self._last_answerability_state = action
        run["answerability_state"] = action
        run["requirement_answerability"] = statuses
        run["viable_candidate_counts"] = dict(counts)
        run["terminal_closure_diagnostic"] = deepcopy(
            getattr(self, "_last_terminal_closure_diagnostic", {}) or {}
        )
        return run

    def prepare_batch_query(self, question, system_message=None, **kwargs):
        prepared = super().prepare_batch_query(
            question, system_message=system_message, **kwargs
        )
        extra = prepared.setdefault("extra", {})
        extra["answerability_contract_version"] = self.ANSWERABILITY_CONTRACT_VERSION
        extra["answerability_state"] = getattr(
            self, "_last_answerability_state", ""
        )
        extra["requirement_answerability"] = dict(
            getattr(self, "_last_requirement_answerability", {}) or {}
        )
        extra["viable_candidate_counts"] = dict(
            getattr(self, "_last_viable_candidate_counts", {}) or {}
        )
        extra["terminal_closure_diagnostic"] = deepcopy(
            getattr(self, "_last_terminal_closure_diagnostic", {}) or {}
        )
        return prepared
