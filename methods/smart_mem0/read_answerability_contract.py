"""Round-level answerability state for SmartMem0 reads.

This layer does not normalize semantics, prove by similarity, render answers, rank context,
or call an LLM. It only summarizes the current execution state as:
RECOVER, TERMINAL, or SYNTHESIZE.
"""

from copy import deepcopy


class ReadAnswerabilityContractMixin:
    """Own the deterministic decision state between retrieval rounds."""

    ANSWERABILITY_CONTRACT_VERSION = "round-action-v2"

    @staticmethod
    def _answerability_action(
        requirement_status, relation_status, *, terminal=False
    ):
        if terminal:
            return "TERMINAL"
        statuses = dict(requirement_status or {})
        relations = dict(relation_status or {})
        if not statuses:
            return "RECOVER"
        if any(status != "FOUND" for status in statuses.values()):
            return "RECOVER"
        if any(status != "PROVEN" for status in relations.values()):
            return "RECOVER"
        return "SYNTHESIZE"

    def _retrieval_status(self, plan, slot_support, selected, relations):
        """Observe retrieval state without changing FOUND/EMPTY proof semantics."""
        statuses, relation_status, complete = super()._retrieval_status(
            plan, slot_support, selected, relations
        )
        self._last_requirement_answerability = dict(statuses)
        self._last_answerability_state = self._answerability_action(
            statuses, relation_status
        )
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
        action = self._answerability_action(
            statuses,
            relation_status,
            terminal=bool(run.get("precomputed_answer")),
        )
        self._last_requirement_answerability = statuses
        self._last_answerability_state = action
        run["answerability_state"] = action
        run["requirement_answerability"] = statuses
        run["terminal_closure_diagnostic"] = deepcopy(
            getattr(self, "_last_terminal_closure_diagnostic", {}) or {}
        )
        return run

    def prepare_batch_query(self, question, system_message=None, **kwargs):
        prepared = super().prepare_batch_query(
            question, system_message=system_message, **kwargs
        )
        extra = prepared.setdefault("extra", {})
        extra[
            "answerability_contract_version"
        ] = self.ANSWERABILITY_CONTRACT_VERSION
        extra["answerability_state"] = getattr(
            self, "_last_answerability_state", ""
        )
        extra["requirement_answerability"] = dict(
            getattr(self, "_last_requirement_answerability", {}) or {}
        )
        extra["terminal_closure_diagnostic"] = deepcopy(
            getattr(self, "_last_terminal_closure_diagnostic", {}) or {}
        )
        return prepared
