"""Generic CandidateSet contract for SmartMem0 READ.

Visible multiple-choice options are only one producer. Any caller may provide a structured
candidate_set mapping. CandidateSet stores propositions, the shared question-owned
predicate, and retrieval evidence views. Empty evidence is UNKNOWN, never false.
"""

from copy import deepcopy


class ReadCandidateSetMixin:
    CANDIDATE_SET_VERSION = "candidate-set-v2"

    @staticmethod
    def _candidate_shared_predicate(question, stemmer=None):
        if callable(stemmer):
            value = stemmer(question)
        else:
            value = question
        return " ".join(str(value or "").split()).strip()

    def _semantic_controller(self, question, seeds, frame, context_map=None):
        supplied = dict(
            getattr(self, "_active_candidate_set_input", {}) or {}
        )
        if supplied:
            merged = dict(context_map or {}) if isinstance(context_map, dict) else {}
            merged["candidate_set"] = supplied
            context_map = merged
        return super()._semantic_controller(
            question, seeds, frame, context_map=context_map
        )

    def _controller_plan(self, ir, question, frame):
        plan = super()._controller_plan(ir, question, frame)
        candidates = dict(
            plan.get("candidate_propositions")
            or ir.get("candidate_propositions")
            or {}
        )
        if not candidates:
            return plan
        predicate = self._candidate_shared_predicate(
            question, getattr(self, "_question_stem", None)
        )
        candidate_set = {
            "version": self.CANDIDATE_SET_VERSION,
            "shared_predicate": predicate,
            "candidates": candidates,
            "evidence_views": {
                candidate_id: []
                for candidate_id in candidates
            },
            "empty_evidence_semantics": "UNKNOWN_NOT_FALSE",
        }
        plan["candidate_set"] = deepcopy(candidate_set)
        plan.setdefault("semantic_ir", {})["candidate_set"] = deepcopy(
            candidate_set
        )
        return plan

    def prepare_batch_query(self, question, system_message=None, **kwargs):
        normalize = getattr(self, "_normalize_candidate_propositions", None)
        supplied = kwargs.pop("candidate_set", None)
        self._active_candidate_set_input = (
            normalize(supplied)
            if callable(normalize) and supplied is not None
            else {}
        )
        try:
            prepared = super().prepare_batch_query(
                question, system_message=system_message, **kwargs
            )
        finally:
            self._active_candidate_set_input = {}

        extra = prepared.setdefault("extra", {})
        existing = extra.get("candidate_set") or {}
        candidates = dict(
            existing.get("candidates")
            or extra.get("candidate_propositions")
            or {}
        )
        if not candidates:
            return prepared

        coverage = deepcopy(
            getattr(self, "_last_proposition_probe_coverage", {}) or {}
        )
        predicate = self._candidate_shared_predicate(
            question, getattr(self, "_question_stem", None)
        )
        extra["candidate_set"] = {
            "version": self.CANDIDATE_SET_VERSION,
            "shared_predicate": predicate,
            "candidates": candidates,
            "evidence_views": {
                candidate_id: list(coverage.get(candidate_id, []) or [])
                for candidate_id in candidates
            },
            "empty_evidence_semantics": "UNKNOWN_NOT_FALSE",
        }
        return prepared
