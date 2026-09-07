"""Unified ProofContext ownership for SmartMem0 READ.

Retrieval completeness and proof are intentionally different questions:
- retrieval asks whether every evidence obligation has at least one structurally viable view;
- proof asks whether a candidate has a conservative certificate;
- terminality asks whether one strict certificate uniquely determines the requested projection.

This class folds the historical final-context owner into ProofContext so there is exactly one
component that decides what reaches LLM #2.
"""

from copy import deepcopy

from .proof_context_contract import ProofContextContractMixin as _StrictProofContextMixin
from .read_proof_context_owner_contract import ReadProofContextOwnerContractMixin


class UnifiedProofContextMixin(
    ReadProofContextOwnerContractMixin,
    _StrictProofContextMixin,
):
    PROOF_CONTEXT_VERSION = "proof-context-v2"
    CONTEXT_OWNER_VERSION = PROOF_CONTEXT_VERSION

    def _retrieval_status(self, plan, slot_support, selected, relations):
        """Measure viable recall without asking a certificate to prove the answer."""
        structural = super(_StrictProofContextMixin, self)
        requirement_status = {}
        for slot in plan.get("required_slots") or []:
            slot_id = str(slot.get("id") or "")
            support_ids = list((slot_support or {}).get(slot_id) or [])
            requirement_status[slot_id] = (
                "FOUND"
                if support_ids
                and structural._slot_covered(
                    slot, support_ids, selected, relations
                )
                else "EMPTY"
            )
        relation_status = structural._relation_status_map(
            plan, slot_support, selected, relations
        )
        complete = bool(requirement_status) and all(
            status == "FOUND" for status in requirement_status.values()
        )
        # Structural relation proof can strengthen diagnostics but does not erase
        # otherwise viable evidence. LLM #2 owns semantic bridges unless a strict
        # terminal certificate closes the answer.
        self._last_retrieval_viability = {
            "requirements": dict(requirement_status),
            "relations": dict(relation_status),
            "complete": complete,
            "semantics": "candidate_viability_not_certificate",
        }
        return requirement_status, relation_status, complete

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
        self._last_retrieval_viability = {}
        run = super()._run_query_retrieval(
            question,
            initial_seeds,
            frame,
            fast_supports,
            gate,
            planning_seeds=planning_seeds,
            planning_context=planning_context,
        )
        run["proof_context_version"] = self.PROOF_CONTEXT_VERSION
        run["context_selection_owner"] = self.PROOF_CONTEXT_VERSION
        run["retrieval_viability"] = deepcopy(
            getattr(self, "_last_retrieval_viability", {}) or {}
        )
        return run

    def prepare_batch_query(self, question, system_message=None, **kwargs):
        prepared = super().prepare_batch_query(
            question, system_message=system_message, **kwargs
        )
        extra = prepared.setdefault("extra", {})
        extra["proof_context_version"] = self.PROOF_CONTEXT_VERSION
        extra["context_selection_owner"] = self.PROOF_CONTEXT_VERSION
        extra["proof_context_single_owner"] = True
        extra["retrieval_viability"] = deepcopy(
            getattr(self, "_last_retrieval_viability", {}) or {}
        )
        return prepared
