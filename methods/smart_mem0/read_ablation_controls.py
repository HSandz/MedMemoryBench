"""Explicit READ ablation controls for SmartMem0.

These switches exist only to isolate architecture authorities during benchmark runs.
Defaults preserve the current v5 behaviour. No switch changes WRITE semantics, model
selection, benchmark-specific routing, or the two-call READ ceiling.
"""

from copy import deepcopy


class ReadAblationControlsMixin:
    """Expose deterministic READ-layer ablations without adding model calls."""

    READ_ABLATION_CONTROLS_VERSION = "read-ablation-controls-v1"

    def _read_ablation_controls(self):
        return {
            "version": self.READ_ABLATION_CONTROLS_VERSION,
            "enable_reasoning_completion": bool(
                getattr(self, "enable_reasoning_completion", True)
            ),
            "enable_proof_status_gate": bool(
                getattr(self, "enable_proof_status_gate", True)
            ),
            "enable_proof_expansion": bool(
                getattr(self, "enable_proof_expansion", True)
            ),
            "enable_reasoning_source_neighbors": bool(
                getattr(self, "enable_reasoning_source_neighbors", True)
            ),
            "enable_reasoning_prompt_strengthening": bool(
                getattr(self, "enable_reasoning_prompt_strengthening", True)
            ),
            "enable_zero_result_recovery": bool(
                getattr(self, "enable_zero_result_recovery", True)
            ),
        }

    def _slot_covered(self, slot, support_ids, selected, relations):
        """Optionally make typed structural coverage independent of proof."""
        enabled = bool(getattr(self, "enable_reasoning_completion", True)) and bool(
            getattr(self, "enable_proof_status_gate", True)
        )
        if enabled:
            return super()._slot_covered(slot, support_ids, selected, relations)

        structural = getattr(self, "_slot_structure_covered", None)
        if callable(structural):
            return bool(structural(slot, support_ids, selected, relations))
        return super()._slot_covered(slot, support_ids, selected, relations)

    def _rcm_requires_proof(self, slot):
        """Use independent proof switches for acquisition and FOUND/EMPTY status."""
        if not bool(getattr(self, "enable_reasoning_completion", True)):
            return False
        if bool(getattr(self, "_read_ablation_status_phase", False)):
            if not bool(getattr(self, "enable_proof_status_gate", True)):
                return False
        elif not bool(getattr(self, "enable_proof_expansion", True)):
            return False
        return super()._rcm_requires_proof(slot)

    def _retrieval_status(self, plan, slot_support, selected, relations):
        """Mark proof calls made while computing completion separately from acquisition."""
        previous = bool(getattr(self, "_read_ablation_status_phase", False))
        self._read_ablation_status_phase = True
        try:
            return super()._retrieval_status(plan, slot_support, selected, relations)
        finally:
            self._read_ablation_status_phase = previous

    def _rcm_reasoning_ids(self, plan):
        if not bool(getattr(self, "enable_reasoning_completion", True)):
            return set()
        return super()._rcm_reasoning_ids(plan)

    def _rcm_source_neighbors(self, slot, rows, existing_ids, limit):
        if not bool(getattr(self, "enable_reasoning_completion", True)) or not bool(
            getattr(self, "enable_reasoning_source_neighbors", True)
        ):
            return []
        return super()._rcm_source_neighbors(slot, rows, existing_ids, limit)

    def _rcm_strengthen_prompt(self, prepared):
        if not bool(getattr(self, "enable_reasoning_completion", True)) or not bool(
            getattr(self, "enable_reasoning_prompt_strengthening", True)
        ):
            return False
        return super()._rcm_strengthen_prompt(prepared)

    def prepare_batch_query(self, question, system_message=None, **kwargs):
        prepared = super().prepare_batch_query(
            question, system_message=system_message, **kwargs
        )
        extra = prepared.setdefault("extra", {})
        extra["read_ablation_controls"] = deepcopy(self._read_ablation_controls())
        extra["active_read_version_stack"] = {
            "orchestrator": getattr(self, "QUERY_ORCHESTRATOR_VERSION", ""),
            "controller_schema": getattr(self, "CONTROLLER_SCHEMA_VERSION", ""),
            "requirement_graph": getattr(self, "REQUIREMENT_GRAPH_VERSION", ""),
            "requirement_graph_runtime": getattr(
                self, "REQUIREMENT_GRAPH_RUNTIME_VERSION", ""
            ),
            "proof": getattr(self, "PROJECTION_PROOF_VERSION", ""),
            "requirement_identity": getattr(
                self, "REQUIREMENT_IDENTITY_VERSION", ""
            ),
            "requirement_resolution": getattr(
                self, "REQUIREMENT_RESOLUTION_VERSION", ""
            ),
            "reasoning_completion": getattr(
                self, "REASONING_COMPLETION_VERSION", ""
            ),
        }
        return prepared
