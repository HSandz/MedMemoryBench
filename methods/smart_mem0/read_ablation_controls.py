"""Explicit READ ablation controls for SmartMem0.

These switches exist only to isolate architecture authorities during benchmark runs.
Defaults preserve the current v5 behaviour. No switch changes WRITE semantics, model
selection, benchmark-specific routing, or the two-call READ ceiling.
"""

from copy import deepcopy

from .read_execution_contract import ReadExecutionContractMixin


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
        """Gate only proof-driven acquisition; status has its own independent switch."""
        if not bool(getattr(self, "enable_reasoning_completion", True)):
            return False
        if not bool(getattr(self, "enable_proof_expansion", True)):
            return False
        return super()._rcm_requires_proof(slot)

    def _retrieval_status(self, plan, slot_support, selected, relations):
        """Optionally make retrieval completion structural rather than proof-gated.

        This deliberately bypasses proof-aware `_slot_covered` and relation-support
        filtering. Structural relation semantics (COMPARE/CAUSES/TEMPORAL_ORDER) remain
        enforced through the base execution contract; only target proof loses authority to
        downgrade FOUND/EMPTY and trigger recovery.
        """
        enabled = bool(getattr(self, "enable_reasoning_completion", True)) and bool(
            getattr(self, "enable_proof_status_gate", True)
        )
        if enabled:
            return super()._retrieval_status(plan, slot_support, selected, relations)

        requirement_status = {}
        for slot in (plan or {}).get("required_slots") or []:
            slot_id = str(slot.get("id") or "")
            support_ids = list((slot_support or {}).get(slot_id) or [])
            requirement_status[slot_id] = (
                "FOUND"
                if support_ids
                and bool(
                    self._slot_structure_covered(
                        slot, support_ids, selected, relations
                    )
                )
                else "EMPTY"
            )

        relation_status = ReadExecutionContractMixin._relation_status_map(
            self, plan or {}, slot_support or {}, selected or [], relations or []
        )
        complete = bool(requirement_status) and all(
            status == "FOUND" for status in requirement_status.values()
        )
        complete = complete and all(
            str(status or "").upper() == "PROVEN"
            for status in relation_status.values()
        )
        self._last_requirement_graph_stop_guard = {
            "materiality_promotes_found": False,
            "strict_requirement_status": dict(requirement_status),
            "strict_relation_status": dict(relation_status),
            "retrieval_complete": bool(complete),
            "proof_status_gate_enabled": False,
            "completion_semantics": "typed_structural_coverage_without_target_proof",
        }
        return requirement_status, relation_status, bool(complete)

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
