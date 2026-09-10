"""Deterministic runtime guards for RequirementGraph READ.

This layer contains no question taxonomy and no domain policy. It converts semantic
RequirementGraph intent into stable runtime invariants: derive canonical memory addresses
when the LLM omits them, keep CURRENT bound to an actual state head, and treat CandidateSet
requirements as shared participant evidence rather than per-option truth lanes.
"""

from copy import deepcopy


class ReadRequirementGraphRuntimeMixin:
    REQUIREMENT_GRAPH_RUNTIME_VERSION = "requirement-graph-runtime-v1"

    def _requirement_slot(self, requirement, ir, compiled_mode):
        slot = super()._requirement_slot(requirement, ir, compiled_mode)

        # Candidate alternatives are propositions. RequirementGraph nodes remain shared
        # participant evidence even when visible alternatives exist. Candidate-local probes
        # may broaden recall, but they must never redefine the material requirement role.
        if ir.get("visible_options") or ir.get("candidate_propositions"):
            slot["evidence_role"] = "REQUIREMENT"

        # LLM anchors are semantic hints, not a mandatory perfect schema. Resolve the
        # requirement `need` against the same durable concept surfaces written by memory
        # construction and merge those deterministic addresses with supplied anchors.
        need = str(
            slot.get("need")
            or requirement.get("need")
            or requirement.get("evidence_family")
            or requirement.get("answer_obligation")
            or ""
        ).strip()
        subject_id = str(slot.get("subject_id") or slot.get("subject") or "")
        resolver = getattr(self, "_rc_resolve_target_keys", None)
        derived = []
        if need and callable(resolver):
            try:
                derived = list(resolver(need, subject_id) or [])
            except Exception:
                derived = []
        unique = getattr(self, "_rg_unique_text", None)
        values = [
            *(slot.get("material_anchors") or []),
            *derived,
            *(slot.get("resolved_keys") or []),
        ]
        slot["material_anchors"] = (
            unique(values, limit=8) if callable(unique) else list(dict.fromkeys(values))[:8]
        )
        slot["canonical_anchor_source"] = {
            "llm": list(requirement.get("material_anchors") or [])[:6],
            "runtime_resolved": derived[:4],
        }
        return slot

    def _rg_selector_compatible(self, slot, memory):
        if not super()._rg_selector_compatible(slot, memory):
            return False
        relation = str(self._rg_selector_relation(slot) or "").upper()
        if relation != "CURRENT":
            return True

        # CURRENT is stronger than merely "not superseded". If the store has a state
        # identity/head for this memory, only an active head is materially compatible.
        head_checker = getattr(self, "_is_state_head", None)
        if callable(head_checker):
            try:
                return bool(head_checker(memory))
            except Exception:
                return False
        return False

    def _execute_operation(self, operation, outputs, seeds, frame=None):
        # Normal query execution initializes these in _run_query_retrieval. Defensive
        # initialization keeps direct operation tests/restored callers deterministic too.
        if not isinstance(getattr(self, "_last_requirement_graph_discoveries", None), dict):
            self._last_requirement_graph_discoveries = {}
        if not isinstance(getattr(self, "_requirement_graph_retrieval_stats", None), dict):
            self._requirement_graph_retrieval_stats = {
                "version": getattr(self, "REQUIREMENT_GRAPH_VERSION", "requirement-graph-v1"),
                "canonical_address_additions": 0,
                "stored_relation_additions": 0,
                "candidate_world_peak": 0,
            }
        return super()._execute_operation(operation, outputs, seeds, frame)

    def prepare_batch_query(self, question, system_message=None, **kwargs):
        prepared = super().prepare_batch_query(
            question, system_message=system_message, **kwargs
        )
        extra = prepared.setdefault("extra", {})
        extra["requirement_graph_runtime_version"] = self.REQUIREMENT_GRAPH_RUNTIME_VERSION
        plan = extra.get("replan") or extra.get("plan") or {}
        slots = list(plan.get("required_slots") or [])
        extra["requirement_canonical_addresses"] = {
            str(slot.get("id") or ""): deepcopy(slot.get("canonical_anchor_source") or {})
            for slot in slots
            if slot.get("id")
        }
        return prepared
