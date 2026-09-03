import json
import time
from typing import Any, Dict, List, Optional, Set, Tuple

STATUS_FILLED = "FILLED"
STATUS_MISSING = "MISSING"
STATUS_AMBIGUOUS = "AMBIGUOUS"
STATUS_CONFLICT = "CONFLICT"


class EvidenceGap:
    """Internal read requirement.

    target_surface is question-owned natural language. resolved_keys are derived
    only from durable write-time fields. Legacy target/option/comparison fields
    remain for runtime compatibility but are not used by the active contract.
    """

    def __init__(
        self,
        id: str,
        role: str,
        required: bool,
        subject_id: str = "",
        target_surface: str = "",
        resolved_keys: List[str] = None,
        required_fields: List[str] = None,
        temporal_axis: str = "",
        temporal_relation: str = "",
        temporal_anchor: str = "",
        temporal_end: str = "",
        side_label: str = "",
        status: str = STATUS_MISSING,
        target_entities: List[str] = None,
        target_property: str = "",
        option_label: str = "",
        option_proposition: str = "",
        comparison_side_label: str = "",
    ):
        self.id = id
        self.role = role
        self.required = required
        self.subject_id = subject_id
        self.target_surface = target_surface
        self.resolved_keys = resolved_keys or []
        self.required_fields = required_fields or []
        self.temporal_axis = temporal_axis
        self.temporal_relation = temporal_relation
        self.temporal_anchor = temporal_anchor
        self.temporal_end = temporal_end
        self.side_label = side_label or comparison_side_label
        self.status = status
        self.description = ""
        self.qrf_operator = "DIRECT"
        self.target_entities = target_entities or []
        self.target_property = target_property
        self.option_label = option_label
        self.option_proposition = option_proposition
        self.comparison_side_label = self.side_label

    def to_dict(self):
        return {
            "id": self.id,
            "role": self.role,
            "required": self.required,
            "subject_id": self.subject_id,
            "target_surface": self.target_surface,
            "resolved_keys": self.resolved_keys,
            "required_fields": self.required_fields,
            "temporal_axis": self.temporal_axis,
            "temporal_relation": self.temporal_relation,
            "temporal_anchor": self.temporal_anchor,
            "temporal_end": self.temporal_end,
            "side_label": self.side_label,
            "status": self.status,
        }


class EvidenceLattice:
    def __init__(self):
        self.gaps = {}

    def add_gap(self, gap: EvidenceGap):
        self.gaps[gap.id] = {
            "gap": gap,
            "status": gap.status,
            "memory_ids": [],
            "evidence_ids": [],
            "relation_ids": [],
        }

    def update_gap(self, gap_id: str, status: str, memory_ids: List[str] = None):
        if gap_id in self.gaps:
            self.gaps[gap_id]["status"] = status
            if memory_ids:
                self.gaps[gap_id]["memory_ids"].extend(memory_ids)
                self.gaps[gap_id]["memory_ids"] = list(set(self.gaps[gap_id]["memory_ids"]))

    def is_sufficient(self) -> bool:
        return all(
            not data["gap"].required or data["status"] == STATUS_FILLED
            for data in self.gaps.values()
        )

    def get_missing_required_gaps(self) -> List[EvidenceGap]:
        return [
            data["gap"]
            for data in self.gaps.values()
            if data["gap"].required
            and data["status"] in (STATUS_MISSING, STATUS_AMBIGUOUS, STATUS_CONFLICT)
        ]

    def to_legacy_slots(self) -> List[Dict[str, Any]]:
        slots = []
        for data in self.gaps.values():
            gap = data["gap"]
            role_to_type = {
                "FOCAL_TRIGGER": "DIRECT",
                "OUTCOME": "DIRECT",
                "PRIOR_TRAJECTORY": "DIRECT",
                "CAUSAL_BRIDGE": "CAUSE_PATH",
                "COMPARAND": "DIRECT",
                "OPTION_CONTEXT": "DIRECT",
                "GENERIC_EVIDENCE": "DIRECT",
                "OPTION_SUPPORT": "DIRECT",
                "COMPARISON_SIDE": "DIRECT",
            }
            slot_type = role_to_type.get(gap.role, "DIRECT")
            qrf_op = getattr(gap, "qrf_operator", "DIRECT")
            if qrf_op == "STATE":
                slot_type = "CURRENT_STATE"
            elif gap.temporal_axis or gap.temporal_relation:
                slot_type = "TEMPORAL"
            elif gap.role == "CAUSAL_BRIDGE":
                slot_type = "CAUSE_PATH"

            slot = {
                "id": gap.id,
                "type": slot_type,
                "evidence_role": gap.role,
                "description": getattr(gap, "description", gap.target_surface or f"Gap for {gap.role}"),
                "required": gap.required,
                "resolution_strategy": "RETRIEVE",
                "subject": gap.subject_id,
                "subject_id": gap.subject_id,
                "target_surface": gap.target_surface,
                "resolved_keys": list(gap.resolved_keys),
                "side_label": gap.side_label or None,
                "time_axis": gap.temporal_axis,
                "time_relation": gap.temporal_relation,
                "time_anchor": gap.temporal_anchor,
                "time_end": gap.temporal_end,
                "required_fields": gap.required_fields,
                "temporal_relation": gap.temporal_relation,
                "qrf_operator": qrf_op,
                "target_entities": gap.target_entities,
                "target_property": gap.target_property,
                "option_label": gap.option_label,
                "option_proposition": gap.option_proposition,
                "comparison_side_label": gap.side_label or None,
            }
            if gap.role == "PRIOR_TRAJECTORY":
                slot["history"] = True
            slots.append(slot)
        return slots


def _evaluate_seed_gate(agent, qrf: Dict[str, Any], seeds: List[Dict[str, Any]], frame: Any) -> Tuple[bool, Optional[List[Dict[str, Any]]], str]:
    """Legacy compatibility seed gate; the two-stage controller owns active routing."""
    if not seeds:
        return False, None, "NO_SEEDS"
    op = qrf.get("operator", "DIRECT")
    agent.enable_slot_support_validation = bool(
        op in {"DECISION", "MULTI_OPTION", "CAUSAL", "COMPARISON"}
        or qrf.get("requires_inference", False)
    )
    if op != "DIRECT":
        return False, None, f"{op}_REQUIRED"
    if qrf.get("requires_inference", False):
        return False, None, "INFERENCE_REQUIRED"
    top_seed = seeds[0]
    if len(seeds) > 1:
        second = seeds[1]
        def state_id(memory):
            return (
                memory.get("subject_id") or memory.get("subject"),
                memory.get("scope"), memory.get("state_key"), memory.get("object_anchor"),
            )
        def mem_val(memory):
            return agent._memory_value(memory) if hasattr(agent, "_memory_value") else memory.get("value") or memory.get("state")
        id1, id2 = state_id(top_seed), state_id(second)
        if id1 == id2 and id1[2] and mem_val(top_seed) != mem_val(second):
            return False, None, "CONFLICTING_CANDIDATES"
        if (
            not id1[2]
            and top_seed.get("semantic_role")
            and top_seed.get("object_anchor")
            and top_seed.get("semantic_role") == second.get("semantic_role")
            and top_seed.get("object_anchor") == second.get("object_anchor")
            and (top_seed.get("subject_id") or top_seed.get("subject")) == (second.get("subject_id") or second.get("subject"))
            and mem_val(top_seed) != mem_val(second)
        ):
            return False, None, "CONFLICTING_CANDIDATES"
    supports = agent._validate_fast_support("$seed0", seeds, frame)
    return (True, supports, "DIRECT_SEED_SUFFICIENT") if supports else (False, None, "SEED_VALIDATION_FAILED")
