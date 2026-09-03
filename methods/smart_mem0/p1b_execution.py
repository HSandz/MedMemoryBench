import json
import time
from typing import Any, Dict, List, Optional, Set, Tuple

# Gap Status Enums
STATUS_FILLED = "FILLED"
STATUS_MISSING = "MISSING"
STATUS_AMBIGUOUS = "AMBIGUOUS"
STATUS_CONFLICT = "CONFLICT"


class EvidenceGap:
    def __init__(
        self,
        id: str,
        role: str,
        required: bool,
        subject_id: str = "",
        target_entities: List[str] = None,
        target_property: str = "",
        required_fields: List[str] = None,
        temporal_axis: str = "",
        temporal_relation: str = "",
        temporal_anchor: str = "",
        temporal_end: str = "",
        option_label: str = "",
        option_proposition: str = "",
        comparison_side_label: str = "",
        status: str = STATUS_MISSING,
    ):
        self.id = id
        self.role = role
        self.required = required
        self.subject_id = subject_id
        self.target_entities = target_entities or []
        self.target_property = target_property
        self.required_fields = required_fields or []
        self.temporal_axis = temporal_axis
        self.temporal_relation = temporal_relation
        self.temporal_anchor = temporal_anchor
        self.temporal_end = temporal_end
        self.option_label = option_label
        self.option_proposition = option_proposition
        self.comparison_side_label = comparison_side_label
        self.status = status
        self.description = ""
        self.qrf_operator = "DIRECT"

    def to_dict(self):
        return {
            "id": self.id,
            "role": self.role,
            "required": self.required,
            "subject_id": self.subject_id,
            "target_entities": self.target_entities,
            "target_property": self.target_property,
            "required_fields": self.required_fields,
            "temporal_axis": self.temporal_axis,
            "temporal_relation": self.temporal_relation,
            "temporal_anchor": self.temporal_anchor,
            "temporal_end": self.temporal_end,
            "option_label": self.option_label,
            "option_proposition": self.option_proposition,
            "comparison_side_label": self.comparison_side_label,
            "status": self.status,
        }


class EvidenceLattice:
    def __init__(self):
        self.gaps = {}  # Dict[str, Dict]

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
                self.gaps[gap_id]["memory_ids"] = list(
                    set(self.gaps[gap_id]["memory_ids"])
                )

    def is_sufficient(self) -> bool:
        for data in self.gaps.values():
            gap = data["gap"]
            if gap.required and data["status"] != STATUS_FILLED:
                return False
        return True

    def get_missing_required_gaps(self) -> List[EvidenceGap]:
        return [
            data["gap"]
            for data in self.gaps.values()
            if data["gap"].required
            and data["status"]
            in (STATUS_MISSING, STATUS_AMBIGUOUS, STATUS_CONFLICT)
        ]

    def to_legacy_slots(self) -> List[Dict[str, Any]]:
        slots = []
        for data in self.gaps.values():
            gap = data["gap"]
            role_to_type = {
                "FOCAL_TRIGGER": "DIRECT",
                "OUTCOME": "DIRECT",
                "OPTION_SUPPORT": "DIRECT",
                "PRIOR_TRAJECTORY": "DIRECT",
                "CAUSAL_BRIDGE": "CAUSE_PATH",
                "COMPARISON_SIDE": "DIRECT",
                "GENERIC_EVIDENCE": "DIRECT",
            }

            slot_type = role_to_type.get(gap.role, "DIRECT")
            qrf_op = getattr(gap, "qrf_operator", "DIRECT")

            if qrf_op == "STATE":
                slot_type = "CURRENT_STATE"
            elif gap.role == "OPTION_SUPPORT":
                # Option evidence remains proposition evidence. Temporal metadata,
                # if any, is a constraint, not a replacement slot type.
                slot_type = "DIRECT"
            elif gap.temporal_axis or gap.temporal_relation:
                slot_type = "TEMPORAL"
            elif gap.role == "CAUSAL_BRIDGE":
                slot_type = "CAUSE_PATH"

            slot = {
                "id": gap.id,
                "type": slot_type,
                "evidence_role": gap.role,
                "description": getattr(
                    gap,
                    "description",
                    f"Gap for {gap.role} concerning {' '.join(gap.target_entities)}",
                ),
                "required": gap.required,
                "resolution_strategy": "RETRIEVE",
                "subject": gap.subject_id,
                "time_axis": gap.temporal_axis,
                "time_relation": gap.temporal_relation,
                "time_anchor": gap.temporal_anchor,
                "time_end": gap.temporal_end,
                "option_label": gap.option_label,
                "option_proposition": gap.option_proposition,
                "target_entities": gap.target_entities,
                "target_property": gap.target_property,
                "required_fields": gap.required_fields,
                "temporal_relation": gap.temporal_relation,
                "comparison_side_label": gap.comparison_side_label or None,
                "qrf_operator": qrf_op,
            }
            if gap.role == "PRIOR_TRAJECTORY":
                slot["history"] = True
            slots.append(slot)
        return slots


def _evaluate_seed_gate(
    agent,
    qrf: Dict[str, Any],
    seeds: List[Dict[str, Any]],
    frame: Any,
) -> Tuple[bool, Optional[List[Dict[str, Any]]], str]:
    if not seeds:
        return False, None, "NO_SEEDS"

    op = qrf.get("operator", "DIRECT")

    # Query-local validation policy. Complex evidence bundles get the existing
    # semantic slot-support check; direct/state/temporal paths remain purely
    # deterministic to preserve the fast path and latency budget.
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
        s2 = seeds[1]

        def _state_id(memory):
            return (
                memory.get("subject_id") or memory.get("subject"),
                memory.get("scope"),
                memory.get("state_key"),
                memory.get("object_anchor"),
            )

        def _mem_val(memory):
            if hasattr(agent, "_memory_value"):
                return agent._memory_value(memory)
            return memory.get("value") or memory.get("state")

        id1 = _state_id(top_seed)
        id2 = _state_id(s2)

        # Stateful conflict requires the same owned state identity, including
        # object anchor when one exists. This avoids conflating two properties
        # merely because they share a generic state key.
        if id1 == id2 and id1[2]:
            if _mem_val(top_seed) != _mem_val(s2):
                return False, None, "CONFLICTING_CANDIDATES"

        role1 = top_seed.get("semantic_role")
        obj1 = top_seed.get("object_anchor")
        if (
            not id1[2]
            and role1
            and obj1
            and role1 == s2.get("semantic_role")
            and obj1 == s2.get("object_anchor")
            and (top_seed.get("subject_id") or top_seed.get("subject"))
            == (s2.get("subject_id") or s2.get("subject"))
        ):
            if _mem_val(top_seed) != _mem_val(s2):
                return False, None, "CONFLICTING_CANDIDATES"

    supports = agent._validate_fast_support("$seed0", seeds, frame)
    if supports:
        return True, supports, "DIRECT_SEED_SUFFICIENT"

    return False, None, "SEED_VALIDATION_FAILED"
