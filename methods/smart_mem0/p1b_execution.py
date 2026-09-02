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
        option_label: str = "",
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
        self.option_label = option_label
        self.status = status
        self.description = ''

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
            "option_label": self.option_label,
            "status": self.status,
        }

class EvidenceLattice:
    def __init__(self):
        self.gaps = {} # Dict[str, Dict]

    def add_gap(self, gap: EvidenceGap):
        self.gaps[gap.id] = {
            "gap": gap,
            "status": gap.status,
            "memory_ids": [],
            "evidence_ids": [],
            "relation_ids": []
        }

    def update_gap(self, gap_id: str, status: str, memory_ids: List[str] = None):
        if gap_id in self.gaps:
            self.gaps[gap_id]["status"] = status
            if memory_ids:
                self.gaps[gap_id]["memory_ids"].extend(memory_ids)
                # Keep unique
                self.gaps[gap_id]["memory_ids"] = list(set(self.gaps[gap_id]["memory_ids"]))

    def is_sufficient(self) -> bool:
        for g_id, data in self.gaps.items():
            gap = data["gap"]
            if gap.required and data["status"] != STATUS_FILLED:
                return False
        return True

    def get_missing_required_gaps(self) -> List[EvidenceGap]:
        return [
            data["gap"] for data in self.gaps.values()
            if data["gap"].required and data["status"] in (STATUS_MISSING, STATUS_AMBIGUOUS, STATUS_CONFLICT)
        ]

    def to_legacy_slots(self) -> List[Dict[str, Any]]:
        slots = []
        for g_id, data in self.gaps.items():
            gap = data["gap"]
            # Convert Gap to Legacy Slot
            role_to_type = {
                "FOCAL_TRIGGER": "DIRECT",
                "OUTCOME": "DIRECT",
                "OPTION_SUPPORT": "DIRECT",
                "PRIOR_TRAJECTORY": "DIRECT",
                "CAUSAL_BRIDGE": "CAUSE_PATH",
                "COMPARISON_SIDE": "DIRECT",
                "GENERIC_EVIDENCE": "DIRECT"
            }
            
            slot = {
                "id": gap.id,
                "type": role_to_type.get(gap.role, "DIRECT"),
                "evidence_role": gap.role,
                "description": getattr(gap, "description", f"Gap for {gap.role} concerning {' '.join(gap.target_entities)}"),
                "required": gap.required,
                "resolution_strategy": "RETRIEVE",
                "subject": gap.subject_id,
                "time_axis": gap.temporal_axis,
            }
            if gap.role == "PRIOR_TRAJECTORY":
                slot["history"] = True
            slots.append(slot)
        return slots


def _evaluate_seed_gate(agent, question: str, frame: Any, seeds: List[Dict[str, Any]]) -> Tuple[bool, Optional[List[Dict[str, Any]]], str]:
    if not seeds:
        return False, None, "NO_SEEDS"

    # Whitelist approach: only allow simple lookup without structured reasoning keywords
    import re
    q_lower = question.lower()
    reject_keywords = [
        "why", "because", "compare", "vs", "versus", "option", "choice", 
        "recommend", "should", "cause", "lead to", "result in", "between",
        "difference", "explain", "reason"
    ]
    for k in reject_keywords:
        # Match as whole word/phrase boundaries
        if re.search(r'' + re.escape(k) + r'', q_lower):
            return False, None, "STRUCTURED_REASONING_REQUIRED"
    
    if hasattr(frame, "options") and frame.options:
         return False, None, "MULTI_OPTION_REQUIRED"

    top_seed = seeds[0]
    
    # Check for conflicts. If seeds[1] has same rank/score but different state value.
    if len(seeds) > 1:
        s2 = seeds[1]
        score1 = top_seed.get("_score")
        score2 = s2.get("_score")
        # Strict conflict check: if scores are close/identical
        if score1 is not None and score2 is not None and abs(float(score1) - float(score2)) < 0.01:
            val1 = top_seed.get("value") or top_seed.get("state")
            val2 = s2.get("value") or s2.get("state")
            if val1 and val2 and val1 != val2:
                return False, None, "CONFLICTING_CANDIDATES"

    # Reuse existing deterministic structural validation
    supports = agent._validate_fast_support(top_seed["id"], seeds, frame)
    if supports:
        return True, supports, "DIRECT_SEED_SUFFICIENT"
        
    return False, None, "SEED_VALIDATION_FAILED"
