from typing import Dict, Any, List, Optional
import re
from datetime import datetime

def effective_time(memory: Dict[str, Any]) -> str:
    event = memory.get("event_time")
    if event and event != "UNKNOWN":
        return event
    return memory.get("document_time") or ""

class StateSpine:
    def __init__(self, identity: str):
        self.identity = identity
        self.versions: List[Dict[str, Any]] = []

    def add_version(self, memory: Dict[str, Any]):
        self.versions.append(memory)
        self.versions.sort(key=lambda x: (effective_time(x), x.get("id")))

    def latest(self) -> Optional[Dict[str, Any]]:
        if not self.versions:
            return None
        return self.versions[-1]
        
    def earliest(self) -> Optional[Dict[str, Any]]:
        if not self.versions:
            return None
        return self.versions[0]
        
    def as_of(self, target_date_str: str) -> Optional[Dict[str, Any]]:
        if not self.versions:
            return None
        valid = None
        for v in self.versions:
            if effective_time(v) <= target_date_str:
                valid = v
        return valid

def canonical_predicate_family(state_key: str) -> str:
    """Extract the broader family from a predicate (e.g. blood_pressure from blood_pressure_sys)."""
    # Just a simple heuristic for now
    if not state_key:
        return ""
    parts = state_key.lower().split("_")
    if len(parts) > 1 and parts[-1] in {"sys", "dia", "level", "status"}:
        return "_".join(parts[:-1])
    return state_key.lower()

def canonical_predicate_key(state_key: str) -> str:
    return str(state_key or "").lower().strip()

def canonical_object_key(object_anchor: str) -> str:
    return str(object_anchor or "").lower().strip()
    
def build_state_identity(subject_id: str, state_key: str, object_anchor: str) -> str:
    pred_key = canonical_predicate_key(state_key)
    if not pred_key:
        return ""
    obj_key = canonical_object_key(object_anchor)
    subject = str(subject_id or "").lower().strip()
    return f"{subject}::{pred_key}::{obj_key}"


MEASUREMENT_ALIASES = {
    "hba1c": "lab.hba1c",
    "hba1c_level": "lab.hba1c",
    "patient_hba1c_level": "lab.hba1c",
    "c-peptide": "lab.c_peptide",
    "c_peptide": "lab.c_peptide",
    "c_peptide_level": "lab.c_peptide",
    "c-peptide concentration": "lab.c_peptide",
    "uacr": "lab.uacr",
    "urine_albumin_creatinine_ratio": "lab.uacr",
}

def measurement_identity(memory: dict) -> str:
    raw_keys = [
        memory.get("object_anchor"),
        memory.get("state_key")
    ]
    raw_keys.extend(memory.get("entities", []))
    
    for raw in raw_keys:
        if not raw:
            continue
        normalized = str(raw).lower().strip().replace(" ", "_")
        if normalized in MEASUREMENT_ALIASES:
            return MEASUREMENT_ALIASES[normalized]
        # Also check without underscores
        if normalized.replace("_", "-") in MEASUREMENT_ALIASES:
            return MEASUREMENT_ALIASES[normalized.replace("_", "-")]
    
    # Try searching the claim
    claim = str(memory.get("claim", "")).lower()
    if "hba1c" in claim:
        return "lab.hba1c"
    if "c-peptide" in claim or "c peptide" in claim:
        return "lab.c_peptide"
    if "uacr" in claim or "albumin" in claim and "creatinine" in claim:
        return "lab.uacr"
        
    return ""

def has_exact_measurement_identity(memory: dict) -> bool:
    return bool(measurement_identity(memory))

def is_state_projection_eligible(memory: dict) -> bool:
    if memory.get("kind") == "STATE":
        return True
    return (
        memory.get("semantic_role") == "MEASUREMENT"
        and memory.get("memory_tier", "HOT") == "HOT"
        and has_exact_measurement_identity(memory)
    )

def state_identity(memory: dict) -> str:
    if not is_state_projection_eligible(memory):
        return ""
        
    subject_id = memory.get("subject_id") or memory.get("subject") or "patient"
    
    # If it's an event projected to a measurement state
    meas_id = measurement_identity(memory)
    if meas_id:
        # Override to measurement identity
        return f"{str(subject_id).lower().strip()}::{meas_id}::"
        
    state_key = memory.get("state_key") or ""
    if not state_key:
        return ""
    object_anchor = memory.get("object_anchor") or ""
    return build_state_identity(subject_id, state_key, object_anchor)

def canonicalize_state(memory: dict) -> dict:
    return memory
