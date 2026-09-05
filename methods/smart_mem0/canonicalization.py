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
        
        def _numeric_memory_seq(m_id: str) -> int:
            if not m_id: return 0
            match = re.search(r'\d+', str(m_id))
            return int(match.group(0)) if match else 0
            
        self.versions.sort(key=lambda x: (
            effective_time(x),
            x.get("document_time", ""),
            x.get("session_idx", 0),
            _numeric_memory_seq(x.get("id", ""))
        ))

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
    
def build_state_identity(
    subject_id: str, scope: str, state_key: str, object_anchor: str
) -> str:
    pred_key = canonical_predicate_key(state_key)
    if not pred_key:
        return ""
    obj_key = canonical_object_key(object_anchor)
    scope_key = str(scope or "general").lower().strip()
    subject = str(subject_id or "").lower().strip()
    return f"{subject}::{scope_key}::{pred_key}::{obj_key}"


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

    # Generic fallback for measurements outside the small alias table. Prefer
    # the discriminative predicate, then the object, and remove representation-only
    # suffixes so "blood_glucose" and "blood_glucose_level" share a family.
    for raw in (memory.get("state_key"), memory.get("object_anchor")):
        normalized = re.sub(r"[^a-z0-9]+", "_", str(raw or "").lower()).strip("_")
        normalized = re.sub(
            r"_(?:level|value|result|reading|measurement|concentration)$", "", normalized
        )
        normalized = re.sub(r"^(?:patient|current|latest)_", "", normalized)
        if normalized and normalized not in {"measurement", "value", "result", "level"}:
            return f"measurement.{normalized}"
        
    return ""

def has_exact_measurement_identity(memory: dict) -> bool:
    return bool(measurement_identity(memory))

def state_identity(memory: dict) -> str:
    subject_id = memory.get("subject_id") or memory.get("subject") or "patient"
    
    # 1. Measurement identity (role-gated)
    if memory.get("semantic_role") == "MEASUREMENT":
        meas_id = measurement_identity(memory)
        if meas_id:
            scope = str(memory.get("scope") or "measurement").lower().strip()
            return f"{str(subject_id).lower().strip()}::{scope}::{meas_id}::"
            
    # 2. Ordinary STATE processing
    if memory.get("kind") != "STATE":
        return ""
        
    state_key = memory.get("state_key") or ""
    if not state_key:
        return ""
        
    object_anchor = memory.get("object_anchor") or ""
    scope = memory.get("scope") or "general"
    return build_state_identity(subject_id, scope, state_key, object_anchor)

def is_state_projection_eligible(memory: dict) -> bool:
    identity = state_identity(memory)
    if not identity:
        return False
        
    if memory.get("semantic_role") == "MEASUREMENT":
        return (
            memory.get("memory_tier", "COLD") == "HOT"
            and memory.get("assertion_mode", "DIRECT") == "DIRECT"
        )
        
    return (
        memory.get("kind") == "STATE"
        and memory.get("memory_tier", "COLD") == "HOT"
        and memory.get("assertion_mode", "DIRECT") == "DIRECT"
    )

def canonicalize_state(memory: dict) -> dict:
    return memory
