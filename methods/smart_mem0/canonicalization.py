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
    obj_key = canonical_object_key(object_anchor)
    subject = str(subject_id or "").lower().strip()
    return f"{subject}::{pred_key}::{obj_key}"

def state_identity(memory: dict) -> str:
    subject_id = memory.get("subject_id") or memory.get("subject") or "patient"
    state_key = memory.get("state_key") or ""
    object_anchor = memory.get("object_anchor") or ""
    return build_state_identity(subject_id, state_key, object_anchor)

def canonicalize_state(memory: dict) -> dict:
    return memory
