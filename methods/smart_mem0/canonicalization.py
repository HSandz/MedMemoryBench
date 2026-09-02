from typing import Dict, Any, List, Optional
import re
from datetime import datetime

class StateSpine:
    def __init__(self, identity: str):
        self.identity = identity
        self.versions: List[Dict[str, Any]] = []

    def add_version(self, memory: Dict[str, Any]):
        self.versions.append(memory)
        # Sort by event_time, fallback to document_time
        self.versions.sort(key=lambda x: (x.get("event_time") or x.get("document_time") or "", x.get("id")))

    def latest(self) -> Optional[Dict[str, Any]]:
        if not self.versions:
            return None
        return self.versions[-1]
        
    def earliest(self) -> Optional[Dict[str, Any]]:
        if not self.versions:
            return None
        return self.versions[0]
        
    def as_of(self, target_date_str: str) -> Optional[Dict[str, Any]]:
        """Return the latest version on or before the target_date_str."""
        if not self.versions:
            return None
        
        valid = None
        for v in self.versions:
            t = v.get("event_time") or v.get("document_time") or ""
            if t <= target_date_str:
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
