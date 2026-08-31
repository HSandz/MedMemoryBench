"""Validation and normalization for untrusted extraction/classifier JSON."""

from __future__ import annotations

import math
import re
from typing import Any, Dict, Iterable, List, Optional, Set

ENUMS = {
    "polarity": {"positive", "negative", "uncertain"},
    "modality": {"asserted", "observed", "planned", "recommended", "hypothetical"},
    "persistence": {"state", "episode", "history"},
}


def enum(value: Any, field: str, default: str) -> str:
    normalized = str(value or default).strip().casefold()
    return normalized if normalized in ENUMS[field] else default


def safe_qualifiers(value: Any) -> Optional[Dict[str, Any]]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return {str(key): item for key, item in value.items()}
    if isinstance(value, list):
        result: Dict[str, Any] = {}
        for item in value:
            if isinstance(item, dict) and "key" in item and "value" in item:
                result[str(item["key"])] = item["value"]
            else:
                return None
        return result
    if isinstance(value, str) and value.strip():
        return {"text": value.strip()}
    return None


def validated_claim(raw: Any, source_turn_ids: Iterable[Any], allowed_turn_ids: Set[Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(raw, dict):
        return None
    predicate = raw.get("predicate")
    value = raw.get("value")
    if not isinstance(predicate, str) or not predicate.strip() or not isinstance(value, str) or not value.strip():
        return None
    qualifiers = safe_qualifiers(raw.get("qualifiers"))
    if qualifiers is None:
        return None
    supplied = raw.get("source_turn_ids")
    if not isinstance(supplied, list) or not supplied:
        return None
    if allowed_turn_ids and any(item not in allowed_turn_ids for item in supplied):
        return None
    confidence = raw.get("confidence", 1.0)
    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        confidence = 0.0
    return {
        **raw,
        "subject": str(raw.get("subject") or ""),
        "subject_id": str(raw.get("subject_id") or "").strip(),
        "predicate": predicate.strip(),
        "value": value.strip(),
        "qualifiers": qualifiers,
        "polarity": enum(raw.get("polarity"), "polarity", "positive"),
        "modality": enum(raw.get("modality"), "modality", "asserted"),
        "persistence": enum(raw.get("persistence"), "persistence", "state"),
        "source_turn_ids": supplied,
        "confidence": max(0.0, min(1.0, confidence if math.isfinite(confidence) else 0.0)),
        "valid_from": raw.get("valid_from") if isinstance(raw.get("valid_from"), str) else None,
        "valid_to": raw.get("valid_to") if isinstance(raw.get("valid_to"), str) else None,
        "valid_time_text": raw.get("valid_time_text") if isinstance(raw.get("valid_time_text"), str) else None,
    }


_META_MARKERS = (
    "request for concise",
    "single-sentence state",
    "asked for summary",
    "question only",
    "not an observation",
    "formatting request",
)


def is_meta_claim(raw: Dict[str, Any]) -> bool:
    """Reject a small set of unambiguous conversation-act pseudo-memories."""
    text = " ".join(str(raw.get(key) or "") for key in ("subject", "predicate", "value")).casefold()
    text = re.sub(r"\s+", " ", text).strip()
    return any(marker in text for marker in _META_MARKERS)
