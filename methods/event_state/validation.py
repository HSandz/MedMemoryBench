"""Validation and normalization for untrusted extraction/classifier JSON."""

from __future__ import annotations

import json
import math
import re
from typing import Any, Dict, Iterable, List, Optional, Set

ENUMS = {
    "polarity": {"positive", "negative", "uncertain"},
    "modality": {"asserted", "observed", "planned", "recommended", "hypothetical"},
    "persistence": {"state", "episode", "history"},
}
NO_INFORMATION_VALUES = {"", "unknown", "unspecified", "not specified", "not provided", "n/a"}


def normalize_state_slot(value: Any) -> str:
    """Normalize a short, value-independent state dimension label."""
    return re.sub(r"[^a-z0-9]+", "_", str(value or "").casefold()).strip("_")


def is_no_information_value(value: Any) -> bool:
    """Reject only explicit absence-of-information placeholders."""
    return re.sub(r"\s+", " ", str(value or "").strip().casefold()) in NO_INFORMATION_VALUES


def canonical_turn_id(value: Any, fallback_index: Any) -> str:
    """Return the single string representation used for turn provenance."""
    text = str(value).strip() if value is not None else ""
    return text or str(fallback_index)


_MODEL_TURN_REFERENCE_PATTERNS = (
    ("bracket_turn_id", re.compile(r"^\[\s*turn_id\s*=\s*([^\[\]]+?)\s*\]$")),
    ("assignment_turn_id", re.compile(r"^turn_id\s*=\s*(.+)$")),
    ("turn_prefix", re.compile(r"^turn_(.+)$")),
)


def resolve_model_turn_reference_with_form(value: Any, allowed_turn_ids: Iterable[Any]) -> tuple[Optional[str], Optional[str]]:
    """Resolve a model reference and report the deterministic wrapper form."""
    allowed = {canonical_turn_id(item, "") for item in allowed_turn_ids}
    text = canonical_turn_id(value, "")
    if text in allowed:
        return text, None
    candidates = []
    for form, pattern in _MODEL_TURN_REFERENCE_PATTERNS:
        match = pattern.fullmatch(text)
        if not match:
            continue
        candidate = canonical_turn_id(match.group(1), "")
        if candidate in allowed:
            candidates.append((candidate, form))
    distinct_candidates = {candidate for candidate, _ in candidates}
    if len(distinct_candidates) == 1:
        return candidates[0]
    return None, None


def resolve_model_turn_reference(value: Any, allowed_turn_ids: Iterable[Any]) -> Optional[str]:
    """Resolve an untrusted model reference to an existing visible turn ID."""
    return resolve_model_turn_reference_with_form(value, allowed_turn_ids)[0]


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


def _normalized_semantic_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key).strip(): _normalized_semantic_value(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, list):
        return [_normalized_semantic_value(item) for item in value]
    return value.strip() if isinstance(value, str) else value


def claim_semantic_fingerprint(raw: Dict[str, Any]) -> str:
    """Return a stable identity for semantic claim fields, excluding provenance."""
    persistence = enum(raw.get("persistence"), "persistence", "state")
    payload = {
        "subject": str(raw.get("subject") or "").strip(),
        "subject_id": str(raw.get("subject_id") or "").strip(),
        "predicate": str(raw.get("predicate") or "").strip(),
        "value": str(raw.get("value") or "").strip(),
        "qualifiers": _normalized_semantic_value(safe_qualifiers(raw.get("qualifiers")) or {}),
        "polarity": enum(raw.get("polarity"), "polarity", "positive"),
        "modality": enum(raw.get("modality"), "modality", "asserted"),
        "persistence": persistence,
        "state_slot": normalize_state_slot(raw.get("state_slot")) if persistence == "state" else None,
        "valid_from": str(raw.get("valid_from")).strip() if isinstance(raw.get("valid_from"), str) else None,
        "valid_to": str(raw.get("valid_to")).strip() if isinstance(raw.get("valid_to"), str) else None,
        "valid_time_text": str(raw.get("valid_time_text")).strip() if isinstance(raw.get("valid_time_text"), str) else None,
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)


def claim_repair_identity(raw: Dict[str, Any]) -> tuple[str, tuple[str, ...]]:
    """Identify a validated claim using semantic fields and canonical evidence IDs."""
    return claim_semantic_fingerprint(raw), tuple(canonical_turn_id(item, "") for item in raw.get("source_turn_ids", []))


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
    # JSON producers may emit numeric IDs while the normalized episode uses
    # strings.  Validate and persist one canonical representation.
    supplied = [canonical_turn_id(item, "") for item in supplied]
    allowed = {canonical_turn_id(item, "") for item in allowed_turn_ids}
    if allowed and any(item not in allowed for item in supplied):
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
        "state_slot": normalize_state_slot(raw.get("state_slot"))
        if enum(raw.get("persistence"), "persistence", "state") == "state"
        else None,
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
