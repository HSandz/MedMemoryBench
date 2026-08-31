"""Canonical subject and visible conversation-scope resolution."""

from __future__ import annotations

import re
from typing import Any, Iterable, Optional


def normalize_scope(text: str = "", explicit: Optional[str] = None) -> str:
    value = str(explicit or "").strip()
    if value:
        lowered = value.casefold()
        if lowered.startswith("third_party:"):
            return "third_party:" + normalize_name(value.split(":", 1)[1])
        if lowered in {"general", "general_non_personal", "non_personal"}:
            return "general_non_personal"
        if lowered in {"primary", "primary_user", "user"}:
            return "primary_user"
    match = re.search(r"\[\s*Health consultation record about\s+([^\]]+)\]", text or "", re.I)
    if match:
        target = re.sub(r"\([^)]*\)", "", match.group(1)).strip()
        return "third_party:" + normalize_name(target)
    if re.search(r"\[\s*Health consultation record\s*\]", text or "", re.I):
        return "general_non_personal"
    return "primary_user"


def normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", (value or "").casefold()).strip("_") or "unknown"


def is_canonical_subject_id(value: Any) -> bool:
    lowered = str(value or "").casefold()
    return lowered in {"primary_user", "general_non_personal"} or lowered.startswith(("speaker:", "third_party:"))


def resolve_subject_id(raw_subject: Any, scope: str, participants: Iterable[str] = (), speaker: Optional[str] = None) -> str:
    raw = str(raw_subject or "").strip()
    canonical = raw.casefold()
    if canonical in {"primary_user", "general_non_personal"}:
        return canonical
    for prefix in ("speaker:", "third_party:"):
        if canonical.startswith(prefix):
            return prefix + normalize_name(raw.split(":", 1)[1])
    key = normalize_name(raw)
    aliases = {"i", "me", "myself", "self", "user", "the_user", "patient", "the_patient", "primary_user", "primary_patient"}
    participant_map = {normalize_name(name): name for name in participants if name}
    if scope == "general_non_personal":
        if key in aliases:
            return "general_non_personal"
        if key in participant_map:
            return "speaker:" + normalize_name(participant_map[key])
        if raw[:1].isupper():
            return "third_party:" + key
        # Generic consultations must never create synthetic people from an
        # attribute or predicate proposed by the extractor.
        return "general_non_personal"
    if scope.startswith("third_party:"):
        target = scope.split(":", 1)[1]
        if key in aliases or not raw:
            return "third_party:" + target
        return "third_party:" + key
    if key in aliases or not raw:
        if speaker and normalize_name(speaker) not in {"user", "assistant", "system", "unknown"}:
            return "speaker:" + normalize_name(speaker)
        return "primary_user"
    if key in participant_map:
        return "speaker:" + normalize_name(participant_map[key])
    # Explicit family/third-party references must not contaminate user state.
    if key in {"father", "mother", "spouse", "partner", "friend", "brother", "sister", "child", "doctor"}:
        return "third_party:" + key
    if speaker and key == normalize_name(speaker):
        return "speaker:" + key
    # An unrecognized subject is not evidence of a person. In primary scope,
    # attribute-like proposals are conservatively attributed to the user when
    # the source turn is the primary user's turn; otherwise they remain user
    # scoped rather than becoming a fabricated speaker identity.
    if speaker and normalize_name(speaker) in {"user", "assistant", "system", "unknown"}:
        return "primary_user"
    if speaker and normalize_name(speaker) in participant_map:
        return "speaker:" + normalize_name(speaker)
    return "primary_user"


def display_subject(subject_id: str, fallback: str = "User") -> str:
    if subject_id == "primary_user":
        return "User"
    if subject_id == "general_non_personal":
        return "General discussion"
    return subject_id.split(":", 1)[1].replace("_", " ").title() if ":" in subject_id else fallback
