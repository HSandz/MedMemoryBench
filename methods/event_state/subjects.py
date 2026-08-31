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
    prefix, _, proposed_name = canonical.partition(":")
    key = normalize_name(proposed_name if prefix in {"speaker", "third_party"} else raw)
    aliases = {"i", "me", "myself", "self", "user", "the_user", "patient", "the_patient", "primary_user", "primary_patient"}
    participant_map = {normalize_name(name): name for name in participants if name}
    visible_speakers = {key for key in participant_map if key not in {"user", "assistant", "system", "unknown"}}
    if key in participant_map and (prefix != "third_party" or key in visible_speakers):
        return "speaker:" + normalize_name(participant_map[key])
    if scope == "general_non_personal":
        # Unknown names and attributes remain non-personal. Capitalization or
        # an extractor-supplied prefix is not proof of a third-party identity.
        return "general_non_personal"
    if scope.startswith("third_party:"):
        target = scope.split(":", 1)[1]
        # The consultation target owns patient-centric claims unless visible
        # participant evidence identifies a different person above.
        return "third_party:" + target
    if canonical == "general_non_personal":
        return "primary_user"
    if canonical == "primary_user" or key in aliases or not raw:
        if speaker and normalize_name(speaker) in visible_speakers:
            return "speaker:" + normalize_name(speaker)
        return "primary_user"
    # Explicit family/third-party references must not contaminate user state.
    if key in {"father", "mother", "spouse", "partner", "friend", "brother", "sister", "child", "doctor"}:
        return "third_party:" + key
    if speaker and normalize_name(speaker) in visible_speakers:
        return "speaker:" + normalize_name(speaker)
    return "primary_user"


def is_visible_subject_identity(raw_subject: Any, participants: Iterable[str]) -> bool:
    """Return whether a raw subject names a visible participant."""
    raw = str(raw_subject or "").strip().casefold()
    prefix, _, name = raw.partition(":")
    key = normalize_name(name if prefix in {"speaker", "third_party"} else raw)
    return key in {normalize_name(participant) for participant in participants if participant}


def display_subject(subject_id: str, fallback: str = "User") -> str:
    if subject_id == "primary_user":
        return "User"
    if subject_id == "general_non_personal":
        return "General discussion"
    return subject_id.split(":", 1)[1].replace("_", " ").title() if ":" in subject_id else fallback
