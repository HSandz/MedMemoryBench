"""Explicit, JSON-serializable records used by Event-State Hybrid Memory."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TurnEvidence:
    turn_id: Optional[Any]
    speaker: str
    role: Optional[str]
    text: str
    timestamp: Optional[str] = None
    image_caption: Optional[str] = None
    source_session_id: Optional[Any] = None
    source_session_index: Optional[int] = None
    source_event_id: Optional[Any] = None


@dataclass
class NormalizedTurn:
    """Canonical turn form shared by extraction, storage, and provenance."""

    source_turn_id: Optional[Any]
    source_session_id: Optional[Any]
    source_session_index: Optional[int]
    source_event_id: Optional[Any]
    role: Optional[str]
    speaker: str
    canonical_speaker_id: str
    text: str
    image_caption: Optional[str]
    timestamp: Optional[str]


@dataclass
class EvidenceRef:
    episode_id: str
    source_session_id: Optional[Any] = None
    source_turn_ids: List[Any] = field(default_factory=list)
    support_type: str = "origin"


@dataclass
class Episode:
    episode_id: str
    context_id: Optional[Any]
    source_session_id: Optional[Any]
    source_session_index: Optional[int]
    source_event_id: Optional[Any]
    recorded_at: Optional[str]
    participants: List[str]
    conversation_scope: Optional[str]
    raw_text: str
    summary: str
    turn_evidence: List[TurnEvidence] = field(default_factory=list)

    def retrieval_text(self) -> str:
        """Return an information-dense but bounded embedding representation."""
        salient = " ".join(
            f"{turn.speaker}: {turn.text} {turn.image_caption or ''}".strip()
            for turn in self.turn_evidence[:8]
        )
        return f"{self.summary} {salient}".strip()[:4000]


@dataclass
class Claim:
    claim_id: str
    subject: str
    subject_key: str
    predicate: str
    value: str
    qualifiers: Dict[str, Any] = field(default_factory=dict)
    polarity: str = "positive"
    modality: str = "asserted"
    persistence: str = "state"
    recorded_at: Optional[str] = None
    valid_from: Optional[str] = None
    valid_to: Optional[str] = None
    valid_time_text: Optional[str] = None
    status: str = "active"
    evidence: List[EvidenceRef] = field(default_factory=list)
    confidence: float = 1.0
    # Added at the end to preserve compatibility with early positional callers.
    subject_id: str = ""
    # A value-independent semantic variable used only by state compilation.
    state_slot: Optional[str] = None

    def semantic_text(self) -> str:
        qualifiers = ", ".join(f"{key}: {value}" for key, value in self.qualifiers.items())
        return f"Subject: {self.subject}. Subject ID: {self.subject_id or self.subject_key}. Predicate: {self.predicate}. Value: {self.value}. Qualifiers: {qualifiers}. Polarity: {self.polarity}. Modality: {self.modality}. Persistence: {self.persistence}."


@dataclass
class StateOperation:
    operation_id: str
    episode_id: str
    observation: Dict[str, Any]
    matched_claim_id: Optional[str]
    result_claim_id: Optional[str]
    operation: str
    confidence: float
    rationale: str
    recorded_at: Optional[str]


def to_dict(value: Any) -> Dict[str, Any]:
    return asdict(value)


def episode_from_dict(value: Dict[str, Any]) -> Episode:
    turns = [TurnEvidence(**item) for item in value.get("turn_evidence", [])]
    return Episode(**{**value, "turn_evidence": turns})


def claim_from_dict(value: Dict[str, Any]) -> Claim:
    evidence = [EvidenceRef(**item) for item in value.get("evidence", [])]
    value = dict(value)
    value.setdefault("subject_id", value.get("subject_key", ""))
    value.setdefault("state_slot", None)
    claim = Claim(**{**value, "evidence": evidence})
    if claim.persistence == "history":
        claim.state_slot = None
    return claim
