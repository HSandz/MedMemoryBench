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

    def semantic_text(self) -> str:
        qualifiers = ", ".join(f"{key}: {value}" for key, value in self.qualifiers.items())
        return f"Subject: {self.subject}. Predicate: {self.predicate}. Value: {self.value}. Qualifiers: {qualifiers}. Polarity: {self.polarity}."


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
    return Claim(**{**value, "evidence": evidence})
