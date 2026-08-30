"""In-memory immutable evidence archive and versioned semantic state store."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from typing import Any, Dict, List, Optional

from .schemas import Claim, Episode, StateOperation, claim_from_dict, episode_from_dict


class EventStateStore:
    """Keeps raw episodes immutable while allowing state metadata to evolve."""

    SCHEMA_VERSION = 2

    def __init__(self, context_id: Optional[Any] = None) -> None:
        self.context_id = context_id
        self.episodes: Dict[str, Episode] = {}
        self.claims: Dict[str, Claim] = {}
        self.operations: List[StateOperation] = []
        self.edges: List[Dict[str, Any]] = []
        self.episode_embeddings: Dict[str, List[float]] = {}
        self.claim_embeddings: Dict[str, List[float]] = {}

    @staticmethod
    def stable_id(prefix: str, value: Any) -> str:
        encoded = json.dumps(value, ensure_ascii=True, sort_keys=True, default=str)
        return f"{prefix}{hashlib.sha256(encoded.encode('utf-8')).hexdigest()[:16]}"

    def add_episode(self, episode: Episode, embedding: List[float]) -> None:
        if episode.episode_id not in self.episodes:
            self.episodes[episode.episode_id] = episode
            self.episode_embeddings[episode.episode_id] = list(embedding)

    def add_claim(self, claim: Claim, embedding: List[float]) -> None:
        self.claims[claim.claim_id] = claim
        self.claim_embeddings[claim.claim_id] = list(embedding)
        for evidence in claim.evidence:
            self.add_edge(claim.claim_id, evidence.episode_id, "CLAIM_SUPPORTED_BY_EPISODE")
            self.add_edge(evidence.episode_id, claim.claim_id, "EPISODE_SUPPORTS_CLAIM")

    def add_edge(self, source_id: str, target_id: str, relation_type: str, weight: float = 1.0) -> None:
        edge = {"source_id": source_id, "target_id": target_id, "relation_type": relation_type, "weight": weight}
        if edge not in self.edges:
            self.edges.append(edge)

    def add_relation_pair(self, newer: str, older: str, relation: str) -> None:
        inverse = {"SUPERSEDES": "SUPERSEDED_BY", "REFINES": "REFINED_BY"}.get(relation)
        self.add_edge(newer, older, relation)
        if inverse:
            self.add_edge(older, newer, inverse)
        elif relation == "CONFLICTS_WITH":
            self.add_edge(older, newer, relation)

    def claim_counts(self) -> Dict[str, int]:
        return {
            "active_claim_count": sum(item.status == "active" for item in self.claims.values()),
            "historical_claim_count": sum(item.status in {"superseded", "historical"} for item in self.claims.values()),
            "superseded_claim_count": sum(item.status == "superseded" for item in self.claims.values()),
            "refined_claim_count": sum(item.status == "refined" for item in self.claims.values()),
            "contested_claim_count": sum(item.status == "contested" for item in self.claims.values()),
            "standalone_claim_count": sum(item.status == "standalone" for item in self.claims.values()),
            "total_claim_count": len(self.claims),
            "total_episode_count": len(self.episodes),
        }

    def export(self) -> Dict[str, Any]:
        return {"schema_version": self.SCHEMA_VERSION, "method": "event_state", "context_id": self.context_id, "episodes": [asdict(item) for item in self.episodes.values()], "claims": [asdict(item) for item in self.claims.values()], "state_operations": [asdict(item) for item in self.operations], "edges": self.edges, "episode_embeddings": self.episode_embeddings, "claim_embeddings": self.claim_embeddings}

    @classmethod
    def from_export(cls, state: Dict[str, Any]) -> "EventStateStore":
        if state.get("method") != "event_state":
            raise ValueError("Not an Event-State Hybrid Memory snapshot")
        if state.get("schema_version") != cls.SCHEMA_VERSION:
            raise ValueError(
                f"Event-State snapshot schema v{state.get('schema_version')} is incompatible with schema v2; rebuild the memory snapshot."
            )
        store = cls(state.get("context_id"))
        store.episodes = {item["episode_id"]: episode_from_dict(item) for item in state.get("episodes", [])}
        store.claims = {item["claim_id"]: claim_from_dict(item) for item in state.get("claims", [])}
        store.operations = [StateOperation(**item) for item in state.get("state_operations", [])]
        store.edges = list(state.get("edges", []))
        store.episode_embeddings = {key: list(value) for key, value in state.get("episode_embeddings", {}).items()}
        store.claim_embeddings = {key: list(value) for key, value in state.get("claim_embeddings", {}).items()}
        return store
