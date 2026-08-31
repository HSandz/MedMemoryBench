"""In-memory immutable evidence archive and versioned semantic state store."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace
from typing import Any, Dict, List, Optional

from .schemas import Claim, Episode, StateOperation, claim_from_dict, episode_from_dict


class EventStateStore:
    """Keeps raw episodes immutable while allowing state metadata to evolve."""

    SCHEMA_VERSION = 3
    SEMANTIC_VERSION = "2.2"

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
            self.attach_claim_evidence(claim.claim_id, evidence, evidence.support_type)

    def attach_claim_evidence(self, claim_id: str, evidence: Any, support_type: str = "origin") -> bool:
        """Attach one provenance reference and keep the heterogeneous graph in sync."""
        claim = self.claims[claim_id]
        normalized = replace(evidence, support_type=support_type)
        for existing in claim.evidence:
            if existing.episode_id == normalized.episode_id and existing.source_turn_ids == normalized.source_turn_ids:
                self.add_edge(claim_id, normalized.episode_id, "CLAIM_SUPPORTED_BY_EPISODE")
                self.add_edge(normalized.episode_id, claim_id, "EPISODE_SUPPORTS_CLAIM")
                return False
        claim.evidence.append(normalized)
        self.add_edge(claim_id, normalized.episode_id, "CLAIM_SUPPORTED_BY_EPISODE")
        self.add_edge(normalized.episode_id, claim_id, "EPISODE_SUPPORTS_CLAIM")
        return True

    def validate_state_invariants(self) -> List[str]:
        errors: List[str] = []
        node_ids = set(self.claims) | set(self.episodes)
        seen_edges = set()
        for edge in self.edges:
            key = tuple(edge.get(field) for field in ("source_id", "target_id", "relation_type", "weight"))
            if key in seen_edges:
                errors.append(f"duplicate edge: {key}")
            seen_edges.add(key)
            if edge.get("source_id") not in node_ids or edge.get("target_id") not in node_ids:
                errors.append(f"edge references missing node: {edge}")
        for claim_id, claim in self.claims.items():
            if claim_id not in self.claim_embeddings:
                errors.append(f"missing claim embedding: {claim_id}")
            for ref in claim.evidence:
                if ref.episode_id not in self.episodes:
                    errors.append(f"missing evidence episode: {claim_id}->{ref.episode_id}")
        for episode_id in self.episodes:
            if episode_id not in self.episode_embeddings:
                errors.append(f"missing episode embedding: {episode_id}")
        for edge in self.edges:
            if edge.get("relation_type") in {"SUPERSEDES", "REFINES"}:
                newer = self.claims.get(edge.get("source_id"))
                older = self.claims.get(edge.get("target_id"))
                if newer and older and newer.status == "active" and older.status == "active":
                    errors.append(f"version target remains active: {edge}")
        return errors

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
        return {"schema_version": self.SCHEMA_VERSION, "semantic_version": self.SEMANTIC_VERSION, "method": "event_state", "context_id": self.context_id, "episodes": [asdict(item) for item in self.episodes.values()], "claims": [asdict(item) for item in self.claims.values()], "state_operations": [asdict(item) for item in self.operations], "edges": self.edges, "episode_embeddings": self.episode_embeddings, "claim_embeddings": self.claim_embeddings}

    @classmethod
    def from_export(cls, state: Dict[str, Any]) -> "EventStateStore":
        if state.get("method") != "event_state":
            raise ValueError("Not an Event-State Hybrid Memory snapshot")
        if state.get("schema_version") != cls.SCHEMA_VERSION:
            raise ValueError(
                f"Event-State snapshot schema v{state.get('schema_version')} is incompatible with schema v3; rebuild the memory snapshot."
            )
        if state.get("semantic_version") != cls.SEMANTIC_VERSION:
            raise ValueError("Event-State snapshot semantic version is incompatible; rebuild the memory snapshot.")
        store = cls(state.get("context_id"))
        store.episodes = {item["episode_id"]: episode_from_dict(item) for item in state.get("episodes", [])}
        store.claims = {item["claim_id"]: claim_from_dict(item) for item in state.get("claims", [])}
        store.operations = [StateOperation(**item) for item in state.get("state_operations", [])]
        store.edges = list(state.get("edges", []))
        store.episode_embeddings = {key: list(value) for key, value in state.get("episode_embeddings", {}).items()}
        store.claim_embeddings = {key: list(value) for key, value in state.get("claim_embeddings", {}).items()}
        return store
