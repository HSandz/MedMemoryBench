"""Dual dense retrieval, optional typed PPR, and redundancy-aware selection."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

from .embeddings import cosine
from .store import EventStateStore
from utils.llm_client import get_usage_tracker


def dense_rank(query: Sequence[float], vectors: Dict[str, Sequence[float]], top_k: int) -> List[Tuple[str, float]]:
    return sorted(((key, cosine(query, value)) for key, value in vectors.items()), key=lambda item: (-item[1], item[0]))[:max(0, top_k)]


class EventStateRetriever:
    def __init__(self, store: EventStateStore, embedder: Any, **config: Any) -> None:
        self.store, self.embedder, self.config = store, embedder, config

    def retrieve(self, question: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        query_vector = self.embedder.embed_query(question)
        claim_rank = dense_rank(query_vector, self.store.claim_embeddings, int(self.config.get("claim_top_k", 30))) if self.config.get("retrieve_claims", True) else []
        episode_rank = dense_rank(query_vector, self.store.episode_embeddings, int(self.config.get("episode_top_k", 20))) if self.config.get("retrieve_episodes", True) else []
        candidates = self._rrf(claim_rank, episode_rank)
        if self.config.get("ppr_enabled", False):
            candidates = self._ppr(candidates)
        candidate_count = int(self.config.get("candidate_count", 40))
        candidates = sorted(candidates, key=lambda item: (-item["score"], item["id"]))[:candidate_count]
        selected = self._select(candidates, int(self.config.get("evidence_count", 8)))
        return selected, {"claim_candidates": len(claim_rank), "episode_candidates": len(episode_rank), "candidate_count": len(candidates), "ppr_enabled": bool(self.config.get("ppr_enabled", False)), "selector_mode": self.config.get("selector_mode", "state_mmr"), "selected_ids": [item["id"] for item in selected]}

    def _rrf(self, claims: Sequence[Tuple[str, float]], episodes: Sequence[Tuple[str, float]]) -> List[Dict[str, Any]]:
        values: Dict[str, Dict[str, Any]] = {}
        rrf_k = float(self.config.get("rrf_k", 60.0))
        for record_type, rows, weight in (("state_claim", claims, float(self.config.get("claim_retrieval_weight", 1.0))), ("episode", episodes, float(self.config.get("episode_retrieval_weight", 1.0)))):
            for rank, (identifier, dense_score) in enumerate(rows, 1):
                item = values.setdefault(identifier, {"id": identifier, "type": record_type, "score": 0.0, "dense_score": dense_score, "ppr_score": 0.0})
                item["score"] += weight / (rrf_k + rank)
        return list(values.values())

    def _ppr(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        with get_usage_tracker().scope("event_state.ppr"):
            return self._ppr_impl(candidates)

    def _ppr_impl(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        identifiers = [item["id"] for item in candidates]
        # Include graph neighbors so PPR can genuinely expand a dense seed set.
        for edge in self.store.edges:
            for identifier in (edge["source_id"], edge["target_id"]):
                if identifier not in identifiers and (identifier in self.store.claims or identifier in self.store.episodes):
                    identifiers.append(identifier)
        if not identifiers:
            return candidates
        initial = {identifier: next((item["score"] for item in candidates if item["id"] == identifier), 0.0) for identifier in identifiers}
        total = sum(initial.values()) or 1.0
        initial = {key: value / total for key, value in initial.items()}
        scores = dict(initial)
        weights = {"SUPERSEDES": self.config.get("ppr_weight_supersedes", 1.2), "SUPERSEDED_BY": self.config.get("ppr_weight_supersedes", 1.2), "REFINES": self.config.get("ppr_weight_refines", 1.0), "REFINED_BY": self.config.get("ppr_weight_refines", 1.0), "CONFLICTS_WITH": self.config.get("ppr_weight_conflict", 0.8), "CLAIM_SUPPORTED_BY_EPISODE": self.config.get("ppr_weight_evidence", 0.7), "EPISODE_SUPPORTS_CLAIM": self.config.get("ppr_weight_evidence", 0.7), "SAME_EPISODE": self.config.get("ppr_weight_same_episode", 0.5), "TEMPORAL_NEIGHBOR": self.config.get("ppr_weight_temporal_neighbor", 0.3)}
        outgoing: Dict[str, List[Tuple[str, float]]] = {key: [] for key in identifiers}
        for edge in self.store.edges:
            if edge["source_id"] in outgoing and edge["target_id"] in outgoing:
                outgoing[edge["source_id"]].append((edge["target_id"], float(weights.get(edge["relation_type"], edge.get("weight", 1.0)))))
        alpha, tolerance = float(self.config.get("ppr_alpha", .85)), float(self.config.get("ppr_tolerance", 1e-6))
        for _ in range(int(self.config.get("ppr_max_iterations", 20))):
            updated = {key: (1 - alpha) * initial[key] for key in identifiers}
            for source, neighbors in outgoing.items():
                normalizer = sum(weight for _, weight in neighbors)
                if normalizer:
                    for target, weight in neighbors:
                        updated[target] += alpha * scores[source] * weight / normalizer
            if sum(abs(updated[key] - scores[key]) for key in identifiers) < tolerance:
                scores = updated
                break
            scores = updated
        known = {item["id"] for item in candidates}
        for item in candidates:
            item["ppr_score"] = scores[item["id"]]
            item["score"] += scores[item["id"]]
        for identifier in identifiers:
            if identifier in known:
                continue
            record_type = "state_claim" if identifier in self.store.claims else "episode"
            candidates.append({"id": identifier, "type": record_type, "score": scores[identifier], "dense_score": 0.0, "ppr_score": scores[identifier]})
        return candidates

    def _select(self, candidates: Sequence[Dict[str, Any]], count: int) -> List[Dict[str, Any]]:
        with get_usage_tracker().scope("event_state.selector"):
            return self._select_impl(candidates, count)

    def _select_impl(self, candidates: Sequence[Dict[str, Any]], count: int) -> List[Dict[str, Any]]:
        mode, selected = self.config.get("selector_mode", "state_mmr"), []
        remaining = list(candidates)
        while remaining and len(selected) < count:
            if mode == "topk":
                choice = max(remaining, key=lambda item: (item["score"], item["id"]))
            else:
                weight = float(self.config.get("mmr_lambda", .7))
                def mmr(item: Dict[str, Any]) -> float:
                    vector = self._vector(item["id"], item["type"])
                    redundancy = max((cosine(vector, self._vector(other["id"], other["type"])) for other in selected), default=0.0)
                    diversity = 0.0
                    if mode == "state_mmr" and selected and all(other["type"] != item["type"] for other in selected):
                        diversity = 0.02
                    return weight * item["score"] - (1 - weight) * redundancy + diversity
                choice = max(remaining, key=lambda item: (mmr(item), item["id"]))
            selected.append(choice)
            remaining.remove(choice)
        for rank, item in enumerate(selected, 1):
            item["selected_rank"] = rank
        return selected

    def _vector(self, identifier: str, record_type: str) -> Sequence[float]:
        return self.store.claim_embeddings.get(identifier, []) if record_type == "state_claim" else self.store.episode_embeddings.get(identifier, [])
