"""Dual dense retrieval, bounded optional PPR, and deterministic MMR selection."""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Any, Dict, List, Sequence, Tuple

from .embeddings import cosine
from .store import EventStateStore
from utils.llm_client import get_usage_tracker


def dense_rank(query: Sequence[float], vectors: Dict[str, Sequence[float]], top_k: int) -> List[Tuple[str, float]]:
    rows = [(key, cosine(query, value)) for key, value in vectors.items()]
    return sorted(rows, key=lambda item: (-item[1], item[0]))[:max(0, int(top_k))]


def normalize_scores(values: Sequence[float]) -> List[float]:
    if not values:
        return []
    low, high = min(values), max(values)
    if high <= low:
        return [1.0] * len(values)
    return [(value - low) / (high - low) for value in values]


class EventStateRetriever:
    def __init__(self, store: EventStateStore, embedder: Any, **config: Any) -> None:
        self.store, self.embedder, self.config = store, embedder, config

    def retrieve(self, question: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        query_vector = self.embedder.embed_query(question)
        claim_rank = dense_rank(query_vector, self.store.claim_embeddings, self.config.get("claim_top_k", 30)) if self.config.get("retrieve_claims", True) else []
        episode_rank = dense_rank(query_vector, self.store.episode_embeddings, self.config.get("episode_top_k", 20)) if self.config.get("retrieve_episodes", True) else []
        candidates = self._rrf(claim_rank, episode_rank)
        if self.config.get("ppr_enabled", False):
            candidates = self._ppr(candidates)
        values = normalize_scores([item.get("score", 0.0) for item in candidates])
        for item, final_score in zip(candidates, values):
            item["final_score"] = final_score
        candidates.sort(key=lambda item: (-item["final_score"], item["id"]))
        candidate_count = int(self.config.get("candidate_count", 40))
        candidates = candidates[:candidate_count]
        selected = self._select(candidates, int(self.config.get("evidence_count", 8)))
        return selected, {
            "claim_candidates": len(claim_rank),
            "episode_candidates": len(episode_rank),
            "candidate_count": len(candidates),
            "ppr_enabled": bool(self.config.get("ppr_enabled", False)),
            "selector_mode": self.config.get("selector_mode", "state_mmr"),
            "selected_ids": [item["id"] for item in selected],
        }

    def _rrf(self, claims: Sequence[Tuple[str, float]], episodes: Sequence[Tuple[str, float]]) -> List[Dict[str, Any]]:
        values: Dict[str, Dict[str, Any]] = {}
        rrf_k = float(self.config.get("rrf_k", 60.0))
        channels = (("state_claim", claims, float(self.config.get("claim_retrieval_weight", 1.0))), ("episode", episodes, float(self.config.get("episode_retrieval_weight", 1.0))))
        for record_type, rows, weight in channels:
            for rank, (identifier, dense_score) in enumerate(rows, 1):
                item = values.setdefault(identifier, {"id": identifier, "type": record_type, "score": 0.0, "dense_score": dense_score, "fusion_score": 0.0, "ppr_score": 0.0})
                item["score"] += weight / (rrf_k + rank)
                item["fusion_score"] = item["score"]
                item["dense_score"] = max(item["dense_score"], dense_score)
        return list(values.values())

    def _ppr(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        with get_usage_tracker().scope("event_state.ppr"):
            return self._ppr_impl(candidates)

    def _ppr_impl(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seeds = [item["id"] for item in candidates]
        if not seeds:
            return candidates
        hops = max(0, int(self.config.get("ppr_expand_hops", 2)))
        adjacency: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
        relation_weights = {"SUPERSEDES": self.config.get("ppr_weight_supersedes", 1.2), "SUPERSEDED_BY": self.config.get("ppr_weight_supersedes", 1.2), "REFINES": self.config.get("ppr_weight_refines", 1.0), "REFINED_BY": self.config.get("ppr_weight_refines", 1.0), "CONFLICTS_WITH": self.config.get("ppr_weight_conflict", 0.8), "CLAIM_SUPPORTED_BY_EPISODE": self.config.get("ppr_weight_evidence", 0.7), "EPISODE_SUPPORTS_CLAIM": self.config.get("ppr_weight_evidence", 0.7)}
        for edge in self.store.edges:
            weight = float(relation_weights.get(edge["relation_type"], 0.0))
            if weight > 0:
                adjacency[edge["source_id"]].append((edge["target_id"], weight))
        identifiers = set(seeds)
        frontier = set(seeds)
        for _ in range(hops):
            frontier = {target for source in frontier for target, _ in adjacency.get(source, []) if target in self.store.claims or target in self.store.episodes} - identifiers
            identifiers.update(frontier)
        identifiers = sorted(identifiers)
        initial = {identifier: next((item["score"] for item in candidates if item["id"] == identifier), 0.0) for identifier in identifiers}
        total = sum(initial.values()) or 1.0
        personalization = {key: value / total for key, value in initial.items()}
        scores = dict(personalization)
        alpha = float(self.config.get("ppr_alpha", .85))
        tolerance = float(self.config.get("ppr_tolerance", 1e-6))
        for _ in range(max(1, int(self.config.get("ppr_max_iterations", 20)))):
            updated = {key: (1 - alpha) * personalization[key] for key in identifiers}
            dangling = 0.0
            for source in identifiers:
                neighbors = [(target, weight) for target, weight in adjacency.get(source, []) if target in scores]
                normalizer = sum(weight for _, weight in neighbors)
                if not normalizer:
                    dangling += scores[source]
                    continue
                for target, weight in neighbors:
                    updated[target] += alpha * scores[source] * weight / normalizer
            for target in identifiers:
                updated[target] += alpha * dangling * personalization[target]
            delta = sum(abs(updated[key] - scores[key]) for key in identifiers)
            scores = updated
            if delta < tolerance:
                break
        known = {item["id"] for item in candidates}
        base_values = normalize_scores([item["score"] for item in candidates])
        gamma = float(self.config.get("ppr_mix_weight", .35))
        for item, base in zip(candidates, base_values):
            item["ppr_score"] = scores[item["id"]]
            item["base_score"] = base
            item["score"] = (1 - gamma) * base + gamma * scores[item["id"]]
            item["final_score"] = item["score"]
        for identifier in identifiers:
            if identifier not in known:
                record_type = "state_claim" if identifier in self.store.claims else "episode"
                candidates.append({"id": identifier, "type": record_type, "score": gamma * scores[identifier], "fusion_score": 0.0, "dense_score": 0.0, "ppr_score": scores[identifier], "base_score": 0.0, "final_score": gamma * scores[identifier]})
        return candidates

    def _select(self, candidates: Sequence[Dict[str, Any]], count: int) -> List[Dict[str, Any]]:
        with get_usage_tracker().scope("event_state.selector"):
            return self._select_impl(candidates, count)

    def _select_impl(self, candidates: Sequence[Dict[str, Any]], count: int) -> List[Dict[str, Any]]:
        mode, selected = self.config.get("selector_mode", "state_mmr"), []
        remaining = list(candidates)
        relevance = normalize_scores([float(item.get("final_score", item.get("score", 0.0))) for item in remaining])
        relevance_by_id = {item["id"]: value for item, value in zip(remaining, relevance)}
        while remaining and len(selected) < count:
            if mode == "topk":
                choice, choice_score = max(((item, item.get("final_score", item.get("score", 0.0))) for item in remaining), key=lambda pair: (pair[1], pair[0]["id"] ))
            else:
                weight = float(self.config.get("mmr_lambda", .7))
                def score(item: Dict[str, Any]) -> float:
                    vector = self._vector(item["id"], item["type"])
                    redundancy = max((cosine(vector, self._vector(other["id"], other["type"])) for other in selected), default=0.0)
                    value = weight * relevance_by_id[item["id"]] - (1 - weight) * redundancy
                    if mode == "state_mmr" and selected:
                        if any(edge["source_id"] == item["id"] and edge["target_id"] == other["id"] or edge["target_id"] == item["id"] and edge["source_id"] == other["id"] for edge in self.store.edges for other in selected):
                            value += float(self.config.get("state_relation_bonus", .05))
                        if item["type"] != selected[-1]["type"]:
                            value += float(self.config.get("representation_balance_bonus", .02))
                        if self._source_ids(item) - set().union(*(self._source_ids(other) for other in selected)):
                            value += float(self.config.get("source_diversity_bonus", .02))
                    return value
                choice = max(remaining, key=lambda item: (score(item), item["id"]))
                choice_score = score(choice)
            choice["selection_score"] = choice_score
            selected.append(choice)
            remaining.remove(choice)
        for rank, item in enumerate(selected, 1):
            item["selected_rank"] = rank
        return selected

    def _vector(self, identifier: str, record_type: str) -> Sequence[float]:
        return self.store.claim_embeddings.get(identifier, []) if record_type == "state_claim" else self.store.episode_embeddings.get(identifier, [])

    def _source_ids(self, item: Dict[str, Any]) -> set[Any]:
        if item["type"] == "episode":
            return {self.store.episodes[item["id"]].source_session_id}
        return {ref.source_session_id for ref in self.store.claims[item["id"]].evidence}
