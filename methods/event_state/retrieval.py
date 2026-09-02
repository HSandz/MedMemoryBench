"""Dual dense retrieval, bounded optional PPR, and deterministic MMR selection."""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from copy import deepcopy
from typing import Any, Dict, List, Sequence, Tuple

from .embeddings import cosine
from .store import EventStateStore
from .temporal import (
    TemporalQueryConstraint,
    claim_temporal_match,
    claim_visible_as_of,
    episode_temporal_match,
    parse_temporal_query as parse_temporal_query_fn,
    parse_stored_date,
)
from utils.llm_client import get_usage_tracker

# Keep the historical module-level hook patchable for compatibility tests.
parse_temporal_query = parse_temporal_query_fn


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

    def rank_candidates(
        self,
        question: str,
        query_vector: Sequence[float] | None = None,
        *,
        temporal_constraint: TemporalQueryConstraint | None = None,
        parse_temporal_query: bool = True,
        retrieve_claims_override: bool | None = None,
        retrieve_episodes_override: bool | None = None,
        state_view: str = "current",
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        query_vector = list(query_vector) if query_vector is not None else self.embedder.embed_query(question)
        temporal = temporal_constraint
        if temporal is None and parse_temporal_query and self.config.get("temporal_retrieval_enabled", True):
            temporal = globals()["parse_temporal_query"](question)
        retrieve_claims = self.config.get("retrieve_claims", True) if retrieve_claims_override is None else retrieve_claims_override
        retrieve_episodes = self.config.get("retrieve_episodes", True) if retrieve_episodes_override is None else retrieve_episodes_override
        claim_vectors, hidden_prior_state_count = self._visible_claim_vectors(temporal, state_view)
        claim_rank = dense_rank(query_vector, claim_vectors, self.config.get("claim_top_k", 30)) if retrieve_claims else []
        episode_rank = dense_rank(query_vector, self.store.episode_embeddings, self.config.get("episode_top_k", 20)) if retrieve_episodes else []
        temporal_claim_rank, temporal_episode_rank = [], []
        if temporal is not None:
            temporal_claim_rank = self._temporal_claim_rank(query_vector, temporal, state_view) if retrieve_claims else []
            temporal_episode_rank = self._temporal_episode_rank(query_vector, temporal) if retrieve_episodes else []
        candidates = self._rrf(claim_rank, episode_rank, temporal_claim_rank, temporal_episode_rank)
        if self.config.get("ppr_enabled", False):
            candidates = self._ppr(candidates)
        candidates = [item for item in candidates if item["type"] != "state_claim" or self._claim_is_directly_visible(item["id"], temporal, state_view)]
        values = normalize_scores([item.get("score", 0.0) for item in candidates])
        for item, final_score in zip(candidates, values):
            item["final_score"] = final_score
        candidates.sort(key=lambda item: (-item["final_score"], item["id"]))
        candidate_count = int(self.config.get("candidate_count", 40))
        candidates = candidates[:candidate_count]
        claim_candidate_statuses = Counter(self.store.claims[identifier].status for identifier, _ in claim_rank)
        temporal_historical = sum(
            1 for identifier, _score, _match in temporal_claim_rank
            if self.store.claims[identifier].status in {"superseded", "refined"}
        )
        future_filtered = 0
        if temporal is not None and temporal.kind == "as_of":
            target = temporal.target_date
            future_filtered = sum(
                1
                for claim in self.store.claims.values()
                if claim.persistence == "state"
                and parse_stored_date(claim.recorded_at) is not None
                and parse_stored_date(claim.recorded_at) > target
            )
        return candidates, {
            "claim_candidates": len(claim_rank),
            "episode_candidates": len(episode_rank),
            "candidate_count": len(candidates),
            "ppr_enabled": bool(self.config.get("ppr_enabled", False)),
            "selector_mode": self.config.get("selector_mode", "state_mmr"),
            "selected_ids": [],
            "claim_candidate_status_counts": dict(sorted(claim_candidate_statuses.items())),
            "selected_claim_status_counts": {},
            "selected_claim_persistence_counts": {},
            "hidden_prior_state_candidate_count": hidden_prior_state_count,
            "temporal_constraint_detected": temporal is not None,
            "temporal_constraint_kind": temporal.kind if temporal else None,
            "temporal_target_date": temporal.target_date.isoformat() if temporal and temporal.target_date else None,
            "temporal_start_date": temporal.start_date.isoformat() if temporal and temporal.start_date else None,
            "temporal_end_date": temporal.end_date.isoformat() if temporal and temporal.end_date else None,
            "temporal_claim_candidate_count": len(temporal_claim_rank),
            "temporal_episode_candidate_count": len(temporal_episode_rank),
            "temporal_historical_state_candidate_count": temporal_historical,
            "temporal_future_state_filtered_count": future_filtered,
            "selected_temporal_claim_count": 0,
            "selected_temporal_episode_count": 0,
        }

    def select_candidates(self, candidates: Sequence[Dict[str, Any]], extra: Dict[str, Any] | None = None) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        selected = self._select(candidates, int(self.config.get("evidence_count", 8)))
        diagnostics = dict(extra or {})
        selected_claims = [self.store.claims[item["id"]] for item in selected if item["type"] == "state_claim"]
        diagnostics.update({
            "selected_ids": [item["id"] for item in selected],
            "selected_claim_status_counts": dict(sorted(Counter(claim.status for claim in selected_claims).items())),
            "selected_claim_persistence_counts": dict(sorted(Counter(claim.persistence for claim in selected_claims).items())),
            "selected_temporal_claim_count": sum(1 for item in selected if item["type"] == "state_claim" and item.get("temporal_score", 0.0)),
            "selected_temporal_episode_count": sum(1 for item in selected if item["type"] == "episode" and item.get("temporal_score", 0.0)),
        })
        return selected, diagnostics

    def retrieve(self, question: str, query_vector: Sequence[float] | None = None) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        candidates, extra = self.rank_candidates(question, query_vector=query_vector)
        return self.select_candidates(candidates, extra)

    def merge_rank_channels(self, channels: Sequence[Sequence[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Fuse base and planner request rank channels with equal-weight RRF."""
        values: Dict[str, Dict[str, Any]] = {}
        rrf_k = float(self.config.get("rrf_k", 60.0))
        for channel_index, channel in enumerate(channels):
            for rank, item in enumerate(channel, 1):
                identifier = item["id"]
                merged = values.setdefault(identifier, deepcopy(item))
                if channel_index:
                    merged.setdefault("planner_request_indices", []).append(channel_index - 1)
                merged["planner_request_indices"] = sorted(set(merged.get("planner_request_indices", [])))
                merged["planner_channel_support_count"] = int(merged.get("planner_channel_support_count", 0)) + 1
                merged["planner_fusion_score"] = float(merged.get("planner_fusion_score", 0.0)) + 1.0 / (rrf_k + rank)
                merged["score"] = merged["planner_fusion_score"]
                merged["fusion_score"] = merged["score"]
                if channel_index == 0:
                    merged["base_rank"] = rank
        rows = list(values.values())
        rows.sort(key=lambda item: (-float(item.get("score", 0.0)), item["id"]))
        return rows[:int(self.config.get("candidate_count", 40))]

    def _claim_is_directly_visible(self, claim_id: str, temporal: TemporalQueryConstraint | None = None, state_view: str = "current") -> bool:
        claim = self.store.claims.get(claim_id)
        if not claim:
            return False
        if temporal is not None and temporal.kind == "as_of":
            if claim.persistence == "state":
                return claim_visible_as_of(claim, temporal.target_date) is not None
            recorded = parse_stored_date(claim.recorded_at)
            return recorded is not None and recorded <= temporal.target_date
        if temporal is not None and temporal.kind == "before" and claim.persistence == "state":
            recorded = parse_stored_date(claim.recorded_at)
            if recorded is not None and recorded >= temporal.target_date:
                return False
            valid_from = parse_stored_date(claim.valid_from)
            if valid_from is not None and valid_from >= temporal.target_date:
                return False
        return claim.persistence != "state" or state_view == "all_versions" or claim.status in {"active", "contested"}

    def _visible_claim_vectors(self, temporal: TemporalQueryConstraint | None = None, state_view: str = "current") -> Tuple[Dict[str, Sequence[float]], int]:
        vectors = {}
        hidden = 0
        for claim_id, embedding in self.store.claim_embeddings.items():
            if self._claim_is_directly_visible(claim_id, temporal, state_view):
                vectors[claim_id] = embedding
            elif self.store.claims.get(claim_id) and self.store.claims[claim_id].status in {"superseded", "refined"}:
                hidden += 1
        return vectors, hidden

    def _temporal_episode_rank(self, query_vector: Sequence[float], temporal: TemporalQueryConstraint) -> List[Tuple[str, float, str]]:
        rows = []
        for identifier, episode in self.store.episodes.items():
            match = episode_temporal_match(episode, temporal)
            if match is not None:
                rows.append((identifier, cosine(query_vector, self.store.episode_embeddings.get(identifier, [])), match.match_type))
        rows.sort(key=lambda item: (-item[1], self.store.episodes[item[0]].source_session_index is None, self.store.episodes[item[0]].source_session_index or 0, item[0]))
        return rows[: max(0, int(self.config.get("episode_top_k", 20)))]

    def _temporal_claim_rank(self, query_vector: Sequence[float], temporal: TemporalQueryConstraint, state_view: str = "current") -> List[Tuple[str, float, str]]:
        rows = []
        for identifier, claim in self.store.claims.items():
            match = claim_temporal_match(claim, self.store.episodes, temporal)
            if match is not None and self._claim_is_directly_visible(identifier, temporal, state_view):
                rows.append((identifier, cosine(query_vector, self.store.claim_embeddings.get(identifier, [])), match.match_type))
        rows.sort(key=lambda item: (-item[1], self.store.claims[item[0]].recorded_at or "", item[0]))
        return rows[: max(0, int(self.config.get("claim_top_k", 30)))]

    def _rrf(
        self,
        claims: Sequence[Tuple[str, float]],
        episodes: Sequence[Tuple[str, float]],
        temporal_claims: Sequence[Tuple[str, float, str]] = (),
        temporal_episodes: Sequence[Tuple[str, float, str]] = (),
    ) -> List[Dict[str, Any]]:
        values: Dict[str, Dict[str, Any]] = {}
        rrf_k = float(self.config.get("rrf_k", 60.0))
        channels = (("state_claim", claims, float(self.config.get("claim_retrieval_weight", 1.0))), ("episode", episodes, float(self.config.get("episode_retrieval_weight", 1.0))))
        for record_type, rows, weight in channels:
            for rank, (identifier, dense_score) in enumerate(rows, 1):
                item = values.setdefault(identifier, {"id": identifier, "type": record_type, "score": 0.0, "dense_score": dense_score, "fusion_score": 0.0, "ppr_score": 0.0})
                item["score"] += weight / (rrf_k + rank)
                item["fusion_score"] = item["score"]
                item["dense_score"] = max(item["dense_score"], dense_score)
        temporal_weight = float(self.config.get("temporal_retrieval_weight", 1.0))
        for record_type, rows in (("state_claim", temporal_claims), ("episode", temporal_episodes)):
            for rank, (identifier, _dense_score, match_type) in enumerate(rows, 1):
                item = values.setdefault(identifier, {"id": identifier, "type": record_type, "score": 0.0, "dense_score": 0.0, "fusion_score": 0.0, "ppr_score": 0.0})
                item["score"] += temporal_weight / (rrf_k + rank)
                item["fusion_score"] = item["score"]
                match_score = 0.5 if match_type == "as_of_fallback" else 1.0
                item["temporal_score"] = max(float(item.get("temporal_score", 0.0)), match_score)
                item["temporal_match_type"] = item.get("temporal_match_type") or match_type
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
