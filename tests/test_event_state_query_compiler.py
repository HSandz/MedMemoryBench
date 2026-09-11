"""Synthetic contracts for the memory-free Event-State query compiler."""

from datetime import date

import pytest

from methods.event_state.planner import QueryPlan, QueryTemporal, salvage_query_compiler_output
from methods.event_state.prompts import QUERY_COMPILER_SYSTEM_PROMPT
from methods.event_state.retrieval import EventStateRetriever
from methods.event_state.store import EventStateStore


class _Embedder:
    def embed_query(self, _text):
        return [1.0, 0.0]


def _candidate(identifier, score):
    return {"id": identifier, "type": "episode", "score": score, "final_score": score}


def test_salvage_preserves_searches_when_temporal_interval_is_invalid():
    plan, warnings, salvaged = salvage_query_compiler_output({
        "searches": [{"query": "Jordan course", "role": "target"}],
        "temporal": {"axis": "event", "relation": "overlap", "start": "2025-03-21", "end": "2025-03-20", "anchor_search": None, "precision": "bounded"},
        "state_view": "current",
    }, 3)
    assert [search.query for search in plan.searches] == ["Jordan course"]
    assert plan.temporal.relation == "none"
    assert salvaged and any(code.startswith("temporal_") for code in warnings)


def test_salvage_accepts_empty_searches_and_canonicalizes_anchor_role():
    empty, _, _ = salvage_query_compiler_output({
        "searches": [], "temporal": {"axis": "none", "relation": "none", "start": None, "end": None, "anchor_search": None, "precision": "unknown"}, "state_view": "current",
    }, 3)
    assert empty.searches == []
    anchored, warnings, _ = salvage_query_compiler_output({
        "searches": [{"query": "job change", "role": "support"}],
        "temporal": {"axis": "event", "relation": "after", "start": None, "end": None, "anchor_search": 0, "precision": "unknown"},
        "state_view": "all_versions",
    }, 3)
    assert anchored.searches[0].role == "anchor"
    assert "anchor_role_canonicalized" in warnings


def test_no_expansion_uses_the_direct_single_channel_path():
    retriever = EventStateRetriever(EventStateStore(), _Embedder(), candidate_count=5, turn_evidence_count=0)
    direct = [_candidate("E1", .8), _candidate("E2", .2)]
    details = {"pre_candidate_truncation_candidates": direct, "post_candidate_truncation_candidates": direct}
    retriever.rank_candidate_pools = lambda *args, **kwargs: (direct, [], details)  # type: ignore[method-assign]
    retriever.merge_rank_channels = lambda *_args: pytest.fail("single-channel plan must not merge")  # type: ignore[method-assign]
    ranked, turns, diagnostics = retriever.rank_query_plan("simple question", QueryPlan([], QueryTemporal(), "current"), [[1.0, 0.0]])
    assert ranked == direct and turns == []
    assert diagnostics["coverage_merge_mode"] == "single_channel"


def test_fusion_keeps_channel_zero_and_aggregates_support_without_global_cap():
    retriever = EventStateRetriever(EventStateStore(), _Embedder(), candidate_count=1, planner_merge_mode="coverage_interleave")
    merged = retriever.merge_rank_channels([
        [_candidate("original", .95)],
        [_candidate("expansion", .99), _candidate("original", .70)],
    ])
    assert [item["id"] for item in merged] == ["expansion", "original"]
    original = next(item for item in merged if item["id"] == "original")
    assert original["original_query_relevance"] == .95
    assert original["planner_channel_support_count"] == 2


def test_temporal_boost_cannot_create_relevance_from_zero_semantics():
    retriever = EventStateRetriever(EventStateStore(), _Embedder(), temporal_retrieval_weight=1.0)
    retriever._candidate_spans = lambda *_args: [{"start": "2025-03-20", "end": "2025-03-20", "precision": "exact"}]  # type: ignore[method-assign]
    plan = QueryPlan([], QueryTemporal("event", "overlap", date(2025, 3, 20), date(2025, 3, 20), None, "exact"), "current")
    rows = retriever._rerank_query_plan_temporal([_candidate("zero", 0.0), _candidate("semantic", .5)], plan, [])
    assert next(item for item in rows if item["id"] == "zero")["final_score"] == 0.0
    assert next(item for item in rows if item["id"] == "semantic")["final_score"] > .5


def test_compiler_prompt_is_memory_free_and_method_general():
    assert "no memory or other context" in QUERY_COMPILER_SYSTEM_PROMPT
    assert all(term not in QUERY_COMPILER_SYSTEM_PROMPT.casefold() for term in ("locomo", "medmemorybench", "benchmark", "gold evidence", "query_type"))
