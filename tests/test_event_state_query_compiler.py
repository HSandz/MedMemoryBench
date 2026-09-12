"""Synthetic contracts for the memory-free Event-State query compiler."""

import hashlib
import json
from datetime import date
from types import SimpleNamespace

import pytest

from benchmarks.locomo.evaluator import LoCoMoEvaluator
from methods.event_state.context import render_claim
from methods.event_state.planner import QueryPlan, QuerySearch, QueryTemporal, salvage_query_compiler_output
from methods.event_state.prompts import QUERY_COMPILER_SYSTEM_PROMPT
from methods.event_state.retrieval import EventStateRetriever
from methods.event_state.schemas import Claim, Episode, EvidenceRef, TurnEvidence
from methods.event_state.store import EventStateStore
from methods.event_state_agent import EventStateAgent


class _Embedder:
    def embed_query(self, _text):
        return [1.0, 0.0]


def _candidate(identifier, score):
    return {"id": identifier, "type": "episode", "score": score, "final_score": score}


def _ranking_fields(rows):
    fields = ("id", "score", "semantic_score", "final_score", "temporal_score", "joint_score")
    return [{field: row[field] for field in fields if field in row} for row in rows]


def _event_none_selection_fixture():
    agent = EventStateAgent(
        llm_client=SimpleNamespace(chat=lambda *_args, **_kwargs: SimpleNamespace(content="answer")),
        memory_llm_client=SimpleNamespace(chat=lambda *_args, **_kwargs: SimpleNamespace(content="answer")),
        embedding_client=_Embedder(), retrieve_episodes=False, candidate_count=3,
        evidence_count=2, turn_evidence_count=2, selector_mode="topk",
    )
    store = EventStateStore("ctx")
    for index, (identifier, claim_vector, event_day) in enumerate((
        ("A", [1.0, 0.0], None),
        ("B", [0.9, 0.1], "2025-03-20"),
        ("C", [0.2, 0.8], "2025-03-20"),
    )):
        episode_id, turn_id = f"E{identifier}", f"T{identifier}"
        episode = Episode(
            episode_id, "ctx", f"session-{identifier}", index, None, "2025-03-21",
            ["User"], "primary_user", "", f"episode {identifier}",
            [TurnEvidence(turn_id, "User", "user", f"event {identifier}")],
        )
        store.add_episode(episode, [0.0, 1.0], [claim_vector])
        event_fields = {} if event_day is None else {
            "event_time_start": event_day, "event_time_end": event_day,
            "event_time_precision": "exact",
        }
        store.add_claim(Claim(
            identifier, "User", "primary_user", "did", f"event {identifier}",
            persistence="episode", evidence=[EvidenceRef(episode_id, f"session-{identifier}", [turn_id])],
            **event_fields,
        ), claim_vector)
    return agent, store


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


def test_event_none_is_temporal_rerank_noop():
    retriever = EventStateRetriever(EventStateStore(), _Embedder(), temporal_retrieval_weight=1.0)
    spans = {
        "B": [{"start": "2025-03-20", "end": "2025-03-20", "precision": "exact"}],
        "C": [{"start": "2025-03-20", "end": "2025-03-20", "precision": "exact"}],
    }
    retriever._candidate_spans = lambda item, _axis: spans.get(item["id"], [])  # type: ignore[method-assign]
    candidates = [_candidate("A", .90), _candidate("B", .89), _candidate("C", .20)]
    no_temporal = retriever._rerank_query_plan_temporal(
        [dict(item) for item in candidates], QueryPlan([], QueryTemporal(), "current"), [],
    )
    event_none = retriever._rerank_query_plan_temporal(
        [dict(item) for item in candidates], QueryPlan([], QueryTemporal("event", "none"), "current"), [],
    )

    assert event_none == no_temporal == candidates
    assert retriever._temporal_compatibility(spans["B"], QueryTemporal("event", "none"), []) == (0.0, None)


def test_event_none_does_not_boost_dated_candidate():
    retriever = EventStateRetriever(EventStateStore(), _Embedder(), temporal_retrieval_weight=1.0)
    spans = {
        "B": [{"start": "2025-03-20", "end": "2025-03-20", "precision": "exact"}],
        "C": [{"start": "2025-03-20", "end": "2025-03-20", "precision": "exact"}],
    }
    retriever._candidate_spans = lambda item, _axis: spans.get(item["id"], [])  # type: ignore[method-assign]
    rows = retriever._rerank_query_plan_temporal(
        [_candidate("A", .90), _candidate("B", .89), _candidate("C", .20)],
        QueryPlan([], QueryTemporal("event", "none"), "current"), [],
    )

    assert [(item["id"], item["final_score"]) for item in rows] == [
        ("A", .90), ("B", .89), ("C", .20),
    ]
    assert all("temporal_score" not in item and "joint_score" not in item for item in rows)


def test_event_none_matches_none_none_selection():
    agent, store = _event_none_selection_fixture()
    plans = {
        "none": QueryPlan([QuerySearch("job start", "target")], QueryTemporal(), "current"),
        "event_none": QueryPlan([QuerySearch("job start", "target")], QueryTemporal("event", "none"), "current"),
    }

    results = {}
    for name, plan in plans.items():
        retriever = EventStateRetriever(store, _Embedder(), **agent._retrieval_config)
        structured, turns, diagnostics = retriever.rank_query_plan(
            "When did someone start a job?", plan, [[1.0, 0.0], [1.0, 0.0]],
        )
        ranking = (_ranking_fields(structured), _ranking_fields(turns))
        selected, selected_diagnostics = agent._select_evidence_pools(
            retriever, structured, turns, diagnostics, [[1.0, 0.0], [1.0, 0.0]],
        )
        results[name] = {
            "ranking": ranking,
            "structured_ids": [item["id"] for item in selected if item["type"] != "turn"],
            "direct_turn_ids": [item["id"] for item in selected if item["type"] == "turn"],
            "diagnostics": selected_diagnostics,
        }

    assert results["event_none"]["ranking"] == results["none"]["ranking"]
    assert results["event_none"]["structured_ids"] == results["none"]["structured_ids"]
    assert results["event_none"]["direct_turn_ids"] == results["none"]["direct_turn_ids"]
    assert results["event_none"]["diagnostics"]["temporalized_candidate_count"] == 0


def test_event_none_temporalized_candidate_count_is_zero():
    retriever = EventStateRetriever(EventStateStore(), _Embedder(), candidate_count=5, turn_evidence_count=0)
    direct = [_candidate("A", .90), _candidate("B", .89)]
    details = {"pre_candidate_truncation_candidates": direct, "post_candidate_truncation_candidates": direct}
    retriever.rank_candidate_pools = lambda *args, **kwargs: (direct, [], details)  # type: ignore[method-assign]

    _ranked, _turns, diagnostics = retriever.rank_query_plan(
        "When did someone start a job?", QueryPlan([], QueryTemporal("event", "none"), "current"), [[1.0, 0.0]],
    )

    assert diagnostics["temporalized_candidate_count"] == 0


def test_real_temporal_relations_still_rerank():
    retriever = EventStateRetriever(EventStateStore(), _Embedder(), temporal_retrieval_weight=1.0)
    spans = {
        "match": [{"start": "2025-03-20", "end": "2025-03-20", "precision": "exact"}],
        "old": [{"start": "2025-01-01", "end": "2025-01-01", "precision": "exact"}],
        "new": [{"start": "2025-04-01", "end": "2025-04-01", "precision": "exact"}],
    }
    retriever._candidate_spans = lambda item, _axis: spans.get(item["id"], [])  # type: ignore[method-assign]
    overlap = retriever._rerank_query_plan_temporal(
        [_candidate("match", .80), _candidate("old", .79)],
        QueryPlan([], QueryTemporal("event", "overlap", date(2025, 3, 20), date(2025, 3, 20), None, "exact"), "current"), [],
    )
    latest = retriever._rerank_query_plan_temporal(
        [_candidate("old", .80), _candidate("new", .80)],
        QueryPlan([], QueryTemporal("event", "latest"), "current"), [],
    )

    assert overlap[0]["id"] == "match" and overlap[0]["final_score"] > .80
    assert latest[0]["id"] == "new"


def test_query_compiler_cache_fingerprint_includes_max_tokens_with_legacy_default_read():
    evaluator = LoCoMoEvaluator.__new__(LoCoMoEvaluator)
    evaluator.method_config = SimpleNamespace(
        model=SimpleNamespace(provider="openai", name="test"),
        raw_config={"retrieval_config": {"query_compiler_max_tokens": 256}},
    )
    payload = evaluator._query_compiler_cache_fingerprint_payload(
        "When did someone start a job?", include_max_tokens=False,
    )
    legacy_fingerprint = hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    ).hexdigest()
    legacy_entry = {"fingerprint": legacy_fingerprint, "raw_model_output": "{}"}

    assert evaluator._query_compiler_cache_fingerprint("When did someone start a job?") != legacy_fingerprint
    assert evaluator._legacy_query_compiler_cache_entry(
        {legacy_fingerprint: legacy_entry}, "When did someone start a job?",
    ) == legacy_entry
    evaluator.method_config.raw_config["retrieval_config"]["query_compiler_max_tokens"] = 128
    assert evaluator._legacy_query_compiler_cache_entry(
        {legacy_fingerprint: legacy_entry}, "When did someone start a job?",
    ) is None


def test_compiler_prompt_is_memory_free_and_method_general():
    assert "no memory or other context" in QUERY_COMPILER_SYSTEM_PROMPT
    assert all(term not in QUERY_COMPILER_SYSTEM_PROMPT.casefold() for term in ("locomo", "medmemorybench", "benchmark", "gold evidence", "query_type"))


@pytest.mark.parametrize(
    ("raw", "relation", "start", "end", "warning"),
    [
        ({"axis": "event", "relation": "before", "start": None, "end": "2025-03-20", "anchor_search": None, "precision": "exact"}, "before", "2025-03-20", "2025-03-20", "before_single_boundary_canonicalized"),
        ({"axis": "event", "relation": "before", "start": "2025-03-20", "end": None, "anchor_search": None, "precision": "exact"}, "before", "2025-03-20", "2025-03-20", "before_single_boundary_canonicalized"),
        ({"axis": "event", "relation": "after", "start": "2025-03-20", "end": None, "anchor_search": None, "precision": "exact"}, "after", "2025-03-20", "2025-03-20", "after_single_boundary_canonicalized"),
        ({"axis": "event", "relation": "after", "start": None, "end": "2025-03-20", "anchor_search": None, "precision": "exact"}, "after", "2025-03-20", "2025-03-20", "after_single_boundary_canonicalized"),
        ({"axis": "event", "relation": "none", "start": "2025-03-20", "end": "2025-03-20", "anchor_search": None, "precision": "exact"}, "none", None, None, "none_fields_cleared"),
    ],
)
def test_temporal_canonicalization_has_one_boundary_contract(raw, relation, start, end, warning):
    plan, warnings, salvaged = salvage_query_compiler_output({"searches": [{"query": "event", "role": "target"}], "temporal": raw, "state_view": "current"}, 3)
    assert salvaged and warning in warnings
    assert plan.temporal.relation == relation
    assert (plan.temporal.start.isoformat() if plan.temporal.start else None) == start
    assert (plan.temporal.end.isoformat() if plan.temporal.end else None) == end


def test_reversed_temporal_constraint_preserves_semantic_searches():
    plan, warnings, _ = salvage_query_compiler_output({
        "searches": [{"query": "useful semantic request", "role": "target"}],
        "temporal": {"axis": "event", "relation": "overlap", "start": "2025-03-21", "end": "2025-03-20", "anchor_search": None, "precision": "bounded"},
        "state_view": "current",
    }, 3)
    assert [search.query for search in plan.searches] == ["useful semantic request"]
    assert plan.temporal == QueryTemporal()
    assert "temporal_reversed_interval" in warnings


def test_knowledge_as_of_requires_its_complete_contract():
    valid, _, _ = salvage_query_compiler_output({
        "searches": [],
        "temporal": {"axis": "knowledge", "relation": "as_of", "start": None, "end": "2025-03-20", "anchor_search": None, "precision": "exact"},
        "state_view": "as_of",
    }, 3)
    invalid, warnings, _ = salvage_query_compiler_output({
        "searches": [],
        "temporal": {"axis": "knowledge", "relation": "as_of", "start": "2025-03-19", "end": "2025-03-20", "anchor_search": None, "precision": "bounded"},
        "state_view": "as_of",
    }, 3)
    assert valid.temporal.relation == "as_of"
    assert invalid.temporal == QueryTemporal()
    assert "temporal_invalid_as_of" in warnings


def test_explicit_before_after_and_overlap_use_strict_canonical_bounds():
    retriever = EventStateRetriever(EventStateStore(), _Embedder())
    spans = lambda day: [{"start": day, "end": day, "precision": "exact"}]
    before = QueryTemporal("event", "before", date(2025, 3, 20), date(2025, 3, 20), None, "exact")
    after = QueryTemporal("event", "after", date(2025, 3, 20), date(2025, 3, 20), None, "exact")
    overlap = QueryTemporal("event", "overlap", date(2025, 3, 20), date(2025, 3, 20), None, "exact")
    assert retriever._temporal_compatibility(spans("2025-03-19"), before, [])[0] > 0
    assert retriever._temporal_compatibility(spans("2025-03-20"), before, [])[0] == 0
    assert retriever._temporal_compatibility(spans("2025-03-21"), after, [])[0] > 0
    assert retriever._temporal_compatibility(spans("2025-03-20"), after, [])[0] == 0
    assert retriever._temporal_compatibility(spans("2025-03-20"), overlap, [])[0] > 0


def test_latest_and_earliest_order_only_semantic_candidates():
    retriever = EventStateRetriever(EventStateStore(), _Embedder(), temporal_retrieval_weight=1.0)
    retriever._candidate_spans = lambda item, _axis: [{"start": item["day"], "end": item["day"], "precision": "exact"}]  # type: ignore[method-assign]
    rows = [_candidate("old", .5) | {"day": "2025-01-01"}, _candidate("new", .5) | {"day": "2025-03-01"}, _candidate("irrelevant", 0.0) | {"day": "2026-01-01"}]
    latest = retriever._rerank_query_plan_temporal([dict(row) for row in rows], QueryPlan([], QueryTemporal("event", "latest"), "current"), [])
    earliest = retriever._rerank_query_plan_temporal([dict(row) for row in rows], QueryPlan([], QueryTemporal("event", "earliest"), "current"), [])
    assert latest[0]["id"] == "new"
    assert earliest[0]["id"] == "old"
    assert latest[-1]["id"] == "irrelevant"


def test_event_time_is_rendered_without_lifecycle_confusion_and_valid_from_is_not_inferred():
    claim = Claim("C", "User", "primary_user", "did", "an activity", recorded_at="2025-03-20", valid_time_text="yesterday", event_time_start="2025-03-19", event_time_end="2025-03-19", event_time_precision="exact")
    rendered = render_claim(claim, [], {claim.claim_id: claim})
    assert "Recorded: 2025-03-20" in rendered
    assert "Event time: 2025-03-19" in rendered
    assert "Event-time precision: exact" in rendered
    assert "Source temporal wording: yesterday" in rendered
    assert EventStateAgent._valid_from(None, "2025-03-20", "yesterday", True) is None


def test_anchor_resolution_uses_claim_or_exact_turn_provenance_not_episode_union():
    store = EventStateStore("ctx")
    episode = Episode("E", "ctx", "s", 0, None, "2025-03-20", ["User"], "primary_user", "", "two events", [TurnEvidence("a", "User", "user", "event A"), TurnEvidence("b", "User", "user", "event B")])
    store.add_episode(episode, [1.0, 0.0], [[1.0, 0.0], [1.0, 0.0]])
    a = Claim("A", "User", "primary_user", "event", "A", evidence=[EvidenceRef("E", "s", ["a"])], event_time_start="2025-01-05", event_time_end="2025-01-05", event_time_precision="exact")
    b = Claim("B", "User", "primary_user", "event", "B", evidence=[EvidenceRef("E", "s", ["b"])], event_time_start="2025-02-12", event_time_end="2025-02-12", event_time_precision="exact")
    store.add_claim(a, [1.0, 0.0])
    store.add_claim(b, [1.0, 0.0])
    store.rebuild_temporal_indexes()
    retriever = EventStateRetriever(store, _Embedder())
    plan = QueryPlan([QuerySearch("A", "anchor")], QueryTemporal("event", "after", anchor_search=0), "current")
    anchor_rows = [{"id": "A", "type": "state_claim", "final_score": 1.0}, {"id": "E", "type": "episode", "final_score": 0.9}]
    spans, diagnostics = retriever._resolve_anchor_spans(plan, [[], anchor_rows], [])
    assert spans[0]["start"] == "2025-01-05"
    assert diagnostics["resolved_anchor_source_ids"] == ["A"]


def test_incompatible_precise_anchor_spans_are_ambiguous_and_missing_metadata_is_neutral():
    store = EventStateStore("ctx")
    episode = Episode("E", "ctx", "s", 0, None, "2025-03-20", ["User"], "primary_user", "", "events", [])
    store.add_episode(episode, [1.0, 0.0])
    first = Claim("A", "User", "primary_user", "event", "first", evidence=[EvidenceRef("E", "s", [])], event_time_start="2025-01-05", event_time_end="2025-01-05", event_time_precision="exact")
    second = Claim("B", "User", "primary_user", "event", "second", evidence=[EvidenceRef("E", "s", [])], event_time_start="2025-02-12", event_time_end="2025-02-12", event_time_precision="exact")
    store.add_claim(first, [1.0, 0.0])
    store.add_claim(second, [1.0, 0.0])
    retriever = EventStateRetriever(store, _Embedder())
    plan = QueryPlan([QuerySearch("anchor", "anchor")], QueryTemporal("event", "before", anchor_search=0), "current")
    rows = [{"id": "A", "type": "state_claim", "final_score": 1.0}, {"id": "B", "type": "state_claim", "final_score": 1.0}]
    spans, diagnostics = retriever._resolve_anchor_spans(plan, [[], rows], [])
    assert spans == [] and diagnostics["anchor_resolution_status"] == "ambiguous"
    spans, diagnostics = retriever._resolve_anchor_spans(plan, [[], [{"id": "E", "type": "episode", "final_score": 1.0}]], [])
    assert spans == [] and diagnostics["anchor_resolution_status"] == "missing_metadata"
