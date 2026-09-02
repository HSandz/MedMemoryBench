from datetime import date
from types import SimpleNamespace

from methods.event_state.planner import validate_planner_output
from methods.event_state.prompts import QUERY_PLANNER_SYSTEM_PROMPT
from methods.event_state.schemas import Claim, Episode, EvidenceRef
from methods.event_state.temporal import (
    TemporalQueryConstraint,
    claim_visible_as_of,
    claim_temporal_match,
    episode_temporal_match,
)
from metrics import MetricsAggregator
from metrics.base import MetricResult
from methods.event_state.store import EventStateStore
from methods.event_state_agent import EventStateAgent


def test_planner_record_modes_round_trip_and_knowledge_as_of_requires_as_of():
    request_base = {"query": "record", "sources": "both", "state_view": "current"}
    cases = (
        ("record_exact", {"date": "2024-01-16"}),
        ("record_before", {"date": "2024-01-16"}),
        ("record_after", {"date": "2024-01-16"}),
        ("record_interval", {"start": "2024-01-01", "end": "2024-01-16"}),
    )
    for mode, date_fields in cases:
        request = validate_planner_output({
            "action": "retrieve",
            "answer": None,
            "requests": [{**request_base, "time": {"mode": mode, **date_fields}}],
        }, 3).requests[0]
        assert request.to_dict()["time"]["mode"] == mode

    as_of = validate_planner_output({
        "action": "retrieve",
        "answer": None,
        "requests": [{
            **request_base,
            "state_view": "as_of",
            "time": {"mode": "knowledge_as_of", "date": "2024-01-16"},
        }],
    }, 3).requests[0]
    assert as_of.to_dict()["time"]["mode"] == "knowledge_as_of"
    invalid = validate_planner_output({
        "action": "retrieve",
        "answer": None,
        "requests": [{**request_base, "time": {"mode": "knowledge_as_of", "date": "2024-01-16"}}],
    }, 3)
    assert invalid.requests == []
    assert invalid.invalid_request_count == 1


def test_planner_rejects_removed_valid_time_modes():
    modes = ("valid_exact", "valid_before", "valid_after", "valid_interval")
    requests = [
        {
            "query": mode,
            "sources": "both",
            "state_view": "current",
            "time": (
                {"mode": mode, "date": "2024-01-16"}
                if mode != "valid_interval"
                else {"mode": mode, "start": "2024-01-01", "end": "2024-01-16"}
            ),
        }
        for mode in modes
    ]
    decision = validate_planner_output(
        {"action": "retrieve", "answer": None, "requests": requests}, 4
    )
    assert decision.requests == []
    assert decision.invalid_request_count == len(modes)


def test_planner_event_date_request_remains_semantic_without_record_constraint():
    class LLM:
        def __init__(self):
            self.outputs = [
                '{"action":"retrieve","answer":null,"requests":[{"query":"glucose measurement on 2024-01-16","sources":"both","state_view":"current","time":null}]}',
                "answer",
            ]

        def chat(self, messages, **kwargs):
            return SimpleNamespace(content=self.outputs.pop(0))

    llm = LLM()
    agent = EventStateAgent(
        llm_client=llm,
        memory_llm_client=llm,
        embedding_client=SimpleNamespace(embed_query=lambda _: [1.0, 0.0]),
        planner_rounds=1,
    )
    response = agent.query("What was my glucose measurement on 2024-01-16?")
    assert response.extra["planner"]["requests"][0]["time"] is None
    assert "valid_*" not in QUERY_PLANNER_SYSTEM_PROMPT
    assert "set time=null" in QUERY_PLANNER_SYSTEM_PROMPT


def test_planner_artifact_keeps_bounded_failure_and_candidate_trace():
    store = EventStateStore()
    store.episodes["E1"] = Episode("E1", None, "s1", 0, None, "2024-01-15", [], None, "", "")
    trace = EventStateAgent._candidate_trace([
        {"id": "E1", "type": "episode", "score": 0.8, "temporal_score": 1.0, "temporal_match_type": "exact_record_time"}
    ], store, 1)
    artifact = EventStateAgent._planner_artifact({
        "planner_json_parse_failure_count": 1,
        "planner_schema_validation_failure_count": 0,
        "planner_parse_failure_count": 1,
        "planner_failure_diagnostics": [{"failure_stage": "json_parse", "failure_reason": "bad", "invalid_output_sha256": "x", "invalid_output_preview": "bad"}],
        "candidate_trace": {"base_ranked": trace, "planner_channels": [], "merged_preselect": trace},
    })
    assert artifact["json_parse_failure_count"] == 1
    assert artifact["failure_diagnostics"][0]["failure_stage"] == "json_parse"
    assert artifact["candidate_trace"]["base_ranked"][0]["source_session_ids"] == ["s1"]
    assert "memory" not in artifact["candidate_trace"]["base_ranked"][0]


def test_record_time_does_not_reinterpret_event_or_valid_date():
    episode = Episode("E1", None, "s1", 0, None, "2024-01-15", [], None, "", "")
    claim = Claim(
        "C1", "User", "primary_user", "glucose", "5", persistence="episode",
        recorded_at="2024-01-15", valid_from="2024-01-16", valid_to=None,
        evidence=[EvidenceRef("E1", "s1")],
    )
    record = TemporalQueryConstraint("exact_record_time", date(2024, 1, 16), intent="record_time")
    assert episode_temporal_match(episode, record) is None
    assert claim_temporal_match(claim, {"E1": episode}, record) is None


def test_unknown_valid_time_has_no_hybrid_exact_match_bonus():
    episode = Episode("E1", None, "s1", 0, None, "2024-01-15", [], None, "", "")
    claim = Claim(
        "C1", "User", "primary_user", "glucose", "5", persistence="episode",
        recorded_at="2024-01-15", evidence=[EvidenceRef("E1", "s1")],
    )
    exact = TemporalQueryConstraint("exact_record_time", date(2024, 1, 16), intent="hybrid")
    assert claim_temporal_match(claim, {"E1": episode}, exact) is None
    removed_valid_mode = TemporalQueryConstraint("exact_valid_time", date(2024, 1, 16), intent="valid_time")
    assert claim_temporal_match(claim, {"E1": episode}, removed_valid_mode) is None
    as_of = claim_visible_as_of(claim, date(2024, 1, 16))
    assert as_of is not None and as_of.match_type == "as_of_fallback"


def test_structured_record_exact_matches_authoritative_episode_record_time():
    request = validate_planner_output({
        "action": "retrieve",
        "answer": None,
        "requests": [{
            "query": "conversation record",
            "sources": "episodes",
            "state_view": "current",
            "time": {"mode": "record_exact", "date": "2024-06-03"},
        }],
    }, 3).requests[0]
    episode = Episode("E1", None, "s1", 0, None, "2024-06-03", [], None, "", "")
    assert episode_temporal_match(episode, request.temporal_constraint) is not None


def test_unscored_execution_is_excluded_from_answer_summary_but_keeps_retrieval_quality():
    scored = MetricResult("q1", "state_update", 1.0, True, "ok", "ok")
    unscored = MetricResult(
        "q2", "state_update", None, None, "answer", "expected",
        details={"evaluation_status": "judge_failed", "metric_groups": {"retrieval_quality": {"evaluated": True}}},
    )
    aggregator = MetricsAggregator()
    aggregator.add_results([scored, unscored])
    summary = aggregator.get_summary()
    assert summary["total"] == 1
    assert summary["executed"] == 2
    assert summary["unscored"] == 1
    assert summary["correct"] == 1
