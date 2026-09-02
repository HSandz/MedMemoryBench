from datetime import date
from methods.event_state.planner import PlannerRequest, validate_planner_output
from methods.event_state.schemas import Claim, Episode, EvidenceRef
from methods.event_state.temporal import (
    TemporalQueryConstraint,
    claim_temporal_match,
    episode_temporal_match,
)
from metrics import MetricsAggregator
from metrics.base import MetricResult
from methods.event_state.store import EventStateStore
from methods.event_state_agent import EventStateAgent


def test_planner_temporal_axis_round_trip_and_distinct_keys():
    record = validate_planner_output({
        "action": "retrieve", "answer": None, "requests": [{
            "query": "record", "sources": "both", "state_view": "current",
            "time": {"mode": "record_exact", "date": "2024-01-16"},
        }]
    }, 3).requests[0]
    valid = validate_planner_output({
        "action": "retrieve", "answer": None, "requests": [{
            "query": "event", "sources": "both", "state_view": "current",
            "time": {"mode": "valid_exact", "date": "2024-01-16"},
        }]
    }, 3).requests[0]
    assert record.to_dict()["time"]["mode"] == "record_exact"
    assert valid.to_dict()["time"]["mode"] == "valid_exact"
    assert record.key() != valid.key()


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


def test_record_and_valid_time_do_not_reinterpret_episode_record_date():
    episode = Episode("E1", None, "s1", 0, None, "2024-01-15", [], None, "", "")
    claim = Claim(
        "C1", "User", "primary_user", "glucose", "5", persistence="episode",
        recorded_at="2024-01-15", valid_from="2024-01-16", valid_to=None,
        evidence=[EvidenceRef("E1", "s1")],
    )
    record = TemporalQueryConstraint("exact_record_time", date(2024, 1, 16), intent="record_time")
    valid = TemporalQueryConstraint("exact_valid_time", date(2024, 1, 16), intent="valid_time")
    assert episode_temporal_match(episode, record) is None
    assert episode_temporal_match(episode, valid) is None
    assert claim_temporal_match(claim, {"E1": episode}, record) is None
    assert claim_temporal_match(claim, {"E1": episode}, valid) is not None


def test_valid_state_interval_is_half_open_and_episode_start_is_conservative():
    episode = Episode("E1", None, "s1", 0, None, "2024-01-15", [], None, "", "")
    state = Claim("S", "User", "primary_user", "residence", "A", persistence="state", valid_from="2024-01-01", valid_to="2024-02-01", evidence=[EvidenceRef("E1")])
    point = Claim("P", "User", "primary_user", "glucose", "5", persistence="episode", valid_from="2024-01-16", valid_to=None, evidence=[EvidenceRef("E1")])
    inside = TemporalQueryConstraint("exact_valid_time", date(2024, 1, 31), intent="valid_time")
    boundary = TemporalQueryConstraint("exact_valid_time", date(2024, 2, 1), intent="valid_time")
    later = TemporalQueryConstraint("exact_valid_time", date(2024, 1, 20), intent="valid_time")
    assert claim_temporal_match(state, {"E1": episode}, inside) is not None
    assert claim_temporal_match(state, {"E1": episode}, boundary) is None
    assert claim_temporal_match(point, {"E1": episode}, later) is None


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
