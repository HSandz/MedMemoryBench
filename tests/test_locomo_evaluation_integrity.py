"""LoCoMo evaluation correctness regressions."""

from benchmarks.locomo.dataset import normalize_locomo_timestamp
from benchmarks.locomo.dataset import LoCoMoQuery
from benchmarks.locomo.evaluator import LoCoMoEvaluator
from metrics.locomo_metrics import LoCoMoF1Metric
from methods.event_state.schemas import Episode
from methods.event_state.temporal import parse_stored_date


def test_official_f1_has_no_substring_or_negation_boost():
    metric = LoCoMoF1Metric()
    numeric = metric.compute("numeric", "single_hop", "It was 2023.", ["2"], category=4)
    negated = metric.compute("negated", "single_hop", "The evidence does not say beach.", ["beach"], category=4)

    assert numeric.score == 0.0
    assert negated.score < 0.5
    assert negated.details["enhanced_f1"] < 0.5


def test_official_multihop_and_open_domain_rules():
    metric = LoCoMoF1Metric()

    assert metric.compute("multi", "multi_hop", "red, blue", ["blue, red"], category=1).score == 1.0
    assert metric.compute("open", "open_domain", "yes", ["yes; explanation"], category=3).score == 1.0


def test_timestamp_normalization_enables_generic_temporal_parsing():
    recorded_at = normalize_locomo_timestamp("1:56 pm on 8 May, 2023")

    assert recorded_at == "2023-05-08T13:56:00"
    assert normalize_locomo_timestamp("12:09 am on 13 September, 2023") == "2023-09-13T00:09:00"
    assert normalize_locomo_timestamp("not a timestamp") is None
    episode = Episode("E", "ctx", 1, 0, None, recorded_at, [], "primary_user", "", "")
    assert parse_stored_date(episode.recorded_at).isoformat() == "2023-05-08"


def test_selected_session_and_answer_visible_turn_metrics_are_distinct():
    evaluator = LoCoMoEvaluator.__new__(LoCoMoEvaluator)
    query = LoCoMoQuery(
        query_id="q", question="", query_type="multi_hop", expected_answers=[""],
        evidence=["D2:4", "D4:7"],
    )
    records = [
        {"type": "episode", "source_session_id": 2, "included_in_context": True, "episode_evidence_turn_ids": ["D2:4"]},
        {"type": "state_claim", "source_session_id": 4, "included_in_context": True, "included_provenance_evidence": []},
    ]

    quality = evaluator._locomo_retrieval_quality(query, records)

    assert quality["selected_memory_session"]["recall"] == 1.0
    assert quality["answer_visible_exact_turn"]["recall"] == 0.5
