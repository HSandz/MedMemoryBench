"""LoCoMo evaluation correctness regressions."""

import json
from pathlib import Path
from types import SimpleNamespace

from benchmarks.locomo.dataset import normalize_locomo_timestamp
from benchmarks.locomo.dataset import LoCoMoQuery
from benchmarks.locomo.evaluator import LoCoMoEvaluator
from benchmarks.medmemorybench.checkpoint import compute_build_config_hash
from metrics import MetricResult, MetricsAggregator
from metrics.locomo_metrics import LoCoMoF1Metric
from methods.event_state.schemas import Episode
from methods.event_state.temporal import parse_stored_date, parse_temporal_query
from src.config import ConfigLoader
from src.result import EvaluationReport, ResultCollector, _efficiency_with_timing_semantics


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


def test_locomo_memory_resume_accepts_v2_manifest(tmp_path: Path):
    loader = ConfigLoader()
    method_config = loader.load_method_config("event_state_gemini")
    dataset_config = loader.load_dataset_config("locomo_1")
    manifest = {
        "format": "locomo.event_state_memory_manifest",
        "version": 2,
        "build_config_hash": compute_build_config_hash(method_config, dataset_config),
    }
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    (memory_dir / "manifest.json").write_text(json.dumps(manifest))

    evaluator = LoCoMoEvaluator.__new__(LoCoMoEvaluator)
    evaluator.resume = True
    evaluator.output_dir = tmp_path
    evaluator.memory_source_run_dir = None
    evaluator.method_config = method_config
    evaluator.dataset_config = dataset_config

    evaluator._start_memory_snapshot_manifest([])

    assert evaluator._memory_snapshot_manifest == manifest
    assert evaluator._memory_snapshot_dir_path == memory_dir


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


def test_gold_evidence_parser_expands_packed_ids_without_inventing_ids():
    evaluator = LoCoMoEvaluator.__new__(LoCoMoEvaluator)

    assert evaluator._gold_evidence_turns(["D2:4", "D4:7"]) == ["D2:4", "D4:7"]
    assert evaluator._gold_evidence_turns(["D8:6; D9:17"]) == ["D8:6", "D9:17"]
    assert evaluator._gold_evidence_turns(["D8:6;D9:17"]) == ["D8:6", "D9:17"]
    assert evaluator._gold_evidence_turns(["D1:3", "D2:4; D3:5"]) == ["D1:3", "D2:4", "D3:5"]
    assert evaluator._gold_evidence_turns(["D1:3; D1:3", "D2:4"]) == ["D1:3", "D2:4"]
    assert evaluator._gold_evidence_turns(["D1", "not evidence", "Dtwo:4", "D1:two:4"]) == []


def test_semicolon_evidence_counts_as_gold_retrieval_evidence():
    evaluator = LoCoMoEvaluator.__new__(LoCoMoEvaluator)
    query = LoCoMoQuery(
        query_id="q", question="", query_type="multi_hop", expected_answers=[""],
        evidence=["D8:6; D9:17"],
    )
    quality = evaluator._locomo_retrieval_quality(query, [])

    assert quality["gold_evidence_turn_ids"] == ["D8:6", "D9:17"]
    assert quality["gold_evidence_session_ids"] == ["8", "9"]
    assert quality["selected_memory_session"]["available"] is True


def test_retrieval_aggregate_counts_packed_evidence_without_a_fixed_total():
    evaluator = LoCoMoEvaluator.__new__(LoCoMoEvaluator)
    packed = LoCoMoQuery(
        query_id="packed", question="", query_type="multi_hop", expected_answers=[""],
        evidence=["D8:6; D9:17"],
    )
    missing = LoCoMoQuery(
        query_id="missing", question="", query_type="multi_hop", expected_answers=[""],
        evidence=["not an evidence id"],
    )
    results = []
    for query in (packed, missing):
        results.append(MetricResult(
            query_id=query.query_id, query_type=query.query_type, score=0.0,
            is_correct=False, model_output="", expected_answer="", question="",
            details={"metric": "locomo_f1", "locomo_retrieval_quality": (
                evaluator._locomo_retrieval_quality(query, [])
            )},
        ))

    summary = evaluator._aggregate_locomo_retrieval(results)

    assert summary["selected_memory_session"]["queries_with_gold_evidence"] == 1
    assert summary["selected_memory_session"]["queries_without_gold_evidence"] == 1


def test_locomo_batch_usage_and_wall_time_are_stage_level(tmp_path):
    evaluator = LoCoMoEvaluator.__new__(LoCoMoEvaluator)
    evaluator._batch_retrieval_preparation_wall_time = 42.0
    evaluator.aggregator = MetricsAggregator()
    for input_tokens, output_tokens in ((100, 10), (200, 20)):
        evaluator.aggregator.add_result(MetricResult(
            query_id=str(input_tokens), query_type="single_hop", score=1.0,
            is_correct=True, model_output="", expected_answer="", question="",
            details={"execution_usage": {"answer": {
                "transport": "batch", "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            }}},
        ))
    manifest_path = tmp_path / "batch.json"
    manifest_path.write_text(json.dumps({"jobs": {"query-final": {
        "state": "completed", "requests": [{}, {}],
        "submitted_at": "2026-09-04T10:00:00+00:00",
        "completed_at": "2026-09-04T10:04:10+00:00",
    }}}), encoding="utf-8")
    evaluator._batch_client = SimpleNamespace(manifest_path=manifest_path)

    stage_usage = evaluator._stage_usage_report({"operations": {"query": {
        "query.retrieval_preparation": {"wall_time": 30.0},
    }}})

    assert stage_usage["answer_generation"]["transport"] == "batch"
    assert stage_usage["answer_generation"]["usage"] == {
        "input_tokens": 300, "output_tokens": 30, "total_tokens": 330,
        "request_count": 2, "successful_requests": 2, "transport": "batch",
    }
    assert stage_usage["answer_generation"]["batch_wall_time_seconds"] == 250.0
    assert stage_usage["batch_stages"][0]["wall_time_seconds"] == 250.0
    assert stage_usage["retrieval_preparation"]["operation_wall_time_seconds"] == 30.0
    assert stage_usage["retrieval_preparation"]["end_to_end_wall_time_seconds"] == 42.0


def test_batch_wall_time_rejects_missing_or_malformed_timestamps():
    assert LoCoMoEvaluator._batch_wall_time_seconds({}) is None
    assert LoCoMoEvaluator._batch_wall_time_seconds({
        "submitted_at": "not a timestamp", "completed_at": "2026-09-04T10:04:10+00:00",
    }) is None


def test_efficiency_prefers_explicit_end_to_end_stage_time():
    report = EvaluationReport(
        method_name="event_state", model_name="test", dataset_name="locomo",
        start_time="", end_time="", duration_seconds=0.0, summary={"efficiency": {}},
        metadata={"stage_usage": {
            "retrieval_preparation": {
                "usage": {"wall_time": 30.0},
                "end_to_end_wall_time_seconds": 42.0,
            },
            "answer_generation": {}, "judge": {}, "batch_stages": [{}],
        }},
    )

    assert _efficiency_with_timing_semantics(report)["stage_wall_time_seconds"]["retrieval_preparation"] == 42.0


def test_result_serializes_locomo_coverage_modality_and_f1_terminology(tmp_path):
    evaluator = LoCoMoEvaluator.__new__(LoCoMoEvaluator)
    summary = {
        "by_type": {"single_hop": {"total": 1, "correct": 1, "accuracy": 1.0, "avg_score": 0.8}},
        "by_metric": {"locomo_f1": {"total": 1, "correct": 1, "accuracy": 1.0, "avg_score": 0.8}},
    }
    evaluator.aggregator = MetricsAggregator()
    evaluator._apply_locomo_summary(summary)
    report = EvaluationReport(
        method_name="event_state", model_name="test", dataset_name="locomo",
        start_time="", end_time="", duration_seconds=0.0, summary=summary,
        metadata={
            "dataset_coverage": {
                "available_sample_count": 10, "evaluated_sample_count": 1,
                "configured_max_samples": 1, "sample_ids": ["conv-26"],
            },
            "input_modality": {
                "image_input_mode": "caption_only", "image_caption_field": "blip_caption",
            },
        },
    )
    result_path, _, _ = ResultCollector().save_reports(
        report, tmp_path, [], include_memory_build=False, include_query_answer=False,
        use_method_subdir=False,
    )
    payload = json.loads(Path(result_path).read_text(encoding="utf-8"))

    assert payload["dataset_coverage"]["evaluated_sample_count"] == 1
    assert payload["input_modality"]["image_input_mode"] == "caption_only"
    assert "accuracy" not in payload["summary"]["by_metric"]["locomo_f1"]
    assert payload["summary"]["by_metric"]["locomo_f1"]["fraction_f1_ge_0_5"] == 1.0


def test_event_state_locomo_reporting_uses_record_and_timing_semantics(tmp_path):
    evaluator = LoCoMoEvaluator.__new__(LoCoMoEvaluator)
    evaluator.method_config = SimpleNamespace(
        method_name="event_state",
        build_config={"event_state_semantic_version": "2.9"},
        retrieval_config={
            "planner_rounds": 0,
            "ppr_enabled": False,
            "temporal_retrieval_enabled": True,
            "selector_mode": "state_mmr",
            "evidence_count": 8,
            "claim_top_k": 30,
            "episode_top_k": 20,
            "candidate_count": 40,
        },
        embedding=SimpleNamespace(model="sentence-transformers/all-MiniLM-L6-v2"),
    )
    evaluator.memory_chunk_size = 0
    evaluator._memory_build_logs = [{
        "reporting_kind": "event_state",
        "unit_id": 0,
        "context_id": "conv-26",
        "session_ids": ["D1", "D2"],
        "session_count": 2,
        "total_time": 1000.0,
        "inserted_record_count": 88,
        "staged_session_count": 2,
        "build_metrics": {
            "chunk_count": 88,
            "inserted_record_count": 88,
            "final_episode_count": 2,
            "final_claim_count": 7,
            "final_memory_object_count": 9,
        },
        "final_store": {
            "final_episode_count": 2,
            "final_claim_count": 7,
            "final_memory_object_count": 9,
        },
    }]

    build_summary = evaluator._summarize_memory_builds()
    compact_metrics = evaluator._compact_build_metrics()
    assert build_summary["inserted_record_count"] == 88
    assert build_summary["avg_time_per_unit"] == 1000.0
    assert "total_memory_chunks" not in build_summary
    assert compact_metrics["schema_version"] == 2
    assert "chunk_count" not in compact_metrics["units"]["conv-26"]
    assert compact_metrics["units"]["conv-26"]["inserted_record_count"] == 88

    aggregator = MetricsAggregator()
    for query_id in ("q1", "q2"):
        aggregator.add_result(MetricResult(
            query_id=query_id, query_type="single_hop", score=1.0,
            is_correct=True, model_output="", expected_answer="", question="",
            memory_construction_time=500.0,
        ))
    efficiency = aggregator.get_summary()["efficiency"]
    assert efficiency["total_memory_construction_time"] == 1000.0
    assert efficiency["amortized_memory_construction_time_per_query"] == 500.0
    assert efficiency["avg_memory_construction_time"] == 500.0
    assert efficiency["avg_memory_construction_time_semantics"] == (
        "legacy_alias_for_amortized_memory_construction_time_per_query"
    )

    report = EvaluationReport(
        method_name="event_state", model_name="test", dataset_name="locomo",
        start_time="", end_time="", duration_seconds=1000.0,
        summary={"total": 2, "efficiency": efficiency},
        metadata={
            "memory_build_summary": build_summary,
            "build_metrics": compact_metrics,
            "memory_size": evaluator._event_state_memory_size(),
            "feature_configuration": evaluator._event_state_feature_configuration(),
        },
    )
    result_path, memory_build_path, _ = ResultCollector().save_reports(
        report, tmp_path, evaluator._memory_build_logs,
        include_query_answer=False, use_method_subdir=False,
    )
    result_data = json.loads(Path(result_path).read_text(encoding="utf-8"))
    build_data = json.loads(Path(memory_build_path).read_text(encoding="utf-8"))

    assert result_data["memory_size"] == {
        "final_episode_count": 2,
        "final_claim_count": 7,
        "final_memory_object_count": 9,
    }
    assert result_data["feature_configuration"]["semantic_version"] == "2.9"
    assert result_data["feature_configuration"]["planner_enabled"] is False
    assert "chunk_count" not in build_data["units"][0]
    assert build_data["units"][0]["inserted_record_count"] == 88
    assert "chunk_count" not in build_data["units"][0]["build_metrics"]


def test_legacy_locomo_chunk_metrics_remain_serialized(tmp_path):
    report = EvaluationReport(
        method_name="embedding_rag", model_name="test", dataset_name="locomo",
        start_time="", end_time="", duration_seconds=0.0, summary={}, metadata={},
    )
    logs = [{
        "unit_id": 0,
        "context_id": "conv-26",
        "build_result": {"method": "embedding_rag", "chunk_count": 4},
    }]

    _, memory_build_path, _ = ResultCollector().save_reports(
        report, tmp_path, logs, include_result=False,
        include_query_answer=False, use_method_subdir=False,
    )
    build_data = json.loads(Path(memory_build_path).read_text(encoding="utf-8"))

    assert build_data["units"][0]["chunk_count"] == 4


def test_month_name_temporal_dates_are_complete_and_generic():
    for question in (
        "What happened on October 13, 2023?",
        "What happened on 13 October 2023?",
        "What happened on Oct 13, 2023?",
    ):
        assert parse_temporal_query(question).target_date.isoformat() == "2023-10-13"
    assert parse_temporal_query("What happened in October?") is None
    assert parse_temporal_query("What happened on October 13?") is None
