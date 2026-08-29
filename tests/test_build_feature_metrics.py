"""Regression coverage for report-ready AMEM build telemetry."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from benchmarks.medmemorybench.evaluator import MedMemoryBenchEvaluator
from src.result import EvaluationReport, ResultCollector
from utils.llm_client import LLMResponse, diff_usage_stats, get_usage_tracker


def _record_success(input_tokens: int, output_tokens: int, latency: float) -> None:
    tracker = get_usage_tracker()
    tracker.record_attempt()
    tracker.record(LLMResponse(
        content="ok",
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency=latency,
        model="build-model",
    ))


def test_scoped_usage_tracks_feature_calls_retries_and_wall_time():
    tracker = get_usage_tracker()
    tracker.reset()
    tracker.set_phase("memorize")
    before = tracker.get_stats()

    with tracker.scope("amem.base.note_analysis"):
        _record_success(10, 2, 0.25)

    with tracker.scope("amem.typed_relations.inference"):
        tracker.record_attempt()
        tracker.record_failure()
        tracker.record_retry()
        _record_success(20, 3, 0.5)

    delta = diff_usage_stats(tracker.get_stats(), before)
    totals = delta["memorize_phase"]
    typed = delta["operations"]["memorize"]["amem.typed_relations.inference"]

    assert totals["input_tokens"] == 30
    assert totals["output_tokens"] == 5
    assert totals["successful_calls"] == 2
    assert totals["attempted_calls"] == 3
    assert totals["failed_attempts"] == 1
    assert totals["retry_count"] == 1
    assert typed["successful_calls"] == 1
    assert typed["attempted_calls"] == 2
    assert typed["failed_attempts"] == 1
    assert typed["operation_count"] == 1
    assert typed["wall_time"] >= 0


def _metrics_evaluator() -> MedMemoryBenchEvaluator:
    evaluator = MedMemoryBenchEvaluator.__new__(MedMemoryBenchEvaluator)
    build_config = {
        "amem_original_evolution": False,
        "amem_typed_relations": True,
        "amem_temporal_state": True,
        "amem_provenance": False,
    }
    evaluator.method_config = SimpleNamespace(
        method_name="amem_test",
        method_type="memory",
        model=SimpleNamespace(name="answer-model"),
        embedding=None,
        memorize_model=None,
        agent_params=build_config,
        build_config=build_config,
        snapshot_build_config=lambda: dict(build_config),
    )
    evaluator.dataset_config = SimpleNamespace(
        dataset_name="medmemorybench",
        language="en",
        data_root_dir="data",
        data_files={},
        evaluation_mode="merged",
        persona_ids=None,
        max_personas=1,
        max_sessions_per_persona=2,
        evaluation_interval=1,
        inject_noise=False,
        raw_config={},
    )
    evaluator._memory_build_logs = []
    evaluator._source_memory_build_logs = []
    evaluator._evaluation_units = []
    evaluator.dataset = None
    return evaluator


def test_build_metrics_report_groups_operations_and_feature_combination():
    tracker = get_usage_tracker()
    tracker.reset()
    tracker.set_phase("memorize")
    before = tracker.get_stats()
    with tracker.scope("amem.base.note_analysis"):
        _record_success(8, 2, 0.1)
    with tracker.scope("amem.typed_relations.inference"):
        _record_success(12, 3, 0.2)
    with tracker.scope("amem.temporal_state.transition"):
        sum(range(1000))
    usage = diff_usage_stats(tracker.get_stats(), before)

    evaluator = _metrics_evaluator()
    evaluator._memory_build_logs = [{
        "unit_id": 0,
        "context_id": 1,
        "session_count": 2,
        "total_time": 1.5,
        "memory_size": {
            "measurement": "serialized_memory_state",
            "bytes": 4096,
            "mib": 0.003906,
            "json_bytes": 3072,
            "embedding_bytes": 1024,
            "memory_entry_count": 4,
            "memory_chunk_count": 2,
        },
        "build_metrics": {"wall_time_seconds": 1.5, "usage": usage},
    }]

    report = evaluator._build_metrics_report()

    assert report["feature_configuration"]["combination_id"] == (
        "base_memory+typed_relations+temporal_state"
    )
    assert report["totals"]["input_tokens"] == 20
    assert report["totals"]["successful_calls"] == 2
    assert report["totals"]["wall_time_seconds"] == 1.5
    assert report["totals"]["memory_size_bytes"] == 4096
    assert report["memory_size"]["overall_bytes"] == 4096
    assert report["memory_size"]["complete"] is True
    assert report["units"][0]["memory_size"][
        "delta_from_previous_unit_bytes"
    ] == 4096
    assert report["by_feature"]["base"]["input_tokens"] == 8
    assert report["by_feature"]["typed_relations"]["input_tokens"] == 12
    assert "temporal_state" in report["by_feature"]
    assert "embedding" in report["by_feature"]
    assert evaluator._feature_for_operation("amem.original_evolution") == (
        "original_evolution"
    )


def test_memory_size_overall_uses_latest_unit_per_context():
    evaluator = _metrics_evaluator()
    evaluator._memory_build_logs = [
        {
            "unit_id": 0,
            "context_id": 1,
            "session_count": 1,
            "memory_size": {"bytes": 100, "mib": 0.000095},
            "build_metrics": {},
        },
        {
            "unit_id": 1,
            "context_id": 1,
            "session_count": 1,
            "memory_size": {"bytes": 150, "mib": 0.000143},
            "build_metrics": {},
        },
        {
            "unit_id": 2,
            "context_id": 2,
            "session_count": 1,
            "memory_size": {"bytes": 80, "mib": 0.000076},
            "build_metrics": {},
        },
    ]

    report = evaluator._build_metrics_report()

    assert report["memory_size"]["overall_bytes"] == 230
    assert report["memory_size"]["unit_snapshot_bytes"] == 330
    assert report["memory_size"]["by_context"] == {
        "1": {"unit_id": 1, "bytes": 150, "mib": 0.000143},
        "2": {"unit_id": 2, "bytes": 80, "mib": 0.000076},
    }
    assert [
        unit["memory_size"]["delta_from_previous_unit_bytes"]
        for unit in report["units"]
    ] == [100, 50, 80]


def test_memory_only_output_persists_full_build_metrics(tmp_path: Path):
    evaluator = _metrics_evaluator()
    metrics = evaluator._build_metrics_report([])
    report = EvaluationReport(
        method_name="amem_test",
        model_name="answer-model",
        dataset_name="medmemorybench",
        start_time="start",
        end_time="end",
        duration_seconds=1.0,
        summary={"total": 0},
        detailed_results=[],
        metadata={
            "memory_build_summary": {"total_units": 0},
            "build_metrics": metrics,
            "memory_size": metrics["memory_size"],
            "feature_configuration": metrics["feature_configuration"],
            "llm_usage": metrics["usage"],
            "api_failures": [],
        },
    )

    _, memory_path, _ = ResultCollector().save_reports(
        report,
        tmp_path,
        [],
        include_result=False,
        include_memory_build=True,
        include_query_answer=False,
    )
    payload = json.loads(memory_path.read_text(encoding="utf-8"))

    assert payload["build_metrics"] == metrics
    assert payload["memory_size"] == metrics["memory_size"]
    assert payload["feature_configuration"]["combination_id"] == (
        "base_memory+typed_relations+temporal_state"
    )
    assert "llm_usage" in payload


def test_query_result_output_carries_source_build_metrics(tmp_path: Path):
    evaluator = _metrics_evaluator()
    metrics = evaluator._build_metrics_report([])
    report = EvaluationReport(
        method_name="amem_test",
        model_name="answer-model",
        dataset_name="medmemorybench",
        start_time="start",
        end_time="end",
        duration_seconds=1.0,
        summary={"total": 0},
        detailed_results=[],
        metadata={
            "memory_build_summary": {"total_builds": 0},
            "build_metrics": metrics,
            "memory_size": metrics["memory_size"],
            "feature_configuration": metrics["feature_configuration"],
            "llm_usage": {},
            "evaluation_coverage": {},
            "api_failures": [],
        },
    )

    result_path, _, _ = ResultCollector().save_reports(
        report,
        tmp_path,
        [],
        include_result=True,
        include_memory_build=False,
        include_query_answer=False,
    )
    payload = json.loads(result_path.read_text(encoding="utf-8"))

    assert payload["build_metrics"] == metrics
    assert payload["memory_size"] == metrics["memory_size"]
    assert payload["feature_configuration"] == metrics["feature_configuration"]
