"""API failures must never be represented as incorrect benchmark answers."""

import json
from types import SimpleNamespace

from benchmarks.base import EvaluationUnit
from benchmarks.medmemorybench.dataset import MedQuery, MedSession
from benchmarks.medmemorybench.evaluator import MedMemoryBenchEvaluator
from metrics import MetricResult
from src.result import EvaluationReport, ResultCollector
from utils.llm_client import (
    EmptyGeminiResponseError,
    LLMAPIError,
    LLMResponse,
    LLMRetryExhaustedError,
    get_usage_tracker,
)


def _retry_exhausted():
    return LLMRetryExhaustedError(
        "API call still failed after retries",
        last_exception=EmptyGeminiResponseError("empty response"),
        attempts=100,
        failure_type="empty_response",
        retry_counts={"http_429": 3, "empty_response": 100},
    )


def test_query_api_failure_is_recorded_without_metric_result():
    evaluator = MedMemoryBenchEvaluator.__new__(MedMemoryBenchEvaluator)
    evaluator.dry_run = False
    evaluator._api_failures = []
    evaluator._log = lambda *args, **kwargs: None
    evaluator.prompt_manager = SimpleNamespace(
        format_query=lambda question, query_type: question
    )
    evaluator.agent_manager = SimpleNamespace(
        send_message=lambda **kwargs: (_ for _ in ()).throw(_retry_exhausted())
    )
    query = MedQuery(
        query_id="q1",
        question="question",
        query_type="entity_exact_match",
        expected_answers=["answer"],
    )

    result = evaluator._evaluate_query(query, context_id=1)

    assert result is None
    assert evaluator._api_failures[0]["query_id"] == "q1"
    assert evaluator._api_failures[0]["attempts"] == 100
    assert evaluator._api_failures[0]["failure_type"] == "empty_response"
    assert evaluator._api_failures[0]["retry_counts"] == {
        "http_429": 3,
        "empty_response": 100,
    }


def test_memory_api_failure_skips_queries_and_checkpoint_injection():
    marked_sessions = []
    evaluator = MedMemoryBenchEvaluator.__new__(MedMemoryBenchEvaluator)
    evaluator.dry_run = False
    evaluator.batch_api = False
    evaluator._api_failures = []
    evaluator._memory_build_logs = []
    evaluator._deferred_judges = []
    evaluator._log = lambda *args, **kwargs: None
    evaluator.prompt_manager = SimpleNamespace(
        format_memorize=lambda context, timestamp: context
    )
    evaluator.agent_manager = SimpleNamespace(
        send_message=lambda **kwargs: (_ for _ in ()).throw(_retry_exhausted())
    )
    evaluator._checkpoint_manager = SimpleNamespace(
        mark_session_injected=lambda session_id: marked_sessions.append(session_id)
    )
    unit = EvaluationUnit(
        unit_id=3,
        context_id=7,
        sessions_to_inject=[MedSession(session_id="s1", content="memory")],
        queries_to_evaluate=[
            MedQuery(
                query_id="q1",
                question="question",
                query_type="entity_exact_match",
                expected_answers=["answer"],
            )
        ],
    )

    assert evaluator._evaluate_unit_with_checkpoint(unit) == []
    assert marked_sessions == []
    assert evaluator._api_failures[0]["affected_query_ids"] == ["q1"]


def test_api_failures_are_saved_outside_result_and_query_files(tmp_path):
    collector = ResultCollector()
    failure = {
        "phase": "query",
        "query_id": "q1",
        "error_type": "LLMRetryExhaustedError",
        "error_message": "failed",
    }
    coverage = {
        "expected_queries": 1,
        "scored_queries": 0,
        "omitted_queries": 1,
        "api_failed_queries": 1,
        "api_failure_events": 1,
        "coverage": 0.0,
        "complete": False,
    }
    report = EvaluationReport(
        method_name="method",
        model_name="model",
        dataset_name="medmemorybench",
        start_time="start",
        end_time="end",
        duration_seconds=1.0,
        summary={"total": 0},
        detailed_results=[],
        metadata={
            "api_failures": [failure],
            "evaluation_coverage": coverage,
        },
    )

    result_path, _, query_path = collector.save_reports(report, tmp_path, [])
    result_data = json.loads(result_path.read_text())
    query_data = json.loads(query_path.read_text())
    failure_data = json.loads(collector.last_api_failure_path.read_text())

    assert result_data["summary"]["total_queries"] == 0
    assert result_data["duration_seconds"] == 1.0
    assert result_data["true_duration_seconds"] == 1.0
    assert "failures" not in result_data
    assert query_data["queries"] == []
    assert failure_data["failures"] == [failure]


def test_failure_counts_are_saved_even_without_terminal_failure_events(tmp_path):
    report = EvaluationReport(
        method_name="method",
        model_name="model",
        dataset_name="medmemorybench",
        start_time="start",
        end_time="end",
        duration_seconds=1.0,
        summary={"total": 0},
        metadata={
            "api_failures": [],
            "llm_usage": {
                "memorize_phase": {"failed_attempts": 4, "retry_count": 3},
                "query_phase": {"failed_attempts": 2, "retry_count": 1},
                "total": {"failed_attempts": 6, "retry_count": 4},
            },
            "evaluation_coverage": {},
        },
    )

    ResultCollector().save_reports(report, tmp_path, [])

    failure_path = next(tmp_path.rglob("*_api_failures.json"))
    failure_data = json.loads(failure_path.read_text())
    assert failure_data["failure_count"] == 0
    assert failure_data["failure_counts"] == {
        "total_failed_attempts": 6,
        "total_retries": 4,
        "by_phase": {
            "memorize": {"failed_attempts": 4, "retry_count": 3},
            "query": {"failed_attempts": 2, "retry_count": 1},
            "judge": {"failed_attempts": 0, "retry_count": 0},
        },
    }


def test_query_answer_uses_compact_retrieval_references_and_full_summary(tmp_path):
    collector = ResultCollector()
    batch_result = MetricResult(
        query_id="batch-query",
        query_type="entity_exact_match",
        score=1.0,
        is_correct=True,
        model_output="batch answer",
        expected_answer="answer",
        retrieved_count=1,
        retrieved_memories=[{"memory_id": "m1", "content": "large payload"}],
        details={
            "artifact_references": {
                "memory_snapshot": {"build_id": "build-1"},
                "answer": {
                    "transport": "batch",
                    "manifest_path": "batch/answers.json",
                    "request_id": "request-batch-query",
                },
                "judge": {
                    "transport": "batch",
                    "manifest_path": "batch/judges.json",
                    "request_id": "judge-batch-query",
                },
            },
            "execution_usage": {
                "answer": {
                    "transport": "batch",
                    "input_tokens": 100,
                    "output_tokens": 10,
                },
                "judge": {"input_tokens": 20, "output_tokens": 5},
            },
        },
    )
    realtime_result = MetricResult(
        query_id="realtime-query",
        query_type="temporal_localization",
        score=0.5,
        is_correct=False,
        model_output="realtime answer",
        expected_answer="answer",
        retrieved_count=1,
        retrieved_memories=[{"memory_id": "m2", "content": "raw only once"}],
        details={"planner": {"rounds_configured": 1, "rounds_used": 1}},
    )
    collector.add_result(batch_result, context_id=1)
    collector.add_result(realtime_result, context_id=2)
    report = EvaluationReport(
        method_name="amem_test",
        model_name="model",
        dataset_name="medmemorybench",
        start_time="start",
        end_time="end",
        duration_seconds=1.0,
        summary={
            "total": 2,
            "correct": 1,
            "overall_accuracy": 0.5,
            "overall_avg_score": 0.75,
            "by_type": {"entity_exact_match": {"total": 1}},
            "by_metric": {"exact_match": {"total": 1}},
        },
        detailed_results=[batch_result.to_dict(), realtime_result.to_dict()],
        metadata={"evaluation_coverage": {"expected_queries": 2, "complete": True}},
    )

    _, _, query_path = collector.save_reports(
        report,
        tmp_path,
        [],
        include_memory_build=False,
    )

    query_data = json.loads(query_path.read_text())
    assert query_data["version"] == 3
    assert query_data["summary"]["overall_accuracy"] == 0.5
    assert query_data["summary"]["evaluation_coverage"]["complete"] is True
    batch_query, realtime_query = query_data["queries"]
    assert "retrieved_memories" not in batch_query
    assert batch_query["retrieved_memory_ids"] == ["m1"]
    assert batch_query["retrieval_reference"] == {
        "source": "answer_batch_manifest",
        "manifest_path": "batch/answers.json",
        "request_id": "request-batch-query",
        "prepared_query_key": "prepared_query",
    }
    assert batch_query["execution_usage"]["judge"]["output_tokens"] == 5
    assert batch_query["timing"] == {
        "kind": "per_request_latency_unavailable_for_batch",
        "query_time_seconds": 0.0,
        "answer_provider_duration_seconds": None,
    }
    assert realtime_query["retrieval_reference"] == {
        "source": "retrieval_records",
        "record_id": "realtime-query",
    }
    retrieval_path = query_path.with_name(
        query_path.name.replace("_query_answer.json", "_retrieval_records.json")
    )
    retrieval_data = json.loads(retrieval_path.read_text())
    assert retrieval_data["records"][0]["retrieved_memories"] == [
        {"memory_id": "m2", "content": "raw only once"}
    ]
    assert query_data["queries"][1]["planner"] == {
        "rounds_configured": 1,
        "rounds_used": 1,
    }
    assert retrieval_data["records"][0]["planner"] == {
        "rounds_configured": 1,
        "rounds_used": 1,
    }


def test_stage_usage_separates_retrieval_answer_and_judge_batch_metrics(tmp_path):
    tracker = get_usage_tracker()
    tracker.reset()
    for operation, tokens in (
        ("query.retrieval_preparation", (11, 2)),
        ("query.answer_batch", (101, 12)),
        ("judge.batch", (21, 3)),
    ):
        tracker.set_phase("judge" if operation.startswith("judge.") else "query")
        with tracker.scope(operation):
            tracker.record(LLMResponse(
                content="ok",
                input_tokens=tokens[0],
                output_tokens=tokens[1],
                model="test-model",
            ))

    manifest_path = tmp_path / "batch.json"
    manifest_path.write_text(json.dumps({
        "project": "openrouter",
        "provider": "openrouter",
        "model": "test-model",
        "jobs": {
            "query-final": {
                "state": "completed",
                "submitted_at": "2026-01-01T00:00:00+00:00",
                "completed_at": "2026-01-01T00:00:03+00:00",
                "requests": [{"request_id": "q1"}],
                "responses": {
                    "q1": {
                        "input_tokens": 101,
                        "output_tokens": 12,
                        "visible_output_tokens": 12,
                        "thinking_tokens": 0,
                        "status": "",
                    }
                },
            }
        },
    }), encoding="utf-8")
    evaluator = MedMemoryBenchEvaluator.__new__(MedMemoryBenchEvaluator)
    evaluator.output_dir = tmp_path
    evaluator.method_config = SimpleNamespace(
        model=SimpleNamespace(
            provider="openrouter",
            name="test-model",
            openrouter_service_tier="flex",
        )
    )
    evaluator._batch_client = SimpleNamespace(manifest_path=manifest_path)
    evaluator._judge_batch_client = None

    stage_usage = evaluator._stage_usage_report(tracker.get_stats())

    assert stage_usage["retrieval_preparation"]["usage"]["input_tokens"] == 11
    assert stage_usage["answer"]["usage"]["output_tokens"] == 12
    assert stage_usage["judge"]["usage"]["input_tokens"] == 21
    assert stage_usage["unattributed_query_usage"]["call_count"] == 0
    assert stage_usage["batch_stages"] == [{
        "stage": "query-final",
        "manifest_path": "batch.json",
        "transport": "batch",
        "provider": "openrouter",
        "model": "test-model",
        "state": "completed",
        "request_count": 1,
        "response_count": 1,
        "successful_response_count": 1,
        "failed_response_count": 0,
        "retry_count": 0,
        "submitted_at": "2026-01-01T00:00:00+00:00",
        "completed_at": "2026-01-01T00:00:03+00:00",
        "remote_elapsed_seconds": 3.0,
        "token_usage": {
            "input_tokens": 101,
            "output_tokens": 12,
            "visible_output_tokens": 12,
            "thinking_tokens": 0,
        },
        "fallback_reason": None,
    }]
    assert stage_usage["answer"]["cost"]["available"] is False


def test_stage_usage_attributes_event_state_planner_calls_once():
    tracker = get_usage_tracker()
    tracker.reset()
    for operation in ("event_state.plan_or_answer", "event_state.plan_or_answer", "event_state.final_answer"):
        tracker.set_phase("query")
        with tracker.scope(operation):
            tracker.record(LLMResponse(content="ok", input_tokens=1, output_tokens=1, model="test-model"))
    evaluator = MedMemoryBenchEvaluator.__new__(MedMemoryBenchEvaluator)
    evaluator.method_config = SimpleNamespace(model=SimpleNamespace(provider="openai", name="test-model"))
    evaluator._batch_client = None
    evaluator._judge_batch_client = None
    stage_usage = evaluator._stage_usage_report(tracker.get_stats())
    assert stage_usage["planner_controller"]["usage"]["call_count"] == 2
    assert stage_usage["answer"]["usage"]["call_count"] == 1
    assert stage_usage["unattributed_query_usage"]["call_count"] == 0


def test_true_duration_subtracts_measured_api_failure_time(tmp_path):
    evaluator = MedMemoryBenchEvaluator.__new__(MedMemoryBenchEvaluator)
    evaluator._api_failures = []
    evaluator._record_api_failure(
        "query",
        LLMAPIError("failed"),
        query_id="q1",
        failure_duration_seconds=2.5,
    )

    report = EvaluationReport(
        method_name="method",
        model_name="model",
        dataset_name="medmemorybench",
        start_time="start",
        end_time="end",
        duration_seconds=10.0,
        summary={"total": 0},
        detailed_results=[],
        metadata={"api_failures": evaluator._api_failures},
    )
    result_path, _, _ = ResultCollector().save_reports(report, tmp_path, [])
    result_data = json.loads(result_path.read_text())

    assert evaluator._api_failures[0]["duration_seconds"] == 2.5
    assert report.to_dict()["true_duration_seconds"] == 7.5
    assert result_data["duration_seconds"] == 10.0
    assert result_data["true_duration_seconds"] == 7.5

def test_true_duration_subtracts_recovered_retry_time():
    report = EvaluationReport(
        method_name="method",
        model_name="model",
        dataset_name="medmemorybench",
        start_time="start",
        end_time="end",
        duration_seconds=10.0,
        summary={"total": 1},
        metadata={"api_failure_duration_seconds": 3.0},
    )

    assert report.to_dict()["true_duration_seconds"] == 7.0

def test_report_failure_duration_combines_recovered_and_terminal_failures():
    evaluator = MedMemoryBenchEvaluator.__new__(MedMemoryBenchEvaluator)
    evaluator._api_failure_duration_seconds = 2.5
    llm_usage = {"total": {"failure_duration_seconds": 4.5}}

    assert evaluator._retry_failure_duration(llm_usage) == 7.0


def test_api_call_timer_attaches_failed_operation_duration(monkeypatch):
    evaluator_module = __import__(
        "benchmarks.medmemorybench.evaluator",
        fromlist=["MedMemoryBenchEvaluator"],
    )
    timestamps = iter((100.0, 103.25))
    monkeypatch.setattr(evaluator_module.time, "perf_counter", lambda: next(timestamps))

    error = LLMAPIError("failed")
    try:
        evaluator_module.MedMemoryBenchEvaluator._run_api_call(
            lambda: (_ for _ in ()).throw(error)
        )
    except LLMAPIError as caught:
        assert caught is error
        assert caught._medmemorybench_api_duration_seconds == 3.25
    else:
        raise AssertionError("Expected the API operation to fail")
