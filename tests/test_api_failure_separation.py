"""API failures must never be represented as incorrect benchmark answers."""

import json
from types import SimpleNamespace

from benchmarks.base import EvaluationUnit
from benchmarks.medmemorybench.dataset import MedQuery, MedSession
from benchmarks.medmemorybench.evaluator import MedMemoryBenchEvaluator
from src.result import EvaluationReport, ResultCollector
from utils.llm_client import EmptyGeminiResponseError, LLMAPIError, LLMRetryExhaustedError


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
