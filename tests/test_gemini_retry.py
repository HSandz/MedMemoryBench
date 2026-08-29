"""Regression tests for the shared managed-Gemini retry policy."""

from types import SimpleNamespace

import pytest

import methods.graph_rag as graph_rag
from metrics.llm_judge import LLMJudge
from utils import llm_client


class ResourceExhausted(Exception):
    """Match the Google API exception class name without a cloud dependency."""


class _HTTPError(Exception):
    def __init__(self, status_code):
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


@pytest.mark.parametrize(
    "error, expected_type",
    [
        (llm_client.EmptyGeminiResponseError("empty"), "empty_response"),
        (llm_client.InvalidLLMResponseError("invalid"), "invalid_response"),
        (_HTTPError(408), "http_408"),
        (_HTTPError(409), "http_409"),
        (_HTTPError(429), "http_429"),
        (_HTTPError(500), "http_500"),
        (_HTTPError(503), "http_503"),
        (_HTTPError(599), "http_599"),
        (type("RateLimitError", (Exception,), {})("limited"), "rate_limit_error"),
        (type("APIConnectionError", (Exception,), {})("offline"), "api_connection_error"),
        (type("APITimeoutError", (Exception,), {})("slow"), "api_timeout_error"),
        (type("OverloadedError", (Exception,), {})("busy"), "overloaded_error"),
        (type("DeadlineExceeded", (Exception,), {})("slow"), "deadline_exceeded"),
        (TimeoutError("slow"), "timeout_error"),
        (ConnectionResetError("reset"), "connection_reset_error"),
        (type("ConnectError", (Exception,), {})("offline"), "connect_error"),
        (type("RemoteProtocolError", (Exception,), {})("disconnected"), "remote_protocol_error"),
        (Exception("HTTP 408"), "http_408"),
        (Exception("HTTP 409"), "http_409"),
        (Exception("HTTP 598"), "http_598"),
        (Exception("resource exhausted"), "resource_exhausted"),
        (Exception("connection reset"), "connection_reset"),
        (Exception("internal server error"), "internal_server_error"),
        (Exception("service unavailable"), "service_unavailable"),
        (Exception("provider is overloaded"), "overloaded"),
        (Exception("provider capacity reached"), "capacity"),
    ],
)
def test_every_retryable_failure_gets_a_stable_pool(error, expected_type):
    retryable, _ = llm_client._is_retryable_exception(error)

    assert retryable is True
    assert llm_client._get_retry_failure_type(error) == expected_type


@pytest.mark.parametrize("status_code", [400, 401, 403, 404, 422])
def test_critical_http_failures_remain_non_retryable(status_code):
    retryable, _ = llm_client._is_retryable_exception(_HTTPError(status_code))

    assert retryable is False


def test_raw_gemini_calls_use_the_shared_retry_policy(monkeypatch):
    attempts = []
    monkeypatch.setattr(llm_client.time, "sleep", lambda delay: None)
    tracker = llm_client.get_usage_tracker()
    tracker.reset()
    tracker.set_phase("memorize")

    def operation():
        attempts.append(None)
        if len(attempts) < 3:
            raise ResourceExhausted("429 quota exceeded")
        return "complete"

    with tracker.scope("amem.typed_relations.inference"):
        assert llm_client.run_with_gemini_retry(operation) == "complete"
    assert len(attempts) == 3
    usage = tracker.get_stats()["operations"]["memorize"][
        "amem.typed_relations.inference"
    ]
    assert usage["successful_calls"] == 1
    assert usage["attempted_calls"] == 3
    assert usage["failed_attempts"] == 2
    assert usage["retry_count"] == 2


def test_empty_gemini_response_is_retried(monkeypatch):
    attempts = []
    monkeypatch.setattr(llm_client.time, "sleep", lambda delay: None)

    def operation():
        attempts.append(None)
        if len(attempts) == 1:
            raise llm_client.EmptyGeminiResponseError("empty response")
        return "complete"

    assert llm_client.run_with_gemini_retry(operation) == "complete"
    assert len(attempts) == 2


def test_all_provider_retry_aliases_use_the_general_budget():
    assert llm_client.DEFAULT_MAX_RETRIES == llm_client.LLM_MAX_RETRIES
    assert llm_client.GEMINI_MAX_RETRIES == llm_client.LLM_MAX_RETRIES
    assert llm_client.GOOGLE_AI_STUDIO_KEY_RETRIES == llm_client.LLM_MAX_RETRIES
    assert (
        llm_client.GOOGLE_VERTEX_SERVICE_ACCOUNT_RETRIES
        == llm_client.LLM_MAX_RETRIES
    )


def test_default_retry_backoff_is_exponential_and_capped(monkeypatch):
    delays = []
    attempts = 0
    monkeypatch.setattr(llm_client.time, "sleep", delays.append)

    @llm_client.with_retry(
        max_retries=11,
        retry_min_delay=1.0,
        retry_max_delay=1000.0,
    )
    def operation():
        nonlocal attempts
        attempts += 1
        if attempts <= 10:
            raise ResourceExhausted("429 quota exceeded")
        return "complete"

    assert operation() == "complete"
    assert delays == [
        1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 100.0, 100.0, 100.0,
    ]


def test_successful_retry_records_failed_attempt_and_wait_duration(monkeypatch):
    timestamps = iter((10.0, 12.0, 12.0, 17.0, 17.0))
    monkeypatch.setattr(llm_client.time, "perf_counter", lambda: next(timestamps))
    monkeypatch.setattr(llm_client.time, "sleep", lambda delay: None)
    tracker = llm_client.get_usage_tracker()
    tracker.reset()

    attempts = []

    @llm_client.with_retry(
        max_retries=2,
        retry_min_delay=5.0,
        retry_max_delay=5.0,
    )
    def operation():
        attempts.append(None)
        if len(attempts) == 1:
            raise ResourceExhausted("429 quota exceeded")
        return "complete"

    assert operation() == "complete"
    assert tracker.get_stats()["total"]["failure_duration_seconds"] == 7.0


def test_retry_budgets_and_backoff_are_separate_by_failure_type(monkeypatch):
    failures = [
        ResourceExhausted("429 quota exceeded"),
        ResourceExhausted("429 quota exceeded"),
        llm_client.EmptyGeminiResponseError("empty response"),
        ResourceExhausted("429 quota exceeded"),
    ]
    delays = []
    monkeypatch.setattr(llm_client.time, "sleep", delays.append)

    @llm_client.with_retry(
        max_retries=4,
        retry_min_delay=1.0,
        retry_max_delay=100.0,
        exponential_backoff=True,
    )
    def operation():
        if failures:
            raise failures.pop(0)
        return "complete"

    assert operation() == "complete"
    assert delays == [1.0, 2.0, 1.0, 4.0]


def test_retry_exhaustion_reports_per_type_and_total_counts(monkeypatch):
    failures = [
        ResourceExhausted("429 quota exceeded"),
        llm_client.EmptyGeminiResponseError("empty response"),
        ResourceExhausted("429 quota exceeded"),
        ResourceExhausted("429 quota exceeded"),
    ]
    monkeypatch.setattr(llm_client.time, "sleep", lambda delay: None)

    @llm_client.with_retry(max_retries=3)
    def operation():
        raise failures.pop(0)

    with pytest.raises(llm_client.LLMRetryExhaustedError) as exc_info:
        operation()

    error = exc_info.value
    assert error.attempts == 4
    assert error.failure_type == "http_429"
    assert error.retry_counts == {"http_429": 3, "empty_response": 1}


def test_invalid_structured_response_is_retryable():
    retryable, reason = llm_client._is_retryable_exception(
        llm_client.InvalidLLMResponseError("malformed JSON")
    )

    assert retryable is True
    assert reason == "Retryable LLM response failure: InvalidLLMResponseError"


def test_judge_retries_malformed_responses(monkeypatch):
    attempts = []
    monkeypatch.setattr(llm_client.time, "sleep", lambda delay: None)

    def chat(**kwargs):
        attempts.append(kwargs)
        content = "not JSON" if len(attempts) == 1 else '{"is_correct": true}'
        return SimpleNamespace(content=content)

    judge = LLMJudge(client=SimpleNamespace(chat=chat))
    judge._initialized = True
    judge._active_judge_provider = "openai"

    assert judge._call_llm("prompt") == {"is_correct": True}
    assert len(attempts) == 2


def test_non_retryable_google_client_error_is_not_retried():
    class ClientError(Exception):
        status_code = 400

    assert llm_client._is_retryable_exception(ClientError("bad request")) == (
        False,
        "Non-retryable HTTP status: 400",
    )


def test_graphrag_does_not_turn_transport_failures_into_answers():
    agent = object.__new__(graph_rag.GraphRAGAgent)
    agent._graph_built = True
    agent._graph_rag = SimpleNamespace(query=lambda question: (_ for _ in ()).throw(ResourceExhausted("429")))

    with pytest.raises(RuntimeError, match="no answer was recorded"):
        agent.query("question")
