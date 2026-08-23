"""LLM client module - unified interface for multiple providers."""

import json
import os
import time
import logging
import re
import threading
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Dict, Any, Iterator, List, Optional, Sequence, Tuple, Union
from dataclasses import dataclass
from functools import wraps

from utils.tokenizer import get_tokenizer, TokenizerProtocol

@dataclass
class TokenUsage:
    """Token usage statistics."""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    call_count: int = 0
    attempt_count: int = 0
    failure_count: int = 0
    retry_count: int = 0
    operation_count: int = 0
    total_latency: float = 0.0
    wall_time: float = 0.0
    failure_duration_seconds: float = 0.0

    def add(self, input_tokens: int, output_tokens: int, latency: float) -> None:
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.total_tokens += input_tokens + output_tokens
        self.call_count += 1
        # Batch and externally supplied responses may not expose an attempt hook.
        if self.attempt_count < self.call_count + self.failure_count:
            self.attempt_count += 1
        self.total_latency += latency

    def add_attempt(self) -> None:
        self.attempt_count += 1

    def add_failure(self) -> None:
        self.failure_count += 1

    def add_retry(self) -> None:
        self.retry_count += 1

    def add_operation(self) -> None:
        self.operation_count += 1

    def add_wall_time(self, duration: float) -> None:
        self.wall_time += max(float(duration), 0.0)

    def add_failure_duration(self, duration: float) -> None:
        self.failure_duration_seconds += max(float(duration), 0.0)

    def merge(self, other: "TokenUsage") -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.total_tokens += other.total_tokens
        self.call_count += other.call_count
        self.attempt_count += other.attempt_count
        self.failure_count += other.failure_count
        self.retry_count += other.retry_count
        self.operation_count += other.operation_count
        self.total_latency += other.total_latency
        self.wall_time += other.wall_time
        self.failure_duration_seconds += other.failure_duration_seconds

    def to_dict(self) -> Dict[str, Any]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "call_count": self.call_count,
            "successful_calls": self.call_count,
            "attempted_calls": self.attempt_count,
            "failed_attempts": self.failure_count,
            "retry_count": self.retry_count,
            "operation_count": self.operation_count,
            "total_latency": round(self.total_latency, 3),
            "avg_latency": round(self.total_latency / self.call_count, 3) if self.call_count > 0 else 0,
            "wall_time": round(self.wall_time, 6),
            "failure_duration_seconds": round(self.failure_duration_seconds, 6),
        }

    @classmethod
    def from_dict(cls, value: Optional[Dict[str, Any]]) -> "TokenUsage":
        value = value or {}
        input_tokens = int(value.get("input_tokens", 0) or 0)
        output_tokens = int(value.get("output_tokens", 0) or 0)
        call_count = int(
            value.get("call_count", value.get("successful_calls", 0)) or 0
        )
        failure_count = int(value.get("failed_attempts", 0) or 0)
        return cls(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=int(
                value.get("total_tokens", input_tokens + output_tokens) or 0
            ),
            call_count=call_count,
            attempt_count=int(
                value.get("attempted_calls", call_count + failure_count) or 0
            ),
            failure_count=failure_count,
            retry_count=int(value.get("retry_count", 0) or 0),
            operation_count=int(value.get("operation_count", 0) or 0),
            total_latency=float(value.get("total_latency", 0.0) or 0.0),
            wall_time=float(value.get("wall_time", 0.0) or 0.0),
            failure_duration_seconds=float(
                value.get("failure_duration_seconds", 0.0) or 0.0
            ),
        )

    def subtract(self, other: "TokenUsage") -> "TokenUsage":
        return TokenUsage(
            input_tokens=max(self.input_tokens - other.input_tokens, 0),
            output_tokens=max(self.output_tokens - other.output_tokens, 0),
            total_tokens=max(self.total_tokens - other.total_tokens, 0),
            call_count=max(self.call_count - other.call_count, 0),
            attempt_count=max(self.attempt_count - other.attempt_count, 0),
            failure_count=max(self.failure_count - other.failure_count, 0),
            retry_count=max(self.retry_count - other.retry_count, 0),
            operation_count=max(self.operation_count - other.operation_count, 0),
            total_latency=max(self.total_latency - other.total_latency, 0.0),
            wall_time=max(self.wall_time - other.wall_time, 0.0),
            failure_duration_seconds=max(
                self.failure_duration_seconds - other.failure_duration_seconds,
                0.0,
            ),
        )


class LLMUsageTracker:
    """Global LLM usage tracker (singleton)."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init()
        return cls._instance

    def _init(self):
        self._memorize_usage = TokenUsage()
        self._query_usage = TokenUsage()
        self._operation_usage: Dict[str, Dict[str, TokenUsage]] = {}
        self._current_phase: ContextVar[str] = ContextVar(
            "llm_usage_phase", default="unknown"
        )
        self._current_operation: ContextVar[str] = ContextVar(
            "llm_usage_operation", default="unscoped"
        )
        self._current_failure_duration: ContextVar[float] = ContextVar(
            "llm_failure_duration", default=0.0
        )
        self._lock = threading.RLock()

    def set_phase(self, phase: str) -> None:
        self._current_phase.set(str(phase or "unknown"))

    def _phase_bucket(self) -> TokenUsage:
        return (
            self._memorize_usage
            if self._current_phase.get() == "memorize"
            else self._query_usage
        )

    def _operation_bucket(self) -> TokenUsage:
        phase = self._current_phase.get()
        operation = self._current_operation.get()
        return self._operation_usage.setdefault(phase, {}).setdefault(
            operation, TokenUsage()
        )

    def record(self, response: "LLMResponse") -> None:
        with self._lock:
            self._phase_bucket().add(
                response.input_tokens, response.output_tokens, response.latency
            )
            self._operation_bucket().add(
                response.input_tokens, response.output_tokens, response.latency
            )

    def record_attempt(self) -> None:
        with self._lock:
            self._phase_bucket().add_attempt()
            self._operation_bucket().add_attempt()

    def record_failure(self) -> None:
        with self._lock:
            self._phase_bucket().add_failure()
            self._operation_bucket().add_failure()

    def record_failure_duration(self, duration: float) -> None:
        duration = max(float(duration), 0.0)
        with self._lock:
            self._phase_bucket().add_failure_duration(duration)
            self._operation_bucket().add_failure_duration(duration)
        self._current_failure_duration.set(
            self._current_failure_duration.get() + duration
        )

    def current_failure_duration(self) -> float:
        return self._current_failure_duration.get()

    def record_retry(self) -> None:
        with self._lock:
            self._phase_bucket().add_retry()
            self._operation_bucket().add_retry()

    def current_successful_calls(self) -> int:
        """Return successful calls in the active phase/operation bucket."""
        with self._lock:
            return self._operation_bucket().call_count

    def record_success_without_usage(self) -> None:
        """Count a successful call when a backend exposes no token metadata."""
        with self._lock:
            self._phase_bucket().add(0, 0, 0.0)
            self._operation_bucket().add(0, 0, 0.0)

    @contextmanager
    def scope(self, operation: str) -> Iterator[None]:
        """Attribute LLM usage and local wall time to one logical operation."""
        token = self._current_operation.set(str(operation or "unscoped"))
        with self._lock:
            self._phase_bucket().add_operation()
            self._operation_bucket().add_operation()
        started_at = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - started_at
            with self._lock:
                self._operation_bucket().add_wall_time(duration)
            self._current_operation.reset(token)

    def reset(self) -> None:
        with self._lock:
            self._memorize_usage = TokenUsage()
            self._query_usage = TokenUsage()
            self._operation_usage = {}
        self._current_phase.set("unknown")
        self._current_operation.set("unscoped")
        self._current_failure_duration.set(0.0)

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            total = TokenUsage()
            total.merge(self._memorize_usage)
            total.merge(self._query_usage)
            return {
                "memorize_phase": self._memorize_usage.to_dict(),
                "query_phase": self._query_usage.to_dict(),
                "total": total.to_dict(),
                "operations": {
                    phase: {
                        operation: usage.to_dict()
                        for operation, usage in sorted(operations.items())
                    }
                    for phase, operations in sorted(self._operation_usage.items())
                },
            }


_usage_tracker: Optional[LLMUsageTracker] = None


def get_usage_tracker() -> LLMUsageTracker:
    global _usage_tracker
    if _usage_tracker is None:
        _usage_tracker = LLMUsageTracker()
    return _usage_tracker


def merge_usage_stats(*stats_values: Dict[str, Any]) -> Dict[str, Any]:
    """Merge tracker snapshots while recomputing all derived fields."""
    phase_totals = {
        "memorize_phase": TokenUsage(),
        "query_phase": TokenUsage(),
    }
    operation_totals: Dict[str, Dict[str, TokenUsage]] = {}
    for stats in stats_values:
        if not isinstance(stats, dict):
            continue
        for phase_name in phase_totals:
            phase_totals[phase_name].merge(TokenUsage.from_dict(stats.get(phase_name)))
        for phase, operations in (stats.get("operations") or {}).items():
            if not isinstance(operations, dict):
                continue
            phase_operations = operation_totals.setdefault(str(phase), {})
            for operation, usage in operations.items():
                phase_operations.setdefault(str(operation), TokenUsage()).merge(
                    TokenUsage.from_dict(usage)
                )

    total = TokenUsage()
    total.merge(phase_totals["memorize_phase"])
    total.merge(phase_totals["query_phase"])
    return {
        "memorize_phase": phase_totals["memorize_phase"].to_dict(),
        "query_phase": phase_totals["query_phase"].to_dict(),
        "total": total.to_dict(),
        "operations": {
            phase: {
                operation: usage.to_dict()
                for operation, usage in sorted(operations.items())
            }
            for phase, operations in sorted(operation_totals.items())
        },
    }


def diff_usage_stats(
    after: Dict[str, Any],
    before: Dict[str, Any],
) -> Dict[str, Any]:
    """Return the non-negative usage accumulated between two snapshots."""
    phase_deltas = {}
    for phase_name in ("memorize_phase", "query_phase"):
        phase_deltas[phase_name] = TokenUsage.from_dict(after.get(phase_name)).subtract(
            TokenUsage.from_dict(before.get(phase_name))
        )

    operation_deltas: Dict[str, Dict[str, TokenUsage]] = {}
    after_operations = after.get("operations") or {}
    before_operations = before.get("operations") or {}
    for phase in set(after_operations) | set(before_operations):
        after_phase = after_operations.get(phase) or {}
        before_phase = before_operations.get(phase) or {}
        for operation in set(after_phase) | set(before_phase):
            delta = TokenUsage.from_dict(after_phase.get(operation)).subtract(
                TokenUsage.from_dict(before_phase.get(operation))
            )
            if any((
                delta.input_tokens,
                delta.output_tokens,
                delta.call_count,
                delta.attempt_count,
                delta.failure_count,
                delta.retry_count,
                delta.operation_count,
                delta.total_latency,
                delta.wall_time,
                delta.failure_duration_seconds,
            )):
                operation_deltas.setdefault(str(phase), {})[str(operation)] = delta

    total = TokenUsage()
    total.merge(phase_deltas["memorize_phase"])
    total.merge(phase_deltas["query_phase"])
    return {
        "memorize_phase": phase_deltas["memorize_phase"].to_dict(),
        "query_phase": phase_deltas["query_phase"].to_dict(),
        "total": total.to_dict(),
        "operations": {
            phase: {
                operation: usage.to_dict()
                for operation, usage in sorted(operations.items())
            }
            for phase, operations in sorted(operation_deltas.items())
        },
    }

# Default retry config (can be overridden via environment variables)
DEFAULT_MAX_RETRIES = int(os.environ.get("LLM_MAX_RETRIES", "10"))
DEFAULT_RETRY_MIN_DELAY = float(os.environ.get("LLM_RETRY_MIN_DELAY", "10.0"))
DEFAULT_RETRY_MAX_DELAY = float(os.environ.get("LLM_RETRY_MAX_DELAY", "20.0"))
GEMINI_MAX_RETRIES = int(os.environ.get("GEMINI_MAX_RETRIES", "100"))
GEMINI_RETRY_INITIAL_DELAY = float(os.environ.get("GEMINI_RETRY_INITIAL_DELAY", "1.0"))
GEMINI_RETRY_MAX_DELAY = float(os.environ.get("GEMINI_RETRY_MAX_DELAY", "100.0"))
GOOGLE_AI_STUDIO_KEY_RETRIES = int(os.environ.get("GOOGLE_AI_STUDIO_KEY_RETRIES", "5"))
GOOGLE_VERTEX_SERVICE_ACCOUNT_RETRIES = int(
    os.environ.get("GOOGLE_VERTEX_SERVICE_ACCOUNT_RETRIES", "5")
)

VERTEX_GEMINI_PROVIDER = "vertex"
GOOGLE_AI_STUDIO_PROVIDER = "ai_studio"
HYBRID_GEMINI_PROVIDER = "gemini"
GEMINI_PROVIDERS = frozenset({
    VERTEX_GEMINI_PROVIDER,
    GOOGLE_AI_STUDIO_PROVIDER,
    HYBRID_GEMINI_PROVIDER,
})

logger = logging.getLogger(__name__)

class LLMAPIError(Exception):
    """LLM API error base class."""
    pass


class RetryableLLMAPIError(LLMAPIError):
    """A provider response failure that is safe to retry."""


class LLMRetryExhaustedError(LLMAPIError):
    """Retry exhausted error."""
    def __init__(
        self,
        message: str,
        last_exception: Exception,
        attempts: int,
        failure_type: Optional[str] = None,
        retry_counts: Optional[Dict[str, int]] = None,
    ):
        super().__init__(message)
        self.last_exception = last_exception
        self.attempts = attempts
        self.failure_type = failure_type
        self.retry_counts = dict(retry_counts or {})


class EmptyLLMResponseError(RetryableLLMAPIError):
    """A completed provider request without generated text."""


class EmptyGeminiResponseError(EmptyLLMResponseError):
    """A completed Gemini request without generated text."""


class InvalidLLMResponseError(RetryableLLMAPIError):
    """A completed provider request with a transiently unusable payload."""


_RETRYABLE_EXCEPTION_TYPES = frozenset({
    "RateLimitError",
    "APIConnectionError",
    "APITimeoutError",
    "InternalServerError",
    "ServiceUnavailableError",
    "OverloadedError",
    "ResourceExhausted",
    "ServiceUnavailable",
    "DeadlineExceeded",
    "ServerError",
    "TooManyRequests",
    "Aborted",
    "BadGateway",
    "GatewayTimeout",
    "TimeoutError",
    "Timeout",
    "TimeoutException",
    "ConnectTimeout",
    "ReadTimeout",
    "WriteTimeout",
    "PoolTimeout",
    "ConnectionError",
    "ConnectionResetError",
    "ConnectionAbortedError",
    "ConnectionRefusedError",
    "BrokenPipeError",
    "ConnectError",
    "ReadError",
    "WriteError",
    "CloseError",
    "NetworkError",
    "RemoteProtocolError",
})

_RETRYABLE_MESSAGE_FAILURE_TYPES = (
    (("rate limit", "rate_limit", "too many requests"), "rate_limit"),
    (("resource exhausted",), "resource_exhausted"),
    (("connection error",), "connection_error"),
    (("connection reset",), "connection_reset"),
    (("timeout", "timed out"), "timeout"),
    (("service unavailable",), "service_unavailable"),
    (("internal server error",), "internal_server_error"),
    (("overloaded",), "overloaded"),
    (("capacity",), "capacity"),
)

_AI_STUDIO_KEY_FAILURE_TYPES = (
    (("invalid api key", "invalid_api_key", "api key", "api_key"), "api_key"),
    (("permission denied", "permission_denied"), "permission_denied"),
    (("quota",), "quota"),
)

_VERTEX_CREDENTIAL_FAILURE_TYPES = (
    (("invalid grant", "invalid_grant"), "invalid_grant"),
    (("invalid jwt", "jwt signature", "signature is invalid"), "invalid_jwt"),
    (("service account", "service_account"), "service_account_auth"),
    (("credential", "credentials"), "credentials"),
    (("unauthenticated",), "unauthenticated"),
    (("permission denied", "permission_denied"), "permission_denied"),
)


def _get_status_code(exc: Exception) -> Optional[int]:
    """Extract an integer HTTP status from common provider exceptions."""
    status_code = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    try:
        return int(status_code)
    except (TypeError, ValueError):
        return None


def _get_message_failure_type(
    message: str,
    failure_types: Sequence[Tuple[Sequence[str], str]],
) -> Optional[str]:
    """Return the stable pool name for a recognized provider message."""
    for keywords, failure_type in failure_types:
        if any(keyword in message for keyword in keywords):
            return failure_type
    return None


def _normalize_exception_type(exc_type: str) -> str:
    """Convert provider exception names to stable snake_case pool names."""
    acronym_split = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", exc_type)
    word_split = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", acronym_split)
    return word_split.lower() or "retryable_error"


def _is_retryable_exception(exc: Exception) -> Tuple[bool, str]:
    """Check if exception is retryable."""
    exc_type = type(exc).__name__
    exc_message = str(exc).lower()

    if isinstance(exc, RetryableLLMAPIError):
        return True, f"Retryable LLM response failure: {exc_type}"

    status_code = _get_status_code(exc)
    if status_code is not None:
        if status_code in {408, 409, 429} or 500 <= status_code < 600:
            return True, f"Retryable HTTP status: {status_code}"
        if 400 <= status_code < 500:
            return False, f"Non-retryable HTTP status: {status_code}"

    if exc_type in _RETRYABLE_EXCEPTION_TYPES:
        return True, f"Retryable exception type: {exc_type}"

    # Check by exception message (compatibility)
    message_failure_type = _get_message_failure_type(
        exc_message,
        _RETRYABLE_MESSAGE_FAILURE_TYPES,
    )
    if message_failure_type:
        return True, f"Contains retryable failure type: {message_failure_type}"

    for status_code in (408, 409, 429, *range(500, 600)):
        if re.search(rf"(?<!\d){status_code}(?!\d)", exc_message):
            return True, f"Contains retryable HTTP status: {status_code}"

    # Non-retryable exceptions
    non_retryable_keywords = [
        "authentication",
        "invalid api key",
        "invalid_api_key",
        "unauthorized",
        "401",
        "invalid request",
        "bad request",
        "400",
        "context_length_exceeded",
        "maximum context length",
    ]

    for keyword in non_retryable_keywords:
        if keyword in exc_message:
            return False, f"Non-retryable exception: {keyword}"

    if isinstance(exc, LLMAPIError):
        return False, f"Non-retryable LLM API failure: {exc_type}"

    # Default: don't retry unknown exceptions
    return False, f"Unknown exception type: {exc_type}"


def _get_retry_failure_type(exc: Exception) -> str:
    """Normalize retryable failures into independent retry-budget pools."""
    if isinstance(exc, EmptyLLMResponseError):
        return "empty_response"
    if isinstance(exc, InvalidLLMResponseError):
        return "invalid_response"

    status_code = _get_status_code(exc)
    if status_code is not None:
        return f"http_{status_code}"

    exc_type = type(exc).__name__
    message = str(exc).lower()
    for code in (408, 409, 429, *range(500, 600)):
        if re.search(rf"(?<!\d){code}(?!\d)", message):
            return f"http_{code}"

    if exc_type in _RETRYABLE_EXCEPTION_TYPES:
        return _normalize_exception_type(exc_type)

    message_failure_type = _get_message_failure_type(
        message,
        (
            *_RETRYABLE_MESSAGE_FAILURE_TYPES,
            *_AI_STUDIO_KEY_FAILURE_TYPES,
            *_VERTEX_CREDENTIAL_FAILURE_TYPES,
        ),
    )
    if message_failure_type:
        return message_failure_type

    if exc_type != "Exception":
        return _normalize_exception_type(exc_type)

    return "retryable_error"


def _get_retry_delay(min_delay: float, max_delay: float) -> float:
    """Get retry delay with jitter (random delay between min and max)."""
    import random
    return random.uniform(min_delay, max_delay)

def _record_untracked_failure_duration(
    tracker: LLMUsageTracker,
    failure_duration_before: float,
    started_at: float,
) -> None:
    """Record elapsed failure time unless an inner client already measured it."""
    if tracker.current_failure_duration() == failure_duration_before:
        tracker.record_failure_duration(time.perf_counter() - started_at)

def _sleep_after_failure(delay: float) -> None:
    """Sleep before a retry and account for the actual failure-induced wait."""
    started_at = time.perf_counter()
    try:
        time.sleep(delay)
    finally:
        get_usage_tracker().record_failure_duration(time.perf_counter() - started_at)


def _log_retry_attempt(
    attempt: int,
    max_retries: int,
    delay: float,
    exc: Exception,
    reason: str,
    failure_type: Optional[str] = None,
) -> None:
    """Log retry attempt."""
    attempt_label = f"{failure_type} attempt" if failure_type else "attempt"
    logger.warning(
        f"API call failed ({attempt_label} {attempt}/{max_retries}): "
        f"{type(exc).__name__}: {exc}\n"
        f"Reason: {reason}\n"
        f"Will retry in {delay:.1f} seconds..."
    )


def with_retry(
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_min_delay: float = DEFAULT_RETRY_MIN_DELAY,
    retry_max_delay: float = DEFAULT_RETRY_MAX_DELAY,
    exponential_backoff: bool = False,
):
    """API call retry decorator with jitter strategy.

    Args:
        max_retries: Maximum number of retry attempts (default: 10)
        retry_min_delay: Minimum delay between retries in seconds (default: 10.0)
        retry_max_delay: Maximum delay between retries in seconds (default: 20.0)
        exponential_backoff: Double the delay after each failure, capped at max delay.

    Each normalized failure type has its own max_retries budget and backoff
    sequence. The actual delay is randomly chosen between retry_min_delay and
    retry_max_delay (jitter strategy) unless exponential_backoff is enabled.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            failure_counts: Dict[str, int] = {}
            total_attempts = 0

            while True:
                tracker = get_usage_tracker()
                successful_calls_before = tracker.current_successful_calls()
                failure_duration_before = tracker.current_failure_duration()
                attempt_started_at = time.perf_counter()
                tracker.record_attempt()
                try:
                    result = func(*args, **kwargs)
                    if tracker.current_successful_calls() == successful_calls_before:
                        tracker.record_success_without_usage()
                    return result
                except Exception as exc:
                    tracker.record_failure()
                    _record_untracked_failure_duration(
                        tracker,
                        failure_duration_before,
                        attempt_started_at,
                    )
                    total_attempts += 1
                    is_retryable, reason = _is_retryable_exception(exc)

                    if not is_retryable:
                        # Non-retryable exception, raise immediately
                        logger.error(f"API call failed (non-retryable): {type(exc).__name__}: {exc}")
                        raise

                    failure_type = _get_retry_failure_type(exc)
                    failure_attempt = failure_counts.get(failure_type, 0) + 1
                    failure_counts[failure_type] = failure_attempt
                    if failure_attempt >= max_retries:
                        logger.error(
                            f"API call retry exhausted for {failure_type} "
                            f"({failure_attempt}/{max_retries}, "
                            f"{total_attempts} total attempts): "
                            f"{type(exc).__name__}: {exc}"
                        )
                        raise LLMRetryExhaustedError(
                            f"API call still failed after {max_retries} "
                            f"{failure_type} failures",
                            last_exception=exc,
                            attempts=total_attempts,
                            failure_type=failure_type,
                            retry_counts=failure_counts,
                        ) from exc

                    if exponential_backoff:
                        delay = min(
                            retry_min_delay * (2 ** (failure_attempt - 1)),
                            retry_max_delay,
                        )
                    else:
                        delay = _get_retry_delay(retry_min_delay, retry_max_delay)
                    _log_retry_attempt(
                        failure_attempt,
                        max_retries,
                        delay,
                        exc,
                        reason,
                        failure_type=failure_type,
                    )
                    tracker.record_retry()
                    _sleep_after_failure(delay)

        return wrapper
    return decorator


@with_retry(
    max_retries=GEMINI_MAX_RETRIES,
    retry_min_delay=GEMINI_RETRY_INITIAL_DELAY,
    retry_max_delay=GEMINI_RETRY_MAX_DELAY,
    exponential_backoff=True,
)
def run_with_gemini_retry(operation, *args, **kwargs):
    """Run a raw Gemini SDK call with the shared retry policy."""
    return operation(*args, **kwargs)


def is_vertex_gemini_provider(provider: Optional[str]) -> bool:
    """Return whether a provider uses service-account-backed Vertex Gemini."""
    return bool(provider and provider.lower() == VERTEX_GEMINI_PROVIDER)


def is_google_ai_studio_provider(provider: Optional[str]) -> bool:
    """Return whether a provider uses the API-key-backed Gemini Developer API."""
    return bool(provider and provider.lower() == GOOGLE_AI_STUDIO_PROVIDER)


def is_hybrid_gemini_provider(provider: Optional[str]) -> bool:
    """Return whether a provider rotates between Vertex and AI Studio."""
    return bool(provider and provider.lower() == HYBRID_GEMINI_PROVIDER)


def is_vertex_batch_provider(provider: Optional[str]) -> bool:
    """Return whether batch requests can be pinned to Vertex AI."""
    return is_vertex_gemini_provider(provider) or is_hybrid_gemini_provider(provider)


def is_gemini_provider(provider: Optional[str]) -> bool:
    """Return whether a provider is any supported Gemini transport."""
    return bool(provider and provider.lower() in GEMINI_PROVIDERS)


@dataclass
class LLMResponse:
    """LLM response result."""
    content: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency: float = 0.0
    model: str = ""
    raw_response: Any = None


class BaseLLMClient:
    """LLM client base class."""

    def __init__(
        self,
        model: str,
        temperature: float = 1.0,
        max_tokens: int = 2000,
        **kwargs
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Tokenizer (prefer local model, fallback to tiktoken)
        self._tokenizer: TokenizerProtocol = get_tokenizer(
            model_name=model,
            prefer_local=True,
        )

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """Send chat request."""
        raise NotImplementedError

    def count_tokens(self, text: str) -> int:
        """Count tokens."""
        return len(self._tokenizer.encode(text))

    def _use_max_completion_tokens(self) -> bool:
        """Check if should use max_completion_tokens param (new models)."""
        new_patterns = ["gpt-5", "o1-", "o3-"]
        return any(p in self.model.lower() for p in new_patterns)


class OpenAIClient(BaseLLMClient):
    """OpenAI client."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 1.0,
        max_tokens: int = 2000,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs
    ):
        super().__init__(model, temperature, max_tokens, **kwargs)

        from openai import OpenAI
        import httpx
        import socket

        # Extended timeout for long-text generation (e.g. gist extraction)
        timeout = httpx.Timeout(
            timeout=180.0,
            connect=30.0,
            read=180.0,
            write=30.0,
        )

        # Minimal connection pooling - proxy connections are unreliable for keepalive
        limits = httpx.Limits(
            max_keepalive_connections=1,  # Minimal keepalive
            max_connections=5,
            keepalive_expiry=10.0,  # Very short expiry - proxy connections die fast
        )

        # Create transport with socket-level options for better dead connection detection
        transport = httpx.HTTPTransport(
            retries=0,  # We handle retries ourselves
            limits=limits,
            socket_options=[
                (socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1),  # Enable TCP keepalive
                (socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 10),  # Keepalive interval: 10s
                (socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 3),  # Max keepalive probes
            ]
        )

        http_client = httpx.Client(timeout=timeout, transport=transport)

        self.client = OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            base_url=base_url or os.environ.get("OPENAI_BASE_URL"),
            http_client=http_client,
        )

    @with_retry()
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        start_time = time.time()

        params = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.temperature,
        }

        # Select token param based on model
        token_limit = max_tokens if max_tokens is not None else self.max_tokens
        if self._use_max_completion_tokens():
            params["max_completion_tokens"] = token_limit
        else:
            params["max_tokens"] = token_limit

        params.update(kwargs)

        response = self.client.chat.completions.create(**params)
        latency = time.time() - start_time

        content = response.choices[0].message.content or ""
        finish_reason = response.choices[0].finish_reason
        refusal = getattr(response.choices[0].message, 'refusal', None)

        if not content.strip():
            raise EmptyLLMResponseError(
                "OpenAI returned an empty response "
                f"(finish_reason={finish_reason}, refusal={refusal}, model={self.model})."
            )

        llm_response = LLMResponse(
            content=content,
            input_tokens=response.usage.prompt_tokens if response.usage else 0,
            output_tokens=response.usage.completion_tokens if response.usage else 0,
            latency=latency,
            model=self.model,
            raw_response=response,
        )
        get_usage_tracker().record(llm_response)
        return llm_response

class ModalClient(OpenAIClient):
    """OpenAI-compatible client for a Modal-hosted vLLM server."""

    def __init__(
        self,
        model: str = "Qwen3-30B-A3B-Instruct-2507-AWQ",
        temperature: float = 1.0,
        max_tokens: int = 2000,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key
            or os.environ.get("MODAL_API_KEY")
            or os.environ.get("MODAL_PROXY_TOKEN"),
            base_url=base_url or os.environ.get("MODAL_BASE_URL"),
            **kwargs,
        )


class AzureOpenAIClient(BaseLLMClient):
    """Azure OpenAI client."""

    def __init__(
        self,
        model: str = "gpt-4o",
        temperature: float = 1.0,
        max_tokens: int = 2000,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        api_version: str = "2024-02-01",
        deployment: Optional[str] = None,
        **kwargs
    ):
        super().__init__(model, temperature, max_tokens, **kwargs)

        from openai import AzureOpenAI

        self.deployment = deployment or model
        self.client = AzureOpenAI(
            api_key=api_key or os.environ.get("AZURE_OPENAI_API_KEY"),
            azure_endpoint=endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT"),
            api_version=api_version,
        )

    @with_retry()
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        start_time = time.time()

        params = {
            "model": self.deployment,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.temperature,
        }

        token_limit = max_tokens if max_tokens is not None else self.max_tokens
        if self._use_max_completion_tokens():
            params["max_completion_tokens"] = token_limit
        else:
            params["max_tokens"] = token_limit

        params.update(kwargs)

        response = self.client.chat.completions.create(**params)
        latency = time.time() - start_time

        content = response.choices[0].message.content or ""
        finish_reason = response.choices[0].finish_reason
        refusal = getattr(response.choices[0].message, 'refusal', None)

        if not content.strip():
            raise EmptyLLMResponseError(
                "Azure OpenAI returned an empty response "
                f"(finish_reason={finish_reason}, refusal={refusal}, model={self.deployment})."
            )

        llm_response = LLMResponse(
            content=content,
            input_tokens=response.usage.prompt_tokens if response.usage else 0,
            output_tokens=response.usage.completion_tokens if response.usage else 0,
            latency=latency,
            model=self.deployment,
            raw_response=response,
        )
        get_usage_tracker().record(llm_response)
        return llm_response


class AnthropicClient(BaseLLMClient):
    """Anthropic Claude client."""

    def __init__(
        self,
        model: str = "claude-3-sonnet-20240229",
        temperature: float = 1.0,
        max_tokens: int = 2000,
        api_key: Optional[str] = None,
        **kwargs
    ):
        super().__init__(model, temperature, max_tokens, **kwargs)

        import anthropic

        self.client = anthropic.Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"),
        )

    @with_retry()
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        start_time = time.time()

        # Separate system message
        system_content = ""
        chat_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_content = msg["content"]
            else:
                chat_messages.append(msg)

        params = {
            "model": self.model,
            "messages": chat_messages,
            "max_tokens": max_tokens if max_tokens is not None else self.max_tokens,
            "temperature": temperature if temperature is not None else self.temperature,
        }
        if system_content:
            params["system"] = system_content

        params.update(kwargs)

        response = self.client.messages.create(**params)
        latency = time.time() - start_time

        content = response.content[0].text if response.content else ""
        if not content.strip():
            raise EmptyLLMResponseError(
                f"Anthropic returned an empty response (model={self.model})."
            )

        llm_response = LLMResponse(
            content=content,
            input_tokens=response.usage.input_tokens if response.usage else 0,
            output_tokens=response.usage.output_tokens if response.usage else 0,
            latency=latency,
            model=self.model,
            raw_response=response,
        )
        get_usage_tracker().record(llm_response)
        return llm_response


class BaseGeminiClient(BaseLLMClient):
    """Shared request and response handling for both Google Gemini transports."""

    def _chat_once(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        client: Any = None,
        **kwargs,
    ) -> LLMResponse:
        from google.genai import types

        start_time = time.time()
        attempt_started_at = time.perf_counter()
        system_messages = []
        contents = []
        for message in messages:
            role = message.get("role")
            content = message.get("content", "")
            if role == "system":
                system_messages.append(content)
                continue
            if role not in ("user", "assistant"):
                raise ValueError(f"Unsupported Gemini message role: {role}")
            contents.append(
                types.Content(
                    role="model" if role == "assistant" else "user",
                    parts=[types.Part.from_text(text=content)],
                )
            )

        if not contents:
            raise ValueError("Gemini requires at least one user or assistant message.")

        config = {
            "max_output_tokens": max_tokens if max_tokens is not None else self.max_tokens,
            "temperature": temperature if temperature is not None else self.temperature,
        }
        if system_messages:
            config["system_instruction"] = "\n\n".join(system_messages)

        # LightMem and the judge use OpenAI's JSON-object marker.
        response_format = kwargs.pop("response_format", None)
        expects_json = bool(response_format and response_format.get("type") == "json_object")
        if expects_json:
            config["response_mime_type"] = "application/json"

        for key in ("top_p", "top_k", "seed", "response_mime_type", "response_json_schema"):
            if key in kwargs:
                config[key] = kwargs.pop(key)

        active_client = client or self.client
        tracker = get_usage_tracker()
        tracker.record_attempt()
        try:
            response = active_client.models.generate_content(
                model=self.model,
                contents=contents,
                config=types.GenerateContentConfig(**config),
            )
        except Exception:
            tracker.record_failure()
            tracker.record_failure_duration(time.perf_counter() - attempt_started_at)
            raise
        latency = time.time() - start_time
        content = getattr(response, "text", "") or ""
        if not content.strip():
            candidates = getattr(response, "candidates", None) or []
            finish_reason = str(getattr(candidates[0], "finish_reason", "")) if candidates else ""
            prompt_feedback = getattr(response, "prompt_feedback", None)
            block_reason = str(getattr(prompt_feedback, "block_reason", "") or "")
            is_blocked = block_reason and "UNSPECIFIED" not in block_reason.upper()
            if "SAFETY" in finish_reason.upper() or is_blocked:
                tracker.record_failure()
                tracker.record_failure_duration(time.perf_counter() - attempt_started_at)
                raise LLMAPIError(
                    "Gemini returned no text because the response was blocked "
                    f"(finish_reason={finish_reason or 'unknown'})."
                )
            tracker.record_failure()
            tracker.record_failure_duration(time.perf_counter() - attempt_started_at)
            raise EmptyGeminiResponseError(
                "Gemini returned an empty response without a blocking reason."
            )
        if expects_json:
            try:
                json.loads(content)
            except (json.JSONDecodeError, TypeError) as exc:
                tracker.record_failure()
                tracker.record_failure_duration(time.perf_counter() - attempt_started_at)
                raise InvalidLLMResponseError(
                    "Gemini returned malformed JSON for a structured response."
                ) from exc
        usage = getattr(response, "usage_metadata", None)
        llm_response = LLMResponse(
            content=content,
            input_tokens=getattr(usage, "prompt_token_count", 0) or 0,
            output_tokens=getattr(usage, "candidates_token_count", 0) or 0,
            latency=latency,
            model=self.model,
            raw_response=response,
        )
        get_usage_tracker().record(llm_response)
        return llm_response


def get_google_service_account_files(
    service_account_files: Union[
        str,
        Path,
        Sequence[Union[str, Path]],
        None,
    ] = None,
) -> List[Path]:
    """Resolve ordered Vertex service-account paths relative to the project root."""
    configured = service_account_files
    if configured is None:
        configured = os.environ.get("GOOGLE_SERVICE_ACCOUNT_FILE")

    if configured is None:
        raw_paths: List[Union[str, Path]] = ["service-account.json"]
    elif isinstance(configured, (str, Path)):
        raw_paths = [
            part.strip()
            for part in re.split(r"[,;\n]+", str(configured))
            if part.strip()
        ]
    else:
        raw_paths = list(configured)

    project_root = Path(__file__).resolve().parent.parent
    resolved_paths: List[Path] = []
    for raw_path in raw_paths:
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = project_root / path
        path = path.resolve()
        if path not in resolved_paths:
            resolved_paths.append(path)

    if not resolved_paths:
        raise ValueError("At least one Vertex service-account file must be configured")

    missing_paths = [path for path in resolved_paths if not path.is_file()]
    if missing_paths:
        raise FileNotFoundError(
            "Vertex Gemini service-account file(s) not found: "
            + ", ".join(str(path) for path in missing_paths)
        )
    return resolved_paths


def _is_vertex_service_account_retryable(exc: Exception) -> Tuple[bool, str]:
    """Include credential/auth failures that can be fixed by account rotation."""
    retryable, reason = _is_retryable_exception(exc)
    if retryable:
        return True, reason

    status_code = _get_status_code(exc)
    message = str(exc).lower()
    credential_failure_type = _get_message_failure_type(
        message,
        _VERTEX_CREDENTIAL_FAILURE_TYPES,
    )
    if status_code in {401, 403} or credential_failure_type:
        return True, "Vertex service-account authentication or permission failure"
    return False, reason


class GeminiVertexClient(BaseGeminiClient):
    """Vertex Gemini client with per-failure-type service-account rotation."""

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        temperature: float = 1.0,
        max_tokens: int = 2000,
        project: Optional[str] = None,
        location: Optional[str] = None,
        service_account_file: Union[str, Path, Sequence[Union[str, Path]], None] = None,
        service_account_files: Union[str, Path, Sequence[Union[str, Path]], None] = None,
        service_account_failure_threshold: Optional[int] = None,
        **kwargs
    ):
        super().__init__(model, temperature, max_tokens, **kwargs)

        try:
            from google import genai
            from google.oauth2 import service_account
        except ImportError as exc:
            raise ImportError(
                "Vertex Gemini support requires google-genai and google-auth. "
                "Install dependencies with: pip install -r requirements.txt"
            ) from exc

        self.location = location or os.environ.get("GOOGLE_CLOUD_LOCATION", "global")
        if service_account_failure_threshold is None:
            service_account_failure_threshold = int(
                os.environ.get(
                    "GOOGLE_VERTEX_SERVICE_ACCOUNT_RETRIES",
                    str(GOOGLE_VERTEX_SERVICE_ACCOUNT_RETRIES),
                )
            )
        if service_account_failure_threshold < 1:
            raise ValueError("service_account_failure_threshold must be at least 1")
        self.service_account_failure_threshold = service_account_failure_threshold

        configured_files = service_account_files or service_account_file
        self.service_account_files = get_google_service_account_files(configured_files)
        self._vertex_accounts: List[Tuple[Path, Any, str, Any]] = []
        for credential_file in self.service_account_files:
            credentials = service_account.Credentials.from_service_account_file(
                str(credential_file),
                scopes=["https://www.googleapis.com/auth/cloud-platform"],
            )
            account_project = project or credentials.project_id
            if not account_project:
                raise ValueError(
                    "Vertex Gemini requires a project_id in every service-account file "
                    "or a project argument."
                )
            # enterprise=True selects Google Agent Platform, not the Developer API.
            account_client = genai.Client(
                enterprise=True,
                credentials=credentials,
                project=account_project,
                location=self.location,
            )
            self._vertex_accounts.append(
                (credential_file, credentials, account_project, account_client)
            )

        self._account_lock = threading.Lock()
        self._active_account_index = 0
        self._failure_counts: Dict[str, int] = {}
        self._activate_account(0)

    @property
    def active_service_account_index(self) -> int:
        """Return the active zero-based service-account index."""
        with self._account_lock:
            return self._active_account_index

    def _activate_account(self, account_index: int) -> None:
        """Expose one account through the legacy single-account attributes."""
        credential_file, credentials, project, client = self._vertex_accounts[account_index]
        self._active_account_index = account_index
        self.service_account_file = credential_file
        self.credentials = credentials
        self.project = project
        self.client = client

    def _active_account(self) -> Tuple[int, Path, Any, str, Any]:
        with self._account_lock:
            account_index = self._active_account_index
            credential_file, credentials, project, client = self._vertex_accounts[account_index]
        return account_index, credential_file, credentials, project, client

    def _select_account(self, account_index: int) -> None:
        """Select an account when an outer hybrid transport owns rotation."""
        with self._account_lock:
            self._activate_account(account_index)

    def _select_project(self, project: Optional[str]) -> bool:
        """Select the first account for a saved Vertex project, if configured."""
        if not project:
            return False
        with self._account_lock:
            for account_index, account in enumerate(self._vertex_accounts):
                if account[2] == project:
                    self._activate_account(account_index)
                    return True
        return False

    def _record_account_success(self, account_index: int) -> None:
        with self._account_lock:
            if self._active_account_index == account_index:
                self._failure_counts.clear()

    def _record_account_failure(
        self,
        account_index: int,
        failure_type: str,
    ) -> Tuple[int, bool, bool, int]:
        with self._account_lock:
            if self._active_account_index != account_index:
                return 0, False, False, self._active_account_index

            failure_count = self._failure_counts.get(failure_type, 0) + 1
            self._failure_counts[failure_type] = failure_count
            threshold_reached = failure_count >= self.service_account_failure_threshold
            rotated = threshold_reached and len(self._vertex_accounts) > 1
            if rotated:
                next_index = (self._active_account_index + 1) % len(self._vertex_accounts)
                self._activate_account(next_index)
            if threshold_reached:
                self._failure_counts.clear()
            return failure_count, threshold_reached, rotated, self._active_account_index

    def _run_with_service_account_rotation(self, operation, operation_name: str):
        """Run a Vertex operation through every configured service account."""
        last_exception: Optional[Exception] = None
        retry_counts: Dict[str, int] = {}
        total_attempts = 0
        exhausted_accounts = 0

        while exhausted_accounts < len(self._vertex_accounts):
            account_index, credential_file, credentials, project, client = self._active_account()
            tracker = get_usage_tracker()
            failure_duration_before = tracker.current_failure_duration()
            attempt_started_at = time.perf_counter()
            try:
                result = operation(client, credentials, project)
                self._record_account_success(account_index)
                return result
            except Exception as exc:
                _record_untracked_failure_duration(
                    tracker,
                    failure_duration_before,
                    attempt_started_at,
                )
                total_attempts += 1
                last_exception = exc
                retryable, reason = _is_vertex_service_account_retryable(exc)
                if not retryable:
                    logger.error(
                        "Vertex %s failed (non-retryable) with service account %d/%d: %s: %s",
                        operation_name,
                        account_index + 1,
                        len(self._vertex_accounts),
                        type(exc).__name__,
                        exc,
                    )
                    raise

                failure_type = _get_retry_failure_type(exc)
                account_attempt, threshold_reached, rotated, active_index = (
                    self._record_account_failure(account_index, failure_type)
                )
                retry_counts[failure_type] = retry_counts.get(failure_type, 0) + 1
                if threshold_reached:
                    exhausted_accounts += 1
                if rotated:
                    logger.warning(
                        "Vertex service account %d/%d (%s) reached %d/%d for %s; "
                        "rotating to account %d/%d (%s).",
                        account_index + 1,
                        len(self._vertex_accounts),
                        credential_file.name,
                        account_attempt,
                        self.service_account_failure_threshold,
                        failure_type,
                        active_index + 1,
                        len(self._vertex_accounts),
                        self.service_account_files[active_index].name,
                    )
                    get_usage_tracker().record_retry()
                    continue
                if threshold_reached:
                    break

                delay = min(
                    GEMINI_RETRY_INITIAL_DELAY * (2 ** (max(account_attempt, 1) - 1)),
                    GEMINI_RETRY_MAX_DELAY,
                )
                _log_retry_attempt(
                    max(account_attempt, 1),
                    self.service_account_failure_threshold,
                    delay,
                    exc,
                    reason,
                    failure_type=failure_type,
                )
                get_usage_tracker().record_retry()
                _sleep_after_failure(delay)

        raise LLMRetryExhaustedError(
            "Vertex call failed after trying every configured service account",
            last_exception=last_exception,
            attempts=total_attempts,
            failure_type=_get_retry_failure_type(last_exception),
            retry_counts=retry_counts,
        )

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        return self._run_with_service_account_rotation(
            lambda client, credentials, project: self._chat_once(
                messages,
                temperature,
                max_tokens,
                client=client,
                **kwargs,
            ),
            "chat call",
        )

    def generate_content(self, **kwargs):
        """Run a raw Vertex generate-content call with account rotation."""
        return self._run_with_service_account_rotation(
            lambda client, credentials, project: client.models.generate_content(**kwargs),
            "generate-content call",
        )


def _split_api_keys(value: Union[str, Sequence[str], None]) -> List[str]:
    """Normalize comma-, semicolon-, whitespace-, or list-separated API keys."""
    if value is None:
        return []
    values = [value] if isinstance(value, str) else list(value)
    keys: List[str] = []
    for item in values:
        keys.extend(part for part in re.split(r"[\s,;]+", str(item).strip()) if part)
    return keys


def get_google_ai_studio_api_keys(
    api_key: Union[str, Sequence[str], None] = None,
    api_keys: Union[str, Sequence[str], None] = None,
) -> List[str]:
    """Load AI Studio keys, preserving order and removing duplicates."""
    if api_keys:
        configured = _split_api_keys(api_keys)
    elif api_key:
        configured = _split_api_keys(api_key)
    else:
        configured = _split_api_keys(os.environ.get("GOOGLE_AI_STUDIO_API_KEYS"))
        if not configured:
            configured = _split_api_keys(os.environ.get("GOOGLE_AI_STUDIO_API_KEY"))
        if not configured:
            # Match Google's documented precedence when both standard variables exist.
            configured = _split_api_keys(
                os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
            )

    return list(dict.fromkeys(configured))


def _is_google_ai_studio_retryable(exc: Exception) -> Tuple[bool, str]:
    """Include authentication/key restriction failures in rotation decisions."""
    retryable, reason = _is_retryable_exception(exc)
    if retryable:
        return True, reason

    status_code = _get_status_code(exc)
    message = str(exc).lower()
    key_failure_type = _get_message_failure_type(message, _AI_STUDIO_KEY_FAILURE_TYPES)
    if status_code in {401, 403} or key_failure_type:
        return True, "AI Studio API key authentication, restriction, or quota failure"
    return False, reason


class GeminiAIStudioClient(BaseGeminiClient):
    """Gemini Developer API client with per-failure-type API key rotation."""

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        temperature: float = 1.0,
        max_tokens: int = 2000,
        api_key: Union[str, Sequence[str], None] = None,
        api_keys: Union[str, Sequence[str], None] = None,
        key_failure_threshold: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(model, temperature, max_tokens, **kwargs)

        try:
            from google import genai
        except ImportError as exc:
            raise ImportError(
                "Google AI Studio support requires google-genai. "
                "Install dependencies with: pip install -r requirements.txt"
            ) from exc

        self.api_keys = get_google_ai_studio_api_keys(api_key=api_key, api_keys=api_keys)
        if not self.api_keys:
            raise ValueError(
                "Google AI Studio requires GOOGLE_AI_STUDIO_API_KEYS, "
                "GOOGLE_AI_STUDIO_API_KEY, GEMINI_API_KEY, GOOGLE_API_KEY, or api_key."
            )
        if key_failure_threshold is None:
            key_failure_threshold = int(
                os.environ.get("GOOGLE_AI_STUDIO_KEY_RETRIES", str(GOOGLE_AI_STUDIO_KEY_RETRIES))
            )
        if key_failure_threshold < 1:
            raise ValueError("key_failure_threshold must be at least 1")

        self.key_failure_threshold = key_failure_threshold
        self._clients = [genai.Client(api_key=key) for key in self.api_keys]
        self._key_lock = threading.Lock()
        self._active_key_index = 0
        self._failure_counts: Dict[str, int] = {}
        self.client = self._clients[0]

    @property
    def active_key_index(self) -> int:
        """Return the zero-based active key index without exposing credentials."""
        with self._key_lock:
            return self._active_key_index

    def _active_client(self) -> Tuple[int, Any]:
        with self._key_lock:
            return self._active_key_index, self._clients[self._active_key_index]

    def _record_success(self, key_index: int) -> None:
        with self._key_lock:
            if self._active_key_index == key_index:
                self._failure_counts.clear()

    def _record_failure(
        self,
        key_index: int,
        failure_type: str,
    ) -> Tuple[int, bool, bool, int]:
        with self._key_lock:
            if self._active_key_index != key_index:
                return 0, False, False, self._active_key_index

            failure_count = self._failure_counts.get(failure_type, 0) + 1
            self._failure_counts[failure_type] = failure_count
            threshold_reached = failure_count >= self.key_failure_threshold
            rotated = threshold_reached and len(self._clients) > 1
            if rotated:
                self._active_key_index = (self._active_key_index + 1) % len(self._clients)
            if threshold_reached:
                self._failure_counts.clear()
                self.client = self._clients[self._active_key_index]
            return failure_count, threshold_reached, rotated, self._active_key_index

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> LLMResponse:
        last_exception: Optional[Exception] = None
        retry_counts: Dict[str, int] = {}
        total_attempts = 0
        exhausted_keys = 0

        while exhausted_keys < len(self._clients):
            key_index, client = self._active_client()
            tracker = get_usage_tracker()
            failure_duration_before = tracker.current_failure_duration()
            attempt_started_at = time.perf_counter()
            try:
                response = self._chat_once(
                    messages,
                    temperature,
                    max_tokens,
                    client=client,
                    **kwargs,
                )
                self._record_success(key_index)
                return response
            except Exception as exc:
                _record_untracked_failure_duration(
                    tracker,
                    failure_duration_before,
                    attempt_started_at,
                )
                total_attempts += 1
                last_exception = exc
                retryable, reason = _is_google_ai_studio_retryable(exc)
                if not retryable:
                    logger.error(
                        f"Google AI Studio call failed (non-retryable): "
                        f"{type(exc).__name__}: {exc}"
                    )
                    raise

                failure_type = _get_retry_failure_type(exc)
                key_attempt, threshold_reached, rotated, active_index = self._record_failure(
                    key_index,
                    failure_type,
                )
                retry_counts[failure_type] = retry_counts.get(failure_type, 0) + 1
                if threshold_reached:
                    exhausted_keys += 1
                if rotated:
                    logger.warning(
                        "Google AI Studio API key %d/%d reached %d/%d for %s; "
                        "rotating to key %d/%d.",
                        key_index + 1,
                        len(self._clients),
                        key_attempt,
                        self.key_failure_threshold,
                        failure_type,
                        active_index + 1,
                        len(self._clients),
                    )
                    get_usage_tracker().record_retry()
                    continue
                if threshold_reached:
                    break

                backoff_attempt = max(key_attempt, 1)
                delay = min(
                    GEMINI_RETRY_INITIAL_DELAY * (2 ** (backoff_attempt - 1)),
                    GEMINI_RETRY_MAX_DELAY,
                )
                _log_retry_attempt(
                    backoff_attempt,
                    self.key_failure_threshold,
                    delay,
                    exc,
                    reason,
                    failure_type=failure_type,
                )
                get_usage_tracker().record_retry()
                _sleep_after_failure(delay)

        raise LLMRetryExhaustedError(
            "Google AI Studio call failed after trying every configured API key",
            last_exception=last_exception,
            attempts=total_attempts,
            failure_type=_get_retry_failure_type(last_exception),
            retry_counts=retry_counts,
        )


class GeminiHybridClient(BaseGeminiClient):
    """Rotate real-time Gemini calls between Vertex and AI Studio transports."""

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        temperature: float = 1.0,
        max_tokens: int = 2000,
        api_key: Union[str, Sequence[str], None] = None,
        api_keys: Union[str, Sequence[str], None] = None,
        key_failure_threshold: Optional[int] = None,
        service_account_failure_threshold: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(model, temperature, max_tokens, **kwargs)

        if service_account_failure_threshold is None and key_failure_threshold is not None:
            service_account_failure_threshold = key_failure_threshold
        vertex_kwargs = {
            key: kwargs[key]
            for key in (
                "project",
                "location",
                "service_account_file",
                "service_account_files",
            )
            if key in kwargs
        }
        if service_account_failure_threshold is not None:
            vertex_kwargs["service_account_failure_threshold"] = (
                service_account_failure_threshold
            )
        self.vertex_client = GeminiVertexClient(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **vertex_kwargs,
        )
        self.ai_studio_client = GeminiAIStudioClient(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
            api_keys=api_keys,
            key_failure_threshold=key_failure_threshold,
        )
        self.key_failure_threshold = self.ai_studio_client.key_failure_threshold

        # Vertex attributes are exposed so the batch transport can always use it.
        self.project = self.vertex_client.project
        self.location = self.vertex_client.location
        self.credentials = self.vertex_client.credentials
        self.service_account_file = self.vertex_client.service_account_file
        self.client = self.vertex_client.client

        vertex_accounts = getattr(
            self.vertex_client,
            "_vertex_accounts",
            [
                (
                    self.vertex_client.service_account_file,
                    self.vertex_client.credentials,
                    self.vertex_client.project,
                    self.vertex_client.client,
                )
            ],
        )
        vertex_account_count = len(vertex_accounts)
        vertex_transports = [
            (
                VERTEX_GEMINI_PROVIDER
                if vertex_account_count == 1
                else f"{VERTEX_GEMINI_PROVIDER}[{index + 1}]",
                account[3],
            )
            for index, account in enumerate(vertex_accounts)
        ]
        ai_studio_transports = [
            (f"{GOOGLE_AI_STUDIO_PROVIDER}[{index}]", client)
            for index, client in enumerate(self.ai_studio_client._clients, start=1)
        ]
        self._transports = [*vertex_transports, *ai_studio_transports]
        self._vertex_transport_accounts = {
            transport_index: transport_index
            for transport_index in range(vertex_account_count)
        }
        self._transport_thresholds = [
            *(
                [
                    getattr(
                        self.vertex_client,
                        "service_account_failure_threshold",
                        service_account_failure_threshold
                        or GOOGLE_VERTEX_SERVICE_ACCOUNT_RETRIES,
                    )
                ]
                * vertex_account_count
            ),
            *([self.key_failure_threshold] * len(ai_studio_transports)),
        ]
        self._transport_lock = threading.Lock()
        self._active_transport_index = 0
        self._failure_counts: Dict[str, int] = {}

    @property
    def active_transport(self) -> str:
        """Return the active real-time transport without exposing credentials."""
        with self._transport_lock:
            return self._transports[self._active_transport_index][0]

    def _active_transport_client(self) -> Tuple[int, str, Any]:
        with self._transport_lock:
            transport_index = self._active_transport_index
            transport_name, client = self._transports[transport_index]
        vertex_account_index = self._vertex_transport_accounts.get(transport_index)
        if vertex_account_index is not None and hasattr(self.vertex_client, "_select_account"):
            self.vertex_client._select_account(vertex_account_index)
        return transport_index, transport_name, client

    def _record_transport_success(self, transport_index: int) -> None:
        with self._transport_lock:
            if self._active_transport_index == transport_index:
                self._failure_counts.clear()

    def _record_transport_failure(
        self,
        transport_index: int,
        failure_type: str,
    ) -> Tuple[int, bool, int]:
        with self._transport_lock:
            if self._active_transport_index != transport_index:
                return 0, False, self._active_transport_index

            failure_count = self._failure_counts.get(failure_type, 0) + 1
            self._failure_counts[failure_type] = failure_count
            threshold = self._transport_thresholds[transport_index]
            rotated = failure_count >= threshold
            if rotated:
                self._active_transport_index = (
                    self._active_transport_index + 1
                ) % len(self._transports)
                self._failure_counts.clear()
            return failure_count, rotated, self._active_transport_index

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> LLMResponse:
        # One complete Vertex + AI Studio-key cycle. Later calls continue from
        # the transport left active by the previous request.
        last_exception: Optional[Exception] = None
        retry_counts: Dict[str, int] = {}
        total_attempts = 0
        exhausted_transports = 0

        while exhausted_transports < len(self._transports):
            transport_index, transport_name, client = self._active_transport_client()
            tracker = get_usage_tracker()
            failure_duration_before = tracker.current_failure_duration()
            attempt_started_at = time.perf_counter()
            try:
                response = self._chat_once(
                    messages,
                    temperature,
                    max_tokens,
                    client=client,
                    **kwargs,
                )
                self._record_transport_success(transport_index)
                return response
            except Exception as exc:
                _record_untracked_failure_duration(
                    tracker,
                    failure_duration_before,
                    attempt_started_at,
                )
                total_attempts += 1
                last_exception = exc
                retryable, reason = _is_google_ai_studio_retryable(exc)
                if not retryable:
                    logger.error(
                        "Gemini %s call failed (non-retryable): %s: %s",
                        transport_name,
                        type(exc).__name__,
                        exc,
                    )
                    raise

                failure_type = _get_retry_failure_type(exc)
                transport_attempt, rotated, active_index = self._record_transport_failure(
                    transport_index,
                    failure_type,
                )
                retry_counts[failure_type] = retry_counts.get(failure_type, 0) + 1
                if rotated:
                    exhausted_transports += 1
                    next_transport = self._transports[active_index][0]
                    transport_threshold = self._transport_thresholds[transport_index]
                    logger.warning(
                        "Gemini %s transport reached %d/%d for %s; rotating to %s.",
                        transport_name,
                        transport_attempt,
                        transport_threshold,
                        failure_type,
                        next_transport,
                    )
                    get_usage_tracker().record_retry()
                    continue

                delay = min(
                    GEMINI_RETRY_INITIAL_DELAY * (2 ** (max(transport_attempt, 1) - 1)),
                    GEMINI_RETRY_MAX_DELAY,
                )
                _log_retry_attempt(
                    max(transport_attempt, 1),
                    self._transport_thresholds[transport_index],
                    delay,
                    exc,
                    reason,
                    failure_type=failure_type,
                )
                get_usage_tracker().record_retry()
                _sleep_after_failure(delay)

        raise LLMRetryExhaustedError(
            "Gemini call failed after trying Vertex and every AI Studio API key",
            last_exception=last_exception,
            attempts=total_attempts,
            failure_type=_get_retry_failure_type(last_exception),
            retry_counts=retry_counts,
        )

def create_llm_client(
    provider: str = "openai",
    model: str = "gpt-4o-mini",
    temperature: float = 1.0,
    max_tokens: int = 2000,
    **kwargs
) -> BaseLLMClient:
    """Create LLM client."""
    provider_map = {
        "openai": OpenAIClient,
        "modal": ModalClient,
        "azure": AzureOpenAIClient,
        "anthropic": AnthropicClient,
        VERTEX_GEMINI_PROVIDER: GeminiVertexClient,
        GOOGLE_AI_STUDIO_PROVIDER: GeminiAIStudioClient,
        HYBRID_GEMINI_PROVIDER: GeminiHybridClient,
    }

    client_class = provider_map.get(provider.lower())
    if client_class is None:
        raise ValueError(f"Unsupported LLM provider: {provider}")

    return client_class(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs
    )


def format_messages(
    user_message: str,
    system_message: Optional[str] = None,
) -> List[Dict[str, str]]:
    """Format chat messages."""
    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    messages.append({"role": "user", "content": user_message})
    return messages

__all__ = [
    # Response class
    "LLMResponse",
    # Client base class
    "BaseLLMClient",
    # Client implementations
    "OpenAIClient",
    "ModalClient",
    "AzureOpenAIClient",
    "AnthropicClient",
    "BaseGeminiClient",
    "GeminiVertexClient",
    "GeminiAIStudioClient",
    "GeminiHybridClient",
    # Exceptions
    "LLMAPIError",
    "RetryableLLMAPIError",
    "LLMRetryExhaustedError",
    "EmptyLLMResponseError",
    "EmptyGeminiResponseError",
    "InvalidLLMResponseError",
    # Factory functions
    "create_llm_client",
    "format_messages",
    # Retry config
    "DEFAULT_MAX_RETRIES",
    "DEFAULT_RETRY_MIN_DELAY",
    "DEFAULT_RETRY_MAX_DELAY",
    "GEMINI_MAX_RETRIES",
    "GEMINI_RETRY_INITIAL_DELAY",
    "GEMINI_RETRY_MAX_DELAY",
    "GOOGLE_AI_STUDIO_KEY_RETRIES",
    "GOOGLE_VERTEX_SERVICE_ACCOUNT_RETRIES",
    "VERTEX_GEMINI_PROVIDER",
    "GOOGLE_AI_STUDIO_PROVIDER",
    "HYBRID_GEMINI_PROVIDER",
    "GEMINI_PROVIDERS",
    "with_retry",
    "run_with_gemini_retry",
    "is_vertex_gemini_provider",
    "is_google_ai_studio_provider",
    "is_hybrid_gemini_provider",
    "is_vertex_batch_provider",
    "is_gemini_provider",
    "get_google_ai_studio_api_keys",
    "get_google_service_account_files",
    # Usage tracking
    "TokenUsage",
    "LLMUsageTracker",
    "get_usage_tracker",
]
