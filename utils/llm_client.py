"""LLM client module - unified interface for multiple providers."""

import os
import time
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Type
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
    total_latency: float = 0.0

    def add(self, input_tokens: int, output_tokens: int, latency: float) -> None:
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.total_tokens += input_tokens + output_tokens
        self.call_count += 1
        self.total_latency += latency

    def merge(self, other: "TokenUsage") -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.total_tokens += other.total_tokens
        self.call_count += other.call_count
        self.total_latency += other.total_latency

    def to_dict(self) -> Dict[str, Any]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "call_count": self.call_count,
            "total_latency": round(self.total_latency, 3),
            "avg_latency": round(self.total_latency / self.call_count, 3) if self.call_count > 0 else 0,
        }


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
        self._current_phase = "unknown"

    def set_phase(self, phase: str) -> None:
        self._current_phase = phase

    def record(self, response: "LLMResponse") -> None:
        if self._current_phase == "memorize":
            self._memorize_usage.add(response.input_tokens, response.output_tokens, response.latency)
        else:
            self._query_usage.add(response.input_tokens, response.output_tokens, response.latency)

    def reset(self) -> None:
        self._memorize_usage = TokenUsage()
        self._query_usage = TokenUsage()
        self._current_phase = "unknown"

    def get_stats(self) -> Dict[str, Any]:
        total = TokenUsage()
        total.merge(self._memorize_usage)
        total.merge(self._query_usage)
        return {
            "memorize_phase": self._memorize_usage.to_dict(),
            "query_phase": self._query_usage.to_dict(),
            "total": total.to_dict(),
        }


_usage_tracker: Optional[LLMUsageTracker] = None


def get_usage_tracker() -> LLMUsageTracker:
    global _usage_tracker
    if _usage_tracker is None:
        _usage_tracker = LLMUsageTracker()
    return _usage_tracker

# Default retry config (can be overridden via environment variables)
DEFAULT_MAX_RETRIES = int(os.environ.get("LLM_MAX_RETRIES", "10"))
DEFAULT_RETRY_MIN_DELAY = float(os.environ.get("LLM_RETRY_MIN_DELAY", "10.0"))
DEFAULT_RETRY_MAX_DELAY = float(os.environ.get("LLM_RETRY_MAX_DELAY", "20.0"))
GEMINI_MAX_RETRIES = int(os.environ.get("GEMINI_MAX_RETRIES", "100"))
GEMINI_RETRY_INITIAL_DELAY = 1.0
GEMINI_RETRY_MAX_DELAY = 100.0

logger = logging.getLogger(__name__)

class LLMAPIError(Exception):
    """LLM API error base class."""
    pass


class LLMRetryExhaustedError(LLMAPIError):
    """Retry exhausted error."""
    def __init__(self, message: str, last_exception: Exception, attempts: int):
        super().__init__(message)
        self.last_exception = last_exception
        self.attempts = attempts


class EmptyGeminiResponseError(LLMAPIError):
    """A completed Gemini request without generated text."""


def _is_retryable_exception(exc: Exception) -> Tuple[bool, str]:
    """Check if exception is retryable."""
    exc_type = type(exc).__name__
    exc_message = str(exc).lower()

    # OpenAI / Azure exceptions
    retryable_openai_errors = [
        "RateLimitError",
        "APIConnectionError",
        "APITimeoutError",
        "InternalServerError",
        "ServiceUnavailableError",
    ]

    # Anthropic exceptions
    retryable_anthropic_errors = [
        "RateLimitError",
        "APIConnectionError",
        "APITimeoutError",
        "InternalServerError",
        "OverloadedError",
    ]

    status_code = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    try:
        status_code = int(status_code)
    except (TypeError, ValueError):
        status_code = None
    if status_code is not None:
        if status_code in {408, 409, 429} or 500 <= status_code < 600:
            return True, f"Retryable HTTP status: {status_code}"
        if 400 <= status_code < 500:
            return False, f"Non-retryable HTTP status: {status_code}"

    # Google API Core exceptions used by the Gemini Enterprise Agent Platform SDK.
    retryable_google_errors = [
        "ResourceExhausted",
        "ServiceUnavailable",
        "InternalServerError",
        "DeadlineExceeded",
        "ServerError",
        "TooManyRequests",
    ]

    if (
        exc_type in retryable_openai_errors
        or exc_type in retryable_anthropic_errors
        or exc_type in retryable_google_errors
    ):
        return True, f"Retryable exception type: {exc_type}"

    # Check by exception message (compatibility)
    retryable_keywords = [
        "rate limit",
        "rate_limit",
        "too many requests",
        "429",
        "resource exhausted",
        "connection error",
        "connection reset",
        "timeout",
        "timed out",
        "service unavailable",
        "503",
        "502",
        "500",
        "internal server error",
        "overloaded",
        "capacity",
    ]

    for keyword in retryable_keywords:
        if keyword in exc_message:
            return True, f"Contains retryable keyword: {keyword}"

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

    # Default: don't retry unknown exceptions
    return False, f"Unknown exception type: {exc_type}"


def _get_retry_delay(min_delay: float, max_delay: float) -> float:
    """Get retry delay with jitter (random delay between min and max)."""
    import random
    return random.uniform(min_delay, max_delay)


def _log_retry_attempt(
    attempt: int,
    max_retries: int,
    delay: float,
    exc: Exception,
    reason: str
) -> None:
    """Log retry attempt."""
    logger.warning(
        f"API call failed (attempt {attempt}/{max_retries}): {type(exc).__name__}: {exc}\n"
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

    The actual delay is randomly chosen between retry_min_delay and retry_max_delay
    (jitter strategy) unless exponential_backoff is enabled.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(1, max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as exc:
                    last_exception = exc
                    is_retryable, reason = _is_retryable_exception(exc)

                    if not is_retryable:
                        # Non-retryable exception, raise immediately
                        logger.error(f"API call failed (non-retryable): {type(exc).__name__}: {exc}")
                        raise

                    if attempt < max_retries:
                        if exponential_backoff:
                            delay = min(
                                retry_min_delay * (2 ** (attempt - 1)),
                                retry_max_delay,
                            )
                        else:
                            delay = _get_retry_delay(retry_min_delay, retry_max_delay)
                        # Log and wait for retry
                        _log_retry_attempt(attempt, max_retries, delay, exc, reason)
                        time.sleep(delay)
                    else:
                        # Retry exhausted
                        logger.error(
                            f"API call retry exhausted ({max_retries} attempts): "
                            f"{type(exc).__name__}: {exc}"
                        )

            # Raise retry exhausted exception
            raise LLMRetryExhaustedError(
                f"API call still failed after {max_retries} retries",
                last_exception=last_exception,
                attempts=max_retries,
            )

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
            "temperature": temperature or self.temperature,
        }

        # Select token param based on model
        token_limit = max_tokens or self.max_tokens
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
            logger.warning(
                f"[OpenAIClient] Empty response - finish_reason={finish_reason}, "
                f"refusal={refusal}, model={self.model}"
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
            "temperature": temperature or self.temperature,
        }

        token_limit = max_tokens or self.max_tokens
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
            logger.warning(
                f"[AzureOpenAIClient] Empty response - finish_reason={finish_reason}, "
                f"refusal={refusal}, model={self.deployment}"
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
            "max_tokens": max_tokens or self.max_tokens,
            "temperature": temperature or self.temperature,
        }
        if system_content:
            params["system"] = system_content

        params.update(kwargs)

        response = self.client.messages.create(**params)
        latency = time.time() - start_time

        llm_response = LLMResponse(
            content=response.content[0].text if response.content else "",
            input_tokens=response.usage.input_tokens if response.usage else 0,
            output_tokens=response.usage.output_tokens if response.usage else 0,
            latency=latency,
            model=self.model,
            raw_response=response,
        )
        get_usage_tracker().record(llm_response)
        return llm_response


class GeminiEnterpriseClient(BaseLLMClient):
    """Gemini client using Gemini Enterprise Agent Platform (formerly Vertex AI)."""

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        temperature: float = 1.0,
        max_tokens: int = 2000,
        project: Optional[str] = None,
        location: Optional[str] = None,
        service_account_file: Optional[str] = None,
        **kwargs
    ):
        super().__init__(model, temperature, max_tokens, **kwargs)

        try:
            from google import genai
            from google.oauth2 import service_account
        except ImportError as exc:
            raise ImportError(
                "Gemini Enterprise support requires google-genai and google-auth. "
                "Install dependencies with: pip install -r requirements.txt"
            ) from exc

        default_credentials_file = Path(__file__).resolve().parent.parent / "service-account.json"
        self.service_account_file = Path(
            service_account_file
            or os.environ.get("GOOGLE_SERVICE_ACCOUNT_FILE")
            or default_credentials_file
        )
        if not self.service_account_file.is_file():
            raise FileNotFoundError(
                "Gemini Enterprise service-account file not found: "
                f"{self.service_account_file}"
            )

        credentials = service_account.Credentials.from_service_account_file(
            str(self.service_account_file),
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )
        # Reused by the optional Vertex batch transport; never serialize it.
        self.credentials = credentials
        self.project = project or credentials.project_id
        self.location = location or os.environ.get("GOOGLE_CLOUD_LOCATION", "global")
        if not self.project:
            raise ValueError(
                "Gemini Enterprise requires a project_id in the service-account file "
                "or a project argument."
            )

        # enterprise=True selects Google Agent Platform, not the Gemini Developer API.
        self.client = genai.Client(
            enterprise=True,
            credentials=credentials,
            project=self.project,
            location=self.location,
        )

    @with_retry(
        max_retries=GEMINI_MAX_RETRIES,
        retry_min_delay=GEMINI_RETRY_INITIAL_DELAY,
        retry_max_delay=GEMINI_RETRY_MAX_DELAY,
        exponential_backoff=True,
    )
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        from google.genai import types

        start_time = time.time()
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

        # LightMem asks for OpenAI's JSON-object response format. Map its equivalent.
        response_format = kwargs.pop("response_format", None)
        if response_format and response_format.get("type") == "json_object":
            config["response_mime_type"] = "application/json"

        for key in ("top_p", "top_k", "seed", "response_mime_type", "response_json_schema"):
            if key in kwargs:
                config[key] = kwargs.pop(key)

        response = self.client.models.generate_content(
            model=self.model,
            contents=contents,
            config=types.GenerateContentConfig(**config),
        )
        latency = time.time() - start_time
        content = getattr(response, "text", "") or ""
        if not content.strip():
            candidates = getattr(response, "candidates", None) or []
            finish_reason = str(getattr(candidates[0], "finish_reason", "")) if candidates else ""
            prompt_feedback = getattr(response, "prompt_feedback", None)
            block_reason = str(getattr(prompt_feedback, "block_reason", "") or "")
            is_blocked = block_reason and "UNSPECIFIED" not in block_reason.upper()
            if "SAFETY" in finish_reason.upper() or is_blocked:
                raise LLMAPIError(
                    "Gemini returned no text because the response was blocked "
                    f"(finish_reason={finish_reason or 'unknown'})."
                )
            raise EmptyGeminiResponseError(
                "Gemini returned an empty response without a blocking reason."
            )
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
        "azure": AzureOpenAIClient,
        "anthropic": AnthropicClient,
        "gemini": GeminiEnterpriseClient,
        "vertex_ai": GeminiEnterpriseClient,
        "vertex": GeminiEnterpriseClient,
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
    "AzureOpenAIClient",
    "AnthropicClient",
    "GeminiEnterpriseClient",
    # Exceptions
    "LLMAPIError",
    "LLMRetryExhaustedError",
    "EmptyGeminiResponseError",
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
    "with_retry",
    "run_with_gemini_retry",
    # Usage tracking
    "TokenUsage",
    "LLMUsageTracker",
    "get_usage_tracker",
]
