"""OpenRouter Batch API transport with real-time fallback."""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote, urlsplit, urlunsplit

import httpx

from utils.llm_client import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_RETRY_MAX_DELAY,
    DEFAULT_RETRY_MIN_DELAY,
    LLMRetryExhaustedError,
    OpenRouterClient,
    _get_exponential_retry_delay,
    _get_retry_failure_type,
    _is_retryable_exception,
    _log_retry_attempt,
    _sleep_after_failure,
    extract_usage_token_counts,
    get_usage_tracker,
)
from utils.vertex_batch import (
    MANIFEST_VERSION,
    BatchChatRequest,
    BatchChatResponse,
    VertexBatchClient,
    VertexBatchError,
    VertexBatchPending,
    _utc_now,
)


OPENROUTER_TERMINAL_STATES = {"completed", "failed", "expired", "cancelled"}


class OpenRouterBatchError(VertexBatchError):
    """Raised for an invalid or unrecoverable OpenRouter batch operation."""


class OpenRouterBatchUnsupported(OpenRouterBatchError):
    """Raised when the configured OpenRouter route cannot use Batch API."""


class OpenRouterBatchPending(VertexBatchPending):
    """Raised when submit-only mode leaves an OpenRouter batch unfinished."""

    def __init__(self, stage: str, job_name: str, manifest_path: Path):
        RuntimeError.__init__(
            self,
            f"OpenRouter batch stage '{stage}' is pending: {job_name}. "
            "Resume with --resume --batch-api after it completes.",
        )
        self.stage = stage
        self.job_name = job_name
        self.manifest_path = manifest_path


class _OpenRouterHTTPError(OpenRouterBatchError):
    def __init__(self, status_code: int, message: str):
        super().__init__(f"OpenRouter Batch API returned HTTP {status_code}: {message}")
        self.status_code = status_code


def _batch_api_base_url(base_url: str) -> str:
    """Derive OpenRouter's beta API root from its OpenAI-compatible base URL."""
    parsed = urlsplit(str(base_url).rstrip("/"))
    path = parsed.path.rstrip("/")
    if path.endswith("/api/v1"):
        path = f"{path[:-len('/api/v1')]}/api/beta"
    elif path.endswith("/v1"):
        path = f"{path[:-len('/v1')]}/beta"
    elif not path.endswith("/beta"):
        path = f"{path}/beta"
    return urlunsplit((parsed.scheme, parsed.netloc, path, "", "")).rstrip("/")


def _provider_slug_matches(configured: str, endpoint: str) -> bool:
    configured = configured.lower().rstrip("/")
    endpoint = endpoint.lower().rstrip("/")
    if configured == endpoint:
        return True
    # Base slugs match regional variants, but not service-tier variants.
    suffix = endpoint[len(configured):] if endpoint.startswith(f"{configured}/") else ""
    return bool(suffix and suffix not in {"/fast", "/priority", "/flex"})


def _eligible_endpoint_tags(
    endpoints: List[Dict[str, Any]],
    provider_routing: Optional[Dict[str, Any]],
    service_tier: Optional[str],
) -> List[str]:
    tags = [
        str(endpoint.get("tag", "")).lower()
        for endpoint in endpoints
        if endpoint.get("tag")
    ]
    routing = provider_routing or {}
    ignored = [str(value) for value in routing.get("ignore", [])]
    allowed = routing.get("only")
    if allowed is None and routing.get("allow_fallbacks") is False:
        allowed = routing.get("order")

    if ignored:
        tags = [
            tag for tag in tags
            if not any(_provider_slug_matches(value, tag) for value in ignored)
        ]
    if allowed:
        tags = [
            tag for tag in tags
            if any(_provider_slug_matches(str(value), tag) for value in allowed)
        ]

    tier = (service_tier or "default").lower()
    if tier in {"priority", "fast"}:
        tags = [tag for tag in tags if tag.endswith(("/fast", "/priority"))]
    elif tier == "flex":
        tags = [tag for tag in tags if tag.endswith("/flex")]
    return tags


class OpenRouterBatchClient(VertexBatchClient):
    """Submit and collect OpenRouter inline batch jobs."""

    def __init__(
        self,
        *,
        direct_client: OpenRouterClient,
        manifest_path: Path,
        wait: bool,
        config_hash: str = "",
        poll_interval: int = 30,
        http_client: Optional[Any] = None,
        progress_callback=None,
    ):
        self.model = direct_client.model.removesuffix(":batch")
        self.project = "openrouter"
        self.location = ""
        self.gcs_uri = ""
        self.manifest_path = Path(manifest_path)
        self.wait_for_completion = wait
        self.config_hash = config_hash
        self.poll_interval = poll_interval
        self._direct_client = direct_client
        self._progress_callback = progress_callback
        self.provider_routing = direct_client.provider_routing
        self.service_tier = direct_client.service_tier
        self.sync_base_url = str(direct_client.client.base_url).rstrip("/")
        self.base_url = _batch_api_base_url(self.sync_base_url)
        self._http_client = http_client or httpx.Client(
            timeout=httpx.Timeout(180.0, connect=30.0),
            headers={
                "Authorization": f"Bearer {direct_client.client.api_key}",
                "Content-Type": "application/json",
            },
        )
        self._support: Optional[Tuple[bool, str]] = None

    @classmethod
    def from_openrouter_client(
        cls,
        client: Any,
        *,
        manifest_path: Path,
        wait: bool,
        config_hash: str = "",
        poll_interval: int = 30,
        progress_callback=None,
        http_client: Optional[Any] = None,
    ) -> "OpenRouterBatchClient":
        if not isinstance(client, OpenRouterClient):
            raise OpenRouterBatchError("OpenRouter batch API requires provider: openrouter.")
        return cls(
            direct_client=client,
            manifest_path=manifest_path,
            wait=wait,
            config_hash=config_hash,
            poll_interval=poll_interval,
            http_client=http_client,
            progress_callback=progress_callback,
        )

    def _progress(self, message: str) -> None:
        if self._progress_callback is not None:
            self._progress_callback(f"[OpenRouter Batch] {message}")

    def _request(self, method: str, url: str, **kwargs) -> Dict[str, Any]:
        retry = kwargs.pop("retry", True)
        failure_counts: Dict[str, int] = {}
        total_attempts = 0
        while True:
            try:
                response = self._http_client.request(method, url, **kwargs)
                if response.status_code >= 400:
                    try:
                        payload = response.json()
                        message = json.dumps(payload, ensure_ascii=False, sort_keys=True)
                    except Exception:
                        message = getattr(response, "text", "") or "request failed"
                    error = _OpenRouterHTTPError(response.status_code, message)
                    if response.status_code in {400, 404, 422}:
                        raise OpenRouterBatchUnsupported(str(error)) from error
                    raise error
                payload = response.json()
                if not isinstance(payload, dict):
                    raise OpenRouterBatchError(
                        "OpenRouter Batch API returned a non-object response."
                    )
                return payload
            except Exception as exc:
                if not retry:
                    raise
                retryable, reason = _is_retryable_exception(exc)
                if not retryable:
                    raise
                total_attempts += 1
                failure_type = _get_retry_failure_type(exc)
                failure_attempt = failure_counts.get(failure_type, 0) + 1
                failure_counts[failure_type] = failure_attempt
                if failure_attempt >= DEFAULT_MAX_RETRIES:
                    raise LLMRetryExhaustedError(
                        "OpenRouter Batch API retry budget exhausted",
                        last_exception=exc,
                        attempts=total_attempts,
                        failure_type=failure_type,
                        retry_counts=failure_counts,
                    ) from exc
                delay = _get_exponential_retry_delay(
                    failure_attempt,
                    DEFAULT_RETRY_MIN_DELAY,
                    DEFAULT_RETRY_MAX_DELAY,
                )
                _log_retry_attempt(
                    failure_attempt,
                    DEFAULT_MAX_RETRIES,
                    delay,
                    exc,
                    reason,
                    failure_type=failure_type,
                )
                get_usage_tracker().record_retry()
                _sleep_after_failure(delay)

    def _check_support(self) -> Tuple[bool, str]:
        if self._support is not None:
            return self._support
        batch_model = quote(f"{self.model}:batch", safe="/")
        try:
            payload = self._request(
                "GET",
                f"{self.sync_base_url}/models/{batch_model}/endpoints",
                retry=False,
                timeout=15.0,
            )
        except OpenRouterBatchUnsupported as exc:
            self._support = (False, str(exc))
            return self._support
        except Exception as exc:
            self._support = (
                False,
                f"batch support could not be confirmed ({type(exc).__name__}: {exc})",
            )
            return self._support

        endpoints = (payload.get("data") or {}).get("endpoints") or []
        tags = _eligible_endpoint_tags(
            endpoints,
            self.provider_routing,
            self.service_tier,
        )
        if tags:
            self._support = (True, f"eligible upstreams: {', '.join(tags)}")
        else:
            route = "configured model/provider/service tier"
            self._support = (
                False,
                f"no batch endpoint matches the {route}",
            )
        return self._support

    def _load_manifest(self) -> Dict[str, Any]:
        if not self.manifest_path.exists():
            return {
                "version": MANIFEST_VERSION,
                "provider": "openrouter",
                "created_at": _utc_now(),
                "model": self.model,
                "config_hash": self.config_hash,
                "run_id": uuid.uuid4().hex,
                "jobs": {},
            }
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        if manifest.get("version") != MANIFEST_VERSION:
            raise OpenRouterBatchError(
                f"Unsupported batch manifest version in {self.manifest_path}."
            )
        if manifest.get("provider") != "openrouter" or manifest.get("model") != self.model:
            raise OpenRouterBatchError("Batch manifest does not match the OpenRouter model.")
        if manifest.get("config_hash", "") != self.config_hash:
            raise OpenRouterBatchError(
                "Batch manifest does not match this evaluator configuration."
            )
        return manifest

    @classmethod
    def _validate_request_correlation(
        cls, requests: List[BatchChatRequest]
    ) -> Dict[str, str]:
        request_ids = [request.request_id for request in requests]
        if len(request_ids) != len(set(request_ids)):
            raise OpenRouterBatchError("OpenRouter batch custom_id values must be unique.")
        return {request_id: request_id for request_id in request_ids}

    def _response_format(self, request: BatchChatRequest) -> Optional[Dict[str, Any]]:
        response_format = request.response_format
        if not response_format:
            return None
        schema = response_format.get("response_json_schema") or response_format.get(
            "response_schema"
        )
        if schema is not None:
            return {
                "type": "json_schema",
                "json_schema": {
                    "name": "batch_response",
                    "strict": True,
                    "schema": schema,
                },
            }
        return {
            key: value
            for key, value in response_format.items()
            if key not in {"response_json_schema", "response_schema"}
        }

    def _request_body(self, request: BatchChatRequest) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            "messages": request.messages,
            "temperature": request.temperature,
        }
        token_key = (
            "max_completion_tokens"
            if self._direct_client._use_max_completion_tokens()
            else "max_tokens"
        )
        body[token_key] = request.max_tokens
        response_format = self._response_format(request)
        if response_format is not None:
            body["response_format"] = response_format
        if self.provider_routing is not None:
            body["provider"] = dict(self.provider_routing)
        if self.service_tier is not None:
            body["service_tier"] = self.service_tier
        reasoning_effort = (
            request.reasoning_effort
            if request.reasoning_effort is not None
            else getattr(self._direct_client, "reasoning_effort", None)
        )
        if reasoning_effort is not None:
            reasoning = dict(body.get("reasoning", {}) or {})
            reasoning.setdefault("effort", reasoning_effort)
            body["reasoning"] = reasoning
        return body

    def _submit(
        self,
        stage: str,
        requests: List[BatchChatRequest],
        run_id: str,
    ) -> Dict[str, Any]:
        self._progress(
            f"Stage '{stage}': submitting {len(requests):,} request(s) for model {self.model}."
        )
        payload = self._request(
            "POST",
            f"{self.base_url}/batches",
            content=json.dumps(
                {
                    "endpoint": "/v1/chat/completions",
                    "model": self.model,
                    "requests": [
                        {
                            "custom_id": request.request_id,
                            "body": self._request_body(request),
                        }
                        for request in requests
                    ],
                },
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        )
        job_name = str(payload.get("id") or "")
        if not job_name:
            raise OpenRouterBatchError("OpenRouter batch submission returned no batch ID.")
        state = str(payload.get("status") or "validating").lower()
        self._progress(f"Stage '{stage}': submitted {job_name} (state: {state}).")
        return {
            "stage": stage,
            "job_name": job_name,
            "state": state,
            "project": self.project,
            "input_uri": None,
            "output_uri": None,
            "submitted_at": _utc_now(),
            "requests": [request.to_manifest_dict() for request in requests],
            "request_fingerprint": self._request_fingerprint(requests),
            "responses": {},
        }

    def _get_job(self, job_name: str, project: Optional[str] = None) -> Dict[str, Any]:
        return self._request("GET", f"{self.base_url}/batches/{quote(job_name, safe='')}")

    def _wait_for_job(self, job_entry: Dict[str, Any]) -> Dict[str, Any]:
        job = self._get_job(job_entry["job_name"])
        state = str(job.get("status") or "").lower()
        attempts = 0
        self._progress(
            f"Stage '{job_entry['stage']}': {job_entry['job_name']} is {state or 'unknown'}; "
            f"polling every {self.poll_interval}s."
        )
        while state not in OPENROUTER_TERMINAL_STATES:
            time.sleep(self.poll_interval)
            attempts += 1
            job = self._get_job(job_entry["job_name"])
            new_state = str(job.get("status") or "").lower()
            if new_state != state:
                self._progress(
                    f"Stage '{job_entry['stage']}': poll {attempts}; "
                    f"{job_entry['job_name']} is {new_state or 'unknown'}."
                )
            state = new_state
        job_entry["state"] = state
        job_entry["completed_at"] = _utc_now()
        return job

    def _wait_or_raise_pending(
        self,
        stage: str,
        job_entry: Dict[str, Any],
        manifest: Dict[str, Any],
    ) -> None:
        job = self._get_job(job_entry["job_name"])
        job_entry["state"] = str(job.get("status") or "").lower()
        if job_entry["state"] in OPENROUTER_TERMINAL_STATES:
            return
        if not self.wait_for_completion:
            self._save_manifest(manifest)
            raise OpenRouterBatchPending(stage, job_entry["job_name"], self.manifest_path)
        self._wait_for_job(job_entry)

    @staticmethod
    def _content(body: Dict[str, Any]) -> str:
        choices = body.get("choices") or []
        if not choices:
            return ""
        content = (choices[0].get("message") or {}).get("content") or ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(
                str(part.get("text", ""))
                for part in content
                if isinstance(part, dict)
            )
        return str(content)

    def _collect(self, job_entry: Dict[str, Any]) -> Dict[str, BatchChatResponse]:
        job = self._get_job(job_entry["job_name"])
        state = str(job.get("status") or "").lower()
        job_entry["state"] = state
        if state != "completed":
            error = job.get("error") or f"batch ended with status {state or 'unknown'}"
            message = (
                json.dumps(error, ensure_ascii=False)
                if isinstance(error, dict)
                else str(error)
            )
            lowered = message.lower()
            if any(
                marker in lowered
                for marker in (
                    "model",
                    "provider",
                    "endpoint",
                    "route",
                    "tier",
                    "quantization",
                    "unsupported",
                    "not support",
                    "not available",
                    "no endpoint",
                    "validation",
                )
            ):
                raise OpenRouterBatchUnsupported(message)
            raise OpenRouterBatchError(message)

        requests = {
            request["request_id"]: BatchChatRequest.from_manifest_dict(request)
            for request in job_entry["requests"]
        }
        responses: Dict[str, BatchChatResponse] = {}
        for result in job.get("results") or []:
            request_id = str(result.get("custom_id") or "")
            if request_id not in requests:
                raise OpenRouterBatchError(
                    f"OpenRouter batch returned unknown custom_id {request_id!r}."
                )
            if request_id in responses:
                raise OpenRouterBatchError(
                    f"OpenRouter batch returned duplicate custom_id {request_id!r}."
                )
            error = result.get("error")
            response = result.get("response") or {}
            body = response.get("body") or {}
            usage = body.get("usage") or {}
            status = ""
            if error:
                status = json.dumps(error, ensure_ascii=False, sort_keys=True)
            elif int(response.get("status_code", 200) or 200) >= 400:
                status = f"HTTP {response.get('status_code')}"
            content = self._content(body)
            if not status and not content.strip():
                status = "Empty response"
            input_tokens, output_tokens, visible_output_tokens, thinking_tokens = (
                extract_usage_token_counts(usage)
            )
            responses[request_id] = BatchChatResponse(
                request_id=request_id,
                content=content,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                visible_output_tokens=visible_output_tokens,
                thinking_tokens=thinking_tokens,
                status=status,
                raw_response=result,
            )

        job_entry["responses"] = {
            request_id: {
                "content": response.content,
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "visible_output_tokens": response.visible_output_tokens,
                "thinking_tokens": response.thinking_tokens,
                "status": response.status,
            }
            for request_id, response in responses.items()
        }
        job_entry["missing_request_ids"] = sorted(set(requests) - set(responses))
        return responses

    def _run_direct(
        self,
        stage: str,
        requests: List[BatchChatRequest],
    ) -> Dict[str, BatchChatResponse]:
        self._progress(
            f"Stage '{stage}': using the existing real-time OpenRouter client for "
            f"{len(requests):,} request(s)."
        )
        responses: Dict[str, BatchChatResponse] = {}
        for request in requests:
            kwargs: Dict[str, Any] = {}
            if self._direct_client.model != self.model:
                kwargs["model"] = self.model
            response_format = self._response_format(request)
            if response_format is not None:
                kwargs["response_format"] = response_format
            get_usage_tracker().set_phase(request.phase)
            started_at = time.perf_counter()
            try:
                response = self._direct_client.chat(
                    request.messages,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens,
                    **kwargs,
                )
                responses[request.request_id] = BatchChatResponse(
                    request_id=request.request_id,
                    content=response.content,
                    input_tokens=response.input_tokens,
                    output_tokens=response.output_tokens,
                    visible_output_tokens=response.visible_output_tokens,
                    thinking_tokens=response.thinking_tokens,
                    duration_seconds=time.perf_counter() - started_at,
                    raw_response=response.raw_response,
                )
            except Exception as exc:
                responses[request.request_id] = BatchChatResponse(
                    request_id=request.request_id,
                    status=f"{type(exc).__name__}: {exc}",
                    duration_seconds=time.perf_counter() - started_at,
                )
        return responses

    def run_stage(
        self,
        stage: str,
        requests: List[BatchChatRequest],
    ) -> Dict[str, BatchChatResponse]:
        if not requests:
            return {}
        supported, reason = self._check_support()
        if not supported:
            self._progress(
                f"Stage '{stage}': {reason}; ignoring --batch-api and using real-time calls."
            )
            return self._run_direct(stage, requests)
        try:
            return super().run_stage(stage, requests)
        except OpenRouterBatchUnsupported as exc:
            manifest = self._load_manifest()
            job_entry = manifest.get("jobs", {}).get(stage)
            if isinstance(job_entry, dict):
                job_entry["state"] = "failed"
                job_entry["fallback_reason"] = str(exc)
                self._save_manifest(manifest)
            self._progress(
                f"Stage '{stage}': batch route is unsupported ({exc}); using real-time calls."
            )
            return self._run_direct(stage, requests)

    def has_unfinished_stage(self) -> Optional[Dict[str, Any]]:
        manifest = self._load_manifest()
        for job in manifest.get("jobs", {}).values():
            if job.get("state") not in OPENROUTER_TERMINAL_STATES:
                return job
            for retry in job.get("retries", []):
                if retry.get("state") not in OPENROUTER_TERMINAL_STATES:
                    return retry
        return None


__all__ = [
    "OpenRouterBatchClient",
    "OpenRouterBatchError",
    "OpenRouterBatchPending",
    "OpenRouterBatchUnsupported",
]
