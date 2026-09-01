"""Vertex Gemini Batch API support for offline evaluator stages.

Vertex batch jobs require Cloud Storage JSONL input.  This module keeps the
provider-specific staging, polling, and result parsing out of evaluators and
retains a local manifest so a submitted job can be collected by ``--resume``.
"""

from __future__ import annotations

import json
import hashlib
import os
import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

from utils.llm_client import (
    LLMResponse,
    extract_usage_token_counts,
    get_usage_tracker,
)
from utils.logger import truncate_error_message


MANIFEST_VERSION = 2
PREPARED_QUERY_METADATA_KEY = "prepared_query"
DEFAULT_MIN_BATCH_REQUESTS = 6
TERMINAL_STATES = {
    "JOB_STATE_SUCCEEDED",
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_PAUSED",
    "JOB_STATE_EXPIRED",
}

# The Vertex SDK exposes a batch job's state, but not a reliable per-request
# completion counter while it is running.  Callers use this hook to surface
# every lifecycle transition rather than looking like a blocking API call.
BatchProgressCallback = Callable[[str], None]


def min_batch_requests() -> int:
    """Return the minimum request count worth sending to Vertex Batch."""
    raw_value = os.environ.get("VERTEX_BATCH_MIN_REQUESTS", str(DEFAULT_MIN_BATCH_REQUESTS))
    try:
        return max(2, int(raw_value))
    except ValueError:
        return DEFAULT_MIN_BATCH_REQUESTS


def should_use_batch(request_count: int) -> bool:
    """Avoid job overhead for empty or very small stages."""
    return request_count >= min_batch_requests()


class VertexBatchError(RuntimeError):
    """Raised for an invalid or unrecoverable Vertex batch operation."""


class VertexBatchPending(RuntimeError):
    """Raised when submit-only mode has created or found an unfinished job."""

    def __init__(self, stage: str, job_name: str, manifest_path: Path):
        super().__init__(
            f"Vertex batch stage '{stage}' is pending: {job_name}. "
            f"Resume with --resume --batch-api after it completes."
        )
        self.stage = stage
        self.job_name = job_name
        self.manifest_path = manifest_path


@dataclass
class BatchChatRequest:
    """A serializable GenerateContent request and its local correlation data."""

    request_id: str
    messages: List[Dict[str, str]]
    temperature: float
    max_tokens: int
    reasoning_effort: Optional[Union[str, int]] = None
    response_format: Optional[Dict[str, Any]] = None
    phase: str = "query"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_vertex_request(
        self,
        model: Optional[str] = None,
        reasoning_effort: Optional[Union[str, int]] = None,
    ) -> Dict[str, Any]:
        """Convert chat messages to one documented GenerateContent request."""
        system_messages: List[str] = []
        contents: List[Dict[str, Any]] = []

        for message in self.messages:
            role = message.get("role")
            content = message.get("content", "")
            if role == "system":
                system_messages.append(content)
                continue
            if role not in {"user", "assistant"}:
                raise VertexBatchError(f"Unsupported batch message role: {role}")
            contents.append(
                {
                    "role": "model" if role == "assistant" else "user",
                    "parts": [{"text": content}],
                }
            )

        if not contents:
            raise VertexBatchError("A batch request requires a user or assistant message.")

        generation_config: Dict[str, Any] = {
            "temperature": self.temperature,
            "maxOutputTokens": self.max_tokens,
        }
        if self.response_format and self.response_format.get("type") == "json_object":
            generation_config["responseMimeType"] = "application/json"
            # Vertex Batch JSONL uses ``responseSchema``.  Keep compatibility
            # with callers that still provide the SDK-style
            # ``response_json_schema`` key, but serialize that to Vertex's
            # documented field as well.
            response_json_schema = self.response_format.get("response_json_schema")
            response_schema = self.response_format.get("response_schema")
            if response_json_schema is not None:
                if not isinstance(response_json_schema, dict):
                    raise VertexBatchError("response_json_schema must be a JSON object.")
                generation_config["responseSchema"] = response_json_schema
            elif response_schema is not None:
                if not isinstance(response_schema, dict):
                    raise VertexBatchError("response_schema must be a JSON object.")
                generation_config["responseSchema"] = response_schema

        effective_reasoning_effort = (
            self.reasoning_effort if self.reasoning_effort is not None else reasoning_effort
        )
        if effective_reasoning_effort is not None:
            model_name = str(model or self.metadata.get("model", "")).lower()
            # The batch client supplies the model name while serializing;
            # standalone requests default to the Gemini 2.5 budget form.
            if isinstance(effective_reasoning_effort, int) and not isinstance(effective_reasoning_effort, bool):
                generation_config["thinkingConfig"] = {
                    "thinkingBudget": effective_reasoning_effort,
                }
            elif isinstance(effective_reasoning_effort, str) and "gemini-3" in model_name:
                level = effective_reasoning_effort.strip().upper()
                if level not in {"MINIMAL", "LOW", "MEDIUM", "HIGH"}:
                    raise VertexBatchError(
                        "Gemini 3 reasoning_effort must be minimal, low, medium, or high."
                    )
                generation_config["thinkingConfig"] = {"thinkingLevel": level}
            elif isinstance(effective_reasoning_effort, str):
                raise VertexBatchError(
                    "Gemini 2.5 reasoning_effort must be an integer thinking budget."
                )
            else:
                raise VertexBatchError("reasoning_effort must be a string or integer.")

        request: Dict[str, Any] = {
            "contents": contents,
            "generationConfig": generation_config,
        }
        if system_messages:
            request["systemInstruction"] = {
                "parts": [{"text": "\n\n".join(system_messages)}]
            }

        return request

    def to_vertex_record(
        self,
        model: Optional[str] = None,
        reasoning_effort: Optional[Union[str, int]] = None,
    ) -> Dict[str, Any]:
        """Wrap a GenerateContent request in the documented GCS JSONL shape."""
        # Vertex's GCS output echoes ``request`` but does not preserve arbitrary
        # top-level fields, so local request IDs remain in the manifest only.
        return {
            "request": self.to_vertex_request(
                model=model,
                reasoning_effort=reasoning_effort,
            )
        }

    def to_manifest_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_manifest_dict(cls, data: Dict[str, Any]) -> "BatchChatRequest":
        return cls(
            request_id=data["request_id"],
            messages=data["messages"],
            temperature=data["temperature"],
            max_tokens=data["max_tokens"],
            reasoning_effort=data.get("reasoning_effort"),
            response_format=data.get("response_format"),
            phase=data.get("phase", "query"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class BatchChatResponse:
    """One parsed Vertex batch output row."""

    request_id: str
    content: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    status: str = ""
    duration_seconds: float = 0.0
    raw_response: Any = field(default_factory=dict)
    visible_output_tokens: Optional[int] = None
    thinking_tokens: int = 0

    def __post_init__(self) -> None:
        self.input_tokens = max(int(self.input_tokens or 0), 0)
        self.output_tokens = max(int(self.output_tokens or 0), 0)
        self.thinking_tokens = min(
            max(int(self.thinking_tokens or 0), 0), self.output_tokens
        )
        if self.visible_output_tokens is None:
            self.visible_output_tokens = self.output_tokens - self.thinking_tokens
        self.visible_output_tokens = min(
            max(int(self.visible_output_tokens or 0), 0),
            self.output_tokens - self.thinking_tokens,
        )

    @property
    def succeeded(self) -> bool:
        return not self.status and bool(self.content)

    def to_llm_response(self, model: str) -> LLMResponse:
        return LLMResponse(
            content=self.content,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            visible_output_tokens=self.visible_output_tokens,
            thinking_tokens=self.thinking_tokens,
            # Job queue time is not a per-request model latency.
            latency=0.0,
            model=model,
            raw_response=self.raw_response,
        )


def make_request_id(prefix: str = "req", key: Optional[str] = None) -> str:
    """Return a concise opaque request ID accepted by JSONL batch input.

    ``key`` makes the identifier stable across a submit/resume invocation.
    """
    safe_prefix = re.sub(r"[^a-z0-9-]", "-", prefix.lower()).strip("-") or "req"
    token = uuid.uuid5(uuid.NAMESPACE_URL, f"medmemorybench:{safe_prefix}:{key}").hex if key else uuid.uuid4().hex
    return f"{safe_prefix}-{token}"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _state_name(state: Any) -> str:
    """Normalize SDK enum values and mocked string states."""
    if state is None:
        return ""
    value = getattr(state, "name", None) or getattr(state, "value", None) or str(state)
    return str(value).split(".")[-1]


def _parse_gcs_uri(uri: str) -> Tuple[str, str]:
    if not uri.startswith("gs://"):
        raise VertexBatchError("GOOGLE_BATCH_GCS_URI must start with gs://")
    bucket_and_path = uri[5:].strip("/")
    bucket, separator, object_prefix = bucket_and_path.partition("/")
    if not bucket or not separator or not object_prefix:
        raise VertexBatchError(
            "GOOGLE_BATCH_GCS_URI must include both a bucket and a non-empty prefix, "
            "for example gs://my-private-bucket/medmemory-batch"
        )
    return bucket, object_prefix.rstrip("/")


def scoped_manifest_path(
    manifest_dir: Path,
    stem: str,
    *,
    model: str,
    config_hash: str,
) -> Path:
    """Return a config-scoped manifest path, preserving a compatible legacy one.

    Older runs stored every method's manifest under the same filename.  That
    makes an otherwise safe configuration check fail when two methods share an
    output directory.  New files are isolated by model and config hash, while
    a matching old manifest remains resumable instead of being re-submitted.
    """
    manifest_dir = Path(manifest_dir)
    legacy_path = manifest_dir / f"{stem}.json"
    safe_model = re.sub(r"[^a-zA-Z0-9_.-]", "-", model).strip("-") or "model"
    safe_hash = re.sub(r"[^a-zA-Z0-9]", "", config_hash) or "nohash"
    scoped_path = manifest_dir / f"{stem}-{safe_model}-{safe_hash}.json"
    if scoped_path.exists() or not legacy_path.exists():
        return scoped_path

    try:
        with legacy_path.open("r", encoding="utf-8") as handle:
            legacy = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return scoped_path
    if legacy.get("model") == model and legacy.get("config_hash", "") == config_hash:
        return legacy_path
    return scoped_path


def snapshot_prepared_query(prepared: Dict[str, Any]) -> Dict[str, Any]:
    """Return a JSON-safe prepared-query snapshot for an exact resume.

    Final-answer retrieval is normally read-only, but it is not necessarily
    deterministic (for example, value-aware retrieval can explore). Keeping
    the prepared response metadata with the submitted request lets a resumed
    evaluator finalize the same prompt without running retrieval again.
    """
    try:
        serialized = json.dumps(prepared, ensure_ascii=False, default=str)
        snapshot = json.loads(serialized)
    except (TypeError, ValueError) as exc:
        raise VertexBatchError("Prepared batch query cannot be saved for resume.") from exc
    if not isinstance(snapshot, dict):
        raise VertexBatchError("Prepared batch query must be a dictionary.")
    return snapshot


def restore_prepared_query(request: BatchChatRequest) -> Optional[Dict[str, Any]]:
    """Return a saved prepared query only when it matches the submitted prompt."""
    prepared = request.metadata.get(PREPARED_QUERY_METADATA_KEY)
    if not isinstance(prepared, dict) or prepared.get("messages") != request.messages:
        return None
    return prepared


class VertexBatchClient:
    """Submit and collect Cloud Storage-backed Vertex Gemini batch jobs."""

    def __init__(
        self,
        *,
        model: str,
        project: str,
        location: str,
        credentials: Any,
        gcs_uri: str,
        manifest_path: Path,
        wait: bool,
        config_hash: str = "",
        poll_interval: int = 30,
        genai_client: Any = None,
        direct_client: Any = None,
        storage_client: Any = None,
        credential_client: Any = None,
        progress_callback: Optional[BatchProgressCallback] = None,
    ):
        self.model = model
        self.project = project
        self.location = location
        self.credentials = credentials
        self.gcs_uri = gcs_uri.rstrip("/")
        self.bucket_name, self.prefix = _parse_gcs_uri(self.gcs_uri)
        self.manifest_path = Path(manifest_path)
        self.wait_for_completion = wait
        self.config_hash = config_hash
        self.poll_interval = poll_interval
        self._genai_client = genai_client
        self._direct_client = direct_client
        self._storage_client = storage_client
        self._credential_client = credential_client
        self._progress_callback = progress_callback

    @classmethod
    def from_gemini_client(
        cls,
        client: Any,
        *,
        gcs_uri: Optional[str],
        manifest_path: Path,
        wait: bool,
        config_hash: str = "",
        poll_interval: int = 30,
        progress_callback: Optional[BatchProgressCallback] = None,
    ) -> "VertexBatchClient":
        """Create a batch transport from the repository's managed Gemini client."""
        from utils.llm_client import GeminiHybridClient, GeminiVertexClient

        if isinstance(client, GeminiHybridClient):
            client = client.vertex_client
        if not isinstance(client, GeminiVertexClient):
            raise VertexBatchError(
                "Vertex batch API requires provider: vertex or gemini."
            )
        configured_uri = gcs_uri or os.environ.get("GOOGLE_BATCH_GCS_URI")
        if not configured_uri:
            raise VertexBatchError(
                "Batch API requires --batch-gcs-uri or GOOGLE_BATCH_GCS_URI."
            )
        return cls(
            model=client.model,
            project=client.project,
            location=client.location,
            credentials=client.credentials,
            gcs_uri=configured_uri,
            manifest_path=manifest_path,
            wait=wait,
            config_hash=config_hash,
            poll_interval=poll_interval,
            # The normal generation client can use the SDK's beta default.
            # Batch jobs use a separate v1 client below while retaining this
            # exact service-account credential, project, and location.
            genai_client=None,
            direct_client=client,
            credential_client=client,
            progress_callback=progress_callback,
        )

    def _run_with_vertex_credentials(self, operation_name: str, operation):
        """Run batch control operations through the Vertex account pool."""
        if self._credential_client is None:
            return operation(self.credentials, self.project)

        def run_for_account(client, credentials, project):
            self.credentials = credentials
            self.project = project
            return operation(credentials, project)

        return self._credential_client._run_with_service_account_rotation(
            run_for_account,
            operation_name,
        )

    @staticmethod
    def _new_storage_client(credentials: Any, project: str):
        try:
            from google.cloud import storage
        except ImportError as exc:
            raise ImportError(
                "Vertex batch support requires google-cloud-storage. "
                "Install dependencies with: pip install -r requirements.txt"
            ) from exc
        return storage.Client(project=project, credentials=credentials)

    def _new_batch_genai_client(self, credentials: Any, project: str):
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise ImportError(
                "Vertex batch support requires google-genai. "
                "Install dependencies with: pip install -r requirements.txt"
            ) from exc
        return genai.Client(
            enterprise=True,
            credentials=credentials,
            project=project,
            location=self.location,
            http_options=types.HttpOptions(api_version="v1"),
        )

    def _progress(self, message: str) -> None:
        """Report batch lifecycle progress without coupling this utility to logging."""
        if self._progress_callback is not None:
            self._progress_callback(f"[Vertex Batch] {message}")

    @property
    def storage_client(self):
        if self._storage_client is None:
            try:
                from google.cloud import storage
            except ImportError as exc:
                raise ImportError(
                    "Vertex batch support requires google-cloud-storage. "
                    "Install dependencies with: pip install -r requirements.txt"
                ) from exc
            self._storage_client = storage.Client(
                project=self.project,
                credentials=self.credentials,
            )
        return self._storage_client

    @property
    def genai_client(self):
        if self._genai_client is None:
            try:
                from google import genai
                from google.genai import types
            except ImportError as exc:
                raise ImportError(
                    "Vertex batch support requires google-genai. "
                    "Install dependencies with: pip install -r requirements.txt"
                ) from exc
            self._genai_client = genai.Client(
                enterprise=True,
                credentials=self.credentials,
                project=self.project,
                location=self.location,
                http_options=types.HttpOptions(api_version="v1"),
            )
        return self._genai_client

    def _load_manifest(self) -> Dict[str, Any]:
        if not self.manifest_path.exists():
            return {
                "version": MANIFEST_VERSION,
                "created_at": _utc_now(),
                "model": self.model,
                "project": self.project,
                "location": self.location,
                "gcs_uri": self.gcs_uri,
                "config_hash": self.config_hash,
                # Do not derive a GCS path from a user-provided method or model
                # name; this also prevents concurrent runs from colliding.
                "run_id": uuid.uuid4().hex,
                "jobs": {},
            }
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        if manifest.get("version") != MANIFEST_VERSION:
            raise VertexBatchError(
                f"Unsupported batch manifest version in {self.manifest_path}."
            )
        if manifest.get("model") != self.model or manifest.get("gcs_uri") != self.gcs_uri:
            raise VertexBatchError(
                "Batch manifest does not match the requested model or Cloud Storage prefix."
            )
        if manifest.get("config_hash", "") != self.config_hash:
            raise VertexBatchError(
                "Batch manifest does not match this evaluator configuration. "
                "Use the original configuration to resume it or choose a new output directory."
            )
        return manifest

    def get_saved_request(
        self,
        stage: str,
        request_id: str,
    ) -> Optional[BatchChatRequest]:
        """Return an already-submitted request without contacting Vertex.

        A few legacy adapters include a display timestamp in an otherwise
        immutable prompt.  Reading its local manifest before re-preparing the
        stage keeps that prompt byte-for-byte stable across submit/resume while
        retaining the normal fingerprint protection for all other fields.
        """
        manifest = self._load_manifest()
        job_entry = manifest.get("jobs", {}).get(stage)
        if not isinstance(job_entry, dict):
            return None
        for request in job_entry.get("requests", []):
            if request.get("request_id") == request_id:
                return BatchChatRequest.from_manifest_dict(request)
        return None

    def get_saved_requests(self, stage: str) -> List[BatchChatRequest]:
        """Return every request already bound to a submitted stage."""
        manifest = self._load_manifest()
        job_entry = manifest.get("jobs", {}).get(stage)
        if not isinstance(job_entry, dict):
            return []
        return [
            BatchChatRequest.from_manifest_dict(request)
            for request in job_entry.get("requests", [])
        ]

    def has_stage(self, stage: str) -> bool:
        """Return whether the manifest already contains ``stage``."""
        manifest = self._load_manifest()
        return isinstance(manifest.get("jobs", {}).get(stage), dict)

    def _save_manifest(self, manifest: Dict[str, Any]) -> None:
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest["updated_at"] = _utc_now()
        with self.manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)

    def _stage_paths(self, stage: str, run_id: str) -> Tuple[str, str]:
        stage_id = re.sub(r"[^a-zA-Z0-9_.-]", "-", stage)
        root = f"{self.prefix}/{run_id}/{stage_id}"
        return f"gs://{self.bucket_name}/{root}/input.jsonl", f"gs://{self.bucket_name}/{root}/output"

    def _upload_jsonl(self, uri: str, requests: Iterable[BatchChatRequest]) -> int:
        bucket_name, object_name = _parse_gcs_uri(uri)
        body = "\n".join(
            json.dumps(
                request.to_vertex_record(
                    model=self.model,
                    reasoning_effort=getattr(self._direct_client, "reasoning_effort", None),
                ),
                ensure_ascii=False,
                separators=(",", ":"),
            )
            for request in requests
        ) + "\n"
        self._progress(
            f"Uploading {len(body.encode('utf-8')):,} bytes of JSONL to {uri}."
        )
        def upload(credentials, project):
            storage_client = (
                self._storage_client
                if self._storage_client is not None
                else self._new_storage_client(credentials, project)
            )
            storage_client.bucket(bucket_name).blob(object_name).upload_from_string(
                body,
                content_type="application/jsonl; charset=utf-8",
            )

        self._run_with_vertex_credentials("batch input upload", upload)
        self._progress(f"Input upload completed: {uri}.")
        return len(body.encode("utf-8"))

    def _submit(
        self,
        stage: str,
        requests: List[BatchChatRequest],
        run_id: str,
    ) -> Dict[str, Any]:
        if not requests:
            return {}
        input_uri, output_uri = self._stage_paths(stage, run_id)
        self._progress(
            f"Stage '{stage}': preparing {len(requests):,} request(s); "
            f"output prefix: {output_uri}."
        )
        self._upload_jsonl(input_uri, requests)
        try:
            from google.genai import types
        except ImportError as exc:  # pragma: no cover - checked by constructor in real runs
            raise ImportError("google-genai is required for Vertex batch jobs") from exc

        self._progress(
            f"Stage '{stage}': creating Vertex job for model {self.model}."
        )
        def create_job(credentials, project):
            genai_client = (
                self._genai_client
                if self._genai_client is not None
                else self._new_batch_genai_client(credentials, project)
            )
            return genai_client.batches.create(
                model=self.model,
                src=input_uri,
                config=types.CreateBatchJobConfig(dest=output_uri),
            )

        job = self._run_with_vertex_credentials("batch job submission", create_job)
        state = _state_name(getattr(job, "state", None)) or "unknown"
        self._progress(
            f"Stage '{stage}': submitted {job.name} (state: {state}; "
            f"requests: {len(requests):,})."
        )
        return {
            "stage": stage,
            "job_name": job.name,
            "state": state,
            "project": self.project,
            "input_uri": input_uri,
            "output_uri": output_uri,
            "submitted_at": _utc_now(),
            "requests": [request.to_manifest_dict() for request in requests],
            "request_fingerprint": self._request_fingerprint(requests),
            "responses": {},
        }

    def _run_direct(
        self,
        stage: str,
        requests: List[BatchChatRequest],
    ) -> Dict[str, BatchChatResponse]:
        """Run a small stage through the normal real-time Vertex AI client."""
        if self._direct_client is None:
            raise VertexBatchError("Direct Vertex AI fallback is unavailable.")

        self._progress(
            f"Stage '{stage}': only {len(requests):,} request(s); "
            "using direct Vertex AI calls instead of creating a batch job."
        )
        responses: Dict[str, BatchChatResponse] = {}
        for request in requests:
            kwargs: Dict[str, Any] = {}
            if request.response_format is not None:
                kwargs["response_format"] = request.response_format
                response_schema = (
                    request.response_format.get("response_json_schema")
                    or request.response_format.get("response_schema")
                )
                if response_schema is not None:
                    kwargs["response_json_schema"] = response_schema

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
                    status=truncate_error_message(
                        f"{type(exc).__name__}: {exc}"
                    ),
                    duration_seconds=time.perf_counter() - started_at,
                )

        failed = sum(1 for response in responses.values() if response.status)
        self._progress(
            f"Stage '{stage}': completed {len(responses) - failed:,}/{len(requests):,} "
            f"direct request(s); {failed:,} failed."
        )
        return responses

    def _get_job(self, job_name: str, project: Optional[str] = None):
        if (
            self._credential_client is not None
            and project
            and hasattr(self._credential_client, "_select_project")
        ):
            self._credential_client._select_project(project)

        def get_job(credentials, project):
            genai_client = (
                self._genai_client
                if self._genai_client is not None
                else self._new_batch_genai_client(credentials, project)
            )
            return genai_client.batches.get(name=job_name)

        return self._run_with_vertex_credentials("batch job polling", get_job)

    def _wait_for_job(self, job_entry: Dict[str, Any]):
        job = self._get_job(job_entry["job_name"], job_entry.get("project"))
        state = _state_name(getattr(job, "state", None))
        attempts = 0
        self._progress(
            f"Stage '{job_entry['stage']}': {job_entry['job_name']} is {state or 'unknown'}; "
            f"polling every {self.poll_interval}s."
        )
        while state not in TERMINAL_STATES:
            time.sleep(self.poll_interval)
            attempts += 1
            job = self._get_job(job_entry["job_name"], job_entry.get("project"))
            polled_state = _state_name(getattr(job, "state", None))
            if polled_state != state:
                self._progress(
                    f"Stage '{job_entry['stage']}': poll {attempts}; "
                    f"{job_entry['job_name']} is {polled_state or 'unknown'}."
                )
            state = polled_state
        job_entry["state"] = state
        job_entry["completed_at"] = _utc_now()
        self._progress(
            f"Stage '{job_entry['stage']}': {job_entry['job_name']} reached terminal state {state}."
        )
        return job

    def _list_output_rows(self, output_uri: str) -> Iterable[Dict[str, Any]]:
        bucket_name, prefix = _parse_gcs_uri(output_uri)

        def download_rows(credentials, project):
            storage_client = (
                self._storage_client
                if self._storage_client is not None
                else self._new_storage_client(credentials, project)
            )
            rows = []
            for blob in storage_client.list_blobs(bucket_name, prefix=prefix):
                if not blob.name.endswith(".jsonl"):
                    continue
                text = blob.download_as_text(encoding="utf-8")
                rows.extend(
                    json.loads(line)
                    for line in text.splitlines()
                    if line.strip()
                )
            return rows

        yield from self._run_with_vertex_credentials("batch output download", download_rows)

    @staticmethod
    def _request_correlation_key(request: Dict[str, Any]) -> str:
        """Return the stable text-message shape echoed by Vertex GCS output."""
        contents = request.get("contents")
        if not isinstance(contents, list):
            raise VertexBatchError("Vertex batch output row does not contain request.contents.")

        messages = []
        for content in contents:
            if not isinstance(content, dict):
                raise VertexBatchError("Vertex batch output contains an invalid request content.")
            parts = content.get("parts")
            if not isinstance(parts, list):
                raise VertexBatchError("Vertex batch output contains an invalid request part list.")
            text_parts = []
            for part in parts:
                if not isinstance(part, dict) or "text" not in part:
                    raise VertexBatchError(
                        "Vertex batch output contains a non-text request part that cannot be correlated."
                    )
                text = part["text"]
                if not isinstance(text, str):
                    raise VertexBatchError("Vertex batch output contains a non-string request text part.")
                text_parts.append(text)
            messages.append({"role": content.get("role", ""), "parts": text_parts})

        return json.dumps(messages, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    @classmethod
    def _validate_request_correlation(cls, requests: List[BatchChatRequest]) -> Dict[str, str]:
        """Map echoed request content to one local ID before a job is submitted.

        Cloud Storage output does not include an application-defined request ID.
        Identical message content would therefore be ambiguous if output order
        changes, so fail before submitting rather than misassigning results.
        """
        request_ids_by_key: Dict[str, str] = {}
        for request in requests:
            key = cls._request_correlation_key(request.to_vertex_request())
            previous_id = request_ids_by_key.get(key)
            if previous_id is not None:
                raise VertexBatchError(
                    "Vertex batch stage contains duplicate message content that cannot be "
                    f"safely correlated ({previous_id!r} and {request.request_id!r})."
                )
            request_ids_by_key[key] = request.request_id
        return request_ids_by_key

    @staticmethod
    def _parse_row(row: Dict[str, Any], request_id: str) -> BatchChatResponse:
        status_value = row.get("status", "") or ""
        if isinstance(status_value, dict):
            # Vertex emits a structured status only for errors. Some emulators
            # include a successful zero code, which is not an error.
            status = "" if str(status_value.get("code", 0)) == "0" else json.dumps(status_value, sort_keys=True)
        else:
            status = str(status_value)
        response = row.get("response") or {}
        usage = response.get("usageMetadata") or response.get("usage_metadata") or {}
        candidates = response.get("candidates") or []
        content = ""
        if candidates:
            parts = (candidates[0].get("content") or {}).get("parts") or []
            content = "".join(part.get("text", "") for part in parts)
        input_tokens, output_tokens, visible_output_tokens, thinking_tokens = (
            extract_usage_token_counts(usage)
        )
        return BatchChatResponse(
            request_id=request_id,
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            visible_output_tokens=visible_output_tokens,
            thinking_tokens=thinking_tokens,
            status=status,
            raw_response=row,
        )

    def _collect(self, job_entry: Dict[str, Any]) -> Dict[str, BatchChatResponse]:
        if self._credential_client is not None and hasattr(
            self._credential_client,
            "_select_project",
        ):
            self._credential_client._select_project(job_entry.get("project"))
        requests = [
            BatchChatRequest.from_manifest_dict(request)
            for request in job_entry["requests"]
        ]
        request_ids_by_key = self._validate_request_correlation(requests)
        responses: Dict[str, BatchChatResponse] = {}
        self._progress(
            f"Stage '{job_entry['stage']}': reading Vertex output for {len(requests):,} request(s)."
        )
        for row in self._list_output_rows(job_entry["output_uri"]):
            request = row.get("request")
            if not isinstance(request, dict):
                raise VertexBatchError("Vertex batch output row does not contain the echoed request.")
            key = self._request_correlation_key(request)
            request_id = request_ids_by_key.get(key)
            if request_id is None:
                raise VertexBatchError("Vertex batch output contains an unknown echoed request.")
            if request_id in responses:
                raise VertexBatchError("Vertex batch output contains a duplicate echoed request.")
            parsed = self._parse_row(row, request_id)
            responses[parsed.request_id] = parsed

        expected_ids = {request.request_id for request in requests}

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
        job_entry["missing_request_ids"] = sorted(expected_ids - set(responses))
        failed = sum(1 for response in responses.values() if response.status)
        self._progress(
            f"Stage '{job_entry['stage']}': collected {len(responses):,}/{len(requests):,} output row(s); "
            f"{failed:,} failed, {len(job_entry['missing_request_ids']):,} missing."
        )
        return responses

    @staticmethod
    def _request_fingerprint(requests: List[BatchChatRequest]) -> str:
        """Bind a resumable stage to exactly the prompts that were submitted."""
        encoded = json.dumps(
            [request.to_manifest_dict() for request in requests],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _wait_or_raise_pending(self, stage: str, job_entry: Dict[str, Any], manifest: Dict[str, Any]):
        job = self._get_job(job_entry["job_name"], job_entry.get("project"))
        job_entry["state"] = _state_name(getattr(job, "state", None))
        if job_entry["state"] in TERMINAL_STATES:
            self._progress(
                f"Stage '{stage}': existing job {job_entry['job_name']} is already "
                f"{job_entry['state']}; collecting its output."
            )
            return
        if not self.wait_for_completion:
            self._save_manifest(manifest)
            self._progress(
                f"Stage '{stage}': job {job_entry['job_name']} is {job_entry['state'] or 'unknown'}. "
                "Manifest saved; exiting without waiting."
            )
            raise VertexBatchPending(stage, job_entry["job_name"], self.manifest_path)
        self._wait_for_job(job_entry)

    @staticmethod
    def _unresolved_requests(
        requests: List[BatchChatRequest],
        responses: Dict[str, BatchChatResponse],
    ) -> List[BatchChatRequest]:
        return [
            request
            for request in requests
            if request.request_id not in responses
            or bool(responses[request.request_id].status)
            or not responses[request.request_id].content
        ]

    def run_stage(
        self,
        stage: str,
        requests: List[BatchChatRequest],
    ) -> Dict[str, BatchChatResponse]:
        """Run one deterministic stage through Vertex Batch AI.

        Every non-empty stage submitted through this explicit batch client uses
        a batch job. In submit-only batch mode this raises
        :class:`VertexBatchPending` for later ``--resume`` collection.
        """
        if not requests:
            return {}
        # Validate before writing GCS input or creating a billable job.
        self._validate_request_correlation(requests)
        manifest = self._load_manifest()
        jobs = manifest.setdefault("jobs", {})
        job_entry = jobs.get(stage)

        if job_entry is None:
            job_entry = self._submit(stage, requests, manifest["run_id"])
            manifest["project"] = self.project
            jobs[stage] = job_entry
            self._save_manifest(manifest)
        elif job_entry.get("request_fingerprint") != self._request_fingerprint(requests):
            raise VertexBatchError(
                f"Batch stage '{stage}' was already submitted with different requests. "
                "Use the matching checkpoint/configuration or a new output directory."
            )
        else:
            self._progress(
                f"Stage '{stage}': resuming {job_entry['job_name']} "
                f"(last saved state: {job_entry.get('state') or 'unknown'}; "
                f"requests: {len(requests):,})."
            )

        self._wait_or_raise_pending(stage, job_entry, manifest)
        responses = self._collect(job_entry)

        # Vertex can finish a failed job with usable partial output. Preserve
        # those rows and submit only missing/failed IDs, never duplicate work.
        retries = job_entry.setdefault("retries", [])
        # First collect a retry that a previous --resume invocation submitted.
        for retry_entry in retries:
            self._wait_or_raise_pending(retry_entry["stage"], retry_entry, manifest)
            responses.update(self._collect(retry_entry))

        unresolved = self._unresolved_requests(requests, responses)
        while unresolved and len(retries) < 3:
            retry_stage = f"{stage}-retry-{len(retries) + 1}"
            self._progress(
                f"Stage '{stage}': retrying {len(unresolved):,} incomplete request(s) "
                f"as '{retry_stage}'."
            )
            retry_job = self._submit(retry_stage, unresolved, manifest["run_id"])
            retry_entry = {
                "stage": retry_stage,
                "job_name": retry_job["job_name"],
                "state": retry_job["state"],
                "project": retry_job["project"],
                "input_uri": retry_job["input_uri"],
                "output_uri": retry_job["output_uri"],
                "submitted_at": retry_job["submitted_at"],
                "request_ids": sorted(request.request_id for request in unresolved),
                "requests": retry_job["requests"],
                "request_fingerprint": retry_job["request_fingerprint"],
                "responses": {},
            }
            retries.append(retry_entry)
            self._save_manifest(manifest)

            self._wait_or_raise_pending(retry_stage, retry_entry, manifest)
            responses.update(self._collect(retry_entry))
            unresolved = self._unresolved_requests(requests, responses)

        job_entry["missing_request_ids"] = sorted(
            request.request_id for request in self._unresolved_requests(requests, responses)
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
        self._save_manifest(manifest)
        self._progress(
            f"Stage '{stage}': finalized {len(responses):,}/{len(requests):,} request(s); "
            f"{len(job_entry['missing_request_ids']):,} remain unresolved."
        )

        # Attribute completed model tokens to their original evaluator phase.
        by_id = {request.request_id: request for request in requests}
        for request_id, response in responses.items():
            request = by_id.get(request_id)
            if request is not None:
                get_usage_tracker().set_phase(request.phase)
                get_usage_tracker().record(response.to_llm_response(self.model))
        return responses

    def has_unfinished_stage(self) -> Optional[Dict[str, Any]]:
        """Return the first unfinished manifest entry, if any."""
        manifest = self._load_manifest()
        for job in manifest.get("jobs", {}).values():
            if job.get("state") not in TERMINAL_STATES:
                return job
            for retry in job.get("retries", []):
                if retry.get("state") not in TERMINAL_STATES:
                    return retry
        return None


__all__ = [
    "BatchChatRequest",
    "BatchChatResponse",
    "VertexBatchClient",
    "VertexBatchError",
    "VertexBatchPending",
    "PREPARED_QUERY_METADATA_KEY",
    "make_request_id",
    "restore_prepared_query",
    "scoped_manifest_path",
    "snapshot_prepared_query",
]
