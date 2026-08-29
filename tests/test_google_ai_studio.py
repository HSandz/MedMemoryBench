"""Offline tests for Google AI Studio provider selection and key rotation."""

import logging
import os
from types import SimpleNamespace
from pathlib import Path

import pytest

from src.agent import AgentManager
from src.config import APIConfig, DatasetConfig, MethodConfig, ModelConfig
from src.evaluator import Evaluator
from utils import llm_client


class ResourceExhausted(Exception):
    """Match the Google SDK's retryable quota exception name."""


class AIStudioClientError(Exception):
    """Expose the structured message field used by google-genai errors."""

    def __init__(self, message: str, status_code: int = 429):
        self.message = message
        self.status_code = status_code
        super().__init__(message)


class _Models:
    def __init__(self, api_key: str, calls: list, operation):
        self.api_key = api_key
        self.calls = calls
        self.operation = operation

    def generate_content(self, **kwargs):
        self.calls.append((self.api_key, kwargs))
        return self.operation(self.api_key)


class _FakeClient:
    def __init__(self, api_key: str, calls: list, operation):
        self.models = _Models(api_key, calls, operation)


class _VertexStub:
    def __init__(self, client):
        self.model = "gemini-2.5-flash"
        self.client = client
        self.project = "test-project"
        self.location = "global"
        self.credentials = object()
        self.service_account_file = "/tmp/service-account.json"


def _success(text: str = "complete"):
    return SimpleNamespace(
        text=text,
        usage_metadata=SimpleNamespace(prompt_token_count=7, candidates_token_count=3),
    )


def _configure_fake_vertex_accounts(monkeypatch, files, calls, operation):
    class Credentials:
        @staticmethod
        def from_service_account_file(path, scopes):
            return SimpleNamespace(
                label=Path(path).stem,
                project_id=f"project-{Path(path).stem}",
            )

    def client_factory(*, api_key=None, enterprise=False, credentials=None, **kwargs):
        label = api_key if api_key is not None else credentials.label
        return _FakeClient(label, calls, operation)

    for path in files:
        path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        "google.oauth2.service_account.Credentials",
        Credentials,
    )
    monkeypatch.setattr("google.genai.Client", client_factory)


def test_ai_studio_rotates_after_configured_consecutive_failures(monkeypatch):
    calls = []
    delays = []

    def operation(api_key: str):
        if api_key == "key-one":
            raise ResourceExhausted("429 quota exceeded")
        return _success()

    monkeypatch.setattr(
        "google.genai.Client",
        lambda *, api_key: _FakeClient(api_key, calls, operation),
    )
    monkeypatch.setattr(llm_client.time, "sleep", delays.append)

    client = llm_client.create_llm_client(
        provider="ai_studio",
        model="gemini-2.5-flash",
        api_keys=["key-one", "key-two"],
        key_failure_threshold=5,
    )
    response = client.chat([{"role": "user", "content": "hello"}])

    assert response.content == "complete"
    assert [api_key for api_key, _ in calls] == ["key-one"] * 5 + ["key-two"]
    assert delays == [1.0, 2.0, 4.0, 8.0, llm_client.AI_STUDIO_KEY_ROTATION_DELAY_SECONDS]
    assert client.active_key_index == 1


def test_ai_studio_defaults_to_sequential_key_rotation(monkeypatch):
    calls = []

    monkeypatch.setattr(
        "google.genai.Client",
        lambda *, api_key: _FakeClient(api_key, calls, lambda _: _success()),
    )
    client = llm_client.GeminiAIStudioClient(api_keys=["key-one", "key-two"])

    for _ in range(3):
        assert client.chat([{"role": "user", "content": "hello"}]).content == "complete"

    assert [api_key for api_key, _ in calls] == ["key-one"] * 3
    assert client.active_key_index == 0


def test_ai_studio_round_robin_rotates_after_configured_successful_calls(monkeypatch):
    calls = []

    monkeypatch.setattr(
        "google.genai.Client",
        lambda *, api_key: _FakeClient(api_key, calls, lambda _: _success()),
    )
    monkeypatch.setenv("GOOGLE_AI_STUDIO_KEY_ROTATION_MODE", "round_robin")
    monkeypatch.setenv("GOOGLE_AI_STUDIO_ROUND_ROBIN_CALLS_PER_KEY", "2")
    client = llm_client.GeminiAIStudioClient(api_keys=["key-one", "key-two"])

    for _ in range(5):
        assert client.chat([{"role": "user", "content": "hello"}]).content == "complete"

    assert [api_key for api_key, _ in calls] == [
        "key-one",
        "key-one",
        "key-two",
        "key-two",
        "key-one",
    ]
    assert client.active_key_index == 0


@pytest.mark.parametrize(
    ("environment", "value", "error"),
    [
        ("GOOGLE_AI_STUDIO_KEY_ROTATION_MODE", "random", "KEY_ROTATION_MODE"),
        ("GOOGLE_AI_STUDIO_ROUND_ROBIN_CALLS_PER_KEY", "0", "CALLS_PER_KEY"),
    ],
)
def test_ai_studio_rejects_invalid_round_robin_settings(
    monkeypatch, environment, value, error
):
    monkeypatch.setenv("GOOGLE_AI_STUDIO_KEY_ROTATION_MODE", "round_robin")
    monkeypatch.setenv(environment, value)

    with pytest.raises(ValueError, match=error):
        llm_client.GeminiAIStudioClient(api_keys=["key-one", "key-two"])


def test_ai_studio_rotation_logs_a_bounded_error_and_waits(monkeypatch, caplog):
    calls = []
    delays = []
    error_message = (
        "429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': "
        "'You exceeded your current quota, please check your plan and billing details. "
        "For more information, see https://example.com/rate-limit.'}}"
    )

    def operation(api_key: str):
        if api_key == "key-one":
            raise ResourceExhausted(error_message)
        return _success()

    monkeypatch.setattr(
        "google.genai.Client",
        lambda *, api_key: _FakeClient(api_key, calls, operation),
    )
    monkeypatch.setattr(llm_client.time, "sleep", delays.append)
    caplog.set_level(logging.WARNING, logger=llm_client.__name__)
    client = llm_client.GeminiAIStudioClient(
        api_keys=["key-one", "key-two"],
        key_failure_threshold=1,
    )

    assert client.chat([{"role": "user", "content": "hello"}]).content == "complete"
    rotation_log = next(
        record.getMessage()
        for record in caplog.records
        if record.getMessage().endswith("| rotating")
    )
    assert "\n" not in rotation_log
    assert rotation_log.startswith(
        "1/2 | 429 | You exceeded your current quota, please check your plan and billing details."
    )
    assert "RESOURCE_EXHAUSTED" not in rotation_log
    assert "code" not in rotation_log
    assert rotation_log.endswith("| rotating")
    assert delays == [llm_client.AI_STUDIO_KEY_ROTATION_DELAY_SECONDS]


def test_ai_studio_quota_message_rotates_immediately(monkeypatch):
    calls = []

    def operation(api_key: str):
        if api_key == "key-one":
            raise AIStudioClientError(
                "You exceeded your current quota, please check your plan and billing details."
            )
        return _success()

    monkeypatch.setattr(
        "google.genai.Client",
        lambda *, api_key: _FakeClient(api_key, calls, operation),
    )
    monkeypatch.setattr(llm_client.time, "sleep", lambda delay: None)
    client = llm_client.GeminiAIStudioClient(
        api_keys=["key-one", "key-two"],
        key_failure_threshold=10,
        resource_exhausted_retries=4,
    )

    assert client.chat([{"role": "user", "content": "hello"}]).content == "complete"
    assert [api_key for api_key, _ in calls] == ["key-one", "key-two"]


def test_ai_studio_retires_permanently_invalid_key_from_source_file(tmp_path, monkeypatch):
    calls = []
    key_file = tmp_path / "keys.txt"
    key_file.write_text("key-one\n# keep this comment\nkey-two\n", encoding="utf-8")
    project_root = Path(llm_client.__file__).resolve().parent.parent
    relative_path = os.path.relpath(key_file, project_root)

    def operation(api_key: str):
        if api_key == "key-one":
            raise AIStudioClientError(
                "The bound service account is deleted or disabled. "
                "The service account bound to the API key must be active.",
                status_code=401,
            )
        return _success()

    monkeypatch.setenv("GOOGLE_AI_STUDIO_API_KEYS_FILE", relative_path)
    monkeypatch.setattr(
        "google.genai.Client",
        lambda *, api_key: _FakeClient(api_key, calls, operation),
    )
    client = llm_client.GeminiAIStudioClient()

    assert client.chat([{"role": "user", "content": "hello"}]).content == "complete"
    assert [api_key for api_key, _ in calls] == ["key-one", "key-two"]
    assert client.api_keys == ["key-two"]
    assert key_file.read_text(encoding="utf-8") == "# keep this comment\nkey-two\n"


def test_ai_studio_resource_exhausted_message_retries_current_key(monkeypatch):
    calls = []

    def operation(api_key: str):
        if api_key == "key-one":
            raise AIStudioClientError("Resource has been exhausted (e.g. check quota).")
        return _success()

    monkeypatch.setattr(
        "google.genai.Client",
        lambda *, api_key: _FakeClient(api_key, calls, operation),
    )
    monkeypatch.setattr(llm_client.time, "sleep", lambda delay: None)
    monkeypatch.setenv("GOOGLE_AI_STUDIO_RESOURCE_EXHAUSTED_RETRIES", "2")
    client = llm_client.GeminiAIStudioClient(
        api_keys=["key-one", "key-two"],
        key_failure_threshold=1,
    )

    assert client.chat([{"role": "user", "content": "hello"}]).content == "complete"
    assert [api_key for api_key, _ in calls] == ["key-one", "key-one", "key-one", "key-two"]


@pytest.mark.parametrize("retries", ("-1", "invalid"))
def test_ai_studio_rejects_invalid_resource_exhausted_retry_counts(monkeypatch, retries):
    monkeypatch.setenv("GOOGLE_AI_STUDIO_RESOURCE_EXHAUSTED_RETRIES", retries)

    with pytest.raises(ValueError, match="must be a non-negative integer"):
        llm_client._resolve_ai_studio_resource_exhausted_retries()


def test_ai_studio_repeats_the_configured_number_of_key_rotation_rounds(monkeypatch):
    calls = []

    def operation(api_key: str):
        raise ResourceExhausted("429 quota exceeded")

    monkeypatch.setattr(
        "google.genai.Client",
        lambda *, api_key: _FakeClient(api_key, calls, operation),
    )
    monkeypatch.setattr(llm_client.time, "sleep", lambda delay: None)
    monkeypatch.setenv("GOOGLE_AI_STUDIO_MAX_ROTATION_ROUNDS", "2")
    client = llm_client.GeminiAIStudioClient(
        api_keys=["key-one", "key-two"],
        key_failure_threshold=1,
    )

    with pytest.raises(llm_client.LLMRetryExhaustedError) as exc_info:
        client.chat([{"role": "user", "content": "hello"}])

    assert [api_key for api_key, _ in calls] == [
        "key-one",
        "key-two",
        "key-one",
        "key-two",
    ]
    assert exc_info.value.attempts == 4


def test_ai_studio_allows_unlimited_key_rotation_rounds(monkeypatch):
    calls = []

    def operation(api_key: str):
        if len(calls) < 5:
            raise ResourceExhausted("429 quota exceeded")
        return _success()

    monkeypatch.setattr(
        "google.genai.Client",
        lambda *, api_key: _FakeClient(api_key, calls, operation),
    )
    monkeypatch.setattr(llm_client.time, "sleep", lambda delay: None)
    monkeypatch.setenv("GOOGLE_AI_STUDIO_MAX_ROTATION_ROUNDS", "-1")
    client = llm_client.GeminiAIStudioClient(
        api_keys=["key-one", "key-two"],
        key_failure_threshold=1,
    )

    assert client.chat([{"role": "user", "content": "hello"}]).content == "complete"
    assert [api_key for api_key, _ in calls] == [
        "key-one",
        "key-two",
        "key-one",
        "key-two",
        "key-one",
    ]


@pytest.mark.parametrize("rounds", ("0", "-2", "invalid"))
def test_ai_studio_rejects_invalid_rotation_round_counts(monkeypatch, rounds):
    monkeypatch.setenv("GOOGLE_AI_STUDIO_MAX_ROTATION_ROUNDS", rounds)

    with pytest.raises(ValueError, match="must be -1 or a positive integer"):
        llm_client._resolve_ai_studio_max_rotation_rounds()


@pytest.mark.parametrize(
    ("model", "effort", "field", "expected"),
    [
        ("gemini-3-pro-preview", "high", "thinking_level", "HIGH"),
        ("gemini-2.5-flash", 1024, "thinking_budget", 1024),
    ],
)
def test_gemini_reasoning_effort_maps_to_thinking_config(model, effort, field, expected):
    client = object.__new__(llm_client.BaseGeminiClient)
    client.model, client.temperature, client.max_tokens = model, 1.0, 100
    client.reasoning_effort = effort
    captured = {}

    class Models:
        def generate_content(self, **kwargs):
            captured.update(kwargs)
            return _success("ok")

    client.client = SimpleNamespace(models=Models())
    client._chat_once([{"role": "user", "content": "hello"}])
    thinking = captured["config"].thinking_config
    assert getattr(thinking, field) == expected


def test_vertex_resolves_relative_service_account_list_and_rotates(tmp_path, monkeypatch):
    calls = []
    delays = []
    files = [tmp_path / "service-one.json", tmp_path / "service-two.json"]

    def operation(account: str):
        if account == "service-one":
            raise ResourceExhausted("429 quota exceeded")
        return _success("vertex-complete")

    _configure_fake_vertex_accounts(monkeypatch, files, calls, operation)
    monkeypatch.setattr(llm_client.time, "sleep", delays.append)
    project_root = Path(llm_client.__file__).resolve().parent.parent
    relative_files = [os.path.relpath(path, project_root) for path in files]
    monkeypatch.setenv("GOOGLE_SERVICE_ACCOUNT_FILE", ",".join(relative_files))

    client = llm_client.GeminiVertexClient(
        service_account_failure_threshold=3,
    )
    response = client.chat([{"role": "user", "content": "hello"}])

    assert response.content == "vertex-complete"
    assert client.service_account_files == [path.resolve() for path in files]
    assert [account for account, _ in calls] == ["service-one"] * 3 + ["service-two"]
    assert delays == [1.0, 2.0]
    assert client.active_service_account_index == 1
    assert client.project == "project-service-two"


def test_vertex_rotation_uses_separate_failure_type_pools(tmp_path, monkeypatch):
    calls = []
    delays = []
    attempts = 0
    files = [tmp_path / "service-one.json", tmp_path / "service-two.json"]

    def operation(account: str):
        nonlocal attempts
        attempts += 1
        if account == "service-one":
            if attempts == 3:
                return _success("")
            raise ResourceExhausted("429 quota exceeded")
        return _success("vertex-complete")

    _configure_fake_vertex_accounts(monkeypatch, files, calls, operation)
    monkeypatch.setattr(llm_client.time, "sleep", delays.append)
    client = llm_client.GeminiVertexClient(
        service_account_files=files,
        service_account_failure_threshold=3,
    )

    assert client.chat([{"role": "user", "content": "hello"}]).content == "vertex-complete"
    assert [account for account, _ in calls] == ["service-one"] * 4 + ["service-two"]
    assert delays == [1.0, 2.0, 1.0]


def test_vertex_rotates_for_service_account_auth_failure(tmp_path, monkeypatch):
    calls = []
    files = [tmp_path / "service-one.json", tmp_path / "service-two.json"]

    class Forbidden(Exception):
        status_code = 403

    def operation(account: str):
        if account == "service-one":
            raise Forbidden("permission denied")
        return _success("authorized")

    _configure_fake_vertex_accounts(monkeypatch, files, calls, operation)
    monkeypatch.setattr(llm_client.time, "sleep", lambda delay: None)
    client = llm_client.GeminiVertexClient(
        service_account_files=files,
        service_account_failure_threshold=2,
    )

    assert client.chat([{"role": "user", "content": "hello"}]).content == "authorized"
    assert [account for account, _ in calls] == ["service-one"] * 2 + ["service-two"]


def test_vertex_does_not_rotate_for_invalid_request(tmp_path, monkeypatch):
    calls = []
    files = [tmp_path / "service-one.json", tmp_path / "service-two.json"]

    class BadRequest(Exception):
        status_code = 400

    def operation(account: str):
        raise BadRequest("unsupported parameter")

    _configure_fake_vertex_accounts(monkeypatch, files, calls, operation)
    client = llm_client.GeminiVertexClient(service_account_files=files)

    with pytest.raises(BadRequest):
        client.chat([{"role": "user", "content": "hello"}])

    assert [account for account, _ in calls] == ["service-one"]
    assert client.active_service_account_index == 0


def test_ai_studio_success_resets_the_consecutive_failure_count(monkeypatch):
    calls = []
    attempts = 0

    def operation(api_key: str):
        nonlocal attempts
        attempts += 1
        if attempts in {1, 2, 4, 5, 6, 7, 8}:
            raise ResourceExhausted("temporary quota failure")
        return _success(f"success-{attempts}")

    monkeypatch.setattr(
        "google.genai.Client",
        lambda *, api_key: _FakeClient(api_key, calls, operation),
    )
    monkeypatch.setattr(llm_client.time, "sleep", lambda delay: None)
    client = llm_client.GeminiAIStudioClient(
        api_keys=["key-one", "key-two"],
        key_failure_threshold=5,
    )

    assert client.chat([{"role": "user", "content": "first"}]).content == "success-3"
    assert client.chat([{"role": "user", "content": "second"}]).content == "success-9"
    assert [api_key for api_key, _ in calls] == ["key-one"] * 8 + ["key-two"]


def test_ai_studio_rotation_uses_separate_failure_type_pools(monkeypatch):
    calls = []
    delays = []
    attempts = 0

    def operation(api_key: str):
        nonlocal attempts
        attempts += 1
        if api_key == "key-one":
            if attempts == 3:
                return _success("")
            raise ResourceExhausted("429 quota exceeded")
        return _success()

    monkeypatch.setattr(
        "google.genai.Client",
        lambda *, api_key: _FakeClient(api_key, calls, operation),
    )
    monkeypatch.setattr(llm_client.time, "sleep", delays.append)
    client = llm_client.GeminiAIStudioClient(
        api_keys=["key-one", "key-two"],
        key_failure_threshold=3,
    )

    assert client.chat([{"role": "user", "content": "hello"}]).content == "complete"
    assert [api_key for api_key, _ in calls] == ["key-one"] * 4 + ["key-two"]
    assert delays == [1.0, 2.0, 1.0, llm_client.AI_STUDIO_KEY_ROTATION_DELAY_SECONDS]


def test_ai_studio_does_not_rotate_for_invalid_requests(monkeypatch):
    calls = []

    class ClientError(Exception):
        status_code = 400

    def operation(api_key: str):
        raise ClientError("bad request: unsupported parameter")

    monkeypatch.setattr(
        "google.genai.Client",
        lambda *, api_key: _FakeClient(api_key, calls, operation),
    )
    client = llm_client.GeminiAIStudioClient(api_keys=["key-one", "key-two"])

    with pytest.raises(ClientError):
        client.chat([{"role": "user", "content": "hello"}])

    assert [api_key for api_key, _ in calls] == ["key-one"]
    assert client.active_key_index == 0


@pytest.mark.parametrize(
    "error, expected_type",
    [
        (type("KeyError", (Exception,), {"status_code": 401})("unauthorized"), "http_401"),
        (type("PermissionError", (Exception,), {"status_code": 403})("forbidden"), "http_403"),
        (Exception("invalid API key"), "api_key"),
        (Exception("permission denied"), "permission_denied"),
        (Exception("daily quota reached"), "quota"),
    ],
)
def test_ai_studio_key_failures_get_separate_rotation_pools(error, expected_type):
    retryable, _ = llm_client._is_google_ai_studio_retryable(error)

    assert retryable is True
    assert llm_client._get_retry_failure_type(error) == expected_type


def test_ai_studio_key_environment_precedence(monkeypatch):
    monkeypatch.delenv("GOOGLE_AI_STUDIO_API_KEYS_FILE", raising=False)
    monkeypatch.setenv("GOOGLE_AI_STUDIO_API_KEYS", "first, second;first\nthird")
    monkeypatch.setenv("GOOGLE_API_KEY", "standard-google")
    monkeypatch.setenv("GEMINI_API_KEY", "standard-gemini")

    assert llm_client.get_google_ai_studio_api_keys() == ["first", "second", "third"]

    monkeypatch.delenv("GOOGLE_AI_STUDIO_API_KEYS")
    assert llm_client.get_google_ai_studio_api_keys() == ["standard-google"]


def test_ai_studio_loads_keys_from_project_relative_text_file(tmp_path, monkeypatch):
    key_file = tmp_path / "ai-studio-keys.txt"
    key_file.write_text("# rotation order\nfirst\n\nsecond\nfirst\n", encoding="utf-8")
    project_root = Path(llm_client.__file__).resolve().parent.parent
    relative_path = os.path.relpath(key_file, project_root)

    monkeypatch.setenv("GOOGLE_AI_STUDIO_API_KEYS_FILE", relative_path)
    monkeypatch.setenv("GOOGLE_AI_STUDIO_API_KEYS", "inline-key")

    assert llm_client.get_google_ai_studio_api_keys() == ["first", "second"]


def test_ai_studio_key_file_must_use_a_relative_path(monkeypatch, tmp_path):
    monkeypatch.setenv("GOOGLE_AI_STUDIO_API_KEYS_FILE", str(tmp_path / "keys.txt"))

    with pytest.raises(ValueError, match="relative to the project root"):
        llm_client.get_google_ai_studio_api_keys()


def test_ai_studio_provider_ignores_vertex_batch_arguments(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "src.evaluator.get_eval_logger",
        lambda *args, **kwargs: SimpleNamespace(info=lambda message: None),
    )
    monkeypatch.setattr(
        "src.evaluator.get_api_config",
        lambda: APIConfig(judge_provider="openai"),
    )
    evaluator = Evaluator(
        method_config=MethodConfig(
            method_name="long_context",
            method_type="baseline",
            model=ModelConfig(provider="ai_studio", name="gemini-2.5-flash"),
        ),
        dataset_config=DatasetConfig(dataset_name="medmemorybench"),
        output_dir=tmp_path,
        batch_api=True,
        batch_gcs_uri="gs://private-bucket/run",
        batch_wait=True,
    )

    assert evaluator.batch_api is False
    assert evaluator.batch_gcs_uri is None
    assert evaluator.batch_wait is False
    assert evaluator._ai_studio_batch_skipped is True


@pytest.mark.parametrize(
    ("provider", "judge_provider", "expected_gcs"),
    [
        ("openai", "vertex", "gs://private-bucket/run"),
        ("ai_studio", "vertex", "gs://private-bucket/run"),
        ("openai", "openrouter", None),
    ],
)
def test_unsupported_query_provider_keeps_batch_for_supported_judge(
    provider, judge_provider, expected_gcs, tmp_path, monkeypatch
):
    monkeypatch.setattr(
        "src.evaluator.get_eval_logger",
        lambda *args, **kwargs: SimpleNamespace(info=lambda message: None),
    )
    monkeypatch.setattr(
        "src.evaluator.get_api_config",
        lambda: APIConfig(judge_provider=judge_provider),
    )
    evaluator = Evaluator(
        method_config=MethodConfig(
            method_name="long_context",
            method_type="baseline",
            model=ModelConfig(provider=provider, name="test-model"),
        ),
        dataset_config=DatasetConfig(dataset_name="medmemorybench"),
        output_dir=tmp_path,
        batch_api=True,
        batch_gcs_uri="gs://private-bucket/run",
        batch_wait=True,
    )

    assert evaluator.batch_api is True
    assert evaluator.batch_gcs_uri == expected_gcs
    assert evaluator.batch_wait is True
    assert evaluator._method_batch_skipped is True


def test_agent_manager_does_not_pass_openai_key_to_ai_studio():
    manager = AgentManager.__new__(AgentManager)
    manager.method_config = MethodConfig(
        method_name="long_context",
        method_type="baseline",
        model=ModelConfig(provider="ai_studio", name="gemini-2.5-flash"),
    )
    manager.dataset_config = DatasetConfig(dataset_name="medmemorybench")
    manager._api_config = APIConfig(openai_api_key="must-not-be-used")
    manager._batch_api = False
    manager._batch_gcs_uri = None
    manager._batch_wait = False
    manager._batch_manifest_dir = None
    manager._batch_config_hash = ""
    manager._batch_progress_callback = None

    assert manager._build_agent_params("long_context")["api_key"] is None


def test_ai_studio_judge_prefers_rotating_judge_keys():
    config = APIConfig(
        openai_api_key="must-not-be-used",
        google_ai_studio_api_keys="shared-one,shared-two",
        judge_provider="ai_studio",
        judge_api_key="single-judge",
        judge_api_keys="judge-one,judge-two",
    )

    assert config.get_judge_api_key() == "judge-one,judge-two"


def test_removed_gemini_provider_aliases_are_rejected():
    for provider in ("google_vertex", "vertex_ai", "google_ai_studio", "gemini_api"):
        with pytest.raises(ValueError, match="Unsupported LLM provider"):
            llm_client.create_llm_client(provider=provider)


def test_hybrid_starts_on_vertex_then_rotates_to_ai_studio(monkeypatch):
    calls = []
    delays = []

    def operation(transport: str):
        if transport == "vertex":
            raise ResourceExhausted("429 quota exceeded")
        return _success("studio-complete")

    vertex = _FakeClient("vertex", calls, operation)
    monkeypatch.setattr(
        llm_client,
        "GeminiVertexClient",
        lambda **kwargs: _VertexStub(vertex),
    )
    monkeypatch.setattr(
        "google.genai.Client",
        lambda *, api_key: _FakeClient(api_key, calls, operation),
    )
    monkeypatch.setattr(llm_client.time, "sleep", delays.append)

    client = llm_client.GeminiHybridClient(
        api_keys=["studio-key"],
        key_failure_threshold=5,
    )
    response = client.chat([{"role": "user", "content": "hello"}])

    assert response.content == "studio-complete"
    assert [transport for transport, _ in calls] == ["vertex"] * 5 + ["studio-key"]
    assert delays == [1.0, 2.0, 4.0, 8.0]
    assert client.active_transport == "ai_studio[1]"


def test_hybrid_rotates_through_vertex_accounts_before_ai_studio(tmp_path, monkeypatch):
    calls = []
    files = [tmp_path / "service-one.json", tmp_path / "service-two.json"]

    def operation(transport: str):
        if transport in {"service-one", "service-two"}:
            raise ResourceExhausted(f"{transport} quota")
        return _success("studio-complete")

    _configure_fake_vertex_accounts(monkeypatch, files, calls, operation)
    monkeypatch.setattr(llm_client.time, "sleep", lambda delay: None)
    client = llm_client.GeminiHybridClient(
        api_keys=["studio-key"],
        service_account_files=files,
        service_account_failure_threshold=2,
        key_failure_threshold=2,
    )

    assert client.chat([{"role": "user", "content": "hello"}]).content == "studio-complete"
    assert [transport for transport, _ in calls] == (
        ["service-one"] * 2 + ["service-two"] * 2 + ["studio-key"]
    )
    assert client.active_transport == "ai_studio[1]"
    assert client.vertex_client.active_service_account_index == 1


def test_hybrid_rotates_back_to_vertex(monkeypatch):
    calls = []
    mode = {"vertex_fails": True, "studio_fails": False}

    def operation(transport: str):
        if transport == "vertex" and mode["vertex_fails"]:
            raise ResourceExhausted("vertex quota")
        if transport == "studio-key" and mode["studio_fails"]:
            raise ResourceExhausted("studio quota")
        return _success(transport)

    vertex = _FakeClient("vertex", calls, operation)
    monkeypatch.setattr(
        llm_client,
        "GeminiVertexClient",
        lambda **kwargs: _VertexStub(vertex),
    )
    monkeypatch.setattr(
        "google.genai.Client",
        lambda *, api_key: _FakeClient(api_key, calls, operation),
    )
    monkeypatch.setattr(llm_client.time, "sleep", lambda delay: None)

    client = llm_client.GeminiHybridClient(api_keys=["studio-key"])
    assert client.chat([{"role": "user", "content": "first"}]).content == "studio-key"

    mode.update(vertex_fails=False, studio_fails=True)
    assert client.chat([{"role": "user", "content": "second"}]).content == "vertex"
    assert client.active_transport == "vertex"


def test_hybrid_rotation_uses_separate_failure_type_pools(monkeypatch):
    calls = []
    delays = []
    vertex_attempts = 0

    def operation(transport: str):
        nonlocal vertex_attempts
        if transport == "vertex":
            vertex_attempts += 1
            if vertex_attempts == 3:
                return _success("")
            raise ResourceExhausted("429 quota exceeded")
        return _success("studio-complete")

    vertex = _FakeClient("vertex", calls, operation)
    monkeypatch.setattr(
        llm_client,
        "GeminiVertexClient",
        lambda **kwargs: _VertexStub(vertex),
    )
    monkeypatch.setattr(
        "google.genai.Client",
        lambda *, api_key: _FakeClient(api_key, calls, operation),
    )
    monkeypatch.setattr(llm_client.time, "sleep", delays.append)
    client = llm_client.GeminiHybridClient(
        api_keys=["studio-key"],
        key_failure_threshold=3,
    )

    assert client.chat([{"role": "user", "content": "hello"}]).content == "studio-complete"
    assert [transport for transport, _ in calls] == ["vertex"] * 4 + ["studio-key"]
    assert delays == [1.0, 2.0, 1.0]


def test_hybrid_rotates_through_each_ai_studio_key_then_vertex(monkeypatch):
    calls = []

    def operation(transport: str):
        raise ResourceExhausted(f"{transport} quota")

    vertex = _FakeClient("vertex", calls, operation)
    monkeypatch.setattr(
        llm_client,
        "GeminiVertexClient",
        lambda **kwargs: _VertexStub(vertex),
    )
    monkeypatch.setattr(
        "google.genai.Client",
        lambda *, api_key: _FakeClient(api_key, calls, operation),
    )
    monkeypatch.setattr(llm_client.time, "sleep", lambda delay: None)

    client = llm_client.GeminiHybridClient(
        api_keys=["studio-one", "studio-two"],
        key_failure_threshold=5,
    )
    with pytest.raises(llm_client.LLMRetryExhaustedError, match="Vertex and every"):
        client.chat([{"role": "user", "content": "hello"}])

    assert [transport for transport, _ in calls] == (
        ["vertex"] * 5 + ["studio-one"] * 5 + ["studio-two"] * 5
    )
    assert client.active_transport == "vertex"


def test_hybrid_batch_transport_is_always_vertex(tmp_path, monkeypatch):
    from utils.vertex_batch import VertexBatchClient

    vertex = object.__new__(llm_client.GeminiVertexClient)
    vertex.model = "gemini-2.5-flash"
    vertex.project = "test-project"
    vertex.location = "global"
    vertex.credentials = object()
    hybrid = object.__new__(llm_client.GeminiHybridClient)
    hybrid.vertex_client = vertex

    batch_client = VertexBatchClient.from_gemini_client(
        hybrid,
        gcs_uri="gs://private-bucket/batch",
        manifest_path=tmp_path / "manifest.json",
        wait=False,
    )

    assert batch_client._direct_client is vertex
    assert batch_client.project == "test-project"
