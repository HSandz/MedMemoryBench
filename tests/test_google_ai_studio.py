"""Offline tests for Google AI Studio provider selection and key rotation."""

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


def test_ai_studio_rotates_after_five_consecutive_failures(monkeypatch):
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
    )
    response = client.chat([{"role": "user", "content": "hello"}])

    assert response.content == "complete"
    assert [api_key for api_key, _ in calls] == ["key-one"] * 5 + ["key-two"]
    assert delays == [1.0, 2.0, 4.0, 8.0]
    assert client.active_key_index == 1


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
    client = llm_client.GeminiAIStudioClient(api_keys=["key-one", "key-two"])

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
    assert delays == [1.0, 2.0, 1.0]


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
    monkeypatch.setenv("GOOGLE_AI_STUDIO_API_KEYS", "first, second;first\nthird")
    monkeypatch.setenv("GOOGLE_API_KEY", "standard-google")
    monkeypatch.setenv("GEMINI_API_KEY", "standard-gemini")

    assert llm_client.get_google_ai_studio_api_keys() == ["first", "second", "third"]

    monkeypatch.delenv("GOOGLE_AI_STUDIO_API_KEYS")
    assert llm_client.get_google_ai_studio_api_keys() == ["standard-google"]


def test_ai_studio_provider_ignores_vertex_batch_arguments(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "src.evaluator.get_eval_logger",
        lambda *args, **kwargs: SimpleNamespace(info=lambda message: None),
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

    client = llm_client.GeminiHybridClient(api_keys=["studio-key"])
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
