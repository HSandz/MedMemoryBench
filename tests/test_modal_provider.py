import ast
from pathlib import Path
from types import SimpleNamespace

from src.agent import AgentManager
from src.config import APIConfig, DatasetConfig, MethodConfig, load_env_config
from utils.llm_client import (
    ModalClient,
    ModalModelNotReadyError,
    OpenAIModelReadinessGate,
    create_llm_client,
)


def test_modal_environment_configuration(tmp_path, monkeypatch):
    env_path = tmp_path / ".env"
    env_path.write_text(
        "MODAL_API_KEY=test-proxy-token\nMODAL_BASE_URL=https://modal.example/v1\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("MODAL_API_KEY", raising=False)
    monkeypatch.delenv("MODAL_PROXY_TOKEN", raising=False)
    monkeypatch.delenv("MODAL_BASE_URL", raising=False)

    config = load_env_config(env_path)

    assert config.modal_api_key == "test-proxy-token"
    assert config.modal_base_url == "https://modal.example/v1"


def test_modal_provider_factory_uses_modal_client(monkeypatch):
    monkeypatch.setenv("MODAL_API_KEY", "test-proxy-token")
    monkeypatch.setenv("MODAL_BASE_URL", "https://modal.example/v1")

    client = create_llm_client(provider="modal", model="served-model")

    assert isinstance(client, ModalClient)
    assert client.model == "served-model"
    assert client.client.api_key == "test-proxy-token"
    assert str(client.client.base_url) == "https://modal.example/v1/"


def test_agent_manager_forwards_modal_configuration():
    method_config = MethodConfig.from_dict({
        "method_name": "long_context",
        "method_type": "baseline",
        "model": {"provider": "modal", "name": "served-model"},
    })
    manager = object.__new__(AgentManager)
    manager.method_config = method_config
    manager.dataset_config = DatasetConfig(dataset_name="medmemorybench")
    manager._api_config = APIConfig(
        modal_api_key="modal-token",
        modal_base_url="https://modal.example/v1",
    )
    manager._batch_api = False
    manager._batch_gcs_uri = None
    manager._batch_wait = False
    manager._batch_manifest_dir = None
    manager._batch_config_hash = ""
    manager._batch_progress_callback = None

    params = manager._build_agent_params("long_context")

    assert params["provider"] == "modal"
    assert params["api_key"] == "modal-token"
    assert params["base_url"] == "https://modal.example/v1"


def test_modal_readiness_waits_for_requested_model(monkeypatch):
    responses = [
        SimpleNamespace(data=[SimpleNamespace(id="old-model")]),
        SimpleNamespace(data=[SimpleNamespace(id="served-model")]),
    ]
    models = SimpleNamespace(list=lambda: responses.pop(0))
    gate = OpenAIModelReadinessGate(SimpleNamespace(models=models), "served-model")
    monkeypatch.setattr("utils.llm_client.time.sleep", lambda _: None)

    gate.wait()

    assert gate._ready is True
    assert responses == []


def test_modal_readiness_timeout_reports_available_models(monkeypatch):
    models = SimpleNamespace(
        list=lambda: SimpleNamespace(data=[SimpleNamespace(id="old-model")])
    )
    gate = OpenAIModelReadinessGate(SimpleNamespace(models=models), "served-model")
    monkeypatch.setenv("MODAL_READY_TIMEOUT_SECONDS", "0")

    try:
        gate.wait()
    except ModalModelNotReadyError as exc:
        message = str(exc)
    else:
        raise AssertionError("Expected readiness timeout")

    assert "served-model" in message
    assert "old-model" in message


def test_modal_deployment_releases_startup_pin_after_ready_idle_window():
    script_path = Path(__file__).parents[1] / "scripts" / "modal_vllm_server.py"
    tree = ast.parse(script_path.read_text(encoding="utf-8"))

    scaledown_window = next(
        node.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id == "SCALEDOWN_WINDOW_SECONDS"
            for target in node.targets
        )
    )
    assert isinstance(scaledown_window, ast.Call)
    getenv_call = scaledown_window.args[0]
    assert isinstance(getenv_call, ast.Call)
    assert [argument.value for argument in getenv_call.args] == [
        "MODAL_VLLM_SCALEDOWN_WINDOW_SECONDS",
        "300",
    ]

    server_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Server"
    )
    server_decorator = next(
        decorator
        for decorator in server_class.decorator_list
        if isinstance(decorator, ast.Call)
        and isinstance(decorator.func, ast.Attribute)
        and decorator.func.attr == "server"
    )
    min_containers = next(
        item for item in server_decorator.keywords if item.arg == "min_containers"
    )
    assert isinstance(min_containers.value, ast.Constant)
    assert min_containers.value.value == 1

    release_function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_release_deploy_start_pin"
    )
    release_calls = [
        node for node in ast.walk(release_function) if isinstance(node, ast.Call)
    ]
    sleep_call = next(
        call
        for call in release_calls
        if isinstance(call.func, ast.Attribute) and call.func.attr == "sleep"
    )
    assert isinstance(sleep_call.args[0], ast.Name)
    assert sleep_call.args[0].id == "SCALEDOWN_WINDOW_SECONDS"

    update_call = next(
        call
        for call in release_calls
        if isinstance(call.func, ast.Attribute)
        and call.func.attr == "update_autoscaler"
    )
    released_minimum = next(
        item for item in update_call.keywords if item.arg == "min_containers"
    )
    assert isinstance(released_minimum.value, ast.Constant)
    assert released_minimum.value.value == 0

    start_method = next(
        node
        for node in server_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "start"
    )
    start_calls = [
        node for node in ast.walk(start_method) if isinstance(node, ast.Call)
    ]
    warmup_call = next(
        call
        for call in start_calls
        if isinstance(call.func, ast.Attribute) and call.func.attr == "_warmup"
    )
    thread_call = next(
        call
        for call in start_calls
        if isinstance(call.func, ast.Attribute) and call.func.attr == "start"
    )
    assert warmup_call.lineno < thread_call.lineno


def test_modal_deployment_loads_repository_env_before_configuration():
    script_path = Path(__file__).parents[1] / "scripts" / "modal_vllm_server.py"
    tree = ast.parse(script_path.read_text(encoding="utf-8"))

    loader = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_load_local_env"
    )
    loader_calls = [
        node for node in ast.walk(loader) if isinstance(node, ast.Call)
    ]
    is_local_call = next(
        call
        for call in loader_calls
        if isinstance(call.func, ast.Attribute) and call.func.attr == "is_local"
    )
    load_dotenv_call = next(
        call
        for call in loader_calls
        if isinstance(call.func, ast.Name) and call.func.id == "load_dotenv"
    )
    assert is_local_call.lineno < load_dotenv_call.lineno
    override = next(
        keyword for keyword in load_dotenv_call.keywords if keyword.arg == "override"
    )
    assert isinstance(override.value, ast.Constant)
    assert override.value.value is False

    loader_invocation = next(
        node
        for node in tree.body
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "_load_local_env"
    )
    app_name_assignment = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "APP_NAME"
            for target in node.targets
        )
    )
    assert loader_invocation.lineno < app_name_assignment.lineno


def test_modal_deployment_injects_hf_token_as_runtime_secret():
    script_path = Path(__file__).parents[1] / "scripts" / "modal_vllm_server.py"
    tree = ast.parse(script_path.read_text(encoding="utf-8"))

    hf_token_assignment = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "HF_TOKEN"
            for target in node.targets
        )
    )
    assert isinstance(hf_token_assignment.value, ast.Call)
    assert isinstance(hf_token_assignment.value.func, ast.Attribute)
    assert hf_token_assignment.value.func.attr == "strip"

    secret_assignment = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "server_secrets"
            for target in node.targets
        )
    )
    secret_calls = [
        node
        for node in ast.walk(secret_assignment)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "from_dict"
    ]
    assert len(secret_calls) == 1
    secret_mapping = secret_calls[0].args[0]
    assert isinstance(secret_mapping, ast.Dict)
    assert secret_mapping.keys[0].value == "HF_TOKEN"
    assert isinstance(secret_mapping.values[0], ast.Name)
    assert secret_mapping.values[0].id == "HF_TOKEN"

    server_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Server"
    )
    server_decorator = next(
        decorator
        for decorator in server_class.decorator_list
        if isinstance(decorator, ast.Call)
        and isinstance(decorator.func, ast.Attribute)
        and decorator.func.attr == "server"
    )
    secrets_keyword = next(
        keyword for keyword in server_decorator.keywords if keyword.arg == "secrets"
    )
    assert isinstance(secrets_keyword.value, ast.Name)
    assert secrets_keyword.value.id == "server_secrets"

    image_assignment = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "image"
            for target in node.targets
        )
    )
    image_env_call = next(
        node
        for node in ast.walk(image_assignment)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "env"
    )
    image_env = image_env_call.args[0]
    assert isinstance(image_env, ast.Dict)
    image_env_keys = {
        key.value for key in image_env.keys if isinstance(key, ast.Constant)
    }
    assert "HF_TOKEN" not in image_env_keys
