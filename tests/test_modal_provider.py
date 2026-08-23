from types import SimpleNamespace

from src.agent import AgentManager
from src.config import APIConfig, DatasetConfig, MethodConfig, load_env_config
from utils.llm_client import ModalClient, create_llm_client


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