"""Regression tests for configurable first-party LLM generation settings."""

from types import SimpleNamespace

import pytest

from metrics.llm_judge import LLMJudge
from methods.memrl_agent import MemRLAgent, TrackedLLMProvider
from src.agent import AgentManager
from src.config import DatasetConfig, MethodConfig, load_env_config


def test_env_loads_all_judge_generation_settings(tmp_path, monkeypatch):
    env_path = tmp_path / ".env"
    env_path.write_text(
        "\n".join([
            "JUDGE_TEMPERATURE=0.25",
            "JUDGE_CLIENT_MAX_TOKENS=9000",
            "JUDGE_MAX_TOKENS=650",
            "JUDGE_MCD_MAX_TOKENS=2400",
        ]),
        encoding="utf-8",
    )
    for name in (
        "JUDGE_TEMPERATURE",
        "JUDGE_CLIENT_MAX_TOKENS",
        "JUDGE_MAX_TOKENS",
        "JUDGE_MCD_MAX_TOKENS",
    ):
        monkeypatch.delenv(name, raising=False)

    config = load_env_config(env_path)

    assert config.get_judge_temperature() == 0.25
    assert config.get_judge_client_max_tokens() == 9000
    assert config.get_judge_max_tokens() == 650
    assert config.get_judge_max_tokens("multi_hop_clinical_deduction") == 2400


def test_judge_uses_configured_realtime_and_batch_generation_settings():
    calls = []
    client = SimpleNamespace(
        chat=lambda **kwargs: calls.append(kwargs)
        or SimpleNamespace(content='{"is_correct": true}')
    )
    judge = LLMJudge(
        client=client,
        judge_temperature=0.15,
        judge_max_tokens=700,
        judge_mcd_max_tokens=2600,
    )
    judge._initialized = True
    judge._active_judge_provider = "openai"

    judge._call_llm("prompt")
    normal_payload = judge.prepare_batch_prompt(
        "state_update", "question", "output", "answer"
    )
    mcd_payload = judge.prepare_batch_prompt(
        "multi_hop_clinical_deduction", "question", "output", "answer"
    )

    assert calls[0]["temperature"] == 0.15
    assert calls[0]["max_tokens"] == 700
    assert normal_payload["temperature"] == 0.15
    assert normal_payload["max_tokens"] == 700
    assert mcd_payload["temperature"] == 0.15
    assert mcd_payload["max_tokens"] == 2600


def test_memrl_internal_script_settings_apply_to_realtime_and_batch():
    calls = []
    provider = TrackedLLMProvider(
        llm_client=SimpleNamespace(
            chat=lambda **kwargs: calls.append(kwargs)
            or SimpleNamespace(content="script", input_tokens=1, output_tokens=1, latency=0.0)
        ),
        model_name="test-model",
        script_temperature=0.22,
        script_max_tokens=333,
    )
    assert provider.generate_script("trajectory") == "script"
    assert calls[0]["temperature"] == 0.22
    assert calls[0]["max_tokens"] == 333

    agent = object.__new__(MemRLAgent)
    agent._context_id = 1
    agent._vertex_batch_stage_index = 0
    agent.memrl_script_temperature = 0.22
    agent.memrl_script_max_tokens = 333
    agent._tracked_llm = SimpleNamespace(
        prepare_script_messages=TrackedLLMProvider.prepare_script_messages,
        record_batch_result=lambda input_tokens, output_tokens: None,
    )
    agent._memory_service = SimpleNamespace(
        strategy_config=SimpleNamespace(build=SimpleNamespace(value="proceduralization"))
    )
    submitted = []
    agent._get_script_batch_client = lambda: SimpleNamespace(
        run_stage=lambda stage, requests: submitted.extend(requests) or {
            request.request_id: SimpleNamespace(
                content="script", status="", input_tokens=1, output_tokens=1
            )
            for request in requests
        }
    )

    agent._batch_build_contents(["trajectory"])

    assert submitted[0].temperature == 0.22
    assert submitted[0].max_tokens == 333


def test_agent_manager_forwards_internal_llm_settings_from_yaml():
    method_config = MethodConfig.from_dict({
        "method_name": "amem_test",
        "method_type": "agentic_memory",
        "model": {"name": "gemini-test"},
        "agent_params": {
            "amem_temperature": 0.11,
            "amem_retry_temperature": 0.12,
            "amem_connectivity_temperature": 0.13,
            "amem_relation_temperature": 0.14,
        },
    })
    manager = object.__new__(AgentManager)
    manager.method_config = method_config
    manager.dataset_config = DatasetConfig(dataset_name="medmemorybench")
    manager._api_config = SimpleNamespace(openai_api_key="", openai_base_url="")
    manager._batch_api = False
    manager._batch_gcs_uri = None
    manager._batch_wait = False
    manager._batch_manifest_dir = None
    manager._batch_config_hash = ""
    manager._batch_progress_callback = None

    params = manager._build_agent_params("amem_test")

    assert params["amem_temperature"] == 0.11
    assert params["amem_retry_temperature"] == 0.12
    assert params["amem_connectivity_temperature"] == 0.13
    assert params["amem_relation_temperature"] == 0.14


@pytest.mark.parametrize("temperature", [0.0, 0.4])
def test_explicit_temperature_is_not_replaced_by_client_default(temperature):
    from utils.llm_client import BaseGeminiClient

    client = object.__new__(BaseGeminiClient)
    client.temperature = 0.9
    client.max_tokens = 100
    client.model = "gemini-test"
    captured = {}

    class _Models:
        def generate_content(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                text="ok",
                usage_metadata=SimpleNamespace(
                    prompt_token_count=0, candidates_token_count=0
                ),
            )

    client.client = SimpleNamespace(models=_Models())
    client._chat_once(
        [{"role": "user", "content": "hello"}],
        temperature=temperature,
    )

    assert captured["config"].temperature == temperature
