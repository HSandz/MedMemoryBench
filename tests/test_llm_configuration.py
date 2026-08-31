"""Regression tests for configurable first-party LLM generation settings."""

import logging
from types import SimpleNamespace

import pytest

from metrics.llm_judge import LLMJudge
from methods.memrl_agent import MemRLAgent, TrackedLLMProvider
from src.agent import AgentManager
from src.config import APIConfig, DatasetConfig, MethodConfig, load_env_config
from src.evaluator import Evaluator
from utils.batch_client import create_batch_client
from utils.llm_client import (
    AnthropicClient,
    OpenAIClient,
    OpenRouterClient,
    create_llm_client,
    extract_usage_token_counts,
    TruncatedLLMResponseError,
)
from utils.openrouter_batch import OpenRouterBatchClient


def test_env_loads_all_judge_generation_settings(tmp_path, monkeypatch):
    env_path = tmp_path / ".env"
    env_path.write_text(
        "\n".join([
            "JUDGE_TEMPERATURE=0.25",
            "JUDGE_CLIENT_MAX_TOKENS=9000",
            "JUDGE_MAX_TOKENS=650",
            "JUDGE_MCD_MAX_TOKENS=2400",
            "JUDGE_REASONING_EFFORT=low",
        ]),
        encoding="utf-8",
    )
    for name in (
        "JUDGE_TEMPERATURE",
        "JUDGE_CLIENT_MAX_TOKENS",
        "JUDGE_MAX_TOKENS",
        "JUDGE_MCD_MAX_TOKENS",
        "JUDGE_REASONING_EFFORT",
    ):
        monkeypatch.delenv(name, raising=False)

    config = load_env_config(env_path)

    assert config.get_judge_temperature() == 0.25
    assert config.get_judge_client_max_tokens() == 9000
    assert config.get_judge_max_tokens() == 650
    assert config.get_judge_max_tokens("multi_hop_clinical_deduction") == 2400
    assert config.get_judge_reasoning_effort() == "low"


def test_reasoning_effort_is_loaded_for_query_and_memorize_models():
    config = MethodConfig.from_dict({
        "method_name": "amem_test",
        "method_type": "agentic_memory",
        "model": {"name": "gpt-5.1", "reasoning_effort": "high"},
        "memorize_model": {"name": "gemini-2.5-flash", "reasoning_effort": 1024},
    })
    assert config.model.reasoning_effort == "high"
    assert config.memorize_model.reasoning_effort == 1024


def test_agent_manager_forwards_reasoning_effort():
    config = MethodConfig.from_dict({
        "method_name": "long_context",
        "method_type": "baseline",
        "model": {"provider": "openai", "name": "gpt-5.1", "reasoning_effort": "high"},
    })
    manager = object.__new__(AgentManager)
    manager.method_config = config
    manager.dataset_config = DatasetConfig(dataset_name="medmemorybench")
    manager._api_config = APIConfig(openai_api_key="key")
    manager._batch_api = False
    manager._batch_gcs_uri = manager._batch_manifest_dir = None
    manager._batch_wait = False
    manager._batch_config_hash = ""
    manager._batch_progress_callback = None
    assert manager._build_agent_params("long_context")["llm_client_kwargs"] == {
        "reasoning_effort": "high"
    }


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
            "amem_note_level": "session",
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

    assert params["amem_note_level"] == "session"
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


def test_openrouter_environment_configuration(tmp_path, monkeypatch):
    env_path = tmp_path / ".env"
    env_path.write_text(
        "OPENROUTER_API_KEY=test-key\n"
        "OPENROUTER_BASE_URL=https://openrouter.example/api/v1\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_BASE_URL", raising=False)

    config = load_env_config(env_path)

    assert config.openrouter_api_key == "test-key"
    assert config.openrouter_base_url == "https://openrouter.example/api/v1"


def test_openrouter_factory_reuses_openai_client_logic(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.delenv("OPENROUTER_BASE_URL", raising=False)

    client = create_llm_client(provider="openrouter", model="openai/gpt-4o-mini")

    assert isinstance(client, OpenRouterClient)
    assert issubclass(OpenRouterClient, OpenAIClient)
    assert client.client.api_key == "test-key"
    assert str(client.client.base_url) == "https://openrouter.ai/api/v1/"


def test_openai_and_openrouter_reasoning_effort_request_mapping():
    calls = []

    class Completions:
        def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="ok", refusal=None), finish_reason="stop")],
                usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
            )

    openai = object.__new__(OpenAIClient)
    openai.model, openai.temperature, openai.max_tokens = "gpt-5.1", 1.0, 10
    openai.reasoning_effort = "high"
    openai.client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    openai.chat([{"role": "user", "content": "hi"}])
    assert calls[-1]["reasoning_effort"] == "high"

    router = object.__new__(OpenRouterClient)
    router.model, router.temperature, router.max_tokens = "openai/gpt-5.1", 1.0, 10
    router.reasoning_effort = "low"
    router.provider_routing = router.service_tier = None
    router.client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    router.chat([{"role": "user", "content": "hi"}])
    assert calls[-1]["extra_body"]["reasoning"] == {"effort": "low"}


def test_nim_thinking_yaml_maps_to_bifrost_passthrough_fields():
    config = MethodConfig.from_dict({
        "method_name": "amem_test",
        "method_type": "agentic_memory",
        "model": {
            "provider": "openai",
            "name": "NIM/nvidia/nemotron-3.5-lightning-30b-a3b",
            "nim": {"enable_thinking": False, "reasoning_budget": 0},
        },
    })
    manager = object.__new__(AgentManager)
    manager.method_config = config
    manager.dataset_config = DatasetConfig(dataset_name="medmemorybench")
    manager._api_config = APIConfig(openai_api_key="key")
    manager._batch_api = False
    manager._batch_gcs_uri = manager._batch_manifest_dir = None
    manager._batch_wait = False
    manager._batch_config_hash = ""
    manager._batch_progress_callback = None

    assert manager._build_agent_params("amem_test")["llm_client_kwargs"] == {
        "nim_thinking_enabled": False,
        "nim_reasoning_budget": 0,
    }

    captured = {}
    client = object.__new__(OpenAIClient)
    client.model, client.temperature, client.max_tokens = config.model.name, 0.0, 100
    client.reasoning_effort = None
    client.nim_thinking_enabled = False
    client.nim_reasoning_budget = 0
    client.client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **kwargs: captured.update(kwargs) or SimpleNamespace(
                    choices=[SimpleNamespace(
                        message=SimpleNamespace(content="9.8", refusal=None),
                        finish_reason="stop",
                    )],
                    usage=SimpleNamespace(prompt_tokens=4, completion_tokens=4),
                )
            )
        )
    )

    OpenAIClient.chat.__wrapped__(client, [{"role": "user", "content": "hello"}])

    assert captured["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": False},
        "reasoning_budget": 0,
    }
    assert captured["extra_headers"] == {
        "x-bf-passthrough-extra-params": "true",
    }


@pytest.mark.parametrize(
    ("nim", "message"),
    [
        (True, "model.nim must be a mapping"),
        ({"enable_thinking": "false"}, "model.nim.enable_thinking must be a boolean"),
        ({"reasoning_budget": 1.5}, "model.nim.reasoning_budget must be an integer"),
        ({"reasoning_budget": 32769}, "model.nim.reasoning_budget must be between -1 and 32768"),
    ],
)
def test_nim_yaml_options_validate_types_and_budget(nim, message):
    with pytest.raises(ValueError, match=message):
        MethodConfig.from_dict({
            "method_name": "long_context",
            "method_type": "baseline",
            "model": {"name": "NIM/nvidia/nemotron-3.5-lightning-30b-a3b", "nim": nim},
        })


@pytest.mark.parametrize("finish_reason", ["length", "max_tokens"])
def test_openai_rejects_truncated_reasoning_response(finish_reason, caplog):
    response = SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(
                content="Here is an unfinished analysis",
                refusal=None,
                reasoning="internal reasoning",
                reasoning_content="provider reasoning",
            ),
            finish_reason=finish_reason,
        )],
        usage=SimpleNamespace(prompt_tokens=4, completion_tokens=10),
    )
    client = object.__new__(OpenAIClient)
    client.model = "NIM/nvidia/nemotron-3.5-lightning-30b-a3b"
    client.temperature = 0.0
    client.max_tokens = 100
    client.client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **kwargs: response)
        )
    )

    caplog.set_level(logging.DEBUG, logger="utils.llm_client")
    with pytest.raises(TruncatedLLMResponseError):
        OpenAIClient.chat.__wrapped__(client, [{"role": "user", "content": "hello"}])

    messages = " ".join(record.getMessage() for record in caplog.records)
    assert "finish_reason" in messages
    assert "message.reasoning" in messages
    assert "message.reasoning_content" in messages


def test_anthropic_reasoning_effort_uses_output_config():
    calls = []

    class Messages:
        def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                content=[SimpleNamespace(text="ok")],
                usage=SimpleNamespace(input_tokens=1, output_tokens=1),
            )

    client = object.__new__(AnthropicClient)
    client.model, client.temperature, client.max_tokens = "claude-sonnet-4-20250514", 1.0, 10
    client.reasoning_effort = "medium"
    client.client = SimpleNamespace(messages=Messages())
    client.chat([{"role": "user", "content": "hi"}])
    assert calls[0]["output_config"] == {"effort": "medium"}


def test_agent_manager_forwards_openrouter_configuration():
    method_config = MethodConfig.from_dict({
        "method_name": "long_context",
        "method_type": "baseline",
        "model": {
            "provider": "openrouter",
            "name": "openai/gpt-5.1",
            "openrouter": {
                "provider": {
                    "order": ["openai", "azure"],
                    "allow_fallbacks": False,
                },
                "service_tier": "priority",
            },
        },
    })
    manager = object.__new__(AgentManager)
    manager.method_config = method_config
    manager.dataset_config = DatasetConfig(dataset_name="medmemorybench")
    manager._api_config = APIConfig(
        openrouter_api_key="openrouter-key",
        openrouter_base_url="https://openrouter.example/api/v1",
    )
    manager._batch_api = False
    manager._batch_gcs_uri = None
    manager._batch_wait = False
    manager._batch_manifest_dir = None
    manager._batch_config_hash = ""
    manager._batch_progress_callback = None

    params = manager._build_agent_params("long_context")

    assert params["provider"] == "openrouter"
    assert params["api_key"] == "openrouter-key"
    assert params["base_url"] == "https://openrouter.example/api/v1"
    assert params["llm_client_kwargs"] == {
        "provider_routing": {
            "order": ["openai", "azure"],
            "allow_fallbacks": False,
        },
        "service_tier": "priority",
    }


def test_amem_build_model_has_independent_openrouter_configuration():
    method_config = MethodConfig.from_dict({
        "method_name": "amem_test",
        "method_type": "agentic_memory",
        "model": {
            "provider": "openrouter",
            "name": "openai/gpt-5-nano",
            "openrouter": {
                "provider": {"order": ["openai"], "allow_fallbacks": False},
            },
        },
        "memorize_model": {
            "provider": "openrouter",
            "name": "openai/gpt-5-nano",
            "temperature": 0.0,
            "max_completion_tokens": 900,
            "api_key": "build-key",
            "base_url": "https://build.openrouter.example/api/v1",
            "openrouter": {
                "provider": {"only": ["openai"], "allow_fallbacks": False},
                "service_tier": "flex",
            },
        },
        "build_config": {},
    })
    manager = object.__new__(AgentManager)
    manager.method_config = method_config
    manager.dataset_config = DatasetConfig(dataset_name="medmemorybench")
    manager._api_config = APIConfig(
        openrouter_api_key="query-key",
        openrouter_base_url="https://query.openrouter.example/api/v1",
    )
    manager._batch_api = True
    manager._batch_gcs_uri = None
    manager._batch_wait = True
    manager._batch_manifest_dir = None
    manager._batch_config_hash = ""
    manager._batch_progress_callback = None

    params = manager._build_agent_params("amem_test")

    assert params["api_key"] == "query-key"
    assert params["base_url"] == "https://query.openrouter.example/api/v1"
    assert params["llm_client_kwargs"] == {
        "provider_routing": {
            "order": ["openai"],
            "allow_fallbacks": False,
        },
    }
    assert params["amem_backend"] == "openrouter"
    assert params["amem_model"] == "openai/gpt-5-nano"
    assert params["amem_max_tokens"] == 900
    assert params["amem_api_key"] == "build-key"
    assert params["amem_base_url"] == "https://build.openrouter.example/api/v1"
    assert params["amem_llm_client_kwargs"] == {
        "provider_routing": {
            "only": ["openai"],
            "allow_fallbacks": False,
        },
        "service_tier": "flex",
    }


def test_agent_manager_enables_internal_batch_for_openrouter():
    method_config = MethodConfig.from_dict({
        "method_name": "graph_rag",
        "method_type": "rag",
        "model": {"provider": "openrouter", "name": "openai/gpt-4o"},
    })
    manager = object.__new__(AgentManager)
    manager.method_config = method_config
    manager.dataset_config = DatasetConfig(dataset_name="medmemorybench")
    manager._api_config = APIConfig(openrouter_api_key="test-key")
    manager._batch_api = True
    manager._batch_gcs_uri = None
    manager._batch_wait = True
    manager._batch_manifest_dir = None
    manager._batch_config_hash = "hash"
    manager._batch_progress_callback = None

    params = manager._build_agent_params("graph_rag")

    assert params["vertex_batch_enabled"] is True
    assert params["vertex_batch_wait"] is True


def test_batch_factory_selects_openrouter_transport(tmp_path):
    client = object.__new__(OpenRouterClient)
    client.model = "openai/gpt-4o"
    client.provider_routing = None
    client.service_tier = None
    client.client = SimpleNamespace(
        base_url="https://openrouter.ai/api/v1/",
        api_key="test-key",
    )

    batch_client = create_batch_client(
        client,
        gcs_uri="gs://ignored/value",
        manifest_path=tmp_path / "batch.json",
        wait=False,
    )

    assert isinstance(batch_client, OpenRouterBatchClient)


def test_openrouter_batch_ignores_gcs_configuration(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "src.evaluator.get_eval_logger",
        lambda *args, **kwargs: SimpleNamespace(info=lambda message: None),
    )
    monkeypatch.setattr(
        "src.evaluator.get_api_config",
        lambda: APIConfig(judge_provider="openai"),
    )
    evaluator = Evaluator(
        method_config=MethodConfig.from_dict({
            "method_name": "long_context",
            "method_type": "baseline",
            "model": {"provider": "openrouter", "name": "openai/gpt-4o"},
        }),
        dataset_config=DatasetConfig(dataset_name="locomo"),
        output_dir=tmp_path,
        batch_api=True,
        batch_gcs_uri="gs://ignored/value",
    )

    assert evaluator.batch_api is True
    assert evaluator.batch_gcs_uri is None
    assert evaluator._batch_gcs_skipped is True


def test_openrouter_request_includes_provider_routing_and_service_tier():
    captured = {}
    client = object.__new__(OpenRouterClient)
    client.model = "openai/gpt-5.1"
    client.temperature = 0.2
    client.max_tokens = 500
    client.provider_routing = {
        "only": ["azure"],
        "allow_fallbacks": False,
    }
    client.service_tier = "flex"
    client.client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **kwargs: captured.update(kwargs) or SimpleNamespace(
                    choices=[SimpleNamespace(
                        message=SimpleNamespace(content="ok", refusal=None),
                        finish_reason="stop",
                    )],
                    usage=SimpleNamespace(
                        prompt_tokens=4,
                        completion_tokens=5,
                        completion_tokens_details=SimpleNamespace(reasoning_tokens=3),
                    ),
                )
            )
        )
    )

    response = client.chat([{"role": "user", "content": "hello"}])

    assert response.content == "ok"
    assert captured["model"] == "openai/gpt-5.1"
    assert captured["extra_body"] == {
        "provider": {
            "only": ["azure"],
            "allow_fallbacks": False,
        },
        "service_tier": "flex",
    }
    assert (response.output_tokens, response.visible_output_tokens, response.thinking_tokens) == (5, 2, 3)


@pytest.mark.parametrize(
    ("openrouter", "message"),
    [
        ([], "model.openrouter must be a mapping"),
        ({"provider": "azure"}, "model.openrouter.provider must be a mapping"),
        ({"service_tier": 1}, "model.openrouter.service_tier must be a string"),
    ],
)
def test_openrouter_yaml_settings_validate_types(openrouter, message):
    with pytest.raises(ValueError, match=message):
        MethodConfig.from_dict({
            "method_name": "long_context",
            "method_type": "baseline",
            "model": {
                "provider": "openrouter",
                "name": "openai/gpt-5.1",
                "openrouter": openrouter,
            },
        })


def test_openrouter_judge_uses_provider_defaults():
    config = APIConfig(
        openai_api_key="must-not-be-used",
        openrouter_api_key="openrouter-key",
        openrouter_base_url="https://openrouter.example/api/v1",
        judge_provider="openrouter",
    )

    assert config.get_judge_api_key() == "openrouter-key"
    assert config.get_judge_base_url() == "https://openrouter.example/api/v1"


def test_usage_breakdown_normalizes_gemini_and_aggregate_provider_schemas():
    assert extract_usage_token_counts({
        "promptTokenCount": 7,
        "candidatesTokenCount": 10,
        "thoughtsTokenCount": 6,
    }) == (7, 10, 4, 6)
    assert extract_usage_token_counts(
        SimpleNamespace(input_tokens=9, output_tokens=3)
    ) == (9, 3, 3, 0)
