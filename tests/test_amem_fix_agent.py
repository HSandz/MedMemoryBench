"""Regression coverage for the paper-aligned A-Mem adapter flow."""

from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from methods.amem_agent import AMemAgent
from methods.amem_fix_agent import AMemFixAgent
from methods.base import AgentResponse
from src.agent import AgentManager
from src.config import DatasetConfig, MethodConfig
from utils.llm_client import EmptyGeminiResponseError, LLMRetryExhaustedError


class _Tokenizer:
    def encode(self, text: str):
        return text.split()

    def decode(self, tokens):
        return " ".join(tokens)


class _LLMClient:
    def count_tokens(self, text: str) -> int:
        return len((text or "").split())


def test_amem_fix_memorizes_atomic_turns_with_real_timestamp():
    calls = []
    memory_system = SimpleNamespace(
        add_note=lambda **kwargs: calls.append(kwargs) or f"note-{len(calls)}"
    )
    agent = object.__new__(AMemFixAgent)
    agent._context_id = 7
    agent._memory_chunks = []
    agent._is_initialized = False
    agent.retrieve_num = 10
    agent.amem_chunk_size_tokens = 10240
    agent._llm_client = _LLMClient()
    agent._get_memory_system = lambda context_id: memory_system

    result = agent.memorize(
        "formatted wrapper that must not be stored",
        timestamp="2025-01-02",
        memory_items=[
            {"role": "user", "content": "My cough is improving."},
            {"role": "assistant", "content": "Continue the medication."},
        ],
    )

    assert calls == [
        {"content": "Speaker Patient says: My cough is improving.", "time": "2025-01-02"},
        {"content": "Speaker Doctor says: Continue the medication.", "time": "2025-01-02"},
    ]
    assert result.method == "amem_fix"
    assert result.extra["turns_received"] == 2
    assert result.extra["notes_created"] == 2
    assert "formatted wrapper" not in result.stored_content


def test_amem_fix_uses_raw_question_keywords_and_expands_links_before_batch_answer():
    keyword_requests = []
    search_queries = []
    memories = {
        "m0": SimpleNamespace(
            content="allergy to penicillin",
            context="Medication allergy",
            keywords=["penicillin", "allergy"],
            tags=["allergy"],
            timestamp="2024-01-01",
            links=[1, "m2", 99],
        ),
        "m1": SimpleNamespace(
            content="amoxicillin was avoided",
            context="Treatment decision",
            keywords=["amoxicillin"],
            tags=["medication"],
            timestamp="2024-02-01",
            links=[],
        ),
        "m2": SimpleNamespace(
            content="alternative antibiotic prescribed",
            context="Treatment",
            keywords=["antibiotic"],
            tags=["treatment"],
            timestamp="2024-02-01",
            links=[],
        ),
    }
    memory_system = SimpleNamespace(
        memories=memories,
        retriever=SimpleNamespace(
            search=lambda query, k: search_queries.append((query, k)) or [0]
        ),
    )
    agent = object.__new__(AMemFixAgent)
    agent._context_id = 4
    agent.retrieve_num = 10
    agent.amem_query_keywords = True
    agent.amem_expand_links = True
    agent.amem_max_context_tokens = 2000
    agent.max_tokens = 100
    agent._tokenizer = _Tokenizer()
    agent._llm_client = SimpleNamespace(
        count_tokens=_LLMClient().count_tokens,
        chat=lambda messages: keyword_requests.append(messages)
        or SimpleNamespace(content='{"keywords": "penicillin, antibiotic allergy"}'),
    )
    agent._get_memory_system = lambda context_id: memory_system

    formatted_question = "Answer from memory. Question: Which antibiotic must be avoided? Answer:"
    prepared = agent.prepare_batch_query(
        formatted_question,
        system_message="system",
        raw_question="Which antibiotic must be avoided?",
    )
    response = agent.finalize_batch_query(prepared, "penicillin")
    agent.record_batch_query_usage(response, 25, 2)

    keyword_prompt = keyword_requests[0][-1]["content"]
    assert "Which antibiotic must be avoided?" in keyword_prompt
    assert "Answer from memory" not in keyword_prompt
    assert search_queries == [("penicillin, antibiotic allergy", 10)]
    assert prepared["extra"]["direct_indices"] == [0]
    assert prepared["extra"]["expanded_indices"] == [0, 1, 2]
    assert prepared["messages"][-1]["content"].endswith(formatted_question)
    assert response.retrieved_count == 3
    assert response.retrieved_memories[1]["linked_expansion"] is True
    assert response.extra["tokens_used"] == {"input": 25, "output": 2}


def test_amem_fix_does_not_hide_exhausted_keyword_api_calls():
    failure = LLMRetryExhaustedError(
        "API call still failed",
        last_exception=EmptyGeminiResponseError("empty response"),
        attempts=100,
    )
    agent = object.__new__(AMemFixAgent)
    agent.amem_query_keywords = True
    agent._llm_client = SimpleNamespace(
        chat=lambda messages: (_ for _ in ()).throw(failure)
    )

    with pytest.raises(LLMRetryExhaustedError):
        agent._generate_retrieval_query("question")


def test_gemini_build_client_is_not_initialized_for_query_only_restore(monkeypatch):
    calls = []
    robust_module = importlib.import_module("memory_layer_robust")

    monkeypatch.setattr(
        "utils.llm_client.create_llm_client",
        lambda **kwargs: calls.append(kwargs)
        or SimpleNamespace(chat=lambda messages, **options: SimpleNamespace(content="ready")),
    )
    controller = robust_module.RobustGeminiController(
        provider="ai_studio",
        model="gemini-test",
    )

    assert controller.client is None
    assert calls == []
    assert controller.get_completion("query-time build-independent prompt") == "ready"
    assert calls == [{
        "provider": "ai_studio",
        "model": "gemini-test",
        "temperature": 0.7,
        "max_tokens": 1000,
        "reasoning_effort": None,
    }]


def test_amem_evolution_does_not_store_after_exhausted_api_call():
    failure = LLMRetryExhaustedError(
        "API call still failed",
        last_exception=EmptyGeminiResponseError("empty response"),
        attempts=100,
    )
    memory_system_class = AMemAgent._load_amem_system_class(object())
    memory_system = object.__new__(memory_system_class)
    memory_system.max_context_chars = 1000
    memory_system.find_related_memories = lambda query, k: ("neighbor memory", [0])
    memory_system.llm_controller = SimpleNamespace(
        llm=SimpleNamespace(
            get_completion=lambda prompt: (_ for _ in ()).throw(failure)
        )
    )
    note = SimpleNamespace(
        id="note-1",
        content="patient memory",
        context="clinical context",
        keywords=["patient"],
    )

    with pytest.raises(LLMRetryExhaustedError):
        memory_system.process_memory(note)


def test_amem_embedding_model_ids_are_not_rewritten_as_paths():
    config = MethodConfig.from_dict({
        "method_name": "amem_fix",
        "method_type": "agentic_memory",
        "agent_params": {"amem_embedding_model": "BAAI/bge-small-zh-v1.5"},
    })
    local_config = MethodConfig.from_dict({
        "method_name": "amem_fix",
        "method_type": "agentic_memory",
        "agent_params": {"amem_embedding_model": "./models/amem-encoder"},
    })

    assert config.agent_params["amem_embedding_model"] == "BAAI/bge-small-zh-v1.5"
    assert Path(local_config.agent_params["amem_embedding_model"]).is_absolute()


def test_all_amem_fix_configs_keep_paper_defaults():
    config_paths = list(Path("configs/method_config").glob("amem_fix_*.yaml"))
    config_paths += list(Path("configs/method_config/persona_1").glob("amem_fix_*.yaml"))

    assert config_paths
    for path in config_paths:
        config = MethodConfig.from_dict(yaml.safe_load(path.read_text()))
        assert config.method_name == "amem_fix"
        assert config.model.temperature == 0.0
        assert config.model.max_completion_tokens == 2000
        assert config.agent_params["retrieve_num"] == 10
        assert config.agent_params["amem_embedding_model"] == "all-MiniLM-L6-v2"
        assert config.agent_params["amem_max_tokens"] == 1000
        assert config.agent_params["amem_query_keywords"] is True
        assert config.agent_params["amem_expand_links"] is True


def test_agent_manager_routes_amem_fix_before_amem_and_forwards_metadata():
    assert list(AgentManager.SUPPORTED_METHODS).index("amem_fix") < list(
        AgentManager.SUPPORTED_METHODS
    ).index("amem")

    calls = {}

    class _Agent:
        memory_size = 0

        def set_context_id(self, context_id):
            calls["context_id"] = context_id

        def memorize(self, message, **kwargs):
            calls["memorize"] = (message, kwargs)
            return {"success": True}

        def query(self, message, system_message=None, **kwargs):
            calls["query"] = (message, system_message, kwargs)
            return AgentResponse(output="answer")

    manager = object.__new__(AgentManager)
    manager._agent = _Agent()
    manager._context_id = None
    manager.method_name = "amem_fix"
    manager.dataset_config = DatasetConfig(dataset_name="medmemorybench")

    manager.send_message(
        "memory",
        memorizing=True,
        context_id=2,
        timestamp="2025-01-02",
        memory_items=[{"role": "user", "content": "fact"}],
    )
    manager.send_message(
        "formatted question",
        raw_question="raw question",
        query_type="entity_exact_match",
    )

    assert calls["memorize"][1]["timestamp"] == "2025-01-02"
    assert calls["memorize"][1]["memory_items"][0]["content"] == "fact"
    assert calls["query"][2] == {
        "raw_question": "raw question",
        "query_type": "entity_exact_match",
    }


def test_amem_fix_remains_eligible_for_vertex_final_answer_batching():
    manager = object.__new__(AgentManager)
    manager.method_config = MethodConfig.from_dict({
        "method_name": "amem_fix",
        "method_type": "agentic_memory",
        "model": {"provider": "gemini", "name": "gemini-2.5-flash"},
    })
    manager._agent = SimpleNamespace(
        prepare_batch_query=lambda *args, **kwargs: {},
        finalize_batch_query=lambda *args, **kwargs: AgentResponse(output="answer"),
    )

    assert manager.supports_batch_queries() is True
