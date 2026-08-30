from types import SimpleNamespace

import methods.event_state_agent as module
from methods.event_state_agent import EventStateAgent


class FakeEmbedder:
    def embed_documents(self, texts):
        return [[1.0, 0.0] for _ in texts]

    def embed_query(self, text):
        return [1.0, 0.0]


class FakeLLM:
    def chat(self, messages):
        system = messages[0]["content"]
        if "Extract conversational memory" in system:
            return SimpleNamespace(content='{"episode_summary":"Alice discussed treatment", "claims":[{"subject":"Alice","predicate":"dose","value":"500 mg","source_turn_ids":["t1"]}]}')
        return SimpleNamespace(content='{"operation":"NEW","matched_claim_id":null,"confidence":1.0,"rationale":"new"}')


def test_normalization_groups_locomo_sessions_and_preserves_turn_provenance():
    sessions = EventStateAgent._normalize_source_sessions("", [{"source_session_id": 2, "speaker": "Bob", "content": "b", "dia_id": "d2"}, {"source_session_id": 1, "speaker": "Alice", "content": "a", "dia_id": "d1"}], {})
    assert [item["source_session_id"] for item in sessions] == [2, 1]
    assert sessions[0]["turns"][0]["source_turn_id"] == "d2"


def test_agent_build_query_snapshot_and_provenance(monkeypatch):
    monkeypatch.setattr(module, "create_llm_client", lambda **kwargs: FakeLLM())
    agent = EventStateAgent(model="query", memory_model="build", embedding_client=FakeEmbedder())
    agent.set_context_id("sample")
    result = agent.memorize("wrapper", memory_items=[{"source_session_id": 7, "speaker": "Alice", "content": "I take medicine", "source_turn_id": "t7"}], timestamp="2024-01-01")
    assert result.extra["episodes_added"] == 1
    prepared = agent.prepare_batch_query("What dose?", raw_question="What dose?")
    assert prepared["retrieved_memories"]
    assert prepared["retrieved_memories"][0].get("source_session_id") == 7
    state = agent.export_memory_state()
    clone = EventStateAgent(model="query", memory_model="build", embedding_client=FakeEmbedder())
    clone.import_memory_state(state, context_id="sample")
    assert clone.prepare_batch_query("What dose?")["retrieved_memories"] == prepared["retrieved_memories"]


def test_agent_routes_low_confidence_classifier_fallback_separately_from_extraction_confidence():
    outputs = iter([
        '{"episode_summary":"one","claims":[{"subject":"patient","predicate":"dose","value":"500 mg","confidence":0.95}]}',
        '{"episode_summary":"two","claims":[{"subject":"patient","predicate":"dose","value":"850 mg","confidence":0.95}]}',
    ])
    class LLM:
        def chat(self, messages, **kwargs):
            if "Extract conversational memory" in messages[0]["content"]:
                return SimpleNamespace(content=next(outputs))
            return SimpleNamespace(content='{"matched_claim_id":"missing","operation":"SUPERSEDE","confidence":0.3}')
    agent = EventStateAgent(llm_client=LLM(), memory_llm_client=LLM(), embedding_client=FakeEmbedder(), state_candidate_min_similarity=0.1)
    agent.memorize("one", source_session_id=1)
    result = agent.memorize("two", source_session_id=2)
    assert result.extra["low_confidence_new_count"] == 1
    assert result.extra["low_extraction_confidence_count"] == 0
