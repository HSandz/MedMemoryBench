from types import SimpleNamespace

import pytest

from methods.event_state_agent import EventStateAgent
from methods.event_state.retrieval import EventStateRetriever, normalize_scores
from methods.event_state.schemas import Claim, EvidenceRef
from methods.event_state.store import EventStateStore
from methods.event_state.subjects import normalize_scope, resolve_subject_id
from methods.event_state.validation import validated_claim
from utils.llm_client import LLMAPIError


class Embedder:
    def embed_documents(self, texts):
        return [[1.0, 0.0] for _ in texts]

    def embed_query(self, text):
        return [1.0, 0.0]


def test_role_only_turns_and_visible_ids_are_preserved():
    session = {"source_session_id": 3, "source_session_index": 8, "timestamp": "2024-03-01"}
    user = EventStateAgent.normalize_turn({"role": "user", "content": "fact", "source_turn_id": "t1", "blip_caption": "x"}, 0, session)
    assistant = EventStateAgent.normalize_turn({"role": "assistant", "content": "advice", "source_turn_id": "t2"}, 1, session)
    assert (user.speaker, assistant.speaker) == ("User", "Assistant")
    assert user.role == "user" and assistant.role == "assistant"
    assert user.source_turn_id == "t1" and user.image_caption == "x"


def test_scope_and_subject_aliases_do_not_contaminate_primary_user():
    assert normalize_scope("[Health consultation record about Mary(mother)]") == "third_party:mary"
    assert normalize_scope("[Health consultation record]") == "general_non_personal"
    assert resolve_subject_id("patient", "primary_user") == "primary_user"
    assert resolve_subject_id("the patient", "primary_user") == "primary_user"
    assert resolve_subject_id("Mary", "general_non_personal") == "third_party:mary"
    assert resolve_subject_id("I", "primary_user", ["Alice"], "Alice") == "speaker:alice"


def test_extraction_prompt_contains_turn_role_speaker_and_ids(monkeypatch):
    captured = []

    class LLM:
        def chat(self, messages, **kwargs):
            captured.append(messages[-1]["content"])
            return SimpleNamespace(content='{"episode_summary":"summary","claims":[]}')

    agent = EventStateAgent(llm_client=LLM(), memory_llm_client=LLM(), embedding_client=Embedder())
    agent.memorize("", memory_items=[{"role": "user", "content": "hello", "source_turn_id": "turn-7"}], source_session_id=1)
    assert "[turn_id=turn-7]" in captured[0]
    assert "[role=user]" in captured[0] and "[speaker=User]" in captured[0]


def test_raw_episode_contains_scope_roles_and_image_caption(monkeypatch):
    class LLM:
        def chat(self, messages, **kwargs):
            return SimpleNamespace(content='{"episode_summary":"summary","claims":[]}')

    agent = EventStateAgent(llm_client=LLM(), memory_llm_client=LLM(), embedding_client=Embedder())
    agent.memorize("[Health consultation record about Mary(mother)]", memory_items=[{"role": "user", "content": "pain", "source_turn_id": 1, "blip_caption": "scan"}], source_session_id=1)
    episode = next(iter(agent._store().episodes.values()))
    assert "conversation_scope=third_party:mary" in episode.raw_text
    assert "role=user" in episode.raw_text and "Shared image: scan" in episode.raw_text


def test_compilation_disabled_keeps_claims_and_evidence_edges():
    outputs = ['{"episode_summary":"s","claims":[{"subject":"patient","predicate":"dose","value":"10 mg"}]}', '{"episode_summary":"s","claims":[{"subject":"patient","predicate":"dose","value":"20 mg"}]}']
    class LLM:
        def chat(self, messages, **kwargs):
            return SimpleNamespace(content=outputs.pop(0))
    agent = EventStateAgent(llm_client=LLM(), memory_llm_client=LLM(), embedding_client=Embedder(), enable_state_compilation=False)
    agent.memorize("x", source_session_id=1)
    agent.memorize("y", source_session_id=2)
    assert agent._store().claim_counts()["total_claim_count"] == 2
    assert agent._store().edges and not agent._store().operations


def test_malformed_extraction_repairs_but_provider_error_propagates():
    class Repair:
        def __init__(self): self.calls = 0
        def chat(self, messages, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return SimpleNamespace(content="not json")
            return SimpleNamespace(content='{"episode_summary":"ok","claims":[]}')
    repair = Repair()
    agent = EventStateAgent(llm_client=repair, memory_llm_client=repair, embedding_client=Embedder())
    result = agent.memorize("text", source_session_id=1)
    assert result.extra["extract_repair_calls"] == 1
    assert repair.calls == 2

    class Failing:
        def chat(self, messages, **kwargs): raise LLMAPIError("provider unavailable")
    failing = EventStateAgent(llm_client=Failing(), memory_llm_client=Failing(), embedding_client=Embedder())
    with pytest.raises(LLMAPIError):
        failing.memorize("text", source_session_id=1)


def test_state_claims_require_episode_archive_for_provenance():
    with pytest.raises(ValueError, match="requires enable_episodes"):
        EventStateAgent(enable_episodes=False, enable_state_claims=True, embedding_client=Embedder())


def test_schema_v1_is_rejected_and_context_fallback_uses_stored_id():
    store = EventStateStore("stored")
    with pytest.raises(ValueError, match="schema v1"):
        EventStateStore.from_export({"method": "event_state", "schema_version": 1})
    agent = EventStateAgent(llm_client=SimpleNamespace(chat=lambda messages, **kwargs: SimpleNamespace(content="ok")), memory_llm_client=SimpleNamespace(chat=lambda messages, **kwargs: SimpleNamespace(content="ok")), embedding_client=Embedder())
    state = store.export()
    agent.import_memory_state(state)
    assert agent._context_id == "stored"


def test_mmr_normalizes_relevance_before_redundancy_penalty():
    assert normalize_scores([0.01, 0.02, 0.02]) == [0.0, 1.0, 1.0]
    store = EventStateStore("p")
    store.episode_embeddings = {"E1": [1, 0], "E2": [0.99, 0.01], "E3": [0, 1]}
    store.episodes = {key: SimpleNamespace(source_session_id=key) for key in store.episode_embeddings}
    retriever = EventStateRetriever(store, Embedder(), selector_mode="mmr", mmr_lambda=0.7)
    selected = retriever._select([{"id": "E1", "type": "episode", "final_score": 0.02}, {"id": "E2", "type": "episode", "final_score": 0.021}, {"id": "E3", "type": "episode", "final_score": 0.01}], 2)
    assert len(selected) == 2 and selected[0]["id"] == "E2"


def test_source_evidence_is_expanded_and_budget_keeps_question_intact():
    class LLM:
        def chat(self, messages, **kwargs):
            return SimpleNamespace(content='{"episode_summary":"summary","claims":[{"subject":"patient","predicate":"dose","value":"10 mg","source_turn_ids":["t1"]}]}')

    agent = EventStateAgent(llm_client=LLM(), memory_llm_client=LLM(), embedding_client=Embedder(), max_context_tokens=1000, max_tokens=5, evidence_count=1, candidate_count=2, retrieve_episodes=False)
    agent.memorize("wrapper", memory_items=[{"role": "user", "content": "I take ten milligrams daily.", "source_turn_id": "t1"}], source_session_id=4)
    prepared = agent.prepare_batch_query("What dose does the user take?", raw_question="What dose does the user take?")
    assert prepared["messages"][-1]["content"].endswith("What dose does the user take?")
    assert prepared["extra"]["included_provenance_evidence"]
