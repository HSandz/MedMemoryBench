from types import SimpleNamespace

import pytest

from methods.event_state_agent import EventStateAgent
from methods.event_state.retrieval import EventStateRetriever, normalize_scores
from methods.event_state.schemas import Claim, EvidenceRef
from methods.event_state.store import EventStateStore
from methods.event_state.subjects import normalize_scope, resolve_subject_id
from methods.event_state.validation import validated_claim
from methods.event_state.context import fit_context
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
    assert resolve_subject_id("Mary", "general_non_personal") == "general_non_personal"
    assert resolve_subject_id("I", "primary_user", ["Alice"], "Alice") == "speaker:alice"
    assert resolve_subject_id("primary_user", "primary_user") == "primary_user"
    assert resolve_subject_id("the user", "primary_user") == "primary_user"
    assert resolve_subject_id("speaker:Alice", "primary_user") == "primary_user"
    assert resolve_subject_id("third_party:Mary", "primary_user") == "primary_user"


def test_claim_source_speaker_comes_from_referenced_turns():
    session = {"source_session_id": 1}
    turns = [
        EventStateAgent.normalize_turn({"speaker": "Alice", "text": "I went to Boston", "source_turn_id": "a"}, 0, session),
        EventStateAgent.normalize_turn({"speaker": "Bob", "text": "I moved to Seattle", "source_turn_id": "b"}, 1, session),
    ]
    assert EventStateAgent.resolve_claim_source_speaker({"subject": "I", "source_turn_ids": ["a"]}, turns) == "Alice"
    assert EventStateAgent.resolve_claim_source_speaker({"subject": "I", "source_turn_ids": ["b"]}, turns) == "Bob"
    assert EventStateAgent.resolve_claim_source_speaker({"subject": "I", "source_turn_ids": ["a", "b"]}, turns) is None


def test_locomo_self_references_are_scoped_to_their_source_speaker():
    class LLM:
        def chat(self, messages, **kwargs):
            return SimpleNamespace(content='{"episode_summary":"two speakers","claims":[{"subject":"I","predicate":"visited","value":"Boston","source_turn_ids":["a"]},{"subject":"I","predicate":"moved","value":"Seattle","source_turn_ids":["b"]}]}')
    agent = EventStateAgent(llm_client=LLM(), memory_llm_client=LLM(), embedding_client=Embedder())
    agent.memorize("", memory_items=[{"speaker": "Alice", "text": "I visited Boston", "source_turn_id": "a"}, {"speaker": "Bob", "text": "I moved to Seattle", "source_turn_id": "b"}], source_session_id=1)
    subjects = {claim.predicate: claim.subject_id for claim in agent._store().claims.values()}
    assert subjects == {"visited": "speaker:alice", "moved": "speaker:bob"}


def test_incorrect_llm_subject_id_cannot_override_source_speaker_or_family_scope():
    class LLM:
        def __init__(self, content): self.content = content
        def chat(self, messages, **kwargs): return SimpleNamespace(content=self.content)
    bob_agent = EventStateAgent(llm_client=LLM('{"episode_summary":"s","claims":[{"subject":"I","subject_id":"speaker:alice","predicate":"moved","value":"Seattle","source_turn_ids":["b"]}]}'), memory_llm_client=LLM('{"episode_summary":"s","claims":[{"subject":"I","subject_id":"speaker:alice","predicate":"moved","value":"Seattle","source_turn_ids":["b"]}]}'), embedding_client=Embedder())
    bob_agent.memorize("", memory_items=[{"speaker":"Alice","text":"hello","source_turn_id":"a"},{"speaker":"Bob","text":"I moved","source_turn_id":"b"}], source_session_id=1)
    assert next(iter(bob_agent._store().claims.values())).subject_id == "speaker:bob"
    family_agent = EventStateAgent(llm_client=LLM('{"episode_summary":"s","claims":[{"subject":"Mary","subject_id":"primary_user","predicate":"dose","value":"10 mg"}]}'), memory_llm_client=LLM('{"episode_summary":"s","claims":[{"subject":"Mary","subject_id":"primary_user","predicate":"dose","value":"10 mg"}]}'), embedding_client=Embedder())
    family_agent.memorize("[Health consultation record about Mary(mother)]", source_session_id=2)
    assert next(iter(family_agent._store().claims.values())).subject_id == "third_party:mary"


def test_ambiguous_multi_speaker_self_reference_is_dropped():
    class LLM:
        def chat(self, messages, **kwargs): return SimpleNamespace(content='{"episode_summary":"s","claims":[{"subject":"I","predicate":"travelled","value":"often","source_turn_ids":["a","b"]}]}')
    agent = EventStateAgent(llm_client=LLM(), memory_llm_client=LLM(), embedding_client=Embedder())
    result = agent.memorize("", memory_items=[{"speaker":"Alice","text":"I travel","source_turn_id":"a"},{"speaker":"Bob","text":"I travel","source_turn_id":"b"}], source_session_id=1)
    assert not agent._store().claims
    assert result.extra["ambiguous_subject_claim_count"] == 1


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


def test_structural_extraction_failures_repair_once_but_empty_claims_are_valid():
    class Structural:
        def __init__(self): self.calls = 0
        def chat(self, messages, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return SimpleNamespace(content='{"episode_summary":"s","claims":{}}')
            return SimpleNamespace(content='{"episode_summary":"s","claims":[]}')
    client = Structural()
    agent = EventStateAgent(llm_client=client, memory_llm_client=client, embedding_client=Embedder())
    result = agent.memorize("text", source_session_id=1)
    assert client.calls == 2 and result.extra["extract_structure_failures"] == 1

    class Empty:
        def chat(self, messages, **kwargs): return SimpleNamespace(content='{"episode_summary":"s","claims":[]}')
    empty = EventStateAgent(llm_client=Empty(), memory_llm_client=Empty(), embedding_client=Embedder())
    result = empty.memorize("text", source_session_id=1)
    assert result.extra["extract_repair_calls"] == 0 and result.extra["extract_structure_failures"] == 0


def test_bitemporal_disabled_clears_normalized_boundaries():
    class LLM:
        def chat(self, messages, **kwargs): return SimpleNamespace(content='{"episode_summary":"s","claims":[{"subject":"patient","predicate":"dose","value":"10 mg","valid_from":"2024-01-01","valid_to":"2024-02-01","valid_time_text":"last month"}]}')
    agent = EventStateAgent(llm_client=LLM(), memory_llm_client=LLM(), embedding_client=Embedder(), enable_bitemporal_time=False)
    agent.memorize("text", source_session_id=1)
    claim = next(iter(agent._store().claims.values()))
    assert claim.valid_from is None and claim.valid_to is None and claim.valid_time_text == "last month"


def test_state_claims_require_episode_archive_for_provenance():
    with pytest.raises(ValueError, match="requires enable_episodes"):
        EventStateAgent(enable_episodes=False, enable_state_claims=True, embedding_client=Embedder())


def test_schema_v1_is_rejected_and_context_fallback_uses_stored_id():
    store = EventStateStore("stored")
    with pytest.raises(ValueError, match="schema v1"):
        EventStateStore.from_export({"method": "event_state", "schema_version": 1})
    with pytest.raises(ValueError, match="schema v2"):
        EventStateStore.from_export({"method": "event_state", "schema_version": 2})
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
    assert all("included_in_context" in record for record in prepared["retrieved_memories"])


def test_numeric_and_string_turn_ids_share_one_grounding_contract():
    class LLM:
        def __init__(self):
            self.calls = 0

        def chat(self, messages, **kwargs):
            self.calls += 1
            return SimpleNamespace(content='{"episode_summary":"summary","claims":[{"subject":"patient","predicate":"dose","value":"10 mg","source_turn_ids":[0]}]}')

    llm = LLM()
    agent = EventStateAgent(llm_client=llm, memory_llm_client=llm, embedding_client=Embedder())
    result = agent.memorize("dose", memory_items=[{"role": "user", "content": "dose", "source_turn_id": "0"}], source_session_id=1)
    claim = next(iter(agent._store().claims.values()))
    assert claim.evidence[0].source_turn_ids == ["0"]
    assert result.extra["unknown_source_turn_id_claim_count"] == 0
    assert result.extra["extract_repair_calls"] == 0 and llm.calls == 1


def test_production_role_content_items_generate_grounded_fallback_ids():
    captured = []

    class LLM:
        def chat(self, messages, **kwargs):
            captured.append(messages[-1]["content"])
            return SimpleNamespace(content='{"episode_summary":"medication","claims":[{"subject":"User","subject_id":"primary_user","predicate":"takes_medication","value":"metformin","source_turn_ids":[0]}]}')

    agent = EventStateAgent(llm_client=LLM(), memory_llm_client=LLM(), embedding_client=Embedder())
    result = agent.memorize("", memory_items=[
        {"role": "user", "content": "I take metformin every day."},
        {"role": "assistant", "content": "Understood."},
    ], source_session_id=1)
    claim = next(iter(agent._store().claims.values()))
    episode = next(iter(agent._store().episodes.values()))
    assert result.extra["accepted_claim_count"] == 1
    assert result.extra["total_claim_count"] == 1
    assert result.extra["primary_user_claim_count"] == 1
    assert result.extra["unknown_source_turn_id_claim_count"] == 0
    assert result.extra["extract_repair_calls"] == 0
    assert result.extra["extract_repair_failures"] == 0
    assert claim.evidence[0].source_turn_ids == ["0"]
    assert [turn.turn_id for turn in episode.turn_evidence] == ["0", "1"]
    assert "[turn_id=0]" in captured[0] and "[turn_id=1]" in captured[0]


def test_claim_limit_skips_invalid_excess_claims_and_allows_zero():
    valid = '{"subject":"patient","predicate":"fact","value":"x","source_turn_ids":["0"]}'
    invalid = '{"subject":"patient","predicate":"bad","value":"x","source_turn_ids":["missing"]}'

    class LLM:
        def __init__(self, content):
            self.content, self.calls = content, 0

        def chat(self, messages, **kwargs):
            self.calls += 1
            return SimpleNamespace(content=self.content)

    limited_llm = LLM('{"episode_summary":"s","claims":[' + valid + ',' + invalid + ']}')
    limited = EventStateAgent(llm_client=limited_llm, memory_llm_client=limited_llm, embedding_client=Embedder(), max_claims_per_episode=1)
    result = limited.memorize("x", memory_items=[{"role": "user", "content": "x"}], source_session_id=1)
    assert result.extra["accepted_claim_count"] == 1
    assert result.extra["excess_claim_count"] == 1
    assert result.extra["unknown_source_turn_id_claim_count"] == 0
    assert result.extra["extract_repair_calls"] == 0 and limited_llm.calls == 1

    zero_llm = LLM('{"episode_summary":"s","claims":[' + valid + ']}')
    zero = EventStateAgent(llm_client=zero_llm, memory_llm_client=zero_llm, embedding_client=Embedder(), max_claims_per_episode=0)
    result = zero.memorize("x", memory_items=[{"role": "user", "content": "x"}], source_session_id=1)
    assert result.extra["accepted_claim_count"] == 0
    assert result.extra["excess_claim_count"] == 1
    assert result.extra["extract_repair_calls"] == 0 and zero_llm.calls == 1


def test_canonical_turn_id_collision_is_namespaced_before_extraction():
    turns = EventStateAgent.normalize_turns({"source_session_id": 1, "turns": [
        {"speaker": "User", "text": "first", "source_turn_id": 1},
        {"speaker": "User", "text": "second", "source_turn_id": "1"},
    ]})
    assert [turn.source_turn_id for turn in turns] == ["1", "1__turn_1"]


def test_extraction_claim_limit_is_deterministic_without_repair():
    claims = ",".join('{"subject":"patient","predicate":"fact%d","value":"x","source_turn_ids":["0"]}' % i for i in range(4))

    class LLM:
        def __init__(self):
            self.calls = 0

        def chat(self, messages, **kwargs):
            self.calls += 1
            return SimpleNamespace(content='{"episode_summary":"summary","claims":[' + claims + ']}')

    llm = LLM()
    agent = EventStateAgent(llm_client=llm, memory_llm_client=llm, embedding_client=Embedder(), max_claims_per_episode=2, enable_state_compilation=False)
    result = agent.memorize("facts", memory_items=[{"role": "user", "content": "facts", "source_turn_id": "0"}], source_session_id=1)
    assert result.extra["claims_extracted"] == 2
    assert result.extra["excess_claim_count"] == 2
    assert result.extra["extract_repair_calls"] == 0 and llm.calls == 1


def test_failed_extraction_fallback_covers_late_turns_and_is_bounded():
    class LLM:
        def chat(self, messages, **kwargs):
            return SimpleNamespace(content="not json")

    items = [{"role": "user", "content": f"turn {index} " + ("x" * 220)} for index in range(6)]
    items[-1]["content"] += " FINAL-LATE-PHRASE"
    agent = EventStateAgent(llm_client=LLM(), memory_llm_client=LLM(), embedding_client=Embedder())
    agent.memorize("wrapper", memory_items=items, source_session_id=1)
    summary = next(iter(agent._store().episodes.values())).summary
    assert "FINAL-LATE-PHRASE" in summary
    assert len(summary) <= 1000


def test_extraction_lifecycle_telemetry_is_reported_before_compilation():
    class LLM:
        def chat(self, messages, **kwargs):
            return SimpleNamespace(content='{"episode_summary":"summary","claims":['
                '{"subject":"User","predicate":"lives_in","value":"Boston","persistence":"state","source_turn_ids":["0"]},'
                '{"subject":"User","predicate":"worked_at","value":"Acme","persistence":"history","source_turn_ids":["0"]},'
                '{"subject":"User","predicate":"visited","value":"Paris","persistence":"episode","source_turn_ids":["0"]}]}')

    agent = EventStateAgent(llm_client=LLM(), memory_llm_client=LLM(), embedding_client=Embedder())
    result = agent.memorize("facts", memory_items=[{"role": "user", "content": "facts", "source_turn_id": "0"}], source_session_id=1)
    assert result.extra["extracted_state_claim_count"] == 1
    assert result.extra["extracted_history_claim_count"] == 1
    assert result.extra["extracted_episode_claim_count"] == 1
    assert result.extra["standalone_history_count"] == 1
    assert result.extra["standalone_episode_count"] == 1


def test_context_budget_accounts_for_instruction_system_question_and_reserve():
    count = lambda text: len(text.split())
    truncate = lambda text, limit: " ".join(text.split()[:limit])
    blocks, memory_tokens = fit_context(
        [{"text": "STATE metadata stays complete", "kind": "state"}, {"text": "source one two three four five six", "kind": "source"}],
        "system words",
        "instruction words",
        "complete question words",
        18,
        2,
        count,
        truncate,
    )
    assert blocks[0] == "STATE metadata stays complete"
    assert blocks[1].startswith("source") and len(blocks[1].split()) < 7
    final_user = "instruction words\n\n" + "\n\n".join(blocks) + "\n\ncomplete question words"
    assert count("system words") + count(final_user) + 2 <= 18
    assert final_user.endswith("complete question words")
