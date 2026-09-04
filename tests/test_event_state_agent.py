from types import SimpleNamespace

import methods.event_state_agent as module
from methods.event_state_agent import EventStateAgent


class FakeEmbedder:
    def embed_documents(self, texts):
        return [[1.0, 0.0] for _ in texts]

    def embed_query(self, text):
        return [1.0, 0.0]


class FakeLLM:
    def chat(self, messages, **kwargs):
        system = messages[0]["content"]
        if "Extract conversational memory" in system:
            return SimpleNamespace(content='{"episode_summary":"Alice discussed treatment", "claims":[{"subject":"Alice","predicate":"dose","value":"500 mg","source_turn_ids":["t1"]}]}')
        return SimpleNamespace(content='{"operation":"NEW","state_value_relation":"uncertain","matched_claim_id":null,"confidence":1.0,"rationale":"new"}')


def test_normalization_groups_locomo_sessions_and_preserves_turn_provenance():
    sessions = EventStateAgent._normalize_source_sessions("", [{"source_session_id": 2, "source_session_index": 42, "speaker": "Bob", "content": "b", "dia_id": "d2"}, {"source_session_id": 1, "source_session_index": 7, "speaker": "Alice", "content": "a", "dia_id": "d1"}], {})
    assert [item["source_session_id"] for item in sessions] == [2, 1]
    assert sessions[0]["turns"][0]["source_turn_id"] == "d2"
    assert [item["source_session_index"] for item in sessions] == [42, 7]


def test_generic_conversation_scope_keeps_mixed_sources_isolated():
    class ScopeLLM:
        def chat(self, messages, **kwargs):
            if "Extract conversational memory" in messages[0]["content"]:
                return SimpleNamespace(content=(
                    '{"episode_summary":"fact","claims":[{"subject":"patient",'
                    '"predicate":"condition","value":"recorded",'
                    '"source_turn_ids":["t"]}]}'
                ))
            return SimpleNamespace(content='{"operation":"NEW","confidence":1}')

    agent = EventStateAgent(
        llm_client=ScopeLLM(),
        memory_llm_client=ScopeLLM(),
        embedding_client=FakeEmbedder(),
        enable_state_compilation=False,
    )
    for source_uid, scope in (
        ("src_p1_r0", "primary_user"),
        ("src_p1_r1", "general_non_personal"),
        ("src_p1_r2", "third_party:maya"),
    ):
        agent.memorize(
            "conversation",
            memory_items=[{
                "source_session_id": source_uid,
                "source_turn_id": "t",
                "speaker": "Patient",
                "content": "A medical fact.",
                "conversation_scope": scope,
            }],
        )

    state = agent.export_memory_state()
    assert len(state["episodes"]) == 3
    assert {item["source_session_id"] for item in state["episodes"]} == {
        "src_p1_r0", "src_p1_r1", "src_p1_r2",
    }
    assert {item["subject_id"] for item in state["claims"]} == {
        "primary_user", "general_non_personal", "third_party:maya",
    }
    assert agent._store().claim_counts()["duplicate_episode_source_id_count"] == 0


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
            return SimpleNamespace(content='{"matched_claim_id":"missing","operation":"SUPERSEDE","state_value_relation":"changed","confidence":0.3}')
    agent = EventStateAgent(llm_client=LLM(), memory_llm_client=LLM(), embedding_client=FakeEmbedder(), state_candidate_min_similarity=0.1)
    agent.memorize("one", source_session_id=1)
    result = agent.memorize("two", source_session_id=2)
    assert result.extra["low_confidence_new_count"] == 1
    assert result.extra["low_extraction_confidence_count"] == 0


def test_agent_reports_cross_session_equivalent_corroboration():
    extractions = iter([
        '{"episode_summary":"residence","claims":[{"subject":"User","subject_id":"primary_user","predicate":"lives_in","state_slot":"residence_location","value":"Boston","source_turn_ids":["0"]}]}',
        '{"episode_summary":"residence","claims":[{"subject":"User","subject_id":"primary_user","predicate":"current_residence","state_slot":"residence_location","value":"still living in Boston","source_turn_ids":["0"]}]}',
    ])

    class LLM:
        def chat(self, messages, **kwargs):
            if "Extract conversational memory" in messages[0]["content"]:
                return SimpleNamespace(content=next(extractions))
            return SimpleNamespace(content='{"matched_claim_id":"C8aa3ca6a3e18e442","operation":"CORROBORATE","state_value_relation":"equivalent","same_state_dimension":true,"same_episode_relation":"none","confidence":1}')

    llm = LLM()
    agent = EventStateAgent(llm_client=llm, memory_llm_client=llm, embedding_client=FakeEmbedder(), state_candidate_min_similarity=0.1)
    first = agent.memorize("Boston", source_session_id=1)
    claim_id = next(iter(agent._store().claims))
    class SecondSessionLLM:
        def chat(self, messages, **kwargs):
            if "Extract conversational memory" in messages[0]["content"]:
                return SimpleNamespace(content=next(extractions))
            return SimpleNamespace(content=f'{{"matched_claim_id":"{claim_id}","operation":"CORROBORATE","state_value_relation":"equivalent","same_state_dimension":true,"same_episode_relation":"none","confidence":1}}')

    agent._memory_llm_client = SecondSessionLLM()
    second = agent.memorize("Boston again", source_session_id=2)

    assert first.extra["new_count"] == 1
    assert second.extra["cross_session_corroborate_count"] == 1
    assert second.extra["claims_with_multiple_provenance_references"] == 1
    assert second.extra["claims_with_evidence_from_multiple_sessions"] == 1


def test_extraction_rejects_absence_placeholders_but_keeps_explicit_negatives():
    class ExtractionLLM:
        def chat(self, messages, **kwargs):
            if "Extract conversational memory" in messages[0]["content"]:
                return SimpleNamespace(content=(
                    '{"episode_summary":"preferences", "claims":['
                    '{"subject":"Alice","predicate":"lives_in","state_slot":"residence_location","value":"not specified","persistence":"state","source_turn_ids":["0"]},'
                    '{"subject":"Alice","predicate":"owns_car","state_slot":"car_ownership","value":"no","polarity":"negative","persistence":"state","source_turn_ids":["0"]},'
                    '{"subject":"Alice","predicate":"notification_preference","state_slot":"notification_preference","value":"none","persistence":"state","source_turn_ids":["0"]}'
                    ']}'))
            return SimpleNamespace(content='{"operation":"NEW","state_value_relation":"uncertain","confidence":1}')

    agent = EventStateAgent(llm_client=ExtractionLLM(), memory_llm_client=ExtractionLLM(), embedding_client=FakeEmbedder())
    agent.set_context_id("ctx")
    result = agent.memorize("I do not own a car and have no notification preference.", context_id="ctx")

    claims = agent.export_memory_state("ctx")["claims"]
    assert {item["predicate"] for item in claims} == {"owns_car", "notification_preference"}
    assert result.extra["no_information_claim_rejected_count"] == 1
    assert result.extra["state_claim_count_with_slot"] == 2
