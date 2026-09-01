import json
from types import SimpleNamespace

import pytest

from methods.event_state.compiler import StateCompiler, parse_json
from methods.event_state.context import render_claim
from methods.event_state.schemas import Claim, Episode, EvidenceRef, TurnEvidence
from methods.event_state.store import EventStateStore
from methods.event_state.subjects import resolve_subject_id
from methods.event_state_agent import EventStateAgent
from methods.event_state.validation import resolve_model_turn_reference, resolve_model_turn_reference_with_form
from benchmarks.medmemorybench.checkpoint import compute_build_config_hash


class Embedder:
    def embed_documents(self, texts):
        return [[1.0, 0.0] for _ in texts]

    def embed_query(self, text):
        return [1.0, 0.0]


def test_attribute_subjects_are_never_fabricated_speakers():
    assert resolve_subject_id("allergic_to", "primary_user", ["User"], "User") == "primary_user"
    assert resolve_subject_id("lives_in", "primary_user", ["User"], "User") == "primary_user"
    assert resolve_subject_id("Alice", "primary_user", ["Alice", "Bob"], "Alice") == "speaker:alice"
    assert resolve_subject_id("patient", "third_party:father", ["User"], "User") == "third_party:father"
    assert resolve_subject_id("lives_in", "general_non_personal", ["User"], "User") == "general_non_personal"


def test_prefixed_subject_proposals_require_visible_identity_and_safe_display():
    class LLM:
        def chat(self, messages, **kwargs):
            return SimpleNamespace(content=json.dumps({"episode_summary": "allergy", "claims": [{"subject": "speaker:allergic_to", "subject_id": "speaker:allergic_to", "predicate": "allergic_to", "value": "cefuroxime", "source_turn_ids": ["u1"]}]}))

    agent = EventStateAgent(llm_client=LLM(), memory_llm_client=LLM(), embedding_client=Embedder())
    agent.memorize("", source_session_id=1, memory_items=[{"speaker": "User", "text": "I am allergic to cefuroxime.", "source_turn_id": "u1"}])
    claim = next(iter(agent._store().claims.values()))
    assert (claim.subject_id, claim.subject) == ("primary_user", "User")
    assert "speaker:allergic_to" not in agent.export_memory_state()["claims"][0]["subject_id"]
    assert resolve_subject_id("third_party:blood_pressure", "third_party:father", ["User"], "User") == "third_party:father"
    assert resolve_subject_id("Blood Glucose", "general_non_personal", ["User"], "User") == "general_non_personal"
    assert resolve_subject_id("speaker:Alice", "primary_user", ["Alice", "Bob"], "Alice") == "speaker:alice"


def _same_session_store(turns):
    store = EventStateStore("p")
    store.add_episode(Episode("s1", "p", "s1", 0, None, None, ["User"], "primary_user", "", "", turns), [1.0, 0.0])
    old = Claim("OLD", "User", "primary_user", "dose", "5 mg", evidence=[EvidenceRef("s1", "s1", [1])])
    store.add_claim(old, [1.0, 0.0])
    return store


def test_same_session_correction_and_unrelated_markers_reach_classifier():
    turns = [
        TurnEvidence(1, "User", "user", "I take 5 mg."),
        TurnEvidence(2, "User", "user", "Sorry, I meant 10 mg."),
        TurnEvidence(3, "User", "user", "The same problem happened again yesterday; I am not nauseated."),
    ]
    store = _same_session_store(turns)
    payloads = []

    class LLM:
        def chat(self, messages, **kwargs):
            payloads.append(json.loads(messages[-1]["content"]))
            return SimpleNamespace(content='{"operation":"SUPERSEDE","matched_claim_id":"OLD","same_state_dimension":true,"same_episode_relation":"correction","confidence":1}')

    compiler = StateCompiler(store, Embedder(), LLM())
    result = compiler.apply(Claim("NEW", "User", "primary_user", "dose", "10 mg", evidence=[EvidenceRef("s1", "s1", [2])]), "s1", [1.0, 0.0])
    assert result.operation == "SUPERSEDE"
    assert [turn["turn_id"] for turn in payloads[0]["new_claim_source_turns"]] == [2]
    assert [turn["turn_id"] for turn in payloads[0]["candidates"][0]["candidate_claim_source_turns"]] == [1]


def test_repair_grounding_telemetry_does_not_count_json_only_repair():
    class LLM:
        def __init__(self):
            self.responses = iter([
                "not json",
                json.dumps({"episode_summary": "fact", "claims": [{"subject": "User", "predicate": "lives_in", "value": "Boston", "source_turn_ids": ["u1"]}]}),
            ])

        def chat(self, messages, **kwargs):
            return SimpleNamespace(content=next(self.responses))

    llm = LLM()
    agent = EventStateAgent(llm_client=llm, memory_llm_client=llm, embedding_client=Embedder())
    result = agent.memorize("", source_session_id=1, memory_items=[{"speaker": "User", "text": "I live in Boston.", "source_turn_id": "u1"}])
    assert result.extra["extract_repair_calls"] == 1
    assert result.extra["repaired_source_turn_id_claim_count"] == 0


def test_repair_grounding_telemetry_counts_only_repaired_claims():
    class LLM:
        def __init__(self):
            self.responses = iter([
                json.dumps({"episode_summary": "fact", "claims": [{"subject": "User", "predicate": "lives_in", "value": "Boston", "source_turn_ids": ["missing"]}]}),
                json.dumps({"episode_summary": "fact", "claims": [{"subject": "User", "predicate": "lives_in", "value": "Boston", "source_turn_ids": ["u1"]}]}),
            ])

        def chat(self, messages, **kwargs):
            return SimpleNamespace(content=next(self.responses))

    llm = LLM()
    agent = EventStateAgent(llm_client=llm, memory_llm_client=llm, embedding_client=Embedder())
    result = agent.memorize("", source_session_id=1, memory_items=[{"speaker": "User", "text": "I live in Boston.", "source_turn_id": "u1"}])
    assert result.extra["repaired_source_turn_id_claim_count"] == 1


def test_parse_json_ignores_reasoning_blocks_and_selects_structured_payload():
    parsed = parse_json('<think>There may be {"wrong": true}</think>\n```json\n{"claims": []}\n```')
    assert parsed == {"claims": []}


def test_pre_fix_snapshot_semantic_version_is_rejected():
    store = EventStateStore("p")
    snapshot = store.export()
    snapshot["schema_version"] = 3
    with pytest.raises(ValueError):
        EventStateStore.from_export(snapshot)
    snapshot["schema_version"] = 4
    snapshot["semantic_version"] = "2.5"
    try:
        EventStateStore.from_export(snapshot)
    except ValueError as exc:
        assert "semantic version" in str(exc)
    else:
        raise AssertionError("pre-fix snapshot was accepted")
    assert EventStateStore.from_export(store.export()).export()["semantic_version"] == "2.6"


@pytest.mark.parametrize(
    ("value", "allowed", "expected", "form"),
    [
        ("0", {"0", "1"}, "0", None),
        (0, {"0", "1"}, "0", None),
        ("[turn_id=0]", {"0"}, "0", "bracket_turn_id"),
        ("turn_id=0", {"0"}, "0", "assignment_turn_id"),
        ("turn_14", {"14"}, "14", "turn_prefix"),
        ("[turn_id=abc-17]", {"abc-17"}, "abc-17", "bracket_turn_id"),
        ("turn_14", {"turn_14"}, "turn_14", None),
        ("turn_14", {"14", "turn_14"}, "turn_14", None),
        ("[turn_id=turn_14]", {"turn_14"}, "turn_14", "bracket_turn_id"),
    ],
)
def test_model_turn_reference_resolution_is_exact_then_narrow(value, allowed, expected, form):
    assert resolve_model_turn_reference(value, allowed) == expected
    assert resolve_model_turn_reference_with_form(value, allowed)[1] == form


def test_model_turn_reference_resolution_rejects_unknown_and_off_by_one():
    assert resolve_model_turn_reference("turn_99", {"0", "1"}) is None
    assert resolve_model_turn_reference("2", {"0", "1"}) is None
    assert resolve_model_turn_reference("turn_id=0", {"0", "id=0"}) is None


def test_event_state_semantic_version_changes_build_hash_only():
    class Config:
        method_name = "event_state"
        method_type = "agentic_memory"
        embedding = None
        memorize_model = None

        def __init__(self, version):
            self.build_config = {"event_state_semantic_version": version}

        def snapshot_build_config(self):
            return self.build_config

    dataset = SimpleNamespace(dataset_name="synthetic")
    assert compute_build_config_hash(Config("2.5"), dataset) != compute_build_config_hash(Config("2.6"), dataset)


def test_preparation_omits_slot_embeddings_for_non_state_claims():
    class RecordingEmbedder(Embedder):
        def __init__(self):
            self.document_calls = []

        def embed_documents(self, texts):
            self.document_calls.append(list(texts))
            return super().embed_documents(texts)

    class LLM:
        def chat(self, messages, **kwargs):
            return SimpleNamespace(content=json.dumps({
                "episode_summary": "state and history",
                "claims": [
                    {"subject": "User", "predicate": "lives_in", "value": "Boston", "persistence": "state", "state_slot": "residence_location", "source_turn_ids": ["u1"]},
                    {"subject": "User", "predicate": "former_employer", "value": "Acme", "persistence": "history", "source_turn_ids": ["u1"]},
                ],
            }))

    embedder = RecordingEmbedder()
    agent = EventStateAgent(llm_client=LLM(), memory_llm_client=LLM(), embedding_client=embedder)
    prepared = agent.prepare_memory_sessions("", source_session_id="s1", memory_items=[{"speaker": "User", "text": "I live in Boston and formerly worked at Acme.", "source_turn_id": "u1"}])[0]
    history_index = next(index for index, claim in enumerate(prepared.claims) if claim.persistence == "history")
    assert prepared.claim_slot_embeddings[history_index] == []
    assert "" not in [text for call in embedder.document_calls for text in call]


def test_state_slot_is_compiler_only_metadata():
    claim = Claim("C", "User", "primary_user", "lives_in", "Boston", state_slot="residence_location")
    assert "State slot" not in claim.semantic_text()
    assert StateCompiler.slot_text(claim) == "residence location"
    assert "State slot" not in render_claim(claim, [], {claim.claim_id: claim})


def test_update_response_format_fallback_preserves_generation_settings():
    calls = []

    class Client:
        def chat(self, messages, **kwargs):
            calls.append(kwargs)
            if "response_format" in kwargs:
                raise TypeError("response_format is unsupported")
            return SimpleNamespace(content=json.dumps({
                "matched_claim_id": "OLD",
                "operation": "SUPERSEDE",
                "same_state_dimension": True,
                "same_episode_relation": "none",
                "confidence": 1.0,
                "rationale": "changed",
            }))

    store = EventStateStore("p")
    store.add_claim(Claim("OLD", "User", "primary_user", "city", "Boston", state_slot="residence_location", evidence=[EvidenceRef("s1", "s1", [1])]), [1.0, 0.0], [1.0, 0.0])
    compiler = StateCompiler(store, Embedder(), Client(), min_similarity=0.1, update_temperature=0.2, update_max_tokens=123)
    result = compiler.apply(Claim("NEW", "User", "primary_user", "city", "Tokyo", state_slot="residence_location", evidence=[EvidenceRef("s2", "s2", [2])]), "s2", [1.0, 0.0], [1.0, 0.0])
    assert result.operation == "SUPERSEDE"
    assert calls[0]["temperature"] == 0.2 and calls[0]["max_tokens"] == 123
    assert "response_format" not in calls[1]
    assert calls[1]["temperature"] == 0.2 and calls[1]["max_tokens"] == 123


def test_update_repair_response_format_fallback_preserves_generation_settings():
    calls = []

    class Client:
        def chat(self, messages, **kwargs):
            calls.append(kwargs)
            if "response_format" in kwargs:
                raise TypeError("response_format is unsupported")
            content = "not json" if len(calls) == 2 else json.dumps({
                "matched_claim_id": "OLD", "operation": "SUPERSEDE",
                "same_state_dimension": True, "same_episode_relation": "none",
                "confidence": 1.0, "rationale": "changed",
            })
            return SimpleNamespace(content=content)

    store = EventStateStore("p")
    store.add_claim(Claim("OLD", "User", "primary_user", "city", "Boston", state_slot="residence_location", evidence=[EvidenceRef("s1", "s1", [1])]), [1.0, 0.0], [1.0, 0.0])
    compiler = StateCompiler(store, Embedder(), Client(), min_similarity=0.1, update_temperature=0.2, update_max_tokens=123)
    result = compiler.apply(Claim("NEW", "User", "primary_user", "city", "Tokyo", state_slot="residence_location", evidence=[EvidenceRef("s2", "s2", [2])]), "s2", [1.0, 0.0], [1.0, 0.0])
    assert result.operation == "SUPERSEDE"
    assert calls[2]["temperature"] == 0.2 and calls[2]["max_tokens"] == 123
    assert "response_format" not in calls[3]
    assert calls[3]["temperature"] == 0.2 and calls[3]["max_tokens"] == 123


def test_ungrounded_claim_is_dropped_but_episode_is_retained():
    class LLM:
        def __init__(self):
            self.calls = 0

        def chat(self, messages, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return SimpleNamespace(content=json.dumps({"episode_summary": "fact", "claims": [{"subject": "patient", "predicate": "city", "value": "Boston", "source_turn_ids": ["missing"]}]}))
            return SimpleNamespace(content="not json")

    llm = LLM()
    agent = EventStateAgent(llm_client=llm, memory_llm_client=llm, embedding_client=Embedder())
    result = agent.memorize("I live in Boston.", source_session_id=1, memory_items=[{"speaker": "User", "text": "I live in Boston.", "source_turn_id": "t1"}])
    assert len(agent._store().episodes) == 1
    assert not agent._store().claims
    assert result.extra["unknown_source_turn_id_claim_count"] == 1
    assert result.extra["rejected_ungrounded_claim_count"] == 1


def test_wrapper_references_preserve_mixed_multi_turn_claims_without_repair():
    class LLM:
        def __init__(self):
            self.calls = 0

        def chat(self, messages, **kwargs):
            self.calls += 1
            return SimpleNamespace(content=json.dumps({
                "episode_summary": "preferences",
                "claims": [
                    {"subject": "User", "predicate": "editor", "value": "vim", "source_turn_ids": ["[turn_id=0]"]},
                    {"subject": "User", "predicate": "deadline", "value": "Friday", "source_turn_ids": ["turn_2"]},
                ],
            }))

    llm = LLM()
    agent = EventStateAgent(llm_client=llm, memory_llm_client=llm, embedding_client=Embedder(), enable_state_compilation=False)
    result = agent.memorize("wrapper", source_session_id=1, memory_items=[
        {"role": "user", "content": "I use vim."},
        {"role": "assistant", "content": "Noted."},
        {"role": "user", "content": "The deadline is Friday."},
    ])
    assert llm.calls == 1
    assert {claim.evidence[0].source_turn_ids[0] for claim in agent._store().claims.values()} == {"0", "2"}
    assert result.extra["normalized_source_turn_reference_count"] == 2
    assert result.extra["unknown_source_turn_id_claim_count"] == 0


def test_invalid_claim_subset_does_not_delete_valid_claims_when_repair_fails():
    class LLM:
        def __init__(self):
            self.calls = 0

        def chat(self, messages, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return SimpleNamespace(content=json.dumps({"episode_summary": "facts", "claims": [
                    {"subject": "User", "predicate": "editor", "value": "vim", "source_turn_ids": ["0"]},
                    {"subject": "User", "predicate": "deadline", "value": "Friday", "source_turn_ids": ["turn_99"]},
                    {"subject": "User", "predicate": "shell", "value": "zsh", "source_turn_ids": ["2"]},
                ]}))
            return SimpleNamespace(content="not json")

    llm = LLM()
    agent = EventStateAgent(llm_client=llm, memory_llm_client=llm, embedding_client=Embedder(), enable_state_compilation=False)
    result = agent.memorize("wrapper", source_session_id=1, memory_items=[
        {"role": "user", "content": "I use vim."},
        {"role": "assistant", "content": "Noted."},
        {"role": "user", "content": "I use zsh."},
    ])
    assert len(agent._store().claims) == 2
    assert result.extra["valid_claims_preserved_despite_other_claim_errors"] == 2
    assert result.extra["invalid_claim_subset_repair_calls"] == 1
    assert result.extra["invalid_claim_subset_repair_failures"] == 1


def test_invalid_claim_subset_repair_merges_only_newly_valid_claims():
    class LLM:
        def __init__(self):
            self.calls = 0

        def chat(self, messages, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return SimpleNamespace(content=json.dumps({"episode_summary": "facts", "claims": [
                    {"subject": "User", "predicate": "editor", "value": "vim", "source_turn_ids": ["0"]},
                    {"subject": "User", "predicate": "deadline", "value": "Friday", "source_turn_ids": ["turn_99"]},
                ]}))
            return SimpleNamespace(content=json.dumps({"claims": [
                {"subject": "User", "predicate": "editor", "value": "vim", "source_turn_ids": ["0"]},
                {"subject": "User", "predicate": "deadline", "value": "Friday", "source_turn_ids": ["1"]},
            ]}))

    llm = LLM()
    agent = EventStateAgent(llm_client=llm, memory_llm_client=llm, embedding_client=Embedder(), enable_state_compilation=False)
    result = agent.memorize("wrapper", source_session_id=1, memory_items=[
        {"role": "user", "content": "I use vim."},
        {"role": "user", "content": "The deadline is Friday."},
    ])
    assert len(agent._store().claims) == 2
    assert result.extra["invalid_claim_subset_repair_successes"] == 1
    assert result.extra["repaired_source_turn_id_claim_count"] == 1


def test_literal_external_wrapper_like_source_id_is_preserved():
    class LLM:
        def chat(self, messages, **kwargs):
            return SimpleNamespace(content=json.dumps({"episode_summary": "fact", "claims": [
                {"subject": "User", "predicate": "editor", "value": "vim", "source_turn_ids": ["turn_2"]},
            ]}))

    agent = EventStateAgent(llm_client=LLM(), memory_llm_client=LLM(), embedding_client=Embedder(), enable_state_compilation=False)
    result = agent.memorize("wrapper", source_session_id=1, memory_items=[{"role": "user", "content": "I use vim.", "source_turn_id": "turn_2"}])
    claim = next(iter(agent._store().claims.values()))
    assert claim.evidence[0].source_turn_ids == ["turn_2"]
    assert result.extra["normalized_source_turn_reference_count"] == 0


def test_multi_source_grounding_is_atomic():
    class LLM:
        def chat(self, messages, **kwargs):
            if "claim subset" in messages[-1]["content"]:
                return SimpleNamespace(content="not json")
            return SimpleNamespace(content=json.dumps({"episode_summary": "fact", "claims": [
                {"subject": "User", "predicate": "editor", "value": "vim", "source_turn_ids": ["0", "turn_99"]},
            ]}))

    agent = EventStateAgent(llm_client=LLM(), memory_llm_client=LLM(), embedding_client=Embedder())
    result = agent.memorize("wrapper", source_session_id=1, memory_items=[{"role": "user", "content": "I use vim."}])
    assert not agent._store().claims
    assert result.extra["rejected_ungrounded_claim_count"] == 1


def test_same_session_restatement_is_classified_from_source_turns():
    store = EventStateStore("p")
    claim = Claim("OLD", "User", "primary_user", "retinal_status", "present", evidence=[EvidenceRef("s1", "s1", [1])])
    store.add_claim(claim, [1.0, 0.0])
    from methods.event_state.schemas import Episode, TurnEvidence
    store.add_episode(Episode("s1", "p", "s1", 0, None, None, ["User"], "primary_user", "", "", [TurnEvidence(1, "User", "user", "mild retinal changes are present"), TurnEvidence(2, "User", "user", "the retinal changes were recalled")]), [1.0, 0.0])
    captured = []
    compiler = StateCompiler(store, Embedder(), SimpleNamespace(chat=lambda messages, **kwargs: (captured.append(messages[-1]["content"]) or SimpleNamespace(content='{"operation":"DUPLICATE","matched_claim_id":"OLD","same_state_dimension":true,"same_episode_relation":"restatement","confidence":1}'))))
    repeated = Claim("NEW", "User", "primary_user", "retinal_status", "recalled", evidence=[EvidenceRef("s1", "s1", [2])])
    assert compiler.apply(repeated, "s1", [1.0, 0.0]).operation == "DUPLICATE"
    assert "new_claim_source_turns" in captured[0]
    assert "candidate_claim_source_turns" in captured[0]
