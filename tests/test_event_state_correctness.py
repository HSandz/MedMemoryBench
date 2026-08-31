import json
from types import SimpleNamespace

from methods.event_state.compiler import StateCompiler, parse_json
from methods.event_state.schemas import Claim, EvidenceRef
from methods.event_state.store import EventStateStore
from methods.event_state.subjects import resolve_subject_id
from methods.event_state_agent import EventStateAgent


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


def test_parse_json_ignores_reasoning_blocks_and_selects_structured_payload():
    parsed = parse_json('<think>There may be {"wrong": true}</think>\n```json\n{"claims": []}\n```')
    assert parsed == {"claims": []}


def test_pre_fix_snapshot_semantic_version_is_rejected():
    store = EventStateStore("p")
    snapshot = store.export()
    snapshot.pop("semantic_version")
    try:
        EventStateStore.from_export(snapshot)
    except ValueError as exc:
        assert "semantic version" in str(exc)
    else:
        raise AssertionError("pre-fix snapshot was accepted")


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


def test_same_session_restatement_is_duplicate_but_correction_is_classified():
    store = EventStateStore("p")
    claim = Claim("OLD", "User", "primary_user", "retinal_status", "present", evidence=[EvidenceRef("s1", "s1", [1])])
    store.add_claim(claim, [1.0, 0.0])
    from methods.event_state.schemas import Episode, TurnEvidence
    store.add_episode(Episode("s1", "p", "s1", 0, None, None, ["User"], "primary_user", "", "", [TurnEvidence(1, "User", "user", "mild retinal changes are present"), TurnEvidence(2, "User", "user", "the retinal changes were recalled")]), [1.0, 0.0])
    compiler = StateCompiler(store, Embedder(), SimpleNamespace(chat=lambda *args, **kwargs: SimpleNamespace(content='{"operation":"CONFLICT","matched_claim_id":"OLD","confidence":1}')))
    repeated = Claim("NEW", "User", "primary_user", "retinal_status", "recalled", evidence=[EvidenceRef("s1", "s1", [1])])
    assert compiler.apply(repeated, "s1", [1.0, 0.0]).operation == "DUPLICATE"
