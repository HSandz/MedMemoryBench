"""Answer-visible retrieval regressions for Event-State."""

from types import SimpleNamespace

from benchmarks.medmemorybench.evaluator import MedMemoryBenchEvaluator
from methods.event_state.schemas import Episode
from methods.event_state.store import EventStateStore
from methods.event_state_agent import EventStateAgent


def _answer_visible_quality(records, session_id):
    evaluator = object.__new__(MedMemoryBenchEvaluator)
    evaluator.method_config = SimpleNamespace(method_name="event_state", build_config={})
    query = SimpleNamespace(source_key_points=[{"session_id": session_id}], metadata={})
    return evaluator._session_retrieval_quality(query, records)["answer_visible"]


def test_event_state_snapshot_messages_use_the_event_state_label():
    evaluator = MedMemoryBenchEvaluator.__new__(MedMemoryBenchEvaluator)
    evaluator.method_config = SimpleNamespace(method_name="event_state")

    assert evaluator._memory_method_label() == "Event-State"


def test_planner_budget_and_record_telemetry_are_unambiguous():
    class LLM:
        def __init__(self):
            self.calls = []

        def chat(self, messages, **kwargs):
            self.calls.append(messages)
            return SimpleNamespace(content='{"action":"answer","answer":"ok","requests":[]}')

    llm = LLM()
    agent = EventStateAgent(llm_client=llm, memory_llm_client=llm, embedding_client=SimpleNamespace(embed_query=lambda _: [1.0, 0.0]), planner_rounds=1)
    agent.set_context_id(1)
    agent.query("What happened?")
    prompt = llm.calls[0][-1]["content"]
    assert "retrieval rounds available including this decision: 1" in prompt
    assert "remaining retrieval rounds: 0" not in prompt

    store = EventStateStore("ctx")
    for identifier in ("A", "C", "D"):
        store.add_episode(Episode(identifier, "ctx", identifier, 0, None, None, ["User"], "primary_user", identifier, identifier, []), [1.0, 0.0])
    records = [
        agent._record(store, {"id": "A", "type": "episode", "planner_request_indices": [0], "planner_added_to_final": False}),
        agent._record(store, {"id": "C", "type": "episode", "planner_request_indices": [], "planner_added_to_final": False}),
        agent._record(store, {"id": "D", "type": "episode", "planner_request_indices": [0], "planner_added_to_final": True}),
    ]
    assert records[0]["planner_channel_retrieval"] is True
    assert records[0]["planner_added_to_final"] is False
    assert records[0]["planner_retrieval"] is True
    assert records[2]["planner_added_to_final"] is True


def test_answer_visible_ignores_state_excluded_without_visible_provenance():
    quality = _answer_visible_quality([
        {
            "type": "state_claim",
            "source_session_id": 7,
            "included_in_context": False,
            "included_provenance_evidence": [],
        },
    ], 7)

    assert quality["available"] is True
    assert quality["retrieved_note_count"] == 0
    assert quality["predicted_session_ids"] == []
    assert "unavailable_reason" not in quality


def test_answer_visible_counts_direct_origin_for_included_state_block():
    quality = _answer_visible_quality([
        {
            "type": "state_claim",
            "source_session_id": 7,
            "included_in_context": True,
            "included_provenance_evidence": [],
        },
    ], 7)

    assert quality["available"] is True
    assert quality["predicted_session_ids"] == ["7"]
    assert quality["f1"] == 1.0


def test_answer_visible_counts_only_included_provenance_for_excluded_state():
    quality = _answer_visible_quality([
        {
            "type": "state_claim",
            "source_session_id": 7,
            "included_in_context": False,
            "included_provenance_evidence": [
                {"evidence": {"source_session_id": 8}},
            ],
        },
    ], 8)

    assert quality["available"] is True
    assert quality["predicted_session_ids"] == ["8"]
    assert quality["f1"] == 1.0


def test_answer_visible_ignores_selected_episode_excluded_by_context_budget():
    quality = _answer_visible_quality([
        {
            "type": "episode",
            "source_session_id": 7,
            "included_in_context": False,
        },
    ], 7)

    assert quality["available"] is True
    assert quality["retrieved_note_count"] == 0
    assert quality["predicted_session_ids"] == []
