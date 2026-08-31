"""Answer-visible retrieval regressions for Event-State."""

from types import SimpleNamespace

from benchmarks.medmemorybench.evaluator import MedMemoryBenchEvaluator


def _answer_visible_quality(records, session_id):
    evaluator = object.__new__(MedMemoryBenchEvaluator)
    evaluator.method_config = SimpleNamespace(method_name="event_state", build_config={})
    query = SimpleNamespace(source_key_points=[{"session_id": session_id}], metadata={})
    return evaluator._session_retrieval_quality(query, records)["answer_visible"]


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
