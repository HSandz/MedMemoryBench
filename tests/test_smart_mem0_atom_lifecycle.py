"""Capture/reconciliation contracts, using a deterministic offline model stub."""

from copy import deepcopy

import pytest

from test_smart_mem0_agent import _bare_agent, JsonClient


def atom(value="8.1%", **changes):
    return dict({"claim": "Recorded value " + value, "kind": "STATE",
                 "subject_id": "primary_user", "scope": "measurement",
                 "state_key": "hba1c", "value": value, "stance": "AFFIRM",
                 "assertion_mode": "DIRECT", "source_turns": [0],
                 "event_time": "2024-01-01", "facets": {"modality": "observed"}}, **changes)


def build(agent, payload, timestamp="2024-01-01"):
    agent._llm_client = JsonClient({"memories": payload})
    agent._effective_runtime_config = lambda: {}
    agent._capsule_seq = getattr(agent, "_capsule_seq", 0)
    result = agent.memorize("source", metadata={"timestamp": timestamp}, memory_items=[{
        "text": "A focal source statement", "speaker_name": "patient", "timestamp": timestamp,
    }])
    assert agent._llm_client.calls == 1
    return result


@pytest.mark.parametrize("value", ["8.1%", "after lunch", "食後", "sau bữa trưa"])
def test_every_extracted_atom_survives_without_reconciliation_call(value):
    agent = _bare_agent()
    first = build(agent, [atom(value)])
    second = build(agent, [atom(value, assertion_mode="RECAP")], "2024-02-01")
    assert len(agent._memories) == 2
    assert [m["value"] for m in agent._memories] == [value, value]
    assert agent._memories[-1]["assertion_mode"] == "RECAP"
    assert agent._memories[-1]["origin_document_time"] == ""
    assert all(m["memory_tier"] == "HOT" for m in agent._memories)
    assert agent._memories[-1]["facets"] == {"modality": "observed"}
    assert first.extra["reconciliation_llm_calls"] == second.extra["reconciliation_llm_calls"] == 0
    assert all(row["disposition"] == "STORED" and row["target_id"] for row in agent._atom_dispositions)


def test_invalid_source_has_explicit_disposition_and_does_not_hide_valid_atom():
    agent = _bare_agent()
    result = build(agent, [atom(), atom("9%", source_turns=[999]), {"claim": ""}])
    dispositions = result.extra["atom_dispositions"]
    assert [d["disposition"] for d in dispositions] == ["STORED", "DISCARDED", "DISCARDED"]
    assert [d["reason"] for d in dispositions[1:]] == ["INVALID_FOCAL_PROVENANCE", "EMPTY_CLAIM"]
    assert len(agent._memories) == 1


def test_facets_and_dispositions_round_trip_in_snapshot():
    agent = _bare_agent()
    build(agent, [atom()])
    saved = agent.export_memory_state()
    restored = _bare_agent()
    restored.import_memory_state(saved)
    assert restored.export_memory_state()["memories"] == saved["memories"]
    assert restored._atom_dispositions == saved["atom_dispositions"]


def test_reconciliation_never_uses_document_time_as_event_time():
    agent = _bare_agent()
    build(agent, [atom(event_time="UNKNOWN")])
    build(agent, [atom("9%", event_time="UNKNOWN")], "2024-02-01")
    assert not any(r["type"] == "SUPERSEDE" for r in agent._relations)
    assert any(r["type"] == "CONFLICT" for r in agent._relations)
    assert {m["value"] for m in agent._memories} == {"8.1%", "9%"}


def test_explicit_state_transition_keeps_history_and_versions_without_regex():
    agent = _bare_agent()
    build(agent, [atom(claim="以前の値")])
    build(agent, [atom("9%", claim="新しい値", event_time="2024-02-01")], "2024-02-01")
    assert len(agent._memories) == 2
    assert any(r["type"] == "SUPERSEDE" for r in agent._relations)
    assert agent._belief_status["m_1"] == "superseded"


def test_different_modalities_do_not_version_each_other():
    agent = _bare_agent()
    build(agent, [atom()])
    build(agent, [atom("9%", event_time="2024-02-01", facets={"modality": "proposed"})], "2024-02-01")
    assert not any(r["type"] in {"SUPERSEDE", "CONFLICT"} for r in agent._relations)


def test_failed_commit_rolls_back_disposition_and_ledger(monkeypatch):
    agent = _bare_agent()
    build(agent, [atom()])
    before = deepcopy(agent.export_memory_state())
    monkeypatch.setattr(agent, "_refresh_index", lambda: (_ for _ in ()).throw(RuntimeError("index failure")))
    with pytest.raises(RuntimeError, match="index failure"):
        build(agent, [atom("9%")], "2024-02-01")
    assert agent.export_memory_state() == before
