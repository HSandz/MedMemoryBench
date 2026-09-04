"""Regression coverage for MedMemoryBench method-facing source identities."""

import json

import pytest

from benchmarks.base import EvaluationUnit
from benchmarks.medmemorybench.dataset import MedMemoryBenchDataset
from benchmarks.medmemorybench.dataset import MedSession
from benchmarks.medmemorybench.evaluator import MedMemoryBenchEvaluator


def _write_fixture(tmp_path, sessions):
    eval_dir = tmp_path / "persona_1" / "eval"
    eval_dir.mkdir(parents=True)
    (eval_dir / "generated_dialogues_with_noise.json").write_text(
        json.dumps({"sessions": sessions}), encoding="utf-8"
    )
    (eval_dir / "generated_dialogues.json").write_text(
        json.dumps({"sessions": [item for item in sessions if "session_id" in item]}),
        encoding="utf-8",
    )
    (eval_dir / "generated_queries.json").write_text(
        json.dumps({"queries": [
            {"query_id": "q1", "question": "one", "query_type": "test", "session_id": 1},
            {"query_id": "q2", "question": "two", "query_type": "test", "session_id": 2},
        ]}), encoding="utf-8"
    )


def test_mixed_source_uids_preserve_order_and_clean_checkpoints(tmp_path):
    # Positions 0 and 2 previously collided with clean benchmark IDs 1 and 2.
    sessions = [
        {"noise_id": "n0", "messages": [{"role": "user", "content": "general fact"}]},
        {"session_id": 1, "messages": [{"role": "user", "content": "clean one"}]},
        {"noise_family_id": "n2", "family_role": {"name": "Maya"}, "messages": [{"role": "user", "content": "family fact"}]},
        {"session_id": 2, "messages": [{"role": "user", "content": "clean two"}]},
    ]
    _write_fixture(tmp_path, sessions)
    dataset = MedMemoryBenchDataset(tmp_path, {
        "evaluation_mode": "independent",
        "evaluation_interval": 1,
        "inject_noise": True,
    })
    dataset.load()

    loaded = dataset._personas[1]["sessions"]
    assert [item.source_uid for item in loaded] == [
        "src_p1_r0", "src_p1_r1", "src_p1_r2", "src_p1_r3",
    ]
    assert [item.benchmark_session_id for item in loaded] == [None, 1, None, 2]
    assert [item.conversation_scope for item in loaded] == [
        "general_non_personal", "primary_user", "third_party:maya", "primary_user",
    ]
    assert dataset.get_source_identity_mapping(1) == {
        "src_p1_r0": None,
        "src_p1_r1": 1,
        "src_p1_r2": None,
        "src_p1_r3": 2,
    }

    units = list(dataset.get_evaluation_units())
    assert [[item.source_uid for item in unit.sessions_to_inject] for unit in units] == [
        ["src_p1_r0", "src_p1_r1"],
        ["src_p1_r2", "src_p1_r3"],
    ]
    assert [unit.metadata["eval_session_id"] for unit in units] == [1, 2]


def test_evaluator_rejects_duplicate_method_source_uids_before_ingestion():
    evaluator = object.__new__(MedMemoryBenchEvaluator)
    evaluator._source_uid_to_benchmark_session_id = {}
    unit = EvaluationUnit(
        unit_id=4,
        context_id=1,
        sessions_to_inject=[
            MedSession(session_id="src_p1_r0", source_uid="src_p1_r0", content="one"),
            MedSession(session_id="src_p1_r0", source_uid="src_p1_r0", content="two"),
        ],
        queries_to_evaluate=[],
    )

    with pytest.raises(ValueError, match="source UID collision"):
        evaluator._validate_unit_source_integrity(unit)
