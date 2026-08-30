"""Tests for exact per-unit A-MEM snapshot and staged-query execution."""

from __future__ import annotations

import importlib
import json
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks.base import EvaluationUnit
from benchmarks.medmemorybench.evaluator import MedMemoryBenchEvaluator
from methods.amem_agent import AMemAgent
from methods.amem_test_agent import AMemTestAgent
from src.result import EvaluationReport, ResultCollector


AMEM_DIR = Path(__file__).resolve().parents[1] / "methods" / "amem" / "A-mem"
if str(AMEM_DIR) not in sys.path:
    sys.path.insert(0, str(AMEM_DIR))


class _FakeEmbeddingModel:
    def encode(self, texts):
        vectors = []
        for text in texts:
            text = str(text).lower()
            vectors.append([
                float(text.count("heart")),
                float(text.count("kidney")),
                float(text.count("lung")),
            ])
        return np.asarray(vectors, dtype=np.float32)


def _note(note_class, memory_id: str, content: str, timestamp: str):
    return note_class(
        content=content,
        id=memory_id,
        keywords=[content.split()[0]],
        links=[],
        importance_score=0.75,
        retrieval_count=2,
        timestamp=timestamp,
        last_accessed=timestamp,
        context="medical",
        evolution_history=[{"event": "created"}],
        category="condition",
        tags=["test"],
    )


def _snapshot_agent(system, *, agent_class=AMemAgent, context_id=4):
    agent = agent_class.__new__(agent_class)
    agent.amem_embedding_model = "fake-embedding"
    agent.amem_evo_threshold = 7
    agent.amem_max_tokens = 1000
    agent.amem_temperature = 0.0
    agent.amem_retry_temperature = 0.0
    agent.amem_connectivity_temperature = 0.0
    agent.amem_build_max_context_tokens = 1234
    agent.amem_max_context_tokens = 1234
    agent.amem_chunk_size_tokens = 321
    if agent_class is AMemTestAgent:
        agent.amem_note_level = "turn"
    agent._amem_backend = "openai"
    agent._amem_model = "build-model"
    agent._amem_systems = {context_id: system}
    agent._memory_chunks = [note.content for note in system.memories.values()]
    agent._is_initialized = True
    agent._context_id = context_id
    agent._amem_class = system.__class__
    agent._get_memory_system = lambda context_id: agent._amem_systems[context_id]
    return agent


def _robust_system(system_class=SimpleNamespace):
    robust_module = importlib.import_module("memory_layer_robust")
    note_class = robust_module.RobustMemoryNote
    system = system_class(
        memories={
            "m1": _note(note_class, "m1", "heart disease", "202401010900"),
            "m2": _note(note_class, "m2", "kidney disease", "202401020900"),
        },
        retriever=SimpleNamespace(
            model=_FakeEmbeddingModel(),
            corpus=["heart disease", "kidney disease"],
            embeddings=np.asarray([[1, 0, 0], [0, 1, 0]], dtype=np.float32),
            document_ids={"heart disease": 0, "kidney disease": 1},
        ),
        evo_cnt=3,
        evo_threshold=7,
        max_context_chars=1851,
    )
    return system


def _search(system, query: str, k: int = 2):
    from sklearn.metrics.pairwise import cosine_similarity

    query_embedding = system.retriever.model.encode([query])[0]
    similarities = cosine_similarity([query_embedding], system.retriever.embeddings)[0]
    return np.argsort(similarities)[-k:][::-1].tolist()


def test_amem_state_round_trip_preserves_order_fields_and_embeddings():
    original = _robust_system()
    agent = _snapshot_agent(original)
    state = agent.export_memory_state(context_id=4)

    original.memories = {}
    original.retriever.corpus = []
    original.retriever.embeddings = None
    agent.import_memory_state(state, context_id=4)

    restored = agent._amem_systems[4]
    assert list(restored.memories) == ["m1", "m2"]
    assert vars(restored.memories["m1"]) == state["system_state"]["memories"][0]["attributes"]
    assert restored.retriever.corpus == ["heart disease", "kidney disease"]
    assert restored.retriever.document_ids == {"heart disease": 0, "kidney disease": 1}
    np.testing.assert_array_equal(
        restored.retriever.embeddings,
        np.asarray([[1, 0, 0], [0, 1, 0]], dtype=np.float32),
    )
    assert restored.retriever.embeddings.dtype == np.float32
    assert _search(restored, "kidney") == [1, 0]


def test_amem_state_import_rejects_changed_build_semantics():
    system = _robust_system()
    agent = _snapshot_agent(system)
    state = agent.export_memory_state(context_id=4)
    agent._amem_model = "different-build-model"

    with pytest.raises(ValueError, match="configuration does not match"):
        agent.import_memory_state(state, context_id=4)


def test_amem_query_import_ignores_build_controller_settings():
    system = _robust_system()
    agent = _snapshot_agent(system)
    state = agent.export_memory_state(context_id=4)
    agent.amem_query_only = True
    agent._amem_model = "different-query-build-controller"
    agent.amem_evo_threshold = 999
    agent.amem_chunk_size_tokens = 9999

    agent.import_memory_state(state, context_id=4)


def test_amem_typed_state_round_trip_preserves_relation_indexes_and_audit():
    typed_system_class = type("TypedRelationMemorySystem", (SimpleNamespace,), {})
    system = _robust_system(typed_system_class)
    system.original_evolution_enabled = False
    system.typed_relations_enabled = True
    system.relation_candidate_count = 5
    edge = {
        "source_id": "m2",
        "target_id": "m1",
        "relation_type": "REFINE",
        "confidence": 0.9,
    }
    system.memories["m2"].typed_relations_out = [dict(edge)]
    system.memories["m1"].typed_relations_in = [dict(edge)]
    system.typed_relations = [dict(edge)]
    system._typed_edge_keys = {("m2", "m1")}
    system.relation_audit = [{"created_memory_id": "m2"}]
    system._relation_audit_by_memory = {"m2": system.relation_audit[0]}
    agent = _snapshot_agent(system)

    state = agent.export_memory_state(context_id=4)
    system.typed_relations = []
    system._typed_edge_keys = set()
    system.relation_audit = []
    system._relation_audit_by_memory = {}
    agent.import_memory_state(state, context_id=4)

    restored = agent._amem_systems[4]
    assert restored.typed_relations == [edge]
    assert restored._typed_edge_keys == {("m2", "m1")}
    assert restored.relation_audit == [{"created_memory_id": "m2"}]
    assert restored._relation_audit_by_memory == {"m2": {"created_memory_id": "m2"}}
    assert restored.memories["m2"].typed_relations_out == [edge]
    assert restored.memories["m1"].typed_relations_in == [edge]


def test_amem_experimental_state_round_trip_preserves_temporal_and_provenance():
    typed_system_class = type("TypedRelationMemorySystem", (SimpleNamespace,), {})
    system = _robust_system(typed_system_class)
    system.original_evolution_enabled = False
    system.typed_relations_enabled = True
    system.temporal_state_enabled = True
    system.provenance_enabled = True
    system.relation_candidate_count = 5
    system.relation_temperature = 0.2
    system.temporal_min_confidence = 0.5
    system.typed_relations = []
    system._typed_edge_keys = set()
    system.relation_audit = []
    system._relation_audit_by_memory = {}
    system.temporal_audit = [{"source_id": "m2", "target_id": "m1"}]
    system._temporal_audit_by_memory = {
        "m1": list(system.temporal_audit),
        "m2": list(system.temporal_audit),
    }
    system.evidence_store = {
        "ev_1": {
            "evidence_id": "ev_1",
            "raw_text": "exact raw statement",
            "source_session_id": 1,
            "source_timestamp": "2024-01-01",
        }
    }
    system.provenance_audit = [{"memory_id": "m1", "evidence_ids": ["ev_1"]}]
    system._provenance_audit_by_memory = {
        "m1": system.provenance_audit[0],
    }
    system.memories["m1"].temporal_state = {
        "status": "superseded",
        "valid_from": "2024-01-01",
        "valid_until": "2024-02-01",
        "superseded_by": ["m2"],
    }
    system.memories["m1"].provenance = {
        "memory_id": "m1",
        "evidence_ids": ["ev_1"],
    }
    agent = _snapshot_agent(system, agent_class=AMemTestAgent)
    agent.amem_original_evolution = False
    agent.amem_typed_relations = True
    agent.amem_temporal_state = True
    agent.amem_provenance = True

    state = agent.export_memory_state(context_id=4)
    system.temporal_audit = []
    system.evidence_store = {}
    agent.import_memory_state(state, context_id=4)

    restored = agent._amem_systems[4]
    assert restored.temporal_state_enabled is True
    assert restored.provenance_enabled is True
    assert restored.temporal_audit == [{"source_id": "m2", "target_id": "m1"}]
    assert restored.evidence_store["ev_1"]["raw_text"] == "exact raw statement"
    assert restored.memories["m1"].temporal_state["status"] == "superseded"
    assert restored.memories["m1"].provenance["evidence_ids"] == ["ev_1"]


def test_amem_test_snapshot_declares_enabled_experimental_features():
    system = _robust_system()
    system.original_evolution_enabled = False
    system.typed_relations_enabled = True
    system.temporal_state_enabled = True
    system.provenance_enabled = True
    agent = _snapshot_agent(system, agent_class=AMemTestAgent)
    agent.amem_original_evolution = False
    agent.amem_typed_relations = True
    agent.amem_temporal_state = True
    agent.amem_provenance = True

    state = agent.export_memory_state(context_id=4)

    assert state["experimental_features"] == {
        "note_level": "turn",
        "original_evolution": False,
        "typed_relations": True,
        "temporal_state": True,
        "provenance": True,
        "provenance_inject_raw_text": False,
    }


def test_amem_test_snapshot_rejects_different_note_level():
    system = _robust_system()
    system.original_evolution_enabled = False
    system.typed_relations_enabled = False
    system.temporal_state_enabled = False
    system.provenance_enabled = False
    agent = _snapshot_agent(system, agent_class=AMemTestAgent)
    agent.amem_original_evolution = False
    agent.amem_typed_relations = False
    agent.amem_temporal_state = False
    agent.amem_provenance = False

    state = agent.export_memory_state(context_id=4)
    agent.amem_note_level = "session"

    with pytest.raises(ValueError, match="note level"):
        agent.import_memory_state(state, context_id=4)


def test_amem_test_accepts_legacy_turn_snapshot_without_note_level():
    system = _robust_system()
    system.original_evolution_enabled = False
    system.typed_relations_enabled = False
    system.temporal_state_enabled = False
    system.provenance_enabled = False
    agent = _snapshot_agent(system, agent_class=AMemTestAgent)
    agent.amem_original_evolution = False
    agent.amem_typed_relations = False
    agent.amem_temporal_state = False
    agent.amem_provenance = False

    state = agent.export_memory_state(context_id=4)
    state["config"].pop("note_level")
    state["experimental_features"].pop("note_level")

    agent.import_memory_state(state, context_id=4)


class _StageManager:
    def __init__(self):
        self.notes = []
        self.events = []

    def supports_memory_snapshots(self):
        return True

    def set_context_id(self, context_id):
        self.context_id = context_id

    def send_message(self, *, message, memorizing, **kwargs):
        assert memorizing is True
        self.events.append(f"memorize:{message}")
        self.notes.append(message)
        from methods.base import MemoryBuildResult
        return MemoryBuildResult(method="amem", time_cost=0.25)

    def export_memory_state(self, context_id=None):
        self.events.append("save")
        return {
            "format": "test",
            "version": 1,
            "context_id": context_id,
            "notes": list(self.notes),
        }

    def import_memory_state(self, state, context_id=None):
        self.events.append("load")
        self.notes = list(state["notes"])

    def prepare_query(self, *, message, **kwargs):
        self.events.append(f"retrieve:{message}:{'|'.join(self.notes)}")
        return {"message": message, "notes": list(self.notes)}

    def answer_prepared_query(self, staged_query):
        self.events.append("answer")
        return {
            "output": "|".join(staged_query["notes"]),
            "query_time": 0.1,
            "retrieved_count": len(staged_query["notes"]),
            "retrieved_memories": list(staged_query["notes"]),
        }


def _staged_evaluator(tmp_path: Path, manager: _StageManager):
    evaluator = MedMemoryBenchEvaluator.__new__(MedMemoryBenchEvaluator)
    evaluator.output_dir = tmp_path
    evaluator.method_config = SimpleNamespace(
        method_name="amem",
        model=SimpleNamespace(name="test-model"),
        raw_config={"method": "amem"},
    )
    evaluator.dataset_config = SimpleNamespace(
        raw_config={"dataset": "test"},
        evaluation_mode="merged",
    )
    evaluator.dry_run = False
    evaluator.force_resume = False
    evaluator.execution_stage = "all"
    evaluator.batch_api = False
    evaluator._checkpoint_manager = None
    evaluator._memory_build_logs = []
    evaluator._deferred_judges = []
    evaluator._api_failures = []
    evaluator._batch_fallback_logged = False
    evaluator.memory_run = None
    evaluator._memory_snapshot_run_dir = (
        tmp_path / "amem_test-model" / "memory" / "20260101_000000_000000"
    )
    evaluator._memory_snapshot_manifest = {
        "format": "medmemorybench.memory_manifest",
        "version": 1,
        "build_id": "test-build",
        "status": "building",
        "method_name": "amem",
        "model_name": "test-model",
        "config_hash": evaluator._batch_config_hash(),
        "unit_ids": [0, 1],
        "snapshots": [],
    }
    evaluator.agent_manager = manager
    evaluator.prompt_manager = SimpleNamespace(
        format_memorize=lambda context, timestamp: context,
        format_query=lambda question, query_type: question,
    )
    evaluator._log = lambda *args, **kwargs: None
    evaluator._score_agent_response = lambda query, response, **kwargs: SimpleNamespace(
        query_id=query.query_id,
        query_type=query.query_type,
        score=1.0,
        is_correct=True,
        query_time=response["query_time"],
        model_output=response["output"],
    )
    return evaluator


def _unit(unit_id: int, session_id: int, content: str, query_id: str):
    session = SimpleNamespace(
        session_id=session_id,
        timestamp=None,
        metadata={},
        to_memory_text=lambda: content,
    )
    query = SimpleNamespace(
        query_id=query_id,
        question=f"question-{query_id}",
        query_type="entity_exact_match",
    )
    return EvaluationUnit(
        unit_id=unit_id,
        context_id=1,
        sessions_to_inject=[session],
        queries_to_evaluate=[query],
        metadata={"eval_session_id": session_id},
    )


def test_evaluator_orders_build_save_load_retrieve_answer_and_isolates_units(tmp_path: Path):
    manager = _StageManager()
    evaluator = _staged_evaluator(tmp_path, manager)

    first = evaluator._evaluate_unit_with_checkpoint(_unit(0, 10, "unit-one", "q1"))
    second = evaluator._evaluate_unit_with_checkpoint(_unit(1, 11, "unit-two", "q2"))

    assert first[0].model_output == "unit-one"
    assert second[0].model_output == "unit-one|unit-two"
    assert manager.events == [
        "memorize:unit-one",
        "save",
        "load",
        "retrieve:question-q1:unit-one",
        "answer",
        "memorize:unit-two",
        "save",
        "load",
        "retrieve:question-q2:unit-one|unit-two",
        "answer",
    ]

    memory_dir = evaluator._memory_snapshot_run_dir
    first_payload = json.loads((memory_dir / "persona_1_unit_0.json").read_text())
    second_payload = json.loads((memory_dir / "persona_1_unit_1.json").read_text())
    manifest = json.loads((memory_dir / "manifest.json").read_text())
    assert first_payload["memory_state"]["notes"] == ["unit-one"]
    assert second_payload["memory_state"]["notes"] == ["unit-one", "unit-two"]
    assert manifest["snapshots"][0]["integrity_hash"] == first_payload["integrity_hash"]
    assert manifest["snapshots"][0]["embedding_sha256"] == ""
    assert evaluator._memory_build_logs[0]["memory_snapshot"].endswith("persona_1_unit_0.json")
    assert evaluator._memory_build_logs[1]["memory_snapshot"].endswith("persona_1_unit_1.json")


def test_medmemorybench_passes_source_ids_only_to_amem_test(tmp_path: Path):
    class _SourceManager(_StageManager):
        def __init__(self):
            super().__init__()
            self.memorize_kwargs = []

        def send_message(self, *, message, memorizing, **kwargs):
            assert memorizing is True
            self.memorize_kwargs.append(kwargs)
            from methods.base import MemoryBuildResult
            return MemoryBuildResult(method="amem_test", time_cost=0.0)

    session = SimpleNamespace(
        session_id=12,
        timestamp="2024-04-03",
        metadata={"event_id": "event-12", "messages": []},
        to_memory_text=lambda: "memory",
    )
    unit = EvaluationUnit(0, [session], [], context_id=1)
    manager = _SourceManager()
    evaluator = _staged_evaluator(tmp_path, manager)
    evaluator.method_config.method_name = "amem_test"
    evaluator._save_and_restore_unit_memory = lambda *args, **kwargs: None
    evaluator._evaluate_unit_queries = lambda *args, **kwargs: []

    evaluator._evaluate_unit_with_checkpoint(unit)

    kwargs = manager.memorize_kwargs[0]
    assert kwargs["source_session_id"] == 12
    assert kwargs["source_session_index"] == 0
    assert kwargs["source_event_id"] == "event-12"

    baseline_manager = _SourceManager()
    baseline = _staged_evaluator(tmp_path, baseline_manager)
    baseline.method_config.method_name = "amem"
    baseline._save_and_restore_unit_memory = lambda *args, **kwargs: None
    baseline._evaluate_unit_queries = lambda *args, **kwargs: []

    baseline._evaluate_unit_with_checkpoint(unit)

    baseline_kwargs = baseline_manager.memorize_kwargs[0]
    assert "source_session_id" not in baseline_kwargs
    assert "source_session_index" not in baseline_kwargs
    assert "source_event_id" not in baseline_kwargs


def test_evaluator_stores_embedding_matrix_as_exact_npy_sidecar(tmp_path: Path):
    system = _robust_system()
    agent = _snapshot_agent(system, context_id=1)
    manager = SimpleNamespace(
        supports_memory_snapshots=lambda: True,
        export_memory_state=agent.export_memory_state,
        import_memory_state=agent.import_memory_state,
    )
    evaluator = _staged_evaluator(tmp_path, _StageManager())
    evaluator.agent_manager = manager
    unit = _unit(0, 10, "unit-one", "q1")

    build_metrics = {
        "wall_time_seconds": 1.5,
        "usage": {
            "memorize_phase": {"input_tokens": 10, "output_tokens": 2},
            "query_phase": {},
            "operations": {},
        },
    }
    path = evaluator._write_memory_snapshot(
        unit,
        memory_build_time=1.5,
        memory_build_metrics=build_metrics,
    )
    payload = json.loads(path.read_text())
    embedding_state = payload["memory_state"]["system_state"]["retriever"]["embeddings"]

    assert "values" not in embedding_state
    sidecar = path.parent / embedding_state["path"]
    assert sidecar.suffix == ".npy"
    np.testing.assert_array_equal(
        np.load(sidecar, allow_pickle=False),
        np.asarray([[1, 0, 0], [0, 1, 0]], dtype=np.float32),
    )
    memory_state_bytes = len(json.dumps(
        payload["memory_state"],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8"))
    assert payload["memory_size"] == {
        "measurement": "serialized_memory_state",
        "bytes": memory_state_bytes + sidecar.stat().st_size,
        "mib": round((memory_state_bytes + sidecar.stat().st_size) / (1024 ** 2), 6),
        "json_bytes": memory_state_bytes,
        "embedding_bytes": sidecar.stat().st_size,
        "memory_entry_count": 2,
        "memory_chunk_count": len(
            payload["memory_state"]["agent_state"]["memory_chunks"]
        ),
    }

    loaded = evaluator._read_memory_snapshot(unit)
    np.testing.assert_array_equal(
        loaded["memory_state"]["system_state"]["retriever"]["embeddings"]["values"],
        np.asarray([[1, 0, 0], [0, 1, 0]], dtype=np.float32),
    )
    assert loaded["memory_build_time"] == 1.5
    assert loaded["memory_build_metrics"] == build_metrics

def test_interrupted_snapshot_replacement_keeps_previous_generation_readable(
    tmp_path: Path,
    monkeypatch,
):
    system = _robust_system()
    agent = _snapshot_agent(system, context_id=1)
    evaluator = _staged_evaluator(tmp_path, _StageManager())
    evaluator.agent_manager = SimpleNamespace(
        supports_memory_snapshots=lambda: True,
        export_memory_state=agent.export_memory_state,
        import_memory_state=agent.import_memory_state,
    )
    unit = _unit(0, 10, "unit-one", "q1")
    path = evaluator._write_memory_snapshot(unit)
    original_payload = json.loads(path.read_text())
    original_sidecar = path.parent / original_payload[
        "memory_state"
    ]["system_state"]["retriever"]["embeddings"]["path"]

    system.retriever.embeddings = np.asarray(
        [[3, 0, 0], [0, 4, 0]], dtype=np.float32
    )
    real_replace = importlib.import_module(
        "benchmarks.medmemorybench.evaluator"
    ).os.replace

    def interrupt_before_json_publish(source, destination):
        if Path(destination) == path:
            raise KeyboardInterrupt()
        return real_replace(source, destination)

    monkeypatch.setattr(
        "benchmarks.medmemorybench.evaluator.os.replace",
        interrupt_before_json_publish,
    )
    with pytest.raises(KeyboardInterrupt):
        evaluator._write_memory_snapshot(unit)

    assert original_sidecar.exists()
    restored = evaluator._read_memory_snapshot(unit)
    np.testing.assert_array_equal(
        restored["memory_state"]["system_state"]["retriever"]["embeddings"]["values"],
        np.asarray([[1, 0, 0], [0, 1, 0]], dtype=np.float32),
    )

def test_ordinary_resume_restores_completed_unit_without_rebuilding(tmp_path: Path):
    first_manager = _StageManager()
    first = _staged_evaluator(tmp_path, first_manager)
    unit = _unit(0, 10, "unit-one", "q1")
    first_result = first._evaluate_unit_with_checkpoint(unit)

    resumed_manager = _StageManager()
    resumed = _staged_evaluator(tmp_path, resumed_manager)
    resumed.resume = True
    resumed_result = resumed._evaluate_unit_with_checkpoint(unit)

    assert first_result[0].model_output == resumed_result[0].model_output == "unit-one"
    assert not any(event.startswith("memorize:") for event in resumed_manager.events)
    assert resumed_manager.events == [
        "load",
        "save",
        "load",
        "retrieve:question-q1:unit-one",
        "answer",
    ]


def test_completed_manifest_summarizes_snapshot_build_metrics(tmp_path: Path):
    system = _robust_system()
    agent = _snapshot_agent(system, context_id=1)
    evaluator = _staged_evaluator(tmp_path, _StageManager())
    evaluator.agent_manager = SimpleNamespace(
        supports_memory_snapshots=lambda: True,
        export_memory_state=agent.export_memory_state,
    )
    evaluator._memory_snapshot_manifest["unit_ids"] = [0]
    unit = _unit(0, 10, "unit-one", "q1")
    build_metrics = {
        "wall_time_seconds": 2.0,
        "usage": {
            "memorize_phase": {
                "input_tokens": 11,
                "output_tokens": 4,
                "successful_calls": 2,
                "attempted_calls": 2,
            },
            "query_phase": {},
            "operations": {},
        },
    }
    evaluator._write_memory_snapshot(
        unit,
        memory_build_time=2.0,
        memory_build_metrics=build_metrics,
    )

    evaluator._complete_memory_snapshot_manifest()

    manifest = json.loads(evaluator._memory_snapshot_manifest_path().read_text())
    assert manifest["status"] == "complete"
    assert manifest["build_metrics"]["totals"]["input_tokens"] == 11
    assert manifest["build_metrics"]["totals"]["successful_calls"] == 2
    assert manifest["build_metrics"]["totals"]["wall_time_seconds"] == 2.0
    assert manifest["memory_size"]["overall_bytes"] > 0
    assert manifest["build_metrics"]["units"][0]["memory_size"]["bytes"] > 0


def test_evaluator_rejects_snapshot_when_manifest_build_hash_differs(tmp_path: Path):
    system = _robust_system()
    agent = _snapshot_agent(system, context_id=1)
    evaluator = _staged_evaluator(tmp_path, _StageManager())
    evaluator.agent_manager = SimpleNamespace(
        supports_memory_snapshots=lambda: True,
        export_memory_state=agent.export_memory_state,
    )
    unit = _unit(0, 10, "unit-one", "q1")
    evaluator._write_memory_snapshot(unit)
    evaluator.dataset_config.inject_noise = True

    with pytest.raises(ValueError, match="build configuration does not match"):
        evaluator._read_memory_snapshot(unit)


def test_evaluator_rejects_snapshot_payload_build_hash_mismatch(tmp_path: Path):
    system = _robust_system()
    agent = _snapshot_agent(system, context_id=1)
    evaluator = _staged_evaluator(tmp_path, _StageManager())
    evaluator.agent_manager = SimpleNamespace(
        supports_memory_snapshots=lambda: True,
        export_memory_state=agent.export_memory_state,
    )
    unit = _unit(0, 10, "unit-one", "q1")
    path = evaluator._write_memory_snapshot(unit)
    payload = json.loads(path.read_text())
    payload["build_config_hash"] = "different-build-hash"
    payload["integrity_hash"] = evaluator._snapshot_integrity_hash(payload)
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="payload build configuration does not match"):
        evaluator._read_memory_snapshot(unit)


def test_query_only_stage_loads_each_saved_unit_without_rebuilding(tmp_path: Path):
    build_manager = _StageManager()
    builder = _staged_evaluator(tmp_path, build_manager)
    builder.execution_stage = "memory"
    units = [_unit(0, 10, "unit-one", "q1"), _unit(1, 11, "unit-two", "q2")]
    for unit in units:
        builder._evaluate_unit_with_checkpoint(unit)
    assert all(not event.startswith("retrieve:") for event in build_manager.events)
    assert "answer" not in build_manager.events
    builder._memory_snapshot_manifest["status"] = "complete"
    builder._memory_snapshot_manifest["completed_at"] = "2026-01-01T00:00:00"
    builder._write_memory_snapshot_manifest()

    query_manager = _StageManager()
    evaluator = _staged_evaluator(tmp_path, query_manager)
    evaluator.execution_stage = "query"
    evaluator.dataset_config = SimpleNamespace(
        raw_config={"dataset": "test"},
        evaluation_mode="merged",
    )
    evaluator.dataset = SimpleNamespace(
        get_evaluation_units=lambda: iter(units),
        get_persona_ids=lambda: [1],
    )
    evaluator._checkpoint_manager = None
    evaluator.aggregator = SimpleNamespace(add_result=lambda result: None)
    collected = []
    evaluator.result_collector = SimpleNamespace(
        add_result=lambda result, persona_id: collected.append(result)
    )
    evaluator._init_agent_for_context = (
        lambda context_id, force_new: evaluator.agent_manager.set_context_id(context_id)
    )
    evaluator._complete_combined_batch_queries = lambda: []
    evaluator._complete_deferred_judges = lambda: []
    evaluator._load_memory_snapshot_manifest()

    evaluator._run_evaluation_loop()

    assert all(not event.startswith("memorize:") for event in query_manager.events)
    assert [result.model_output for result in collected] == [
        "unit-one",
        "unit-one|unit-two",
    ]


def test_query_only_stage_runs_units_in_parallel_with_isolated_agents(
    tmp_path: Path,
    monkeypatch,
):
    build_manager = _StageManager()
    builder = _staged_evaluator(tmp_path, build_manager)
    builder.execution_stage = "memory"
    units = [_unit(0, 10, "unit-one", "q1"), _unit(1, 11, "unit-two", "q2")]
    for unit in units:
        builder._evaluate_unit_with_checkpoint(unit)
    builder._memory_snapshot_manifest["status"] = "complete"
    builder._memory_snapshot_manifest["completed_at"] = "2026-01-01T00:00:00"
    builder._write_memory_snapshot_manifest()

    barrier = threading.Barrier(2)
    managers = []

    class ParallelStageManager(_StageManager):
        def prepare_query(self, *, message, **kwargs):
            managers.append(self)
            barrier.wait(timeout=5)
            return super().prepare_query(message=message, **kwargs)

        def reset(self):
            return None

    monkeypatch.setattr(
        "benchmarks.medmemorybench.evaluator.AgentManager",
        lambda **kwargs: ParallelStageManager(),
    )

    evaluator = _staged_evaluator(tmp_path, _StageManager())
    evaluator.execution_stage = "query"
    evaluator.workers = 2
    evaluator.dataset = SimpleNamespace(
        get_evaluation_units=lambda: iter(units),
        get_persona_ids=lambda: [1],
    )
    evaluator.aggregator = SimpleNamespace(add_result=lambda result: None)
    collected = []
    evaluator.result_collector = SimpleNamespace(
        add_result=lambda result, persona_id: collected.append(result)
    )
    evaluator._complete_combined_batch_queries = lambda: []
    evaluator._complete_deferred_judges = lambda: []
    evaluator._load_memory_snapshot_manifest()

    evaluator._run_evaluation_loop()

    assert len(managers) == 2
    assert [result.model_output for result in collected] == [
        "unit-one",
        "unit-one|unit-two",
    ]


def test_query_only_stage_applies_workers_as_a_global_query_limit(
    tmp_path: Path,
    monkeypatch,
):
    build_manager = _StageManager()
    builder = _staged_evaluator(tmp_path, build_manager)
    builder.execution_stage = "memory"
    units = [_unit(0, 10, "unit-one", "q1"), _unit(1, 11, "unit-two", "q3")]
    units[0].queries_to_evaluate.append(
        SimpleNamespace(query_id="q2", question="question-q2", query_type="entity_exact_match")
    )
    units[1].queries_to_evaluate.append(
        SimpleNamespace(query_id="q4", question="question-q4", query_type="entity_exact_match")
    )
    for unit in units:
        builder._evaluate_unit_with_checkpoint(unit)
    builder._memory_snapshot_manifest["status"] = "complete"
    builder._memory_snapshot_manifest["completed_at"] = "2026-01-01T00:00:00"
    builder._write_memory_snapshot_manifest()

    active_queries = 0
    max_active_queries = 0
    query_lock = threading.Lock()

    class CountingStageManager(_StageManager):
        def prepare_query(self, *, message, **kwargs):
            nonlocal active_queries, max_active_queries
            with query_lock:
                active_queries += 1
                max_active_queries = max(max_active_queries, active_queries)
            try:
                time.sleep(0.05)
                return super().prepare_query(message=message, **kwargs)
            finally:
                with query_lock:
                    active_queries -= 1

        def reset(self):
            return None

    monkeypatch.setattr(
        "benchmarks.medmemorybench.evaluator.AgentManager",
        lambda **kwargs: CountingStageManager(),
    )

    evaluator = _staged_evaluator(tmp_path, _StageManager())
    evaluator.execution_stage = "query"
    evaluator.workers = 2
    evaluator.dataset = SimpleNamespace(
        get_evaluation_units=lambda: iter(units),
        get_persona_ids=lambda: [1],
    )
    evaluator.aggregator = SimpleNamespace(add_result=lambda result: None)
    collected = []
    evaluator.result_collector = SimpleNamespace(
        add_result=lambda result, persona_id: collected.append(result)
    )
    evaluator._complete_combined_batch_queries = lambda: []
    evaluator._complete_deferred_judges = lambda: []
    evaluator._load_memory_snapshot_manifest()

    evaluator._run_evaluation_loop()

    assert max_active_queries == 2
    assert [result.query_id for result in collected] == ["q1", "q2", "q3", "q4"]


def test_fresh_memory_stage_initializes_manifest_after_resume_state():
    evaluator = MedMemoryBenchEvaluator.__new__(MedMemoryBenchEvaluator)
    evaluator.execution_stage = "memory"
    evaluator.method_config = SimpleNamespace(method_name="amem_test")
    evaluator.dry_run = False
    evaluator.resume = False
    evaluator._checkpoint_enabled = True
    evaluator._checkpoint_manager = None
    evaluator._init_dataset = lambda: None
    evaluator._run_evaluation_loop = lambda: None
    evaluator._create_new_checkpoint = lambda: None
    evaluator._complete_memory_snapshot_manifest = lambda: None
    evaluator._generate_report = lambda *args: "report"
    manifest_calls = []
    evaluator._start_memory_snapshot_manifest = (
        lambda *, resume_existing: manifest_calls.append(resume_existing)
    )

    assert evaluator.evaluate() == "report"
    assert manifest_calls == [False]


def test_fresh_query_stage_creates_unique_batch_scope():
    evaluator = MedMemoryBenchEvaluator.__new__(MedMemoryBenchEvaluator)
    evaluator.execution_stage = "query"
    evaluator.method_config = SimpleNamespace(method_name="amem_test")
    evaluator.dry_run = False
    evaluator.resume = False
    evaluator.batch_api = True
    evaluator._checkpoint_enabled = True
    evaluator._checkpoint_manager = SimpleNamespace(
        mark_completed=lambda: None,
        delete=lambda: None,
    )
    evaluator._init_dataset = lambda: None
    evaluator._load_memory_snapshot_manifest = lambda: None
    evaluator._run_evaluation_loop = lambda: None
    evaluator._generate_report = lambda *args: "report"
    checkpoint_calls = []
    evaluator._create_new_checkpoint = lambda: checkpoint_calls.append(True)
    evaluator._log = lambda *args, **kwargs: None

    assert evaluator.evaluate() == "report"
    assert checkpoint_calls == [True]


def test_fresh_memory_runs_use_distinct_timestamp_directories(tmp_path: Path):
    evaluator = _staged_evaluator(tmp_path, _StageManager())
    evaluator.dataset = SimpleNamespace(get_evaluation_units=lambda: iter([]))
    run_names = iter(("20260101_010101_000001", "20260101_010102_000002"))
    evaluator._new_memory_run_name = lambda: next(run_names)

    evaluator._start_memory_snapshot_manifest()
    first_dir = evaluator._memory_snapshot_run_dir
    evaluator._start_memory_snapshot_manifest()
    second_dir = evaluator._memory_snapshot_run_dir

    assert first_dir != second_dir
    assert (first_dir / "manifest.json").exists()
    assert (second_dir / "manifest.json").exists()


def test_query_stage_selects_newest_complete_compatible_memory_run(tmp_path: Path):
    evaluator = _staged_evaluator(tmp_path, _StageManager())
    evaluator.dataset = SimpleNamespace(get_evaluation_units=lambda: iter([]))
    root = evaluator._memory_snapshot_root()
    for run_name, status in (
        ("20260101_010101_000001", "complete"),
        ("20260101_010102_000002", "complete"),
        ("20260101_010103_000003", "building"),
    ):
        run_dir = root / run_name
        run_dir.mkdir(parents=True)
        (run_dir / "manifest.json").write_text(json.dumps({
            "format": "medmemorybench.memory_manifest",
            "version": 1,
            "status": status,
            "method_name": "amem",
            "model_name": "test-model",
            "config_hash": evaluator._batch_config_hash(),
            "unit_ids": [],
        }))

    evaluator._load_memory_snapshot_manifest()

    assert evaluator._memory_snapshot_run_dir.name == "20260101_010102_000002"


def test_query_stage_can_select_an_explicit_older_memory_run(tmp_path: Path):
    evaluator = _staged_evaluator(tmp_path, _StageManager())
    evaluator.dataset = SimpleNamespace(get_evaluation_units=lambda: iter([]))
    evaluator.memory_run = "20260101_010101_000001"
    run_dir = evaluator._memory_snapshot_root() / evaluator.memory_run
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(json.dumps({
        "format": "medmemorybench.memory_manifest",
        "version": 1,
        "status": "complete",
        "method_name": "amem",
        "model_name": "test-model",
        "config_hash": evaluator._batch_config_hash(),
        "unit_ids": [],
    }))

    evaluator._load_memory_snapshot_manifest()

    assert evaluator._memory_snapshot_run_dir == run_dir


def test_query_stage_falls_back_to_complete_legacy_memory_layout(tmp_path: Path):
    evaluator = _staged_evaluator(tmp_path, _StageManager())
    evaluator.dataset = SimpleNamespace(get_evaluation_units=lambda: iter([]))
    root = evaluator._memory_snapshot_root()
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_text(json.dumps({
        "format": "medmemorybench.memory_manifest",
        "version": 1,
        "status": "complete",
        "method_name": "amem",
        "model_name": "test-model",
        "config_hash": evaluator._batch_config_hash(),
        "unit_ids": [],
    }))

    evaluator._load_memory_snapshot_manifest()

    assert evaluator._memory_snapshot_run_dir == root


def test_query_stage_can_explicitly_select_legacy_memory_layout(tmp_path: Path):
    evaluator = _staged_evaluator(tmp_path, _StageManager())
    evaluator.dataset = SimpleNamespace(get_evaluation_units=lambda: iter([]))
    evaluator.memory_run = "legacy"
    root = evaluator._memory_snapshot_root()
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_text(json.dumps({
        "format": "medmemorybench.memory_manifest",
        "version": 1,
        "status": "complete",
        "method_name": "amem",
        "model_name": "test-model",
        "config_hash": evaluator._batch_config_hash(),
        "unit_ids": [],
    }))

    evaluator._load_memory_snapshot_manifest()

    assert evaluator._memory_snapshot_run_dir == root


def _report():
    return EvaluationReport(
        method_name="amem",
        model_name="test-model",
        dataset_name="medmemorybench",
        start_time="start",
        end_time="end",
        duration_seconds=1.0,
        summary={"total": 0},
        detailed_results=[],
        metadata={"api_failures": []},
    )


def test_memory_stage_writes_only_memory_build_report(tmp_path: Path):
    collector = ResultCollector()

    paths = collector.save_reports(
        _report(),
        tmp_path,
        [],
        include_result=False,
        include_memory_build=True,
        include_query_answer=False,
    )

    assert paths[0] is None
    assert paths[1] is not None and paths[1].exists()
    assert paths[2] is None
    output_files = {path.name for path in paths[1].parent.iterdir() if path.is_file()}
    assert output_files == {paths[1].name}
    assert collector.last_api_failure_path is None


def test_query_stage_writes_result_and_query_answer_without_memory_build(tmp_path: Path):
    collector = ResultCollector()

    paths = collector.save_reports(
        _report(),
        tmp_path,
        [],
        include_result=True,
        include_memory_build=False,
        include_query_answer=True,
    )

    assert paths[0] is not None and paths[0].exists()
    assert paths[1] is None
    assert paths[2] is not None and paths[2].exists()
    output_files = {path.name for path in paths[0].parent.iterdir() if path.is_file()}
    assert output_files == {paths[0].name, paths[2].name}
    assert collector.last_api_failure_path is None


def test_stage_report_writes_api_failures_only_when_present(tmp_path: Path):
    collector = ResultCollector()
    report = _report()
    report.metadata["api_failures"] = [{"phase": "memory_build", "error": "failed"}]

    collector.save_reports(
        report,
        tmp_path,
        [],
        include_result=False,
        include_memory_build=True,
        include_query_answer=False,
    )

    assert collector.last_api_failure_path is not None
    assert collector.last_api_failure_path.exists()


def test_full_run_report_keeps_all_output_files(tmp_path: Path):
    collector = ResultCollector()

    paths = collector.save_reports(_report(), tmp_path, [])

    assert all(path is not None and path.exists() for path in paths)
    output_files = {path.name for path in paths[0].parent.iterdir() if path.is_file()}
    assert output_files == {path.name for path in paths}


def test_run_scoped_reports_write_directly_in_run_directory(tmp_path: Path):
    collector = ResultCollector()

    paths = collector.save_reports(
        _report(),
        tmp_path,
        [],
        use_method_subdir=False,
    )

    assert all(path is not None and path.parent == tmp_path for path in paths)
    assert not (tmp_path / "amem_test-model").exists()


def test_run_scoped_memory_build_uses_the_run_memory_directory(tmp_path: Path):
    experiment_dir = tmp_path / "amem_test-model"
    run_dir = experiment_dir / "20260101_010101"
    run_dir.mkdir(parents=True)
    evaluator = _staged_evaluator(run_dir, _StageManager())
    evaluator.run_scoped_output = True
    evaluator.dataset = SimpleNamespace(get_evaluation_units=lambda: iter([]))

    evaluator._start_memory_snapshot_manifest()

    assert evaluator._memory_snapshot_run_dir == run_dir / "memory"
    manifest = json.loads((run_dir / "memory" / "manifest.json").read_text())
    assert manifest["run_name"] == run_dir.name


def test_run_scoped_query_selects_sibling_memory_and_records_source(tmp_path: Path):
    experiment_dir = tmp_path / "amem_test-model"
    query_run = experiment_dir / "20260101_010103"
    query_run.mkdir(parents=True)
    evaluator = _staged_evaluator(query_run, _StageManager())
    evaluator.run_scoped_output = True
    evaluator.dataset = SimpleNamespace(get_evaluation_units=lambda: iter([]))

    for run_name, status in (
        ("20260101_010101", "complete"),
        ("20260101_010102", "complete"),
        ("20260101_010104", "building"),
    ):
        memory_dir = experiment_dir / run_name / "memory"
        memory_dir.mkdir(parents=True)
        (memory_dir / "manifest.json").write_text(json.dumps({
            "format": "medmemorybench.memory_manifest",
            "version": 1,
            "status": status,
            "method_name": "amem",
            "model_name": "test-model",
            "config_hash": evaluator._batch_config_hash(),
            "build_id": f"build-{run_name}",
            "unit_ids": [],
        }))

    evaluator._load_memory_snapshot_manifest()

    assert evaluator._memory_snapshot_run_dir == experiment_dir / "20260101_010102" / "memory"
    source = json.loads((query_run / "memory_source.json").read_text())
    assert source["source_run_id"] == "20260101_010102"
    assert source["build_id"] == "build-20260101_010102"


def test_nested_query_run_selects_its_parent_memory_and_records_source(tmp_path: Path):
    experiment_dir = tmp_path / "amem_test-model"
    source_run = experiment_dir / "20260101_010102"
    query_run = source_run / "query_runs" / "20260101_010103"
    memory_dir = source_run / "memory"
    query_run.mkdir(parents=True)
    memory_dir.mkdir()
    evaluator = _staged_evaluator(query_run, _StageManager())
    evaluator.run_scoped_output = True
    evaluator.memory_run = source_run.name
    evaluator.memory_run_explicit = True
    evaluator.memory_source_run_dir = source_run.resolve()
    evaluator.dataset = SimpleNamespace(get_evaluation_units=lambda: iter([]))
    (memory_dir / "manifest.json").write_text(json.dumps({
        "format": "medmemorybench.memory_manifest",
        "version": 1,
        "status": "complete",
        "method_name": "amem",
        "model_name": "test-model",
        "config_hash": evaluator._batch_config_hash(),
        "build_id": "parent-build",
        "unit_ids": [],
    }))

    evaluator._load_memory_snapshot_manifest()

    assert evaluator._memory_snapshot_run_dir == memory_dir
    source = json.loads((query_run / "memory_source.json").read_text())
    assert source["source_run_id"] == source_run.name
    assert source["selection"] == "explicit"
    assert (query_run / source["manifest_path"]).resolve() == (
        memory_dir / "manifest.json"
    ).resolve()


def test_run_scoped_query_resume_reuses_recorded_memory_source(tmp_path: Path):
    experiment_dir = tmp_path / "amem_test-model"
    query_run = experiment_dir / "20260101_010103"
    source_run = experiment_dir / "20260101_010101" / "memory"
    query_run.mkdir(parents=True)
    source_run.mkdir(parents=True)
    evaluator = _staged_evaluator(query_run, _StageManager())
    evaluator.run_scoped_output = True
    evaluator.dataset = SimpleNamespace(get_evaluation_units=lambda: iter([]))
    manifest = {
        "format": "medmemorybench.memory_manifest",
        "version": 1,
        "status": "complete",
        "method_name": "amem",
        "model_name": "test-model",
        "config_hash": evaluator._batch_config_hash(),
        "build_id": "original-build",
        "unit_ids": [],
    }
    (source_run / "manifest.json").write_text(json.dumps(manifest))
    (query_run / "memory_source.json").write_text(json.dumps({
        "format": "medmemorybench.memory_source",
        "version": 1,
        "manifest_path": "../20260101_010101/memory/manifest.json",
    }))

    newer_memory = experiment_dir / "20260101_010104" / "memory"
    newer_memory.mkdir(parents=True)
    newer_manifest = dict(manifest, build_id="newer-build")
    (newer_memory / "manifest.json").write_text(json.dumps(newer_manifest))

    evaluator._load_memory_snapshot_manifest()

    assert evaluator._memory_snapshot_run_dir == source_run
    assert evaluator._memory_snapshot_manifest["build_id"] == "original-build"
