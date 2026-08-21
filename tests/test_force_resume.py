"""Tests for forced session-level MedMemoryBench resume."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.base import EvaluationUnit
from benchmarks.medmemorybench.checkpoint import MedMemoryBenchCheckpointManager
from benchmarks.medmemorybench.dataset import MedSession
from benchmarks.medmemorybench.evaluator import MedMemoryBenchEvaluator
from methods.base import MemoryBuildResult


def _manager(tmp_path: Path, config_hash: str) -> MedMemoryBenchCheckpointManager:
    return MedMemoryBenchCheckpointManager(
        method_name="method",
        model_name="model",
        checkpoint_dir=tmp_path,
        config_hash=config_hash,
    )


def test_start_persona_can_preserve_injected_session_progress(tmp_path: Path):
    manager = _manager(tmp_path, "hash")
    manager.create(total_personas=1, total_queries=2, evaluation_mode="independent")
    manager.start_persona(1)
    manager.mark_session_injected(11)

    manager.start_persona(1, preserve_progress=True)

    assert manager.is_session_injected(11, persona_id=1) is True
    assert manager.get_resume_info()["current_persona_injected_sessions"] == 1


def test_force_resume_accepts_and_adopts_changed_config(tmp_path: Path):
    original = _manager(tmp_path, "old-hash")
    original.create(total_personas=1, total_queries=2, evaluation_mode="independent")
    original.start_persona(1)
    original.mark_session_injected(11)

    evaluator = MedMemoryBenchEvaluator.__new__(MedMemoryBenchEvaluator)
    evaluator._checkpoint_manager = _manager(tmp_path, "new-hash")
    evaluator.force_resume = True
    evaluator._force_resume_persona_id = None
    evaluator._load_checkpoint_results = lambda: None
    evaluator._log = lambda *args, **kwargs: None

    assert evaluator._try_resume_from_checkpoint() is True
    assert evaluator._force_resume_persona_id == 1
    assert evaluator._checkpoint_manager.validate_config() is True
    assert evaluator._checkpoint_manager.is_session_injected(11, persona_id=1) is True
    assert evaluator._checkpoint_manager.get_resume_info()["active_session_id"] is None

def test_force_resume_rejects_ambiguous_persistent_backend_session(tmp_path: Path):
    original = _manager(tmp_path, "hash")
    original.create(total_personas=1, total_queries=0, evaluation_mode="independent")
    original.start_persona(1)
    original.mark_session_started(12, unit_id=1)

    evaluator = MedMemoryBenchEvaluator.__new__(MedMemoryBenchEvaluator)
    evaluator.method_config = SimpleNamespace(method_name="method")
    evaluator._checkpoint_manager = _manager(tmp_path, "hash")
    evaluator.force_resume = True
    evaluator._load_checkpoint_results = lambda: None
    evaluator._log = lambda *args, **kwargs: None

    with pytest.raises(RuntimeError, match="may already have committed"):
        evaluator._try_resume_from_checkpoint()

    assert evaluator._checkpoint_manager.get_resume_info()["active_session_id"] == 12

def test_amem_force_resume_rejects_changed_build_config_before_adoption(
    tmp_path: Path,
):
    original = MedMemoryBenchCheckpointManager(
        method_name="amem",
        model_name="model",
        checkpoint_dir=tmp_path,
        config_hash="old-hash",
    )
    original.create(total_personas=1, total_queries=0, evaluation_mode="independent")

    evaluator = MedMemoryBenchEvaluator.__new__(MedMemoryBenchEvaluator)
    evaluator.method_config = SimpleNamespace(method_name="amem")
    evaluator._checkpoint_manager = MedMemoryBenchCheckpointManager(
        method_name="amem",
        model_name="model",
        checkpoint_dir=tmp_path,
        config_hash="new-hash",
    )
    evaluator.force_resume = True
    evaluator._log = lambda *args, **kwargs: None

    with pytest.raises(ValueError, match="memory-build configuration changes"):
        evaluator._try_resume_from_checkpoint()

    reloaded = MedMemoryBenchCheckpointManager(
        method_name="amem",
        model_name="model",
        checkpoint_dir=tmp_path,
        config_hash="old-hash",
    )
    assert reloaded.load() is not None
    assert reloaded.validate_config() is True


def test_force_resume_skips_recorded_sessions_and_injects_the_rest(tmp_path: Path):
    manager = _manager(tmp_path, "hash")
    manager.create(total_personas=1, total_queries=0, evaluation_mode="independent")
    manager.start_persona(1)
    manager.mark_session_injected(11)

    injected_messages = []
    evaluator = MedMemoryBenchEvaluator.__new__(MedMemoryBenchEvaluator)
    evaluator.force_resume = True
    evaluator.dry_run = False
    evaluator.batch_api = False
    evaluator._checkpoint_manager = manager
    evaluator._memory_build_logs = []
    evaluator._deferred_judges = []
    evaluator._log = lambda *args, **kwargs: None
    evaluator.prompt_manager = SimpleNamespace(
        format_memorize=lambda context, timestamp: context
    )

    def send_message(**kwargs):
        injected_messages.append(kwargs["message"])
        return MemoryBuildResult(method="test")

    evaluator.agent_manager = SimpleNamespace(send_message=send_message)
    unit = EvaluationUnit(
        unit_id=1,
        context_id=1,
        sessions_to_inject=[
            MedSession(session_id=11, content="session eleven"),
            MedSession(session_id=12, content="session twelve"),
        ],
        queries_to_evaluate=[],
    )

    assert evaluator._evaluate_unit_with_checkpoint(unit) == []
    assert len(injected_messages) == 1
    assert "session twelve" in injected_messages[0]
    assert manager.is_session_injected(12, persona_id=1) is True


def test_corrupted_primary_recovers_previous_good_checkpoint(tmp_path: Path):
    manager = _manager(tmp_path, "hash")
    manager.create(total_personas=1, total_queries=2, evaluation_mode="independent")
    manager.start_persona(1)
    manager.mark_session_injected(11)
    manager.mark_session_started(12, unit_id=1)
    manager.checkpoint_path.write_text("{corrupted", encoding="utf-8")

    recovered = _manager(tmp_path, "hash")
    checkpoint = recovered.load()

    assert checkpoint is not None
    assert recovered.recovered_from_backup is True
    assert recovered.is_session_injected(11, persona_id=1) is True
    assert recovered.is_session_injected(12, persona_id=1) is False
    assert recovered.get_resume_info()["active_session_id"] is None


def test_integrity_mismatch_recovers_previous_good_checkpoint(tmp_path: Path):
    manager = _manager(tmp_path, "hash")
    manager.create(total_personas=1, total_queries=2, evaluation_mode="independent")
    manager.start_persona(1)
    payload = json.loads(manager.checkpoint_path.read_text(encoding="utf-8"))
    payload["completed_query_count"] = 999
    manager.checkpoint_path.write_text(json.dumps(payload), encoding="utf-8")

    recovered = _manager(tmp_path, "hash")
    checkpoint = recovered.load()

    assert checkpoint is not None
    assert recovered.recovered_from_backup is True
    assert recovered.get_resume_info()["completed_queries"] == 0


def test_unfinished_session_marker_is_rolled_back_for_retry(tmp_path: Path):
    manager = _manager(tmp_path, "hash")
    manager.create(total_personas=1, total_queries=2, evaluation_mode="independent")
    manager.start_persona(1)
    manager.mark_session_injected(11)
    manager.mark_session_started(12, unit_id=1)

    resumed = _manager(tmp_path, "hash")
    assert resumed.load() is not None
    rollback = resumed.rollback_incomplete_session()

    assert rollback["session_id"] == 12
    assert rollback["unit_id"] == 1
    assert resumed.is_session_injected(11, persona_id=1) is True
    assert resumed.is_session_injected(12, persona_id=1) is False
    assert resumed.get_resume_info()["active_session_id"] is None


def test_interrupt_during_memory_build_leaves_retry_marker(tmp_path: Path):
    manager = _manager(tmp_path, "hash")
    manager.create(total_personas=1, total_queries=0, evaluation_mode="independent")
    manager.start_persona(1)

    evaluator = MedMemoryBenchEvaluator.__new__(MedMemoryBenchEvaluator)
    evaluator.force_resume = False
    evaluator.dry_run = False
    evaluator.batch_api = False
    evaluator._checkpoint_manager = manager
    evaluator._memory_build_logs = []
    evaluator._deferred_judges = []
    evaluator._log = lambda *args, **kwargs: None
    evaluator.prompt_manager = SimpleNamespace(
        format_memorize=lambda context, timestamp: context
    )
    evaluator.agent_manager = SimpleNamespace(
        send_message=lambda **kwargs: (_ for _ in ()).throw(KeyboardInterrupt())
    )
    unit = EvaluationUnit(
        unit_id=1,
        context_id=1,
        sessions_to_inject=[MedSession(session_id=12, content="session twelve")],
        queries_to_evaluate=[],
    )

    with pytest.raises(KeyboardInterrupt):
        evaluator._evaluate_unit_with_checkpoint(unit)

    assert manager.get_resume_info()["active_session_id"] == 12
    assert manager.get_resume_info()["active_unit_id"] == 1
