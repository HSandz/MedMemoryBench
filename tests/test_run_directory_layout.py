from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import main as cli
from src.config import (
    DatasetConfig,
    MethodConfig,
    ModelConfig,
    dataset_config_from_snapshot,
    method_config_from_snapshot,
)
from src.evaluator import DATASET_EVALUATOR_REGISTRY, Evaluator, create_evaluator
from src.result import EvaluationReport, ResultCollector
from benchmarks.medmemorybench.checkpoint import (
    compute_config_hash,
    derive_legacy_build_config_hash,
    is_manifest_build_compatible,
)
from utils.vertex_batch import VertexBatchPending


def _configs(model_name: str = "test-model", temperature: float = 1.0):
    method = MethodConfig(
        method_name="amem",
        method_type="agentic_memory",
        model=ModelConfig(
            provider="openai",
            name=model_name,
            temperature=temperature,
            api_key="do-not-write-this-secret",
        ),
        raw_config={
            "method_name": "amem",
            "model": {
                "provider": "openai",
                "name": model_name,
                "temperature": temperature,
                "api_key": "do-not-write-this-secret",
            },
        },
    )
    dataset = DatasetConfig(
        dataset_name="run-layout-test",
        raw_config={"dataset_name": "run-layout-test"},
    )
    return method, dataset


def _fake_evaluate(**kwargs):
    report = EvaluationReport(
        method_name=kwargs["method_config"].method_name,
        model_name=kwargs["method_config"].model.name,
        dataset_name=kwargs["dataset_config"].dataset_name,
        start_time="2026-01-01T00:00:00",
        end_time="2026-01-01T00:00:01",
        duration_seconds=1.0,
        summary={"total": 0, "correct": 0, "overall_accuracy": 0.0},
        detailed_results=[],
        config={
            "method_config": kwargs["method_config"].raw_config,
            "dataset_config": kwargs["dataset_config"].raw_config,
        },
        metadata={"api_failures": []},
    )
    ResultCollector().save_reports(
        report,
        kwargs["output_dir"],
        [],
        use_method_subdir=not kwargs["run_scoped_output"],
    )
    return report


def _evaluator(
    tmp_path: Path,
    *,
    resume: bool = False,
    model_name: str = "test-model",
    temperature: float = 1.0,
    config_inference=None,
    execution_stage: str = "all",
    memory_run: str | None = None,
    memory_source_run_dir: Path | None = None,
):
    method, dataset = _configs(model_name, temperature)
    return Evaluator(
        method_config=method,
        dataset_config=dataset,
        output_dir=tmp_path,
        resume=resume,
        execution_stage=execution_stage,
        memory_run=memory_run,
        memory_source_run_dir=memory_source_run_dir,
        method_config_name="persona_1/amem_test",
        dataset_config_name="run-layout-test",
        method_config_path=tmp_path / "configs" / "method.yaml",
        dataset_config_path=tmp_path / "configs" / "dataset.yaml",
        config_inference=config_inference,
    )


def _write_memory_run(
    output_root: Path,
    run_id: str,
    *,
    experiment: str = "amem_test-model",
    status: str = "complete",
    method_config_name: str = "persona_1/amem_test",
    dataset_config_name: str = "medmemorybench",
    stored_method_name: str = "amem",
    manifest_method_name: str = "amem",
    stored_model_name: str = "test-model",
    manifest_model_name: str = "test-model",
    config_hash: str = "test-config-hash",
    write_run_config: bool = True,
    manifest_updates=None,
) -> Path:
    run_dir = output_root / experiment / run_id
    memory_dir = run_dir / "memory"
    memory_dir.mkdir(parents=True)
    manifest = {
        "format": "medmemorybench.memory_manifest",
        "version": 1,
        "status": status,
        "method_name": manifest_method_name,
        "model_name": manifest_model_name,
        "config_hash": config_hash,
    }
    if manifest_updates:
        manifest.update(manifest_updates)
    (memory_dir / "manifest.json").write_text(json.dumps(manifest))
    if write_run_config:
        (run_dir / "run_config.json").write_text(json.dumps({
            "format": "medmemorybench.run_config",
            "version": 1,
        "method_config_name": method_config_name,
        "dataset_config_name": dataset_config_name,
        "method_config": {
            "method_name": stored_method_name,
            "method_type": "agentic_memory",
            "model": {"provider": "openai", "name": stored_model_name},
            "agent_params": {},
            "raw_config": {
                "method_name": stored_method_name,
                "method_type": "agentic_memory",
                "model": {"provider": "openai", "name": stored_model_name},
            },
        },
        "dataset_config": {
            "dataset_name": dataset_config_name,
            "language": "en",
            "data_root_dir": "data",
            "data_files": {},
            "evaluation_mode": "independent",
            "persona_ids": [1],
            "max_personas": None,
            "max_sessions_per_persona": 1,
            "evaluation_interval": 10,
            "inject_noise": False,
            "query_types": [],
            "save_intermediate": True,
            "save_retrieved_context": True,
            "raw_config": {"dataset_name": dataset_config_name},
        },
    }))
    return run_dir


def test_legacy_manifest_derives_build_hash_from_stored_run_config(tmp_path: Path):
    run_dir = _write_memory_run(
        tmp_path,
        "20260813_143538",
        config_hash="legacy-full-config-hash",
    )
    manifest_path = run_dir / "memory" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    run_config = json.loads((run_dir / "run_config.json").read_text())
    method = method_config_from_snapshot(run_config["method_config"])
    dataset = dataset_config_from_snapshot(run_config["dataset_config"])
    expected_hash = compute_config_hash(method, dataset)

    assert derive_legacy_build_config_hash(manifest, manifest_path) == expected_hash
    assert is_manifest_build_compatible(manifest, expected_hash, manifest_path)

    manifest["build_config_hash"] = "explicit-but-wrong"
    assert not is_manifest_build_compatible(manifest, expected_hash, manifest_path)


def test_full_run_keeps_config_log_and_reports_in_one_timestamp_directory(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setitem(DATASET_EVALUATOR_REGISTRY, "run-layout-test", _fake_evaluate)
    evaluator = _evaluator(tmp_path)

    evaluator.run()

    assert evaluator.output_dir.parent == tmp_path / "amem_test-model"
    assert evaluator.output_dir.name[:8].isdigit()
    assert (evaluator.output_dir / "evaluation.log").exists()
    config_path = evaluator.output_dir / "run_config.json"
    assert config_path.exists()
    config = json.loads(config_path.read_text())
    assert config["status"] == "complete"
    assert config["execution"]["stage"] == "all"
    assert config["method_config"]["model"]["api_key"] == "<redacted>"
    assert config["method_config"]["raw_config"]["model"]["api_key"] == "<redacted>"
    assert config["config_sources"]["method"].endswith("configs/method.yaml")
    report_names = {path.name for path in evaluator.output_dir.glob("*.json")}
    assert any(name.endswith("_result.json") for name in report_names)
    assert any(name.endswith("_memory_build.json") for name in report_names)
    assert any(name.endswith("_query_answer.json") for name in report_names)
    assert not (evaluator.output_dir / "amem_test-model").exists()


def test_resume_reuses_only_an_incomplete_exactly_matching_run(tmp_path: Path):
    first = _evaluator(tmp_path)
    first._write_run_config(status="pending")

    resumed = _evaluator(tmp_path, resume=True)
    changed = _evaluator(tmp_path, resume=True, temperature=0.25)

    assert resumed.output_dir == first.output_dir
    assert changed.output_dir != first.output_dir
    resumed_config = json.loads((resumed.output_dir / "run_config.json").read_text())
    assert resumed_config["started_at"] == first.run_started_at.isoformat()
    assert len(resumed_config["invocations"]) == 2


def test_completed_run_is_never_reused_by_resume(tmp_path: Path):
    first = _evaluator(tmp_path)
    first._write_run_config(status="complete")

    resumed = _evaluator(tmp_path, resume=True)

    assert resumed.output_dir != first.output_dir


def test_explicit_query_runs_are_nested_under_the_memory_run(tmp_path: Path):
    method, dataset = _configs()
    source_run = _write_memory_run(
        tmp_path,
        "20260814_093118",
        config_hash=compute_config_hash(method, dataset),
    )

    first = _evaluator(
        tmp_path,
        execution_stage="query",
        memory_run=source_run.name,
        memory_source_run_dir=source_run,
    )
    first._write_run_config(status="pending")
    resumed = _evaluator(
        tmp_path,
        resume=True,
        execution_stage="query",
        memory_run=source_run.name,
        memory_source_run_dir=source_run,
    )

    assert first.output_dir.parent == source_run / "query_runs"
    assert resumed.output_dir == first.output_dir
    config = json.loads((first.output_dir / "run_config.json").read_text())
    assert config["execution"]["memory_source_run_dir"] == str(source_run.resolve())


def test_pending_batch_run_records_pending_status_for_resume(tmp_path: Path, monkeypatch):
    manifest_path = tmp_path / "batch" / "manifest.json"

    def pending_evaluate(**kwargs):
        raise VertexBatchPending("query-final", "projects/test/jobs/1", manifest_path)

    monkeypatch.setitem(
        DATASET_EVALUATOR_REGISTRY,
        "run-layout-test",
        pending_evaluate,
    )
    evaluator = _evaluator(tmp_path)

    with pytest.raises(VertexBatchPending):
        evaluator.run()

    config = json.loads((evaluator.output_dir / "run_config.json").read_text())
    assert config["status"] == "pending"
    assert config["batch_pending"] == {
        "stage": "query-final",
        "job_name": "projects/test/jobs/1",
        "manifest_path": str(manifest_path),
    }


def test_query_config_is_inferred_from_one_completed_memory_run(tmp_path: Path):
    run_dir = _write_memory_run(tmp_path, "20260813_143538")

    inferred = cli.infer_query_config_from_memory_run(
        tmp_path,
        "20260813_143538",
    )

    assert inferred["run_dir"] == run_dir
    assert inferred["method_config_name"] == "persona_1/amem_test"
    assert inferred["dataset_config_name"] == "medmemorybench"
    assert inferred["base_output_dir"] == tmp_path
    assert inferred["method_config_snapshot"]["raw_config"]["method_name"] == "amem"
    assert inferred["dataset_config_snapshot"]["raw_config"]["dataset_name"] == "medmemorybench"


def test_explicit_query_uses_run_snapshot_without_loading_current_yaml(tmp_path: Path):
    method, dataset = _configs()
    dataset.dataset_name = "medmemorybench"
    dataset.raw_config = {"dataset_name": "medmemorybench"}
    source_run = _write_memory_run(
        tmp_path,
        "20260813_143538",
        config_hash=compute_config_hash(method, dataset),
    )

    class ExplodingLoader:
        method_config_dir = tmp_path / "configs" / "method_config"
        dataset_config_dir = tmp_path / "configs" / "dataset_config"

        def load_method_config(self, _name):
            raise AssertionError("query must not load the current method YAML")

        def load_dataset_config(self, _name):
            raise AssertionError("query must not load the current dataset YAML")

    evaluator = create_evaluator(
        method_config_name="persona_1/amem_test",
        dataset_name="medmemorybench",
        config_loader=ExplodingLoader(),
        method_config=method,
        dataset_config=dataset,
        output_dir=tmp_path,
        execution_stage="query",
        memory_run=source_run.name,
        memory_source_run_dir=source_run,
    )

    assert evaluator.method_config is method
    assert evaluator.dataset_config is dataset


def test_append_create_evaluator_uses_the_source_snapshot_and_nested_output(
    tmp_path: Path,
):
    method, dataset = _configs()
    dataset.dataset_name = "medmemorybench"
    dataset.raw_config = {"dataset_name": "medmemorybench"}
    source_run = _write_memory_run(
        tmp_path,
        "20260813_143538",
        status="building",
        config_hash=compute_config_hash(method, dataset),
    )

    evaluator = create_evaluator(
        method_config_name="persona_1/amem_test",
        dataset_name="medmemorybench",
        method_config=method,
        dataset_config=dataset,
        output_dir=tmp_path,
        execution_stage="memory",
        memory_run=source_run.name,
        append=True,
        append_persona=1,
        append_unit=0,
        memory_source_run_dir=source_run,
    )

    assert evaluator.append is True
    assert evaluator.append_persona == 1
    assert evaluator.append_unit == 0
    assert evaluator.output_dir.parent == source_run / "query_runs"


def test_memory_run_accepts_underscore_cli_alias(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["main.py", "--stage", "query", "--memory_run", "20260813_143538"],
    )

    args = cli.parse_args()

    assert args.memory_run == "20260813_143538"


@pytest.mark.parametrize("stage", ["memory", "all"])
def test_append_cli_accepts_stage_and_target(monkeypatch, stage):
    argv = [
        "main.py",
        "--append",
        "--memory-run",
        "20260813_143538",
        "--persona",
        "4",
        "--unit",
        "12",
    ]
    if stage != "all":
        argv.extend(["--stage", stage])
    monkeypatch.setattr(sys, "argv", argv)

    args = cli.parse_args()

    assert args.append is True
    assert args.stage == stage
    assert args.persona == 4
    assert args.unit == 12


def test_append_cli_rejects_dry_run(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "--append",
            "--dry-run",
            "--memory-run",
            "20260813_143538",
            "--persona",
            "1",
            "--unit",
            "0",
        ],
    )

    with pytest.raises(SystemExit):
        cli.parse_args()


def test_append_config_inference_accepts_building_source_and_forwards_target(
    tmp_path: Path,
    monkeypatch,
):
    source_run = _write_memory_run(tmp_path, "20260813_143538", status="building")
    captured = {}

    class FakeEvaluator:
        output_dir = source_run / "query_runs" / "new-append-run"

        @staticmethod
        def run():
            return SimpleNamespace(
                summary={"correct": 0, "total": 0, "overall_accuracy": 0.0}
            )

    def fake_create_evaluator(**kwargs):
        captured.update(kwargs)
        return FakeEvaluator()

    monkeypatch.setattr(cli, "create_evaluator", fake_create_evaluator)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "--append",
            "--stage",
            "memory",
            "--memory-run",
            "20260813_143538",
            "--persona",
            "1",
            "--unit",
            "3",
            "-o",
            str(tmp_path),
        ],
    )

    assert cli.main() == 0
    assert captured["append"] is True
    assert captured["append_persona"] == 1
    assert captured["append_unit"] == 3
    assert captured["execution_stage"] == "memory"
    assert captured["memory_source_run_dir"] == source_run
    assert captured["config_inference"]["source_memory_status"] == "building"


def test_append_returns_without_creating_a_run_when_exact_target_exists(
    tmp_path: Path,
    monkeypatch,
):
    source_run = _write_memory_run(
        tmp_path,
        "20260813_143538",
        manifest_updates={
            "snapshots": [{
                "context_id": 1,
                "unit_id": 7,
                "path": "persona_1_unit_7.json",
            }],
        },
    )
    payload = {
        "format": "medmemorybench.memory_snapshot",
        "version": 1,
        "build_id": None,
        "context_id": 1,
        "unit_id": 7,
        "memory_state": {},
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    import hashlib
    payload["integrity_hash"] = hashlib.sha256(encoded).hexdigest()
    (source_run / "memory" / "persona_1_unit_7.json").write_text(
        json.dumps(payload)
    )

    def unexpected_create(**kwargs):
        raise AssertionError("already-processed append must not create a run")

    monkeypatch.setattr(cli, "create_evaluator", unexpected_create)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "--append",
            "--memory-run",
            source_run.name,
            "--persona",
            "1",
            "--unit",
            "7",
            "-o",
            str(tmp_path),
        ],
    )

    assert cli.main() == 0

def test_append_corrupt_exact_target_does_not_return_false_success(
    tmp_path: Path,
    monkeypatch,
):
    source_run = _write_memory_run(
        tmp_path,
        "20260813_143538",
        manifest_updates={
            "snapshots": [{
                "context_id": 1,
                "unit_id": 7,
                "path": "persona_1_unit_7.json",
            }],
        },
    )
    (source_run / "memory" / "persona_1_unit_7.json").write_text("{}")
    created = []

    class FakeEvaluator:
        output_dir = source_run / "query_runs" / "append"

        @staticmethod
        def run():
            return SimpleNamespace(
                summary={"correct": 0, "total": 0, "overall_accuracy": 0.0}
            )

    monkeypatch.setattr(
        cli,
        "create_evaluator",
        lambda **kwargs: created.append(kwargs) or FakeEvaluator(),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "--append",
            "--memory-run",
            source_run.name,
            "--persona",
            "1",
            "--unit",
            "7",
            "-o",
            str(tmp_path),
        ],
    )

    assert cli.main() == 0
    assert len(created) == 1


def test_append_does_not_treat_a_later_unit_as_the_requested_target():
    manifest = {
        "snapshots": [{
            "context_id": 1,
            "unit_id": 8,
            "path": "persona_1_unit_8.json",
        }],
    }

    assert cli.append_target_already_processed(manifest, 1, 7) is False
    assert cli.append_target_already_processed(manifest, 1, 8) is True


def test_nested_append_memory_run_infers_original_output_root(tmp_path: Path):
    source_run = _write_memory_run(tmp_path, "20260813_143538")
    child_run = source_run / "query_runs" / "20260813_143539"
    child_memory = child_run / "memory"
    child_memory.mkdir(parents=True)
    child_manifest = json.loads(
        (source_run / "memory" / "manifest.json").read_text()
    )
    child_manifest.update({
        "status": "complete",
        "build_id": "child-build",
        "unit_ids": [0],
        "snapshots": [],
    })
    (child_memory / "manifest.json").write_text(json.dumps(child_manifest))
    child_config = json.loads((source_run / "run_config.json").read_text())
    child_config["run_id"] = child_run.name
    child_config["output_dir"] = str(child_run)
    child_run.joinpath("run_config.json").write_text(json.dumps(child_config))

    inferred = cli.infer_query_config_from_memory_run(tmp_path, child_run.name)

    assert inferred["run_dir"] == child_run
    assert inferred["base_output_dir"] == tmp_path


@pytest.mark.parametrize(
    ("fixture_kwargs", "expected_message"),
    [
        ({"status": "building"}, "not 'complete'"),
        ({"write_run_config": False}, "missing run_config.json"),
        (
            {
                "stored_method_name": "other",
                "manifest_method_name": "amem",
            },
            "method identity disagrees",
        ),
        (
            {
                "stored_model_name": "other-model",
                "manifest_model_name": "test-model",
            },
            "model identity disagrees",
        ),
    ],
)
def test_query_config_inference_rejects_invalid_memory_sources(
    tmp_path: Path,
    fixture_kwargs,
    expected_message: str,
):
    _write_memory_run(tmp_path, "20260813_143538", **fixture_kwargs)

    with pytest.raises(FileNotFoundError, match=expected_message):
        cli.infer_query_config_from_memory_run(tmp_path, "20260813_143538")


def test_query_config_inference_rejects_duplicate_run_ids(tmp_path: Path):
    _write_memory_run(tmp_path, "20260813_143538", experiment="experiment-one")
    _write_memory_run(tmp_path, "20260813_143538", experiment="experiment-two")

    with pytest.raises(ValueError, match="Use a narrower -o/--output-dir"):
        cli.infer_query_config_from_memory_run(tmp_path, "20260813_143538")


def test_main_uses_inferred_method_dataset_and_output_root(tmp_path: Path, monkeypatch):
    source_run = _write_memory_run(tmp_path, "20260813_143538")
    captured = {}

    class FakeEvaluator:
        output_dir = source_run / "query_runs" / "new-query-run"

        @staticmethod
        def run():
            return SimpleNamespace(
                summary={"correct": 0, "total": 0, "overall_accuracy": 0.0}
            )

    def fake_create_evaluator(**kwargs):
        captured.update(kwargs)
        return FakeEvaluator()

    monkeypatch.setattr(cli, "create_evaluator", fake_create_evaluator)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "--stage",
            "query",
            "--memory-run",
            "20260813_143538",
            "-o",
            str(tmp_path),
        ],
    )

    assert cli.main() == 0
    assert captured["method_config_name"] == "persona_1/amem_test"
    assert captured["dataset_name"] == "medmemorybench"
    assert captured["output_dir"] == tmp_path
    assert captured["config_inference"]["source_run_dir"] == str(source_run)
    assert captured["memory_source_run_dir"] == source_run


def test_query_stage_accepts_an_explicit_method_config_file(
    tmp_path: Path,
    monkeypatch,
):
    source_run = _write_memory_run(tmp_path, "20260813_143538")
    captured = {}

    class FakeEvaluator:
        output_dir = source_run / "query_runs" / "explicit-query-run"

        @staticmethod
        def run():
            return SimpleNamespace(
                summary={"correct": 0, "total": 0, "overall_accuracy": 0.0}
            )

    def fake_create_evaluator(**kwargs):
        captured.update(kwargs)
        return FakeEvaluator()

    method_path = "configs/method_config/persona_1/amem_test_gemini.yaml"
    monkeypatch.setattr(cli, "create_evaluator", fake_create_evaluator)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "--stage",
            "query",
            "-m",
            method_path,
            "--memory-run",
            "20260813_143538",
            "-o",
            str(tmp_path),
        ],
    )

    assert cli.main() == 0
    assert captured["method_config_name"] == method_path
    assert captured["method_config"] is None
    assert captured["dataset_config"] is not None


def test_main_rejects_supplied_method_that_disagrees_with_memory_run(
    tmp_path: Path,
    monkeypatch,
):
    _write_memory_run(tmp_path, "20260813_143538")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "-m",
            "persona_1/wrong_method",
            "--stage",
            "query",
            "--memory-run",
            "20260813_143538",
            "-o",
            str(tmp_path),
        ],
    )

    assert cli.main() == 1


def test_run_config_records_inferred_configuration_source(tmp_path: Path):
    inference = {
        "source_run_id": "20260813_143538",
        "source_run_dir": str(tmp_path / "amem_test-model" / "20260813_143538"),
        "source_config_hash": "test-config-hash",
    }

    evaluator = _evaluator(tmp_path, config_inference=inference)

    config = json.loads((evaluator.output_dir / "run_config.json").read_text())
    assert config["config_inference"] == inference
