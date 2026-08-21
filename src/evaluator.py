import os
import json
import sys
from dataclasses import asdict, is_dataclass
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable, Dict, Any, List, Tuple

from src.config import MethodConfig, DatasetConfig, ConfigLoader, PROJECT_ROOT, get_api_config
from src.result import EvaluationReport, ResultCollector
from utils.llm_client import is_google_ai_studio_provider, is_vertex_batch_provider
from utils.logger import get_eval_logger
from utils.vertex_batch import VertexBatchPending


# {dataset_name: evaluate_function}
DATASET_EVALUATOR_REGISTRY: Dict[str, Callable] = {}


def register_evaluator(dataset_name: str):
    """
    Example:
        @register_evaluator("medmemorybench")
        def evaluate_medmemorybench(method_config, dataset_config, **kwargs):
            ...
    """
    def decorator(func: Callable):
        DATASET_EVALUATOR_REGISTRY[dataset_name.lower()] = func
        return func
    return decorator


class Evaluator:

    def __init__(
        self,
        method_config: MethodConfig,
        dataset_config: DatasetConfig,
        output_dir: Optional[Path] = None,
        dry_run: bool = False,
        verbose: bool = True,
        resume: bool = False,
        force_resume: bool = False,
        execution_stage: str = "all",
        memory_run: Optional[str] = None,
        append: bool = False,
        append_persona: Optional[int] = None,
        append_unit: Optional[int] = None,
        method_config_name: Optional[str] = None,
        dataset_config_name: Optional[str] = None,
        method_config_path: Optional[Path] = None,
        dataset_config_path: Optional[Path] = None,
        config_inference: Optional[Dict[str, Any]] = None,
        memory_source_run_dir: Optional[Path] = None,
        batch_api: bool = False,
        batch_gcs_uri: Optional[str] = None,
        batch_wait: bool = False,
    ):
        self.method_config = method_config
        self.dataset_config = dataset_config
        self.dry_run = dry_run
        self.verbose = verbose
        self.resume = resume
        self.force_resume = force_resume
        self.execution_stage = execution_stage
        self.memory_run = memory_run
        self.memory_run_explicit = memory_run is not None
        self.append = bool(append)
        self.append_persona = append_persona
        self.append_unit = append_unit
        self.method_config_name = method_config_name
        self.dataset_config_name = dataset_config_name
        self.method_config_path = method_config_path
        self.dataset_config_path = dataset_config_path
        self.config_inference = (
            self._redact_secrets(dict(config_inference))
            if config_inference
            else None
        )
        self._ai_studio_batch_skipped = bool(
            batch_api and is_google_ai_studio_provider(method_config.model.provider)
        )
        self.batch_api = batch_api and not self._ai_studio_batch_skipped
        self.batch_gcs_uri = (
            None
            if self._ai_studio_batch_skipped
            else batch_gcs_uri or os.environ.get("GOOGLE_BATCH_GCS_URI")
        )
        self.batch_wait = False if self._ai_studio_batch_skipped else batch_wait

        self.base_output_dir = output_dir or (PROJECT_ROOT / "outputs")
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
        safe_method = method_config.method_name.replace("/", "-").replace("\\", "-")
        safe_model = method_config.model.name.replace("/", "-").replace("\\", "-")
        self.experiment_dir = self.base_output_dir / f"{safe_method}_{safe_model}"
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        self.memory_source_run_dir = self._resolve_memory_source_run_dir(
            memory_source_run_dir
        )
        self.run_container_dir = (
            self.memory_source_run_dir / "query_runs"
            if self.memory_source_run_dir is not None
            else self.experiment_dir
        )
        self.run_container_dir.mkdir(parents=True, exist_ok=True)
        self.run_started_at = datetime.now()
        self._selected_run_config: Dict[str, Any] = {}
        self.run_id, self.output_dir = self._select_run_directory()
        if self._selected_run_config.get("started_at"):
            try:
                self.run_started_at = datetime.fromisoformat(
                    self._selected_run_config["started_at"]
                )
            except (TypeError, ValueError):
                pass
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.logger = get_eval_logger(
            method_config.method_name,
            dataset_config.dataset_name,
            log_dir=self.output_dir,
            log_filename="evaluation.log",
        )

        self.result_collector = ResultCollector()
        self._write_run_config(status="running", append_invocation=True)

    @staticmethod
    def _redact_secrets(value: Any) -> Any:
        if isinstance(value, dict):
            redacted = {}
            for key, item in value.items():
                lowered = str(key).lower()
                if any(
                    marker in lowered
                    for marker in (
                        "api_key", "secret", "password", "credential",
                        "access_token", "private_key",
                    )
                ):
                    redacted[key] = "<redacted>" if item else item
                else:
                    redacted[key] = Evaluator._redact_secrets(item)
            return redacted
        if isinstance(value, (list, tuple)):
            return [Evaluator._redact_secrets(item) for item in value]
        return value

    def _config_value(self, value: Any) -> Any:
        if is_dataclass(value):
            value = asdict(value)
        return self._redact_secrets(value)

    def _new_run_id(self) -> str:
        base = self.run_started_at.strftime("%Y%m%d_%H%M%S")
        candidate = base
        counter = 2
        while (self.run_container_dir / candidate).exists():
            candidate = f"{base}_{counter}"
            counter += 1
        return candidate

    def _resolve_memory_source_run_dir(
        self,
        configured_source: Optional[Path],
    ) -> Optional[Path]:
        """Resolve and validate the parent memory run for an explicit query."""
        if not self.memory_run or (self.execution_stage != "query" and not self.append):
            return None
        if self.memory_run == "legacy":
            if self.append:
                raise ValueError("--append cannot use the legacy memory layout")
            return None

        source_run = (
            Path(configured_source).resolve()
            if configured_source is not None
            else (self.experiment_dir / self.memory_run).resolve()
        )
        if source_run.name != self.memory_run:
            raise ValueError(
                "Memory source directory does not match --memory-run: "
                f"{source_run}"
            )

        manifest_path = source_run / "memory" / "manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Memory manifest not found for run '{self.memory_run}': "
                f"{manifest_path}"
            ) from exc
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"Cannot read memory manifest {manifest_path}: {exc}"
            ) from exc

        from benchmarks.medmemorybench.checkpoint import (
            compute_config_hash,
            is_manifest_build_compatible,
        )

        compatible = (
            isinstance(manifest, dict)
            and manifest.get("format") == "medmemorybench.memory_manifest"
            and manifest.get("version") == 1
            and manifest.get("status") in ({"complete", "building"} if self.append else {"complete"})
            and manifest.get("method_name") == self.method_config.method_name
            and is_manifest_build_compatible(
                manifest,
                compute_config_hash(self.method_config, self.dataset_config),
                manifest_path,
            )
        )
        if not compatible:
            raise ValueError(
                f"Memory run '{self.memory_run}' is unavailable or incompatible "
                "with the selected method and dataset configuration"
            )
        return source_run

    def _resume_identity(self) -> Dict[str, Any]:
        """Return the configuration fields that must agree for a safe resume."""
        return {
            "method_config_name": self.method_config_name,
            "dataset_config_name": self.dataset_config_name,
            "method_config": self._config_value(self.method_config),
            "dataset_config": self._config_value(self.dataset_config),
            "api_config": self._config_value(get_api_config()),
            "stage": self.execution_stage,
            "append": self.append,
            "append_persona": self.append_persona,
            "append_unit": self.append_unit,
            "dry_run": self.dry_run,
            "memory_run": self.memory_run,
            "batch_api": self.batch_api,
            "batch_gcs_uri": self.batch_gcs_uri,
        }

    @staticmethod
    def _identity_hash(identity: Dict[str, Any]) -> str:
        encoded = json.dumps(
            identity,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
        import hashlib
        return hashlib.sha256(encoded).hexdigest()

    def _resume_run_candidates(self) -> List[Path]:
        candidates = []
        for path in self.run_container_dir.iterdir():
            config_path = path / "run_config.json"
            if not path.is_dir() or not config_path.exists():
                continue
            try:
                payload = json.loads(config_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if payload.get("status") == "complete":
                continue
            expected_hash = self._identity_hash(self._resume_identity())
            matches = payload.get("resume_identity_hash") == expected_hash
            if (
                not matches
                and self.force_resume
                and not self.method_config.method_name.lower().startswith("amem")
            ):
                matches = (
                    payload.get("method_config_name") == self.method_config_name
                    and payload.get("dataset_config_name") == self.dataset_config_name
                    and payload.get("execution", {}).get("stage") == self.execution_stage
                )
            if not matches:
                continue
            candidates.append(path)
        return sorted(candidates, key=lambda path: path.name, reverse=True)

    def _select_run_directory(self) -> Tuple[str, Path]:
        if self.resume:
            for candidate in self._resume_run_candidates():
                config_path = candidate / "run_config.json"
                try:
                    self._selected_run_config = json.loads(
                        config_path.read_text(encoding="utf-8")
                    )
                except (OSError, json.JSONDecodeError):
                    continue
                return candidate.name, candidate
        run_id = self._new_run_id()
        return run_id, self.run_container_dir / run_id

    def _run_config_payload(self, status: str, **updates: Any) -> Dict[str, Any]:
        payload = {
            "format": "medmemorybench.run_config",
            "version": 1,
            "run_id": self.run_id,
            "status": status,
            "started_at": self.run_started_at.isoformat(),
            "updated_at": datetime.now().isoformat(),
            "method_config_name": self.method_config_name,
            "dataset_config_name": self.dataset_config_name,
            "method_config": self._config_value(self.method_config),
            "dataset_config": self._config_value(self.dataset_config),
            "api_config": self._config_value(get_api_config()),
            "config_sources": {
                "method": str(self.method_config_path) if self.method_config_path else None,
                "dataset": str(self.dataset_config_path) if self.dataset_config_path else None,
            },
            "resume_identity_hash": self._identity_hash(self._resume_identity()),
            "execution": {
                "stage": self.execution_stage,
                "append": self.append,
                "append_persona": self.append_persona,
                "append_unit": self.append_unit,
                "dry_run": self.dry_run,
                "resume": self.resume,
                "force_resume": self.force_resume,
                "memory_run": self.memory_run,
                "memory_source_run_dir": (
                    str(self.memory_source_run_dir)
                    if self.memory_source_run_dir is not None
                    else None
                ),
                "batch_api": self.batch_api,
                "batch_gcs_uri": self.batch_gcs_uri,
                "batch_wait": self.batch_wait,
            },
            "command": [sys.executable, *sys.argv],
            "output_dir": str(self.output_dir),
        }
        if self.config_inference:
            payload["config_inference"] = self.config_inference
        payload.update(updates)
        return payload

    def _write_run_config(
        self,
        status: str,
        *,
        append_invocation: bool = False,
        **updates: Any,
    ) -> None:
        path = self.output_dir / "run_config.json"
        existing = {}
        if path.exists():
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                existing = {}
        payload = dict(existing)
        payload.update(self._run_config_payload(status))
        payload.update(updates)
        payload["started_at"] = existing.get(
            "started_at", self.run_started_at.isoformat()
        )
        if append_invocation:
            invocations = list(existing.get("invocations", []))
            invocations.append({
                "invoked_at": datetime.now().isoformat(),
                "command": [sys.executable, *sys.argv],
                "resume": self.resume,
                "force_resume": self.force_resume,
                "batch_wait": self.batch_wait,
            })
            payload["invocations"] = invocations
        payload["status"] = status
        payload["updated_at"] = datetime.now().isoformat()
        if status not in {"failed", "interrupted"}:
            payload.pop("error", None)
        if status == "complete":
            payload.pop("batch_pending", None)
        temporary_path = path.with_suffix(".tmp")
        with temporary_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        try:
            directory_descriptor = os.open(path.parent, os.O_RDONLY)
        except OSError:
            directory_descriptor = None
        if directory_descriptor is not None:
            try:
                os.fsync(directory_descriptor)
            except OSError:
                pass
            finally:
                os.close(directory_descriptor)

    def _log(self, message: str, level: str = "INFO") -> None:
        if self.verbose:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] [{level}] {message}")
        self.logger.info(message)

    def run(self) -> EvaluationReport:
        _ensure_evaluators_registered()

        dataset_name = self.dataset_config.dataset_name.lower()

        evaluate_func = DATASET_EVALUATOR_REGISTRY.get(dataset_name)
        if evaluate_func is None:
            raise ValueError(
                f"No evaluator found for dataset '{dataset_name}', "
                f"available: {list(DATASET_EVALUATOR_REGISTRY.keys())}"
            )

        start_time = datetime.now()
        self._log("=" * 60)
        self._log("Starting evaluation")
        self._log(f"  Method: {self.method_config.method_name}")
        self._log(f"  Provider: {self.method_config.model.provider}")
        self._log(f"  Model: {self.method_config.model.name}")
        self._log(f"  Dataset: {self.dataset_config.dataset_name}")
        self._log(f"  Dry Run: {self.dry_run}")
        self._log(f"  Resume: {self.resume}")
        self._log(f"  Force Resume: {self.force_resume}")
        self._log(f"  Execution Stage: {self.execution_stage}")
        if self.memory_run:
            self._log(f"  Memory Run: {self.memory_run}")
        if self.memory_source_run_dir is not None:
            self._log(f"  Memory Source Directory: {self.memory_source_run_dir}")
        if self.config_inference:
            self._log(
                "  Configuration Source: "
                f"{self.config_inference.get('source_run_dir')}"
            )
        if self._ai_studio_batch_skipped:
            self._log(
                "  Google AI Studio does not use this repository's Vertex batch path; "
                "ignoring --batch-api, --batch-gcs-uri, and --batch-wait.",
                level="WARNING",
            )
        self._log(f"  Vertex Batch API: {self.batch_api}")
        self._log("=" * 60)

        try:
            if self.batch_api and not self.dry_run:
                self._validate_batch_eligibility()
            report = evaluate_func(
                method_config=self.method_config,
                dataset_config=self.dataset_config,
                output_dir=self.output_dir,
                dry_run=self.dry_run,
                verbose=self.verbose,
                logger=self.logger,
                resume=self.resume,
                force_resume=self.force_resume,
                execution_stage=self.execution_stage,
                memory_run=self.memory_run,
                append=self.append,
                append_persona=self.append_persona,
                append_unit=self.append_unit,
                memory_source_run_dir=self.memory_source_run_dir,
                memory_run_explicit=self.memory_run_explicit,
                run_scoped_output=True,
                batch_api=self.batch_api,
                batch_gcs_uri=self.batch_gcs_uri,
                batch_wait=self.batch_wait,
            )
        except VertexBatchPending as exc:
            self._write_run_config(
                status="pending",
                batch_pending={
                    "stage": exc.stage,
                    "job_name": exc.job_name,
                    "manifest_path": str(exc.manifest_path),
                },
            )
            raise
        except BaseException as exc:
            self._write_run_config(
                status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                error={"type": type(exc).__name__, "message": str(exc)},
            )
            raise

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        self._log("=" * 60)
        self._log(f"Evaluation completed in {duration:.2f}s")
        self._log(f"  Total Queries: {report.summary.get('total', 0)}")
        self._log(f"  Accuracy: {report.summary.get('overall_accuracy', 0):.2%}")
        self._log("=" * 60)

        self._write_run_config(
            status="complete",
            completed_at=end_time.isoformat(),
            duration_seconds=(end_time - self.run_started_at).total_seconds(),
            last_invocation_duration_seconds=duration,
            summary=report.summary,
        )

        return report

    def _validate_batch_eligibility(self) -> None:
        """Fail early when --batch-api cannot reach any safe Gemini stage."""
        provider = self.method_config.model.provider.lower()
        method_name = self.method_config.method_name.lower()
        eligible_methods = (
            "long_context", "embedding_rag", "bm25_rag", "lightmem",
            "zep", "remem", "graph_rag", "amem", "mem0", "memos",
            "memrl", "mirix",
        )
        agent_params = getattr(self.method_config, "raw_config", {}).get("agent_params", {})
        has_method_stage = (
            is_vertex_batch_provider(provider)
            and any(name in method_name for name in eligible_methods)
            and not (
                "mirix" in method_name
                and agent_params.get("use_native_query", True)
            )
        )
        has_medmemorybench_judge = (
            self.dataset_config.dataset_name.lower() == "medmemorybench"
            and is_vertex_batch_provider(get_api_config().get_judge_provider())
        )
        if not has_method_stage and not has_medmemorybench_judge:
            raise ValueError(
                "--batch-api requires at least one eligible Gemini stage. "
                "Use a supported Gemini method or configure a Gemini MedMemoryBench judge."
            )
        if not self.batch_gcs_uri and not os.environ.get("GOOGLE_BATCH_GCS_URI"):
            raise ValueError(
                "--batch-api requires --batch-gcs-uri or GOOGLE_BATCH_GCS_URI before evaluation starts."
            )


def _ensure_evaluators_registered():
    """Lazy import to avoid circular import."""
    if "medmemorybench" not in DATASET_EVALUATOR_REGISTRY:
        from benchmarks.medmemorybench.evaluator import evaluate_medmemorybench  # noqa: F401
    if "locomo" not in DATASET_EVALUATOR_REGISTRY:
        from benchmarks.locomo.evaluator import evaluate_locomo  # noqa: F401


def create_evaluator(
    method_config_name: str,
    dataset_name: str,
    config_loader: Optional[ConfigLoader] = None,
    method_config: Optional[MethodConfig] = None,
    dataset_config: Optional[DatasetConfig] = None,
    **kwargs
) -> Evaluator:
    if method_config is None or dataset_config is None:
        if config_loader is None:
            config_loader = ConfigLoader()
        method_config = method_config or config_loader.load_method_config(method_config_name)
        dataset_config = dataset_config or config_loader.load_dataset_config(dataset_name)

    dataset_overrides = method_config.raw_config.get("dataset_overrides", {})
    if dataset_overrides:
        allowed_overrides = {
            "persona_ids", "max_personas", "max_sessions_per_persona",
            "evaluation_interval", "inject_noise", "evaluation_mode",
        }
        unexpected = set(dataset_overrides) - allowed_overrides
        if unexpected:
            raise ValueError(
                f"Unsupported dataset overrides in {method_config_name}: "
                f"{', '.join(sorted(unexpected))}"
            )
        dataset_config = replace(dataset_config, **dataset_overrides)

    if config_loader is None:
        config_loader = ConfigLoader()

    return Evaluator(
        method_config=method_config,
        dataset_config=dataset_config,
        method_config_name=method_config_name,
        dataset_config_name=dataset_name,
        method_config_path=config_loader.method_config_dir / f"{method_config_name}.yaml",
        dataset_config_path=config_loader.dataset_config_dir / f"{dataset_name}.yaml",
        **kwargs
    )


def list_available_evaluators() -> list:
    return list(DATASET_EVALUATOR_REGISTRY.keys())
