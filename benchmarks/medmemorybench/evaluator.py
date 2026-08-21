"""MedMemoryBench evaluation module."""

import hashlib
import copy
import json
import os
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging

import numpy as np

from src.config import MethodConfig, DatasetConfig, PROJECT_ROOT, get_api_config
from src.evaluator import register_evaluator
from src.agent import AgentManager, AgentResponse, MemoryBuildResult
from src.result import EvaluationReport, ResultCollector
from benchmarks.medmemorybench.dataset import MedMemoryBenchDataset, MedQuery, MedSession
from benchmarks.base import EvaluationUnit
from metrics import MetricsCalculator, MetricsAggregator, MetricResult
from metrics.base import MetricResult as BaseMetricResult
from utils.templates import get_prompt_manager
from utils.llm_client import (
    LLMAPIError,
    LLMRetryExhaustedError,
    TokenUsage,
    diff_usage_stats,
    get_usage_tracker,
    merge_usage_stats,
)
from utils.vertex_batch import (
    BatchChatRequest,
    VertexBatchClient,
    VertexBatchError,
    make_request_id,
    PREPARED_QUERY_METADATA_KEY,
    restore_prepared_query,
    scoped_manifest_path,
    snapshot_prepared_query,
    should_use_batch,
)

from benchmarks.medmemorybench.checkpoint import (
    MedMemoryBenchCheckpointManager,
    compute_config_hash,
    compute_query_config_hash,
    derive_legacy_build_config_hash,
    is_manifest_build_compatible,
)


class MedMemoryBenchEvaluator:
    """MedMemoryBench evaluator with incremental evaluation and checkpoint support."""

    def __init__(
        self,
        method_config: MethodConfig,
        dataset_config: DatasetConfig,
        output_dir: Path,
        dry_run: bool = False,
        verbose: bool = True,
        logger: Optional[logging.Logger] = None,
        resume: bool = False,
        force_resume: bool = False,
        execution_stage: str = "all",
        memory_run: Optional[str] = None,
        append: bool = False,
        append_persona: Optional[int] = None,
        append_unit: Optional[int] = None,
        memory_source_run_dir: Optional[Path] = None,
        memory_run_explicit: bool = False,
        run_scoped_output: bool = False,
        batch_api: bool = False,
        batch_gcs_uri: Optional[str] = None,
        batch_wait: bool = False,
    ):
        self.method_config = method_config
        self.dataset_config = dataset_config
        self.output_dir = output_dir
        self.dry_run = dry_run
        self.verbose = verbose
        self.logger = logger
        self.resume = resume
        self.force_resume = force_resume
        if execution_stage not in {"all", "memory", "query"}:
            raise ValueError(f"Unsupported execution stage: {execution_stage}")
        self.execution_stage = execution_stage
        self.memory_run = memory_run
        self.append = bool(append)
        self.append_persona = append_persona
        self.append_unit = append_unit
        self.memory_source_run_dir = (
            Path(memory_source_run_dir).resolve()
            if memory_source_run_dir is not None
            else None
        )
        self.memory_run_explicit = memory_run_explicit
        self.run_scoped_output = run_scoped_output
        self.batch_api = batch_api
        self.batch_gcs_uri = batch_gcs_uri
        self.batch_wait = batch_wait

        self.prompt_manager = get_prompt_manager(
            dataset=dataset_config.dataset_name,
            method=method_config.method_name,
            language=dataset_config.language,
        )

        self.agent_manager: Optional[AgentManager] = None
        self.dataset: Optional[MedMemoryBenchDataset] = None

        api_config = get_api_config()
        metric_mapping = {
            query_type.name: query_type.metric
            for query_type in dataset_config.query_types
            if query_type.name and query_type.metric
        }
        self.metrics_calculator = MetricsCalculator(
            custom_mapping=metric_mapping,
            judge_model=api_config.judge_model or None,
            judge_api_key=api_config.judge_api_key or None,
            judge_base_url=api_config.judge_base_url or None,
            judge_temperature=getattr(api_config, "judge_temperature", 1.0),
            judge_client_max_tokens=getattr(api_config, "judge_client_max_tokens", 10000),
            judge_max_tokens=getattr(api_config, "judge_max_tokens", 500),
            judge_mcd_max_tokens=getattr(api_config, "judge_mcd_max_tokens", 2000),
            language=dataset_config.language,
        )
        self.aggregator = MetricsAggregator()
        self.result_collector = ResultCollector()

        self._memory_build_logs: List[Dict[str, Any]] = []
        self._source_memory_build_logs: List[Dict[str, Any]] = []
        self._batch_client: Optional[VertexBatchClient] = None
        self._judge_batch_client: Optional[VertexBatchClient] = None
        self._batch_fallback_logged = False
        self._judge_batch_fallback_logged = False
        self._deferred_judges: List[Dict[str, Any]] = []
        self._pending_batch_queries: List[Dict[str, Any]] = []
        self._api_failures: List[Dict[str, Any]] = []
        self._api_failure_duration_seconds = 0.0

        self._checkpoint_manager: Optional[MedMemoryBenchCheckpointManager] = None
        self._checkpoint_enabled = False
        self._force_resume_persona_id: Optional[int] = None
        self._memory_snapshot_manifest: Optional[Dict[str, Any]] = None
        self._memory_snapshot_run_dir: Optional[Path] = None
        self._evaluation_units: List[EvaluationUnit] = []
        self._memory_unit_ids: Optional[set[int]] = None
        self._append_target_index: Optional[int] = None
        self._append_target_unit_id: Optional[int] = None
        self._append_target_reached = False
        self._append_source_snapshot_keys: set[tuple[int, int]] = set()
        self._append_source_snapshot_records: Dict[tuple[int, int], Dict[str, Any]] = {}
        self._append_source_manifest: Optional[Dict[str, Any]] = None

        if self._should_enable_checkpoint():
            self._init_checkpoint_manager()

    def _log(self, message: str, level: str = "INFO") -> None:
        if self.verbose:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] [{level}] {message}")
        if self.logger:
            self.logger.info(message)

    def _amem_feature_configuration(self) -> Dict[str, Any]:
        """Describe the build feature combination represented by this run."""
        method_config = getattr(self, "method_config", None)
        method_name = str(getattr(method_config, "method_name", ""))
        build_config = getattr(method_config, "build_config", None)
        if not isinstance(build_config, dict):
            build_config = (
                method_config.snapshot_build_config()
                if hasattr(method_config, "snapshot_build_config")
                else {}
            )
        is_experimental = method_name.lower().startswith("amem_test")
        features = {
            "base_memory": True,
            "original_evolution": bool(
                build_config.get("amem_original_evolution", not is_experimental)
            ),
            "typed_relations": bool(build_config.get("amem_typed_relations", False)),
            "temporal_state": bool(build_config.get("amem_temporal_state", False)),
            "provenance": bool(build_config.get("amem_provenance", False)),
        }
        enabled = [name for name, enabled_flag in features.items() if enabled_flag]
        combination = "+".join(enabled) if enabled else "none"
        dependencies = []
        if features["temporal_state"] and not features["typed_relations"]:
            dependencies.append(
                "temporal_state records timestamps, but inferred transitions require typed_relations"
            )
        return {
            "schema_version": 1,
            "method_name": method_name,
            "combination_id": combination,
            "enabled_features": enabled,
            "features": features,
            "dependencies": dependencies,
            "build_config_hash": (
                self._batch_config_hash()
                if method_config is not None and hasattr(self, "dataset_config")
                else ""
            ),
            "build_config": copy.deepcopy(build_config),
        }

    @staticmethod
    def _feature_for_operation(operation: str) -> str:
        parts = str(operation).split(".")
        if len(parts) >= 2 and parts[0] == "amem":
            return parts[1]
        return "unscoped"

    def _build_metrics_report(
        self,
        logs: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Aggregate per-unit telemetry into a report-ready feature breakdown."""
        selected_logs = list(
            logs
            if logs is not None
            else self._combined_memory_build_logs()
        )
        usage_values = []
        units = []
        total_wall_time = 0.0
        total_sessions = 0
        complete_telemetry = True
        measured_memory_units = 0
        memory_snapshot_bytes = 0
        previous_memory_bytes: Dict[Any, int] = {}
        latest_memory_by_context: Dict[Any, Dict[str, Any]] = {}
        for log in selected_logs:
            metrics = log.get("build_metrics") or {}
            usage = metrics.get("usage") or {}
            if usage:
                usage_values.append(usage)
            wall_time = float(
                metrics.get("wall_time_seconds", log.get("total_time", 0.0)) or 0.0
            )
            total_wall_time += wall_time
            total_sessions += int(log.get("session_count", 0) or 0)
            legacy_metrics_unavailable = bool(
                metrics.get("legacy_metrics_unavailable", False)
            )
            complete_telemetry = complete_telemetry and not legacy_metrics_unavailable
            context_id = log.get("context_id")
            memory_size = log.get("memory_size") or metrics.get("memory_size")
            unit_memory_size: Optional[Dict[str, Any]] = None
            if isinstance(memory_size, dict) and memory_size.get("bytes") is not None:
                unit_memory_size = copy.deepcopy(memory_size)
                memory_bytes = int(unit_memory_size["bytes"])
                prior_bytes = previous_memory_bytes.get(context_id, 0)
                delta_bytes = memory_bytes - prior_bytes
                unit_memory_size.update({
                    "delta_from_previous_unit_bytes": delta_bytes,
                    "delta_from_previous_unit_mib": round(
                        delta_bytes / (1024 ** 2), 6
                    ),
                })
                previous_memory_bytes[context_id] = memory_bytes
                measured_memory_units += 1
                memory_snapshot_bytes += memory_bytes
                latest_memory_by_context[context_id] = {
                    "unit_id": log.get("unit_id"),
                    "bytes": memory_bytes,
                    "mib": round(memory_bytes / (1024 ** 2), 6),
                }
            units.append({
                "unit_id": log.get("unit_id"),
                "context_id": context_id,
                "session_count": log.get("session_count", 0),
                "wall_time_seconds": round(wall_time, 6),
                "restored_from_snapshot": bool(log.get("restored_from_snapshot", False)),
                "legacy_metrics_unavailable": legacy_metrics_unavailable,
                "memory_size": unit_memory_size,
                "usage": usage,
            })

        merged_usage = merge_usage_stats(*usage_values)
        operations = (
            merged_usage.get("operations", {}).get("memorize", {})
        )
        feature_configuration = self._amem_feature_configuration()
        feature_totals: Dict[str, TokenUsage] = {
            "base": TokenUsage(),
            "embedding": TokenUsage(),
        }
        for feature, enabled in feature_configuration["features"].items():
            if enabled and feature != "base_memory":
                feature_totals.setdefault(feature, TokenUsage())
        for operation, usage in operations.items():
            feature = self._feature_for_operation(operation)
            feature_totals.setdefault(feature, TokenUsage()).merge(
                TokenUsage.from_dict(usage)
            )
        scoped_wall_time = sum(
            float(usage.get("wall_time", 0.0) or 0.0)
            for usage in operations.values()
        )
        memorize_totals = dict(merged_usage.get("memorize_phase", {}))
        memorize_totals.pop("wall_time", None)
        memorize_totals.update({
            "wall_time_seconds": round(total_wall_time, 6),
            "scoped_operation_wall_time_seconds": round(scoped_wall_time, 6),
            "unattributed_wall_time_seconds": round(
                max(total_wall_time - scoped_wall_time, 0.0), 6
            ),
        })
        overall_memory_bytes = sum(
            item["bytes"] for item in latest_memory_by_context.values()
        )
        memory_size_report = {
            "schema_version": 1,
            "measurement": "serialized_memory_state",
            "unit": "bytes",
            "available": measured_memory_units > 0,
            "complete": bool(units) and measured_memory_units == len(units),
            "overall_bytes": overall_memory_bytes,
            "overall_mib": round(overall_memory_bytes / (1024 ** 2), 6),
            "unit_snapshot_bytes": memory_snapshot_bytes,
            "unit_snapshot_mib": round(memory_snapshot_bytes / (1024 ** 2), 6),
            "context_count": len({unit.get("context_id") for unit in units}),
            "measured_unit_count": measured_memory_units,
            "by_context": {
                str(context_id): value
                for context_id, value in sorted(
                    latest_memory_by_context.items(),
                    key=lambda item: str(item[0]),
                )
            },
        }
        memorize_totals.update({
            "memory_size_bytes": overall_memory_bytes,
            "memory_size_mib": memory_size_report["overall_mib"],
        })
        return {
            "schema_version": 2,
            "feature_configuration": feature_configuration,
            "unit_count": len(units),
            "session_count": total_sessions,
            "complete_telemetry": complete_telemetry,
            "totals": memorize_totals,
            "memory_size": memory_size_report,
            "by_feature": {
                feature: usage.to_dict()
                for feature, usage in sorted(feature_totals.items())
            },
            "by_operation": operations,
            "units": units,
            "usage": merged_usage,
        }

    def _combined_memory_build_logs(self) -> List[Dict[str, Any]]:
        """Return one cumulative build record per evaluation unit."""
        combined: Dict[tuple[Any, Any], Dict[str, Any]] = {}
        for log in getattr(self, "_source_memory_build_logs", []):
            combined[(log.get("context_id"), log.get("unit_id"))] = log
        for log in getattr(self, "_memory_build_logs", []):
            combined[(log.get("context_id"), log.get("unit_id"))] = log
        unit_order = {
            (unit.context_id, unit.unit_id): index
            for index, unit in enumerate(self._get_evaluation_units())
        }
        return sorted(
            combined.values(),
            key=lambda log: unit_order.get(
                (log.get("context_id"), log.get("unit_id")),
                len(unit_order),
            ),
        )

    def _record_source_build_metrics(
        self,
        unit: EvaluationUnit,
        payload: Dict[str, Any],
    ) -> None:
        metrics = payload.get("memory_build_metrics")
        if not isinstance(metrics, dict):
            metrics = {
                "wall_time_seconds": float(
                    payload.get("memory_build_time", 0.0) or 0.0
                ),
                "usage": {},
                "legacy_metrics_unavailable": True,
            }
        logs = getattr(self, "_source_memory_build_logs", None)
        if logs is None:
            self._source_memory_build_logs = []
            logs = self._source_memory_build_logs
        logs[:] = [item for item in logs if item.get("unit_id") != unit.unit_id]
        logs.append({
            "unit_id": unit.unit_id,
            "context_id": unit.context_id,
            "session_count": len(unit.sessions_to_inject),
            "total_time": payload.get("memory_build_time", 0.0),
            "build_metrics": copy.deepcopy(metrics),
            "memory_size": copy.deepcopy(
                payload.get("memory_size") or metrics.get("memory_size")
            ),
            "restored_from_snapshot": True,
        })

    def _record_api_failure(
        self,
        phase: str,
        error: Any,
        failure_duration_seconds: Optional[float] = None,
        **context: Any,
    ) -> None:
        """Record provider failures without constructing a metric result."""
        if failure_duration_seconds is None:
            failure_duration_seconds = getattr(
                error,
                "_medmemorybench_api_duration_seconds",
                0.0,
            )
        try:
            failure_duration_seconds = max(float(failure_duration_seconds), 0.0)
        except (TypeError, ValueError):
            failure_duration_seconds = 0.0

        root_error = (
            error.last_exception
            if isinstance(error, LLMRetryExhaustedError)
            else error
        )
        failure = {
            "timestamp": datetime.now().isoformat(),
            "phase": phase,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "root_error_type": type(root_error).__name__,
            "root_error_message": str(root_error),
            **{key: value for key, value in context.items() if value is not None},
        }
        if failure_duration_seconds:
            failure["duration_seconds"] = failure_duration_seconds
            self._api_failure_duration_seconds = (
                getattr(self, "_api_failure_duration_seconds", 0.0)
                + failure_duration_seconds
            )
        if isinstance(error, LLMRetryExhaustedError):
            failure["attempts"] = error.attempts
            if error.failure_type:
                failure["failure_type"] = error.failure_type
            if error.retry_counts:
                failure["retry_counts"] = error.retry_counts
        self._api_failures.append(failure)

    @staticmethod
    def _run_api_call(operation, *args, **kwargs):
        """Attach elapsed time to an API exception without changing its type."""
        started_at = time.perf_counter()
        try:
            return operation(*args, **kwargs)
        except LLMAPIError as error:
            previous_duration = getattr(
                error,
                "_medmemorybench_api_duration_seconds",
                0.0,
            )
            error._medmemorybench_api_duration_seconds = (
                max(float(previous_duration), 0.0)
                + time.perf_counter() - started_at
            )
            raise

    def _init_dataset(self) -> None:
        self._log(f"Loading dataset: {self.dataset_config.dataset_name}")
        data_dir = PROJECT_ROOT / self.dataset_config.data_root_dir

        self.dataset = MedMemoryBenchDataset(
            data_dir=data_dir,
            config={
                "evaluation_mode": self.dataset_config.evaluation_mode,
                "persona_ids": self.dataset_config.persona_ids,
                "max_personas": self.dataset_config.max_personas,
                "max_sessions_per_persona": self.dataset_config.max_sessions_per_persona,
                "evaluation_interval": self.dataset_config.evaluation_interval,
                "inject_noise": self.dataset_config.inject_noise,
                "query_types": [qt.name for qt in self.dataset_config.query_types] if self.dataset_config.query_types else None,
            }
        )
        self.dataset.load()
        self._evaluation_units = list(self.dataset.get_evaluation_units())

        self._log(f"  Total Sessions: {self.dataset.get_total_sessions()}")
        self._log(f"  Total Queries: {self.dataset.get_total_queries()}")
        self._log(f"  Evaluation Mode: {self.dataset_config.evaluation_mode}")
        self._log(f"  Evaluation Interval: {self.dataset_config.evaluation_interval}")

    def _should_enable_checkpoint(self) -> bool:
        return self.dataset_config.evaluation_mode == "independent"

    def _session_level_force_resume(self) -> bool:
        """Use forced session skipping only for backends without A-MEM snapshots."""
        method_name = str(
            getattr(getattr(self, "method_config", None), "method_name", "")
        ).lower()
        return bool(getattr(self, "force_resume", False) and not method_name.startswith("amem"))

    def _get_evaluation_units(self) -> List[EvaluationUnit]:
        """Return the loaded units without consuming the dataset iterator twice."""
        cached = getattr(self, "_evaluation_units", None)
        if cached:
            return list(cached)
        if self.dataset is None:
            return []
        return list(self.dataset.get_evaluation_units())

    def _prepare_append_plan(self) -> None:
        """Resolve the append target and validate the source snapshot prefix."""
        if not getattr(self, "append", False):
            return
        if self.dataset_config.evaluation_mode != "independent":
            raise ValueError("--append currently requires independent evaluation mode")
        if self.memory_source_run_dir is None:
            raise ValueError("--append requires a memory source run")
        if getattr(self, "append_persona", None) is None or getattr(self, "append_unit", None) is None:
            raise ValueError("--append requires both --persona and --unit")

        units = self._get_evaluation_units()
        target_matches = [
            (index, unit)
            for index, unit in enumerate(units)
            if unit.context_id == getattr(self, "append_persona", None)
            and unit.unit_id == getattr(self, "append_unit", None)
        ]
        if len(target_matches) != 1:
            raise ValueError(
                f"Append target persona={getattr(self, 'append_persona', None)}, "
                f"unit={getattr(self, 'append_unit', None)} "
                "does not identify exactly one evaluation unit in the stored dataset"
            )
        target_index, target_unit = target_matches[0]
        self._append_target_index = target_index
        self._append_target_unit_id = target_unit.unit_id
        target_key = (target_unit.context_id, target_unit.unit_id)

        manifest_path = self.memory_source_run_dir / "memory" / "manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Cannot read append source manifest {manifest_path}: {exc}") from exc
        if (
            not isinstance(manifest, dict)
            or manifest.get("format") != "medmemorybench.memory_manifest"
            or manifest.get("version") != 1
            or manifest.get("status") not in {"building", "complete"}
        ):
            raise ValueError(f"Append source manifest is unavailable or invalid: {manifest_path}")

        unit_lookup = {(unit.context_id, unit.unit_id): index for index, unit in enumerate(units)}
        records: Dict[tuple[int, int], Dict[str, Any]] = {}
        processed_indices: List[int] = []
        for record in manifest.get("snapshots", []):
            if not isinstance(record, dict):
                raise ValueError(f"Append source manifest contains an invalid snapshot record: {manifest_path}")
            try:
                key = (int(record["context_id"]), int(record["unit_id"]))
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Append source snapshot record is missing its unit identity: {manifest_path}") from exc
            if key not in unit_lookup:
                raise ValueError(
                    f"Append source snapshot {key} does not belong to the stored dataset configuration"
                )
            if key in records:
                raise ValueError(f"Append source contains duplicate snapshot record {key}")
            records[key] = record
            processed_indices.append(unit_lookup[key])

        processed_indices.sort()
        if processed_indices:
            expected_prefix = list(range(processed_indices[-1] + 1))
            if processed_indices != expected_prefix:
                raise ValueError(
                    "Append source snapshots are not a contiguous evaluation prefix; "
                    "repair or rebuild the source run before appending"
                )
        self._append_target_reached = target_key in records

        self._append_source_manifest = manifest
        self._append_source_snapshot_records = records
        self._append_source_snapshot_keys = set(records)

    def _init_checkpoint_manager(self) -> None:
        config_hash = compute_config_hash(self.method_config, self.dataset_config)

        self._checkpoint_manager = MedMemoryBenchCheckpointManager(
            method_name=self.method_config.method_name,
            model_name=self.method_config.model.name,
            checkpoint_dir=self.output_dir / "checkpoints",
            config_hash=config_hash,
        )
        self._checkpoint_enabled = True

    def _try_resume_from_checkpoint(self) -> bool:
        if not self._checkpoint_manager or not self._checkpoint_manager.exists():
            self._log("No checkpoint found, starting from scratch")
            return False

        checkpoint = self._checkpoint_manager.load()
        if checkpoint is None:
            self._log(
                "Checkpoint and previous-good backup are unavailable or corrupted; "
                "starting from scratch",
                level="WARNING",
            )
            self._checkpoint_manager.delete()
            return False

        if self._checkpoint_manager.recovered_from_backup:
            self._log(
                "Primary checkpoint was corrupted; restored the previous-good backup.",
                level="WARNING",
            )

        config_matches = self._checkpoint_manager.validate_config()
        force_requested = bool(getattr(self, "force_resume", False))
        force_resume = self._session_level_force_resume()
        method_name = str(
            getattr(getattr(self, "method_config", None), "method_name", "")
        ).lower()
        if (
            not config_matches
            and force_requested
            and method_name.startswith("amem")
        ):
            raise ValueError(
                "Cannot force-resume an A-MEM run after memory-build configuration "
                "changes. Resume with the stored configuration or start a new run."
            )
        if not config_matches and not force_requested:
            self._log("Config changed, checkpoint invalid, starting from scratch")
            self._checkpoint_manager.delete()
            return False

        if not config_matches:
            self._log(
                "Config changed; --force is keeping the existing checkpoint and "
                "continuing with the new configuration.",
                level="WARNING",
            )
            self._checkpoint_manager.adopt_config_hash()

        if not self._checkpoint_manager.is_independent_mode():
            self._log("Checkpoint not in independent mode, cannot resume")
            return False

        resume_info = self._checkpoint_manager.get_resume_info()
        if force_resume and resume_info.get("active_session_id") is not None:
            raise RuntimeError(
                "Cannot safely force-resume because the last session may already have "
                "committed to the persistent memory backend. Resolve or rebuild that "
                "backend before resuming."
            )

        rollback = self._checkpoint_manager.rollback_incomplete_session()
        if rollback is not None:
            self._log(
                f"Rolled back unfinished session {rollback['session_id']} "
                f"from evaluation unit {rollback['unit_id']}; it will be retried.",
                level="WARNING",
            )

        self._load_checkpoint_results()

        info = self._checkpoint_manager.get_resume_info()
        if force_resume:
            self._force_resume_persona_id = info["current_persona"]
            self._log(
                "Forced session-level resume trusts the memory backend to retain all "
                "sessions recorded by the checkpoint.",
                level="WARNING",
            )
        self._log("=" * 50)
        self._log("Resuming from checkpoint")
        self._log(f"  Checkpoint ID: {info['checkpoint_id'][:8]}...")
        self._log(f"  Completed Personas: {info['completed_personas']}/{info['total_personas']}")
        self._log(f"  Completed Queries: {info['completed_queries']}/{info['total_queries']}")
        if info['current_persona'] is not None:
            resume_action = "continuing stored memory" if force_resume else "rebuilding memory"
            self._log(
                f"  Current Persona: {info['current_persona']} "
                f"({info['current_persona_completed_queries']} queries done, "
                f"{info['current_persona_injected_sessions']} sessions injected, "
                f"{resume_action})"
            )
        self._log("=" * 50)

        return True

    def _create_new_checkpoint(self) -> None:
        if not self._checkpoint_manager:
            return

        total_personas = len(self.dataset.get_persona_ids())
        total_queries = self.dataset.get_total_queries()

        self._checkpoint_manager.create(
            total_personas=total_personas,
            total_queries=total_queries,
            evaluation_mode=self.dataset_config.evaluation_mode,
        )
        self._log(f"Created checkpoint: {self._checkpoint_manager.checkpoint_path}")

    def _load_checkpoint_results(self) -> None:
        if not self._checkpoint_manager:
            return

        completed_results = self._checkpoint_manager.get_completed_results()

        for persona_id, results in completed_results.items():
            for result_dict in results:
                result = MetricResult(
                    query_id=result_dict.get("query_id", ""),
                    query_type=result_dict.get("query_type", ""),
                    score=result_dict.get("score", 0.0),
                    is_correct=result_dict.get("is_correct", False),
                    model_output=result_dict.get("model_output", ""),
                    expected_answer=result_dict.get("expected_answer", ""),
                    question=result_dict.get("question", ""),
                    details=result_dict.get("details", {}),
                )
                result.memory_construction_time = result_dict.get("memory_construction_time", 0.0)
                result.query_time = result_dict.get("query_time", 0.0)

                self.aggregator.add_result(result)
                self.result_collector.add_result(result, persona_id)

    def _init_agent_for_context(self, context_id: int, force_new: bool = True) -> None:
        if force_new or self.agent_manager is None:
            # Clean up old agent's resources before creating new one
            if self.agent_manager is not None:
                try:
                    self._log(f"[DEBUG] Starting reset of old agent...")
                    # Reset releases resources (closes Qdrant client, etc.)
                    self.agent_manager.reset()
                    self._log(f"[DEBUG] Reset completed")

                    # Explicitly delete old agent manager to release all references
                    old_manager = self.agent_manager
                    self.agent_manager = None
                    del old_manager

                    # Force garbage collection to release resources
                    import gc
                    gc.collect()

                    self._log(f"[DEBUG] Old agent deleted, sleeping 1s for resource cleanup...")
                    # Give time for resources to be fully released
                    import time
                    time.sleep(1.0)
                    self._log(f"[DEBUG] Sleep done, creating new agent...")
                except Exception as e:
                    self._log(f"Warning: Failed to reset old agent: {e}")

            self._log(f"[DEBUG] Creating AgentManager for context {context_id}...")
            self.agent_manager = AgentManager(
                method_config=self.method_config,
                dataset_config=self.dataset_config,
                batch_api=self.batch_api,
                batch_gcs_uri=self.batch_gcs_uri,
                batch_wait=self.batch_wait,
                batch_manifest_dir=self.output_dir / "batch",
                batch_config_hash=self._batch_manifest_config_hash(),
                batch_progress_callback=self._log,
            )
            self._log(f"[DEBUG] AgentManager created successfully")

        self.agent_manager.set_context_id(context_id)
        self._log(f"[DEBUG] Context ID set to {context_id}")

    def _batch_config_hash(self) -> str:
        return compute_config_hash(self.method_config, self.dataset_config)

    def _judge_batch_config_hash(self) -> str:
        """Include `.env` judge settings in judge submit/resume identity."""
        api_config = get_api_config()
        payload = {
            "method_config_hash": self._batch_manifest_config_hash(),
            "provider": api_config.get_judge_provider(),
            "model": api_config.get_judge_model(),
            "temperature": getattr(api_config, "judge_temperature", 1.0),
            "client_max_tokens": getattr(api_config, "judge_client_max_tokens", 10000),
            "max_tokens": getattr(api_config, "judge_max_tokens", 500),
            "mcd_max_tokens": getattr(api_config, "judge_mcd_max_tokens", 2000),
        }
        encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()[:48]

    def _supports_memory_snapshots(self) -> bool:
        support = getattr(self.agent_manager, "supports_memory_snapshots", None)
        return bool(callable(support) and support())

    def _memory_snapshot_root(self) -> Path:
        if getattr(self, "run_scoped_output", False):
            return self.output_dir / "memory"
        safe_method = self.method_config.method_name.replace("/", "-").replace("\\", "-")
        safe_model = self.method_config.model.name.replace("/", "-").replace("\\", "-")
        return self.output_dir / f"{safe_method}_{safe_model}" / "memory"

    def _run_scoped_experiment_dir(self) -> Path:
        memory_source_run_dir = getattr(self, "memory_source_run_dir", None)
        if memory_source_run_dir is not None:
            return memory_source_run_dir.parent
        if self.output_dir.parent.name == "query_runs":
            return self.output_dir.parent.parent.parent
        return self.output_dir.parent

    def _legacy_memory_snapshot_root(self) -> Path:
        """Return the pre-run-directory memory root for compatibility."""
        if getattr(self, "run_scoped_output", False):
            return self._run_scoped_experiment_dir() / "memory"
        return self._memory_snapshot_root()

    def _memory_snapshot_dir(self) -> Path:
        if self._memory_snapshot_run_dir is None:
            raise RuntimeError("A-MEM memory snapshot run has not been selected")
        return self._memory_snapshot_run_dir

    def _memory_snapshot_path(self, unit: EvaluationUnit) -> Path:
        return self._memory_snapshot_dir() / (
            f"persona_{unit.context_id}_unit_{unit.unit_id}.json"
        )

    def _memory_snapshot_manifest_path(self) -> Path:
        return self._memory_snapshot_dir() / "manifest.json"

    def _memory_snapshot_embeddings_path(self, unit: EvaluationUnit) -> Path:
        return self._memory_snapshot_path(unit).with_suffix(".embeddings.npy")

    @staticmethod
    def _file_sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _write_memory_snapshot_manifest(self) -> None:
        if self._memory_snapshot_manifest is None:
            return
        path = self._memory_snapshot_manifest_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_name(f"{path.name}.tmp")
        with temporary_path.open("w", encoding="utf-8") as handle:
            json.dump(
                self._memory_snapshot_manifest,
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        self._fsync_directory(path.parent)

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        """Make a completed atomic replacement durable on supported filesystems."""
        try:
            descriptor = os.open(path, os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(descriptor)
        except OSError:
            pass
        finally:
            os.close(descriptor)

    @staticmethod
    def _new_memory_run_name() -> str:
        return datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    def _memory_run_candidates(self) -> List[Path]:
        if getattr(self, "run_scoped_output", False):
            candidates = []
            experiment_dir = self._run_scoped_experiment_dir()
            for run_dir in sorted(
                [path for path in experiment_dir.iterdir() if path.is_dir()],
                key=lambda path: path.name,
                reverse=True,
            ):
                memory_dir = run_dir / "memory"
                if memory_dir != self._memory_snapshot_root():
                    candidates.append(memory_dir)

            legacy_root = self._legacy_memory_snapshot_root()
            if legacy_root.exists():
                candidates.extend(sorted(
                    [path for path in legacy_root.iterdir() if path.is_dir()],
                    key=lambda path: path.name,
                    reverse=True,
                ))
                if (legacy_root / "manifest.json").exists():
                    candidates.append(legacy_root)
            return candidates

        root = self._memory_snapshot_root()
        if not root.exists():
            return []
        candidates = sorted(
            [path for path in root.iterdir() if path.is_dir()],
            key=lambda path: path.name,
            reverse=True,
        )
        # Runs produced before timestamped directories stored their manifest
        # and unit snapshots directly in the memory root.
        if (root / "manifest.json").exists():
            candidates.append(root)
        return candidates

    def _saved_memory_source_candidate(self) -> Optional[Path]:
        if not getattr(self, "run_scoped_output", False):
            return None
        source_path = self.output_dir / "memory_source.json"
        if not source_path.exists():
            return None
        try:
            payload = json.loads(source_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Cannot read saved memory source {source_path}: {exc}") from exc
        manifest_value = payload.get("manifest_path")
        if not isinstance(manifest_value, str) or not manifest_value:
            raise ValueError(f"Saved memory source is invalid: {source_path}")
        manifest_path = Path(manifest_value)
        if not manifest_path.is_absolute():
            manifest_path = self.output_dir / manifest_path
        return manifest_path.resolve().parent

    def _explicit_memory_run_candidates(self, requested_run: str) -> List[Path]:
        if requested_run == "legacy":
            return [self._legacy_memory_snapshot_root()]
        if Path(requested_run).name != requested_run:
            raise ValueError("--memory-run must be a run timestamp directory name")
        if getattr(self, "run_scoped_output", False):
            candidates = []
            memory_source_run_dir = getattr(self, "memory_source_run_dir", None)
            if memory_source_run_dir is not None:
                candidates.append(memory_source_run_dir / "memory")
            candidates.extend([
                self._run_scoped_experiment_dir() / requested_run / "memory",
                self._legacy_memory_snapshot_root() / requested_run,
            ])
            return list(dict.fromkeys(candidates))
        return [self._memory_snapshot_root() / requested_run]

    def _write_memory_source(self, run_dir: Path, manifest: Dict[str, Any]) -> None:
        if not getattr(self, "run_scoped_output", False):
            return
        manifest_path = run_dir / "manifest.json"
        relative_manifest = os.path.relpath(manifest_path, self.output_dir)
        source_run_id = (
            run_dir.parent.name
            if (
                run_dir.name == "memory"
                and run_dir.parent.parent == self._run_scoped_experiment_dir()
            )
            else f"legacy:{run_dir.name}"
        )
        payload = {
            "format": "medmemorybench.memory_source",
            "version": 1,
            "selected_at": datetime.now().isoformat(),
            "selection": (
                "explicit"
                if getattr(
                    self,
                    "memory_run_explicit",
                    bool(getattr(self, "memory_run", None)),
                )
                else "newest_compatible"
            ),
            "source_run_id": source_run_id,
            "manifest_path": relative_manifest,
            "build_id": manifest.get("build_id"),
            "config_hash": manifest.get("config_hash"),
            "build_config_hash": derive_legacy_build_config_hash(
                manifest, manifest_path
            ),
            "feature_configuration": manifest.get("feature_configuration", {}),
            "build_metrics": manifest.get("build_metrics", {}),
            "memory_size": manifest.get("memory_size", {}),
            "status": manifest.get("status"),
        }
        path = self.output_dir / "memory_source.json"
        temporary_path = path.with_suffix(".tmp")
        temporary_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        temporary_path.replace(path)

    def _select_memory_snapshot_run(
        self,
        *,
        require_complete: bool,
        resume_current: bool = False,
    ) -> Path:
        requested_run = getattr(self, "memory_run", None)
        saved_source = (
            self._saved_memory_source_candidate()
            if require_complete and not requested_run
            else None
        )
        if resume_current and getattr(self, "run_scoped_output", False):
            candidates = [self._memory_snapshot_root()]
        elif saved_source is not None:
            candidates = [saved_source]
        elif requested_run:
            candidates = self._explicit_memory_run_candidates(requested_run)
        else:
            candidates = self._memory_run_candidates()

        for run_dir in candidates:
            manifest_path = run_dir / "manifest.json"
            if not manifest_path.exists():
                continue
            try:
                with manifest_path.open("r", encoding="utf-8") as handle:
                    manifest = json.load(handle)
            except (OSError, json.JSONDecodeError):
                continue
            if (
                manifest.get("format") != "medmemorybench.memory_manifest"
                or manifest.get("version") != 1
                or not is_manifest_build_compatible(
                    manifest, self._batch_config_hash(), manifest_path
                )
                or manifest.get("method_name") != self.method_config.method_name
                or (require_complete and manifest.get("status") != "complete")
                or (not require_complete and manifest.get("status") != "building")
            ):
                continue
            self._memory_snapshot_run_dir = run_dir
            self._memory_snapshot_manifest = manifest
            if require_complete:
                self._write_memory_source(run_dir, manifest)
            return run_dir

        selection = f" '{requested_run}'" if requested_run else ""
        status = "complete" if require_complete else "in-progress"
        raise FileNotFoundError(
            f"No compatible {status} A-MEM memory run{selection} for experiment "
            f"{self._run_scoped_experiment_dir() if getattr(self, 'run_scoped_output', False) else self._memory_snapshot_root()}"
        )

    def _copy_append_source_snapshots(self) -> None:
        """Materialize source snapshots into the derived append run."""
        if self._append_source_manifest is None or self.memory_source_run_dir is None:
            return
        units = self._get_evaluation_units()
        unit_lookup = {(unit.context_id, unit.unit_id): unit for unit in units}
        source_memory_dir = self.memory_source_run_dir / "memory"
        snapshots = self._memory_snapshot_manifest["snapshots"]
        for key, record in sorted(
            self._append_source_snapshot_records.items(),
            key=lambda item: unit_lookup[item[0]].unit_id,
        ):
            unit = unit_lookup[key]
            source_name = record.get("path")
            if not isinstance(source_name, str) or Path(source_name).name != source_name:
                raise ValueError(f"Append source snapshot path is invalid for unit {key}")
            source_path = source_memory_dir / source_name
            source_payload = self._read_snapshot_payload(
                source_path,
                self._append_source_manifest,
                unit,
                require_current_config=True,
            )
            child_payload = copy.deepcopy(source_payload)
            child_payload["build_id"] = self._memory_snapshot_manifest["build_id"]
            child_payload["append_source"] = {
                "run_id": self.memory_source_run_dir.name,
                "build_id": self._append_source_manifest.get("build_id"),
            }
            target_path = self._memory_snapshot_path(unit)
            child_payload = self._write_snapshot_payload(child_payload, target_path)
            snapshots[:] = [
                item
                for item in snapshots
                if not (
                    item.get("context_id") == unit.context_id
                    and item.get("unit_id") == unit.unit_id
                )
            ]
            snapshots.append({
                "unit_id": unit.unit_id,
                "context_id": unit.context_id,
                "path": target_path.name,
                "source": self.memory_source_run_dir.name,
                "session_count": len(child_payload.get("session_ids", [])),
                "memory_build_time": child_payload.get("memory_build_time", 0.0),
                "memory_build_metrics": copy.deepcopy(
                    child_payload.get("memory_build_metrics") or {}
                ),
                "memory_size": copy.deepcopy(child_payload.get("memory_size")),
            })
        snapshots.sort(key=lambda item: item["unit_id"])
        self._write_memory_snapshot_manifest()

    def _start_append_memory_snapshot_manifest(self) -> None:
        """Create a derived memory run seeded with the source prefix."""
        self._prepare_append_plan()
        units = self._get_evaluation_units()
        if self._append_target_index is None:
            raise RuntimeError("Append target was not resolved")
        selected_units = units[: self._append_target_index + 1]
        run_name = self.output_dir.name if self.run_scoped_output else self._new_memory_run_name()
        run_dir = self._memory_snapshot_root()
        run_dir.mkdir(parents=True, exist_ok=False)
        self._memory_snapshot_run_dir = run_dir
        source_manifest = self._append_source_manifest or {}
        self._memory_snapshot_manifest = {
            "format": "medmemorybench.memory_manifest",
            "version": 1,
            "build_id": str(uuid.uuid4()),
            "run_name": run_name,
            "status": "building",
            "method_name": self.method_config.method_name,
            "model_name": self.method_config.model.name,
            "config_hash": self._batch_config_hash(),
            "build_config_hash": self._batch_config_hash(),
            "retrieval_config_hash": compute_query_config_hash(
                self.method_config, self.dataset_config
            ),
            "feature_configuration": self._amem_feature_configuration(),
            "created_at": datetime.now().isoformat(),
            "completed_at": None,
            "unit_ids": [unit.unit_id for unit in selected_units],
            "dataset_unit_ids": [unit.unit_id for unit in units],
            "append": {
                "source_run_id": self.memory_source_run_dir.name,
                "source_build_id": source_manifest.get("build_id"),
                "target_persona": getattr(self, "append_persona", None),
                "target_unit": getattr(self, "append_unit", None),
                "target_index": self._append_target_index,
                "source_snapshot_count": len(self._append_source_snapshot_records),
            },
            "snapshots": [],
        }
        self._write_memory_snapshot_manifest()
        self._copy_append_source_snapshots()
        self._log(f"Created append A-MEM memory run: {run_dir}")

    def _start_memory_snapshot_manifest(self, *, resume_existing: bool = False) -> None:
        units = self._get_evaluation_units()
        # A process can stop after publishing the memory manifest but before the
        # checkpoint is created. Reuse that current run on the next --resume.
        if (
            not resume_existing
            and getattr(self, "resume", False)
            and getattr(self, "run_scoped_output", False)
            and (self._memory_snapshot_root() / "manifest.json").exists()
        ):
            resume_existing = True
        if resume_existing:
            run_dir = self._select_memory_snapshot_run(
                require_complete=False,
                resume_current=True,
            )
            expected_unit_ids = [unit.unit_id for unit in units]
            stored_unit_ids = self._memory_snapshot_manifest.get("unit_ids", [])
            if getattr(self, "append", False):
                if not self._memory_snapshot_manifest.get("append"):
                    raise ValueError(
                        f"A-MEM resume target is not an append run: {run_dir}"
                    )
                if self._append_target_index is None:
                    self._prepare_append_plan()
                if stored_unit_ids != expected_unit_ids[: self._append_target_index + 1]:
                    raise ValueError(
                        f"A-MEM append run does not match current evaluation units: {run_dir}"
                    )
                # Reconcile source snapshots after an interruption during the
                # initial copy, before building any missing evaluation units.
                self._copy_append_source_snapshots()
            elif stored_unit_ids != expected_unit_ids:
                raise ValueError(
                    f"A-MEM memory run does not match current evaluation units: {run_dir}"
                )
            run_label = run_dir.name if run_dir != self._memory_snapshot_root() else "legacy"
            self._log(f"Resuming A-MEM memory run: {run_label}")
            return

        if getattr(self, "append", False):
            self._start_append_memory_snapshot_manifest()
            return

        if getattr(self, "run_scoped_output", False):
            run_name = self.output_dir.name
            run_dir = self._memory_snapshot_root()
        else:
            run_name = self._new_memory_run_name()
            run_dir = self._memory_snapshot_root() / run_name
        run_dir.mkdir(parents=True, exist_ok=False)
        self._memory_snapshot_run_dir = run_dir
        self._memory_snapshot_manifest = {
            "format": "medmemorybench.memory_manifest",
            "version": 1,
            "build_id": str(uuid.uuid4()),
            "run_name": run_name,
            "status": "building",
            "method_name": self.method_config.method_name,
            "model_name": self.method_config.model.name,
            "config_hash": self._batch_config_hash(),
            "build_config_hash": self._batch_config_hash(),
            "retrieval_config_hash": compute_query_config_hash(
                self.method_config, self.dataset_config
            ),
            "feature_configuration": self._amem_feature_configuration(),
            "created_at": datetime.now().isoformat(),
            "completed_at": None,
            "unit_ids": [unit.unit_id for unit in units],
            "dataset_unit_ids": [unit.unit_id for unit in units],
            "snapshots": [],
        }
        self._write_memory_snapshot_manifest()
        self._log(f"Created A-MEM memory run: {run_dir}")

    def _load_memory_snapshot_manifest(self) -> None:
        run_dir = self._select_memory_snapshot_run(require_complete=True)
        manifest = self._memory_snapshot_manifest
        expected_unit_ids = [unit.unit_id for unit in self._get_evaluation_units()]
        stored_unit_ids = manifest.get("unit_ids", [])
        if manifest.get("append"):
            if not set(stored_unit_ids).issubset(set(expected_unit_ids)):
                raise ValueError(
                    f"A-MEM append manifest contains unknown evaluation units: {run_dir}"
                )
        elif stored_unit_ids != expected_unit_ids:
            raise ValueError(
                f"A-MEM memory manifest does not match current evaluation units: {run_dir}"
            )
        snapshot_unit_ids = sorted(item.get("unit_id") for item in manifest.get("snapshots", []))
        if snapshot_unit_ids != sorted(stored_unit_ids):
            raise ValueError(f"A-MEM memory manifest snapshot list is incomplete: {run_dir}")
        self._memory_unit_ids = set(stored_unit_ids)
        run_label = run_dir.name if run_dir != self._memory_snapshot_root() else "legacy"
        self._log(f"Using A-MEM memory run: {run_label}")

    def _complete_memory_snapshot_manifest(self) -> None:
        if self._memory_snapshot_manifest is None:
            return
        expected = self._memory_snapshot_manifest.get("unit_ids", [])
        completed = sorted(item["unit_id"] for item in self._memory_snapshot_manifest["snapshots"])
        expected = sorted(expected)
        if completed != expected:
            raise RuntimeError(
                f"A-MEM snapshot build is incomplete: expected units {expected}, got {completed}"
            )
        snapshot_logs = [
            {
                "unit_id": item.get("unit_id"),
                "context_id": item.get("context_id"),
                "session_count": item.get("session_count", 0),
                "total_time": item.get("memory_build_time", 0.0),
                "build_metrics": item.get("memory_build_metrics") or {},
                "memory_size": item.get("memory_size"),
                "restored_from_snapshot": bool(item.get("source")),
            }
            for item in self._memory_snapshot_manifest.get("snapshots", [])
        ]
        self._memory_snapshot_manifest["build_metrics"] = self._build_metrics_report(
            snapshot_logs
        )
        self._memory_snapshot_manifest["memory_size"] = copy.deepcopy(
            self._memory_snapshot_manifest["build_metrics"]["memory_size"]
        )
        self._memory_snapshot_manifest["status"] = "complete"
        self._memory_snapshot_manifest["completed_at"] = datetime.now().isoformat()
        self._write_memory_snapshot_manifest()

    @staticmethod
    def _snapshot_integrity_hash(payload: Dict[str, Any]) -> str:
        hash_payload = dict(payload)
        hash_payload.pop("integrity_hash", None)
        content = json.dumps(
            hash_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    @staticmethod
    def _measure_snapshot_memory_size(
        payload: Dict[str, Any],
        snapshot_path: Path,
    ) -> Dict[str, Any]:
        """Measure the serialized retrieval state without snapshot metadata."""
        memory_state = payload.get("memory_state") or {}
        json_bytes = len(json.dumps(
            memory_state,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8"))
        system_state = memory_state.get("system_state") or {}
        embedding_state = (system_state.get("retriever") or {}).get("embeddings")
        embedding_bytes = 0
        if isinstance(embedding_state, dict):
            embedding_name = embedding_state.get("path")
            if (
                isinstance(embedding_name, str)
                and Path(embedding_name).name == embedding_name
            ):
                embedding_path = snapshot_path.parent / embedding_name
                if embedding_path.is_file():
                    embedding_bytes = embedding_path.stat().st_size
        memory_entries = system_state.get("memories") or []
        memory_chunks = (memory_state.get("agent_state") or {}).get(
            "memory_chunks"
        ) or []
        total_bytes = json_bytes + embedding_bytes
        return {
            "measurement": "serialized_memory_state",
            "bytes": total_bytes,
            "mib": round(total_bytes / (1024 ** 2), 6),
            "json_bytes": json_bytes,
            "embedding_bytes": embedding_bytes,
            "memory_entry_count": len(memory_entries),
            "memory_chunk_count": len(memory_chunks),
        }

    def _write_snapshot_payload(
        self,
        payload: Dict[str, Any],
        path: Path,
    ) -> Dict[str, Any]:
        """Publish a snapshot without invalidating the previous generation."""
        payload = copy.deepcopy(payload)
        embedding_state = (
            payload.get("memory_state", {})
            .get("system_state", {})
            .get("retriever", {})
            .get("embeddings")
        )
        published_embedding_path: Optional[Path] = None
        if embedding_state is not None:
            embedding_values = np.asarray(embedding_state.pop("values"))
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary_embedding_path = path.parent / (
                f".{path.stem}.embeddings.{uuid.uuid4().hex}.tmp"
            )
            with temporary_embedding_path.open("wb") as handle:
                np.save(handle, embedding_values, allow_pickle=False)
                handle.flush()
                os.fsync(handle.fileno())
            embedding_sha256 = self._file_sha256(temporary_embedding_path)
            published_embedding_path = path.parent / (
                f"{path.stem}.embeddings.{embedding_sha256}.npy"
            )
            os.replace(temporary_embedding_path, published_embedding_path)
            self._fsync_directory(path.parent)
            embedding_state.update({
                "storage": "npy",
                "path": published_embedding_path.name,
                "sha256": embedding_sha256,
            })

        payload["memory_size"] = self._measure_snapshot_memory_size(payload, path)
        payload["integrity_hash"] = self._snapshot_integrity_hash(payload)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_name(f"{path.name}.tmp")
        with temporary_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        self._fsync_directory(path.parent)

        referenced_embedding = (
            published_embedding_path.name
            if published_embedding_path is not None
            else None
        )
        embedding_candidates = list(
            path.parent.glob(f"{path.stem}.embeddings.*.npy")
        )
        embedding_candidates.append(path.with_suffix(".embeddings.npy"))
        for candidate in embedding_candidates:
            if candidate.name != referenced_embedding:
                candidate.unlink(missing_ok=True)
        return payload

    def _read_snapshot_payload(
        self,
        path: Path,
        manifest: Dict[str, Any],
        unit: EvaluationUnit,
        *,
        require_current_config: bool = True,
    ) -> Dict[str, Any]:
        """Read and validate a snapshot from any memory-run directory."""
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Cannot read A-MEM snapshot {path}: {exc}") from exc
        if (
            payload.get("format") != "medmemorybench.memory_snapshot"
            or payload.get("version") != 1
            or payload.get("context_id") != unit.context_id
            or payload.get("unit_id") != unit.unit_id
            or payload.get("build_id") != manifest.get("build_id")
        ):
            raise ValueError(f"A-MEM snapshot metadata does not match unit {unit.unit_id}: {path}")
        expected_hash = self._snapshot_integrity_hash(payload)
        if payload.get("integrity_hash") != expected_hash:
            raise ValueError(f"A-MEM snapshot integrity check failed: {path}")
        if (
            require_current_config
            and not is_manifest_build_compatible(
                manifest,
                self._batch_config_hash(),
                path.parent / "manifest.json",
            )
        ):
            raise ValueError(
                f"A-MEM snapshot build configuration does not match: {path}"
            )
        payload_build_hash = str(payload.get("build_config_hash") or "")
        if (
            require_current_config
            and payload_build_hash
            and payload_build_hash != self._batch_config_hash()
        ):
            raise ValueError(
                f"A-MEM snapshot payload build configuration does not match: {path}"
            )
        payload["memory_size"] = self._measure_snapshot_memory_size(payload, path)
        embedding_state = (
            payload.get("memory_state", {})
            .get("system_state", {})
            .get("retriever", {})
            .get("embeddings")
        )
        if embedding_state is not None:
            if embedding_state.get("storage") != "npy":
                raise ValueError(f"A-MEM embedding snapshot metadata is invalid: {path}")
            embedding_path = path.parent / embedding_state["path"]
            if (
                not embedding_path.exists()
                or self._file_sha256(embedding_path) != embedding_state.get("sha256")
            ):
                raise ValueError(f"A-MEM embedding snapshot integrity check failed: {embedding_path}")
            embedding_values = np.load(embedding_path, allow_pickle=False)
            if (
                str(embedding_values.dtype) != embedding_state.get("dtype")
                or list(embedding_values.shape) != embedding_state.get("shape")
            ):
                raise ValueError(f"A-MEM embedding snapshot shape or dtype is invalid: {embedding_path}")
            embedding_state["values"] = embedding_values
        return payload

    def _write_memory_snapshot(
        self,
        unit: EvaluationUnit,
        memory_build_time: float = 0.0,
        memory_build_metrics: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Persist the exact post-build state, then atomically publish it."""
        path = self._memory_snapshot_path(unit)
        state = self.agent_manager.export_memory_state(context_id=unit.context_id)
        payload = {
            "format": "medmemorybench.memory_snapshot",
            "version": 1,
            "method_name": self.method_config.method_name,
            "model_name": self.method_config.model.name,
            "config_hash": self._batch_config_hash(),
            "build_config_hash": self._batch_config_hash(),
            "retrieval_config_hash": compute_query_config_hash(
                self.method_config, self.dataset_config
            ),
            "build_id": self._memory_snapshot_manifest["build_id"],
            "context_id": unit.context_id,
            "unit_id": unit.unit_id,
            "evaluation_session_id": unit.metadata.get("eval_session_id"),
            "session_ids": [session.session_id for session in unit.sessions_to_inject],
            "memory_build_time": memory_build_time,
            "memory_build_metrics": copy.deepcopy(memory_build_metrics or {}),
            "feature_configuration": self._amem_feature_configuration(),
            "created_at": datetime.now().isoformat(),
            "memory_state": state,
        }
        payload = self._write_snapshot_payload(payload, path)
        memory_size = copy.deepcopy(payload.get("memory_size"))
        snapshot_record = {
            "unit_id": unit.unit_id,
            "context_id": unit.context_id,
            "path": path.name,
            "session_count": len(unit.sessions_to_inject),
            "memory_build_time": memory_build_time,
            "memory_build_metrics": copy.deepcopy(memory_build_metrics or {}),
            "memory_size": memory_size,
        }
        snapshots = self._memory_snapshot_manifest["snapshots"]
        snapshots[:] = [item for item in snapshots if item["unit_id"] != unit.unit_id]
        snapshots.append(snapshot_record)
        snapshots.sort(key=lambda item: item["unit_id"])
        for log in reversed(getattr(self, "_memory_build_logs", [])):
            if (
                log.get("unit_id") == unit.unit_id
                and log.get("context_id") == unit.context_id
            ):
                log["memory_size"] = copy.deepcopy(memory_size)
                break
        self._write_memory_snapshot_manifest()
        return path

    def _read_memory_snapshot(
        self,
        unit: EvaluationUnit,
        *,
        require_current_config: bool = True,
    ) -> Optional[Dict[str, Any]]:
        path = self._memory_snapshot_path(unit)
        if not path.exists():
            return None
        if self._memory_snapshot_manifest is None:
            raise RuntimeError("A-MEM memory snapshot manifest has not been loaded")
        payload = self._read_snapshot_payload(
            path,
            self._memory_snapshot_manifest,
            unit,
            require_current_config=require_current_config,
        )
        return payload

    def _restore_memory_snapshot(self, unit: EvaluationUnit) -> Path:
        """Reload the just-written state before retrieval starts."""
        payload = self._read_memory_snapshot(unit)
        if payload is None:
            raise RuntimeError(f"A-MEM snapshot is unavailable for unit {unit.unit_id}")
        self.agent_manager.import_memory_state(
            payload["memory_state"],
            context_id=unit.context_id,
        )
        return self._memory_snapshot_path(unit)

    def _save_and_restore_unit_memory(
        self,
        unit: EvaluationUnit,
        memory_build_time: float = 0.0,
        memory_build_metrics: Optional[Dict[str, Any]] = None,
    ) -> Optional[Path]:
        if self.dry_run or not self._supports_memory_snapshots():
            return None
        self._log("    --- Memory Snapshot Start ---")
        path = self._write_memory_snapshot(
            unit,
            memory_build_time=memory_build_time,
            memory_build_metrics=memory_build_metrics,
        )
        self._restore_memory_snapshot(unit)
        self._log(f"      Saved and restored retrieval state: {path}")
        self._log("    --- Memory Snapshot Done ---")
        return path

    def _restore_completed_unit_for_resume(
        self,
        unit: EvaluationUnit,
    ) -> Optional[Dict[str, Any]]:
        if (
            not (
                getattr(self, "resume", False)
                or getattr(self, "force_resume", False)
            )
            or self.dry_run
            or not self._supports_memory_snapshots()
        ):
            return None
        payload = self._read_memory_snapshot(unit)
        if payload is None:
            return None
        self.agent_manager.import_memory_state(
            payload["memory_state"],
            context_id=unit.context_id,
        )
        self._log(
            f"      [Resume] Restored completed unit state from "
            f"{self._memory_snapshot_path(unit)}"
        )
        return payload

    def _batch_manifest_config_hash(self) -> str:
        """Namespace manifests by checkpoint so only --resume reuses a run."""
        config_hash = self._batch_config_hash()
        checkpoint_manager = getattr(self, "_checkpoint_manager", None)
        scope = (
            checkpoint_manager.get_batch_manifest_scope()
            if checkpoint_manager
            else ""
        )
        # Older checkpoints have no scope and retain their legacy manifest path.
        return f"{config_hash}{scope.replace('-', '')}" if scope else config_hash

    def _batch_manifest_path(self, stem: str, model: str) -> Path:
        """Keep manifests isolated when different methods share one output directory."""
        return scoped_manifest_path(
            self.output_dir / "batch",
            stem,
            model=model,
            config_hash=self._batch_manifest_config_hash(),
        )

    def _deferred_judge_state_paths(self) -> List[Path]:
        batch_dir = self.output_dir / "batch"
        return sorted(batch_dir.glob(
            f"medmemorybench_deferred_judges-*-{self._judge_batch_config_hash()}.json"
        ))

    def _save_deferred_judges(self, judge_client) -> None:
        """Persist judge inputs before submitting so resume never regenerates them."""
        path = scoped_manifest_path(
            self.output_dir / "batch",
            "medmemorybench_deferred_judges",
            model=judge_client.model,
            config_hash=self._judge_batch_config_hash(),
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "config_hash": self._judge_batch_config_hash(),
            "judge_model": judge_client.model,
            "items": self._deferred_judges,
        }
        temporary_path = path.with_suffix(".tmp")
        with temporary_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        temporary_path.replace(path)

    def _load_deferred_judges(self) -> None:
        """Restore submitted-or-ready judge work before rebuilding evaluation units."""
        paths = self._deferred_judge_state_paths()
        if not paths:
            return
        if len(paths) != 1:
            raise VertexBatchError(
                "Found multiple deferred-judge state files for this configuration. "
                "Use a separate output directory to resume safely."
            )
        path = paths[0]
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise VertexBatchError(f"Cannot read deferred judge state {path}: {exc}") from exc
        if (
            payload.get("version") != 1
            or payload.get("config_hash") != self._judge_batch_config_hash()
        ):
            raise VertexBatchError(f"Deferred judge state does not match this evaluator: {path}")
        items = payload.get("items")
        if not isinstance(items, list):
            raise VertexBatchError(f"Deferred judge state is invalid: {path}")
        self._deferred_judges = items
        self._log(f"[Vertex Batch] Restored {len(items):,} deferred LLM-judge request(s) from {path}.")

    def _clear_deferred_judges(self) -> None:
        for path in self._deferred_judge_state_paths():
            path.unlink(missing_ok=True)
        self._deferred_judges = []

    def _is_deferred_judge_query(self, query_id: str) -> bool:
        return any(item.get("query_id") == query_id for item in self._deferred_judges)

    def evaluate(self) -> EvaluationReport:
        start_time = datetime.now()

        get_usage_tracker().reset()

        self._init_dataset()

        if self.execution_stage != "all" and not self.method_config.method_name.lower().startswith("amem"):
            raise ValueError(
                "Separated memory/query execution is currently implemented for AMem methods only"
            )
        if getattr(self, "append", False):
            if not self.method_config.method_name.lower().startswith("amem"):
                raise ValueError("--append is currently implemented for AMem methods only")
            self._prepare_append_plan()

        resumed = False
        if self.resume and self._checkpoint_enabled:
            resumed = self._try_resume_from_checkpoint()

        if self.execution_stage == "query":
            self._load_memory_snapshot_manifest()
        elif not self.dry_run and self.method_config.method_name.lower().startswith("amem"):
            self._start_memory_snapshot_manifest(resume_existing=resumed)

        # Deferred judge inputs belong to the loaded checkpoint namespace.
        # Do not attach a fresh run to an older namespace when no checkpoint
        # was successfully resumed.
        if resumed and self.batch_api and not self.dry_run:
            self._load_deferred_judges()

        # Query-only runs still submit batch work, so they need the same unique
        # checkpoint-backed manifest scope as full runs. Without this, a fresh
        # query invocation falls back to the config-only legacy manifest name.
        if not resumed and self._checkpoint_enabled:
            self._create_new_checkpoint()

        self._run_evaluation_loop()

        if (
            self.execution_stage in {"all", "memory"}
            and not self.dry_run
            and self.method_config.method_name.lower().startswith("amem")
        ):
            self._complete_memory_snapshot_manifest()

        if self._checkpoint_manager and self.execution_stage in {"all", "query"}:
            self._checkpoint_manager.mark_completed()
            self._checkpoint_manager.delete()
            self._log("Evaluation completed, checkpoint deleted")
        elif self._checkpoint_manager and self.execution_stage == "memory":
            self._checkpoint_manager.delete()
            self._log("Memory build completed, checkpoint deleted")

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        report = self._generate_report(start_time, end_time, duration)

        return report

    def _run_evaluation_loop(self) -> None:
        current_context_id = None

        units = self._get_evaluation_units()
        if getattr(self, "append", False) and self._append_target_index is not None:
            units = units[: self._append_target_index + 1]
        elif self.execution_stage == "query" and self._memory_unit_ids is not None:
            units = [unit for unit in units if unit.unit_id in self._memory_unit_ids]

        for unit in units:
            persona_id = unit.context_id

            if self._checkpoint_manager and self._checkpoint_manager.is_persona_completed(persona_id):
                self._log(f"Skipping completed Persona: {persona_id}")
                continue

            if persona_id != current_context_id:
                if current_context_id is not None and self._checkpoint_manager and not self.batch_api:
                    self._checkpoint_manager.complete_persona(current_context_id)

                if current_context_id is not None:
                    self._log(f"\nSwitching Persona: {current_context_id} -> {persona_id}")

                if not self.dry_run:
                    self._init_agent_for_context(
                        context_id=persona_id,
                        force_new=(self.dataset_config.evaluation_mode == "independent")
                    )

                current_context_id = persona_id

                if self._checkpoint_manager:
                    preserve_progress = (
                        self._session_level_force_resume()
                        and persona_id == getattr(self, "_force_resume_persona_id", None)
                    )
                    self._checkpoint_manager.start_persona(
                        persona_id,
                        preserve_progress=preserve_progress,
                    )

            if getattr(self, "execution_stage", "all") == "query":
                snapshot_payload = self._read_memory_snapshot(unit)
                if snapshot_payload is None:
                    raise FileNotFoundError(
                        f"No compatible A-MEM snapshot for evaluation unit {unit.unit_id}: "
                        f"{self._memory_snapshot_path(unit)}"
                    )
                if not self._supports_memory_snapshots():
                    raise RuntimeError(
                        f"Method {self.method_config.method_name} does not support query-only snapshots"
                    )
                self.agent_manager.import_memory_state(
                    snapshot_payload["memory_state"],
                    context_id=unit.context_id,
                )
                self._record_source_build_metrics(unit, snapshot_payload)
                self._log(
                    f"\n  Evaluation Unit {unit.unit_id}: restored query-stage memory "
                    f"from {self._memory_snapshot_path(unit)}"
                )
                unit_results = self._evaluate_unit_queries(
                    unit,
                    total_memory_time=float(snapshot_payload.get("memory_build_time", 0.0)),
                )
            elif getattr(self, "append", False):
                unit_results = self._evaluate_append_unit(unit)
            else:
                unit_results = self._evaluate_unit_with_checkpoint(unit)

            for result in unit_results:
                self.aggregator.add_result(result)
                self.result_collector.add_result(result, persona_id)

        if getattr(self, "execution_stage", "all") == "memory":
            return

        for item in self._complete_combined_batch_queries():
            result = item["result"]
            persona_id = item["persona_id"]
            self.aggregator.add_result(result)
            self.result_collector.add_result(result, persona_id)
            if self._checkpoint_manager:
                self._checkpoint_manager.mark_query_completed(
                    result.query_id,
                    result.to_dict(),
                    persona_id=persona_id,
                )

        for item in self._complete_deferred_judges():
            result = item["result"]
            persona_id = item["persona_id"]
            self.aggregator.add_result(result)
            self.result_collector.add_result(result, persona_id)
            if self._checkpoint_manager:
                self._checkpoint_manager.mark_query_completed(
                    result.query_id,
                    result.to_dict(),
                    persona_id=persona_id,
                )

        if self._checkpoint_manager and self.batch_api:
            for persona_id in self.dataset.get_persona_ids():
                self._checkpoint_manager.complete_persona(persona_id)
        elif current_context_id is not None and self._checkpoint_manager:
            self._checkpoint_manager.complete_persona(current_context_id)

    def _evaluate_append_unit(self, unit: EvaluationUnit) -> List[MetricResult]:
        """Reuse a carried-forward snapshot or build the next missing unit."""
        snapshot_payload = self._read_memory_snapshot(unit)
        if snapshot_payload is not None:
            if not self._supports_memory_snapshots():
                raise RuntimeError(
                    f"Method {self.method_config.method_name} does not support append snapshots"
                )
            self.agent_manager.import_memory_state(
                snapshot_payload["memory_state"],
                context_id=unit.context_id,
            )
            self._record_source_build_metrics(unit, snapshot_payload)
            self._log(
                f"\n  Evaluation Unit {unit.unit_id}: restored append memory from "
                f"{self._memory_snapshot_path(unit)}"
            )
            if self.execution_stage == "memory":
                return []
            return self._evaluate_unit_queries(
                unit,
                total_memory_time=float(snapshot_payload.get("memory_build_time", 0.0)),
            )
        return self._evaluate_unit_with_checkpoint(unit)

    def _evaluate_unit_with_checkpoint(self, unit: EvaluationUnit) -> List[MetricResult]:
        """Evaluate unit with checkpoint support."""
        self._log(f"\n  Evaluation Unit {unit.unit_id}:")
        self._log(f"    Persona: {unit.context_id}")
        self._log(f"    Sessions to inject: {len(unit.sessions_to_inject)}")
        self._log(f"    Queries to evaluate: {len(unit.queries_to_evaluate)}")

        results = []

        self._log(f"    --- Memory Build Start ---")

        memory_build_failed = False
        total_memory_time = 0.0
        session_build_results = []  # Store per-session build results
        restored_payload = self._restore_completed_unit_for_resume(unit)
        restored_completed_unit = restored_payload is not None
        if restored_payload is not None:
            total_memory_time = float(restored_payload.get("memory_build_time", 0.0))

        if self.dry_run:
            self._log(f"    [Dry Run] Skipping memory build")
        else:
            total_sessions = len(unit.sessions_to_inject)
            session_ids = []

            for idx, session in enumerate(unit.sessions_to_inject):
                session_ids.append(session.session_id)
                if restored_completed_unit:
                    self._log(
                        f"      [Resume] Session {session.session_id} restored from snapshot; skipping"
                    )
                    continue
                if (
                    self._session_level_force_resume()
                    and not self._supports_memory_snapshots()
                    and self._checkpoint_manager
                    and self._checkpoint_manager.is_session_injected(
                        session.session_id,
                        persona_id=unit.context_id,
                    )
                ):
                    self._log(
                        f"      [Resume] Session {session.session_id} already injected; skipping"
                    )
                    continue
                session_succeeded = False
                memory_text = session.to_memory_text()

                # Show progress
                self._log(f"      [Progress] Session {idx + 1}/{total_sessions} (ID: {session.session_id})")

                formatted_text = self.prompt_manager.format_memorize(
                    context=memory_text,
                    timestamp=None,
                )

                is_last_session = (idx == total_sessions - 1)

                mark_session_started = getattr(
                    self._checkpoint_manager,
                    "mark_session_started",
                    None,
                )
                if callable(mark_session_started):
                    mark_session_started(session.session_id, unit.unit_id)

                usage_tracker = get_usage_tracker()
                usage_before = usage_tracker.get_stats()
                session_started_at = time.perf_counter()
                session_build_record: Dict[str, Any] = {
                    "session_id": session.session_id,
                    "session_index": idx,
                }
                try:
                    experimental_source = {}
                    method_name = getattr(
                        getattr(self, "method_config", None),
                        "method_name",
                        "",
                    )
                    if method_name == "amem_test":
                        experimental_source = {
                            "source_session_id": session.session_id,
                            "source_session_index": idx,
                            "source_event_id": session.metadata.get("event_id"),
                        }
                    memory_result = self._run_api_call(
                        self.agent_manager.send_message,
                        message=formatted_text,
                        memorizing=True,
                        context_id=unit.context_id,
                        is_last_session=is_last_session,
                        timestamp=session.timestamp,
                        memory_items=session.metadata.get("messages", []),
                        **experimental_source,
                    )

                    if memory_result is not None and isinstance(memory_result, MemoryBuildResult):
                        session_succeeded = True
                        total_memory_time += memory_result.time_cost

                        # Log brief progress info
                        passages_count = len(memory_result.all_passages) if memory_result.all_passages else memory_result.extra.get("inserted_count", 0)
                        self._log(f"        → Stored {passages_count} passages, time={memory_result.time_cost:.2f}s")

                        # Show extraction preview (first 100 chars)
                        if memory_result.extraction_result:
                            preview = memory_result.extraction_result[:150].replace('\n', ' ')
                            self._log(f"        → Extraction: {preview}...")

                        session_build_record["build_result"] = memory_result.to_dict()

                except LLMAPIError as e:
                    session_build_record["error"] = str(e)
                    self._log(
                        f"        [API ERROR] Session {session.session_id} failed: {e}",
                        level="ERROR"
                    )
                    memory_build_failed = True
                    self._record_api_failure(
                        "memory_build",
                        e,
                        unit_id=unit.unit_id,
                        context_id=unit.context_id,
                        session_id=session.session_id,
                        session_index=idx,
                        affected_query_ids=[
                            query.query_id for query in unit.queries_to_evaluate
                        ],
                    )
                    rollback_incomplete = getattr(
                        self._checkpoint_manager,
                        "rollback_incomplete_session",
                        None,
                    )
                    if callable(rollback_incomplete):
                        rollback_incomplete()
                finally:
                    session_wall_time = time.perf_counter() - session_started_at
                    session_build_record["wall_time_seconds"] = session_wall_time
                    session_build_record["build_metrics"] = {
                        "wall_time_seconds": session_wall_time,
                        "usage": diff_usage_stats(
                            usage_tracker.get_stats(), usage_before
                        ),
                    }
                    session_build_results.append(session_build_record)

                if self._checkpoint_manager and session_succeeded:
                    self._checkpoint_manager.mark_session_injected(session.session_id)
                if memory_build_failed:
                    break

            # Log summary
            total_passages = sum(
                len(r.get("build_result", {}).get("all_passages", []))
                for r in session_build_results if "build_result" in r
            )
            self._log(f"      [Summary] Total passages stored: {total_passages}")

            if restored_payload is not None and not session_build_results:
                unit_build_metrics = copy.deepcopy(
                    restored_payload.get("memory_build_metrics") or {
                        "wall_time_seconds": total_memory_time,
                        "usage": {},
                    }
                )
            else:
                unit_usage = merge_usage_stats(*[
                    record.get("build_metrics", {}).get("usage", {})
                    for record in session_build_results
                ])
                unit_build_metrics = {
                    "wall_time_seconds": sum(
                        float(record.get("wall_time_seconds", 0.0) or 0.0)
                        for record in session_build_results
                    ),
                    "usage": unit_usage,
                }

            # Store all session build results
            self._memory_build_logs.append({
                "unit_id": unit.unit_id,
                "context_id": unit.context_id,
                "session_ids": session_ids,
                "session_count": total_sessions,
                "total_time": total_memory_time,
                "total_passages": total_passages,
                "session_builds": session_build_results,
                "build_metrics": unit_build_metrics,
                "feature_configuration": self._amem_feature_configuration(),
                "restored_from_snapshot": bool(
                    restored_payload is not None and not session_build_results
                ),
            })

        self._log(f"    --- Memory Build Done, time={total_memory_time:.2f}s ---")

        if memory_build_failed:
            self._log(
                "    [API ERROR] Skipping this unit's queries because memory construction "
                "did not complete; no metric results will be recorded.",
                level="ERROR",
            )
            return results

        # Snapshot boundaries are part of the staged AMem execution, but their
        # disk I/O is intentionally excluded from method construction/query time.
        snapshot_path = self._save_and_restore_unit_memory(
            unit,
            memory_build_time=total_memory_time,
            memory_build_metrics=(
                self._memory_build_logs[-1].get("build_metrics", {})
                if self._memory_build_logs else {}
            ),
        )
        if snapshot_path is not None and self._memory_build_logs:
            self._memory_build_logs[-1]["memory_snapshot"] = str(snapshot_path)

        if getattr(self, "execution_stage", "all") == "memory":
            self._log("    --- Query Evaluation Skipped (memory stage only) ---")
            return results

        return self._evaluate_unit_queries(unit, total_memory_time)

    def _evaluate_unit_queries(
        self,
        unit: EvaluationUnit,
        total_memory_time: float,
    ) -> List[MetricResult]:
        """Retrieve, answer, and score one unit from its active memory state."""
        results: List[MetricResult] = []

        total_query_time = 0.0
        self._log(f"    --- Query Evaluation Start ---")

        query_count = len(unit.queries_to_evaluate)
        memory_time_per_query = total_memory_time / query_count if query_count > 0 else 0.0

        pending_query_count = sum(
            1
            for query in unit.queries_to_evaluate
            if not (
                self._checkpoint_manager
                and self._checkpoint_manager.is_query_completed(
                    query.query_id,
                    persona_id=unit.context_id,
                )
            )
            and not self._is_deferred_judge_query(query.query_id)
        )
        if not pending_query_count:
            self._log(f"    --- Query Evaluation Done, time={total_query_time:.2f}s ---")
            return results

        if pending_query_count and self._supports_batch_queries():
            batch_client = self._get_batch_client()
            combined_stage = "query-final"
            legacy_stage = f"query-unit-{unit.unit_id}"
            if not batch_client.has_stage(combined_stage) and batch_client.has_stage(legacy_stage):
                query_results = self._evaluate_batch_queries(unit, memory_time_per_query)
            else:
                prepared_count = self._prepare_combined_batch_queries(
                    unit,
                    memory_time_per_query,
                )
                query_results = []
                self._log(
                    f"    [Vertex Batch] Prepared {prepared_count:,} request(s) for "
                    f"combined stage '{combined_stage}'."
                )
            for result in query_results:
                results.append(result)
                total_query_time += result.query_time
                status = "✓" if result.is_correct else "✗"
                self._log(f"    [{status}] {result.query_id} ({result.query_type}): {result.score:.2f}")
                if self._checkpoint_manager:
                    self._checkpoint_manager.mark_query_completed(
                        result.query_id,
                        result.to_dict(),
                        persona_id=unit.context_id,
                    )
        else:
            if self.batch_api and not self.dry_run and not self._batch_fallback_logged:
                self._log(
                    "Vertex batch API is unavailable for this adapter; "
                    "using real-time final-answer generation.",
                    level="WARNING",
                )
                self._batch_fallback_logged = True

            for query in unit.queries_to_evaluate:
                if (
                    self._checkpoint_manager
                    and self._checkpoint_manager.is_query_completed(
                        query.query_id,
                        persona_id=unit.context_id,
                    )
                ):
                    self._log(f"    [Skip] {query.query_id} (completed)")
                    continue
                if self._is_deferred_judge_query(query.query_id):
                    self._log(f"    [Skip] {query.query_id} (awaiting saved Vertex judge result)")
                    continue

                if self._supports_memory_snapshots():
                    result = self._evaluate_query_staged(
                        query,
                        unit.context_id,
                        memory_time_per_query=memory_time_per_query,
                    )
                else:
                    result = self._evaluate_query(
                        query,
                        unit.context_id,
                        memory_time_per_query=memory_time_per_query,
                    )
                if result is None:
                    continue
                results.append(result)
                total_query_time += result.query_time

                status = "✓" if result.is_correct else "✗"
                self._log(f"    [{status}] {query.query_id} ({query.query_type}): {result.score:.2f}")

                if self._checkpoint_manager:
                    self._checkpoint_manager.mark_query_completed(
                        query.query_id,
                        result.to_dict(),
                        persona_id=unit.context_id,
                    )

        self._log(f"    --- Query Evaluation Done, time={total_query_time:.2f}s ---")

        return results

    def _supports_batch_queries(self) -> bool:
        return bool(
            self.batch_api
            and self.agent_manager
            and self.agent_manager.supports_batch_queries()
        )

    def _get_batch_client(self) -> VertexBatchClient:
        if self._batch_client is None:
            if self.agent_manager is None:
                raise VertexBatchError("Agent manager is not initialized for batch execution.")
            llm_client = self.agent_manager.get_batch_llm_client()
            if llm_client is None:
                raise VertexBatchError("This method does not expose a managed Gemini batch client.")
            self._batch_client = VertexBatchClient.from_gemini_client(
                llm_client,
                gcs_uri=self.batch_gcs_uri,
                manifest_path=self._batch_manifest_path("medmemorybench_batch_manifest", llm_client.model),
                wait=self.batch_wait,
                config_hash=self._batch_manifest_config_hash(),
                progress_callback=self._log,
            )
        return self._batch_client

    def _get_judge_batch_client(self, judge_client) -> VertexBatchClient:
        """Create an independent manifest because judge and agent models may differ."""
        if self._judge_batch_client is None:
            self._judge_batch_client = VertexBatchClient.from_gemini_client(
                judge_client,
                gcs_uri=self.batch_gcs_uri,
                manifest_path=scoped_manifest_path(
                    self.output_dir / "batch",
                    "medmemorybench_judge_batch_manifest",
                    model=judge_client.model,
                    config_hash=self._judge_batch_config_hash(),
                ),
                wait=self.batch_wait,
                config_hash=self._judge_batch_config_hash(),
                progress_callback=self._log,
            )
        return self._judge_batch_client

    def _evaluate_batch_queries(
        self,
        unit: EvaluationUnit,
        memory_time_per_query: float,
    ) -> List[MetricResult]:
        """Prepare local retrieval for a unit, then batch its final generations."""
        prepared_by_id: Dict[str, tuple[Any, Dict[str, Any]]] = {}
        requests: List[BatchChatRequest] = []
        stage = f"query-unit-{unit.unit_id}"
        batch_client = self._get_batch_client()

        for query in unit.queries_to_evaluate:
            if (
                self._checkpoint_manager
                and self._checkpoint_manager.is_query_completed(
                    query.query_id,
                    persona_id=unit.context_id,
                )
            ):
                self._log(f"    [Skip] {query.query_id} (completed)")
                continue
            if self._is_deferred_judge_query(query.query_id):
                self._log(f"    [Skip] {query.query_id} (awaiting saved Vertex judge result)")
                continue
            request_id = make_request_id(
                "query",
                f"{self.method_config.method_name}:{unit.unit_id}:{query.query_id}",
            )
            saved_request = batch_client.get_saved_request(stage, request_id)
            prepared = restore_prepared_query(saved_request) if saved_request else None
            if prepared is None:
                # Reuse a persisted display timestamp for legacy manifests that
                # predate prepared-query snapshots. New manifests do not rerun
                # retrieval at all when collecting a completed stage.
                batch_request_time = (
                    saved_request.metadata.get("batch_request_time")
                    if saved_request is not None
                    else datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                )
                formatted_question = self.prompt_manager.format_query(
                    question=query.question,
                    query_type=query.query_type,
                )
                try:
                    prepared = self._run_api_call(
                        self.agent_manager.prepare_batch_query,
                        formatted_question,
                        query_id=query.query_id,
                        context_id=unit.context_id,
                        batch_request_time=batch_request_time,
                        raw_question=query.question,
                        query_type=query.query_type,
                    )
                except LLMAPIError as e:
                    self._log(
                        f"    [API ERROR] {query.query_id}: query preparation failed; "
                        f"skipping this query. Error: {e}",
                        level="ERROR",
                    )
                    self._record_api_failure(
                        "query_preparation",
                        e,
                        query_id=query.query_id,
                        query_type=query.query_type,
                        context_id=unit.context_id,
                        unit_id=unit.unit_id,
                    )
                    continue

            if saved_request is not None:
                # Reuse the exact staged request. This protects a resume from
                # both timestamp changes and non-deterministic retrieval.
                requests.append(saved_request)
            else:
                requests.append(
                    BatchChatRequest(
                        request_id=request_id,
                        messages=prepared["messages"],
                        temperature=self.method_config.model.temperature,
                        max_tokens=(
                            self.method_config.model.max_completion_tokens
                            or self.method_config.model.max_tokens
                        ),
                        phase="query",
                        metadata={
                            "query_id": query.query_id,
                            "unit_id": unit.unit_id,
                            "batch_request_time": batch_request_time,
                            PREPARED_QUERY_METADATA_KEY: snapshot_prepared_query(prepared),
                        },
                    )
                )
            prepared_by_id[request_id] = (query, prepared)

        if not requests:
            return []

        self._log(
            f"    [Vertex Batch] Stage '{stage}': local preparation complete; "
            f"submitting {len(requests):,} final-answer request(s)."
        )
        get_saved_requests = getattr(batch_client, "get_saved_requests", None)
        saved_requests = get_saved_requests(stage) if callable(get_saved_requests) else []
        submitted_requests = saved_requests or requests
        responses = batch_client.run_stage(stage, submitted_requests)
        results: List[MetricResult] = []
        for request_id, (query, prepared) in prepared_by_id.items():
            batch_response = responses.get(request_id)
            if batch_response is None or batch_response.status:
                error = batch_response.status if batch_response else "No output row returned"
                self._record_batch_api_failure(
                    query,
                    f"Vertex batch request failed: {error}",
                    failure_duration_seconds=(
                        getattr(batch_response, "duration_seconds", 0.0)
                        if batch_response is not None else 0.0
                    ),
                    context_id=unit.context_id,
                    unit_id=unit.unit_id,
                )
                continue
            try:
                response = self._run_api_call(
                    self.agent_manager.finalize_batch_query,
                    prepared,
                    batch_response.content,
                    input_tokens=batch_response.input_tokens,
                    output_tokens=batch_response.output_tokens,
                )
                result = self._run_api_call(
                    self._score_agent_response,
                    query,
                    response,
                    context_id=unit.context_id,
                    memory_construction_time=memory_time_per_query,
                )
            except LLMAPIError as e:
                self._log(
                    f"    [API ERROR] {query.query_id}: judge API call failed; "
                    f"skipping this query. Error: {e}",
                    level="ERROR",
                )
                self._record_api_failure(
                    "judge",
                    e,
                    query_id=query.query_id,
                    query_type=query.query_type,
                    context_id=unit.context_id,
                    unit_id=unit.unit_id,
                )
                continue
            if result is not None:
                results.append(result)
        return results

    def _prepare_combined_batch_queries(
        self,
        unit: EvaluationUnit,
        memory_time_per_query: float,
    ) -> int:
        """Freeze this unit's prompts while its current memory snapshot is active."""
        stage = "query-final"
        batch_client = self._get_batch_client()
        prepared_count = 0

        for query in unit.queries_to_evaluate:
            if (
                self._checkpoint_manager
                and self._checkpoint_manager.is_query_completed(
                    query.query_id,
                    persona_id=unit.context_id,
                )
            ):
                self._log(f"    [Skip] {query.query_id} (completed)")
                continue
            if self._is_deferred_judge_query(query.query_id):
                self._log(f"    [Skip] {query.query_id} (awaiting saved Vertex judge result)")
                continue

            request_id = make_request_id(
                "query",
                f"{self.method_config.method_name}:{unit.unit_id}:{query.query_id}",
            )
            saved_request = batch_client.get_saved_request(stage, request_id)
            batch_request_time = (
                saved_request.metadata.get("batch_request_time")
                if saved_request is not None
                else datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            )
            prepared = restore_prepared_query(saved_request) if saved_request else None
            if prepared is None:
                formatted_question = self.prompt_manager.format_query(
                    question=query.question,
                    query_type=query.query_type,
                )
                try:
                    prepared = self._run_api_call(
                        self.agent_manager.prepare_batch_query,
                        formatted_question,
                        query_id=query.query_id,
                        context_id=unit.context_id,
                        batch_request_time=batch_request_time,
                        raw_question=query.question,
                        query_type=query.query_type,
                    )
                except LLMAPIError as e:
                    self._log(
                        f"    [API ERROR] {query.query_id}: query preparation failed; "
                        f"skipping this query. Error: {e}",
                        level="ERROR",
                    )
                    self._record_api_failure(
                        "query_preparation",
                        e,
                        query_id=query.query_id,
                        query_type=query.query_type,
                        context_id=unit.context_id,
                        unit_id=unit.unit_id,
                    )
                    continue

            request = saved_request or BatchChatRequest(
                request_id=request_id,
                messages=prepared["messages"],
                temperature=self.method_config.model.temperature,
                max_tokens=(
                    self.method_config.model.max_completion_tokens
                    or self.method_config.model.max_tokens
                ),
                phase="query",
                metadata={
                    "query_id": query.query_id,
                    "unit_id": unit.unit_id,
                    "context_id": unit.context_id,
                    "memory_time_per_query": memory_time_per_query,
                    "batch_request_time": batch_request_time,
                    PREPARED_QUERY_METADATA_KEY: snapshot_prepared_query(prepared),
                },
            )
            self._pending_batch_queries.append({
                "request": request,
                "query": query,
                "prepared": prepared,
                "persona_id": unit.context_id,
                "unit_id": unit.unit_id,
                "memory_time_per_query": request.metadata.get(
                    "memory_time_per_query",
                    memory_time_per_query,
                ),
            })
            prepared_count += 1

        return prepared_count

    def _complete_combined_batch_queries(self) -> List[Dict[str, Any]]:
        """Submit all frozen final-answer prompts in one Vertex batch job."""
        if not self._pending_batch_queries:
            return []

        stage = "query-final"
        batch_client = self._get_batch_client()
        saved_requests = batch_client.get_saved_requests(stage)
        requests = saved_requests or [
            item["request"] for item in self._pending_batch_queries
        ]
        self._log(
            f"[Vertex] Stage '{stage}': dispatching {len(requests):,} combined "
            "final-answer request(s) from all prepared units."
        )
        responses = batch_client.run_stage(stage, requests)
        finalized: List[Dict[str, Any]] = []

        for item in self._pending_batch_queries:
            request = item["request"]
            query = item["query"]
            batch_response = responses.get(request.request_id)
            if batch_response is None or batch_response.status:
                error = batch_response.status if batch_response else "No output row returned"
                self._record_batch_api_failure(
                    query,
                    f"Vertex batch request failed: {error}",
                    failure_duration_seconds=(
                        getattr(batch_response, "duration_seconds", 0.0)
                        if batch_response is not None else 0.0
                    ),
                    context_id=item["persona_id"],
                    unit_id=item["unit_id"],
                )
                continue
            try:
                response = self._run_api_call(
                    self.agent_manager.finalize_batch_query,
                    item["prepared"],
                    batch_response.content,
                    input_tokens=batch_response.input_tokens,
                    output_tokens=batch_response.output_tokens,
                )
                result = self._run_api_call(
                    self._score_agent_response,
                    query,
                    response,
                    context_id=item["persona_id"],
                    memory_construction_time=item["memory_time_per_query"],
                )
            except LLMAPIError as e:
                self._log(
                    f"[API ERROR] {query.query_id}: judge API call failed; "
                    f"skipping this query. Error: {e}",
                    level="ERROR",
                )
                self._record_api_failure(
                    "judge",
                    e,
                    query_id=query.query_id,
                    query_type=query.query_type,
                    context_id=item["persona_id"],
                    unit_id=item["unit_id"],
                )
                continue
            if result is not None:
                status = "✓" if result.is_correct else "✗"
                self._log(
                    f"[{status}] {result.query_id} ({result.query_type}): {result.score:.2f}"
                )
                finalized.append({
                    "persona_id": item["persona_id"],
                    "result": result,
                })

        self._pending_batch_queries = []
        return finalized

    def _evaluate_query(
        self,
        query,
        context_id: int,
        memory_time_per_query: float = 0.0,
    ) -> Optional[MetricResult]:
        if self.dry_run:
            return MetricResult(
                query_id=query.query_id,
                query_type=query.query_type,
                score=0.0,
                is_correct=False,
                model_output="[DRY RUN]",
                expected_answer=", ".join(query.get_correct_answers()),
                question=query.question,
                details={"dry_run": True},
            )

        formatted_question = self.prompt_manager.format_query(
            question=query.question,
            query_type=query.query_type,
        )

        try:
            response = self._run_api_call(
                self.agent_manager.send_message,
                message=formatted_question,
                memorizing=False,
                query_id=query.query_id,
                context_id=context_id,
                raw_question=query.question,
                query_type=query.query_type,
            )
        except LLMAPIError as e:
            self._log(
                f"    [API ERROR] {query.query_id}: answer API call failed, "
                f"skipping this query. Error: {e}",
                level="ERROR"
            )
            self._record_api_failure(
                "query",
                e,
                query_id=query.query_id,
                query_type=query.query_type,
                context_id=context_id,
            )
            return None

        try:
            return self._run_api_call(
                self._score_agent_response,
                query,
                response,
                context_id=context_id,
                memory_construction_time=memory_time_per_query,
            )
        except LLMAPIError as e:
            self._log(
                f"    [API ERROR] {query.query_id}: judge API call failed, "
                f"skipping this query. Error: {e}",
                level="ERROR"
            )
            self._record_api_failure(
                "judge",
                e,
                query_id=query.query_id,
                query_type=query.query_type,
                context_id=context_id,
            )
            return None

    def _evaluate_query_staged(
        self,
        query,
        context_id: int,
        memory_time_per_query: float = 0.0,
    ) -> Optional[MetricResult]:
        """Run retrieval, final answering, and scoring as explicit stages."""
        formatted_question = self.prompt_manager.format_query(
            question=query.question,
            query_type=query.query_type,
        )
        try:
            staged_query = self._run_api_call(
                self.agent_manager.prepare_query,
                message=formatted_question,
                query_id=query.query_id,
                context_id=context_id,
                raw_question=query.question,
                query_type=query.query_type,
            )
            response = self._run_api_call(
                self.agent_manager.answer_prepared_query,
                staged_query,
            )
        except LLMAPIError as e:
            self._log(
                f"    [API ERROR] {query.query_id}: answer API call failed, "
                f"skipping this query. Error: {e}",
                level="ERROR",
            )
            self._record_api_failure(
                "query",
                e,
                query_id=query.query_id,
                query_type=query.query_type,
                context_id=context_id,
            )
            return None

        try:
            return self._run_api_call(
                self._score_agent_response,
                query,
                response,
                context_id=context_id,
                memory_construction_time=memory_time_per_query,
            )
        except LLMAPIError as e:
            self._log(
                f"    [API ERROR] {query.query_id}: judge API call failed, "
                f"skipping this query. Error: {e}",
                level="ERROR",
            )
            self._record_api_failure(
                "judge",
                e,
                query_id=query.query_id,
                query_type=query.query_type,
                context_id=context_id,
            )
            return None

    def _record_batch_api_failure(
        self,
        query,
        error_message: str,
        **context: Any,
    ) -> None:
        failure_duration_seconds = context.pop("failure_duration_seconds", None)
        self._log(
            f"    [API ERROR] {query.query_id}: {error_message}; "
            "skipping this query.",
            level="ERROR",
        )
        self._record_api_failure(
            "batch_query",
            VertexBatchError(error_message),
            query_id=query.query_id,
            query_type=query.query_type,
            failure_duration_seconds=failure_duration_seconds,
            **context,
        )

    def _score_agent_response(
        self,
        query,
        response: Any,
        context_id: Optional[int] = None,
        memory_construction_time: float = 0.0,
    ) -> Optional[MetricResult]:
        """Score a normal or batch-finalized agent response with unchanged metrics."""
        answers_data = query.answers_data if isinstance(query, MedQuery) else None

        if isinstance(response, dict):
            model_output = response.get("output", "")
            query_time = response.get("query_time", 0.0)
            retrieved_memories = response.get("retrieved_memories", [])
            retrieved_count = response.get("retrieved_count", 0)
        elif hasattr(response, "output"):
            model_output = response.output
            query_time = getattr(response, "query_time", 0.0)
            retrieved_memories = getattr(response, "retrieved_memories", [])
            retrieved_count = getattr(response, "retrieved_count", 0)
        else:
            model_output = str(response)
            query_time = 0.0
            retrieved_memories = []
            retrieved_count = 0

        metric_kwargs = {
            "answers_data": answers_data,
            "metadata": query.metadata,
        }
        prepared_metric = None
        judge_client = None
        if self.batch_api and not self.dry_run:
            prepared_metric = self.metrics_calculator.prepare_batch(
                query_id=query.query_id,
                query_type=query.query_type,
                model_output=model_output,
                expected_answers=query.get_correct_answers(),
                question=query.question,
                **metric_kwargs,
            )
            if prepared_metric is not None:
                judge_client = self.metrics_calculator.get_batch_judge_client(query.query_type)

        if prepared_metric is not None and judge_client is not None:
            judge_payload = prepared_metric["prepared"]["judge_payload"]
            if "immediate" not in judge_payload:
                self._deferred_judges.append({
                    "query_id": query.query_id,
                    "persona_id": context_id,
                    "prepared_metric": prepared_metric,
                    "expected_answers": query.get_correct_answers(),
                    "metric_kwargs": metric_kwargs,
                    "query_time": query_time,
                    "memory_construction_time": memory_construction_time,
                    "retrieved_memories": retrieved_memories,
                    "retrieved_count": retrieved_count,
                })
                return None
            result = self.metrics_calculator.finalize_batch(prepared_metric, "")
        else:
            if prepared_metric is not None and not self._judge_batch_fallback_logged:
                self._log(
                    "Vertex batch API cannot batch the configured LLM judge; "
                    "using its existing real-time client.",
                    level="WARNING",
                )
                self._judge_batch_fallback_logged = True
            result = self.metrics_calculator.compute(
                query_id=query.query_id,
                query_type=query.query_type,
                model_output=model_output,
                expected_answers=query.get_correct_answers(),
                question=query.question,
                **metric_kwargs,
            )

        result.query_time = query_time
        result.memory_construction_time = memory_construction_time
        result.retrieved_memories = retrieved_memories
        result.retrieved_count = retrieved_count

        return result

    def _complete_deferred_judges(self) -> List[Dict[str, Any]]:
        """Submit all independent MedMemoryBench judge prompts as one final stage."""
        if not self._deferred_judges:
            return []

        can_fallback_to_realtime = all(
            "expected_answers" in item and "metric_kwargs" in item
            for item in self._deferred_judges
        )
        if not should_use_batch(len(self._deferred_judges)) and can_fallback_to_realtime:
            self._log(
                f"[Vertex Batch] Only {len(self._deferred_judges):,} judge request(s); "
                "using real-time judging."
            )
            finalized: List[Dict[str, Any]] = []
            for item in self._deferred_judges:
                prepared = item["prepared_metric"]["prepared"]
                try:
                    result = self._run_api_call(
                        self.metrics_calculator.compute,
                        query_id=prepared["query_id"],
                        query_type=prepared["query_type"],
                        model_output=prepared["model_output"],
                        expected_answers=item["expected_answers"],
                        question=prepared["question"],
                        **item["metric_kwargs"],
                    )
                except LLMAPIError as e:
                    self._log(
                        f"[API ERROR] Judge failed for {prepared['query_id']}; "
                        f"no metric result recorded. Error: {e}",
                        level="ERROR",
                    )
                    self._record_api_failure(
                        "judge",
                        e,
                        query_id=prepared["query_id"],
                        query_type=prepared["query_type"],
                        context_id=item["persona_id"],
                    )
                    continue
                result.query_time = item["query_time"]
                result.memory_construction_time = item["memory_construction_time"]
                result.retrieved_memories = item["retrieved_memories"]
                result.retrieved_count = item["retrieved_count"]
                finalized.append({"persona_id": item["persona_id"], "result": result})
            self._clear_deferred_judges()
            return finalized

        first_prepared = self._deferred_judges[0]["prepared_metric"]["prepared"]
        judge_client = self.metrics_calculator.get_batch_judge_client(first_prepared["query_type"])
        if judge_client is None:
            raise VertexBatchError(
                "Deferred LLM-judge work requires the original Gemini judge configuration."
            )
        # This is written before the remote submission.  If the process exits
        # afterward, resume can collect the same job without regenerating answers.
        self._save_deferred_judges(judge_client)
        requests: List[BatchChatRequest] = []
        by_id: Dict[str, Dict[str, Any]] = {}
        for item in self._deferred_judges:
            payload = item["prepared_metric"]["prepared"]["judge_payload"]
            request_id = make_request_id(
                "judge",
                f"{self.method_config.method_name}:{item['persona_id']}:{item['query_id']}",
            )
            requests.append(
                BatchChatRequest(
                    request_id=request_id,
                    messages=[{"role": "user", "content": payload["prompt"]}],
                    temperature=payload.get(
                        "temperature", getattr(get_api_config(), "judge_temperature", 1.0)
                    ),
                    max_tokens=payload["max_tokens"],
                    response_format={"type": "json_object"},
                    phase="query",
                    metadata={"query_id": item["query_id"], "phase": "judge"},
                )
            )
            by_id[request_id] = item

        self._log(
            f"[Vertex Batch] Stage 'judge-final': prepared {len(requests):,} LLM-judge request(s)."
        )
        responses = self._get_judge_batch_client(judge_client).run_stage("judge-final", requests)
        finalized: List[Dict[str, Any]] = []
        for request_id, item in by_id.items():
            response = responses.get(request_id)
            prepared = item["prepared_metric"]["prepared"]
            if response is None or response.status:
                error = response.status if response else "No output row returned"
                self._log(
                    f"[API ERROR] Batch judge failed for {item['query_id']}; "
                    f"no metric result recorded. Error: {error}",
                    level="ERROR",
                )
                self._record_api_failure(
                    "batch_judge",
                    VertexBatchError(str(error)),
                    failure_duration_seconds=(
                        getattr(response, "duration_seconds", 0.0)
                        if response is not None else 0.0
                    ),
                    query_id=item["query_id"],
                    query_type=prepared["query_type"],
                    context_id=item["persona_id"],
                )
                continue
            try:
                result = self._run_api_call(
                    self.metrics_calculator.finalize_batch,
                    item["prepared_metric"],
                    response.content,
                )
            except LLMAPIError as e:
                self._log(
                    f"[API ERROR] Batch judge returned an unusable response for "
                    f"{item['query_id']}; no metric result recorded. Error: {e}",
                    level="ERROR",
                )
                self._record_api_failure(
                    "batch_judge",
                    e,
                    query_id=item["query_id"],
                    query_type=prepared["query_type"],
                    context_id=item["persona_id"],
                )
                continue
            result.query_time = item["query_time"]
            result.memory_construction_time = item["memory_construction_time"]
            result.retrieved_memories = item["retrieved_memories"]
            result.retrieved_count = item["retrieved_count"]
            finalized.append({"persona_id": item["persona_id"], "result": result})
        self._clear_deferred_judges()
        return finalized

    def _generate_report(
        self,
        start_time: datetime,
        end_time: datetime,
        duration: float,
    ) -> EvaluationReport:
        summary = self.aggregator.get_summary()

        memory_build_summary = self._summarize_memory_builds()
        build_metrics = self._build_metrics_report()

        llm_usage = get_usage_tracker().get_stats()
        failed_query_ids = set()
        for failure in self._api_failures:
            if failure.get("query_id"):
                failed_query_ids.add(failure["query_id"])
            failed_query_ids.update(failure.get("affected_query_ids", []))
        expected_queries = self._expected_query_count()
        scored_queries = len(self.aggregator.results)
        evaluation_coverage = {
            "expected_queries": expected_queries,
            "scored_queries": scored_queries,
            "omitted_queries": max(expected_queries - scored_queries, 0),
            "api_failed_queries": len(failed_query_ids),
            "api_failure_events": len(self._api_failures),
            "coverage": (
                scored_queries / expected_queries if expected_queries > 0 else 1.0
            ),
            "complete": scored_queries == expected_queries,
        }

        report = EvaluationReport(
            method_name=self.method_config.method_name,
            model_name=self.method_config.model.name,
            dataset_name=self.dataset_config.dataset_name,
            start_time=start_time.isoformat(),
            end_time=end_time.isoformat(),
            duration_seconds=duration,
            summary=summary,
            detailed_results=self.aggregator.get_detailed_results(),
            config={
                "method_config": self.method_config.raw_config,
                "dataset_config": self.dataset_config.raw_config,
                "judge_config": {
                    "provider": get_api_config().get_judge_provider(),
                    "model": get_api_config().get_judge_model(),
                    "temperature": getattr(get_api_config(), "judge_temperature", 1.0),
                    "client_max_tokens": getattr(
                        get_api_config(), "judge_client_max_tokens", 10000
                    ),
                    "max_tokens": getattr(get_api_config(), "judge_max_tokens", 500),
                    "mcd_max_tokens": getattr(
                        get_api_config(), "judge_mcd_max_tokens", 2000
                    ),
                },
                "dry_run": self.dry_run,
            },
            metadata={
                "evaluation_mode": self.dataset_config.evaluation_mode,
                "evaluation_interval": self.dataset_config.evaluation_interval,
                "total_personas": len(self.result_collector.get_context_ids()),
                "memory_build_summary": memory_build_summary,
                "build_metrics": build_metrics,
                "memory_size": build_metrics["memory_size"],
                "feature_configuration": build_metrics["feature_configuration"],
                "llm_usage": llm_usage,
                "evaluation_coverage": evaluation_coverage,
                "api_failures": self._api_failures,
                "api_failure_duration_seconds": self._api_failure_duration_seconds,
            }
        )

        execution_stage = getattr(self, "execution_stage", "all")
        result_path, memory_build_path, query_answer_path = self.result_collector.save_reports(
            report=report,
            output_dir=self.output_dir,
            memory_build_logs=self._combined_memory_build_logs(),
            include_result=execution_stage in {"all", "query"},
            include_memory_build=execution_stage in {"all", "memory"},
            include_query_answer=execution_stage in {"all", "query"},
            include_api_failures=True,
            use_method_subdir=not getattr(self, "run_scoped_output", False),
        )

        if result_path is not None:
            self._log(f"Results saved to: {result_path}")
        if memory_build_path is not None:
            self._log(f"Memory build details saved to: {memory_build_path}")
        if query_answer_path is not None:
            self._log(f"Query answer details saved to: {query_answer_path}")
        if self.result_collector.last_api_failure_path is not None:
            self._log(
                f"API failure details saved separately to: "
                f"{self.result_collector.last_api_failure_path}"
            )

        return report

    def _expected_query_count(self) -> int:
        """Return coverage denominator for the units selected by this run."""
        units = self._get_evaluation_units()
        if getattr(self, "append", False) and self._append_target_index is not None:
            units = units[: self._append_target_index + 1]
            return sum(len(unit.queries_to_evaluate) for unit in units)
        if self.execution_stage == "query" and self._memory_unit_ids is not None:
            units = [unit for unit in units if unit.unit_id in self._memory_unit_ids]
            return sum(len(unit.queries_to_evaluate) for unit in units)
        return self.dataset.get_total_queries()

    def _summarize_memory_builds(self) -> Dict[str, Any]:
        build_logs = self._combined_memory_build_logs()
        if not build_logs:
            return {"total_builds": 0}

        total_units = len(build_logs)

        total_sessions = sum(
            log.get("session_count", 0)
            for log in build_logs
        )

        # New structure: sum time from each unit's total_time
        total_time = sum(
            log.get("total_time", 0)
            for log in build_logs
        )

        # Count total passages from session_builds
        total_passages = sum(
            log.get("total_passages", 0)
            for log in build_logs
        )

        # Count memory entries from session builds
        total_memory_entries = 0
        methods = {}

        for log in build_logs:
            session_builds = log.get("session_builds", [])
            for sb in session_builds:
                build_result = sb.get("build_result", {})
                total_memory_entries += len(build_result.get("memory_entries", []))

                method = build_result.get("method", "unknown")
                if method not in methods:
                    methods[method] = {"count": 0, "time_cost": 0, "passages": 0}
                methods[method]["count"] += 1
                methods[method]["time_cost"] += build_result.get("time_cost", 0)
                methods[method]["passages"] += len(build_result.get("all_passages", []))

        return {
            "total_units": total_units,
            "total_sessions": total_sessions,
            "total_time": total_time,
            "avg_time_per_session": total_time / total_sessions if total_sessions > 0 else 0,
            "total_passages": total_passages,
            "total_memory_entries": total_memory_entries,
            "by_method": methods,
        }


@register_evaluator("medmemorybench")
def evaluate_medmemorybench(
    method_config: MethodConfig,
    dataset_config: DatasetConfig,
    output_dir: Path,
    dry_run: bool = False,
    verbose: bool = True,
    logger: Optional[logging.Logger] = None,
    resume: bool = False,
    force_resume: bool = False,
    execution_stage: str = "all",
    memory_run: Optional[str] = None,
    append: bool = False,
    append_persona: Optional[int] = None,
    append_unit: Optional[int] = None,
    memory_source_run_dir: Optional[Path] = None,
    memory_run_explicit: bool = False,
    run_scoped_output: bool = False,
    batch_api: bool = False,
    batch_gcs_uri: Optional[str] = None,
    batch_wait: bool = False,
    **kwargs
) -> EvaluationReport:
    """MedMemoryBench evaluation entry point."""
    evaluator = MedMemoryBenchEvaluator(
        method_config=method_config,
        dataset_config=dataset_config,
        output_dir=output_dir,
        dry_run=dry_run,
        verbose=verbose,
        logger=logger,
        resume=resume,
        force_resume=force_resume,
        execution_stage=execution_stage,
        memory_run=memory_run,
        append=append,
        append_persona=append_persona,
        append_unit=append_unit,
        memory_source_run_dir=memory_source_run_dir,
        memory_run_explicit=memory_run_explicit,
        run_scoped_output=run_scoped_output,
        batch_api=batch_api,
        batch_gcs_uri=batch_gcs_uri,
        batch_wait=batch_wait,
    )
    return evaluator.evaluate()
