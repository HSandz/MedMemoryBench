"""MedMemoryBench evaluation module."""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging

from src.config import MethodConfig, DatasetConfig, PROJECT_ROOT, get_api_config
from src.evaluator import register_evaluator
from src.agent import AgentManager, AgentResponse, MemoryBuildResult
from src.result import EvaluationReport, ResultCollector
from src.memory_snapshot import (
    MemorySnapshotKey,
    MemorySnapshotStore,
    compute_memory_config_hash,
)
from benchmarks.medmemorybench.dataset import MedMemoryBenchDataset, MedQuery, MedSession
from benchmarks.base import EvaluationUnit
from metrics import MetricsCalculator, MetricsAggregator, MetricResult
from metrics.base import MetricResult as BaseMetricResult
from utils.templates import get_prompt_manager
from utils.llm_client import get_usage_tracker, LLMRetryExhaustedError
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
        rebuild_memory: bool = False,
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
        self.rebuild_memory = rebuild_memory
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
        self.metrics_calculator = MetricsCalculator(
            judge_model=api_config.judge_model or None,
            judge_api_key=api_config.judge_api_key or None,
            judge_base_url=api_config.judge_base_url or None,
            language=dataset_config.language,
        )
        self.aggregator = MetricsAggregator()
        self.result_collector = ResultCollector()

        self._memory_build_logs: List[Dict[str, Any]] = []
        self._batch_client: Optional[VertexBatchClient] = None
        self._judge_batch_client: Optional[VertexBatchClient] = None
        self._batch_fallback_logged = False
        self._judge_batch_fallback_logged = False
        self._deferred_judges: List[Dict[str, Any]] = []

        self._checkpoint_manager: Optional[MedMemoryBenchCheckpointManager] = None
        self._checkpoint_enabled = False
        self._memory_snapshot_store = MemorySnapshotStore(
            self.output_dir / "memory_snapshots",
            dataset=dataset_config.dataset_name,
            method=method_config.method_name,
            model=method_config.model.name,
            config_hash=compute_memory_config_hash(method_config, dataset_config),
        )
        self._memory_snapshot_lineage: Dict[str, str] = {}

        if self._should_enable_checkpoint():
            self._init_checkpoint_manager()

    def _log(self, message: str, level: str = "INFO") -> None:
        if self.verbose:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] [{level}] {message}")
        if self.logger:
            self.logger.info(message)

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

        self._log(f"  Total Sessions: {self.dataset.get_total_sessions()}")
        self._log(f"  Total Queries: {self.dataset.get_total_queries()}")
        self._log(f"  Evaluation Mode: {self.dataset_config.evaluation_mode}")
        self._log(f"  Evaluation Interval: {self.dataset_config.evaluation_interval}")

    def _should_enable_checkpoint(self) -> bool:
        return self.dataset_config.evaluation_mode == "independent"

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
            self._log("Checkpoint file corrupted, starting from scratch")
            self._checkpoint_manager.delete()
            return False

        if not self._checkpoint_manager.validate_config():
            self._log("Config changed, checkpoint invalid, starting from scratch")
            self._checkpoint_manager.delete()
            return False

        if not self._checkpoint_manager.is_independent_mode():
            self._log("Checkpoint not in independent mode, cannot resume")
            return False

        self._load_checkpoint_results()

        info = self._checkpoint_manager.get_resume_info()
        self._log("=" * 50)
        self._log("Resuming from checkpoint")
        self._log(f"  Checkpoint ID: {info['checkpoint_id'][:8]}...")
        self._log(f"  Completed Personas: {info['completed_personas']}/{info['total_personas']}")
        self._log(f"  Completed Queries: {info['completed_queries']}/{info['total_queries']}")
        if info['current_persona'] is not None:
            self._log(f"  Current Persona: {info['current_persona']} "
                     f"({info['current_persona_completed_queries']} queries done, rebuilding memory)")
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

    def _memory_snapshot_key(self, unit: EvaluationUnit) -> MemorySnapshotKey:
        context_key = str(unit.context_id)
        key = self._memory_snapshot_store.make_key(
            unit,
            parent_fingerprint=self._memory_snapshot_lineage.get(context_key, ""),
        )
        self._memory_snapshot_lineage[context_key] = key.unit_fingerprint
        return key

    def _try_load_memory_snapshot(
        self,
        key: MemorySnapshotKey,
    ) -> bool:
        if (
            self.rebuild_memory
            or self.agent_manager is None
            or not self.agent_manager.supports_memory_snapshots()
        ):
            return False
        state = self._memory_snapshot_store.load(key)
        if state is None:
            return False
        try:
            self.agent_manager.import_memory_state(state)
        except (RuntimeError, ValueError, TypeError) as exc:
            self._log(f"      [Memory Cache] Invalid snapshot ignored: {exc}", level="WARNING")
            return False
        self._log(
            f"      [Memory Cache] Loaded exact unit snapshot: "
            f"{self._memory_snapshot_store.path_for(key)}"
        )
        return True

    def _save_memory_snapshot(
        self,
        key: MemorySnapshotKey,
        session_ids: List[str],
    ) -> Optional[Path]:
        if self.agent_manager is None or not self.agent_manager.supports_memory_snapshots():
            return None
        state = self.agent_manager.export_memory_state()
        path = self._memory_snapshot_store.save(
            key,
            state,
            session_ids=session_ids,
        )
        self._log(f"      [Memory Cache] Saved unit snapshot: {path}")
        return path

    def _deferred_judge_state_paths(self) -> List[Path]:
        batch_dir = self.output_dir / "batch"
        return sorted(batch_dir.glob(
            f"medmemorybench_deferred_judges-*-{self._batch_manifest_config_hash()}.json"
        ))

    def _save_deferred_judges(self, judge_client) -> None:
        """Persist judge inputs before submitting so resume never regenerates them."""
        path = self._batch_manifest_path("medmemorybench_deferred_judges", judge_client.model)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "config_hash": self._batch_manifest_config_hash(),
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
            or payload.get("config_hash") != self._batch_manifest_config_hash()
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

        resumed = False
        if self.resume and self._checkpoint_enabled:
            resumed = self._try_resume_from_checkpoint()

        # Deferred judge inputs belong to the loaded checkpoint namespace.
        # Do not attach a fresh run to an older namespace when no checkpoint
        # was successfully resumed.
        if resumed and self.batch_api and not self.dry_run:
            self._load_deferred_judges()

        if not resumed and self._checkpoint_enabled:
            self._create_new_checkpoint()

        self._run_evaluation_loop()

        if self._checkpoint_manager:
            self._checkpoint_manager.mark_completed()
            self._checkpoint_manager.delete()
            self._log("Evaluation completed, checkpoint deleted")

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        report = self._generate_report(start_time, end_time, duration)

        return report

    def _run_evaluation_loop(self) -> None:
        current_context_id = None

        for unit in self.dataset.get_evaluation_units():
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
                    # Do not wipe progress if we are resuming the same persona that was interrupted
                    if self._checkpoint_manager.get_current_persona_id() != persona_id:
                        self._checkpoint_manager.start_persona(persona_id)

            unit_results = self._evaluate_unit_with_checkpoint(unit)

            for result in unit_results:
                self.aggregator.add_result(result)
                self.result_collector.add_result(result, persona_id)

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
        snapshot_key = self._memory_snapshot_key(unit)
        snapshot_loaded = False

        if self.dry_run:
            self._log(f"    [Dry Run] Skipping memory build")
        else:
            total_sessions = len(unit.sessions_to_inject)
            session_ids = [str(session.session_id) for session in unit.sessions_to_inject]
            snapshot_loaded = self._try_load_memory_snapshot(snapshot_key)

            for idx, session in enumerate(unit.sessions_to_inject):
                if snapshot_loaded:
                    if self._checkpoint_manager:
                        self._checkpoint_manager.mark_session_injected(session.session_id)
                    continue
                memory_text = session.to_memory_text()

                # Show progress
                self._log(f"      [Progress] Session {idx + 1}/{total_sessions} (ID: {session.session_id})")

                formatted_text = self.prompt_manager.format_memorize(
                    context=memory_text,
                    timestamp=None,
                )

                is_last_session = (idx == total_sessions - 1)

                try:
                    memory_result = self.agent_manager.send_message(
                        message=formatted_text,
                        memorizing=True,
                        context_id=unit.context_id,
                        is_last_session=is_last_session,
                    )

                    if memory_result is not None and isinstance(memory_result, MemoryBuildResult):
                        total_memory_time += memory_result.time_cost

                        # Log brief progress info
                        passages_count = len(memory_result.all_passages) if memory_result.all_passages else memory_result.extra.get("inserted_count", len(memory_result.memory_entries))
                        self._log(f"        → Stored {passages_count} passages, time={memory_result.time_cost:.2f}s")

                        # Show extraction preview (first 100 chars)
                        if memory_result.extraction_result:
                            preview = memory_result.extraction_result[:150].replace('\n', ' ')
                            self._log(f"        → Extraction: {preview}...")

                        # Store detailed result for this session
                        session_build_results.append({
                            "session_id": session.session_id,
                            "session_index": idx,
                            "build_result": memory_result.to_dict(),
                        })

                    if self._checkpoint_manager:
                        self._checkpoint_manager.mark_session_injected(session.session_id)

                except LLMRetryExhaustedError as e:
                    self._log(
                        f"        [API ERROR] Session {session.session_id} failed: {e}",
                        level="ERROR"
                    )
                    memory_build_failed = True
                    session_build_results.append({
                        "session_id": session.session_id,
                        "session_index": idx,
                        "error": str(e),
                    })

            snapshot_path = None
            if not snapshot_loaded and not memory_build_failed:
                snapshot_path = self._save_memory_snapshot(snapshot_key, session_ids)

            # Log summary
            total_passages = sum(
                len(r.get("build_result", {}).get("all_passages", []) or r.get("build_result", {}).get("memory_entries", []))
                for r in session_build_results if "build_result" in r
            )
            self._log(f"      [Summary] Total passages stored: {total_passages}")

            # Store all session build results
            self._memory_build_logs.append({
                "unit_id": unit.unit_id,
                "context_id": unit.context_id,
                "session_ids": session_ids,
                "session_count": total_sessions,
                "total_time": total_memory_time,
                "total_passages": total_passages,
                "memory_snapshot_hit": snapshot_loaded,
                "memory_snapshot_path": str(
                    snapshot_path or self._memory_snapshot_store.path_for(snapshot_key)
                ),
                "session_builds": session_build_results,
            })

        self._log(f"    --- Memory Build Done, time={total_memory_time:.2f}s ---")

        if memory_build_failed:
            raise RuntimeError(
                f"Memory build failed for context {unit.context_id}, unit {unit.unit_id}; "
                "queries were not run against an incomplete memory state."
            )

        total_query_time = 0.0
        self._log(f"    --- Query Evaluation Start ---")

        query_count = len(unit.queries_to_evaluate)
        memory_time_per_query = total_memory_time / query_count if query_count > 0 else 0.0

        pending_query_count = sum(
            1
            for query in unit.queries_to_evaluate
            if not (
                self._checkpoint_manager
                and self._checkpoint_manager.is_query_completed(query.query_id)
            )
            and not self._is_deferred_judge_query(query.query_id)
        )
        if self._can_batch_queries(pending_query_count):
            query_results = self._evaluate_batch_queries(unit, memory_time_per_query)
            for result in query_results:
                results.append(result)
                total_query_time += result.query_time
                status = "✓" if result.is_correct else "✗"
                self._log(f"    [{status}] {result.query_id} ({result.query_type}): {result.score:.2f}")
                self._log_query_timing(result)
                if self._checkpoint_manager:
                    self._checkpoint_manager.mark_query_completed(
                        result.query_id,
                        result.to_dict(),
                    )
        else:
            if self.batch_api and not self.dry_run and not self._batch_fallback_logged:
                self._log(
                    "Vertex batch API is unavailable or this stage is too small; "
                    "using real-time generation for dependent requests.",
                    level="WARNING",
                )
                self._batch_fallback_logged = True

            for query in unit.queries_to_evaluate:
                if self._checkpoint_manager and self._checkpoint_manager.is_query_completed(query.query_id):
                    self._log(f"    [Skip] {query.query_id} (completed)")
                    continue
                if self._is_deferred_judge_query(query.query_id):
                    self._log(f"    [Skip] {query.query_id} (awaiting saved Vertex judge result)")
                    continue

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
                self._log_query_timing(result)

                if self._checkpoint_manager:
                    self._checkpoint_manager.mark_query_completed(
                        query.query_id,
                        result.to_dict(),
                    )

        self._log(f"    --- Query Evaluation Done, time={total_query_time:.2f}s ---")

        return results

    def _can_batch_queries(self, request_count: int) -> bool:
        return bool(
            should_use_batch(request_count)
            and self.batch_api
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
                manifest_path=self._batch_manifest_path(
                    "medmemorybench_judge_batch_manifest", judge_client.model
                ),
                wait=self.batch_wait,
                config_hash=self._batch_manifest_config_hash(),
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
            if self._checkpoint_manager and self._checkpoint_manager.is_query_completed(query.query_id):
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
                prepared = self.agent_manager.prepare_batch_query(
                    formatted_question,
                    query_id=query.query_id,
                    query_type=query.query_type,
                    context_id=unit.context_id,
                    batch_request_time=batch_request_time,
                )

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

        self._log(
            f"    [Vertex Batch] Stage '{stage}': local preparation complete; "
            f"submitting {len(requests):,} final-answer request(s)."
        )
        responses = batch_client.run_stage(stage, requests)
        results: List[MetricResult] = []
        for request_id, (query, prepared) in prepared_by_id.items():
            batch_response = responses.get(request_id)
            if batch_response is None or batch_response.status:
                error = batch_response.status if batch_response else "No output row returned"
                results.append(self._api_error_result(query, f"Vertex batch request failed: {error}"))
                continue
            response = self.agent_manager.finalize_batch_query(
                prepared,
                batch_response.content,
                input_tokens=batch_response.input_tokens,
                output_tokens=batch_response.output_tokens,
            )
            result = self._score_agent_response(
                query,
                response,
                context_id=unit.context_id,
                memory_construction_time=memory_time_per_query,
            )
            if result is not None:
                results.append(result)
        return results

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

        answers_data = None
        if isinstance(query, MedQuery):
            answers_data = query.answers_data

        formatted_question = self.prompt_manager.format_query(
            question=query.question,
            query_type=query.query_type,
        )

        try:
            response = self.agent_manager.send_message(
                message=formatted_question,
                memorizing=False,
                query_id=query.query_id,
                query_type=query.query_type,
                context_id=context_id,
            )
        except LLMRetryExhaustedError as e:
            # Log the error and return a failed result instead of crashing
            self._log(
                f"    [API ERROR] {query.query_id}: API call failed after retries, "
                f"skipping this query. Error: {e}",
                level="ERROR"
            )
            return MetricResult(
                query_id=query.query_id,
                query_type=query.query_type,
                score=0.0,
                is_correct=False,
                model_output="[API_ERROR] Connection failed after retries",
                expected_answer=", ".join(query.get_correct_answers()),
                question=query.question,
                details={
                    "api_error": True,
                    "error_type": "LLMRetryExhaustedError",
                    "error_message": str(e),
                },
            )

        return self._score_agent_response(
            query,
            response,
            context_id=context_id,
            memory_construction_time=memory_time_per_query,
        )

    def _api_error_result(self, query, error_message: str) -> MetricResult:
        return MetricResult(
            query_id=query.query_id,
            query_type=query.query_type,
            score=0.0,
            is_correct=False,
            model_output="[API_ERROR] Connection failed after retries",
            expected_answer=", ".join(query.get_correct_answers()),
            question=query.question,
            details={
                "api_error": True,
                "error_type": "LLMRetryExhaustedError",
                "error_message": error_message,
            },
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
            agent_extra = response.get("extra", {})
        elif hasattr(response, "output"):
            model_output = response.output
            query_time = getattr(response, "query_time", 0.0)
            retrieved_memories = getattr(response, "retrieved_memories", [])
            retrieved_count = getattr(response, "retrieved_count", 0)
            agent_extra = getattr(response, "extra", {})
        else:
            model_output = str(response)
            query_time = 0.0
            retrieved_memories = []
            retrieved_count = 0
            agent_extra = {}

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
                    "agent_extra": agent_extra,
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
        if agent_extra:
            result.details["agent_telemetry"] = agent_extra

        return result

    def _log_query_timing(self, result: MetricResult) -> None:
        """Expose per-stage query latency without changing evaluation semantics."""
        telemetry = result.details.get("agent_telemetry", {})
        timing = telemetry.get("query_latency")
        if not isinstance(timing, dict):
            return
        stages = []
        for name in ("fast_gate", "planner", "slot_validation", "replan", "answer"):
            value = timing.get(name)
            if value is not None and float(value or 0.0) > 0:
                stages.append(f"{name}={float(value):.2f}s")
        if stages:
            self._log(
                f"      [Timing] total={result.query_time:.2f}s "
                + " ".join(stages)
            )

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
                result = self.metrics_calculator.compute(
                    query_id=prepared["query_id"],
                    query_type=prepared["query_type"],
                    model_output=prepared["model_output"],
                    expected_answers=item["expected_answers"],
                    question=prepared["question"],
                    **item["metric_kwargs"],
                )
                result.query_time = item["query_time"]
                result.memory_construction_time = item["memory_construction_time"]
                result.retrieved_memories = item["retrieved_memories"]
                result.retrieved_count = item["retrieved_count"]
                if item.get("agent_extra"):
                    result.details["agent_telemetry"] = item["agent_extra"]
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
                    temperature=1.0,
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
            result_text = response.content if response is not None and not response.status else ""
            result = self.metrics_calculator.finalize_batch(item["prepared_metric"], result_text)
            result.query_time = item["query_time"]
            result.memory_construction_time = item["memory_construction_time"]
            result.retrieved_memories = item["retrieved_memories"]
            result.retrieved_count = item["retrieved_count"]
            if item.get("agent_extra"):
                result.details["agent_telemetry"] = item["agent_extra"]
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

        llm_usage = get_usage_tracker().get_stats()

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
                "dry_run": self.dry_run,
            },
            metadata={
                "evaluation_mode": self.dataset_config.evaluation_mode,
                "evaluation_interval": self.dataset_config.evaluation_interval,
                "total_personas": len(self.result_collector.get_context_ids()),
                "memory_build_summary": memory_build_summary,
                "llm_usage": llm_usage,
            }
        )

        result_path, memory_build_path, query_answer_path = self.result_collector.save_reports(
            report=report,
            output_dir=self.output_dir,
            memory_build_logs=self._memory_build_logs,
        )

        self._log(f"Results saved to: {result_path}")
        self._log(f"Memory build details saved to: {memory_build_path}")
        self._log(f"Query answer details saved to: {query_answer_path}")

        return report

    def _summarize_memory_builds(self) -> Dict[str, Any]:
        if not self._memory_build_logs:
            return {"total_builds": 0}

        total_units = len(self._memory_build_logs)

        total_sessions = sum(
            log.get("session_count", 0)
            for log in self._memory_build_logs
        )

        # New structure: sum time from each unit's total_time
        total_time = sum(
            log.get("total_time", 0)
            for log in self._memory_build_logs
        )

        # Count total passages from session_builds
        total_passages = sum(
            log.get("total_passages", 0)
            for log in self._memory_build_logs
        )

        # Count memory entries from session builds
        total_memory_entries = 0
        methods = {}

        for log in self._memory_build_logs:
            session_builds = log.get("session_builds", [])
            for sb in session_builds:
                build_result = sb.get("build_result", {})
                total_memory_entries += len(build_result.get("memory_entries", []))

                method = build_result.get("method", "unknown")
                if method not in methods:
                    methods[method] = {"count": 0, "time_cost": 0, "passages": 0}
                methods[method]["count"] += 1
                methods[method]["time_cost"] += build_result.get("time_cost", 0)
                methods[method]["passages"] += len(build_result.get("all_passages", []) or build_result.get("memory_entries", []))

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
    rebuild_memory: bool = False,
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
        rebuild_memory=rebuild_memory,
        batch_api=batch_api,
        batch_gcs_uri=batch_gcs_uri,
        batch_wait=batch_wait,
    )
    return evaluator.evaluate()
