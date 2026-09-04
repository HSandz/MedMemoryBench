"""LoCoMo evaluation module."""

import hashlib
import json
import os
import re
import subprocess
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import logging

from src.config import MethodConfig, DatasetConfig, PROJECT_ROOT, get_api_config
from src.evaluator import register_evaluator
from src.agent import AgentManager
from src.result import EvaluationReport, ResultCollector
from benchmarks.locomo.dataset import LoCoMoDataset, LoCoMoQuery, LoCoMoSession
from benchmarks.base import EvaluationUnit
from methods.base import MemoryBuildResult
from metrics import MetricsCalculator, MetricsAggregator, MetricResult
from metrics.retrieval_quality import compute_session_retrieval_quality
from utils.templates import get_prompt_manager
from utils.logger import truncate_error_message
from utils.batch_client import create_batch_client
from utils.llm_client import get_usage_tracker
from utils.vertex_batch import (
    BatchChatRequest,
    VertexBatchClient,
    VertexBatchError,
    make_request_id,
    PREPARED_QUERY_METADATA_KEY,
    restore_prepared_query,
    scoped_manifest_path,
    snapshot_prepared_query,
)
from benchmarks.medmemorybench.checkpoint import (
    compute_build_config_hash,
    compute_memory_query_compatibility_hash,
    compute_query_config_hash,
    is_manifest_build_compatible,
    is_manifest_query_compatible,
)


# Default chunk size for memory injection (in characters)
# ~32K chars ≈ 8K tokens, safe for GPT-5.1/Qwen3-235B (128K context)
DEFAULT_MEMORY_CHUNK_SIZE = 32000


class LoCoMoEvaluator:

    def __init__(
        self,
        method_config: MethodConfig,
        dataset_config: DatasetConfig,
        output_dir: Path,
        dry_run: bool = False,
        verbose: bool = True,
        logger: Optional[logging.Logger] = None,
        resume: bool = False,
        execution_stage: str = "all",
        memory_run: Optional[str] = None,
        memory_source_run_dir: Optional[Path] = None,
        run_scoped_output: bool = False,
        batch_api: bool = False,
        batch_gcs_uri: Optional[str] = None,
        batch_wait: bool = False,
        workers: int = 1,
    ):
        self.method_config = method_config
        self.dataset_config = dataset_config
        self.output_dir = output_dir
        self.dry_run = dry_run
        self.verbose = verbose
        self.logger = logger
        self.resume = resume
        if execution_stage not in {"all", "memory", "query"}:
            raise ValueError(f"Unsupported execution stage: {execution_stage}")
        self.execution_stage = execution_stage
        self.memory_run = memory_run
        self.memory_source_run_dir = (
            Path(memory_source_run_dir).resolve()
            if memory_source_run_dir is not None else None
        )
        self.run_scoped_output = run_scoped_output
        self.batch_api = batch_api
        self.batch_gcs_uri = batch_gcs_uri
        self.batch_wait = batch_wait
        if workers < 1:
            raise ValueError("workers must be at least 1")
        self.workers = workers

        self.prompt_manager = get_prompt_manager(
            dataset=dataset_config.dataset_name,
            method=method_config.method_name,
        )

        self.agent_manager: Optional[AgentManager] = None
        self.dataset: Optional[LoCoMoDataset] = None

        api_config = get_api_config()
        self.metrics_calculator = MetricsCalculator(
            dataset="locomo",
            judge_model=api_config.judge_model or None,
            judge_api_key=api_config.judge_api_key or None,
            judge_base_url=api_config.judge_base_url or None,
            judge_temperature=getattr(api_config, "judge_temperature", 1.0),
            judge_reasoning_effort=getattr(api_config, "judge_reasoning_effort", None),
            judge_client_max_tokens=getattr(api_config, "judge_client_max_tokens", 10000),
            judge_max_tokens=getattr(api_config, "judge_max_tokens", 500),
            judge_mcd_max_tokens=getattr(api_config, "judge_mcd_max_tokens", 2000),
        )
        self.aggregator = MetricsAggregator()
        self.result_collector = ResultCollector()

        self._memory_build_logs: List[Dict[str, Any]] = []
        self._batch_client: Optional[VertexBatchClient] = None
        self._batch_fallback_logged = False
        self._pending_batch_queries: List[Dict[str, Any]] = []
        self._memory_snapshot_manifest: Optional[Dict[str, Any]] = None
        self._memory_snapshot_dir_path: Optional[Path] = None
        self._query_checkpoint: Dict[str, Any] = {"results": {}}
        self._batch_retrieval_preparation_wall_time = 0.0

        # Memory chunk configuration
        # Get from dataset config or use default
        eval_config = dataset_config.raw_config.get("evaluation", {})
        self.memory_chunk_size = eval_config.get("memory_chunk_size", DEFAULT_MEMORY_CHUNK_SIZE)

    def _batch_config_hash(self) -> str:
        """Bind a resumable batch manifest to the evaluated configuration."""
        payload = {
            "method": self.method_config.raw_config,
            "dataset": self.dataset_config.raw_config,
        }
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]

    def _log(self, message: str, level: str = "INFO") -> None:
        if level.upper() in {"ERROR", "WARNING"}:
            message = truncate_error_message(message)
        if self.verbose:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] [{level}] {message}")
        if self.logger:
            self.logger.info(message)

    @staticmethod
    def _git_metadata() -> Dict[str, Any]:
        """Record local repository identity without making Git a runtime dependency."""
        def run(*args: str) -> Optional[str]:
            try:
                value = subprocess.run(
                    ["git", *args], cwd=PROJECT_ROOT, text=True,
                    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=True,
                ).stdout.strip()
                return value or None
            except (OSError, subprocess.CalledProcessError):
                return None
        status = run("status", "--porcelain")
        return {
            "git_commit_sha": run("rev-parse", "HEAD"),
            "git_branch": run("branch", "--show-current"),
            "git_dirty": None if status is None else bool(status),
        }

    def _init_dataset(self) -> None:
        self._log(f"Loading dataset: {self.dataset_config.dataset_name}")
        data_dir = PROJECT_ROOT / self.dataset_config.data_root_dir

        self.dataset = LoCoMoDataset(
            data_dir=data_dir,
            config={
                "data_file": self.dataset_config.raw_config.get("data", {}).get("data_file", "locomo10.json"),
                "sample_ids": self.dataset_config.raw_config.get("evaluation", {}).get("sample_ids"),
                "max_samples": self.dataset_config.raw_config.get("evaluation", {}).get("max_samples"),
                "category_filter": self.dataset_config.raw_config.get("evaluation", {}).get("category_filter"),
                "include_images": self.dataset_config.raw_config.get("evaluation", {}).get("include_images", True),
            }
        )
        self.dataset.load()

        self._log(f"  Total Samples: {len(self.dataset.get_sample_ids())}")
        self._log(f"  Total Sessions: {self.dataset.get_total_sessions()}")
        self._log(f"  Total Queries: {self.dataset.get_total_queries()}")
        self._log(f"  Category Distribution: {self.dataset.get_category_distribution()}")
        self._log(f"  Memory Chunk Size: {self.memory_chunk_size:,} chars (~{self.memory_chunk_size // 4:,} tokens)")

    def _init_agent_for_context(self, context_id: Any, force_new: bool = True) -> None:
        if force_new or self.agent_manager is None:
            # Clean up old agent if exists
            if self.agent_manager is not None:
                try:
                    self.agent_manager.reset()
                except Exception as e:
                    self._log(
                        f"Warning: Failed to reset old agent: {truncate_error_message(e)}",
                        level="WARNING",
                    )

            self.agent_manager = AgentManager(
                method_config=self.method_config,
                dataset_config=self.dataset_config,
                batch_api=self.batch_api,
                batch_gcs_uri=self.batch_gcs_uri,
                batch_wait=self.batch_wait,
                workers=self.workers,
                batch_manifest_dir=self.output_dir / "batch",
                batch_config_hash=self._batch_config_hash(),
                batch_progress_callback=self._log,
            )

        self.agent_manager.set_context_id(context_id)

    def _supports_event_state_snapshots(self) -> bool:
        return bool(
            self.method_config.method_name.lower() == "event_state"
            and self.agent_manager
            and self.agent_manager.supports_memory_snapshots()
        )

    def _memory_snapshot_root(self) -> Path:
        if self.memory_source_run_dir is not None:
            return self.memory_source_run_dir / "memory"
        return self.output_dir / "memory"

    @staticmethod
    def _snapshot_integrity_hash(payload: Dict[str, Any]) -> str:
        content = dict(payload)
        content.pop("integrity_hash", None)
        encoded = json.dumps(content, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    @staticmethod
    def _write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        with temporary_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)

    def _snapshot_path(self, unit: EvaluationUnit) -> Path:
        if self._memory_snapshot_dir_path is None:
            raise RuntimeError("LoCoMo memory snapshot run has not been selected")
        safe_sample = str(unit.context_id).replace("/", "-").replace("\\", "-")
        return self._memory_snapshot_dir_path / f"sample_{unit.unit_id}_{safe_sample}.json"

    def _start_memory_snapshot_manifest(self, units: List[EvaluationUnit]) -> None:
        path = self._memory_snapshot_root()
        manifest_path = path / "manifest.json"
        if self.resume and manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(f"Cannot resume LoCoMo memory manifest: {exc}") from exc
            if (
                manifest.get("format") != "locomo.event_state_memory_manifest"
                or manifest.get("version") != 1
                or not is_manifest_build_compatible(
                    manifest, compute_build_config_hash(self.method_config, self.dataset_config), manifest_path
                )
            ):
                raise ValueError("LoCoMo memory manifest is incompatible with the current Event-State build configuration")
            self._memory_snapshot_manifest = manifest
            self._memory_snapshot_dir_path = path
            return
        path.mkdir(parents=True, exist_ok=False)
        self._memory_snapshot_dir_path = path
        self._memory_snapshot_manifest = {
            "format": "locomo.event_state_memory_manifest",
            "version": 2,
            "build_id": str(uuid.uuid4()),
            "status": "building",
            "method_name": self.method_config.method_name,
            "model_name": self.method_config.model.name,
            "config_hash": compute_build_config_hash(self.method_config, self.dataset_config),
            "build_config_hash": compute_build_config_hash(self.method_config, self.dataset_config),
            "retrieval_config_hash": compute_query_config_hash(self.method_config, self.dataset_config),
            "retrieval_compatibility_hash": compute_memory_query_compatibility_hash(self.method_config, self.dataset_config),
            "sample_ids": [str(unit.context_id) for unit in units],
            "snapshots": [],
            "created_at": datetime.now().isoformat(),
            "completed_at": None,
        }
        self._write_json_atomic(path / "manifest.json", self._memory_snapshot_manifest)

    def _load_memory_snapshot_manifest(self, units: List[EvaluationUnit]) -> None:
        path = self._memory_snapshot_root()
        manifest_path = path / "manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise FileNotFoundError(f"Cannot read LoCoMo memory manifest: {manifest_path}") from exc
        if (
            manifest.get("format") != "locomo.event_state_memory_manifest"
            or manifest.get("version") not in {1, 2}
            or manifest.get("status") != "complete"
            or manifest.get("sample_ids") != [str(unit.context_id) for unit in units]
            or not is_manifest_query_compatible(manifest, self.method_config, self.dataset_config, manifest_path)
        ):
            raise ValueError("LoCoMo Event-State memory manifest is incomplete or incompatible with the query configuration")
        records = manifest.get("snapshots", [])
        if {str(record.get("sample_id")) for record in records} != set(manifest["sample_ids"]):
            raise ValueError("LoCoMo Event-State memory manifest has an incomplete sample snapshot list")
        self._memory_snapshot_manifest = manifest
        self._memory_snapshot_dir_path = path

    def _write_memory_snapshot(self, unit: EvaluationUnit, build_time: float, build_metrics: Dict[str, Any]) -> Path:
        path = self._snapshot_path(unit)
        payload = {
            "format": "locomo.event_state_memory_snapshot",
            "version": 2,
            "build_id": self._memory_snapshot_manifest["build_id"],
            "sample_id": str(unit.context_id),
            "unit_id": unit.unit_id,
            "session_ids": [session.session_id for session in unit.sessions_to_inject],
            "session_timestamps": [session.date_time for session in unit.sessions_to_inject],
            "memory_build_time": build_time,
            "memory_build_metrics": build_metrics,
            "created_at": datetime.now().isoformat(),
            "memory_state": self.agent_manager.export_memory_state(context_id=unit.context_id),
        }
        payload["integrity_hash"] = self._snapshot_integrity_hash(payload)
        self._write_json_atomic(path, payload)
        record = {
            "sample_id": str(unit.context_id), "unit_id": unit.unit_id,
            "path": path.name, "integrity_hash": payload["integrity_hash"],
            "session_count": len(unit.sessions_to_inject), "memory_build_time": build_time,
            "memory_build_metrics": build_metrics,
        }
        records = self._memory_snapshot_manifest["snapshots"]
        records[:] = [item for item in records if str(item.get("sample_id")) != str(unit.context_id)]
        records.append(record)
        records.sort(key=lambda item: item["unit_id"])
        self._write_json_atomic(self._memory_snapshot_dir_path / "manifest.json", self._memory_snapshot_manifest)
        return path

    def _read_memory_snapshot(self, unit: EvaluationUnit) -> Dict[str, Any]:
        record = next((item for item in self._memory_snapshot_manifest["snapshots"] if str(item.get("sample_id")) == str(unit.context_id)), None)
        if record is None:
            raise ValueError(f"No LoCoMo snapshot exists for sample {unit.context_id}")
        path = self._memory_snapshot_dir_path / record["path"]
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Cannot read LoCoMo snapshot {path}") from exc
        if (
            payload.get("format") != "locomo.event_state_memory_snapshot"
            or payload.get("version") not in {1, 2}
            or payload.get("build_id") != self._memory_snapshot_manifest.get("build_id")
            or str(payload.get("sample_id")) != str(unit.context_id)
            or payload.get("integrity_hash") != self._snapshot_integrity_hash(payload)
        ):
            raise ValueError(f"LoCoMo snapshot integrity or identity check failed: {path}")
        return payload

    def _complete_memory_snapshot_manifest(self) -> None:
        expected = self._memory_snapshot_manifest.get("sample_ids", [])
        actual = [str(item.get("sample_id")) for item in self._memory_snapshot_manifest.get("snapshots", [])]
        if sorted(actual) != sorted(expected):
            raise RuntimeError("LoCoMo Event-State memory build is incomplete")
        self._memory_snapshot_manifest["status"] = "complete"
        self._memory_snapshot_manifest["completed_at"] = datetime.now().isoformat()
        self._write_json_atomic(self._memory_snapshot_dir_path / "manifest.json", self._memory_snapshot_manifest)

    def _query_checkpoint_path(self) -> Path:
        return self.output_dir / "locomo_query_checkpoint.json"

    def _load_query_checkpoint(self) -> None:
        if not self.resume or not self._query_checkpoint_path().exists():
            return
        try:
            payload = json.loads(self._query_checkpoint_path().read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("Cannot read the LoCoMo query resume checkpoint") from exc
        if (
            payload.get("format") != "locomo.query_checkpoint"
            or payload.get("version") != 1
            or payload.get("query_config_hash") != compute_query_config_hash(self.method_config, self.dataset_config)
            or payload.get("integrity_hash") != self._snapshot_integrity_hash(payload)
            or not isinstance(payload.get("results"), dict)
        ):
            raise ValueError("LoCoMo query resume checkpoint is incompatible or corrupt")
        self._query_checkpoint = payload

    def _completed_query_results(self, unit: EvaluationUnit) -> List[MetricResult]:
        saved = self._query_checkpoint.get("results", {}).get(str(unit.context_id), {})
        if not isinstance(saved, dict):
            return []
        return [MetricResult(**saved[query.query_id]) for query in unit.queries_to_evaluate if query.query_id in saved]

    def _pending_query_unit(self, unit: EvaluationUnit) -> EvaluationUnit:
        saved = self._query_checkpoint.get("results", {}).get(str(unit.context_id), {})
        pending = [query for query in unit.queries_to_evaluate if query.query_id not in saved]
        return replace(unit, queries_to_evaluate=pending)

    def _record_completed_query(self, sample_id: Any, result: MetricResult) -> None:
        results = self._query_checkpoint.setdefault("results", {}).setdefault(str(sample_id), {})
        if result.query_id in results:
            return
        results[result.query_id] = result.to_dict()
        self._query_checkpoint.update({
            "format": "locomo.query_checkpoint",
            "version": 1,
            "query_config_hash": compute_query_config_hash(self.method_config, self.dataset_config),
        })
        self._query_checkpoint["integrity_hash"] = self._snapshot_integrity_hash(self._query_checkpoint)
        self._write_json_atomic(self._query_checkpoint_path(), self._query_checkpoint)

    def evaluate(self) -> EvaluationReport:
        start_time = datetime.now()

        get_usage_tracker().reset()

        self._init_dataset()
        self._load_query_checkpoint()
        units = list(self.dataset.get_evaluation_units())
        if self.execution_stage != "all" and self.method_config.method_name.lower() != "event_state":
            raise ValueError("LoCoMo --stage memory/query requires Event-State memory snapshots")
        if self.method_config.method_name.lower() == "event_state" and not self.dry_run:
            if self.execution_stage == "query":
                self._load_memory_snapshot_manifest(units)
            else:
                self._start_memory_snapshot_manifest(units)

        self._run_evaluation_loop(units)
        if (
            self.method_config.method_name.lower() == "event_state"
            and not self.dry_run
            and self.execution_stage != "query"
        ):
            self._complete_memory_snapshot_manifest()

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        report = self._generate_report(start_time, end_time, duration)

        return report

    def _run_evaluation_loop(self, units: List[EvaluationUnit]) -> None:
        if self.execution_stage == "query":
            self._run_snapshot_queries(units)
            return
        for unit in units:
            sample_id = unit.context_id

            self._log(f"\n{'='*60}")
            self._log(f"Sample: {sample_id}")
            self._log(f"  Sessions: {len(unit.sessions_to_inject)}")
            self._log(f"  Queries: {len(unit.queries_to_evaluate)}")

            for result in self._completed_query_results(unit):
                self.aggregator.add_result(result)
                self.result_collector.add_result(result, sample_id)
            unit = self._pending_query_unit(unit)

            if not self.dry_run:
                self._init_agent_for_context(context_id=sample_id, force_new=True)

            unit_results = self._evaluate_unit(unit)

            for result in unit_results:
                self.aggregator.add_result(result)
                self.result_collector.add_result(result, sample_id)
                self._record_completed_query(sample_id, result)

        for item in self._complete_combined_batch_queries():
            result = item["result"]
            sample_id = item["sample_id"]
            self.aggregator.add_result(result)
            self.result_collector.add_result(result, sample_id)
            self._record_completed_query(sample_id, result)

    def _run_snapshot_queries(self, units: List[EvaluationUnit]) -> None:
        """Restore each complete conversation into an isolated Event-State store."""
        if self.batch_api:
            # The run-wide manifest stays single-writer, while each prompt is
            # frozen immediately after restoring its own sample snapshot.
            for unit in units:
                for result in self._completed_query_results(unit):
                    self.aggregator.add_result(result)
                    self.result_collector.add_result(result, unit.context_id)
                unit = self._pending_query_unit(unit)
                self._init_agent_for_context(unit.context_id, force_new=True)
                self.agent_manager.import_memory_state(
                    self._read_memory_snapshot(unit)["memory_state"], context_id=unit.context_id
                )
                for result in self._evaluate_unit_queries(unit, memory_time=0.0):
                    self.aggregator.add_result(result)
                    self.result_collector.add_result(result, unit.context_id)
                    self._record_completed_query(unit.context_id, result)
            for item in self._complete_combined_batch_queries():
                self.aggregator.add_result(item["result"])
                self.result_collector.add_result(item["result"], item["sample_id"])
                self._record_completed_query(item["sample_id"], item["result"])
            return

        def query_sample(unit: EvaluationUnit) -> Tuple[EvaluationUnit, List[MetricResult]]:
            manager = AgentManager(
                method_config=self.method_config,
                dataset_config=self.dataset_config,
                batch_api=False,
                workers=self.workers,
            )
            manager.import_memory_state(
                self._read_memory_snapshot(unit)["memory_state"], context_id=unit.context_id
            )
            return unit, self._evaluate_realtime_queries(
                unit.queries_to_evaluate, unit.context_id, manager=manager
            )

        completed_prior = []
        pending_units = []
        for unit in units:
            completed_prior.extend((unit.context_id, result) for result in self._completed_query_results(unit))
            pending_units.append(self._pending_query_unit(unit))
        for sample_id, result in completed_prior:
            self.aggregator.add_result(result)
            self.result_collector.add_result(result, sample_id)

        if self.workers > 1 and len(pending_units) > 1:
            with ThreadPoolExecutor(max_workers=min(self.workers, len(pending_units))) as executor:
                completed = list(executor.map(query_sample, pending_units))
        else:
            completed = [query_sample(unit) for unit in pending_units]
        for unit, results in completed:
            for result in results:
                result.memory_construction_time = 0.0
                self.aggregator.add_result(result)
                self.result_collector.add_result(result, unit.context_id)
                self._record_completed_query(unit.context_id, result)

    def _split_sessions_into_chunks(
        self,
        sessions: List[LoCoMoSession],
    ) -> List[Tuple[List[LoCoMoSession], str]]:
        """Split sessions into chunks based on character size limit.

        Returns:
            List of tuples: (sessions_in_chunk, combined_memory_text)
        """
        chunks = []
        current_chunk_sessions = []
        current_chunk_texts = []
        current_chunk_size = 0

        for session in sessions:
            memory_text = session.to_memory_text()
            text_size = len(memory_text)

            # If adding this session exceeds the limit and we have content, start a new chunk
            if current_chunk_size + text_size > self.memory_chunk_size and current_chunk_sessions:
                # Save current chunk
                combined_text = "\n\n".join(current_chunk_texts)
                chunks.append((current_chunk_sessions.copy(), combined_text))

                # Start new chunk
                current_chunk_sessions = []
                current_chunk_texts = []
                current_chunk_size = 0

            # Add session to current chunk
            current_chunk_sessions.append(session)
            current_chunk_texts.append(memory_text)
            current_chunk_size += text_size

        # Don't forget the last chunk
        if current_chunk_sessions:
            combined_text = "\n\n".join(current_chunk_texts)
            chunks.append((current_chunk_sessions, combined_text))

        return chunks

    def _evaluate_unit(self, unit: EvaluationUnit) -> List[MetricResult]:
        if self.method_config.method_name.lower() == "event_state":
            return self._evaluate_event_state_unit(unit)
        results = []

        self._log(f"  --- Memory Build Phase ---")

        if self.dry_run:
            self._log(f"  [Dry Run] Skipping memory build")
            total_memory_time = 0.0
            chunk_build_results = []
        else:
            # Split sessions into chunks
            chunks = self._split_sessions_into_chunks(unit.sessions_to_inject)
            total_chunks = len(chunks)

            self._log(f"  Split into {total_chunks} chunks (chunk_size={self.memory_chunk_size:,} chars)")

            total_memory_time = 0.0
            chunk_build_results = []
            all_session_ids = []

            speakers = unit.metadata.get("speaker_a", "A") + " and " + unit.metadata.get("speaker_b", "B")

            for chunk_idx, (chunk_sessions, chunk_text) in enumerate(chunks):
                chunk_session_ids = [s.session_id for s in chunk_sessions]
                all_session_ids.extend(chunk_session_ids)

                chunk_chars = len(chunk_text)
                chunk_tokens_est = chunk_chars // 4

                self._log(f"    [Chunk {chunk_idx + 1}/{total_chunks}] "
                         f"Sessions: {chunk_session_ids[0]}-{chunk_session_ids[-1]} "
                         f"({len(chunk_sessions)} sessions, ~{chunk_tokens_est:,} tokens)")

                # Format the chunk text with prompt template
                formatted_text = self.prompt_manager.format_memorize(
                    context=chunk_text,
                    timestamp=speakers,
                )

                chunk_start_time = time.time()

                try:
                    memory_items = []
                    for session_index, session in enumerate(chunk_sessions):
                        for turn in session.dialogues:
                            memory_item = {
                                "speaker": turn.get("speaker", "Unknown"),
                                "content": turn.get("text", ""),
                                "blip_caption": turn.get("blip_caption", ""),
                                "timestamp": session.date_time,
                            }
                            if self.method_config.method_name in {"amem_test", "event_state"}:
                                memory_item.update({
                                    "source_session_id": session.session_id,
                                    "source_session_index": session_index if self.method_config.method_name == "event_state" else session.session_id,
                                    "source_turn_id": turn.get("dia_id"),
                                    "source_event_id": session.metadata.get("session_key"),
                                })
                            memory_items.append(memory_item)
                    memory_result = self.agent_manager.send_message(
                        message=formatted_text,
                        memorizing=True,
                        context_id=unit.context_id,
                        memory_items=memory_items,
                    )

                    chunk_time = time.time() - chunk_start_time
                    total_memory_time += chunk_time

                    if isinstance(memory_result, MemoryBuildResult):
                        # Log brief info
                        entries_count = len(memory_result.memory_entries) if memory_result.memory_entries else 0
                        chunk_count = memory_result.chunk_count or 0

                        self._log(f"      → Stored: entries={entries_count}, chunks={chunk_count}, "
                                 f"time={chunk_time:.2f}s")

                        # Store detailed result for this chunk
                        chunk_build_results.append({
                            "chunk_index": chunk_idx,
                            "session_ids": chunk_session_ids,
                            "session_count": len(chunk_sessions),
                            "input_chars": chunk_chars,
                            "input_tokens_est": chunk_tokens_est,
                            "time_cost": chunk_time,
                            "build_result": memory_result.to_dict(),
                        })
                    else:
                        # Fallback for non-standard result
                        chunk_build_results.append({
                            "chunk_index": chunk_idx,
                            "session_ids": chunk_session_ids,
                            "session_count": len(chunk_sessions),
                            "input_chars": chunk_chars,
                            "input_tokens_est": chunk_tokens_est,
                            "time_cost": chunk_time,
                            "build_result": {"raw_result": str(memory_result)},
                        })

                except Exception as e:
                    chunk_time = time.time() - chunk_start_time
                    self._log(
                        f"      [ERROR] Chunk {chunk_idx + 1} failed: {truncate_error_message(e)}",
                        level="ERROR",
                    )

                    chunk_build_results.append({
                        "chunk_index": chunk_idx,
                        "session_ids": chunk_session_ids,
                        "session_count": len(chunk_sessions),
                        "input_chars": chunk_chars,
                        "input_tokens_est": chunk_tokens_est,
                        "time_cost": chunk_time,
                        "error": truncate_error_message(e),
                    })

            # Calculate summary stats
            total_entries = sum(
                len(r.get("build_result", {}).get("memory_entries", []))
                for r in chunk_build_results if "build_result" in r
            )
            total_stored_chunks = sum(
                r.get("build_result", {}).get("chunk_count", 0)
                for r in chunk_build_results if "build_result" in r
            )

            self._log(f"  [Summary] Total: {total_chunks} chunks, "
                     f"{len(all_session_ids)} sessions, "
                     f"entries={total_entries}, stored_chunks={total_stored_chunks}")

            # Store all chunk build results
            self._memory_build_logs.append({
                "unit_id": unit.unit_id,
                "context_id": unit.context_id,
                "session_ids": all_session_ids,
                "session_count": len(unit.sessions_to_inject),
                "chunk_count": total_chunks,
                "chunk_size_config": self.memory_chunk_size,
                "total_time": total_memory_time,
                "total_entries": total_entries,
                "total_stored_chunks": total_stored_chunks,
                "chunk_builds": chunk_build_results,
            })

        self._log(f"  Memory Build Done, total_time={total_memory_time:.2f}s")

        return self._evaluate_unit_queries(unit, memory_time=total_memory_time)

    def _event_state_memory_items(self, unit: EvaluationUnit) -> List[Dict[str, Any]]:
        """Keep LoCoMo's source sessions intact, with sample-global chronology."""
        items: List[Dict[str, Any]] = []
        for session_index, session in enumerate(unit.sessions_to_inject):
            for turn in session.dialogues:
                items.append({
                    "speaker": turn.get("speaker", "Unknown"),
                    "content": turn.get("text", ""),
                    "blip_caption": turn.get("blip_caption", ""),
                    # The adapter owns LoCoMo-specific timestamp parsing; the
                    # memory method receives a generic canonical record time.
                    "timestamp": session.metadata.get("recorded_at", session.date_time),
                    "recorded_at_raw": session.metadata.get("recorded_at_raw"),
                    "source_session_id": session.session_id,
                    "source_session_index": session_index,
                    "source_turn_id": turn.get("dia_id"),
                    "source_event_id": session.metadata.get("session_key"),
                })
        return items

    @staticmethod
    def _event_state_store_diagnostics(
        state: Dict[str, Any], input_session_count: int,
    ) -> Dict[str, Any]:
        episodes = state.get("episodes", []) if isinstance(state, dict) else []
        claims = state.get("claims", []) if isinstance(state, dict) else []
        source_ids = [item.get("source_session_id") for item in episodes if item.get("source_session_id") is not None]
        episode_count = len(episodes)
        unique_source_count = len(set(source_ids))
        duplicate_source_count = len(source_ids) - unique_source_count
        return {
            "input_session_count": input_session_count,
            "unique_source_session_id_count": unique_source_count,
            "final_episode_count": episode_count,
            "final_claim_count": len(claims),
            "final_memory_object_count": episode_count + len(claims),
            "duplicate_episode_source_id_count": duplicate_source_count,
            "source_identity_integrity": {
                "expected": input_session_count,
                "actual_unique_source_session_ids": unique_source_count,
                "actual_episodes": episode_count,
                "valid": (
                    input_session_count == unique_source_count == episode_count
                ),
            },
        }

    def _evaluate_event_state_unit(self, unit: EvaluationUnit) -> List[MetricResult]:
        if self.dry_run:
            return [] if self.execution_stage == "memory" else self._evaluate_unit_queries(unit, memory_time=0.0)
        existing = None
        if self.resume:
            try:
                existing = self._read_memory_snapshot(unit)
            except ValueError:
                existing = None
        if existing is not None:
            self.agent_manager.import_memory_state(existing["memory_state"], context_id=unit.context_id)
            build_time = float(existing.get("memory_build_time", 0.0) or 0.0)
            snapshot_state = existing.get("memory_state", {})
            build_metrics = dict(existing.get("memory_build_metrics") or {})
            build_metrics.update(self._event_state_store_diagnostics(
                snapshot_state, len(unit.sessions_to_inject)
            ))
            self._memory_build_logs.append({
                "unit_id": unit.unit_id, "context_id": unit.context_id,
                "session_ids": existing.get("session_ids", []),
                "session_count": len(unit.sessions_to_inject), "chunk_count": 0,
                "total_time": build_time, "inserted_record_count": 0,
                "total_stored_chunks": 0, "build_metrics": build_metrics,
                "final_store": self._event_state_store_diagnostics(
                    snapshot_state, len(unit.sessions_to_inject)
                ), "restored_from_snapshot": True,
            })
        else:
            started = time.time()
            items = self._event_state_memory_items(unit)
            prepared = self.agent_manager.prepare_memory_sessions(
                "", context_id=unit.context_id, memory_items=items
            )
            # Event-State preparation can run concurrently internally; this
            # commit is deliberately single-threaded and source ordered.
            memory_result = self.agent_manager.commit_prepared_memory(
                prepared, context_id=unit.context_id
            )
            build_time = time.time() - started
            if isinstance(memory_result, MemoryBuildResult):
                # Keep compact build telemetry, not a second copy of raw memory.
                build_metrics = {
                    key: value for key, value in memory_result.to_dict().items()
                    if key not in {
                        "memory_entries", "all_passages", "input_content",
                        "stored_content", "extraction_result",
                    }
                }
            else:
                build_metrics = {"raw_result": str(memory_result)}
            final_store = self._event_state_store_diagnostics(
                self.agent_manager.export_memory_state(context_id=unit.context_id),
                len(unit.sessions_to_inject),
            )
            build_metrics.update(final_store)
            if not final_store["source_identity_integrity"]["valid"]:
                raise RuntimeError(
                    "LoCoMo Event-State ingestion lost or duplicated source sessions: "
                    f"{final_store['source_identity_integrity']}"
                )
            self._memory_build_logs.append({
                "unit_id": unit.unit_id, "context_id": unit.context_id,
                "session_ids": [session.session_id for session in unit.sessions_to_inject],
                "session_count": len(unit.sessions_to_inject), "chunk_count": 0,
                "total_time": build_time,
                "inserted_record_count": len(getattr(memory_result, "memory_entries", []) or []),
                "total_stored_chunks": 0, "build_metrics": build_metrics,
                "final_store": final_store, "staged_session_count": len(prepared),
            })
            self._write_memory_snapshot(unit, build_time, build_metrics)
        if self.execution_stage == "memory":
            return []
        return self._evaluate_unit_queries(unit, memory_time=build_time)

    def _evaluate_unit_queries(self, unit: EvaluationUnit, memory_time: float) -> List[MetricResult]:
        self._log("  --- Query Evaluation Phase ---")
        query_count = len(unit.queries_to_evaluate)
        memory_time_per_query = memory_time / query_count if query_count else 0.0
        if unit.queries_to_evaluate and self._supports_batch_queries():
            batch_client = self._get_batch_client()
            if not batch_client.has_stage("query-final") and batch_client.has_stage(f"query-unit-{unit.unit_id}"):
                query_results = self._evaluate_batch_queries(unit)
            else:
                self._prepare_combined_batch_queries(unit, memory_time_per_query)
                query_results = []
        else:
            if self.batch_api and not self.dry_run and not self._batch_fallback_logged:
                self._log("Batch API is unavailable for this adapter; using real-time final-answer generation.", level="WARNING")
                self._batch_fallback_logged = True
            query_results = self._evaluate_realtime_queries(unit.queries_to_evaluate, unit.context_id)
        for result in query_results:
            result.memory_construction_time = memory_time_per_query
            self._log(f"    [{'✓' if result.is_correct else '✗'}] {result.query_id} ({result.query_type}): {result.score:.2f}")
        return query_results

    def _supports_batch_queries(self) -> bool:
        return bool(
            self.batch_api
            and self.agent_manager
            and self.agent_manager.supports_batch_queries()
        )

    def _evaluate_realtime_queries(
        self,
        queries: List[LoCoMoQuery],
        context_id: Any,
        manager: Optional[AgentManager] = None,
    ) -> List[MetricResult]:
        """Run independent real-time query evaluations with bounded concurrency."""
        explicit_manager = manager is not None
        manager = manager or getattr(self, "agent_manager", None)
        if len(queries) < 2 or self.workers == 1:
            if not explicit_manager:
                return [self._evaluate_query(query, context_id) for query in queries]
            return [self._evaluate_query(query, context_id, manager=manager) for query in queries]

        worker_count = min(self.workers, len(queries))
        self._log(f"  [Workers] Running {len(queries):,} real-time queries with {worker_count} workers.")
        if getattr(getattr(self, "method_config", None), "method_name", "").lower() == "event_state":
            # Event-State retrieval is logically read-only, but separate stores
            # make that guarantee explicit and keep future adapter changes safe.
            state = manager.export_memory_state(context_id=context_id)

            def evaluate_isolated(query: LoCoMoQuery) -> MetricResult:
                isolated = AgentManager(
                    method_config=self.method_config,
                    dataset_config=self.dataset_config,
                    batch_api=False,
                    workers=1,
                )
                isolated.import_memory_state(state, context_id=context_id)
                return self._evaluate_query(query, context_id, manager=isolated)

            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                return list(executor.map(evaluate_isolated, queries))
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            # executor.map preserves dataset order for deterministic reports.
            if not explicit_manager:
                return list(executor.map(
                    lambda query: self._evaluate_query(query, context_id), queries
                ))
            return list(executor.map(
                # Every worker receives an isolated manager for snapshot-only
                # execution; legacy in-sample calls retain their current agent.
                lambda query: self._evaluate_query(query, context_id, manager=manager),
                queries,
            ))

    def _get_batch_client(self) -> VertexBatchClient:
        if self._batch_client is None:
            if self.agent_manager is None:
                raise VertexBatchError("Agent manager is not initialized for batch execution.")
            llm_client = self.agent_manager.get_batch_llm_client()
            if llm_client is None:
                raise VertexBatchError("This method does not expose a managed batch client.")
            self._batch_client = create_batch_client(
                llm_client,
                gcs_uri=self.batch_gcs_uri,
                manifest_path=scoped_manifest_path(
                    self.output_dir / "batch",
                    "locomo_batch_manifest",
                    model=llm_client.model,
                    config_hash=self._batch_config_hash(),
                ),
                wait=self.batch_wait,
                config_hash=self._batch_config_hash(),
                progress_callback=self._log,
                vertex_batch_class=VertexBatchClient,
            )
        return self._batch_client

    def _evaluate_batch_queries(self, unit: EvaluationUnit) -> List[MetricResult]:
        prepared_by_id: Dict[str, tuple[LoCoMoQuery, Dict[str, Any]]] = {}
        requests: List[BatchChatRequest] = []
        stage = f"query-unit-{unit.unit_id}"
        batch_client = self._get_batch_client()
        for query in unit.queries_to_evaluate:
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
                prepared = self.agent_manager.prepare_batch_query(
                    formatted_question,
                    query_id=query.query_id,
                    context_id=unit.context_id,
                    batch_request_time=batch_request_time,
                    raw_question=query.question,
                    query_type=query.query_type,
                )

            if saved_request is not None:
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

        get_saved_requests = getattr(batch_client, "get_saved_requests", None)
        saved_requests = get_saved_requests(stage) if callable(get_saved_requests) else []
        submitted_requests = saved_requests or requests
        responses = batch_client.run_stage(stage, submitted_requests)
        results: List[MetricResult] = []
        for request_id, (query, prepared) in prepared_by_id.items():
            batch_response = responses.get(request_id)
            if batch_response is None or batch_response.status:
                error = batch_response.status if batch_response else "No output row returned"
                results.append(
                    self._api_error_result(
                        query, f"Batch request failed: {truncate_error_message(error)}"
                    )
                )
                continue
            response = self.agent_manager.finalize_batch_query(
                prepared,
                batch_response.content,
                input_tokens=batch_response.input_tokens,
                output_tokens=batch_response.output_tokens,
            )
            result = self._score_agent_response(query, response)
            result.details.setdefault("execution_usage", {})["answer"] = {
                "transport": "batch",
                "input_tokens": batch_response.input_tokens,
                "output_tokens": batch_response.output_tokens,
            }
            results.append(result)
        return results

    def _prepare_combined_batch_queries(
        self,
        unit: EvaluationUnit,
        memory_time_per_query: float,
    ) -> int:
        """Freeze a sample's prompts before its agent is reset for the next sample."""
        stage = "query-final"
        batch_client = self._get_batch_client()
        prepared_count = 0

        query_items = []
        for query in unit.queries_to_evaluate:
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
            query_items.append((query, request_id, saved_request, batch_request_time, prepared))

        event_state_snapshot = None
        if (
            getattr(self.method_config, "method_name", "").lower() == "event_state"
            and self.agent_manager is not None
        ):
            event_state_snapshot = self.agent_manager.export_memory_state(unit.context_id)

        def prepare_item(item):
            query, _, _, batch_request_time, prepared = item
            if prepared is not None:
                return prepared
            formatted_question = self.prompt_manager.format_query(
                question=query.question, query_type=query.query_type
            )
            manager = self.agent_manager
            if event_state_snapshot is not None:
                # Retrieval preparation is parallel but never shares an active
                # Event-State store with another query worker.
                manager = AgentManager(
                    method_config=self.method_config,
                    dataset_config=self.dataset_config,
                    batch_api=self.batch_api,
                    batch_gcs_uri=self.batch_gcs_uri,
                    batch_wait=self.batch_wait,
                    workers=1,
                )
                manager.import_memory_state(event_state_snapshot, context_id=unit.context_id)
            return manager.prepare_batch_query(
                formatted_question,
                query_id=query.query_id,
                context_id=unit.context_id,
                batch_request_time=batch_request_time,
                raw_question=query.question,
                query_type=query.query_type,
            )

        worker_count = getattr(self, "workers", 1)
        preparation_started = time.perf_counter()
        if worker_count > 1 and len(query_items) > 1:
            with ThreadPoolExecutor(max_workers=min(worker_count, len(query_items))) as executor:
                prepared_items = list(executor.map(prepare_item, query_items))
        else:
            prepared_items = [prepare_item(item) for item in query_items]
        self._batch_retrieval_preparation_wall_time = (
            getattr(self, "_batch_retrieval_preparation_wall_time", 0.0)
            + time.perf_counter() - preparation_started
        )

        for item, prepared in zip(query_items, prepared_items):
            query, request_id, saved_request, batch_request_time, _ = item

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
                "sample_id": unit.context_id,
                "memory_time_per_query": request.metadata.get(
                    "memory_time_per_query",
                    memory_time_per_query,
                ),
            })
            prepared_count += 1

        return prepared_count

    def _complete_combined_batch_queries(self) -> List[Dict[str, Any]]:
        """Submit all LoCoMo final-answer prompts as one Vertex stage."""
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
            "final-answer request(s) from all prepared samples."
        )
        responses = batch_client.run_stage(stage, requests)
        finalized: List[Dict[str, Any]] = []

        for item in self._pending_batch_queries:
            request = item["request"]
            query = item["query"]
            batch_response = responses.get(request.request_id)
            if batch_response is None or batch_response.status:
                error = batch_response.status if batch_response else "No output row returned"
                result = self._api_error_result(
                    query,
                    f"Batch request failed: {truncate_error_message(error)}",
                )
            else:
                response = self.agent_manager.finalize_batch_query(
                    item["prepared"],
                    batch_response.content,
                    input_tokens=batch_response.input_tokens,
                    output_tokens=batch_response.output_tokens,
                )
                result = self._score_agent_response(query, response)
                if hasattr(result, "details"):
                    result.details.setdefault("execution_usage", {})["answer"] = {
                        "transport": "batch",
                        "input_tokens": batch_response.input_tokens,
                        "output_tokens": batch_response.output_tokens,
                    }

            result.memory_construction_time = item["memory_time_per_query"]
            status = "✓" if result.is_correct else "✗"
            self._log(
                f"  [{status}] {result.query_id} ({result.query_type}): {result.score:.2f}"
            )
            finalized.append({
                "sample_id": item["sample_id"],
                "result": result,
            })

        self._pending_batch_queries = []
        return finalized

    def _evaluate_query(
        self,
        query: LoCoMoQuery,
        context_id: Any,
        manager: Optional[AgentManager] = None,
    ) -> MetricResult:
        if self.dry_run:
            return MetricResult(
                query_id=query.query_id,
                query_type=query.query_type,
                score=0.0,
                is_correct=False,
                model_output="[DRY RUN]",
                expected_answer=", ".join(query.get_correct_answers()),
                question=query.question,
                details={"dry_run": True, "category": query.category},
            )

        formatted_question = self.prompt_manager.format_query(
            question=query.question,
            query_type=query.query_type,
        )

        response = (manager or self.agent_manager).send_message(
            message=formatted_question,
            memorizing=False,
            context_id=context_id,
            raw_question=query.question,
            query_type=query.query_type,
        )

        return self._score_agent_response(query, response)

    def _api_error_result(self, query: LoCoMoQuery, error_message: str) -> MetricResult:
        return MetricResult(
            query_id=query.query_id,
            query_type=query.query_type,
            score=0.0,
            is_correct=False,
            model_output="[API_ERROR] Batch request failed",
            expected_answer=", ".join(query.get_correct_answers()),
            question=query.question,
            details={
                "api_error": True,
                "error_message": truncate_error_message(error_message),
            },
        )

    @staticmethod
    def _gold_evidence_turns(evidence: List[str]) -> List[str]:
        """Keep only canonical LoCoMo evidence IDs (for evaluator-only use)."""
        values, seen = [], set()
        for value in evidence or []:
            normalized = str(value).strip()
            if not re.fullmatch(r"D\d+:[^\s:]+", normalized) or normalized in seen:
                continue
            values.append(normalized)
            seen.add(normalized)
        return values

    @staticmethod
    def _turn_quality(gold_turn_ids: List[str], visible_turn_ids: List[str]) -> Dict[str, Any]:
        """Score exact evidence turns rendered in final answer context."""
        predicted = list(dict.fromkeys(str(value) for value in visible_turn_ids))
        gold = list(dict.fromkeys(gold_turn_ids))
        predicted_set, gold_set = set(predicted), set(gold)
        matched = [value for value in predicted if value in gold_set]
        precision = len(matched) / len(predicted) if predicted else 0.0
        recall = len(matched) / len(gold) if gold else None
        return {
            "stage": "answer_visible_exact_source_evidence",
            "unit": "locomo_turn_id",
            "available": bool(gold),
            "gold_turn_ids": gold,
            "answer_visible_turn_ids": predicted,
            "matched_turn_ids": matched,
            "precision": precision if gold else None,
            "recall": recall,
            "hit": bool(matched) if gold else None,
            "true_positive_count": len(matched),
            "false_positive_count": len(predicted_set - gold_set),
            "false_negative_count": len(gold_set - predicted_set),
        }

    @staticmethod
    def _locomo_turn_id(source_session_id: Any, turn_id: Any) -> Optional[str]:
        value = str(turn_id).strip()
        if re.fullmatch(r"D\d+:[^\s:]+", value):
            return value
        if source_session_id is None or not value:
            return None
        return f"D{source_session_id}:{value}"

    def _locomo_retrieval_quality(
        self, query: LoCoMoQuery, retrieved_memories: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Derive gold diagnostics after retrieval, never before method execution."""
        gold_turn_ids = self._gold_evidence_turns(query.evidence)
        gold_session_ids = [match.group(1) for item in gold_turn_ids if (match := re.fullmatch(r"D(\d+):[^\s:]+", item))]
        session_quality = compute_session_retrieval_quality(
            retrieved_memories, [], {"gold_session_ids": gold_session_ids},
        )
        session_quality["stage"] = "selected_memory_objects"
        visible_turn_ids = []
        for record in retrieved_memories or []:
            if not isinstance(record, dict):
                continue
            source_session = record.get("source_session_id")
            for turn_id in record.get("episode_evidence_turn_ids", []) if record.get("included_in_context") else []:
                normalized = self._locomo_turn_id(source_session, turn_id)
                if normalized:
                    visible_turn_ids.append(normalized)
            for item in record.get("included_provenance_evidence", []):
                evidence = item.get("evidence", item) if isinstance(item, dict) else {}
                if not isinstance(evidence, dict):
                    continue
                source_session = evidence.get("source_session_id")
                for turn_id in evidence.get("source_turn_ids", []):
                    normalized = self._locomo_turn_id(source_session, turn_id)
                    if normalized:
                        visible_turn_ids.append(normalized)
        return {
            "gold_evidence_turn_ids": gold_turn_ids,
            "gold_evidence_session_ids": list(dict.fromkeys(gold_session_ids)),
            "selected_memory_session": session_quality,
            "answer_visible_exact_turn": self._turn_quality(gold_turn_ids, visible_turn_ids),
        }

    def _score_agent_response(self, query: LoCoMoQuery, response: Any) -> MetricResult:
        if isinstance(response, dict):
            model_output = response.get("output", "")
            query_time = response.get("query_time", 0.0)
            retrieved_memories = response.get("retrieved_memories", [])
            retrieved_count = response.get("retrieved_count", 0)
            response_extra = response.get("extra", {})
        elif hasattr(response, "output"):
            model_output = response.output
            query_time = getattr(response, "query_time", 0.0)
            retrieved_memories = getattr(response, "retrieved_memories", [])
            retrieved_count = getattr(response, "retrieved_count", 0)
            response_extra = getattr(response, "extra", {})
        else:
            model_output = str(response)
            query_time = 0.0
            retrieved_memories = []
            retrieved_count = 0
            response_extra = {}

        tracker = get_usage_tracker()
        tracker.set_phase("judge")
        with tracker.scope("judge.realtime"):
            result = self.metrics_calculator.compute(
                query_id=query.query_id,
                query_type=query.query_type,
                model_output=model_output,
                expected_answers=query.get_correct_answers(),
                question=query.question,
                category=query.category,
                evidence=query.evidence,
                adversarial_answer=query.adversarial_answer,
                metadata=query.metadata,
            )

        result.query_time = query_time
        result.retrieved_memories = retrieved_memories
        result.retrieved_count = retrieved_count
        if isinstance(response_extra, dict):
            result.details["method_retrieval"] = response_extra

        if "category" not in result.details:
            result.details["category"] = query.category
        if "evidence" not in result.details:
            result.details["evidence"] = query.evidence
        result.details["locomo_retrieval_quality"] = self._locomo_retrieval_quality(
            query, retrieved_memories
        )

        return result

    def _generate_report(
        self,
        start_time: datetime,
        end_time: datetime,
        duration: float,
    ) -> EvaluationReport:
        summary = self.aggregator.get_summary()
        self._apply_locomo_summary(summary)

        memory_build_summary = self._summarize_memory_builds()
        build_metrics = self._compact_build_metrics()

        llm_usage = get_usage_tracker().get_stats()
        stage_usage = self._stage_usage_report(llm_usage)
        results = self.aggregator.results
        expected_queries = self.dataset.get_total_queries()
        scored_queries = sum(result.score is not None for result in results)
        api_error_count = sum(
            bool(result.details.get("api_error")) for result in results
        )
        category_filter = self.dataset.category_filter
        evaluation_coverage = {
            "expected_query_count": expected_queries,
            "scored_query_count": scored_queries,
            "api_error_count": api_error_count,
            "skipped_query_count": max(expected_queries - len(results), 0),
            "unscored_query_count": max(expected_queries - scored_queries, 0),
            "category_filter": category_filter,
            "category_coverage": self.dataset.get_category_distribution(),
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
                    "reasoning_effort": getattr(
                        get_api_config(), "judge_reasoning_effort", None
                    ),
                    "client_max_tokens": getattr(
                        get_api_config(), "judge_client_max_tokens", 10000
                    ),
                    "max_tokens": getattr(get_api_config(), "judge_max_tokens", 500),
                    "mcd_max_tokens": getattr(
                        get_api_config(), "judge_mcd_max_tokens", 2000
                    ),
                },
                "dry_run": self.dry_run,
                "locomo_scoring": {
                    "primary_metric": "official_token_stem_f1",
                    "enhanced_f1": "diagnostic_only",
                },
            },
            metadata={
                "run_metadata": self._git_metadata(),
                "total_samples": len(self.dataset.get_sample_ids()),
                "evaluated_sample_count": len(self.dataset.get_sample_ids()),
                "configured_max_samples": self.dataset.max_samples,
                "sample_ids": self.dataset.get_sample_ids(),
                "category_distribution": self.dataset.get_category_distribution(),
                "image_input_mode": "caption_only" if self.dataset.include_images else "disabled",
                "image_caption_field": "blip_caption" if self.dataset.include_images else None,
                "memory_build_summary": memory_build_summary,
                "build_metrics": build_metrics,
                "memory_chunk_size": self.memory_chunk_size,
                "llm_usage": llm_usage,
                "stage_usage": stage_usage,
                "evaluation_coverage": evaluation_coverage,
            }
        )

        result_path, memory_build_path, query_answer_path = self.result_collector.save_reports(
            report=report,
            output_dir=self.output_dir,
            memory_build_logs=self._memory_build_logs,
            use_method_subdir=not self.run_scoped_output,
        )

        self._log(f"Results saved to: {result_path}")
        self._log(f"Memory build details saved to: {memory_build_path}")
        self._log(f"Query answer details saved to: {query_answer_path}")

        return report

    @staticmethod
    def _aggregate_locomo_retrieval(results: List[MetricResult]) -> Dict[str, Any]:
        """Macro aggregate evaluator-only selected-session and visible-turn metrics."""
        stages = {
            "selected_memory_session": "selected_memory_session",
            "answer_visible_exact_turn": "answer_visible_exact_turn",
        }
        aggregated: Dict[str, Any] = {}
        for output_name, source_name in stages.items():
            groups: Dict[str, List[Dict[str, Any]]] = {}
            for result in results:
                quality = result.details.get("locomo_retrieval_quality", {})
                value = quality.get(source_name) if isinstance(quality, dict) else None
                if isinstance(value, dict):
                    groups.setdefault(result.query_type, []).append(value)

            def summarize(items: List[Dict[str, Any]]) -> Dict[str, Any]:
                available = [item for item in items if item.get("available")]
                def mean(field: str) -> Optional[float]:
                    values = [float(item[field]) for item in available if item.get(field) is not None]
                    return sum(values) / len(values) if values else None
                return {
                    "queries_with_gold_evidence": len(available),
                    "queries_without_gold_evidence": len(items) - len(available),
                    "hit_rate": mean("hit"), "recall": mean("recall"),
                    "precision": mean("precision"),
                    "mean_average_precision": mean("average_precision"),
                    "mean_reciprocal_rank": mean("reciprocal_rank"),
                }
            all_items = [item for items in groups.values() for item in items]
            aggregated[output_name] = {
                "stage": source_name,
                **summarize(all_items),
                "by_query_type": {
                    query_type: summarize(items) for query_type, items in sorted(groups.items())
                },
            }
        return aggregated

    def _apply_locomo_summary(self, summary: Dict[str, Any]) -> None:
        """Make continuous official F1 the LoCoMo headline, not a threshold count."""
        results = [
            result for result in self.aggregator.results
            if result.score is not None and result.details.get("metric") == "locomo_f1"
        ]
        summary["f1_query_count"] = len(results)
        summary["mean_f1"] = (
            sum(float(result.score) for result in results) / len(results)
            if results else 0.0
        )
        summary["queries_f1_ge_0_5"] = sum(
            float(result.score) >= 0.5 for result in results
        )
        summary["fraction_f1_ge_0_5"] = (
            summary["queries_f1_ge_0_5"] / len(results) if results else 0.0
        )
        for query_type, stats in summary.get("by_type", {}).items():
            if query_type == "adversarial":
                continue
            stats["mean_f1"] = stats.get("avg_score", 0.0)
            stats["queries_f1_ge_0_5"] = stats.pop("correct", 0)
            stats["fraction_f1_ge_0_5"] = stats.pop("accuracy", 0.0)
        summary["retrieval_quality"] = self._aggregate_locomo_retrieval(results)

    def _stage_usage_report(self, llm_usage: Dict[str, Any]) -> Dict[str, Any]:
        """Expose local phase accounting and batch lifecycle without fake latency."""
        operations = llm_usage.get("operations", {})
        query_operations = operations.get("query", {})
        batch_stages = []
        manifest_path = getattr(getattr(self, "_batch_client", None), "manifest_path", None)
        if manifest_path:
            try:
                manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
                for name, stage in (manifest.get("jobs") or {}).items():
                    batch_stages.append({
                        "stage": name, "state": stage.get("state"),
                        "request_count": len(stage.get("requests") or []),
                        "submitted_at": stage.get("submitted_at"),
                        "completed_at": stage.get("completed_at"),
                    })
            except (OSError, json.JSONDecodeError):
                pass
        return {
            "schema_version": 1,
            "memory": {"usage": llm_usage.get("memorize_phase", {})},
            "retrieval_preparation": {
                "usage": query_operations.get("query.retrieval_preparation", {}),
                "wall_time_seconds": getattr(
                    self, "_batch_retrieval_preparation_wall_time", 0.0
                ),
            },
            "answer_generation": {
                "usage": query_operations.get("query.answer_realtime", {}),
            },
            "judge": {"usage": llm_usage.get("judge_phase", {})},
            "batch_stages": batch_stages,
        }

    def _summarize_memory_builds(self) -> Dict[str, Any]:
        if not self._memory_build_logs:
            return {"total_builds": 0}

        total_units = len(self._memory_build_logs)

        total_sessions = sum(
            log.get("session_count", 0)
            for log in self._memory_build_logs
        )

        total_chunks = sum(
            log.get("chunk_count", 0)
            for log in self._memory_build_logs
        )

        total_time = sum(
            log.get("total_time", 0)
            for log in self._memory_build_logs
        )

        inserted_record_count = sum(
            log.get("inserted_record_count", log.get("total_entries", 0))
            for log in self._memory_build_logs
        )

        total_stored_chunks = sum(
            log.get("total_stored_chunks", 0)
            for log in self._memory_build_logs
        )

        return {
            "total_units": total_units,
            "total_sessions": total_sessions,
            "total_memory_chunks": total_chunks,
            "total_time": total_time,
            "inserted_record_count": inserted_record_count,
            "total_stored_chunks": total_stored_chunks,
            "avg_time_per_unit": total_time / total_units if total_units > 0 else 0,
            "avg_chunks_per_unit": total_chunks / total_units if total_units > 0 else 0,
            "chunk_size_config": self.memory_chunk_size,
        }

    def _compact_build_metrics(self) -> Dict[str, Any]:
        """Keep useful Event-State diagnostics in result.json without raw memory."""
        units = {}
        excluded = {
            "memory_entries", "all_passages", "input_content", "stored_content",
            "extraction_result", "raw_result",
        }
        for log in self._memory_build_logs:
            metrics = log.get("build_metrics", {})
            if not isinstance(metrics, dict):
                continue
            units[str(log.get("context_id"))] = {
                key: value for key, value in metrics.items() if key not in excluded
            }
        return {
            "schema_version": 1,
            "units": units,
        }


@register_evaluator("locomo")
def evaluate_locomo(
    method_config: MethodConfig,
    dataset_config: DatasetConfig,
    output_dir: Path,
    dry_run: bool = False,
    verbose: bool = True,
    logger: Optional[logging.Logger] = None,
    resume: bool = False,
    execution_stage: str = "all",
    memory_run: Optional[str] = None,
    memory_source_run_dir: Optional[Path] = None,
    run_scoped_output: bool = False,
    batch_api: bool = False,
    batch_gcs_uri: Optional[str] = None,
    batch_wait: bool = False,
    workers: int = 1,
    **kwargs
) -> EvaluationReport:
    evaluator = LoCoMoEvaluator(
        method_config=method_config,
        dataset_config=dataset_config,
        output_dir=output_dir,
        dry_run=dry_run,
        verbose=verbose,
        logger=logger,
        resume=resume,
        execution_stage=execution_stage,
        memory_run=memory_run,
        memory_source_run_dir=memory_source_run_dir,
        run_scoped_output=run_scoped_output,
        batch_api=batch_api,
        batch_gcs_uri=batch_gcs_uri,
        batch_wait=batch_wait,
        workers=workers,
    )
    return evaluator.evaluate()
