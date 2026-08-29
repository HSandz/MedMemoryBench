"""LoCoMo evaluation module."""

import hashlib
import json
import time
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
from utils.templates import get_prompt_manager
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
        self.run_scoped_output = run_scoped_output
        self.batch_api = batch_api
        self.batch_gcs_uri = batch_gcs_uri
        self.batch_wait = batch_wait

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
        if self.verbose:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] [{level}] {message}")
        if self.logger:
            self.logger.info(message)

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
                    self._log(f"Warning: Failed to reset old agent: {e}", level="WARNING")

            self.agent_manager = AgentManager(
                method_config=self.method_config,
                dataset_config=self.dataset_config,
                batch_api=self.batch_api,
                batch_gcs_uri=self.batch_gcs_uri,
                batch_wait=self.batch_wait,
                batch_manifest_dir=self.output_dir / "batch",
                batch_config_hash=self._batch_config_hash(),
                batch_progress_callback=self._log,
            )

        self.agent_manager.set_context_id(context_id)

    def evaluate(self) -> EvaluationReport:
        start_time = datetime.now()

        get_usage_tracker().reset()

        self._init_dataset()

        self._run_evaluation_loop()

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        report = self._generate_report(start_time, end_time, duration)

        return report

    def _run_evaluation_loop(self) -> None:
        for unit in self.dataset.get_evaluation_units():
            sample_id = unit.context_id

            self._log(f"\n{'='*60}")
            self._log(f"Sample: {sample_id}")
            self._log(f"  Sessions: {len(unit.sessions_to_inject)}")
            self._log(f"  Queries: {len(unit.queries_to_evaluate)}")

            if not self.dry_run:
                self._init_agent_for_context(context_id=sample_id, force_new=True)

            unit_results = self._evaluate_unit(unit)

            for result in unit_results:
                self.aggregator.add_result(result)
                self.result_collector.add_result(result, sample_id)

        for item in self._complete_combined_batch_queries():
            result = item["result"]
            sample_id = item["sample_id"]
            self.aggregator.add_result(result)
            self.result_collector.add_result(result, sample_id)

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
                    for session in chunk_sessions:
                        for turn in session.dialogues:
                            memory_item = {
                                "speaker": turn.get("speaker", "Unknown"),
                                "content": turn.get("text", ""),
                                "blip_caption": turn.get("blip_caption", ""),
                                "timestamp": session.date_time,
                            }
                            if self.method_config.method_name == "amem_test":
                                memory_item.update({
                                    "source_session_id": session.session_id,
                                    "source_session_index": session.session_id,
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
                    self._log(f"      [ERROR] Chunk {chunk_idx + 1} failed: {e}", level="ERROR")

                    chunk_build_results.append({
                        "chunk_index": chunk_idx,
                        "session_ids": chunk_session_ids,
                        "session_count": len(chunk_sessions),
                        "input_chars": chunk_chars,
                        "input_tokens_est": chunk_tokens_est,
                        "time_cost": chunk_time,
                        "error": str(e),
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

        self._log(f"  --- Query Evaluation Phase ---")

        query_count = len(unit.queries_to_evaluate)
        memory_time_per_query = total_memory_time / query_count if query_count > 0 else 0.0

        if unit.queries_to_evaluate and self._supports_batch_queries():
            batch_client = self._get_batch_client()
            combined_stage = "query-final"
            legacy_stage = f"query-unit-{unit.unit_id}"
            if not batch_client.has_stage(combined_stage) and batch_client.has_stage(legacy_stage):
                query_results = self._evaluate_batch_queries(unit)
            else:
                prepared_count = self._prepare_combined_batch_queries(
                    unit,
                    memory_time_per_query,
                )
                query_results = []
                self._log(
                    f"  [Batch] Prepared {prepared_count:,} request(s) for "
                    f"combined stage '{combined_stage}'."
                )
        else:
            if self.batch_api and not self.dry_run and not self._batch_fallback_logged:
                self._log(
                    "Batch API is unavailable for this adapter; "
                    "using real-time final-answer generation.",
                    level="WARNING",
                )
                self._batch_fallback_logged = True
            query_results = [
                self._evaluate_query(query, unit.context_id)
                for query in unit.queries_to_evaluate
            ]

        for result in query_results:
            result.memory_construction_time = memory_time_per_query
            results.append(result)

            status = "✓" if result.is_correct else "✗"
            self._log(f"    [{status}] {query.query_id} ({query.query_type}): {result.score:.2f}")

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
                results.append(self._api_error_result(query, f"Batch request failed: {error}"))
                continue
            response = self.agent_manager.finalize_batch_query(
                prepared,
                batch_response.content,
                input_tokens=batch_response.input_tokens,
                output_tokens=batch_response.output_tokens,
            )
            results.append(self._score_agent_response(query, response))
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
                    f"Batch request failed: {error}",
                )
            else:
                response = self.agent_manager.finalize_batch_query(
                    item["prepared"],
                    batch_response.content,
                    input_tokens=batch_response.input_tokens,
                    output_tokens=batch_response.output_tokens,
                )
                result = self._score_agent_response(query, response)

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

    def _evaluate_query(self, query: LoCoMoQuery, context_id: Any) -> MetricResult:
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

        response = self.agent_manager.send_message(
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
            details={"api_error": True, "error_message": error_message},
        )

    def _score_agent_response(self, query: LoCoMoQuery, response: Any) -> MetricResult:
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

        if "category" not in result.details:
            result.details["category"] = query.category
        if "evidence" not in result.details:
            result.details["evidence"] = query.evidence

        return result

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
            },
            metadata={
                "total_samples": len(self.dataset.get_sample_ids()),
                "category_distribution": self.dataset.get_category_distribution(),
                "memory_build_summary": memory_build_summary,
                "memory_chunk_size": self.memory_chunk_size,
                "llm_usage": llm_usage,
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

        total_entries = sum(
            log.get("total_entries", 0)
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
            "total_entries": total_entries,
            "total_stored_chunks": total_stored_chunks,
            "avg_time_per_unit": total_time / total_units if total_units > 0 else 0,
            "avg_chunks_per_unit": total_chunks / total_units if total_units > 0 else 0,
            "chunk_size_config": self.memory_chunk_size,
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
    run_scoped_output: bool = False,
    batch_api: bool = False,
    batch_gcs_uri: Optional[str] = None,
    batch_wait: bool = False,
    **kwargs
) -> EvaluationReport:
    if execution_stage != "all":
        raise ValueError("Separated memory/query execution is currently implemented for AMem on MedMemoryBench only")
    evaluator = LoCoMoEvaluator(
        method_config=method_config,
        dataset_config=dataset_config,
        output_dir=output_dir,
        dry_run=dry_run,
        verbose=verbose,
        logger=logger,
        resume=resume,
        run_scoped_output=run_scoped_output,
        batch_api=batch_api,
        batch_gcs_uri=batch_gcs_uri,
        batch_wait=batch_wait,
    )
    return evaluator.evaluate()
