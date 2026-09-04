"""MedMemoryBench batch integration for SmartMem0 prepared-query answers.

SmartMem0 can finish an atomic query during local preparation. Vertex batch must
not submit an unused second generation for such a prepared query. This helper
patches only SmartMem0 evaluator transport/accounting; every other method
continues through the evaluator's original implementation unchanged.
"""

from datetime import datetime
from math import ceil
from statistics import mean, median
from typing import Any, Dict, List

from src.result import EvaluationReport
from utils.llm_client import get_usage_tracker
from utils.vertex_batch import (
    BatchChatRequest,
    PREPARED_QUERY_METADATA_KEY,
    make_request_id,
    restore_prepared_query,
    snapshot_prepared_query,
)


def _method_llm_usage(evaluator) -> Dict[str, Any]:
    controller = answer = total = covered = 0
    token_totals: List[int] = []
    token_stages: Dict[str, int] = {}
    for result in evaluator.aggregator.results:
        details = getattr(result, "details", {}) or {}
        telemetry = details.get("agent_telemetry") or {}
        calls = telemetry.get("method_llm_calls") or {}
        if not isinstance(calls, dict):
            continue
        covered += 1
        controller += int(calls.get("controller", 0) or 0)
        answer += int(calls.get("answer", 0) or 0)
        total += int(calls.get("total", 0) or 0)
        query_tokens = telemetry.get("query_tokens") or {}
        if isinstance(query_tokens, dict):
            value = int(query_tokens.get("total", 0) or 0)
            token_totals.append(value)
            for stage, stage_tokens in query_tokens.items():
                if stage == "total":
                    continue
                token_stages[stage] = token_stages.get(stage, 0) + int(
                    stage_tokens or 0
                )

    ordered = sorted(token_totals)

    def percentile(p: float) -> int:
        if not ordered:
            return 0
        return ordered[max(0, ceil(p * len(ordered)) - 1)]

    return {
        "controller_calls": controller,
        "answer_calls": answer,
        "total_calls": total,
        "queries_with_telemetry": covered,
        "total_tokens": sum(token_totals),
        "tokens_by_stage": token_stages,
        "tokens_per_query": {
            "mean": round(mean(token_totals), 3) if token_totals else 0,
            "median": round(median(token_totals), 3) if token_totals else 0,
            "p90": percentile(0.90),
            "p95": percentile(0.95),
            "max": max(token_totals, default=0),
        },
        "excludes_evaluator_judges": True,
    }


def _evaluation_llm_usage(global_usage: Dict[str, Any], method_usage: Dict[str, Any]) -> Dict[str, Any]:
    """Expose benchmark-side calls without pretending they belong to SmartMem0."""
    query_usage = global_usage.get("query_phase") or {}
    global_calls = int(query_usage.get("call_count", 0) or 0)
    global_tokens = int(query_usage.get("total_tokens", 0) or 0)
    method_calls = int(method_usage.get("total_calls", 0) or 0)
    method_tokens = int(method_usage.get("total_tokens", 0) or 0)
    return {
        "call_count": max(0, global_calls - method_calls),
        "total_tokens": max(0, global_tokens - method_tokens),
        "includes_judges_and_other_benchmark_query_calls": True,
        "derived_from_global_query_usage": True,
    }


def install_smart_mem0_batch_integration(evaluator_cls) -> None:
    """Install resume-safe SmartMem0 local-finalization and usage reporting."""
    if getattr(evaluator_cls, "_smart_mem0_batch_integration_installed", False):
        return

    original_evaluate_batch_queries = evaluator_cls._evaluate_batch_queries
    original_generate_report = evaluator_cls._generate_report

    def _evaluate_batch_queries(self, unit, memory_time_per_query):
        if str(self.method_config.method_name) != "smart_mem0":
            return original_evaluate_batch_queries(self, unit, memory_time_per_query)

        prepared_by_id = {}
        local_precomputed = set()
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

            # Preserve a previously staged request exactly when resuming an old
            # manifest. For a fresh preparation, a surviving precomputed answer
            # is already final and must not pay for an unused remote generation.
            if saved_request is not None:
                requests.append(saved_request)
            elif str(prepared.get("precomputed_answer") or "").strip():
                local_precomputed.add(request_id)
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
            f"submitting {len(requests):,} final-answer request(s); "
            f"finalizing {len(local_precomputed):,} precomputed answer(s) locally."
        )
        responses = batch_client.run_stage(stage, requests) if requests else {}
        results = []

        for request_id, (query, prepared) in prepared_by_id.items():
            if request_id in local_precomputed:
                response = self.agent_manager.finalize_batch_query(
                    prepared,
                    "",
                    input_tokens=0,
                    output_tokens=0,
                )
            else:
                batch_response = responses.get(request_id)
                if batch_response is None or batch_response.status:
                    error = batch_response.status if batch_response else "No output row returned"
                    results.append(
                        self._api_error_result(
                            query,
                            f"Vertex batch request failed: {error}",
                        )
                    )
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

    def _generate_report(self, start_time, end_time, duration):
        if str(self.method_config.method_name) != "smart_mem0":
            return original_generate_report(self, start_time, end_time, duration)

        summary = self.aggregator.get_summary()
        memory_build_summary = self._summarize_memory_builds()
        llm_usage = get_usage_tracker().get_stats()
        method_llm_usage = _method_llm_usage(self)
        evaluation_llm_usage = _evaluation_llm_usage(llm_usage, method_llm_usage)

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
                # Global usage intentionally includes evaluator/judge calls.
                "llm_usage": llm_usage,
                # SmartMem0's semantic-generation budget only.
                "method_llm_usage": method_llm_usage,
                # Benchmark-side calls, derived from global minus method calls.
                "evaluation_llm_usage": evaluation_llm_usage,
            },
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

    evaluator_cls._evaluate_batch_queries = _evaluate_batch_queries
    evaluator_cls._generate_report = _generate_report
    evaluator_cls._smart_mem0_batch_integration_installed = True
