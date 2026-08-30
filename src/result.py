"""Result collection and report generation module."""

import json
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Tuple

from metrics import MetricResult


def _true_duration_seconds(report: "EvaluationReport") -> float:
    """Return wall time with measured failed API operation time removed."""
    failure_duration = report.metadata.get("api_failure_duration_seconds")
    if failure_duration is None:
        failure_duration = sum(
            float(failure.get("duration_seconds", 0.0) or 0.0)
            for failure in report.metadata.get("api_failures", [])
        )
    return max(
        report.duration_seconds - max(float(failure_duration), 0.0),
        0.0,
    )


def _failure_counts_from_usage(llm_usage: Dict[str, Any]) -> Dict[str, Any]:
    """Summarize failed attempts and retries without duplicating failure logs."""
    phases = {
        "memorize": dict(llm_usage.get("memorize_phase") or {}),
        "query": dict(llm_usage.get("query_phase") or {}),
    }
    by_phase = {
        phase: {
            "failed_attempts": int(stats.get("failed_attempts", 0) or 0),
            "retry_count": int(stats.get("retry_count", 0) or 0),
        }
        for phase, stats in phases.items()
    }
    total = dict(llm_usage.get("total") or {})
    return {
        "total_failed_attempts": int(total.get("failed_attempts", 0) or 0),
        "total_retries": int(total.get("retry_count", 0) or 0),
        "by_phase": by_phase,
    }


def _efficiency_with_timing_semantics(report: "EvaluationReport") -> Dict[str, Any]:
    """Avoid presenting unavailable batch per-request latency as a real zero."""
    efficiency = dict(report.summary.get("efficiency", {}))
    stage_usage = report.metadata.get("stage_usage", {})
    batch_stages = stage_usage.get("batch_stages", []) if isinstance(stage_usage, dict) else []
    if not batch_stages:
        return efficiency
    efficiency["query_time_kind"] = "per_request_latency_unavailable_for_batch"
    efficiency["stage_wall_time_seconds"] = {
        name: (stage_usage.get(name, {}).get("usage", {}).get("wall_time"))
        for name in ("retrieval_preparation", "answer", "judge")
    }
    efficiency["batch_stage_count"] = len(batch_stages)
    return efficiency


@dataclass
class EvaluationReport:
    """Evaluation report data structure."""
    method_name: str
    model_name: str
    dataset_name: str
    start_time: str
    end_time: str
    duration_seconds: float
    summary: Dict[str, Any]
    detailed_results: List[Dict[str, Any]] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["true_duration_seconds"] = _true_duration_seconds(self)
        return data

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


class ResultCollector:
    """Result collector for generating evaluation reports."""

    def __init__(self):
        self._results: List[MetricResult] = []
        self._results_by_context: Dict[int, List[MetricResult]] = {}
        self.last_api_failure_path: Optional[Path] = None

    def add_result(self, result: MetricResult, context_id: Optional[int] = None) -> None:
        self._results.append(result)

        if context_id is not None:
            if context_id not in self._results_by_context:
                self._results_by_context[context_id] = []
            self._results_by_context[context_id].append(result)

    def get_all_results(self) -> List[MetricResult]:
        return self._results.copy()

    def get_results_by_context(self, context_id: int) -> List[MetricResult]:
        return self._results_by_context.get(context_id, []).copy()

    def get_context_ids(self) -> List[int]:
        return list(self._results_by_context.keys())

    def save_reports(
        self,
        report: EvaluationReport,
        output_dir: Path,
        memory_build_logs: List[Dict[str, Any]],
        *,
        include_result: bool = True,
        include_memory_build: bool = True,
        include_query_answer: bool = True,
        include_api_failures: bool = True,
        use_method_subdir: bool = True,
    ) -> Tuple[Optional[Path], Optional[Path], Optional[Path]]:
        """Save only the report files produced by the active execution stage."""
        method_subdir = self._get_method_subdir(report.method_name, report.model_name)
        method_output_dir = (
            output_dir / method_subdir if use_method_subdir else output_dir
        )
        method_output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = f"{report.dataset_name}_{report.method_name}_{report.model_name}_{timestamp}"
        prefix = prefix.replace("/", "-").replace("\\", "-")

        result_path = (
            self._save_result_json(report, method_output_dir, prefix)
            if include_result else None
        )
        memory_build_path = (
            self._save_memory_build_json(
                report, memory_build_logs, method_output_dir, prefix
            )
            if include_memory_build else None
        )
        query_answer_path = (
            self._save_query_answer_json(report, method_output_dir, prefix)
            if include_query_answer else None
        )
        self.last_api_failure_path = None
        failure_counts = _failure_counts_from_usage(
            report.metadata.get("llm_usage") or {}
        )
        if include_api_failures and (
            report.metadata.get("api_failures")
            or failure_counts["total_failed_attempts"]
        ):
            self.last_api_failure_path = self._save_api_failures_json(
                report, method_output_dir, prefix
            )

        return result_path, memory_build_path, query_answer_path

    def _save_result_json(
        self,
        report: EvaluationReport,
        output_dir: Path,
        prefix: str,
    ) -> Path:
        """Save evaluation metrics file (result.json)."""
        filepath = output_dir / f"{prefix}_result.json"

        result_summary = {
            "total_queries": report.summary.get("total", 0),
            "correct_count": report.summary.get("correct", 0),
            "overall_accuracy": report.summary.get("overall_accuracy", 0.0),
            "overall_avg_score": report.summary.get("overall_avg_score", 0.0),
            "by_type": report.summary.get("by_type", {}),
            "by_metric": report.summary.get("by_metric", {}),
        }
        metric_groups = report.summary.get("metric_groups")
        if isinstance(metric_groups, dict) and metric_groups:
            result_summary["metric_groups"] = metric_groups

        result_data = {
            "method_name": report.method_name,
            "model_name": report.model_name,
            "dataset_name": report.dataset_name,
            "start_time": report.start_time,
            "end_time": report.end_time,
            "duration_seconds": report.duration_seconds,
            "true_duration_seconds": _true_duration_seconds(report),
            "summary": result_summary,
            "efficiency": _efficiency_with_timing_semantics(report),
            "memory_build_summary": report.metadata.get("memory_build_summary", {}),
            "build_metrics": report.metadata.get("build_metrics", {}),
            "memory_size": report.metadata.get("memory_size", {}),
            "feature_configuration": report.metadata.get(
                "feature_configuration", {}
            ),
            "llm_usage": report.metadata.get("llm_usage", {}),
            "stage_usage": report.metadata.get("stage_usage", {}),
            "evaluation_coverage": report.metadata.get("evaluation_coverage", {}),
            "config": {
                "evaluation_mode": report.metadata.get("evaluation_mode", ""),
                "evaluation_interval": report.metadata.get("evaluation_interval", 0),
                "total_personas": report.metadata.get("total_personas", 0),
                "method_config": report.config.get("method_config", {}),
                "dataset_config": report.config.get("dataset_config", {}),
            },
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2)

        return filepath

    def _save_api_failures_json(
        self,
        report: EvaluationReport,
        output_dir: Path,
        prefix: str,
    ) -> Path:
        """Save provider failures outside metric and query-answer outputs."""
        filepath = output_dir / f"{prefix}_api_failures.json"
        failures = report.metadata.get("api_failures", [])
        failure_counts = _failure_counts_from_usage(
            report.metadata.get("llm_usage") or {}
        )
        failure_data = {
            "method_name": report.method_name,
            "model_name": report.model_name,
            "dataset_name": report.dataset_name,
            "summary": report.metadata.get("evaluation_coverage", {}),
            "failure_count": len(failures),
            "failure_counts": failure_counts,
            "failures": failures,
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(failure_data, f, ensure_ascii=False, indent=2)

        return filepath

    def _save_memory_build_json(
        self,
        report: EvaluationReport,
        memory_build_logs: List[Dict[str, Any]],
        output_dir: Path,
        prefix: str,
    ) -> Path:
        """Save memory build details file (memory_build.json)."""
        filepath = output_dir / f"{prefix}_memory_build.json"

        processed_units = []
        for log in memory_build_logs:
            # Check if this is the new per-session format (MedMemoryBench)
            if "session_builds" in log:
                # New format: per-session builds
                processed_sessions = []
                for sb in log.get("session_builds", []):
                    build_result = sb.get("build_result", {})
                    processed_sessions.append({
                        "session_id": sb.get("session_id"),
                        "session_index": sb.get("session_index"),
                        "method": build_result.get("method", ""),
                        "action": build_result.get("action", ""),
                        "time_cost": build_result.get("time_cost", 0.0),
                        "input_content": build_result.get("input_content", ""),
                        "stored_content": build_result.get("stored_content", ""),
                        "extraction_result": build_result.get("extraction_result", ""),
                        "all_passages": build_result.get("all_passages", []),
                        "memory_entries": build_result.get("memory_entries", []),
                        "chunk_count": build_result.get("chunk_count", 0),
                        "wall_time_seconds": sb.get(
                            "wall_time_seconds", build_result.get("time_cost", 0.0)
                        ),
                        "build_metrics": sb.get("build_metrics", {}),
                        "extra": {
                            k: v for k, v in build_result.items()
                            if k not in ["method", "action", "time_cost", "input_content",
                                        "stored_content", "extraction_result", "all_passages",
                                        "memory_entries", "chunk_count", "success"]
                        },
                        "error": sb.get("error"),  # Include error if present
                    })

                processed_unit = {
                    "unit_id": log.get("unit_id"),
                    "context_id": log.get("context_id"),
                    "session_ids": log.get("session_ids", []),
                    "session_count": log.get("session_count", 0),
                    "total_time": log.get("total_time", 0.0),
                    "total_passages": log.get("total_passages", 0),
                    "build_metrics": log.get("build_metrics", {}),
                    "memory_size": log.get("memory_size"),
                    "feature_configuration": log.get("feature_configuration", {}),
                    "restored_from_snapshot": log.get("restored_from_snapshot", False),
                    "session_builds": processed_sessions,
                }

            # Check if this is the new chunk_builds format (LoCoMo)
            elif "chunk_builds" in log:
                # LoCoMo format: per-chunk builds
                processed_chunks = []
                for cb in log.get("chunk_builds", []):
                    build_result = cb.get("build_result", {})
                    processed_chunks.append({
                        "chunk_index": cb.get("chunk_index"),
                        "session_ids": cb.get("session_ids", []),
                        "session_count": cb.get("session_count", 0),
                        "input_chars": cb.get("input_chars", 0),
                        "input_tokens_est": cb.get("input_tokens_est", 0),
                        "time_cost": cb.get("time_cost", 0.0),
                        "method": build_result.get("method", ""),
                        "action": build_result.get("action", ""),
                        "input_content": build_result.get("input_content", ""),
                        "stored_content": build_result.get("stored_content", ""),
                        "extraction_result": build_result.get("extraction_result", ""),
                        "all_passages": build_result.get("all_passages", []),
                        "memory_entries": build_result.get("memory_entries", []),
                        "chunk_count": build_result.get("chunk_count", 0),
                        "extra": {
                            k: v for k, v in build_result.items()
                            if k not in ["method", "action", "time_cost", "input_content",
                                        "stored_content", "extraction_result", "all_passages",
                                        "memory_entries", "chunk_count", "success"]
                        },
                        "error": cb.get("error"),  # Include error if present
                    })

                processed_unit = {
                    "unit_id": log.get("unit_id"),
                    "context_id": log.get("context_id"),
                    "session_ids": log.get("session_ids", []),
                    "session_count": log.get("session_count", 0),
                    "chunk_count": log.get("chunk_count", 0),
                    "chunk_size_config": log.get("chunk_size_config", 0),
                    "total_time": log.get("total_time", 0.0),
                    "total_entries": log.get("total_entries", 0),
                    "total_stored_chunks": log.get("total_stored_chunks", 0),
                    "chunk_builds": processed_chunks,
                }

            else:
                # Legacy format: single build_result per unit
                build_result = log.get("build_result", {})
                processed_unit = {
                    "unit_id": log.get("unit_id"),
                    "context_id": log.get("context_id"),
                    "session_ids": log.get("session_ids", []),
                    "session_count": log.get("session_count", 0),
                    "method": build_result.get("method", ""),
                    "action": build_result.get("action", ""),
                    "time_cost": build_result.get("time_cost", 0.0),
                    "input_content": build_result.get("input_content", ""),
                    "stored_content": build_result.get("stored_content", ""),
                    "extraction_result": build_result.get("extraction_result", ""),
                    "all_passages": build_result.get("all_passages", []),
                    "memory_entries": build_result.get("memory_entries", []),
                    "chunk_count": build_result.get("chunk_count", 0),
                    "extra": {
                        k: v for k, v in build_result.items()
                        if k not in ["method", "action", "time_cost", "input_content",
                                    "stored_content", "extraction_result", "all_passages",
                                    "memory_entries", "chunk_count", "success"]
                    },
                }

            processed_units.append(processed_unit)

        memory_build_data = {
            "method_name": report.method_name,
            "model_name": report.model_name,
            "dataset_name": report.dataset_name,
            "summary": report.metadata.get("memory_build_summary", {}),
            "build_metrics": report.metadata.get("build_metrics", {}),
            "memory_size": report.metadata.get("memory_size", {}),
            "feature_configuration": report.metadata.get(
                "feature_configuration", {}
            ),
            "llm_usage": report.metadata.get("llm_usage", {}),
            "stage_usage": report.metadata.get("stage_usage", {}),
            "memory_chunk_size": report.metadata.get("memory_chunk_size"),
            "total_units": len(processed_units),
            "units": processed_units,
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(memory_build_data, f, ensure_ascii=False, indent=2)

        return filepath

    def _save_query_answer_json(
        self,
        report: EvaluationReport,
        output_dir: Path,
        prefix: str,
    ) -> Path:
        """Save query answer details file (query_answer.json)."""
        filepath = output_dir / f"{prefix}_query_answer.json"

        context_by_result = {
            id(result): context_id
            for context_id, results in self._results_by_context.items()
            for result in results
        }
        retrieval_records = []
        query_details = []
        for index, result in enumerate(report.detailed_results):
            result_object = self._results[index] if index < len(self._results) else None
            retrieved_memories = result.get("retrieved_memories", [])
            if not isinstance(retrieved_memories, list):
                retrieved_memories = []
            memory_ids = []
            for memory in retrieved_memories:
                if not isinstance(memory, dict):
                    continue
                memory_id = memory.get("memory_id", memory.get("id"))
                if memory_id is not None:
                    memory_ids.append(str(memory_id))
            details = result.get("details", {})
            if not isinstance(details, dict):
                details = {}
            artifact_references = details.get("artifact_references", {})
            if not isinstance(artifact_references, dict):
                artifact_references = {}
            execution_usage = details.get("execution_usage", {})
            if not isinstance(execution_usage, dict):
                execution_usage = {}
            answer_usage = execution_usage.get("answer", {})
            if not isinstance(answer_usage, dict):
                answer_usage = {}
            answer_reference = artifact_references.get("answer", {})
            if not isinstance(answer_reference, dict):
                answer_reference = {}
            has_batch_retrieval = bool(
                answer_reference.get("transport") == "batch"
                and answer_reference.get("manifest_path")
                and answer_reference.get("request_id")
            )
            retrieval_reference: Dict[str, Any]
            if has_batch_retrieval:
                retrieval_reference = {
                    "source": "answer_batch_manifest",
                    "manifest_path": answer_reference["manifest_path"],
                    "request_id": answer_reference["request_id"],
                    "prepared_query_key": "prepared_query",
                }
            elif retrieved_memories:
                retrieval_records.append({
                    "query_id": result.get("query_id", ""),
                    "context_id": (
                        context_by_result.get(id(result_object))
                        if result_object is not None else None
                    ),
                    "retrieved_memories": retrieved_memories,
                })
                retrieval_reference = {
                    "source": "retrieval_records",
                    "record_id": result.get("query_id", ""),
                }
            else:
                retrieval_reference = {"source": "none"}
            query_detail = {
                "context_id": (
                    context_by_result.get(id(result_object))
                    if result_object is not None
                    else None
                ),
                "query_id": result.get("query_id", ""),
                "query_type": result.get("query_type", ""),
                "question": result.get("question", ""),
                "expected_answer": result.get("expected_answer", ""),
                "model_output": result.get("model_output", ""),
                "score": result.get("score", 0.0),
                "is_correct": result.get("is_correct", False),
                "retrieved_count": result.get("retrieved_count", 0),
                "retrieved_memory_ids": memory_ids,
                "retrieval_reference": retrieval_reference,
                "query_time": result.get("query_time", 0.0),
                "execution_references": artifact_references,
                "execution_usage": execution_usage,
                "timing": {
                    "kind": (
                        "per_request_latency_unavailable_for_batch"
                        if answer_usage.get("transport") == "batch"
                        else "end_to_end_local"
                    ),
                    "query_time_seconds": result.get("query_time", 0.0),
                    "answer_provider_duration_seconds": answer_usage.get(
                        "duration_seconds"
                    ),
                },
                "evaluation_details": {
                    key: value
                    for key, value in details.items()
                    if key not in {"artifact_references", "execution_usage"}
                },
            }
            query_details.append(query_detail)

        by_context = {}
        for result in self._results:
            for ctx_id, ctx_results in self._results_by_context.items():
                if result in ctx_results:
                    if ctx_id not in by_context:
                        by_context[ctx_id] = {
                            "total": 0,
                            "correct": 0,
                            "query_ids": [],
                        }
                    by_context[ctx_id]["total"] += 1
                    by_context[ctx_id]["correct"] += 1 if result.is_correct else 0
                    by_context[ctx_id]["query_ids"].append(result.query_id)
                    break

        retrieval_records_path = None
        if retrieval_records:
            retrieval_records_path = output_dir / f"{prefix}_retrieval_records.json"
            retrieval_data = {
                "format": "medmemorybench.query_retrieval_records",
                "version": 1,
                "method_name": report.method_name,
                "model_name": report.model_name,
                "dataset_name": report.dataset_name,
                "records": retrieval_records,
            }
            with open(retrieval_records_path, "w", encoding="utf-8") as f:
                json.dump(retrieval_data, f, ensure_ascii=False, indent=2)

        query_summary = {
            "total_queries": len(query_details),
            "correct_count": sum(1 for q in query_details if q["is_correct"]),
            "overall_accuracy": report.summary.get("overall_accuracy", 0.0),
            "overall_avg_score": report.summary.get("overall_avg_score", 0.0),
            "by_type": report.summary.get("by_type", {}),
            "by_metric": report.summary.get("by_metric", {}),
            "evaluation_coverage": report.metadata.get("evaluation_coverage", {}),
            "stage_usage": report.metadata.get("stage_usage", {}),
            "efficiency": _efficiency_with_timing_semantics(report),
            "total_query_time": sum(q["query_time"] for q in query_details),
            "avg_query_time": (
                sum(q["query_time"] for q in query_details) / len(query_details)
                if query_details else 0.0
            ),
            "avg_retrieved_count": (
                sum(q["retrieved_count"] for q in query_details) / len(query_details)
                if query_details else 0.0
            ),
        }
        metric_groups = report.summary.get("metric_groups")
        if isinstance(metric_groups, dict) and metric_groups:
            query_summary["metric_groups"] = metric_groups

        query_answer_data = {
            "format": "medmemorybench.query_answers",
            "version": 2,
            "method_name": report.method_name,
            "model_name": report.model_name,
            "dataset_name": report.dataset_name,
            "summary": query_summary,
            "retrieval_records_path": (
                retrieval_records_path.name if retrieval_records_path is not None else None
            ),
            "by_context": by_context,
            "queries": query_details,
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(query_answer_data, f, ensure_ascii=False, indent=2)

        return filepath

    @staticmethod
    def _get_method_subdir(method_name: str, model_name: str) -> str:
        """Generate method subdirectory name."""
        safe_method = method_name.replace("/", "-").replace("\\", "-")
        safe_model = model_name.replace("/", "-").replace("\\", "-")
        return f"{safe_method}_{safe_model}"

    def save_report(self, report: EvaluationReport, output_dir: Path) -> Path:
        """Save evaluation report (backward compatible method)."""
        memory_build_logs = report.metadata.get("memory_build_logs", [])
        result_path, _, _ = self.save_reports(report, output_dir, memory_build_logs)
        if result_path is None:
            raise RuntimeError("Result report was not written")
        return result_path

    def clear(self) -> None:
        self._results.clear()
        self._results_by_context.clear()
        self.last_api_failure_path = None


def generate_comparison_report(
    reports: List[EvaluationReport],
    output_dir: Path,
) -> Path:
    """Generate multi-method comparison report."""
    comparison = {
        "generated_at": datetime.now().isoformat(),
        "methods": [],
    }

    for report in reports:
        comparison["methods"].append({
            "method_name": report.method_name,
            "model_name": report.model_name,
            "dataset_name": report.dataset_name,
            "duration_seconds": report.duration_seconds,
            "true_duration_seconds": _true_duration_seconds(report),
            "summary": report.summary,
        })

    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = output_dir / f"comparison_{timestamp}.json"

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(comparison, ensure_ascii=False, indent=2, fp=f)

    return filepath
