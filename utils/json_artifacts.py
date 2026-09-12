"""Readable, stable ordering for persisted JSON artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, TextIO


# Put identity and the one authoritative summary for an artifact ahead of details.
_KEY_PRIORITY = (
    "format", "version", "schema_version", "method_name", "model_name",
    "dataset_name", "run_id", "build_id", "status", "start_time",
    "end_time", "started_at", "completed_at", "created_at", "updated_at",
    "duration_seconds", "true_duration_seconds", "score_summary",
    "evaluation_coverage", "execution_summary", "build_summary", "llm_usage",
    "efficiency", "artifact_references", "retrieval_records_path", "batch_jobs",
    "summary", "totals", "total",
    "total_queries", "correct_count", "overall_accuracy", "overall_avg_score",
    "avg_score", "total_query_time", "avg_query_time", "avg_retrieved_count",
    "usage", "memory_build_metrics", "memory_size",
    "memory_build_summary", "build_metrics", "feature_configuration",
    "selection", "run_metadata", "config", "config_hash",
    "build_config_hash", "retrieval_config_hash",
    "source_revision", "method_config_name", "dataset_config_name",
    "judge_configuration", "config_inference", "config_sources",
    "method_config", "dataset_config", "execution", "resume_identity_hash",
    "last_invocation_duration_seconds", "batch_pending", "error", "invocations",
    "commit_sha", "branch", "dirty", "method_type", "description", "model",
    "memorize_model", "embedding", "language", "data_root_dir", "data_files",
    "evaluation_mode", "persona_ids", "max_personas", "max_sessions_per_persona",
    "evaluation_interval", "inject_noise", "query_types", "save_intermediate",
    "save_retrieved_context", "build_config", "retrieval_config", "agent_params",
    "effective_agent",
    "source_run_id", "selection", "selected_at", "manifest_path", "memory_identity",
)
_DETAIL_KEYS = (
    "by_context", "by_type", "by_metric", "metric_groups", "by_feature",
    "by_operation", "operations", "units",
    "snapshots", "queries", "records", "results", "detailed_results",
    "session_builds", "memory_state", "jobs", "requests", "responses",
    "items", "invocations", "api_failures",
    "constructor_params", "raw_config",
)
_PRIORITY_INDEX = {key: index for index, key in enumerate(_KEY_PRIORITY)}
_DETAIL_INDEX = {key: index for index, key in enumerate(_DETAIL_KEYS)}
_SCORE_SUMMARY_KEYS = (
    "total_queries", "correct_count", "overall_accuracy", "overall_avg_score",
    "mean_f1", "queries_f1_ge_0_5", "fraction_f1_ge_0_5", "by_type",
    "by_metric", "metric_variants",
)


_SKIP_DEEP_ORDER_KEYS = {
    "requests", "responses", "messages", "memory_state", "channel_semantic_candidates",
    "retrieval_stage_candidates", "merged_semantic_union", "temporally_reranked_union",
    "raw_config",
}


def score_summary(summary: Any) -> dict[str, Any]:
    """Return score headlines without retrieval, timing, or provider detail."""
    if not isinstance(summary, dict):
        return {}
    return {
        key: summary[key]
        for key in _SCORE_SUMMARY_KEYS
        if key in summary
    }


def enrich_json_artifact(value: Any) -> Any:
    """Return an artifact unchanged.

    Writers explicitly choose their summary and detail fields.  Automatically
    copying a nested summary to the root made persisted artifacts look concise
    while silently duplicating scores and usage data.
    """
    return value


def order_json_artifact(value: Any) -> Any:
    """Recursively move summary fields ahead of verbose artifact payloads."""
    if isinstance(value, list):
        return [order_json_artifact(item) for item in value]
    if not isinstance(value, dict):
        return value

    def sort_key(item: tuple[int, tuple[str, Any]]) -> tuple[int, int, int]:
        original_index, (key, _) = item
        if key in _PRIORITY_INDEX:
            return (0, _PRIORITY_INDEX[key], original_index)
        if key in _DETAIL_INDEX:
            return (2, _DETAIL_INDEX[key], original_index)
        return (1, 0, original_index)

    return {
        key: (item if key in _SKIP_DEEP_ORDER_KEYS else order_json_artifact(item))
        for _, (key, item) in sorted(enumerate(value.items()), key=sort_key)
    }


def dump_json_artifact(value: Any, handle: TextIO, *, indent: int = 2) -> None:
    """Write an artifact with stable summary-first ordering without blowing up memory."""
    if not isinstance(value, dict):
        json.dump(
            order_json_artifact(value),
            handle,
            ensure_ascii=False,
            indent=indent,
        )
        handle.write("\n")
        return

    def sort_key(item: tuple[int, tuple[str, Any]]) -> tuple[int, int, int]:
        original_index, (key, _) = item
        if key in _PRIORITY_INDEX:
            return (0, _PRIORITY_INDEX[key], original_index)
        if key in _DETAIL_INDEX:
            return (2, _DETAIL_INDEX[key], original_index)
        return (1, 0, original_index)

    sorted_items = [v for _, v in sorted(enumerate(value.items()), key=sort_key)]
    has_large_list = any(isinstance(v, list) and len(v) > 10 for _, v in sorted_items)
    if not has_large_list:
        json.dump(
            order_json_artifact(value),
            handle,
            ensure_ascii=False,
            indent=indent,
        )
        handle.write("\n")
        return

    # Stream large list fields item-by-item to avoid cloning gigabytes of memory
    handle.write("{\n")
    num_items = len(sorted_items)
    for i, (k, v) in enumerate(sorted_items):
        comma = "," if i < num_items - 1 else ""
        if isinstance(v, list) and len(v) > 10:
            handle.write(f"  {json.dumps(k, ensure_ascii=False)}: [")
            if not v:
                handle.write(f"]{comma}\n")
            else:
                handle.write("\n")
                num_sub = len(v)
                for j, sub_item in enumerate(v):
                    sub_comma = "," if j < num_sub - 1 else ""
                    ordered_sub = sub_item if k in _SKIP_DEEP_ORDER_KEYS else order_json_artifact(sub_item)
                    rendered = json.dumps(ordered_sub, ensure_ascii=False, indent=indent)
                    indented = "\n".join("    " + line for line in rendered.split("\n"))
                    handle.write(f"{indented}{sub_comma}\n")
                handle.write(f"  ]{comma}\n")
        else:
            ordered_v = v if k in _SKIP_DEEP_ORDER_KEYS else order_json_artifact(v)
            rendered = json.dumps(ordered_v, ensure_ascii=False, indent=indent)
            indented = "\n".join("  " + line if idx > 0 else line for idx, line in enumerate(rendered.split("\n")))
            handle.write(f"  {json.dumps(k, ensure_ascii=False)}: {indented}{comma}\n")
    handle.write("}\n")


def rewrite_json_artifact(
    path: Path,
    *,
    indent: int = 2,
    additions: Mapping[str, Any] | None = None,
) -> bool:
    """Atomically rewrite one JSON artifact when summary-first formatting differs."""
    source = path.read_text(encoding="utf-8")
    value = json.loads(source)
    if additions and isinstance(value, dict):
        value = dict(value)
        for key, item in additions.items():
            value.setdefault(key, item)
    ordered = order_json_artifact(value)
    rendered = json.dumps(ordered, ensure_ascii=False, indent=indent) + "\n"
    if source == rendered:
        return False
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_text(rendered, encoding="utf-8")
    temporary_path.replace(path)
    return True
