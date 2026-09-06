#!/usr/bin/env python3
"""Normalize report artifacts into non-overlapping, summary-first schemas."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from utils.json_artifacts import order_json_artifact, score_summary


_USAGE_TOTAL_FIELDS = (
    "input_tokens", "output_tokens", "visible_output_tokens", "thinking_tokens",
    "total_tokens", "call_count", "successful_calls", "attempted_calls",
    "failed_attempts", "retry_count", "operation_count", "total_latency",
    "wall_time", "failure_duration_seconds",
)


def _usage_for_phases(
    usage: object,
    phases: tuple[str, ...],
    *,
    include_total: bool = True,
) -> dict[str, object]:
    """Return a token ledger containing only the named execution phases."""
    if not isinstance(usage, dict):
        return {}
    selected: dict[str, object] = {}
    totals = {field: 0 for field in _USAGE_TOTAL_FIELDS}
    for phase in phases:
        phase_usage = usage.get(f"{phase}_phase")
        if not isinstance(phase_usage, dict):
            continue
        selected[f"{phase}_phase"] = phase_usage
        for field in _USAGE_TOTAL_FIELDS:
            value = phase_usage.get(field, 0)
            if isinstance(value, (int, float)):
                totals[field] += value
    operations = usage.get("operations")
    if isinstance(operations, dict):
        selected_operations = {
            phase: operations[phase]
            for phase in phases
            if isinstance(operations.get(phase), dict)
        }
        if selected_operations:
            selected["operations"] = selected_operations
    if selected and include_total:
        successful_calls = totals["successful_calls"]
        totals["avg_latency"] = (
            totals["total_latency"] / successful_calls if successful_calls else 0.0
        )
        selected["total"] = totals
    return selected


def _references(path: Path, root: Path) -> dict[str, str]:
    """Link related artifacts without reproducing their payloads."""
    suffixes = {
        "result": "_result.json",
        "memory_build": "_memory_build.json",
        "query_answer": "_query_answer.json",
    }
    own_name = next((name for name, suffix in suffixes.items() if path.name.endswith(suffix)), None)
    stem = path.name[:-len(suffixes[own_name])] if own_name else ""
    references: dict[str, str] = {}
    for name, suffix in suffixes.items():
        if name == own_name:
            continue
        sibling = path.with_name(f"{stem}{suffix}")
        if sibling.is_file():
            references[name] = sibling.name
    if own_name in {"result", "query_answer"} and "memory_build" not in references:
        build_paths = sorted(root.glob("*_memory_build.json"))
        if len(build_paths) == 1:
            references["memory_build"] = os.path.relpath(build_paths[0], path.parent)
    if (path.parent / "run_config.json").is_file():
        references["run_config"] = "run_config.json"
    if (path.parent / "memory_source.json").is_file():
        references["memory_source"] = "memory_source.json"
    return references


def _migrate_result(data: dict[str, object], path: Path, root: Path) -> dict[str, object]:
    summary = data.get("score_summary")
    if not isinstance(summary, dict):
        summary = score_summary(data.get("summary"))
    output = {
        key: data[key]
        for key in (
            "method_name", "model_name", "dataset_name", "start_time", "end_time",
            "duration_seconds", "true_duration_seconds",
        )
        if key in data
    }
    output["score_summary"] = summary
    for key in ("evaluation_coverage", "run_metadata", "dataset_coverage", "input_modality"):
        if key in data:
            output[key] = data[key]
    output["artifact_references"] = _references(path, root)
    return output


def _migrate_query_answer(data: dict[str, object], path: Path, root: Path) -> dict[str, object]:
    summary = data.get("execution_summary")
    legacy_summary = data.get("summary")
    if not isinstance(summary, dict):
        source = legacy_summary if isinstance(legacy_summary, dict) else {}
        summary = {
            key: source[key]
            for key in ("total_queries", "total_query_time", "avg_query_time", "avg_retrieved_count")
            if key in source
        }
    stage_usage = data.get("stage_usage")
    if isinstance(stage_usage, dict):
        legacy_efficiency = legacy_summary.get("efficiency", {}) if isinstance(legacy_summary, dict) else {}
        for key in ("query_time_kind", "stage_wall_time_seconds", "batch_stage_count"):
            if key in legacy_efficiency and key not in summary:
                summary[key] = legacy_efficiency[key]
    batch_jobs = data.get("batch_jobs")
    if not isinstance(batch_jobs, list):
        batch_jobs = []
        if isinstance(stage_usage, dict):
            batch_jobs = [
                {key: value for key, value in item.items() if key != "token_usage"}
                for item in stage_usage.get("batch_stages", [])
                if isinstance(item, dict)
            ]
    output = {
        key: data[key]
        for key in ("format", "version", "method_name", "model_name", "dataset_name")
        if key in data
    }
    output["execution_summary"] = summary
    output["llm_usage"] = _usage_for_phases(data.get("llm_usage", {}), ("query", "judge"))
    output["artifact_references"] = _references(path, root)
    if batch_jobs:
        output["batch_jobs"] = batch_jobs
    for key in ("retrieval_records_path", "by_context", "queries"):
        if key in data:
            output[key] = data[key]
    return output


def _migrate_memory_build(data: dict[str, object], path: Path, root: Path) -> dict[str, object]:
    summary = data.get("build_summary")
    if isinstance(summary, dict):
        summary = dict(summary)
    else:
        summary = dict(data.get("summary", {})) if isinstance(data.get("summary"), dict) else {}
        summary["total_units"] = data.get("total_units", len(data.get("units", [])))
        for key in ("memory_chunk_size", "memory_size", "feature_configuration"):
            if key in data:
                summary[key] = data[key]
    metrics = data.get("build_metrics")
    raw_build_usage = metrics.get("usage") if isinstance(metrics, dict) else None
    if not isinstance(raw_build_usage, dict):
        raw_build_usage = summary.get("llm_usage")
    if not isinstance(raw_build_usage, dict):
        raw_build_usage = data.get("llm_usage")
    build_usage = _usage_for_phases(
        raw_build_usage, ("memorize",), include_total=False
    )
    if build_usage:
        summary["llm_usage"] = build_usage
    elif isinstance(raw_build_usage, dict) and raw_build_usage:
        summary["llm_usage"] = raw_build_usage
    detailed_metrics = {
        key: value
        for key, value in metrics.items()
        if key not in {"feature_configuration", "memory_size", "usage", "totals", "unit_count", "session_count"}
    } if isinstance(metrics, dict) else {}
    output = {
        key: data[key]
        for key in ("method_name", "model_name", "dataset_name")
        if key in data
    }
    output["build_summary"] = summary
    output["build_metrics"] = detailed_metrics
    if "run_metadata" in data:
        output["run_metadata"] = data["run_metadata"]
    output["artifact_references"] = _references(path, root)
    if "units" in data:
        output["units"] = data["units"]
    return output


def _compact_method_config(snapshot: object) -> object:
    """Remove an agent-parameter mirror when scoped configs fully define it."""
    if not isinstance(snapshot, dict):
        return snapshot
    compact = dict(snapshot)
    agent_params = compact.get("agent_params")
    build_config = compact.get("build_config")
    retrieval_config = compact.get("retrieval_config")
    if isinstance(agent_params, dict) and isinstance(build_config, dict) and isinstance(retrieval_config, dict):
        if agent_params == {**build_config, **retrieval_config}:
            compact.pop("agent_params")
    return compact


def _migrate_run_config(data: dict[str, object]) -> dict[str, object]:
    """Keep reproducible configuration and lifecycle audit data once."""
    output = {
        key: data[key]
        for key in (
            "format", "run_id", "status", "started_at", "completed_at", "updated_at",
            "duration_seconds", "last_invocation_duration_seconds",
        )
        if key in data
    }
    output["version"] = 2
    source_revision = data.get("source_revision")
    if not isinstance(source_revision, dict):
        source_revision = {
            key.removeprefix("git_"): data.get(key)
            for key in ("git_commit_sha", "git_dirty", "git_branch")
            if key in data
        }
    if source_revision:
        output["source_revision"] = source_revision
    for key in ("method_config_name", "dataset_config_name", "config_inference", "config_sources"):
        if key in data:
            output[key] = data[key]
    judge_configuration = data.get("judge_configuration")
    if not isinstance(judge_configuration, dict):
        api_config = data.get("api_config")
        if isinstance(api_config, dict):
            judge_configuration = {
                "provider": api_config.get("judge_provider"),
                "model": api_config.get("judge_model"),
                "max_tokens": api_config.get("judge_max_tokens"),
                "mcd_max_tokens": api_config.get("judge_mcd_max_tokens"),
                "client_max_tokens": api_config.get("judge_client_max_tokens"),
                "temperature": api_config.get("judge_temperature"),
                "reasoning_effort": api_config.get("judge_reasoning_effort"),
            }
    if isinstance(judge_configuration, dict):
        output["judge_configuration"] = judge_configuration
    if "method_config" in data:
        output["method_config"] = _compact_method_config(data["method_config"])
    if "dataset_config" in data:
        output["dataset_config"] = data["dataset_config"]
    for key in ("execution", "resume_identity_hash", "batch_pending", "error", "invocations"):
        if key in data:
            output[key] = data[key]
    return output


def _migrate_memory_source(data: dict[str, object]) -> dict[str, object]:
    """Keep a compact source selection; the manifest owns build telemetry."""
    output = {
        key: data[key]
        for key in ("format", "source_run_id", "selection", "selected_at", "manifest_path")
        if key in data
    }
    output["version"] = 2
    identity = data.get("memory_identity")
    if not isinstance(identity, dict):
        identity = {
            key: data.get(key)
            for key in ("build_id", "status", "config_hash", "build_config_hash")
            if key in data
        }
    if identity:
        output["memory_identity"] = identity
    return output


def migrate_artifact(data: object, path: Path, root: Path) -> object:
    """Assign each persisted report type one authoritative set of fields."""
    if not isinstance(data, dict):
        return data
    if path.name == "run_config.json":
        return _migrate_run_config(data)
    if path.name == "memory_source.json":
        return _migrate_memory_source(data)
    if path.name.endswith("_result.json"):
        return _migrate_result(data, path, root)
    if path.name.endswith("_query_answer.json"):
        return _migrate_query_answer(data, path, root)
    if path.name.endswith("_memory_build.json"):
        return _migrate_memory_build(data, path, root)
    return data


def _rewrite(path: Path, root: Path) -> bool:
    source = path.read_text(encoding="utf-8")
    value = migrate_artifact(json.loads(source), path, root)
    rendered = json.dumps(order_json_artifact(value), ensure_ascii=False, indent=2) + "\n"
    if source == rendered:
        return False
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_text(rendered, encoding="utf-8")
    temporary_path.replace(path)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Run directory to rewrite recursively")
    args = parser.parse_args()
    root = args.root.resolve()
    if not root.is_dir():
        parser.error(f"not a directory: {root}")

    total = 0
    rewritten = 0
    for path in sorted(root.rglob("*.json")):
        total += 1
        rewritten += int(
            _rewrite(path, root)
        )
    print(f"JSON artifacts checked: {total}; reformatted: {rewritten}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
