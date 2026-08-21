#!/usr/bin/env python3
"""Conservatively migrate legacy evaluation artifacts into per-run directories."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
from zoneinfo import ZoneInfo

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
LOCAL_ZONE = ZoneInfo("Asia/Bangkok")
LOG_TIMESTAMP = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
REPORT_PATTERN = re.compile(
    r"^(?P<prefix>.+)_(?P<kind>result|memory_build|query_answer|api_failures)\.json$"
)
REJUDGE_PATTERN = re.compile(r"^(?P<prefix>.+_query_answer)_rejudge_(?P<number>\d+)\.json$")
RUN_DIRECTORY_PATTERN = re.compile(r"^\d{8}_\d{6}(?:_\d+)?$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="Print the migration plan only")
    mode.add_argument("--apply", action="store_true", help="Move all confirmed artifacts")
    parser.add_argument("--outputs", type=Path, default=PROJECT_ROOT / "outputs")
    parser.add_argument("--logs", type=Path, default=PROJECT_ROOT / "logs")
    parser.add_argument(
        "--report",
        type=Path,
        help="Migration report path; defaults inside the outputs directory for --apply",
    )
    return parser.parse_args()


def parse_datetime(value: Any) -> Optional[datetime]:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(LOCAL_ZONE).replace(tzinfo=None)
    return parsed


def run_id(value: datetime) -> str:
    return value.strftime("%Y%m%d_%H%M%S")


def read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def config_hash(method_config: Dict[str, Any], dataset_config: Dict[str, Any]) -> str:
    content = json.dumps(
        {"method": method_config, "dataset": dataset_config},
        sort_keys=True,
    )
    return hashlib.md5(content.encode()).hexdigest()[:16]


def redact(value: Any) -> Any:
    if isinstance(value, dict):
        output = {}
        for key, item in value.items():
            lowered = str(key).lower()
            if any(
                marker in lowered
                for marker in (
                    "api_key", "secret", "password", "credential",
                    "access_token", "private_key",
                )
            ):
                output[key] = "<redacted>" if item else item
            else:
                output[key] = redact(item)
        return output
    if isinstance(value, list):
        return [redact(item) for item in value]
    return value


def resolve_logged_path(raw_value: str) -> Path:
    value = raw_value.strip().strip("'\"")
    path = Path(value)
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def parse_log(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    timestamps = []
    for line in lines:
        match = LOG_TIMESTAMP.match(line)
        if match:
            timestamps.append(datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S"))

    def field(label: str) -> Optional[str]:
        match = re.search(rf"\s{re.escape(label)}:\s*(.+)$", text, re.MULTILINE)
        return match.group(1).strip() if match else None

    explicit_paths: Set[Path] = set()
    for match in re.finditer(
        r"(?:saved(?: separately)? to|Created A-MEM memory run|Created checkpoint):\s*(.+)$",
        text,
        re.MULTILINE,
    ):
        explicit_paths.add(resolve_logged_path(match.group(1)))
    for match in re.finditer(r"restored query-stage memory from\s+(.+)$", text, re.MULTILINE):
        explicit_paths.add(resolve_logged_path(match.group(1)))

    started = None
    start_match = re.search(
        r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*Starting evaluation$",
        text,
        re.MULTILINE,
    )
    if start_match:
        started = datetime.strptime(start_match.group(1), "%Y-%m-%d %H:%M:%S")
    elif timestamps:
        started = timestamps[0]

    stage = field("Execution Stage") or "all"
    return {
        "path": path.resolve(),
        "text": text,
        "start": started,
        "end": timestamps[-1] if timestamps else started,
        "method": field("Method"),
        "provider": field("Provider"),
        "model": field("Model"),
        "dataset": field("Dataset"),
        "dry_run": field("Dry Run") == "True",
        "resume": field("Resume") == "True",
        "force_resume": field("Force Resume") == "True",
        "stage": stage,
        "batch_api": field("Vertex Batch API") == "True",
        "explicit_paths": explicit_paths,
        "completed": "Evaluation completed in" in text,
    }


def safe_experiment_name(method: str, model: str) -> str:
    safe_method = method.replace("/", "-").replace("\\", "-")
    safe_model = model.replace("/", "-").replace("\\", "-")
    return f"{safe_method}_{safe_model}"


def infer_log_experiment(log: Dict[str, Any], outputs_dir: Path) -> Optional[Path]:
    report_paths = [
        path for path in log["explicit_paths"]
        if REPORT_PATTERN.match(path.name) or REJUDGE_PATTERN.match(path.name)
    ]
    if report_paths:
        return report_paths[0].parent
    memory_paths = [path for path in log["explicit_paths"] if "memory" in path.parts]
    for path in memory_paths:
        parts = list(path.parts)
        memory_index = len(parts) - 1 - parts[::-1].index("memory")
        return Path(*parts[:memory_index])
    if log.get("method") and log.get("model"):
        return outputs_dir / safe_experiment_name(log["method"], log["model"])
    return None


def matching_log(
    logs: List[Dict[str, Any]],
    artifacts: Iterable[Path],
    *,
    start: Optional[datetime] = None,
    method: Optional[str] = None,
    model: Optional[str] = None,
    dataset: Optional[str] = None,
) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    artifact_set = {path.resolve() for path in artifacts}
    explicit = [log for log in logs if artifact_set & log["explicit_paths"]]
    if len(explicit) == 1:
        evidence = ["log contains the exact legacy artifact path"]
        if start and explicit[0]["start"] == start.replace(microsecond=0):
            evidence.append("embedded result/manifest start second equals log start second")
        return explicit[0], evidence
    if len(explicit) > 1:
        return None, ["multiple logs contain the same artifact path"]

    if start:
        candidates = [
            log for log in logs
            if log["start"] == start.replace(microsecond=0)
            and (not method or log.get("method") == method)
            and (not model or log.get("model") == model)
            and (not dataset or log.get("dataset") == dataset)
        ]
        if len(candidates) == 1:
            return candidates[0], [
                "embedded result/manifest start second equals the unique matching log"
            ]
    return None, ["no unique exact log match"]


def load_config_catalog() -> Dict[str, Any]:
    from src.config import DatasetConfig, MethodConfig

    method_entries = []
    method_root = PROJECT_ROOT / "configs" / "method_config"
    for path in method_root.rglob("*.yaml"):
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        try:
            effective = asdict(MethodConfig.from_dict(raw))
        except Exception:
            effective = None
        method_entries.append({
            "name": str(path.relative_to(method_root).with_suffix("")),
            "path": path,
            "raw": raw,
            "effective": effective,
        })

    dataset_entries = []
    dataset_root = PROJECT_ROOT / "configs" / "dataset_config"
    for path in dataset_root.rglob("*.yaml"):
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        try:
            effective = asdict(DatasetConfig.from_dict(raw))
        except Exception:
            effective = None
        dataset_entries.append({
            "name": str(path.relative_to(dataset_root).with_suffix("")),
            "path": path,
            "raw": raw,
            "effective": effective,
        })
    return {"methods": method_entries, "datasets": dataset_entries}


def identify_config(
    catalog: Dict[str, Any],
    *,
    method_raw: Optional[Dict[str, Any]],
    dataset_raw: Optional[Dict[str, Any]],
    expected_hash: Optional[str],
    dataset_name: Optional[str],
) -> Dict[str, Any]:
    method_matches = []
    dataset_matches = []
    if method_raw is not None:
        method_matches = [item for item in catalog["methods"] if item["raw"] == method_raw]
    if dataset_raw is not None:
        dataset_matches = [item for item in catalog["datasets"] if item["raw"] == dataset_raw]
    elif dataset_name:
        dataset_matches = [
            item for item in catalog["datasets"]
            if item["raw"].get("dataset_name") == dataset_name
        ]

    if expected_hash and not method_matches and len(dataset_matches) == 1:
        method_matches = [
            item for item in catalog["methods"]
            if config_hash(item["raw"], dataset_matches[0]["raw"]) == expected_hash[:16]
        ]

    method = method_matches[0] if len(method_matches) == 1 else None
    dataset = dataset_matches[0] if len(dataset_matches) == 1 else None
    return {
        "method_name": method["name"] if method else None,
        "method_path": str(method["path"]) if method else None,
        "method_raw": method_raw if method_raw is not None else (method["raw"] if method else None),
        "method_effective": method["effective"] if method else None,
        "dataset_name": dataset["name"] if dataset else None,
        "dataset_path": str(dataset["path"]) if dataset else None,
        "dataset_raw": dataset_raw if dataset_raw is not None else (dataset["raw"] if dataset else None),
        "dataset_effective": dataset["effective"] if dataset else None,
        "method_candidates": [item["name"] for item in method_matches],
        "dataset_candidates": [item["name"] for item in dataset_matches],
    }


def new_plan(experiment: Path, started: datetime) -> Dict[str, Any]:
    target = experiment / run_id(started)
    return {
        "experiment": experiment.resolve(),
        "run_id": target.name,
        "target": target.resolve(),
        "started": started,
        "ended": started,
        "method": None,
        "model": None,
        "dataset": None,
        "provider": None,
        "stage": "all",
        "status": "interrupted",
        "dry_run": False,
        "resume": False,
        "force_resume": False,
        "batch_api": False,
        "config_hash": None,
        "method_raw": None,
        "dataset_raw": None,
        "summary": None,
        "sources": [],
        "source_paths": set(),
        "evidence": [],
        "log": None,
        "memory_manifest": None,
        "memory_source_plan": None,
        "migration_kind": "evaluation",
    }


def get_plan(plans: Dict[Path, Dict[str, Any]], experiment: Path, started: datetime) -> Dict[str, Any]:
    target = (experiment / run_id(started)).resolve()
    if target not in plans:
        plans[target] = new_plan(experiment, started)
    return plans[target]


def add_source(plan: Dict[str, Any], source: Path, destination: Path, reason: str) -> None:
    source = source.resolve()
    destination = destination.resolve()
    if source in plan["source_paths"]:
        return
    plan["source_paths"].add(source)
    plan["sources"].append({
        "source": source,
        "destination": destination,
        "reason": reason,
    })


def apply_log(plan: Dict[str, Any], log: Dict[str, Any], evidence: List[str]) -> None:
    plan["log"] = log
    plan["started"] = log["start"] or plan["started"]
    plan["ended"] = max(filter(None, (plan["ended"], log["end"])))
    for field in ("method", "model", "dataset", "provider"):
        plan[field] = plan[field] or log.get(field)
    plan["stage"] = log.get("stage") or plan["stage"]
    plan["dry_run"] = log.get("dry_run", False)
    plan["resume"] = log.get("resume", False)
    plan["force_resume"] = log.get("force_resume", False)
    plan["batch_api"] = log.get("batch_api", False)
    if log.get("completed"):
        plan["status"] = "complete"
    plan["evidence"].extend(evidence)
    destination = plan["target"] / "evaluation.log"
    add_source(plan, log["path"], destination, "exact run log")


def is_legacy_artifact(path: Path) -> bool:
    parts = set(path.parts)
    if "deduplication_backups" in parts:
        return False
    if any(parent.name == "run_config.json" for parent in path.parents):
        return False
    if RUN_DIRECTORY_PATTERN.match(path.parent.name) and (path.parent / "run_config.json").exists():
        return False
    return True


def discover_report_groups(outputs_dir: Path) -> Tuple[Dict[Tuple[Path, str], List[Path]], List[Path]]:
    groups: Dict[Tuple[Path, str], List[Path]] = {}
    rejudge = []
    for path in outputs_dir.rglob("*.json"):
        if not is_legacy_artifact(path):
            continue
        if any(part in {"batch", "memory", "checkpoints", "deduplication_backups"} for part in path.parts):
            continue
        if REJUDGE_PATTERN.match(path.name):
            rejudge.append(path)
            continue
        match = REPORT_PATTERN.match(path.name)
        if match:
            groups.setdefault((path.parent.resolve(), match.group("prefix")), []).append(path)
    return groups, rejudge


def plan_reports(
    plans: Dict[Path, Dict[str, Any]],
    groups: Dict[Tuple[Path, str], List[Path]],
    logs: List[Dict[str, Any]],
    used_logs: Set[Path],
    leftovers: List[Dict[str, Any]],
) -> None:
    for (experiment, prefix), files in sorted(groups.items(), key=lambda item: str(item[0])):
        result_path = next((path for path in files if path.name.endswith("_result.json")), None)
        result = read_json(result_path) if result_path else None
        metadata = result or read_json(files[0]) or {}
        started = parse_datetime(result.get("start_time")) if result else None
        method = metadata.get("method_name")
        model = metadata.get("model_name")
        dataset = metadata.get("dataset_name")
        log, evidence = matching_log(
            logs,
            files,
            start=started,
            method=method,
            model=model,
            dataset=dataset,
        )
        if started is None and log is not None:
            started = log["start"]
        if started is None:
            leftovers.append({
                "path": str(files[0]),
                "reason": "report group has no result start_time and no exact matching log",
            })
            continue

        plan = get_plan(plans, experiment, started)
        plan["method"] = plan["method"] or method
        plan["model"] = plan["model"] or model
        plan["dataset"] = plan["dataset"] or dataset
        plan["status"] = "complete" if result else plan["status"]
        if result:
            plan["ended"] = parse_datetime(result.get("end_time")) or plan["ended"]
            plan["summary"] = result.get("summary")
            report_config = result.get("config") or {}
            plan["method_raw"] = report_config.get("method_config")
            plan["dataset_raw"] = report_config.get("dataset_config")
            if plan["method_raw"] is not None and plan["dataset_raw"] is not None:
                plan["config_hash"] = config_hash(plan["method_raw"], plan["dataset_raw"])
            plan["evidence"].append("result JSON embeds the exact evaluation start and end timestamps")
        plan["evidence"].append("all report files share one save-operation filename prefix")
        for path in files:
            add_source(plan, path, plan["target"] / path.name, "same report prefix")
        if log is not None:
            apply_log(plan, log, evidence)
            used_logs.add(log["path"])


def memory_experiment_and_files(manifest_path: Path) -> Optional[Tuple[Path, Path, List[Path]]]:
    parts = list(manifest_path.resolve().parts)
    if "memory" not in parts:
        return None
    memory_index = len(parts) - 1 - parts[::-1].index("memory")
    experiment = Path(*parts[:memory_index])
    memory_root = Path(*parts[:memory_index + 1])
    run_memory_dir = manifest_path.parent
    if run_memory_dir == memory_root:
        files = [path for path in memory_root.iterdir() if path.is_file()]
    else:
        files = [path for path in run_memory_dir.rglob("*") if path.is_file()]
    return experiment, run_memory_dir, files


def plan_memory(
    plans: Dict[Path, Dict[str, Any]],
    outputs_dir: Path,
    logs: List[Dict[str, Any]],
    used_logs: Set[Path],
    leftovers: List[Dict[str, Any]],
) -> None:
    for manifest_path in sorted(outputs_dir.rglob("memory/**/manifest.json")):
        if (manifest_path.parent.parent / "run_config.json").exists():
            continue
        manifest = read_json(manifest_path)
        layout = memory_experiment_and_files(manifest_path)
        if not manifest or not layout:
            leftovers.append({"path": str(manifest_path), "reason": "invalid memory manifest"})
            continue
        started = parse_datetime(manifest.get("created_at"))
        if not started:
            leftovers.append({"path": str(manifest_path), "reason": "memory manifest has no created_at"})
            continue
        experiment, memory_dir, files = layout
        log, evidence = matching_log(
            logs,
            [manifest_path, memory_dir],
            start=started,
            method=manifest.get("method_name"),
            model=manifest.get("model_name"),
            dataset="medmemorybench",
        )
        if log is None:
            leftovers.append({
                "path": str(manifest_path),
                "reason": "memory manifest has no unique same-second method/model log",
            })
            continue
        plan = get_plan(plans, experiment, started)
        plan["method"] = plan["method"] or manifest.get("method_name")
        plan["model"] = plan["model"] or manifest.get("model_name")
        plan["dataset"] = plan["dataset"] or "medmemorybench"
        plan["config_hash"] = plan["config_hash"] or manifest.get("config_hash")
        plan["memory_manifest"] = manifest
        if manifest.get("status") == "complete" and log.get("completed"):
            plan["status"] = "complete"
        completed_at = parse_datetime(manifest.get("completed_at"))
        if completed_at:
            plan["ended"] = max(plan["ended"], completed_at)
        plan["evidence"].extend([
            *evidence,
            "memory manifest method/model/config hash is internally consistent",
        ])
        for path in files:
            relative = path.relative_to(memory_dir)
            add_source(plan, path, plan["target"] / "memory" / relative, "matched AMem manifest")
        apply_log(plan, log, evidence)
        used_logs.add(log["path"])


def plan_standalone_logs(
    plans: Dict[Path, Dict[str, Any]],
    logs: List[Dict[str, Any]],
    used_logs: Set[Path],
    outputs_dir: Path,
    leftovers: List[Dict[str, Any]],
) -> None:
    for log in logs:
        if log["path"] in used_logs:
            continue
        if not log.get("start"):
            leftovers.append({"path": str(log["path"]), "reason": "log has no start timestamp"})
            continue
        experiment = infer_log_experiment(log, outputs_dir)
        if experiment is None:
            leftovers.append({"path": str(log["path"]), "reason": "log has no method/model experiment identity"})
            continue
        try:
            experiment.resolve().relative_to(outputs_dir.resolve())
        except ValueError:
            leftovers.append({
                "path": str(log["path"]),
                "reason": f"log points to an output directory outside {outputs_dir}",
            })
            continue
        plan = get_plan(plans, experiment, log["start"])
        apply_log(plan, log, [
            "log filename timestamp and first Starting evaluation timestamp agree",
            "log embeds method, model, dataset, and execution-stage identity",
        ])
        used_logs.add(log["path"])


def plan_checkpoints(
    plans: Dict[Path, Dict[str, Any]],
    outputs_dir: Path,
    logs: List[Dict[str, Any]],
    leftovers: List[Dict[str, Any]],
) -> None:
    checkpoint_root = outputs_dir / "checkpoints"
    if not checkpoint_root.exists():
        return
    grouped: Dict[Tuple[str, str, str], List[Path]] = {}
    for path in checkpoint_root.rglob("checkpoint*.json"):
        data = read_json(path)
        if not data:
            continue
        key = (data.get("checkpoint_id", ""), data.get("method_name", ""), data.get("model_name", ""))
        grouped.setdefault(key, []).append(path)

    for (_, method, model), files in grouped.items():
        checkpoint = read_json(next((p for p in files if p.name == "checkpoint.json"), files[0])) or {}
        created = parse_datetime(checkpoint.get("created_at"))
        candidates = [
            log for log in logs
            if created and log.get("start")
            and abs((log["start"] - created).total_seconds()) <= 2
            and log.get("method") == method
            and log.get("model") == model
        ]
        if len(candidates) != 1:
            leftovers.append({
                "path": str(files[0]),
                "reason": "checkpoint has no unique same-time method/model log",
            })
            continue
        log = candidates[0]
        experiment = infer_log_experiment(log, outputs_dir)
        plan = get_plan(plans, experiment, log["start"])
        plan["config_hash"] = plan["config_hash"] or checkpoint.get("config_hash")
        plan["evidence"].append(
            "checkpoint creation time, checkpoint ID, method, model, and log start agree"
        )
        for path in files:
            add_source(plan, path, plan["target"] / "checkpoints" / path.name, "matched checkpoint")


def plan_rejudge(
    plans: Dict[Path, Dict[str, Any]],
    paths: List[Path],
    leftovers: List[Dict[str, Any]],
) -> Set[Path]:
    used_batch: Set[Path] = set()
    for path in paths:
        data = read_json(path)
        rejudge = (data or {}).get("rejudge")
        if not isinstance(rejudge, dict):
            leftovers.append({"path": str(path), "reason": "rejudge output has no rejudge metadata"})
            continue
        batch_path = None
        raw_batch_path = rejudge.get("batch_manifest")
        if raw_batch_path:
            candidate = resolve_logged_path(raw_batch_path)
            if candidate.exists():
                batch_path = candidate
        batch_data = read_json(batch_path) if batch_path else None
        started = parse_datetime((batch_data or {}).get("created_at"))
        if started is None:
            started = parse_datetime(rejudge.get("created_at"))
        if started is None:
            leftovers.append({"path": str(path), "reason": "rejudge output has no recoverable start"})
            continue
        plan = get_plan(plans, path.parent, started)
        plan["migration_kind"] = "rejudge"
        plan["stage"] = "rejudge"
        plan["status"] = "complete"
        plan["method"] = (data or {}).get("method_name")
        plan["model"] = (data or {}).get("model_name")
        plan["dataset"] = (data or {}).get("dataset_name")
        plan["batch_api"] = bool(rejudge.get("batch_api"))
        plan["summary"] = (data or {}).get("summary")
        plan["evidence"].append("rejudge output embeds source, judge configuration, and run number")
        add_source(plan, path, plan["target"] / path.name, "rejudge output")
        if batch_path and batch_data:
            add_source(plan, batch_path, plan["target"] / "batch" / batch_path.name, "exact batch_manifest path embedded in rejudge output")
            used_batch.add(batch_path.resolve())
            plan["evidence"].append("rejudge output points to this exact batch manifest")
        plan["rejudge"] = rejudge
    return used_batch


def plan_batch_manifests(
    plans: Dict[Path, Dict[str, Any]],
    outputs_dir: Path,
    used_batch: Set[Path],
    leftovers: List[Dict[str, Any]],
) -> None:
    for path in sorted(outputs_dir.rglob("*batch_manifest*.json")):
        resolved = path.resolve()
        if resolved in used_batch or (path.parent.parent / "run_config.json").exists():
            continue
        data = read_json(path)
        created = parse_datetime((data or {}).get("created_at"))
        if not data or not created:
            leftovers.append({"path": str(path), "reason": "batch manifest has no created_at"})
            continue
        stem = path.name.lower()
        manifest_hash = str(data.get("config_hash", ""))
        candidates = []
        for plan in plans.values():
            if plan["migration_kind"] != "evaluation" or not plan.get("log"):
                continue
            log = plan["log"]
            interval_end = max(plan["ended"], log.get("end") or plan["ended"]) + timedelta(minutes=2)
            if not (plan["started"] <= created <= interval_end):
                continue
            if "judge_batch" in stem and plan["stage"] not in {"all", "query"}:
                continue
            if "graphrag" in stem and plan.get("method") != "graph_rag":
                continue
            if "lightmem" in stem and plan.get("method") != "lightmem":
                continue
            candidates.append(plan)

        exact_hash = [
            plan for plan in candidates
            if plan.get("config_hash") and manifest_hash.startswith(plan["config_hash"])
        ]
        selected = exact_hash if len(exact_hash) == 1 else candidates
        if len(selected) != 1:
            leftovers.append({
                "path": str(path),
                "reason": f"batch manifest matches {len(selected)} possible run intervals",
            })
            continue
        plan = selected[0]
        reason = (
            "batch config hash prefix and creation interval match this unique run"
            if exact_hash
            else "batch creation interval and stage match this unique run"
        )
        add_source(plan, path, plan["target"] / "batch" / path.name, reason)
        plan["evidence"].append(reason)


def attach_memory_sources(plans: Dict[Path, Dict[str, Any]]) -> None:
    memory_plans = [
        plan for plan in plans.values()
        if plan.get("memory_manifest")
        and plan["memory_manifest"].get("status") == "complete"
    ]
    for plan in plans.values():
        if plan.get("stage") != "query" or not plan.get("log"):
            continue
        log_text = plan["log"].get("text", "")
        if "restored query-stage memory from" not in log_text and "Using A-MEM memory run" not in log_text:
            continue
        candidates = [
            memory_plan for memory_plan in memory_plans
            if memory_plan["experiment"] == plan["experiment"]
            and memory_plan["started"] < plan["started"]
            and (
                not plan.get("config_hash")
                or memory_plan.get("config_hash") == plan.get("config_hash")
            )
        ]
        if not candidates:
            continue
        candidates.sort(key=lambda item: item["started"], reverse=True)
        source_plan = candidates[0]
        if len(candidates) > 1 and candidates[0]["started"] == candidates[1]["started"]:
            continue
        plan["memory_source_plan"] = source_plan
        plan["evidence"].append(
            "query log restores legacy snapshot paths and the selected source is the newest earlier compatible complete manifest"
        )


def attach_configs(plans: Dict[Path, Dict[str, Any]], catalog: Dict[str, Any]) -> None:
    for plan in plans.values():
        plan["identified_config"] = identify_config(
            catalog,
            method_raw=plan.get("method_raw"),
            dataset_raw=plan.get("dataset_raw"),
            expected_hash=plan.get("config_hash"),
            dataset_name=plan.get("dataset"),
        )


def serialize_plan(plan: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "target": str(plan["target"]),
        "run_id": plan["run_id"],
        "started_at": plan["started"].isoformat(),
        "ended_at": plan["ended"].isoformat() if plan.get("ended") else None,
        "method": plan.get("method"),
        "model": plan.get("model"),
        "dataset": plan.get("dataset"),
        "stage": plan.get("stage"),
        "status": plan.get("status"),
        "config_hash": plan.get("config_hash"),
        "identified_config": redact(plan.get("identified_config")),
        "memory_source_run_id": (
            plan["memory_source_plan"]["run_id"]
            if plan.get("memory_source_plan") else None
        ),
        "evidence": sorted(set(plan["evidence"])),
        "moves": [
            {
                "source": str(item["source"]),
                "destination": str(item["destination"]),
                "reason": item["reason"],
            }
            for item in plan["sources"]
        ],
    }


def run_config_payload(plan: Dict[str, Any], move_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    config = plan["identified_config"]
    complete_config = bool(
        config.get("method_name")
        and config.get("dataset_name")
        and config.get("method_effective")
        and config.get("dataset_effective")
    )
    payload = {
        "format": "medmemorybench.run_config",
        "version": 1,
        "run_id": plan["run_id"],
        "status": plan["status"],
        "started_at": plan["started"].isoformat(),
        "updated_at": datetime.now().isoformat(),
        "completed_at": plan["ended"].isoformat() if plan["status"] == "complete" else None,
        "method_config_name": config.get("method_name"),
        "dataset_config_name": config.get("dataset_name"),
        "method_config": redact(config.get("method_effective") or plan.get("method_raw") or {}),
        "dataset_config": redact(config.get("dataset_effective") or plan.get("dataset_raw") or {}),
        "config_sources": {
            "method": config.get("method_path"),
            "dataset": config.get("dataset_path"),
        },
        "execution": {
            "stage": plan["stage"],
            "dry_run": plan["dry_run"],
            "resume": plan["resume"],
            "force_resume": plan["force_resume"],
            "memory_run": (
                plan["memory_source_plan"]["run_id"]
                if plan.get("memory_source_plan") else None
            ),
            "batch_api": plan["batch_api"],
            "batch_gcs_uri": None,
            "batch_wait": None,
        },
        "command": None,
        "output_dir": str(plan["target"]),
        "summary": plan.get("summary"),
        "migration": {
            "migrated_at": datetime.now().isoformat(),
            "confidence": "confirmed",
            "configuration_complete": complete_config,
            "configuration_note": (
                "Exact method/dataset YAML was identified; historical environment and command-line secrets cannot be reconstructed."
                if complete_config else
                "Only the configuration embedded in artifacts could be recovered; historical environment and command line are unavailable."
            ),
            "config_hash": plan.get("config_hash"),
            "evidence": sorted(set(plan["evidence"])),
            "moves": move_records,
        },
    }
    if plan.get("memory_manifest"):
        payload["memory"] = {
            key: plan["memory_manifest"].get(key)
            for key in ("build_id", "config_hash", "status", "created_at", "completed_at")
        }
    if plan.get("memory_source_plan"):
        source_plan = plan["memory_source_plan"]
        source_manifest = source_plan["memory_manifest"]
        payload["memory_source"] = {
            "source_run_id": source_plan["run_id"],
            "build_id": source_manifest.get("build_id"),
            "config_hash": source_manifest.get("config_hash"),
            "status": source_manifest.get("status"),
        }
    if plan.get("rejudge"):
        payload["rejudge"] = redact(plan["rejudge"])
    return payload


def write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def apply_plans(plans: Dict[Path, Dict[str, Any]]) -> List[Dict[str, Any]]:
    destinations: Set[Path] = set()
    sources: Set[Path] = set()
    for plan in plans.values():
        config_path = plan["target"] / "run_config.json"
        if config_path.exists():
            raise FileExistsError(f"Refusing to overwrite existing run config: {config_path}")
        for item in plan["sources"]:
            source = item["source"]
            destination = item["destination"]
            if not source.exists():
                raise FileNotFoundError(f"Migration source disappeared: {source}")
            if source in sources:
                raise ValueError(f"Migration source appears in multiple plans: {source}")
            if destination in destinations:
                raise ValueError(f"Migration destination appears more than once: {destination}")
            if destination.exists():
                raise FileExistsError(f"Migration destination already exists: {destination}")
            sources.add(source)
            destinations.add(destination)
        if plan.get("memory_source_plan"):
            memory_source_path = plan["target"] / "memory_source.json"
            if memory_source_path.exists():
                raise FileExistsError(
                    f"Migration destination already exists: {memory_source_path}"
                )

    records = []
    for plan in sorted(plans.values(), key=lambda item: str(item["target"])):
        plan["target"].mkdir(parents=True, exist_ok=True)
        move_records = []
        for item in plan["sources"]:
            source = item["source"]
            destination = item["destination"]
            source_hash = sha256(source)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(destination))
            destination_hash = sha256(destination)
            if destination_hash != source_hash:
                raise RuntimeError(f"Hash changed while moving {source} to {destination}")
            record = {
                "source": str(source),
                "destination": str(destination),
                "sha256": source_hash,
                "verified": True,
                "reason": item["reason"],
            }
            move_records.append(record)
            records.append(record)

        if plan.get("memory_source_plan"):
            source_plan = plan["memory_source_plan"]
            source_manifest = source_plan["memory_manifest"]
            relative_manifest = os.path.relpath(
                source_plan["target"] / "memory" / "manifest.json",
                plan["target"],
            )
            memory_source_payload = {
                "format": "medmemorybench.memory_source",
                "version": 1,
                "selected_at": plan["started"].isoformat(),
                "selection": "migrated_from_legacy_query_log",
                "source_run_id": source_plan["run_id"],
                "manifest_path": relative_manifest,
                "build_id": source_manifest.get("build_id"),
                "config_hash": source_manifest.get("config_hash"),
                "status": source_manifest.get("status"),
            }
            write_json_atomic(
                plan["target"] / "memory_source.json",
                memory_source_payload,
            )

        config_path = plan["target"] / "run_config.json"
        write_json_atomic(config_path, run_config_payload(plan, move_records))

    for path in sorted(
        {Path(record["source"]).parent for record in records},
        key=lambda candidate: len(candidate.parts),
        reverse=True,
    ):
        try:
            path.rmdir()
        except OSError:
            pass
    return records


def verify_applied(plans: Dict[Path, Dict[str, Any]], records: List[Dict[str, Any]]) -> Dict[str, Any]:
    errors = []
    for record in records:
        source = Path(record["source"])
        destination = Path(record["destination"])
        if source.exists():
            errors.append(f"source still exists: {source}")
        if not destination.exists() or sha256(destination) != record["sha256"]:
            errors.append(f"destination hash mismatch: {destination}")
    for plan in plans.values():
        config_path = plan["target"] / "run_config.json"
        if not config_path.exists():
            errors.append(f"missing run config: {config_path}")
        if plan.get("memory_source_plan") and not (plan["target"] / "memory_source.json").exists():
            errors.append(f"missing memory source: {plan['target'] / 'memory_source.json'}")
        manifest_path = plan["target"] / "memory" / "manifest.json"
        if manifest_path.exists():
            manifest = read_json(manifest_path) or {}
            if plan.get("config_hash") and manifest.get("config_hash") != plan["config_hash"]:
                errors.append(f"memory config hash mismatch: {manifest_path}")
    return {"ok": not errors, "errors": errors}


def main() -> int:
    args = parse_args()
    outputs_dir = args.outputs.resolve()
    logs_dir = args.logs.resolve()
    logs = [parse_log(path) for path in sorted(logs_dir.glob("*.log"))]
    plans: Dict[Path, Dict[str, Any]] = {}
    leftovers: List[Dict[str, Any]] = []
    used_logs: Set[Path] = set()

    report_groups, rejudge_paths = discover_report_groups(outputs_dir)
    plan_reports(plans, report_groups, logs, used_logs, leftovers)
    plan_memory(plans, outputs_dir, logs, used_logs, leftovers)
    plan_standalone_logs(plans, logs, used_logs, outputs_dir, leftovers)
    plan_checkpoints(plans, outputs_dir, logs, leftovers)
    used_batch = plan_rejudge(plans, rejudge_paths, leftovers)
    plan_batch_manifests(plans, outputs_dir, used_batch, leftovers)
    attach_memory_sources(plans)
    attach_configs(plans, load_config_catalog())

    serialized = [serialize_plan(plan) for plan in sorted(plans.values(), key=lambda item: str(item["target"]))]
    planned_moves = sum(len(plan["moves"]) for plan in serialized)
    report = {
        "format": "medmemorybench.artifact_migration",
        "version": 1,
        "mode": "apply" if args.apply else "dry-run",
        "created_at": datetime.now().isoformat(),
        "summary": {
            "confirmed_runs": len(serialized),
            "planned_moves": planned_moves,
            "leftovers": len(leftovers),
        },
        "runs": serialized,
        "leftovers": leftovers,
    }

    if args.dry_run:
        print(json.dumps(report["summary"], indent=2))
        for item in serialized:
            print(
                f"{item['run_id']} {item['method']}/{item['model']} "
                f"stage={item['stage']} files={len(item['moves'])} -> {item['target']}"
            )
        print(f"Ambiguous/unmatched artifacts left in place: {len(leftovers)}")
        for item in leftovers:
            print(f"LEFT: {item['path']} ({item['reason']})")
        if args.report:
            write_json_atomic(args.report.resolve(), report)
        return 0

    records = apply_plans(plans)
    verification = verify_applied(plans, records)
    report["verification"] = verification
    report_path = (
        args.report.resolve()
        if args.report
        else outputs_dir / f"run_artifact_migration_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    write_json_atomic(report_path, report)
    print(json.dumps({
        **report["summary"],
        "verification_ok": verification["ok"],
        "report": str(report_path),
    }, indent=2))
    return 0 if verification["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
