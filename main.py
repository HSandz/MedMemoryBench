#!/usr/bin/env python3
"""
Personalization Lab - Memory Evaluation System
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    ConfigLoader,
    PROJECT_ROOT,
    dataset_config_from_snapshot,
    method_config_from_snapshot,
)
from src.evaluator import create_evaluator
from src.agent import list_available_methods
from benchmarks.medmemorybench.rejudge import rejudge_medmemorybench
from utils.logger import format_limited_traceback, truncate_error_message


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Personalization Lab - Memory Evaluation System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "-m",
        "--method",
        type=str,
        help="Method config name or YAML config file (query stage may override the build config)",
    )
    parser.add_argument("-d", "--dataset", type=str, help="Dataset name")
    parser.add_argument("-o", "--output-dir", type=str, default="./outputs", help="Output directory")
    parser.add_argument("--dry-run", action="store_true", help="Dry run mode")
    parser.add_argument("-q", "--quiet", action="store_true", help="Quiet mode")
    parser.add_argument("--list-methods", action="store_true", help="List available methods")
    parser.add_argument("--list-datasets", action="store_true", help="List available datasets")
    parser.add_argument("--list-agents", action="store_true", help="List available agent types")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument(
        "--stage",
        choices=("all", "memory", "query"),
        default="all",
        help=(
            "Run the full evaluation, build AMem snapshots only, or answer/score "
            "from existing AMem snapshots"
        ),
    )
    parser.add_argument(
        "--memory-run",
        "--memory_run",
        dest="memory_run",
        type=str,
        metavar="TIMESTAMP",
        help=(
            "Run directory to use for --stage query or --append; append also accepts "
            "an in-progress memory run. When a run is specified, omitted -m/-d "
            "values are inferred from its run_config.json. Use 'legacy' for "
            "snapshots stored directly in the old memory directory. Derived runs "
            "are written under <memory-run>/query_runs/"
        ),
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help=(
            "Build incrementally from --memory-run through the requested --persona "
            "and --unit, writing a derived run under its query_runs directory"
        ),
    )
    parser.add_argument(
        "--persona",
        type=int,
        metavar="PERSONA_ID",
        help="Target persona for --append",
    )
    parser.add_argument(
        "--unit",
        type=int,
        metavar="UNIT_ID",
        help="Target evaluation-unit ID for --append",
    )
    parser.add_argument(
        "--rejudge",
        type=str,
        metavar="QUERY_ANSWER_JSON",
        help="Re-run LLM judge metrics from a saved query_answer.json file",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Force resume after config changes and skip sessions already recorded as "
            "injected; requires the memory backend to retain its state"
        ),
    )
    parser.add_argument(
        "--batch-api",
        action="store_true",
        help="Use the configured provider's Batch API for eligible offline stages",
    )
    parser.add_argument(
        "--batch-gcs-uri",
        type=str,
        help="Cloud Storage staging prefix (defaults to GOOGLE_BATCH_GCS_URI)",
    )
    parser.add_argument(
        "--batch-wait",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Wait for Batch API jobs to complete (default: true)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        metavar="N",
        help=(
            "Maximum concurrent workers for independent query/build preparation "
            "tasks (default: 1)"
        ),
    )

    args = parser.parse_args()
    if args.force and not args.resume:
        parser.error("--force requires --resume")
    if args.stage == "query" and args.dry_run:
        parser.error("--stage query cannot be combined with --dry-run")
    if args.memory_run and args.stage != "query" and not args.append:
        parser.error("--memory-run requires --stage query")
    if args.append and not args.memory_run:
        parser.error("--append requires --memory-run")
    if args.append and args.stage == "query":
        parser.error("--append cannot be combined with --stage query")
    if args.append and args.dry_run:
        parser.error("--append cannot be combined with --dry-run")
    if args.append and (args.persona is None or args.unit is None):
        parser.error("--append requires both --persona and --unit")
    if args.append and (args.persona < 0 or args.unit < 0):
        parser.error("--persona and --unit must be non-negative")
    if not args.append and (args.persona is not None or args.unit is not None):
        parser.error("--persona and --unit require --append")
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    return args


def infer_query_config_from_memory_run(
    output_dir: Path,
    memory_run: str,
    *,
    allow_incomplete: bool = False,
) -> Dict[str, Any]:
    """Load identity and effective config snapshots from one memory run."""
    if memory_run == "legacy":
        raise ValueError(
            "Cannot infer configuration from the legacy memory layout; "
            "specify -m and -d explicitly"
        )
    if Path(memory_run).name != memory_run:
        raise ValueError("--memory-run must be a run timestamp directory name")

    output_root = Path(output_dir).resolve()
    if not output_root.exists():
        raise FileNotFoundError(f"Output directory not found: {output_root}")

    candidates = []
    rejected = []
    for manifest_path in output_root.rglob("manifest.json"):
        if manifest_path.parent.name != "memory":
            continue
        run_dir = manifest_path.parent.parent
        if run_dir.name != memory_run:
            continue
        run_config_path = run_dir / "run_config.json"
        if not run_config_path.exists():
            rejected.append(f"{run_dir}: missing run_config.json")
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            rejected.append(f"{run_dir}: invalid JSON ({truncate_error_message(exc)})")
            continue

        if not isinstance(manifest, dict) or not isinstance(run_config, dict):
            rejected.append(f"{run_dir}: manifest and run config must be JSON objects")
            continue
        if (
            manifest.get("format") != "medmemorybench.memory_manifest"
            or manifest.get("version") != 1
        ):
            rejected.append(f"{run_dir}: unsupported memory manifest")
            continue
        if (
            run_config.get("format") != "medmemorybench.run_config"
            or run_config.get("version") != 1
        ):
            rejected.append(f"{run_dir}: unsupported run_config.json")
            continue
        allowed_statuses = {"complete", "building"} if allow_incomplete else {"complete"}
        if manifest.get("status") not in allowed_statuses:
            rejected.append(
                f"{run_dir}: memory status is {manifest.get('status')!r}, not 'complete'"
            )
            continue

        method_config_name = run_config.get("method_config_name")
        dataset_config_name = run_config.get("dataset_config_name")
        if not isinstance(method_config_name, str) or not method_config_name:
            rejected.append(f"{run_dir}: method_config_name is unavailable")
            continue
        if not isinstance(dataset_config_name, str) or not dataset_config_name:
            rejected.append(f"{run_dir}: dataset_config_name is unavailable")
            continue

        stored_method = run_config.get("method_config", {})
        if not isinstance(stored_method, dict):
            rejected.append(f"{run_dir}: stored method configuration is invalid")
            continue
        stored_method_name = stored_method.get("method_name")
        stored_model = stored_method.get("model", {})
        if not isinstance(stored_model, dict):
            rejected.append(f"{run_dir}: stored model configuration is invalid")
            continue
        stored_model_name = stored_model.get("name")
        if stored_method_name and stored_method_name != manifest.get("method_name"):
            rejected.append(f"{run_dir}: method identity disagrees with its manifest")
            continue
        if stored_model_name and stored_model_name != manifest.get("model_name"):
            rejected.append(f"{run_dir}: model identity disagrees with its manifest")
            continue

        stored_dataset = run_config.get("dataset_config")
        if not isinstance(stored_dataset, dict):
            rejected.append(f"{run_dir}: stored dataset configuration is unavailable")
            continue
        if not isinstance(stored_method.get("raw_config"), dict):
            rejected.append(f"{run_dir}: stored method raw_config is unavailable")
            continue
        if not isinstance(stored_dataset.get("raw_config"), dict):
            rejected.append(f"{run_dir}: stored dataset raw_config is unavailable")
            continue

        candidates.append({
            "run_dir": run_dir,
            "manifest_path": manifest_path,
            "run_config_path": run_config_path,
            "method_config_name": method_config_name,
            "dataset_config_name": dataset_config_name,
            "method_config_snapshot": stored_method,
            "dataset_config_snapshot": stored_dataset,
            "config_hash": manifest.get("config_hash"),
            "base_output_dir": _infer_base_output_dir(run_dir, manifest),
            "memory_status": manifest.get("status"),
            "memory_manifest": manifest,
        })

    if not candidates:
        detail = f" ({'; '.join(rejected)})" if rejected else ""
        raise FileNotFoundError(
            f"No compatible memory run named '{memory_run}' under {output_root}{detail}"
        )
    if len(candidates) != 1:
        paths = ", ".join(str(item["run_dir"]) for item in candidates)
        raise ValueError(
            f"Memory run '{memory_run}' is ambiguous under {output_root}: {paths}. "
            "Use a narrower -o/--output-dir."
        )
    return candidates[0]


def _infer_base_output_dir(run_dir: Path, manifest: Dict[str, Any]) -> Path:
    """Find the output root for both top-level and nested query runs."""
    method_name = str(manifest.get("method_name", "")).replace("/", "-").replace("\\", "-")
    model_name = str(manifest.get("model_name", "")).replace("/", "-").replace("\\", "-")
    experiment_name = f"{method_name}_{model_name}"
    for parent in run_dir.parents:
        if parent.name == experiment_name:
            return parent.parent
    # Preserve the historical layout fallback for old artifacts.
    return run_dir.parent.parent


def append_target_already_processed(
    memory_manifest: Dict[str, Any],
    persona_id: int,
    unit_id: int,
    *,
    memory_dir: Path | None = None,
) -> bool:
    """Return whether the exact target snapshot is present and intact."""
    for record in memory_manifest.get("snapshots", []):
        if not isinstance(record, dict):
            continue
        try:
            if (
                int(record.get("context_id")) == persona_id
                and int(record.get("unit_id")) == unit_id
            ):
                snapshot_name = record.get("path")
                if not isinstance(snapshot_name, str) or Path(snapshot_name).name != snapshot_name:
                    continue
                if memory_dir is not None:
                    snapshot_path = memory_dir / snapshot_name
                    try:
                        payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
                    except (OSError, json.JSONDecodeError):
                        continue
                    if (
                        not isinstance(payload, dict)
                        or payload.get("format") != "medmemorybench.memory_snapshot"
                        or payload.get("version") != 1
                        or payload.get("context_id") != persona_id
                        or payload.get("unit_id") != unit_id
                        or payload.get("build_id") != memory_manifest.get("build_id")
                    ):
                        continue
                    expected_integrity = payload.get("integrity_hash")
                    hash_payload = dict(payload)
                    hash_payload.pop("integrity_hash", None)
                    encoded = json.dumps(
                        hash_payload,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                    if expected_integrity != hashlib.sha256(encoded).hexdigest():
                        continue
                    embedding_state = (
                        payload.get("memory_state", {})
                        .get("system_state", {})
                        .get("retriever", {})
                        .get("embeddings")
                    )
                    if embedding_state is not None:
                        embedding_name = embedding_state.get("path")
                        if (
                            embedding_state.get("storage") != "npy"
                            or not isinstance(embedding_name, str)
                            or Path(embedding_name).name != embedding_name
                        ):
                            continue
                        embedding_path = memory_dir / embedding_name
                        try:
                            digest = hashlib.sha256(embedding_path.read_bytes()).hexdigest()
                        except OSError:
                            continue
                        if digest != embedding_state.get("sha256"):
                            continue
                return True
        except (TypeError, ValueError):
            continue
    return False


def list_methods(config_loader: ConfigLoader) -> None:
    print("\nAvailable methods:")
    print("=" * 60)

    configs = config_loader.list_method_configs()
    if not configs:
        print("  (none)")
    else:
        for name in sorted(configs):
            try:
                cfg = config_loader.load_method_config(name)
                print(f"  {name}")
                print(f"    type: {cfg.method_type}, model: {cfg.model.name}")
            except Exception as e:
                print(f"  {name} (load failed: {truncate_error_message(e)})")

    print("=" * 60)


def list_datasets(config_loader: ConfigLoader) -> None:
    print("\nAvailable datasets:")
    print("=" * 60)

    configs = config_loader.list_dataset_configs()
    if not configs:
        print("  (none)")
    else:
        for name in sorted(configs):
            try:
                cfg = config_loader.load_dataset_config(name)
                print(f"  {name} ({cfg.language})")
            except Exception as e:
                print(f"  {name} (load failed: {truncate_error_message(e)})")

    print("=" * 60)


def list_agents_info() -> None:
    print("\nAvailable agent types:")
    print("=" * 60)
    for method in list_available_methods():
        print(f"  - {method}")
    print("=" * 60)


def main() -> int:
    args = parse_args()
    config_loader = ConfigLoader()
    config_inference = None
    memory_source_run_dir = None

    if args.list_methods:
        list_methods(config_loader)
        return 0

    if args.list_datasets:
        list_datasets(config_loader)
        return 0

    if args.list_agents:
        list_agents_info()
        return 0

    if args.rejudge:
        try:
            output_path, output_data = rejudge_medmemorybench(
                query_answer_path=Path(args.rejudge),
                config_loader=config_loader,
                dataset_name=args.dataset,
                verbose=not args.quiet,
                batch_api=args.batch_api,
                batch_gcs_uri=args.batch_gcs_uri,
                batch_wait=args.batch_wait,
                resume=args.resume,
            )
        except (FileNotFoundError, ValueError) as e:
            print(f"Rejudge failed: {truncate_error_message(e)}")
            return 1
        except Exception as e:
            from utils.vertex_batch import VertexBatchPending
            if isinstance(e, VertexBatchPending):
                print(f"\nBatch rejudge submitted: {e.job_name}")
                print(f"Manifest: {e.manifest_path}")
                print(
                    "Resume after completion with the same --rejudge command plus "
                    "--resume --batch-api."
                )
                return 0
            print(f"Rejudge failed: {truncate_error_message(e)}")
            print(format_limited_traceback(e), end="", file=sys.stderr)
            return 1

        summary = output_data.get("summary", {})
        print(
            f"\nRejudge results: {summary.get('correct_count', 0)}/"
            f"{summary.get('total_queries', 0)} "
            f"({summary.get('overall_accuracy', 0):.2%})"
        )
        print(f"Output: {output_path}")
        return 0

    snapshot_method_config = None
    snapshot_dataset_config = None
    if args.memory_run and (args.stage == "query" or args.append):
        try:
            inferred = infer_query_config_from_memory_run(
                Path(args.output_dir),
                args.memory_run,
                allow_incomplete=args.append,
            )
        except (FileNotFoundError, ValueError) as exc:
            print(f"Memory run inference failed: {truncate_error_message(exc)}")
            return 1
        if args.dataset and args.dataset != inferred["dataset_config_name"]:
            print(
                "Memory run inference failed: supplied dataset "
                f"'{args.dataset}' does not match stored dataset "
                f"'{inferred['dataset_config_name']}'"
            )
            return 1
        supplied_query_method = bool(args.method)
        args.method = args.method or inferred["method_config_name"]
        args.dataset = args.dataset or inferred["dataset_config_name"]
        try:
            if not supplied_query_method:
                snapshot_method_config = method_config_from_snapshot(
                    inferred["method_config_snapshot"]
                )
            snapshot_dataset_config = dataset_config_from_snapshot(
                inferred["dataset_config_snapshot"]
            )
        except (TypeError, ValueError) as exc:
            print(f"Memory run configuration failed: {truncate_error_message(exc)}")
            return 1
        args.output_dir = str(inferred["base_output_dir"])
        memory_source_run_dir = inferred["run_dir"]
        config_inference = {
            "source_run_id": args.memory_run,
            "source_run_dir": str(inferred["run_dir"]),
            "source_run_config": str(inferred["run_config_path"]),
            "source_manifest": str(inferred["manifest_path"]),
            "source_config_hash": inferred["config_hash"],
        }
        if args.append:
            config_inference.update({
                "operation": "append",
                "target_persona": args.persona,
                "target_unit": args.unit,
                "source_memory_status": inferred.get("memory_status"),
            })
        source_label = "append" if args.append else "query"
        method_label = "selected" if supplied_query_method else "inferred"
        print(
            f"Inferred {source_label} dataset from memory run {args.memory_run}; "
            f"{method_label} method={args.method}"
        )
        if args.append and append_target_already_processed(
            inferred["memory_manifest"],
            args.persona,
            args.unit,
            memory_dir=inferred["manifest_path"].parent,
        ):
            print(
                f"Append target persona={args.persona}, unit={args.unit} is already "
                f"present in memory run {args.memory_run}; nothing to do."
            )
            return 0

    if not args.method:
        print("Error: please specify method (-m/--method)")
        return 1

    if not args.dataset:
        print("Error: please specify dataset (-d/--dataset)")
        return 1

    try:
        evaluator = create_evaluator(
            method_config_name=args.method,
            dataset_name=args.dataset,
            config_loader=config_loader,
            output_dir=Path(args.output_dir),
            dry_run=args.dry_run,
            verbose=not args.quiet,
            resume=args.resume,
            force_resume=args.force,
            execution_stage=args.stage,
            memory_run=args.memory_run,
            append=args.append,
            append_persona=args.persona,
            append_unit=args.unit,
            method_config=snapshot_method_config,
            dataset_config=snapshot_dataset_config,
            config_inference=config_inference,
            memory_source_run_dir=memory_source_run_dir,
            batch_api=args.batch_api,
            batch_gcs_uri=args.batch_gcs_uri,
            batch_wait=args.batch_wait,
            workers=args.workers,
        )
    except FileNotFoundError as e:
        print(f"Error: {truncate_error_message(e)}")
        return 1
    except Exception as e:
        print(f"Init failed: {truncate_error_message(e)}")
        return 1

    print(f"\nMethod: {args.method}, Dataset: {args.dataset}, Dry Run: {args.dry_run}")
    print(f"Run directory: {evaluator.output_dir}\n")

    try:
        report = evaluator.run()
        summary = report.summary
        print(f"\nResults: {summary.get('correct', 0)}/{summary.get('total', 0)} "
              f"({summary.get('overall_accuracy', 0):.2%})")
        print(f"Output: {evaluator.output_dir}")
        return 0
    except KeyboardInterrupt:
        print("\nInterrupted")
        return 130
    except Exception as e:
        from utils.vertex_batch import VertexBatchPending
        if isinstance(e, VertexBatchPending):
            print(f"\nBatch job submitted: {e.job_name}")
            print(f"Manifest: {e.manifest_path}")
            print("Resume after completion with the same command plus --resume --batch-api.")
            return 0
        print(f"\nFailed: {truncate_error_message(e)}")
        print(format_limited_traceback(e), end="", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
