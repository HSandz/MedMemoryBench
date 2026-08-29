"""Re-run MedMemoryBench metrics from previously saved method answers."""

import copy
import hashlib
import json
import os
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from benchmarks.medmemorybench.dataset import MedMemoryBenchDataset, MedQuery
from metrics import MetricResult, MetricsAggregator, MetricsCalculator
from src.config import ConfigLoader, DatasetConfig, PROJECT_ROOT, get_api_config
from utils.batch_client import create_batch_client
from utils.llm_client import (
    get_usage_tracker,
    is_batch_provider,
    is_google_ai_studio_provider,
)
from utils.vertex_batch import (
    BatchChatRequest,
    VertexBatchClient,
    VertexBatchError,
    make_request_id,
    scoped_manifest_path,
)


LLM_JUDGE_METRICS = {"llm_judge", "eem_judge", "llm_judge_mcd"}
BATCH_STATE_VERSION = 1


def _load_dataset(dataset_config: DatasetConfig) -> MedMemoryBenchDataset:
    """Load all personas so older subset runs remain re-judgeable."""
    dataset = MedMemoryBenchDataset(
        data_dir=PROJECT_ROOT / dataset_config.data_root_dir,
        config={
            "evaluation_mode": dataset_config.evaluation_mode,
            "persona_ids": None,
            "max_personas": None,
            "max_sessions_per_persona": None,
            "evaluation_interval": dataset_config.evaluation_interval,
            "inject_noise": dataset_config.inject_noise,
            "query_types": None,
        },
    )
    dataset.load()
    return dataset


def _query_signature(query: Any) -> Tuple[str, str, str]:
    return (
        str(query.get("query_id", "") if isinstance(query, dict) else query.query_id),
        str(query.get("query_type", "") if isinstance(query, dict) else query.query_type),
        str(query.get("question", "") if isinstance(query, dict) else query.question),
    )


def _index_dataset_queries(
    dataset: MedMemoryBenchDataset,
) -> Tuple[
    Dict[Tuple[int, str, str, str], MedQuery],
    Dict[Tuple[str, str, str], List[Tuple[int, MedQuery]]],
]:
    by_context: Dict[Tuple[int, str, str, str], MedQuery] = {}
    by_signature: Dict[Tuple[str, str, str], List[Tuple[int, MedQuery]]] = {}

    for persona_id, persona_data in dataset._personas.items():
        for query in persona_data["queries"]:
            signature = _query_signature(query)
            by_context[(persona_id, *signature)] = query
            by_signature.setdefault(signature, []).append((persona_id, query))

    return by_context, by_signature


def _find_dataset_query(
    saved_query: Dict[str, Any],
    by_context: Dict[Tuple[int, str, str, str], MedQuery],
    by_signature: Dict[Tuple[str, str, str], List[Tuple[int, MedQuery]]],
) -> Tuple[int, MedQuery]:
    signature = _query_signature(saved_query)
    context_id = saved_query.get("context_id")
    if context_id is not None:
        try:
            normalized_context_id = int(context_id)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid context_id for saved query {signature[0]}: {context_id!r}"
            ) from exc
        query = by_context.get((normalized_context_id, *signature))
        if query is None:
            raise ValueError(
                f"Saved query {signature[0]} does not match persona {normalized_context_id} "
                "in the current dataset."
            )
        return normalized_context_id, query

    matches = by_signature.get(signature, [])
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise ValueError(
            f"Saved query {signature[0]} was not found in the current dataset."
        )
    raise ValueError(
        f"Saved query {signature[0]} is ambiguous across personas. Re-run the benchmark "
        "with the current code so query_answer.json includes context_id."
    )


def _metric_result_from_saved(saved_query: Dict[str, Any]) -> MetricResult:
    return MetricResult(
        query_id=str(saved_query.get("query_id", "")),
        query_type=str(saved_query.get("query_type", "")),
        score=float(saved_query.get("score", 0.0)),
        is_correct=bool(saved_query.get("is_correct", False)),
        model_output=str(saved_query.get("model_output", "")),
        expected_answer=str(saved_query.get("expected_answer", "")),
        question=str(saved_query.get("question", "")),
        details=copy.deepcopy(saved_query.get("evaluation_details", {})),
        query_time=float(saved_query.get("query_time", 0.0)),
        retrieved_memories=copy.deepcopy(saved_query.get("retrieved_memories", [])),
        retrieved_count=int(saved_query.get("retrieved_count", 0)),
    )


def _artifact_path(source_path: Path, value: str) -> Path:
    """Resolve an artifact path stored relative to a query report."""
    path = Path(value)
    return path if path.is_absolute() else source_path.parent / path


def _hydrate_retrieved_memories(
    source_path: Path,
    source_data: Dict[str, Any],
    saved_queries: List[Dict[str, Any]],
) -> None:
    """Restore compact v2 retrieval references for rejudge output fidelity."""
    records_by_id: Dict[str, List[Dict[str, Any]]] = {}
    records_path = source_data.get("retrieval_records_path")
    if isinstance(records_path, str) and records_path:
        path = _artifact_path(source_path, records_path)
        try:
            records_payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Cannot read retrieval records {path}: {exc}") from exc
        for record in records_payload.get("records", []):
            if isinstance(record, dict) and isinstance(record.get("retrieved_memories"), list):
                records_by_id[str(record.get("query_id", ""))] = record["retrieved_memories"]

    batch_requests: Dict[Path, Dict[str, Dict[str, Any]]] = {}
    for saved_query in saved_queries:
        if isinstance(saved_query.get("retrieved_memories"), list):
            continue
        reference = saved_query.get("retrieval_reference", {})
        if not isinstance(reference, dict):
            continue
        source = reference.get("source")
        if source == "none":
            saved_query["retrieved_memories"] = []
        elif source == "retrieval_records":
            record_id = str(reference.get("record_id", ""))
            if record_id not in records_by_id:
                raise ValueError(f"Retrieval record is missing for query {record_id!r}.")
            saved_query["retrieved_memories"] = copy.deepcopy(records_by_id[record_id])
        elif source == "answer_batch_manifest":
            manifest_value = reference.get("manifest_path")
            request_id = reference.get("request_id")
            if not isinstance(manifest_value, str) or not isinstance(request_id, str):
                raise ValueError("Batch retrieval reference is missing its manifest path or request ID.")
            manifest_path = _artifact_path(source_path, manifest_value)
            if manifest_path not in batch_requests:
                try:
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as exc:
                    raise ValueError(f"Cannot read answer batch manifest {manifest_path}: {exc}") from exc
                batch_requests[manifest_path] = {
                    str(request.get("request_id")): request
                    for job in manifest.get("jobs", {}).values()
                    if isinstance(job, dict)
                    for request in job.get("requests", [])
                    if isinstance(request, dict) and request.get("request_id")
                }
            request = batch_requests[manifest_path].get(request_id)
            prepared = (
                request.get("metadata", {}).get("prepared_query")
                if isinstance(request, dict) else None
            )
            memories = prepared.get("retrieved_memories") if isinstance(prepared, dict) else None
            if not isinstance(memories, list):
                raise ValueError(
                    f"Batch retrieval payload is missing for request {request_id!r}."
                )
            saved_query["retrieved_memories"] = copy.deepcopy(memories)


def _next_output_path(source_path: Path) -> Tuple[Path, int]:
    base_stem = re.sub(r"_rejudge_\d+$", "", source_path.stem)
    pattern = re.compile(rf"^{re.escape(base_stem)}_rejudge_(\d+)\.json$")
    existing_runs = []
    for candidate in source_path.parent.glob(f"{base_stem}_rejudge_*.json"):
        match = pattern.match(candidate.name)
        if match:
            existing_runs.append(int(match.group(1)))
    state_path = source_path.with_name(f"{base_stem}_rejudge_batch_state.json")
    if state_path.is_file():
        try:
            reserved_run = int(json.loads(state_path.read_text(encoding="utf-8"))["run_number"])
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            pass
        else:
            existing_runs.append(reserved_run)
    run_number = max(existing_runs, default=0) + 1
    return source_path.with_name(
        f"{base_stem}_rejudge_{run_number}{source_path.suffix}"
    ), run_number


def _batch_state_path(source_path: Path) -> Path:
    """Keep one active batch-rejudge state beside the source answers."""
    base_stem = re.sub(r"_rejudge_\d+$", "", source_path.stem)
    return source_path.with_name(f"{base_stem}_rejudge_batch_state.json")


def _source_hash(source_path: Path) -> str:
    digest = hashlib.sha256()
    with source_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _batch_config_hash(
    source_path: Path,
    *,
    source_hash: str,
    judge_provider: str,
    judge_model: str,
    judge_temperature: float,
    judge_max_tokens: int,
    judge_mcd_max_tokens: int,
    run_scope: str,
) -> str:
    payload = json.dumps({
        "source_path": str(source_path),
        "source_hash": source_hash,
        "judge_provider": judge_provider,
        "judge_model": judge_model,
        "judge_temperature": judge_temperature,
        "judge_max_tokens": judge_max_tokens,
        "judge_mcd_max_tokens": judge_mcd_max_tokens,
        "run_scope": run_scope,
    }, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:48]


def _write_batch_state(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary_path, path)


def _load_batch_state(
    state_path: Path,
    *,
    source_path: Path,
    source_hash: str,
    judge_provider: str,
    judge_model: str,
    judge_temperature: float,
    judge_max_tokens: int,
    judge_mcd_max_tokens: int,
) -> Dict[str, Any]:
    if not state_path.is_file():
        raise FileNotFoundError(
            f"No pending batch rejudge state found: {state_path}. "
            "Start without --resume first."
        )
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid batch rejudge state: {state_path}") from exc
    if (
        state.get("version") != BATCH_STATE_VERSION
        or state.get("source_path") != str(source_path)
        or state.get("source_hash") != source_hash
        or state.get("judge_provider") != judge_provider
        or state.get("judge_model") != judge_model
        or state.get("judge_temperature", 1.0) != judge_temperature
        or state.get("judge_max_tokens", 500) != judge_max_tokens
        or state.get("judge_mcd_max_tokens", 2000) != judge_mcd_max_tokens
    ):
        raise ValueError(
            "Batch rejudge state does not match the source file or current judge "
            "configuration. Resume with the original settings or start a fresh run."
        )
    return state


def _query_output(
    saved_query: Dict[str, Any],
    result: MetricResult,
    context_id: int,
    *,
    compact_retrieval: bool = False,
) -> Dict[str, Any]:
    output = copy.deepcopy(saved_query)
    output.update({
        "context_id": context_id,
        "question": result.question,
        "expected_answer": result.expected_answer,
        "model_output": result.model_output,
        "score": result.score,
        "is_correct": result.is_correct,
        "retrieved_count": result.retrieved_count,
        "query_time": result.query_time,
        "evaluation_details": result.details,
    })
    if compact_retrieval:
        output.pop("retrieved_memories", None)
    else:
        output["retrieved_memories"] = result.retrieved_memories
    return output


def rejudge_medmemorybench(
    query_answer_path: Path,
    config_loader: Optional[ConfigLoader] = None,
    dataset_name: Optional[str] = None,
    verbose: bool = True,
    batch_api: bool = False,
    batch_gcs_uri: Optional[str] = None,
    batch_wait: bool = False,
    resume: bool = False,
) -> Tuple[Path, Dict[str, Any]]:
    """Re-run only LLM-judge metrics while preserving saved method answers."""
    source_path = Path(query_answer_path).resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"Query answer file not found: {source_path}")

    try:
        source_data = json.loads(source_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid query answer JSON: {source_path}") from exc

    saved_queries = source_data.get("queries")
    if not isinstance(saved_queries, list):
        raise ValueError(
            "Rejudge input must be a *_query_answer.json file containing a queries list."
        )
    _hydrate_retrieved_memories(source_path, source_data, saved_queries)
    compact_retrieval = source_data.get("format") == "medmemorybench.query_answers" and (
        source_data.get("version") == 2
    )

    resolved_dataset_name = dataset_name or source_data.get("dataset_name")
    if resolved_dataset_name != "medmemorybench":
        raise ValueError(
            "Rejudge currently supports only the medmemorybench dataset."
        )

    config_loader = config_loader or ConfigLoader()
    dataset_config = config_loader.load_dataset_config(resolved_dataset_name)
    dataset = _load_dataset(dataset_config)
    by_context, by_signature = _index_dataset_queries(dataset)

    api_config = get_api_config()
    judge_provider = api_config.get_judge_provider().lower()
    judge_model = api_config.get_judge_model()
    judge_temperature = getattr(api_config, "judge_temperature", 1.0)
    judge_reasoning_effort = getattr(api_config, "judge_reasoning_effort", None)
    judge_client_max_tokens = getattr(api_config, "judge_client_max_tokens", 10000)
    judge_max_tokens = getattr(api_config, "judge_max_tokens", 500)
    judge_mcd_max_tokens = getattr(api_config, "judge_mcd_max_tokens", 2000)
    if batch_api and is_google_ai_studio_provider(judge_provider):
        if verbose:
            print(
                "Google AI Studio does not support the Vertex batch path; "
                "ignoring --batch-api, --batch-gcs-uri, --batch-wait, and --resume."
            )
        batch_api = False
        resume = False
    elif batch_api and not is_batch_provider(judge_provider):
        if verbose:
            print(
                f"Judge provider '{judge_provider}' cannot use a batch API; "
                "using the existing real-time judge client."
            )
        batch_api = False
        resume = False
    get_usage_tracker().reset()
    metric_mapping = {
        query_type.name: query_type.metric
        for query_type in dataset_config.query_types
        if query_type.name and query_type.metric
    }
    calculator = MetricsCalculator(
        custom_mapping=metric_mapping,
        judge_model=api_config.judge_model or None,
        judge_api_key=api_config.judge_api_key or None,
        judge_base_url=api_config.judge_base_url or None,
        judge_temperature=judge_temperature,
        judge_reasoning_effort=judge_reasoning_effort,
        judge_client_max_tokens=judge_client_max_tokens,
        judge_max_tokens=judge_max_tokens,
        judge_mcd_max_tokens=judge_mcd_max_tokens,
        language=dataset_config.language,
    )
    aggregator = MetricsAggregator()
    output_queries: List[Dict[str, Any]] = []
    results_by_context: Dict[int, List[MetricResult]] = {}
    rejudged_count = 0
    prepared_batch_items: List[Dict[str, Any]] = []

    for index, saved_query in enumerate(saved_queries, start=1):
        if not isinstance(saved_query, dict):
            raise ValueError(f"Invalid query entry at position {index}.")
        context_id, dataset_query = _find_dataset_query(
            saved_query, by_context, by_signature
        )
        metric_name = calculator.get_metric_name(dataset_query.query_type)

        if metric_name in LLM_JUDGE_METRICS:
            if verbose and not batch_api:
                print(
                    f"[Rejudge {index}/{len(saved_queries)}] "
                    f"persona={context_id} query={dataset_query.query_id} "
                    f"type={dataset_query.query_type}"
                )
            if batch_api:
                prepared_metric = calculator.prepare_batch(
                    query_id=dataset_query.query_id,
                    query_type=dataset_query.query_type,
                    model_output=str(saved_query.get("model_output", "")),
                    expected_answers=dataset_query.get_correct_answers(),
                    question=dataset_query.question,
                    answers_data=dataset_query.answers_data,
                    metadata=dataset_query.metadata,
                )
                judge_payload = prepared_metric["prepared"]["judge_payload"]
                if "immediate" in judge_payload:
                    result = calculator.finalize_batch(prepared_metric, "")
                else:
                    prepared_batch_items.append({
                        "index": index - 1,
                        "context_id": context_id,
                        "saved_query": saved_query,
                        "prepared_metric": prepared_metric,
                    })
                    output_queries.append({})
                    rejudged_count += 1
                    continue
            else:
                result = calculator.compute(
                    query_id=dataset_query.query_id,
                    query_type=dataset_query.query_type,
                    model_output=str(saved_query.get("model_output", "")),
                    expected_answers=dataset_query.get_correct_answers(),
                    question=dataset_query.question,
                    answers_data=dataset_query.answers_data,
                    metadata=dataset_query.metadata,
                )
            result.query_time = float(saved_query.get("query_time", 0.0))
            result.retrieved_memories = copy.deepcopy(
                saved_query.get("retrieved_memories", [])
            )
            result.retrieved_count = int(saved_query.get("retrieved_count", 0))
            rejudged_count += 1
        else:
            result = _metric_result_from_saved(saved_query)

        aggregator.add_result(result)
        results_by_context.setdefault(context_id, []).append(result)
        output_queries.append(
            _query_output(
                saved_query,
                result,
                context_id,
                compact_retrieval=compact_retrieval,
            )
        )

    batch_manifest_path: Optional[Path] = None
    active_state_path: Optional[Path] = None
    if prepared_batch_items:
        first_prepared = prepared_batch_items[0]["prepared_metric"]["prepared"]
        judge_client = calculator.get_batch_judge_client(first_prepared["query_type"])
        if judge_client is None:
            raise VertexBatchError(
                "Batch rejudge requires a Vertex Gemini or OpenRouter judge."
            )

        state_path = _batch_state_path(source_path)
        active_state_path = state_path
        source_hash = _source_hash(source_path)
        if resume:
            state = _load_batch_state(
                state_path,
                source_path=source_path,
                source_hash=source_hash,
                judge_provider=judge_provider,
                judge_model=judge_model,
                judge_temperature=judge_temperature,
                judge_max_tokens=judge_max_tokens,
                judge_mcd_max_tokens=judge_mcd_max_tokens,
            )
        else:
            output_path, run_number = _next_output_path(source_path)
            run_scope = uuid.uuid4().hex
            state = {
                "version": BATCH_STATE_VERSION,
                "source_path": str(source_path),
                "source_hash": source_hash,
                "judge_provider": judge_provider,
                "judge_model": judge_model,
                "judge_temperature": judge_temperature,
                "judge_max_tokens": judge_max_tokens,
                "judge_mcd_max_tokens": judge_mcd_max_tokens,
                "run_scope": run_scope,
                "run_number": run_number,
                "output_path": str(output_path),
                "created_at": datetime.now().isoformat(),
            }
            state["config_hash"] = _batch_config_hash(
                source_path,
                source_hash=source_hash,
                judge_provider=judge_provider,
                judge_model=judge_model,
                judge_temperature=judge_temperature,
                judge_max_tokens=judge_max_tokens,
                judge_mcd_max_tokens=judge_mcd_max_tokens,
                run_scope=run_scope,
            )
            _write_batch_state(state_path, state)

        output_path = Path(state["output_path"])
        run_number = int(state["run_number"])
        batch_manifest_path = scoped_manifest_path(
            source_path.parent / "batch",
            "medmemorybench_rejudge_batch_manifest",
            model=judge_client.model,
            config_hash=state["config_hash"],
        )
        batch_client = create_batch_client(
            judge_client,
            gcs_uri=batch_gcs_uri,
            manifest_path=batch_manifest_path,
            wait=batch_wait,
            config_hash=state["config_hash"],
            progress_callback=print if verbose else None,
            vertex_batch_class=VertexBatchClient,
        )
        requests: List[BatchChatRequest] = []
        by_request_id: Dict[str, Dict[str, Any]] = {}
        for item in prepared_batch_items:
            prepared = item["prepared_metric"]["prepared"]
            payload = prepared["judge_payload"]
            request_id = make_request_id(
                "rejudge",
                f"{item['context_id']}:{prepared['query_id']}:{prepared['query_type']}",
            )
            requests.append(BatchChatRequest(
                request_id=request_id,
                messages=[{"role": "user", "content": payload["prompt"]}],
                temperature=payload.get("temperature", judge_temperature),
                max_tokens=payload["max_tokens"],
                reasoning_effort=payload.get("reasoning_effort", judge_reasoning_effort),
                response_format={"type": "json_object"},
                phase="query",
                metadata={
                    "query_id": prepared["query_id"],
                    "context_id": item["context_id"],
                    "phase": "rejudge",
                },
            ))
            by_request_id[request_id] = item

        responses = batch_client.run_stage("rejudge-final", requests)
        for request_id, item in by_request_id.items():
            response = responses.get(request_id)
            if response is None or response.status or not response.content:
                error = response.status if response is not None else "No output row returned"
                raise VertexBatchError(
                    f"Batch rejudge failed for {item['prepared_metric']['prepared']['query_id']}: {error}"
                )
            result = calculator.finalize_batch(
                item["prepared_metric"], response.content
            )
            saved_query = item["saved_query"]
            result.query_time = float(saved_query.get("query_time", 0.0))
            result.retrieved_memories = copy.deepcopy(
                saved_query.get("retrieved_memories", [])
            )
            result.retrieved_count = int(saved_query.get("retrieved_count", 0))
            aggregator.add_result(result)
            results_by_context.setdefault(item["context_id"], []).append(result)
            output_queries[item["index"]] = _query_output(
                saved_query,
                result,
                item["context_id"],
                compact_retrieval=compact_retrieval,
            )

    summary = aggregator.get_summary()
    total_query_time = sum(result.query_time for result in aggregator.results)
    total_retrieved = sum(result.retrieved_count for result in aggregator.results)
    total = len(aggregator.results)
    if not prepared_batch_items:
        output_path, run_number = _next_output_path(source_path)

    output_data = copy.deepcopy(source_data)
    output_data["summary"] = {
        "total_queries": total,
        "correct_count": summary.get("correct", 0),
        "overall_accuracy": summary.get("overall_accuracy", 0.0),
        "overall_avg_score": summary.get("overall_avg_score", 0.0),
        "by_type": summary.get("by_type", {}),
        "total_query_time": total_query_time,
        "avg_query_time": total_query_time / total if total else 0.0,
        "avg_retrieved_count": total_retrieved / total if total else 0.0,
    }
    output_data["queries"] = output_queries
    output_data["by_context"] = {
        str(context_id): {
            "total": len(results),
            "correct": sum(1 for result in results if result.is_correct),
            "query_ids": [result.query_id for result in results],
        }
        for context_id, results in results_by_context.items()
    }
    output_data["rejudge"] = {
        "run_number": run_number,
        "created_at": datetime.now().isoformat(),
        "source_file": source_path.name,
        "judge_provider": api_config.get_judge_provider(),
        "judge_model": api_config.get_judge_model(),
        "judge_temperature": judge_temperature,
        "judge_reasoning_effort": judge_reasoning_effort,
        "judge_client_max_tokens": judge_client_max_tokens,
        "judge_max_tokens": judge_max_tokens,
        "judge_mcd_max_tokens": judge_mcd_max_tokens,
        "batch_api": batch_api,
        "batch_requests": len(prepared_batch_items),
        "batch_manifest": str(batch_manifest_path) if batch_manifest_path else None,
        "rejudged_queries": rejudged_count,
        "preserved_local_metric_queries": total - rejudged_count,
        "source_summary": source_data.get("summary", {}),
        "llm_usage": get_usage_tracker().get_stats(),
    }

    temporary_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
    temporary_path.write_text(
        json.dumps(output_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary_path.replace(output_path)
    if active_state_path is not None:
        active_state_path.unlink(missing_ok=True)
    return output_path, output_data
