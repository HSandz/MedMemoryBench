# A-MEM Build Metrics

This document describes the report-ready telemetry produced by A-MEM memory builds. It applies to `amem`, `amem_fix`, and `amem_test` runs on MedMemoryBench.

## What Is Recorded

Each new build records three top-level objects:

- `feature_configuration` identifies the exact build feature combination.
- `build_metrics` contains whole-run, per-feature, per-operation, and per-unit measurements. The memory-build report also retains per-session metrics under each unit's `session_builds` list.
- `memory_size` is the general final memory-footprint summary, also available as `build_metrics.memory_size`.

The combination identity comes from the effective build flags, not the YAML filename. This matters because files with similar names may intentionally enable different feature combinations.

## Output Locations

| Artifact | Metrics location | Purpose |
|---|---|---|
| `*_memory_build.json` | Top-level `feature_configuration`, `build_metrics`, `memory_size`, and `llm_usage`; unit/session details under `units` | Construction metrics recorded by that invocation, including memory-only runs |
| `*_result.json` | Top-level `feature_configuration`, `build_metrics`, and `memory_size` | Accuracy and available construction metrics in one file |
| `memory/manifest.json` | Top-level `feature_configuration`, `build_metrics`, and `memory_size` | Canonical completed snapshot-run summary, including copied or restored units |
| `memory/persona_*_unit_*.json` | `feature_configuration`, `memory_build_metrics`, and `memory_size` | Authoritative metrics and serialized footprint for one evaluation unit |
| `memory_source.json` | `feature_configuration`, `build_metrics`, and `memory_size` copied from the selected manifest | Source-build report for query-only runs |

`build_metrics` is the build-only view. For ordinary fresh runs, the memory-build report and completed manifest describe the same construction. For resumed or append workflows, use the completed manifest when reporting the full snapshot lifecycle because invocation reports intentionally do not recount restored work. The `llm_usage` field reports `memorize_phase`, `query_phase`, and `judge_phase` separately; `total` is their sum.

## Feature Combination

Example:

```json
{
  "schema_version": 1,
  "method_name": "amem_test",
  "combination_id": "base_memory+original_evolution+typed_relations+temporal_state+provenance",
  "enabled_features": [
    "base_memory",
    "original_evolution",
    "typed_relations",
    "temporal_state",
    "provenance"
  ],
  "features": {
    "base_memory": true,
    "original_evolution": true,
    "typed_relations": true,
    "temporal_state": true,
    "provenance": true
  },
  "dependencies": [],
  "build_config_hash": "...",
  "build_config": {}
}
```

Always group experiments by `combination_id` or the explicit `features` map. When comparing note granularity, also group by `build_config.amem_note_level`; an omitted value means `turn`. Do not infer behavior from names such as `amem_test2`.

## Metrics Schema

`build_metrics` has these report levels:

The current `build_metrics.schema_version` is `2`; schema version `1` artifacts remain readable but do not contain the dedicated memory-size summary.

| Field | Meaning |
|---|---|
| `totals` | End-to-end construction metrics for the complete run |
| `by_feature` | Direct metrics grouped by feature family |
| `by_operation` | Fine-grained instrumented operations |
| `units` | Per-evaluation-unit metrics and source/restoration status |
| `usage` | Raw merged tracker structure for programmatic analysis |
| `complete_telemetry` | `false` when a legacy snapshot has build time but no historical token/call details |

## Memory Size Schema

Memory size measures the serialized retrieval state, excluding snapshot metadata such as timestamps, build metrics, and integrity fields. It includes the compact JSON memory state plus the exact on-disk NumPy embedding sidecar.

The top-level `memory_size` object contains:

| Field | Meaning |
|---|---|
| `overall_bytes` / `overall_mib` | Final benchmark memory footprint: the latest unit snapshot for each persona/context, summed once |
| `unit_snapshot_bytes` / `unit_snapshot_mib` | Sum of every cumulative unit snapshot; useful for artifact-series accounting, not final footprint |
| `by_context` | Latest unit ID and final footprint for each persona/context |
| `available` | At least one unit has a measured serialized state |
| `complete` | Every reported evaluation unit has a measured serialized state |
| `measured_unit_count` | Number of units with a measurement |

Each `build_metrics.units[]` entry has a `memory_size` object with `bytes`, `mib`, `json_bytes`, `embedding_bytes`, `memory_entry_count`, `memory_chunk_count`, and signed `delta_from_previous_unit_bytes`/`delta_from_previous_unit_mib` fields. The first unit for a context uses zero as its baseline. A negative delta is valid when evolution, consolidation, or replacement reduces the serialized state.

Evaluation-unit snapshots are cumulative within a persona. Therefore, do not sum unit `bytes` to report final memory use. Use `memory_size.overall_bytes`; use `unit_snapshot_bytes` only when measuring the complete stored snapshot series.

Common metric fields:

| Metric | Definition |
|---|---|
| `input_tokens` | Provider-reported LLM prompt tokens |
| `output_tokens` | Provider-reported total generated tokens, including thinking/reasoning tokens when the provider includes them |
| `visible_output_tokens` | Provider-reported generated tokens excluding separately reported thinking/reasoning tokens |
| `thinking_tokens` | Provider-reported hidden thinking/reasoning tokens; `0` when the provider does not expose a separate count |
| `total_tokens` | Input plus output tokens |
| `successful_calls` / `call_count` | Successful LLM calls |
| `memorize_phase` | LLM calls used to construct memory |
| `query_phase` | LLM calls used for retrieval preparation and final answers |
| `judge_phase` | LLM calls used to score answers with an LLM judge |
| `attempted_calls` | All LLM attempts, including failed attempts before retries |
| `failed_attempts` | Attempts that raised an error or returned an unusable response |
| `retry_count` | Failures followed by another attempt |
| `operation_count` | Number of times an instrumented logical operation ran |
| `total_latency` | Sum of successful provider-call latency in seconds |
| `avg_latency` | Mean successful provider-call latency |
| `wall_time` | Direct wall time for one feature or operation scope |
| `wall_time_seconds` | End-to-end session/unit/run construction wall time |
| `scoped_operation_wall_time_seconds` | Sum of directly instrumented operation wall time |
| `unattributed_wall_time_seconds` | Build wall time outside the named operation scopes, including orchestration overhead |

Token counts depend on provider metadata. Backends that do not expose token usage still report successful calls, attempts, retries, operation counts, and wall time, but their token fields can be zero.

## Feature and Operation Mapping

| Feature group | Operations |
|---|---|
| `base` | `amem.base.chunking`, `amem.base.note_analysis` |
| `original_evolution` | `amem.original_evolution` |
| `embedding` | `amem.embedding.index`, `amem.embedding.consolidation` |
| `typed_relations` | `amem.typed_relations.candidate_search`, `amem.typed_relations.inference`, `amem.typed_relations.store` |
| `temporal_state` | `amem.temporal_state.initialize`, `amem.temporal_state.transition` |
| `provenance` | `amem.provenance.preprocessing`, `amem.provenance.attach` |

`by_feature` measures direct scoped work. It is not a causal marginal-cost estimate. Enabling provenance can change the number of notes and therefore increase base analysis and embedding work; typed relations can drive temporal transitions. Use `totals` to compare complete combinations and use `by_feature` to explain where directly instrumented work occurred.

## Running Feature Combinations

Declare every build feature explicitly:

```yaml
build_config:
  amem_note_level: turn
  amem_original_evolution: false
  amem_typed_relations: true
  amem_temporal_state: false
  amem_provenance: false
```

Useful controlled combinations include:

| Experiment | Evolution | Typed | Temporal | Provenance |
|---|---:|---:|---:|---:|
| Base atomic-note build | Off | Off | Off | Off |
| Original evolution | On | Off | Off | Off |
| Typed relations | Off | On | Off | Off |
| Typed plus temporal | Off | On | On | Off |
| Provenance | Off | Off | Off | On |
| Typed plus temporal plus provenance | Off | On | On | On |
| All features | On | On | On | On |

Temporal timestamp state can exist without typed relations, but inferred temporal transitions require typed relations. The generated `dependencies` field records this partial combination.

Run construction separately when you need an isolated build artifact:

```bash
python main.py -m METHOD_CONFIG -d medmemorybench --stage memory
```

Run construction and evaluation together when you want cost and accuracy in the same result:

```bash
python main.py -m METHOD_CONFIG -d medmemorybench
```

For causal comparisons, keep the dataset selection, build model, embedding model, temperatures, chunking, prompt budget, and temporal threshold fixed. Change only the intended feature flags.

## Creating a Cross-Run CSV

The following script scans completed memory manifests and writes one row per snapshot run. Using manifests keeps resumed and append runs complete.

```python
import csv
import json
from pathlib import Path

rows = []
for path in Path("outputs").rglob("manifest.json"):
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format") != "medmemorybench.memory_manifest":
        continue
    if payload.get("status") != "complete":
        continue
    metrics = payload.get("build_metrics", {})
    totals = metrics.get("totals", {})
    feature_config = payload.get("feature_configuration", {})
    rows.append({
        "path": str(path),
        "method": payload.get("method_name", ""),
        "model": payload.get("model_name", ""),
        "combination": feature_config.get("combination_id", "unknown"),
        "complete_telemetry": metrics.get("complete_telemetry", False),
        "build_wall_time_seconds": totals.get("wall_time_seconds", 0),
        "input_tokens": totals.get("input_tokens", 0),
        "output_tokens": totals.get("output_tokens", 0),
        "successful_calls": totals.get("successful_calls", 0),
        "attempted_calls": totals.get("attempted_calls", 0),
        "failed_attempts": totals.get("failed_attempts", 0),
        "retries": totals.get("retry_count", 0),
        "llm_latency_seconds": totals.get("total_latency", 0),
        "memory_size_bytes": metrics.get("memory_size", {}).get("overall_bytes", 0),
        "memory_size_mib": metrics.get("memory_size", {}).get("overall_mib", 0),
    })

with Path("amem_build_report.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
    if rows:
        writer.writeheader()
        writer.writerows(rows)
```

For feature-level columns, read `build_metrics.by_feature.<feature>`. For operation-level analysis, read `build_metrics.by_operation.<operation>`.

## Resume, Append, and Legacy Behavior

- Resumed builds reuse metrics embedded in completed unit snapshots rather than counting restored work as new construction. Use the completed manifest for the full resumed-run total.
- Append manifests preserve copied source-unit metrics and add metrics for newly built units. The append invocation report can contain only the work performed during that invocation.
- Query-only runs expose source construction metrics through their result and `memory_source.json`.
- Legacy snapshots remain compatible. When detailed historical metrics do not exist, `complete_telemetry` is `false`; available build wall time is retained.

## Verification

```bash
python -m pytest tests/test_build_feature_metrics.py
python -m pytest tests/test_amem_staged_memory.py tests/test_gemini_retry.py
```
