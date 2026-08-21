# A-MEM Method Guide

This guide covers the A-MEM integration in MedMemoryBench. Repository-wide Python, testing, and secret-handling rules still apply.

## Integration Structure

- `methods/amem_agent.py` provides the shared `AMemAgent` adapter and snapshot support.
- `methods/amem_fix_agent.py` implements the paper-aligned `AMemFixAgent` flow.
- `methods/amem_test_agent.py` extends it with experimental typed relations, temporal state, and provenance.
- `methods/amem/A-mem/memory_layer_robust.py` is the active plain-text LLM memory layer; `memory_layer_typed.py` is used only by `amem_test`.
- `methods/amem/A-mem/memory_layer.py` and the scripts in that directory are the upstream/standalone implementation and LoCoMo experiments. They are not the normal MedMemoryBench entry point.
- Configs are in `configs/method_config/amem*.yaml` and `configs/method_config/persona_1/`. Regression coverage is in `tests/test_amem_fix_agent.py`, `tests/test_amem_test_agent.py`, and `tests/test_amem_staged_memory.py`.

## Runtime Pipeline

1. `AgentManager` selects `amem`, `amem_fix`, or `amem_test` from the YAML `method_name` and forwards `agent_params`.
2. Each persona/context receives its own memory system. Notes use a SentenceTransformer embedding retriever plus an internal metadata/evolution LLM.
3. `amem` splits oversized text into token-bounded chunks and passes each chunk directly to `add_note()`.
4. `amem_fix` converts structured dialogue turns into timestamped atomic notes, optionally generates keyword queries, retrieves semantic seeds, and expands linked notes.
5. `amem_test` keeps the atomic-note flow and optionally infers typed edges (`SUPPORT`, `REFINE`, `SUPERSEDE`, `CONFLICT`, `RELATED`), temporal states, and evidence records.
6. Retrieval is local and read-only during query preparation, apart from the optional LLM call that turns a question into keywords. Retrieved notes are placed in the final prompt; Vertex batch transport applies to final answer generation. `--stage memory` and `--stage query` can build and reuse snapshots.

The robust layer may make up to three conditional evolution calls when adding a note: decide whether to evolve, strengthen details, and update neighbors. `amem_evo_threshold` controls periodic retriever consolidation.

## Configuration

AMEM method files separate `build_config` from `retrieval_config`. Build settings define the stored snapshot: the internal `amem_backend`/`amem_model`, embedding, chunking, metadata/evolution settings, and `amem_build_max_context_tokens`. Retrieval settings can change on a frozen snapshot, including `retrieve_num`, keyword use, graph budgets, `amem_relation_min_confidence`, temporal ordering, provenance injection, and `amem_max_context_tokens`. The top-level `model` controls final-answer generation and does not replace the internal build model.

For `amem_test`, the important build switches are `amem_original_evolution`, `amem_typed_relations`, `amem_temporal_state`, and `amem_provenance`. `amem_temporal_transition_min_confidence` controls which inferred relations update stored temporal state. Query-time `amem_relation_min_confidence` only filters retrieved relations. Temporal state requires typed relations. Snapshot selection validates build semantics while allowing retrieval-only ablations; legacy manifests derive a build hash from their stored `run_config.json` when possible.

## Build Metrics and Feature Reports

Every new A-MEM build records report-ready `feature_configuration`, `build_metrics`, and `memory_size` objects. The combination ID comes from the actual build flags rather than the configuration filename, so independently run ablations and custom combinations remain comparable. Metrics include build wall time, provider-reported input/output tokens, successful calls, attempted calls, failed attempts, retries, LLM latency, operation counts, direct wall time for each instrumented operation, and serialized memory footprint.

The operation breakdown groups construction work under `base`, `original_evolution`, `typed_relations`, `temporal_state`, `provenance`, and `embedding`. The total run cost remains authoritative when features interact: provenance can create a different number of base notes, for example, and temporal transitions normally depend on typed-relation inference. Use combination totals for end-to-end comparisons and the feature/operation breakdown for direct attribution.

Metrics are written to the normal result JSON, the memory-build JSON, every unit snapshot, the memory manifest, and `memory_source.json` for query-only runs. Each unit reports compact JSON bytes, embedding-sidecar bytes, entry/chunk counts, and growth from its preceding unit. The overall footprint sums only the latest cumulative snapshot for each persona, so it does not double-count earlier evaluation units. Memory-only runs therefore retain cost and size statistics even though they do not write a query result. Older snapshots still load; their serialized size is measured when read, while unavailable historical call telemetry remains marked incomplete.

See [A-MEM Build Metrics](AMEM_BUILD_METRICS.md) for the complete schema, operation mapping, controlled-combination guidance, and a cross-run CSV example.

## Development and Verification

```bash
python -m pytest tests/test_amem_fix_agent.py tests/test_amem_test_agent.py tests/test_amem_staged_memory.py tests/test_build_feature_metrics.py
python main.py -m amem_fix_gemini -d medmemorybench --dry-run
python main.py -m persona_1/amem_fix_gemini -d medmemorybench --stage memory
python main.py --stage query --memory-run YYYYMMDD_HHMMSS
```

## Incremental Append

An exported A-MEM run can be extended without rebuilding its completed prefix:

```bash
python main.py --memory-run SOURCE_RUN --append --stage memory \
  --persona PERSONA_ID --unit UNIT_ID
python main.py --memory-run SOURCE_RUN --append \
  --persona PERSONA_ID --unit UNIT_ID
```

The first form only builds and exports memory. The second form restores the source snapshots, builds missing units through the inclusive target, and evaluates all units in that prefix. `UNIT_ID` is the dataset's global evaluation-unit ID; `PERSONA_ID` selects the persona that owns it. The source may be complete or interrupted, and its exact effective method/dataset configuration is read from `run_config.json`. Append output is a child run in `SOURCE_RUN/query_runs/`. Use `--resume` after interruption, and use a completed child run as the source for the next increment. If the source already contains the exact target snapshot, append is a no-op.

Prefer mocked provider tests. When changing retrieval, note serialization, feature flags, or batch preparation, update the corresponding A-MEM regression tests and verify snapshot round-trips. Keep audit fields and stable memory IDs intact; they are used by resume, rejudge, and experiment analysis.
