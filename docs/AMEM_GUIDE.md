# A-MEM Method Guide

This guide covers the A-MEM integration in MedMemoryBench. Repository-wide Python, testing, and secret-handling rules still apply.

## Integration Structure

- `methods/amem_agent.py` provides the shared `AMemAgent` adapter and snapshot support.
- `methods/amem_fix_agent.py` implements the paper-aligned `AMemFixAgent` flow.
- `methods/amem_test_agent.py` extends it with experimental typed relations, temporal state, provenance, and chain-preserving evidence selection.
- `methods/amem/A-mem/memory_layer_robust.py` is the active plain-text LLM memory layer; `memory_layer_typed.py` is used only by `amem_test`.
- `methods/amem/A-mem/memory_layer.py` and the scripts in that directory are the upstream/standalone implementation and LoCoMo experiments. They are not the normal MedMemoryBench entry point.
- Configs are in `configs/method_config/amem*.yaml` and `configs/method_config/persona_1/`. Regression coverage is in `tests/test_amem_fix_agent.py`, `tests/test_amem_test_agent.py`, and `tests/test_amem_staged_memory.py`.

## Runtime Pipeline

1. `AgentManager` selects `amem`, `amem_fix`, or `amem_test` from the YAML `method_name` and forwards `agent_params`.
2. Each persona/context receives its own memory system. Notes use a SentenceTransformer embedding retriever plus an internal metadata/evolution LLM.
3. `amem` splits oversized text into token-bounded chunks and passes each chunk directly to `add_note()`.
4. `amem_fix` converts structured dialogue turns into timestamped atomic notes, optionally generates keyword queries, retrieves semantic seeds, and expands linked notes.
5. `amem_test` defaults to the atomic turn-note flow, but `amem_note_level` can combine all turns from each injected benchmark session into one note input. Both modes use the same A-MEM analysis, evolution, typed-edge (`SUPPORT`, `REFINE`, `SUPERSEDE`, `CONFLICT`, `RELATED`), temporal-state, provenance, indexing, and retrieval paths.
6. Retrieval is local and read-only during query preparation, apart from the optional LLM call that turns a question into keywords. That rewrite uses the top-level query `model`, while note metadata and evolution use `memorize_model`. Gemini/AI Studio build-client initialization is deferred while restoring a snapshot, unless a build operation invokes it. Retrieved notes are placed in the final prompt; Vertex batch transport applies to final answer generation. `--stage memory` and `--stage query` can build and reuse snapshots.

The robust layer may make up to three conditional evolution calls when adding a note: decide whether to evolve, strengthen details, and update neighbors. `amem_evo_threshold` controls periodic retriever consolidation.

## Configuration

AMEM method files separate `build_config` from `retrieval_config`. Build settings define the stored snapshot: embedding, chunking, metadata/evolution settings, and `amem_build_max_context_tokens`. The optional top-level `memorize_model` block independently configures the internal build LLM, including its provider, model, credentials, OpenRouter routing, and service tier. The top-level `model` configures both query keyword rewrites and final-answer generation. Legacy `build_config.amem_backend` and `build_config.amem_model` remain supported when `memorize_model` is absent. Retrieval settings can change on a frozen snapshot, including `retrieve_num`, keyword use, graph budgets, `amem_relation_min_confidence`, temporal ordering, provenance injection, and `amem_max_context_tokens`.

For example, use OpenRouter flex for memory construction while leaving query answering on OpenRouter's default tier so it remains independently eligible for Batch API routing:

```yaml
model:
  provider: openrouter
  name: openai/gpt-5-nano
  openrouter:
    provider:
      only: [openai]
      allow_fallbacks: false
  temperature: 0.0
  max_completion_tokens: 5000
  reasoning_effort: high

memorize_model:
  provider: openrouter
  name: openai/gpt-5-nano
  openrouter:
    provider:
      only: [openai]
      allow_fallbacks: false
    service_tier: flex
  temperature: 0.0
  max_completion_tokens: 1000
  reasoning_effort: high

build_config:
  amem_embedding_model: all-MiniLM-L6-v2
  amem_evo_threshold: 100
  amem_build_max_context_tokens: 200000
```

Omitting `model.openrouter.service_tier` selects normal/default query routing; `memorize_model.openrouter.service_tier` affects only build calls. A-MEM construction remains real-time because it is stateful. With `--batch-api`, only the independently configured query-answer and judge stages use batch when their selected routes support it. Keep API keys in `.env`, or use the separate `api_key` and `base_url` fields in each model block for non-secret local configuration.

For `amem_test`, the important build switches are `amem_note_level`, `amem_original_evolution`, `amem_typed_relations`, `amem_temporal_state`, and `amem_provenance`. `amem_temporal_transition_min_confidence` controls which inferred relations update stored temporal state. Query-time `amem_relation_min_confidence` only filters retrieved relations. Temporal state requires typed relations. Snapshot selection validates build semantics while allowing retrieval-only ablations; legacy manifests derive a build hash from their stored `run_config.json` when possible.

Set note granularity under `build_config`:

```yaml
build_config:
  amem_note_level: turn     # default; one note input per dialogue turn
```

```yaml
build_config:
  amem_note_level: session  # one combined note input per benchmark session
```

`turn` preserves the historical `amem_test` behavior, including turn text, timestamps, note attributes, and turn-level provenance. `session` combines the same normalized speaker turns before `add_note()`, uses the session timestamp, and records session-level provenance with `source_turn_id: null` and `source_text_scope: source_session`. Oversized combined sessions still split at `amem_chunk_size_tokens`, so one session can produce multiple token-bounded note parts, matching the base A-MEM chunking rule. Retrieval configuration and query behavior are unchanged. Because granularity changes stored memory, snapshots built with `turn` and `session` are intentionally incompatible; older snapshots without this field load as `turn`.

For controlled retrieval experiments, set `amem_hybrid_retrieval`, `amem_graph_ranking_mode`, and `amem_chain_selection` under `retrieval_config`. Hybrid RRF channels have independent non-negative weights; zero disables a channel. `fixed_bfs`, `amem_hybrid_retrieval: false`, and `amem_chain_selection: false` preserve prior behavior. `typed_ppr` requires typed retrieval and weights transitions by relation type, confidence, direction, temporal compatibility, and deterministic query intent. These settings are query-only, so they can be crossed over compatible frozen snapshots without changing the build hash. Match `amem_max_context_tokens` when comparing policies.

Session-level `amem_test` runs automatically report source-session retrieval quality. Gold session IDs come from each query's `source_key_points`; predicted IDs come from the retrieved notes' stored `source_session_id`. Per-query output is under `evaluation_details.metric_groups.retrieval_quality`, with deduplicated predicted/gold/matched IDs, TP/FP/FN counts, precision, recall, F1, average precision, reciprocal rank, hit, and exact match. Aggregate macro, micro, coverage, MAP, MRR, hit-rate, exact-match-rate, and per-query-type summaries are under `result.json -> summary.metric_groups.retrieval_quality`. Turn-level A-MEM runs do not emit this group. Queries without gold IDs, and legacy retrieved notes that lack session identity, are marked unavailable rather than scored as zero; rebuild older session-level snapshots to obtain complete metrics.

Use `amem_regex_intent_conditioning` to run the intent-free retrieval ablation:

```yaml
retrieval_config:
  amem_regex_intent_conditioning: true  # existing regex-conditioned behavior
```

```yaml
retrieval_config:
  amem_regex_intent_conditioning: false  # intent-free graph/temporal semantics
```

When the flag is `false`, regex-derived causal/conflict/detail/change graph intents and current/historical/change temporal intents do not affect RRF state channels, typed-PPR adjacency, temporal ordering/expansion, or chain structural facets. Explicit targets such as `2025`, `2025-03`, and `March 2025` remain parsed and can still drive timestamp and validity matching. Typed PPR still uses configured relation weights multiplied by relation confidence, plus explicit temporal compatibility when a parsed target is available. Retrieval audits expose `regex_intent_conditioning`, `graph_relation_intents`, and the resulting `temporal_query`.

The disabled-mode query flow is:

```text
raw question
  -> keyword rewrite -> dense/BM25/entity overlap/timestamp/explicit-state channels
  -> weighted RRF -> hybrid seeds -> typed PPR using relation weight x confidence
  -> explicit-time temporal expansion/order (when enabled) -> optional chain selection
```

When enabled, chain selection runs after graph/temporal retrieval and before final provenance formatting. It fairly fuses retrieval, hybrid, graph, and dense rankings, limits the fused pool with `amem_chain_candidate_count`, and selects up to `amem_chain_evidence_count` memories; no upstream ID is forced into the result. It scores calibrated relevance, question-facet coverage, soft graph connectivity, complete short paths, available temporal compatibility, and bounded redundancy, then adds fitting singletons or complete paths by marginal utility per added memory. Path search is deterministically bounded: `amem_chain_max_hops` is clamped to 3, each memory keeps up to eight strongest path neighbors, and at most 100 path actions are retained. The exact final A-MEM context formatter supplies each proposal's token cost, so accepted evidence—including relation, temporal, provenance, raw-source, and note text—fits the remaining prompt budget without post-selection truncation. When a target-sized set fits, exact-cost completion reservation prevents an expensive early choice from making `amem_chain_evidence_count` unreachable. The process uses the raw question, any temporal target resolved by normal retrieval, and stored memory; unresolved relative dates do not match every timestamp. Benchmark query labels, expected answers, gold evidence, and scoring metadata are not inputs.

```yaml
retrieval_config:
  amem_chain_selection: false
  amem_chain_candidate_count: 50
  amem_chain_evidence_count: 30
  amem_chain_max_hops: 2
  amem_chain_max_groups: 3
  amem_chain_relevance_weight: 1.0
  amem_chain_coverage_weight: 1.0
  amem_chain_connectivity_weight: 0.35
  amem_chain_path_weight: 0.75
  amem_chain_temporal_weight: 0.5
  amem_chain_redundancy_weight: 0.25
```

`amem_chain_max_groups` is a soft preference, so independently relevant evidence is not rejected solely because the typed graph is sparse. State-based temporal utility is used only when temporal retrieval is active; explicit dates still use direct source-timestamp matching. Use `extra.chain_selection` in query results to inspect fused ranks, candidates, selected paths, utility components, exact selected tokens, target count, greedy steps, and rejection reasons. All shipped `amem_test*` configs explicitly declare the complete base, hybrid, graph, and chain retrieval surface.

## JSON Artifact Auditability

`run_config.json` preserves the original method YAML under
`method_config.raw_config`, while `method_config` records the parsed default
values. `method_config.effective_agent` is the authoritative runtime view: it
names the selected adapter and stores the fully resolved constructor keyword
arguments after adapter defaults, model routing, token limits, and configured
embedding fallbacks are applied. Credentials in both views are redacted. For
A-MEM adapters, `method_config.build_config`, `retrieval_config`, and
`agent_params` also contain their resolved values, including defaults omitted
from the YAML. This lets sparse configs be audited and used for staged-run
reconstruction without guessing which defaults were in effect.

New query reports use `medmemorybench.query_answers` schema version `2`.
They keep the ordinary `*_query_answer.json` small enough to inspect and diff:
each query has result fields, retrieved-memory IDs, a retrieval reference, and
`execution_references` for the memory snapshot, answer request, and judge
request. Batch references include the manifest path, request ID, and stable
prompt/response hashes; snapshot references include the build ID, config
hashes, paths, and integrity digests. Real-time calls explicitly use a local
correlation ID rather than claiming a provider request ID.

Full retrieval payloads are not copied into version-2 query reports. Batch
runs reference the persisted `prepared_query` in the answer batch manifest.
Real-time runs write the full payload once to the sibling
`*_retrieval_records.json` named by `retrieval_records_path`. The query report
summary also contains accuracy, score, type/metric groups, retrieval metric
groups, and selected-run coverage, so it can be interpreted without joining
the result report.

Every result, memory-build, and query-answer summary also writes
`stage_usage` schema version `1`. It separates `memory_build`,
`retrieval_preparation`, `answer`, and `judge` usage instead of combining them
under a generic query total. Each stage reports successful/attempted calls,
failures/retries, input/output/visible/thinking tokens, provider latency, and
local wall time. `batch_stages` adds the persisted manifest path, model,
provider, request/response outcomes, retries, provider token totals, and
submission-to-completion elapsed time. Per-query `execution_usage` carries
the exact answer and judge provider token counts when a batch response exposes
them. `unattributed_query_usage` is retained explicitly so unsupported
adapters cannot silently be misclassified. Monetary cost is deliberately not
estimated because the repository does not configure a versioned provider price
schedule; `cost.available` is `false` with the reason, rather than an invented
number.

For batch answers, `query_time` remains a compatibility field and is not a
per-request latency measurement. The per-query `timing.kind` explicitly says
`per_request_latency_unavailable_for_batch`, while result and query summaries
report stage wall time and batch lifecycle elapsed time instead.

`memory/manifest.json` records each snapshot's `integrity_hash` and embedding
sidecar `embedding_sha256` in addition to its path. `expected_queries` always
counts the evaluation units selected by the run, rather than every query in
the underlying dataset. Older artifacts remain readable but do not gain these
additive fields retroactively.

## Build Metrics and Feature Reports

For the completed hybrid-retrieval, graph-ranking, and chain-selection run matrix, see [A-MEM Completed Run Results](AMEM_COMPLETED_RUNS_RESULTS_20260823.md). It links each policy group to its frozen build root, child `query_runs/`, coverage, and aggregate/per-query-type outcomes. The detailed selector interpretation is in [A-MEM Chain-Selection Audit](AMEM_CHAIN_SELECTION_AUDIT_20260823.md).

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
