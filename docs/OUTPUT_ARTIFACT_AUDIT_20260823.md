# Output Artifact Audit — 2026-08-23

## Scope

Read-only review of the A-MEM build run
`outputs/amem_test_gemini-2.5-flash/20260822_115258` and representative
chain-enabled query run `query_runs/20260823_113111`, including serializers,
memory snapshots, batch manifests, result files, query-answer files, logs,
and checkpoint implementation.

The complete recent run-family results and child-run index are in [`AMEM_COMPLETED_RUNS_RESULTS_20260823.md`](AMEM_COMPLETED_RUNS_RESULTS_20260823.md).

## Verdict

The repository saves **substantially more diagnostic information than a typical
benchmark harness**. It is sufficient to reconstruct most A-MEM retrieval
decisions and to inspect batch prompts, outputs, and MCD judge rationales.

It is not yet ideal for controlled retrieval research or fast failure triage.
The main weakness is not missing raw evidence; it is that the evidence is
duplicated across large artifacts and lacks explicit, per-query links and
content hashes. A reviewer must manually join a query result to its answer
request, judge request, batch response, memory snapshot, and feature settings.

The highest-value change is a small, normalized per-query execution record.
Do not add another full copy of retrieval context or prompt text to
`*_query_answer.json`.

## What Is Already Strong

| Concern | Existing evidence | Assessment |
| --- | --- | --- |
| Run identity and effective configuration | `run_config.json` records command, selected configs, full method/dataset configuration, redacted API settings, execution stage, invocation history, and completion state. | Strong. |
| Memory lineage | `memory/manifest.json` records `build_id`, build and retrieval hashes, unit snapshot paths, timestamps, feature configuration, build telemetry, and memory size. Child query runs additionally write `memory_source.json`. | Strong for parent-child provenance. |
| Snapshot reproducibility | Each memory unit has a serialized state JSON plus a hash-addressed embedding sidecar. The manifest identifies the selected unit snapshot. | Strong, although immutable content verification can be improved. |
| Retrieval diagnostics | A-MEM query records retain seed IDs/scores, channel rankings, RRF contributions, graph scores and convergence, expansions, relations, final IDs, and chain selector diagnostics. The actual relation-aware context is retained. | Excellent for algorithm debugging. |
| Final answer requests | Query batch manifests retain stable request IDs, full messages, decoding parameters, a request fingerprint, timestamps, and a prepared-query snapshot. | Strong and sufficient for batch-resume prompt replay. |
| Judge inspection | Judge manifests retain full judge prompts, request IDs, decoding settings, parsed response text, token counts, and MCD node-level assessments. `evaluation_details` also retains the final reason and MCD node validations. | Strong. |
| Reliability and coverage | Build telemetry includes failures and retries by feature/operation; result files record coverage; terminal API failures are written to a separate artifact when present. Atomic run config and checkpoint writes are implemented. | Strong operational foundation. |

## Verified Limitations

### 1. Cross-stage lineage is implicit

The query-answer record has `query_id`, retrieved memories, final answer, and
metric details, but does not directly store:

- the final-answer batch `request_id` and manifest path;
- the judge `request_id` and manifest path;
- the source `build_id`, snapshot path, or snapshot-content hash used for that
  individual query;
- a prompt hash, retrieval audit hash, or selector-output hash.

These records can be joined today, but only by deriving stable IDs from code or
searching large manifests. This makes comparisons and automated debugging
needlessly fragile.

### 2. Replayability exists for batch resume, not as a first-class experiment control

`prepared_query` snapshots persist the exact batch messages and retrieval
state, which is excellent for resuming a submitted stage. However, retrieval
query rewriting, candidate ranking, selector output, prompt construction,
answer generation, and judging are not represented as separately versioned
replay stages with reusable artifact IDs.

Consequently, an ablation cannot simply say “reuse candidate set X and prompt
Y, regenerate only answers.” The data exists but needs bespoke extraction from
the batch manifest.

### 3. The storage layout is highly redundant

The query audit is nested in the first retrieved-memory object and includes the
full relation-aware context. The same prepared query and full prompt are then
copied into the query batch manifest. Retrieved memory content and relations
are also repeated across query records.

For the inspected run:

- root output directory: approximately **706 MiB**;
- root `*_query_answer.json`: approximately **16.7 MB**;
- root answer batch manifest: approximately **37.1 MB**;
- one chain query-run `*_query_answer.json`: approximately **21.3 MB**;
- its judge manifest: approximately **188 KB**.

The redundancy improves forensic convenience but makes ordinary inspection,
diffing, and archival expensive. It also obscures the small set of fields most
useful for experiment comparison.

### 4. The local batch manifest retains parsed responses, not provider rows

The manifest preserves response content, tokens, and status, but not the full
provider output row (`raw_response`) or an immutable local copy of the fetched
GCS output. This limits post-hoc diagnosis of provider schema changes,
safety/filter metadata, candidate metadata, malformed output, or unexpected
usage fields.

### 5. Query timing is not actionable for batch runs

The inspected query-answer artifacts report `query_time: 0.0`. This is expected
because a batch job has queue and stage timing rather than meaningful
per-request latency, but the field currently looks like a measured zero.
There is no explicit timing provenance such as `timing_kind: "batch_unavailable"`
or separate local preparation, submission, queue, collection, and finalization
durations per request.

### 6. Environment and code identity are absent

`run_config.json` preserves the executable path and command, but not a Git
revision/dirty state, Python version, platform, dependency lock hash, or a
canonical configuration-file hash. This makes it difficult to prove that two
runs used identical code and dependency environments.

### 7. Snapshot integrity is not explicit at the manifest level

Embedding filenames include content-like hashes, but the manifest snapshot
records do not provide a clear SHA-256 digest for every state JSON and sidecar.
The checkpoint itself has an integrity hash, but completed root/child output
directories may have no retained checkpoint file after successful cleanup.

## Recommended Additions

### Priority 0 — Add one normalized per-query record

Write `artifacts/queries/<query_id>.json` or append JSONL records under
`artifacts/query_executions.jsonl`. Keep the existing files backwards
compatible. Each record should be small and joinable:

```text
schema_version, run_id, query_id, context_id, unit_id, query_type
source: {build_id, snapshot_path, snapshot_sha256, build_config_hash,
         retrieval_config_hash}
retrieval: {raw_question_hash, retrieval_query, retrieval_query_hash,
            candidate_ids_hash, final_memory_ids, final_ids_hash,
            audit_path, audit_hash, selector_version}
answer: {request_id, manifest_path, prompt_hash, response_hash,
         model, decoding, token_counts, status, timing_kind}
judge: {request_id, manifest_path, prompt_hash, response_hash,
        model, decoding, status}
outcome: {score, is_correct, metric, evaluation_details_path}
```

This is the preferred entry point for debugging and analysis. It should point
to full payloads rather than copy them.

### Priority 0 — Add immutable hashes and stage IDs

Record SHA-256 hashes for each serialized memory snapshot, embedding sidecar,
retrieval audit, final prompt, answer text, judge prompt, and judge response.
Add a `stage_id`/`replay_id` derived from the inputs that affect that stage.

This enables precise ablations:

| Desired control | Frozen IDs/hashes |
| --- | --- |
| Compare selectors only | snapshot, retrieval query, candidate set, selector input |
| Compare prompts only | final evidence IDs and relation-aware context |
| Estimate answer variance | final prompt hash |
| Estimate judge variance | answer hash and judge prompt hash |

### Priority 1 — Separate index records from verbose payloads

Keep complete prompt/context payloads, but move them to content-addressed
sidecars, for example:

```text
artifacts/prompts/<sha256>.txt
artifacts/retrieval_audits/<sha256>.json
artifacts/provider_rows/<request_id>.json.gz
artifacts/judgments/<request_id>.json
```

The query-answer file should retain concise retrieved-memory summaries and the
reference/hash/path to verbose data. This removes the current practice of
attaching the entire `retrieval_audit` to the first memory record.

### Priority 1 — Retain raw provider output safely

For batch jobs, save the fetched GCS output rows either verbatim in compressed
JSONL or as per-request raw sidecars. Link each to the parsed manifest response
using a SHA-256 digest. Apply the same policy to real-time provider responses,
subject to privacy and retention requirements.

This is specifically useful when a parsed response is empty, filtered,
malformed, or has unexpected token accounting.

### Priority 1 — Make timing semantics explicit

Replace ambiguous zero values with fields such as:

```text
timing_kind: "batch_stage" | "realtime_latency" | "unavailable"
prepare_seconds, local_finalize_seconds, submit_to_complete_seconds,
provider_latency_seconds, queue_seconds
```

For batch mode, stage-level durations are sufficient when per-request provider
latency is unavailable; do not invent an allocation.

### Priority 2 — Capture execution environment

At run start, add `environment.json` with:

- Git commit, branch, dirty flag, and optional diff hash;
- Python version, OS/platform, and executable path;
- `requirements.txt` SHA-256 and installed-package snapshot or lockfile hash;
- config-file SHA-256 values; and
- model/provider SDK versions when available.

No credentials, environment variables, private dataset text, or service-account
contents should be recorded.

### Priority 2 — Add explicit retention levels

Define `output_detail: summary | audit | forensic`:

- **summary**: metrics, config, IDs, hashes, and concise evidence metadata;
- **audit**: summary plus prompts, retrieval audits, and parsed responses;
- **forensic**: audit plus raw provider rows and verbose snapshots.

The current behavior is close to `audit`, but without a clear contract or
normalization. Defaulting to audit is reasonable for this research repository;
compressed sidecars should reduce its cost.

## What Not To Add

- Do not duplicate the entire prompt in `query_answer.json` when the batch
  manifest/sidecar already owns it.
- Do not add embedding vectors to per-query outputs; record model/version and
  snapshot-sidecar hash instead.
- Do not persist API keys, service-account contents, raw `.env` values, or
  unredacted credential-bearing request metadata.
- Do not serialize a separate full copy of every memory snapshot for every
  query; a snapshot hash plus selected memory IDs is sufficient.

## Suggested Migration

1. **Additive schema only.** Preserve existing `run_config.json`, result,
   query-answer, manifest, and checkpoint formats.
2. Introduce a versioned query-execution index and write it alongside existing
   outputs.
3. Store prompt/audit/provider payloads in hash-addressed sidecars and reference
   them from both the index and legacy records where useful.
4. Add a small reader utility that joins a query ID to its memory source,
   retrieval audit, answer request/response, judge request/response, and final
   metric outcome.
5. After several runs validate storage savings and replay parity, optionally
   reduce verbose duplication in new query-answer schema versions.

## Bottom Line

Add more **structure and linkage**, not substantially more duplicated content.
The current artifacts can already explain most observed A-MEM failures,
including selector inputs/outputs and MCD judge-node failures. The next schema
should make those investigations deterministic, one-command, and suitable for
frozen-candidate and frozen-prompt experiments.