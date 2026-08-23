# A-MEM Completed Run Results: Hybrid Retrieval, Graph Ranking, and Chain Selection

Date: 2026-08-23

## Scope and decision summary

This document consolidates the completed A-MEM runs added on 2026-08-21–23, including their fixed-snapshot child query runs. It is the current results index for the recent `amem_test` experiment families. The detailed implementation and chain-selector audit remains in [`AMEM_CHAIN_SELECTION_AUDIT_20260823.md`](AMEM_CHAIN_SELECTION_AUDIT_20260823.md); the output-schema review remains in [`OUTPUT_ARTIFACT_AUDIT_20260823.md`](OUTPUT_ARTIFACT_AUDIT_20260823.md).

All recent evaluations use Persona 1, 50 sessions, 788 final notes, and 49 scored queries:

| Query type | Count |
|---|---:|
| Entity exact match (EEM) | 10 |
| Temporal localization (TLA) | 10 |
| State update (SUA) | 5 |
| Multiple choice (MQ) | 10 |
| Inference generation (IG) | 10 |
| Multi-hop clinical deduction (MCD) | 4 |
| **Total** | **49** |

The main descriptive conclusions are:

1. The 2026-08-22 frozen typed snapshot with hybrid retrieval and `typed_ppr` produced the strongest recent no-chain repeated mean: `0.657 ± 0.015` accuracy and `0.678 ± 0.009` average score.
2. Enabling the current chain selector on that same snapshot reduced the repeated mean to `0.641 ± 0.044` accuracy and `0.648 ± 0.041` average score. MCD binary correctness increased from `1/20` to `3/20`, but the detailed audit found the MCD composite score decreased from `0.158` to `0.138`.
3. The 2026-08-23 fixed-BFS child runs used a different evolution-only memory build and are not a causal comparison with the hybrid/typed-PPR runs. They scored `0.547 ± 0.008` accuracy and `0.579 ± 0.011` average score.
4. The 2026-08-21 policy groups show the same general limitation: untyped links had the best repeated mean on that all-feature snapshot, while typed and temporal traversal did not produce a repeatable overall gain.

These are conditional, descriptive results. Keyword rewriting, final answer generation, batch behavior, and judging remain stochastic. A score difference across independently built snapshots or unmatched query policies is not a construction-level causal claim.

## 1. Run lineage and build identities

The recent artifacts contain three relevant root builds. A child under `query_runs/` reuses the source snapshot unless explicitly described as an append/build child; the child `run_config.json` and `memory_source.json` are authoritative for the exact lineage.

| Root run | Build combination | Build ID | Build hash | Final memory | Child query runs |
|---|---|---|---|---:|---:|
| `20260821_023755` | Base + original evolution + typed relations + temporal state + provenance | `3ed8da80-b766-4a9d-8ea1-8a751f8f1236` | `21c89b46414f775b` | 788 notes; 18.780 MiB serialized state | 20 |
| `20260822_115258` | Base + typed relations; evolution, temporal state, and provenance off | `ef0ee1bd-e4eb-45e9-a8af-32f0d0f6fe1b` | `a063bcd9e00f6d3d` | 788 notes; 15.075 MiB serialized state | 10 |
| `20260822_181747` | Base + original evolution; typed relations, temporal state, and provenance off | `bfcee78b-b343-4bc1-989b-b44769298572` | `40ae7c7fef51367b` | 788 notes; 9.225 MiB serialized state | 5 |

The root `20260822_115258` result itself is a completed build-plus-query artifact with `32/49` correct and average score `0.669`. Its ten children are the controlled recent comparison: five no-chain runs and five chain-enabled runs over the same frozen snapshot.

The root `20260821_023755` is an all-feature build. Its children are query-policy runs over that snapshot; a child with typed or temporal retrieval disabled is not a separately built `amem_fix` or typed-only memory.

## 2. Policy groups and repeated results

Values are mean ± population standard deviation across five completed child runs. Accuracy is binary correctness; average score is the metric-aware score stored in each result JSON. The per-type column reports aggregate correct outcomes over the 25 executions in each group, using `MQ / EEM / TLA / SUA / IG / MCD` order.

| Group | Root snapshot | Query policy | Child count | Accuracy | Average score | Per-type correct over 25 |
|---|---|---|---:|---:|---:|---|
| 2026-08-21 links | `20260821_023755` | Untyped links | 5 | `0.600 ± 0.033` | `0.627 ± 0.032` | 32 / 47 / 38 / 16 / 9 / 5 |
| 2026-08-21 typed | `20260821_023755` | Typed retrieval | 5 | `0.551 ± 0.026` | `0.569 ± 0.026` | 33 / 40 / 39 / 5 / 14 / 4 |
| 2026-08-21 temporal | `20260821_023755` | Temporal retrieval/order | 5 | `0.547 ± 0.020` | `0.577 ± 0.012` | 30 / 41 / 40 / 7 / 16 / 0 |
| 2026-08-21 provenance | `20260821_023755` | Provenance with raw-source injection | 5 | `0.584 ± 0.028` | `0.604 ± 0.023` | 31 / 39 / 45 / 6 / 14 / 8 |
| 2026-08-22 hybrid/PPR, no chain | `20260822_115258` | Hybrid retrieval + `typed_ppr`; chain off | 5 | **`0.657 ± 0.015`** | **`0.678 ± 0.009`** | 39 / 50 / 42 / 10 / 19 / 1 |
| 2026-08-23 hybrid/PPR, chain | `20260822_115258` | Hybrid retrieval + `typed_ppr` + chain selection | 5 | `0.641 ± 0.044` | `0.648 ± 0.041` | 41 / 46 / 41 / 9 / 17 / 3 |
| 2026-08-23 fixed BFS | `20260822_181747` | Evolution-only build + `fixed_bfs` query | 5 | `0.547 ± 0.008` | `0.579 ± 0.011` | 30 / 44 / 38 / 11 / 11 / 0 |

The 2026-08-21 groups have a shared snapshot but different retrieval features. The 2026-08-22 hybrid/PPR groups have the same snapshot and upstream query settings; only chain selection differs. The fixed-BFS group has a different source build and should be treated as a reference, not a matched ablation.

### 2.1 Exact child-run index

Every child below completed 49/49 declared queries. The score pairs are `correct / 49` and average score.

| Group | Child runs and outcomes |
|---|---|
| 2026-08-21 links | `171423` 27/49, 0.578; `173021` 29/49, 0.616; `174135` 30/49, 0.636; `175227` 32/49, 0.677; `180422` 29/49, 0.626 |
| 2026-08-21 typed | `182104` 29/49, 0.605; `183147` 26/49, 0.546; `184129` 28/49, 0.595; `185224` 26/49, 0.554; `190324` 26/49, 0.544 |
| 2026-08-21 temporal | `193715` 27/49, 0.575; `194929` 25/49, 0.558; `200122` 28/49, 0.595; `201424` 27/49, 0.575; `202552` 27/49, 0.582 |
| 2026-08-21 provenance | `204736` 30/49, 0.626; `210340` 30/49, 0.630; `211739` 27/49, 0.578; `213042` 29/49, 0.609; `214719` 27/49, 0.575 |
| 2026-08-22 hybrid/PPR, no chain | `163906` 31/49, 0.670; `165202` 33/49, 0.681; `170415` 32/49, 0.672; `171549` 33/49, 0.694; `173813` 32/49, 0.673 |
| 2026-08-23 hybrid/PPR, chain | `113111` 29/49, 0.602; `114657` 30/49, 0.606; `120523` 34/49, 0.697; `121751` 34/49, 0.691; `123207` 30/49, 0.644 |
| 2026-08-23 fixed BFS | `050401` 27/49, 0.580; `052418` 27/49, 0.580; `053401` 26/49, 0.558; `054447` 27/49, 0.587; `055449` 27/49, 0.587 |

Full paths are under `outputs/amem_test_gemini-2.5-flash/<root>/query_runs/<child>/`. The shortened IDs in this table are unambiguous within their parent root.

## 3. Exact query policies

### Frozen typed snapshot: hybrid retrieval and `typed_ppr`

The ten children of `20260822_115258` share these upstream settings:

```text
hybrid retrieval = true
retrieve_num = 10
graph ranking = typed_ppr
typed expansion budget = 20
ordinary-link expansion = false
temporal retrieval = false
provenance retrieval = false
answer temperature = 0
```

No-chain children:

- `20260822_163906`
- `20260822_165202`
- `20260822_170415`
- `20260822_171549`
- `20260822_173813`

Chain-enabled children add:

```text
chain selection = true
candidate pool = 50
selected evidence target = 30
maximum path hops = 2
preferred graph components = 3 (soft preference)
weights = relevance 1.0, coverage 1.0, connectivity 0.35,
          path 0.75, temporal 0.5, redundancy 0.25
```

The selector selected exactly 30 memories in every audited query. It retained all ten semantic seeds and replaced only a small fraction of the fused top-30 pool with lower-ranked candidates. The detailed chain audit reports a mean of 6.30 disconnected components, so the current selector should be interpreted as a broad 30-note reranker rather than a sparse causal-chain extractor.

### All-feature snapshot: 2026-08-21 policy groups

The 20 children of `20260821_023755` comprise four five-run query groups. Their common build is `base_memory + original_evolution + typed_relations + temporal_state + provenance`. The groups vary query-time retrieval/context policy; the exact effective settings are recorded in each child `run_config.json` and `memory_source.json`. Use the group labels above as the stable analysis labels rather than inferring the build from a child directory name.

### Evolution-only snapshot: fixed BFS

The five children of `20260822_181747` use a separate `base_memory + original_evolution` snapshot and `fixed_bfs` retrieval. They are useful for continuity with the previous graph implementation, but differences from the hybrid/PPR family combine build changes and query-policy changes.

## 4. Per-query-type interpretation

The aggregate type counts indicate where the policies were strong or weak, but they are not supporting-evidence recall measurements.

- **Hybrid + `typed_ppr`, no chain:** strongest aggregate EEM (`50/50`) and IG (`19/50`) among the recent groups; MCD remained weak (`1/20`).
- **Hybrid + `typed_ppr`, chain:** MQ increased descriptively (`41/50` versus `39/50`), while EEM, TLA, SUA, and IG decreased. MCD binary correctness increased to `3/20`, but this did not translate into a better MCD composite score; see [`AMEM_CHAIN_SELECTION_AUDIT_20260823.md`](AMEM_CHAIN_SELECTION_AUDIT_20260823.md).
- **All-feature temporal retrieval:** TLA was relatively strong (`40/50`), but MCD was `0/20` and SUA remained low (`7/25`). This supports further temporal-index work, not a claim that temporal traversal improves accuracy.
- **All-feature provenance:** TLA was highest among the 2026-08-21 groups (`45/50`), and MCD reached `8/20`, but provenance changes prompt content and should be evaluated with matched evidence and token budgets.
- **Fixed BFS:** MCD was `0/20`; this group is not directly comparable to the frozen typed/PPR family because its memory build differs.

For clinical-looking MCD questions, the current final answer can remain generic even when related notes are retrieved. The selector audit therefore recommends measuring endpoint recall, minimal-chain recall, node-level NCR/CRC/CC, and evidence-to-claim citations before treating answer scores as retrieval evidence.

## 5. Artifact navigation

For each root or child run, inspect artifacts in this order:

1. `run_config.json` — effective command, method/dataset configuration, stage, status, and invocation metadata.
2. `memory/manifest.json` — build ID, build/retrieval hashes, feature configuration, snapshot paths, build metrics, and memory size.
3. `memory_source.json` in query children — source-run and snapshot lineage.
4. `*_result.json` — aggregate and per-query-type scores, coverage, feature metadata, and build summary.
5. `*_query_answer.json` — per-query answers, retrieved-memory records, evaluation details, and embedded retrieval audit.
6. `batch/` — prepared-query snapshots and final-answer request/response manifests.
7. `query_runs/<child>/` — repeated query-policy executions over the parent snapshot.

The current artifacts are sufficient for detailed forensic inspection, but cross-stage IDs and hashes are not yet normalized. See [`OUTPUT_ARTIFACT_AUDIT_20260823.md`](OUTPUT_ARTIFACT_AUDIT_20260823.md) before designing another frozen-candidate experiment.

## 6. Interpretation limits and next experiment

Do not compare these `49`-query results directly with the historical `97`-query raw `amem` result. Do not infer build-feature gains from the fixed-BFS group or from the all-feature groups without matched source snapshots. Do not interpret MCD answer scores as medical correctness or clinical suitability.

The next controlled experiment should freeze, for every question:

1. the generated retrieval query;
2. dense/BM25/hybrid candidate IDs and scores;
3. graph scores and eligible edges;
4. the exact final prompt and token budget; and
5. the answer and judge stages as separate repeats.

Then compare ranked bags, the current 30-note selector, and genuinely sparse directed-chain selectors at matched token budgets. Advance a new selector only if it improves annotated minimal-chain recall and MCD NCR/CRC/CC without unacceptable overall loss.

## Source references

- Recent hybrid/PPR root and children: `outputs/amem_test_gemini-2.5-flash/20260822_115258/`
- Recent fixed-BFS root and children: `outputs/amem_test_gemini-2.5-flash/20260822_181747/`
- Earlier all-feature root and children: `outputs/amem_test_gemini-2.5-flash/20260821_023755/`
- Experimental adapter: `methods/amem_test_agent.py`
- Typed retrieval and selector: `methods/amem/A-mem/memory_layer_typed.py`
- Evaluation and result serialization: `benchmarks/medmemorybench/evaluator.py`, `src/result.py`
- Historical aggregate analysis: [`AMEM_EXPERIMENT_ANALYSIS.md`](AMEM_EXPERIMENT_ANALYSIS.md)
- Implementation/configuration reference: [`AMEM_GUIDE.md`](AMEM_GUIDE.md), [`AMEM_IMPLEMENTATION.md`](AMEM_IMPLEMENTATION.md)
