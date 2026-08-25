# A-MEM Completed Run Results: Typed Retrieval, Graph Ranking, and Chain Selection

Date: 2026-08-24

## Scope and decision summary

This document consolidates the retained completed A-MEM runs from 2026-08-22–24, including their fixed-snapshot child query runs. It is the current results index for the recent `amem_test` experiment families. The detailed implementation and chain-selector audit remains in [`AMEM_CHAIN_SELECTION_AUDIT_20260823.md`](AMEM_CHAIN_SELECTION_AUDIT_20260823.md); the output-schema review remains in [`OUTPUT_ARTIFACT_AUDIT_20260823.md`](OUTPUT_ARTIFACT_AUDIT_20260823.md).

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

## Aggregate results by experiment

Each row averages all completed child query runs assigned to the experiment. `Correct/49`, accuracy, overall score, and each query-type cell are arithmetic means across those children; query token and call fields are also child-run means. Query-type cells use `score / binary accuracy`. Calls use `successful / attempted / failed / retried`. Build telemetry is the shared root-build total, not averaged across child runs. The pooled `2026-08-22–24 typed - hybrid/PPR` group has ten children; all other groups have five.

| Experiment name | Correct/49 | Accuracy | Score | EEM | MQ | TLA | SUA | IG | MCD | Build input/output/calls | Query input/output/calls |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 2026-08-22–24 typed - hybrid/PPR | 32.60/49 | 0.665 | 0.682 | 1.000/1.000 | 0.845/0.790 | 0.840/0.840 | 0.420/0.420 | 0.390/0.390 | 0.138/0.075 | 3,514,359 / 88,666 / 1,575/2,054/479/479 | 1,046,039 / 9,910 / 137.0/155.6/18.6/18.6 |
| 2026-08-23 typed - hybrid/PPR + chain | 31.40/49 | 0.641 | 0.648 | 0.920/0.920 | 0.860/0.820 | 0.820/0.820 | 0.360/0.360 | 0.340/0.340 | 0.138/0.150 | 3,514,359 / 88,666 / 1,575/2,054/479/479 | 1,037,125 / 9,987 / 137.0/153.6/16.6/16.6 |
| 2026-08-24 typed - fixed BFS | 26.00/49 | 0.531 | 0.558 | 0.800/0.800 | 0.730/0.600 | 0.720/0.720 | 0.360/0.360 | 0.260/0.260 | 0.106/0.100 | 3,514,359 / 88,666 / 1,575/2,054/479/479 | 925,094 / 9,601 / 137.0/147.8/10.8/10.8 |
| 2026-08-23 original evolution - fixed BFS | 26.80/49 | 0.547 | 0.579 | 0.880/0.880 | 0.747/0.600 | 0.760/0.760 | 0.440/0.440 | 0.220/0.220 | 0.020/0.000 | 7,823,162 / 131,570 / 2,718/3,416/698/698 | 837,898 / 9,869 / 137.0/148.2/11.2/11.2 |

The main descriptive conclusions are:

1. The pooled `2026-08-22–24 typed - hybrid/PPR` ten-run group on the frozen typed snapshot produced `0.665 ± 0.023` accuracy and `0.682 ± 0.019` average score. All children used the same A-MEM retrieval settings and effective Gemini model, so the pooled result is the appropriate summary of this experiment.
2. Enabling the current chain selector on that same snapshot reduced the repeated mean to `0.641 ± 0.044` accuracy and `0.648 ± 0.041` average score. MCD binary correctness increased from `1/20` to `3/20`, but the detailed audit found the MCD composite score decreased from `0.158` to `0.138`.
3. The `2026-08-23 original evolution - fixed BFS` child runs used a separate evolution-only memory build and are not a causal comparison with the hybrid/typed-PPR runs. They scored `0.547 ± 0.008` accuracy and `0.579 ± 0.011` average score.
4. The ten pooled hybrid/PPR repeats range from the earlier `0.657` mean accuracy to the later `0.673` mean accuracy. Pooling them gives `0.665`; the difference between repeat batches is descriptive stochastic variation, not evidence of a retrieval-feature gain.
5. The `2026-08-24 typed - fixed BFS` group scored `0.531 ± 0.037` accuracy and `0.558 ± 0.032` average score. It is a typed-only query policy with effective `fixed_bfs` traversal, not a no-graph configuration.
These are conditional, descriptive results. Keyword rewriting, final answer generation, batch behavior, and judging remain stochastic. A score difference across independently built snapshots or unmatched query policies is not a construction-level causal claim.

## 1. Run lineage and build identities

The retained artifacts contain two relevant root builds. A child under `query_runs/` reuses the source snapshot unless explicitly described as an append/build child; the child `run_config.json` and `memory_source.json` are authoritative for the exact lineage.

| Root run | Build combination | Build ID | Build hash | Final memory | Child query runs |
|---|---|---|---|---:|---:|
| `20260822_115258` | Base + typed relations; evolution, temporal state, and provenance off | `ef0ee1bd-e4eb-45e9-a8af-32f0d0f6fe1b` | `a063bcd9e00f6d3d` | 788 notes; 15.075 MiB serialized state | 20 |
| `20260822_181747` | Base + original evolution; typed relations, temporal state, and provenance off | `bfcee78b-b343-4bc1-989b-b44769298572` | `40ae7c7fef51367b` | 788 notes; 9.225 MiB serialized state | 5 |

The root `20260822_115258` result itself is a completed build-plus-query artifact with `32/49` correct and average score `0.669`. Its 20 children are query-only runs over the same frozen snapshot: ten pooled `2026-08-22–24 typed - hybrid/PPR` runs, five `2026-08-23 typed - hybrid/PPR + chain` runs, and five `2026-08-24 typed - fixed BFS` runs.

## 2. Policy groups and repeated results

Values are mean ± population standard deviation across the completed child runs listed for each group. Accuracy is binary correctness; average score is the metric-aware score stored in each result JSON. The per-type column reports aggregate correct outcomes over all child executions in each group, using `MQ / EEM / TLA / SUA / IG / MCD` order. The pooled `2026-08-22–24 typed - hybrid/PPR` group has ten children; all other groups have five.

| Group | Root snapshot | Query policy | Child count | Accuracy | Average score | Per-type correct |
|---|---|---|---:|---:|---:|---|
| 2026-08-22–24 typed - hybrid/PPR | `20260822_115258` | Hybrid retrieval + `typed_ppr`; chain off | 10 | **`0.665 ± 0.023`** | **`0.682 ± 0.019`** | 79 / 100 / 84 / 21 / 39 / 3 |
| 2026-08-23 typed - hybrid/PPR + chain | `20260822_115258` | Hybrid retrieval + `typed_ppr` + chain selection | 5 | `0.641 ± 0.044` | `0.648 ± 0.041` | 41 / 46 / 41 / 9 / 17 / 3 |
| 2026-08-24 typed - fixed BFS | `20260822_115258` | Typed retrieval only; hybrid retrieval off; `fixed_bfs`; chain off | 5 | `0.531 ± 0.037` | `0.558 ± 0.032` | 30 / 40 / 36 / 9 / 13 / 2 |
| 2026-08-23 original evolution - fixed BFS | `20260822_181747` | Evolution-only build + `fixed_bfs` query | 5 | `0.547 ± 0.008` | `0.579 ± 0.011` | 30 / 44 / 38 / 11 / 11 / 0 |

The pooled `2026-08-22–24 typed - hybrid/PPR` group combines ten runs with the same snapshot, retrieval settings, and effective Gemini model. The `2026-08-23 typed - hybrid/PPR + chain` group uses that same policy with chain selection enabled. The `2026-08-24 typed - fixed BFS` group disables hybrid retrieval and chain selection while retaining the effective `fixed_bfs` graph mode. The `2026-08-23 original evolution - fixed BFS` group has a different source build and should be treated as a reference, not a matched ablation.

### 2.1 Exact child-run index

Every child below completed 49/49 declared queries. The score pairs are `correct / 49` and average score.

| Group | Child runs and outcomes |
|---|---|
| 2026-08-22–24 typed - hybrid/PPR | `20260822_163906` 31/49, 0.670; `20260822_165202` 33/49, 0.681; `20260822_170415` 32/49, 0.672; `20260822_171549` 33/49, 0.694; `20260822_173813` 32/49, 0.673; `20260824_003237` 33/49, 0.685; `20260824_004505` 35/49, 0.724; `20260824_005717` 33/49, 0.680; `20260824_010848` 31/49, 0.645; `20260824_012239` 33/49, 0.691 |
| 2026-08-23 typed - hybrid/PPR + chain | `113111` 29/49, 0.602; `114657` 30/49, 0.606; `120523` 34/49, 0.697; `121751` 34/49, 0.691; `123207` 30/49, 0.644 |
| 2026-08-24 typed - fixed BFS | `022727` 23/49, 0.505; `023645` 27/49, 0.575; `024725` 25/49, 0.536; `025800` 27/49, 0.580; `030843` 28/49, 0.592 |
| 2026-08-23 original evolution - fixed BFS | `050401` 27/49, 0.580; `052418` 27/49, 0.580; `053401` 26/49, 0.558; `054447` 27/49, 0.587; `055449` 27/49, 0.587 |

Full paths are under `outputs/amem_test_gemini-2.5-flash/<root>/query_runs/<child>/`. The shortened IDs in this table are unambiguous within their parent root and date-labeled group.

### 2.2 Full root build and query telemetry

The root table reports each root artifact's build phase and its one root query phase. Token counts are provider-reported input/output/total tokens. Calls are `successful / attempted / failed / retried`; latency is the sum of successful-call latencies. The root query columns are not the sum of child repeats.

| Root | Build features | Root correct/49 | Root accuracy | Root score | Build input | Build output | Build total | Build calls | Build latency (s) | Build wall (s) | Root query input | Root query output | Root query total | Root query calls | Root query latency (s) | Memory (MiB) |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---|---:|---:|
| `20260822_115258` | `base_memory+typed_relations` | 32/49 | 0.653 | 0.669 | 3,514,359 | 88,666 | 3,603,025 | 1,575/2,054/479/479 | 8,336.083 | 11,075.421 | 1,037,257 | 9,727 | 1,046,984 | 137/160/23/23 | 132.122 | 15.075 |
| `20260822_181747` | `base_memory+original_evolution` | 26/49 | 0.531 | 0.565 | 7,823,162 | 131,570 | 7,954,732 | 2,718/3,416/698/698 | 15,564.798 | 22,447.119 | 860,087 | 9,576 | 869,663 | 137/154/17/17 | 101.291 | 9.225 |

### 2.2.1 Build telemetry by feature

The aggregate build columns above can be decomposed by feature as follows. Feature `wall` is the recorded feature wall time; it is not necessarily additive with root wall time because operations can overlap or include unscoped work.

| Root | Build feature | Input tokens | Output tokens | Total tokens | Successful calls | Attempted calls | Failed attempts | Retries | Latency (s) | Feature wall (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `20260822_115258` | `base` | 544,708 | 58,737 | 603,445 | 788 | 886 | 98 | 98 | 3,578.993 | 4,006.361 |
| `20260822_115258` | `embedding` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.000 | 16.284 |
| `20260822_115258` | `typed_relations` | 2,969,651 | 29,929 | 2,999,580 | 787 | 1,168 | 381 | 381 | 4,757.090 | 7,045.154 |
| `20260822_181747` | `base` | 544,708 | 58,437 | 603,145 | 788 | 834 | 46 | 46 | 3,554.760 | 4,320.969 |
| `20260822_181747` | `embedding` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.000 | 93.224 |
| `20260822_181747` | `original_evolution` | 7,278,454 | 73,133 | 7,351,587 | 1,930 | 2,582 | 652 | 652 | 12,010.038 | 18,024.111 |

### 2.3 Full child query-run result table

Each child row is one completed 49-query evaluation. Type cells use `average score / binary accuracy (correct / total)`, in `MQ, EEM, TLA, SUA, IG, MCD` order. Query telemetry is from that child's `llm_usage.total`; it excludes the reused source-build telemetry.

| Group | Child | Endpoint | Correct/49 | Accuracy | Overall score | MQ | EEM | TLA | SUA | IG | MCD | Query input | Query output | Query total | Query calls | Query latency (s) |
|---|---|---|---:|---:|---:|---|---|---|---|---|---|---:|---:|---:|---|---:|
| 2026-08-22–24 typed - hybrid/PPR | `20260822_163906` | standard | 31/49 | 0.633 | 0.670 | 0.800/0.700 (7/10) | 1.000/1.000 (10/10) | 0.800/0.800 (8/10) | 0.400/0.400 (2/5) | 0.400/0.400 (4/10) | 0.203/0.000 (0/4) | 1,052,042 | 10,183 | 1,062,225 | 137/155/18/18 | 118.352 |
| 2026-08-22–24 typed - hybrid/PPR | `20260822_165202` | standard | 33/49 | 0.673 | 0.681 | 0.900/0.900 (9/10) | 1.000/1.000 (10/10) | 0.900/0.900 (9/10) | 0.400/0.400 (2/5) | 0.300/0.300 (3/10) | 0.093/0.000 (0/4) | 1,039,778 | 9,862 | 1,049,640 | 137/154/17/17 | 84.879 |
| 2026-08-22–24 typed - hybrid/PPR | `20260822_170415` | standard | 32/49 | 0.653 | 0.672 | 0.850/0.800 (8/10) | 1.000/1.000 (10/10) | 0.900/0.900 (9/10) | 0.400/0.400 (2/5) | 0.300/0.300 (3/10) | 0.112/0.000 (0/4) | 1,048,249 | 10,308 | 1,058,557 | 137/153/16/16 | 119.397 |
| 2026-08-22–24 typed - hybrid/PPR | `20260822_171549` | standard | 33/49 | 0.673 | 0.694 | 0.850/0.800 (8/10) | 1.000/1.000 (10/10) | 0.800/0.800 (8/10) | 0.400/0.400 (2/5) | 0.500/0.500 (5/10) | 0.129/0.000 (0/4) | 1,043,627 | 10,232 | 1,053,859 | 137/151/14/14 | 143.239 |
| 2026-08-22–24 typed - hybrid/PPR | `20260822_173813` | standard | 32/49 | 0.653 | 0.673 | 0.800/0.700 (7/10) | 1.000/1.000 (10/10) | 0.800/0.800 (8/10) | 0.400/0.400 (2/5) | 0.400/0.400 (4/10) | 0.250/0.250 (1/4) | 1,055,513 | 9,974 | 1,065,487 | 137/158/21/21 | 130.102 |
| 2026-08-23 typed - hybrid/PPR + chain | `20260823_113111` | standard | 29/49 | 0.592 | 0.602 | 0.850/0.800 (8/10) | 0.900/0.900 (9/10) | 0.700/0.700 (7/10) | 0.400/0.400 (2/5) | 0.300/0.300 (3/10) | 0.000/0.000 (0/4) | 1,045,038 | 11,003 | 1,056,041 | 137/155/18/18 | 211.401 |
| 2026-08-23 typed - hybrid/PPR + chain | `20260823_114657` | standard | 30/49 | 0.612 | 0.606 | 0.900/0.900 (9/10) | 0.900/0.900 (9/10) | 0.700/0.700 (7/10) | 0.400/0.400 (2/5) | 0.200/0.200 (2/10) | 0.169/0.250 (1/4) | 1,033,280 | 9,663 | 1,042,943 | 137/154/17/17 | 140.270 |
| 2026-08-23 typed - hybrid/PPR + chain | `20260823_120523` | standard | 34/49 | 0.694 | 0.697 | 0.850/0.800 (8/10) | 1.000/1.000 (10/10) | 0.900/0.900 (9/10) | 0.200/0.200 (1/5) | 0.500/0.500 (5/10) | 0.169/0.250 (1/4) | 1,031,696 | 9,310 | 1,041,006 | 137/146/9/9 | 185.783 |
| 2026-08-23 typed - hybrid/PPR + chain | `20260823_121751` | standard | 34/49 | 0.694 | 0.691 | 0.900/0.900 (9/10) | 0.900/0.900 (9/10) | 0.900/0.900 (9/10) | 0.400/0.400 (2/5) | 0.400/0.400 (4/10) | 0.212/0.250 (1/4) | 1,039,777 | 10,070 | 1,049,847 | 137/159/22/22 | 160.110 |
| 2026-08-23 typed - hybrid/PPR + chain | `20260823_123207` | standard | 30/49 | 0.612 | 0.644 | 0.800/0.700 (7/10) | 0.900/0.900 (9/10) | 0.900/0.900 (9/10) | 0.400/0.400 (2/5) | 0.300/0.300 (3/10) | 0.139/0.000 (0/4) | 1,035,832 | 9,887 | 1,045,719 | 137/154/17/17 | 113.649 |
| 2026-08-22–24 typed - hybrid/PPR | `20260824_003237` | standard | 33/49 | 0.673 | 0.685 | 0.850/0.800 (8/10) | 1.000/1.000 (10/10) | 0.900/0.900 (9/10) | 0.200/0.200 (1/5) | 0.500/0.500 (5/10) | 0.021/0.000 (0/4) | 1,044,725 | 9,442 | 1,054,167 | 137/164/27/27 | 111.270 |
| 2026-08-22–24 typed - hybrid/PPR | `20260824_004505` | standard | 35/49 | 0.714 | 0.724 | 0.850/0.800 (8/10) | 1.000/1.000 (10/10) | 0.800/0.800 (8/10) | 0.600/0.600 (3/5) | 0.500/0.500 (5/10) | 0.249/0.250 (1/4) | 1,047,985 | 10,026 | 1,058,011 | 137/157/20/20 | 99.741 |
| 2026-08-22–24 typed - hybrid/PPR | `20260824_005717` | standard | 33/49 | 0.673 | 0.680 | 0.850/0.800 (8/10) | 1.000/1.000 (10/10) | 0.800/0.800 (8/10) | 0.600/0.600 (3/5) | 0.300/0.300 (3/10) | 0.209/0.250 (1/4) | 1,046,676 | 9,993 | 1,056,669 | 137/142/5/5 | 124.718 |
| 2026-08-22–24 typed - hybrid/PPR | `20260824_010848` | standard | 31/49 | 0.633 | 0.645 | 0.850/0.800 (8/10) | 1.000/1.000 (10/10) | 0.800/0.800 (8/10) | 0.400/0.400 (2/5) | 0.300/0.300 (3/10) | 0.021/0.000 (0/4) | 1,035,793 | 9,216 | 1,045,009 | 137/162/25/25 | 163.394 |
| 2026-08-22–24 typed - hybrid/PPR | `20260824_012239` | standard | 33/49 | 0.673 | 0.691 | 0.850/0.800 (8/10) | 1.000/1.000 (10/10) | 0.900/0.900 (9/10) | 0.400/0.400 (2/5) | 0.400/0.400 (4/10) | 0.093/0.000 (0/4) | 1,046,006 | 9,863 | 1,055,869 | 137/160/23/23 | 123.789 |
| 2026-08-24 typed - fixed BFS | `20260824_022727` | standard | 23/49 | 0.469 | 0.505 | 0.683/0.500 (5/10) | 0.800/0.800 (8/10) | 0.600/0.600 (6/10) | 0.400/0.400 (2/5) | 0.100/0.100 (1/10) | 0.228/0.250 (1/4) | 929,607 | 9,429 | 939,036 | 137/151/14/14 | 106.685 |
| 2026-08-24 typed - fixed BFS | `20260824_023645` | standard | 27/49 | 0.551 | 0.575 | 0.717/0.600 (6/10) | 0.800/0.800 (8/10) | 0.800/0.800 (8/10) | 0.400/0.400 (2/5) | 0.300/0.300 (3/10) | 0.000/0.000 (0/4) | 926,561 | 9,719 | 936,280 | 137/148/11/11 | 124.983 |
| 2026-08-24 typed - fixed BFS | `20260824_024725` | standard | 25/49 | 0.510 | 0.536 | 0.717/0.600 (6/10) | 0.800/0.800 (8/10) | 0.600/0.600 (6/10) | 0.400/0.400 (2/5) | 0.300/0.300 (3/10) | 0.025/0.000 (0/4) | 930,469 | 9,555 | 940,024 | 137/149/12/12 | 105.734 |
| 2026-08-24 typed - fixed BFS | `20260824_025800` | standard | 27/49 | 0.551 | 0.580 | 0.767/0.600 (6/10) | 0.800/0.800 (8/10) | 0.800/0.800 (8/10) | 0.200/0.200 (1/5) | 0.300/0.300 (3/10) | 0.188/0.250 (1/4) | 925,496 | 9,800 | 935,296 | 137/147/10/10 | 134.470 |
| 2026-08-24 typed - fixed BFS | `20260824_030843` | standard | 28/49 | 0.571 | 0.592 | 0.767/0.700 (7/10) | 0.800/0.800 (8/10) | 0.800/0.800 (8/10) | 0.400/0.400 (2/5) | 0.300/0.300 (3/10) | 0.087/0.000 (0/4) | 913,336 | 9,500 | 922,836 | 137/144/7/7 | 113.984 |
| 2026-08-23 original evolution - fixed BFS | `20260823_050401` | standard | 27/49 | 0.551 | 0.580 | 0.733/0.600 (6/10) | 0.800/0.800 (8/10) | 0.600/0.600 (6/10) | 0.600/0.600 (3/5) | 0.400/0.400 (4/10) | 0.025/0.000 (0/4) | 839,294 | 9,727 | 849,021 | 137/147/10/10 | 104.473 |
| 2026-08-23 original evolution - fixed BFS | `20260823_052418` | standard | 27/49 | 0.551 | 0.580 | 0.733/0.600 (6/10) | 0.800/0.800 (8/10) | 0.900/0.900 (9/10) | 0.400/0.400 (2/5) | 0.200/0.200 (2/10) | 0.025/0.000 (0/4) | 841,914 | 9,974 | 851,888 | 137/152/15/15 | 98.006 |
| 2026-08-23 original evolution - fixed BFS | `20260823_053401` | standard | 26/49 | 0.531 | 0.558 | 0.733/0.600 (6/10) | 0.900/0.900 (9/10) | 0.800/0.800 (8/10) | 0.400/0.400 (2/5) | 0.100/0.100 (1/10) | 0.000/0.000 (0/4) | 839,684 | 9,931 | 849,615 | 137/148/11/11 | 96.244 |
| 2026-08-23 original evolution - fixed BFS | `20260823_054447` | standard | 27/49 | 0.551 | 0.587 | 0.767/0.600 (6/10) | 0.900/0.900 (9/10) | 0.800/0.800 (8/10) | 0.400/0.400 (2/5) | 0.200/0.200 (2/10) | 0.025/0.000 (0/4) | 839,063 | 9,987 | 849,050 | 137/146/9/9 | 110.271 |
| 2026-08-23 original evolution - fixed BFS | `20260823_055449` | standard | 27/49 | 0.551 | 0.587 | 0.767/0.600 (6/10) | 1.000/1.000 (10/10) | 0.700/0.700 (7/10) | 0.400/0.400 (2/5) | 0.200/0.200 (2/10) | 0.025/0.000 (0/4) | 829,533 | 9,724 | 839,257 | 137/148/11/11 | 342.934 |

## 3. Exact query policies

### Frozen typed snapshot: hybrid retrieval and `typed_ppr`

The 20 children of `20260822_115258` share the same frozen memory snapshot. The query-policy groups are pooled `2026-08-22–24 typed - hybrid/PPR`, `2026-08-23 typed - hybrid/PPR + chain`, and `2026-08-24 typed - fixed BFS`.

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

Pooled no-chain children:

- `20260822_163906`
- `20260822_165202`
- `20260822_170415`
- `20260822_171549`
- `20260822_173813`
- `20260824_003237`
- `20260824_004505`
- `20260824_005717`
- `20260824_010848`
- `20260824_012239`

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

The pooled no-chain query parameters are hybrid retrieval enabled, `typed_ppr`, typed expansion budget 20, ordinary-link expansion disabled, temporal and provenance retrieval disabled, chain selection disabled, and answer temperature 0. All ten children are standard-model query repeats over the same frozen snapshot.

The `2026-08-24 typed - fixed BFS` children are:

- `20260824_022727`
- `20260824_023645`
- `20260824_024725`
- `20260824_025800`
- `20260824_030843`

Their effective query parameters are typed retrieval enabled, hybrid retrieval disabled, graph ranking mode `fixed_bfs`, ordinary-link expansion disabled, chain selection disabled, temporal and provenance retrieval disabled, and answer temperature 0. The graph mode is reported as the effective artifact value; “typed-only” here means that hybrid retrieval, chain selection, and ordinary links are disabled, not that graph traversal is absent.

### Evolution-only snapshot: fixed BFS

The five children of `20260822_181747` use a separate `base_memory + original_evolution` snapshot and `fixed_bfs` retrieval. They are useful for continuity with the previous graph implementation, but differences from the hybrid/PPR family combine build changes and query-policy changes.

## 4. Per-query-type interpretation

The aggregate type counts indicate where the policies were strong or weak, but they are not supporting-evidence recall measurements.

- **Hybrid + `typed_ppr`, no chain:** the pooled group reached EEM `100/100`, IG `39/100`, and MCD correctness `3/40` with mean composite score `0.138`.
- **Hybrid + `typed_ppr`, chain:** MQ increased descriptively (`41/50` versus `39/50`), while EEM, TLA, SUA, and IG decreased. MCD binary correctness increased to `3/20`, but this did not translate into a better MCD composite score; see [`AMEM_CHAIN_SELECTION_AUDIT_20260823.md`](AMEM_CHAIN_SELECTION_AUDIT_20260823.md).
- **Repeat variation:** with the same frozen snapshot, A-MEM retrieval settings, and effective Gemini model, the ten hybrid/PPR repeats ranged from `0.657` to `0.673` batch means. This should not be attributed to retrieval alone.
- **Fixed BFS:** MCD was `0/20`; this group is not directly comparable to the frozen typed/PPR family because its memory build differs.
- **Typed-only fixed BFS:** hybrid retrieval and chain selection were disabled while the artifact retained `fixed_bfs`; MCD correctness was `2/20` and the mean composite score was `0.106`.

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

Do not compare these `49`-query results directly with the historical `97`-query raw `amem` result. Do not infer build-feature gains from the fixed-BFS group without matched source snapshots. Do not interpret MCD answer scores as medical correctness or clinical suitability.

The next controlled experiment should freeze, for every question:

1. the generated retrieval query;
2. dense/BM25/hybrid candidate IDs and scores;
3. graph scores and eligible edges;
4. the query and answer model endpoints, retry behavior, and token budgets;
5. the exact final prompt; and
6. the answer and judge stages as separate repeats.

Then compare ranked bags, the current 30-note selector, and genuinely sparse directed-chain selectors at matched token budgets. Advance a new selector only if it improves annotated minimal-chain recall and MCD NCR/CRC/CC without unacceptable overall loss.

## Source references

- Recent hybrid/PPR root and children, including the 2026-08-24 query groups: `outputs/amem_test_gemini-2.5-flash/20260822_115258/`
- Recent fixed-BFS root and children: `outputs/amem_test_gemini-2.5-flash/20260822_181747/`
- Experimental adapter: `methods/amem_test_agent.py`
- Typed retrieval and selector: `methods/amem/A-mem/memory_layer_typed.py`
- Evaluation and result serialization: `benchmarks/medmemorybench/evaluator.py`, `src/result.py`
- Historical aggregate analysis: [`AMEM_EXPERIMENT_ANALYSIS.md`](AMEM_EXPERIMENT_ANALYSIS.md)
- Implementation/configuration reference: [`AMEM_GUIDE.md`](AMEM_GUIDE.md), [`AMEM_IMPLEMENTATION.md`](AMEM_IMPLEMENTATION.md)
