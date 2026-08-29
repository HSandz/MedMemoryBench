# A-MEM Implementation Reference

This is the concise, code-oriented guide to the A-MEM integrations in MedMemoryBench. The active adapter files are outside this directory; the vendored implementation is under `methods/amem/A-mem/`.

## Components

| Component | Role |
|---|---|
| `methods/amem_agent.py` | Shared adapter, embedding retriever setup, query answering, and snapshot serialization. |
| `methods/amem_fix_agent.py` | Paper-aligned atomic-note and linked-retrieval flow (`amem_fix`). |
| `methods/amem_test_agent.py` | Experimental typed-relation, temporal-state, provenance, and chain-selection flow (`amem_test`). |
| `methods/amem/A-mem/memory_layer_robust.py` | Active plain-text LLM memory layer used by `amem` and `amem_fix`. |
| `methods/amem/A-mem/memory_layer_typed.py` | Experimental layer used by `amem_test`. |
| `configs/method_config/*amem*.yaml` | Method and persona-specific configurations. |
| `tests/test_amem_*.py` | Adapter, feature, and snapshot regression coverage. |

The older `memory_layer.py` and standalone scripts in `methods/amem/A-mem/` support upstream/LoCoMo experiments; they are not the normal MedMemoryBench entry point.

## Runtime Pipeline

1. `AgentManager` selects `amem`, `amem_fix`, or `amem_test` from the YAML `method_name` and forwards `agent_params`.
2. Each persona/context gets an independent A-MEM system with a SentenceTransformer retriever and an internal metadata/evolution LLM.
3. During memorization, input is split at `amem_chunk_size_tokens` and passed to `add_note()`.
4. A-MEM analyzes each note, extracts metadata, and may evolve neighboring notes. The robust layer can make up to three conditional calls: evolution decision, strengthening, and neighbor update. `amem_evo_threshold` controls retriever consolidation.
5. At query time, the adapter retrieves semantic seeds, formats the notes into the answer prompt, and records retrieval metadata. `amem_fix` optionally generates keyword queries and expands untyped links.
6. `amem_test` can fuse deterministic hybrid candidate ranks, apply a configurable graph-ranking policy, select temporal states, optionally select chain-preserving evidence under the final prompt budget, and attach provenance evidence before final prompt construction. Typed-relation inference occurs during memory construction; query-time ranking and selection are local after optional keyword generation. Vertex batch applies to eligible final-answer generation.

Memory snapshots contain notes, retriever corpus/embeddings, IDs, configuration, and feature audits. Query-stage runs can reuse a completed memory snapshot with `--memory-run`; imports reject incompatible state.

### Session retrieval-quality metrics

When `method_name: amem_test` and `build_config.amem_note_level: session`, each stored note also carries its source session ID independently of optional provenance text/evidence retrieval. The ID is serialized with the note and copied to each retrieved-memory record as `source_session_id`, so retrieval scoring remains available when `amem_provenance` or `amem_provenance_retrieval` is disabled. Turn-level notes and retrieval records are unchanged.

For each query, the evaluator deduplicates retrieved session IDs in retrieval order and gold session IDs from `source_key_points`. It computes:

$$
P=\frac{|R \cap G|}{|R|},\qquad
Recall=\frac{|R \cap G|}{|G|},\qquad
F1=\frac{2P\,Recall}{P+Recall}.
$$

It also reports TP/FP/FN, average precision over the ranked unique session IDs, reciprocal rank of the first relevant session, any-hit, and exact set match. An empty retrieval with valid gold IDs is a scored miss. Missing gold IDs or any retrieved note without a source session ID makes that query unavailable, preventing incomplete legacy metadata from being interpreted as a false retrieval miss.

Per-query metrics and audit IDs are written to `query_answer.json` under `evaluation_details.metric_groups.retrieval_quality`. `result.json` writes aggregate macro averages, micro totals/precision/recall/F1, coverage, MAP, MRR, hit rate, exact-match rate, average predicted/gold session counts, and the same aggregate split by query type under `summary.metric_groups.retrieval_quality`. These diagnostics do not alter answer correctness, answer score, or `by_metric` scoring.

## Snapshot and Append Lifecycle

Normal A-MEM memory builds publish one snapshot per evaluation unit under `<run>/memory/`. The manifest remains `building` until every expected unit has a snapshot, then becomes `complete`. Each snapshot records the source context, unit, build ID, config hash, memory construction time, and an integrity hash; embedding arrays are stored in adjacent `.embeddings.npy` sidecars.

Build telemetry is captured as a usage delta around each injected session and aggregated per unit and run. The tracker uses context-local phase and operation scopes so concurrent work cannot relabel another call. AMEM scopes distinguish note analysis, original evolution, chunking, embedding/indexing, typed candidate search and inference, temporal initialization/transitions, and provenance preprocessing/storage. Reports keep successful calls separate from attempts, failures, and retries. Snapshot publication also measures the compact serialized memory-state JSON and exact embedding-sidecar bytes. Unit metrics include cumulative size and signed growth; the overall footprint sums the latest unit for each context instead of all cumulative snapshots. Unit metrics are embedded in snapshots and summarized in the completed manifest, allowing memory-only, resume, append, and query-only workflows to expose the same source-build measurements. See [A-MEM Build Metrics](AMEM_BUILD_METRICS.md) for the serialized schema and reporting workflow.

An append creates a child run under `<source-run>/query_runs/<child-run>/`. It copies the source's contiguous snapshot prefix into the child, rewrites the snapshots to the child build ID, preserves source provenance, and builds only missing units through the inclusive `(persona, unit)` target. `--stage memory` stops after this publication. The default/all append restores both copied and newly built snapshots and evaluates the complete prefix, writing reports directly in the child run. A child manifest may contain a prefix of the full dataset and is therefore accepted by normal `--stage query` loading.

Append requires independent evaluation mode and an A-MEM method. The target uses the dataset's global `unit_id` plus its `context_id`/persona for validation. Source manifests may be `building` or `complete`; a building source must still contain a contiguous snapshot prefix. `--resume` reuses the child run and its checkpoint, including the small window between manifest publication and checkpoint creation.

## Variants and Features

| Variant | Write path | Query path |
|---|---|---|
| `amem` | Raw token-bounded chunks passed directly to `add_note()`. | Direct semantic retrieval. |
| `amem_fix` | One timestamped atomic note per dialogue turn; structured `memory_items` are preferred. | Keyword query -> semantic seeds -> one-hop untyped link expansion. |
| `amem_test` | Configurable turn-level (default) or session-level notes plus optional typed relation/provenance metadata; original evolution is configurable. | Dense or hybrid seeds, selectable graph ranking, optional temporal expansion, chain-preserving evidence selection, then provenance context. |

Typed relation labels are `SUPPORT`, `REFINE`, `SUPERSEDE`, `CONFLICT`, and `RELATED`. Temporal state is derived from typed transitions and therefore requires `amem_typed_relations: true`. Provenance creates stable source evidence records; `amem_provenance_inject_raw_text` optionally adds selected source turns to the answer prompt.

### Experimental retrieval policies

`amem_hybrid_retrieval: true` replaces dense-only seeds with weighted reciprocal-rank fusion over dense similarity, BM25, deterministic query-to-note token overlap, direct source-timestamp match, stored temporal-state match, and graph proximity. `amem_hybrid_*_weight: 0` disables an individual channel. `amem_hybrid_candidate_count` bounds each channel's candidate pool; `retrieve_num` remains the final seed count. The default `amem_hybrid_retrieval: false` preserves dense seeds.

`amem_regex_intent_conditioning` is a query-only ablation flag and defaults to `true`. With the default, `amem_test` applies its existing regex semantic heuristics for temporal intents (`current`, `historical`, `change`) and graph relation intents (`causal`, `conflict`, `detail`, `change`). With `false`, those question-derived multipliers and state preferences are disabled: graph ranking keeps configured relation-type weights and confidence, while explicit date/time parsing remains active for timestamp, validity, and temporal compatibility matching. The flag does not change memory construction or keyword rewriting.

`amem_graph_ranking_mode` selects the graph policy:

| Value | Behavior |
|---|---|
| `none` | Keep seeds only; omit typed relation expansion and typed relation prompt context. |
| `fixed_bfs` | Preserve the existing bounded typed BFS (default). |
| `untyped_ppr` | Personalized PageRank over enabled ordinary and typed edges with unit edge weights; omit typed relation labels from the prompt. |
| `typed_ppr` | Personalized PageRank weighted by relation type, confidence, direction, temporal compatibility, and deterministic query intent. Requires `amem_typed_retrieval: true`. |

PageRank uses `amem_graph_alpha`, `amem_graph_iterations`, and `amem_graph_tolerance`. Per-relation weights use `amem_graph_{supersede,conflict,refine,support,related}_weight`. `amem_expand_links`, `amem_typed_retrieval`, `amem_expand_related`, and `amem_relation_min_confidence` still gate which edges are eligible. Hybrid channel rankings, RRF contributions, seed scores, graph scores, convergence, and selected IDs are written to the retrieval audit.

### Chain-preserving evidence selection

`amem_chain_selection: true` enables a deterministic query-time stage after graph and temporal retrieval but before final relation/provenance formatting. It fuses retrieval, hybrid, graph, and dense rankings with RRF before limiting the pool to `amem_chain_candidate_count`; no source ranking or upstream ID is mandatory. The selector builds an undirected evidence graph from the ordinary and typed edges allowed by the existing retrieval switches. It then identifies lexical IDF facets, direct timestamp facets, available temporal-state facets, and general operations such as change, comparison, conflict, refinement, support/causal connection, and enumeration from the raw question.

Selection maximizes the configured utility

$$
U(S)=\omega_r Rel(S)+\omega_c Cov(S)+\omega_g Conn(S)+\omega_p Path(S)+\omega_t Temp(S)-\omega_d Red(S).
$$

`Rel` combines max-calibrated fusion, hybrid, and graph scores; `Cov` rewards distinct question facets; `Conn` is an accumulated maximum-spanning-forest reward with a soft excess-component penalty; `Path` rewards complete paths of at most `amem_chain_max_hops`; `Temp` rewards direct date matches and, only when temporal retrieval is active, stored state/transition compatibility; and `Red` is bounded average lexical, embedding, and source-evidence duplication. Greedy actions are either one memory or one complete path, ranked by marginal utility per added memory. `amem_chain_max_hops` is clamped to 3; path traversal keeps the eight strongest eligible neighbors per memory and at most 100 ranked path actions to bound dense-graph work. `amem_chain_max_groups` is the preferred number of disconnected evidence components, not a hard rejection limit. All ties follow stable fused rank.

The selector aims for `amem_chain_evidence_count` final memories while respecting `amem_chain_candidate_count`, complete-path atomicity, and the exact prompt budget. It does not preserve upstream IDs by rule and does not stop merely because a bounded objective term has saturated. If a target-sized set is feasible, each accepted action must leave enough exact formatted budget for a deterministic cheapest-singleton completion, preventing one expensive high-ranked note from consuming the budget needed to reach the target. The token callback formats every proposal exactly as the answer prompt will see it, including eligible relation labels, temporal state, provenance headers/raw evidence, and note metadata. The budget is the remaining `amem_max_context_tokens` after the system message, question, output allowance, and the existing 200-token reserve. Enabled selection is not followed by context truncation; selection stops below the target only when the target is not feasible and no remaining action fits.

The selector uses only the raw question and stored memory state. It receives any temporal target already resolved by the normal temporal-retrieval stage; an unresolved relative date without concrete bounds does not match every timestamped note. It does not receive benchmark query types, reference answers, gold evidence, or scoring metadata. Its retrieval audit is stored under `extra.chain_selection` and includes candidate/selected IDs, paths, facets, per-candidate scores, evidence-graph edges, utility components, exact selected tokens, greedy steps, and rejection reasons.

## Configuration

Common fields:

```yaml
model:
  provider: gemini
  name: gemini-2.5-flash
  temperature: 0.0
  max_completion_tokens: 2000

build_config:
  amem_backend: vertex
  amem_model: gemini-2.5-flash
  amem_embedding_model: all-MiniLM-L6-v2
  amem_evo_threshold: 100
  amem_max_tokens: 1000
  amem_chunk_size_tokens: 10240
  amem_note_level: turn
  amem_build_max_context_tokens: 200000

retrieval_config:
  retrieve_num: 10
  amem_max_context_tokens: 200000
  amem_hybrid_retrieval: false
  amem_regex_intent_conditioning: true
  amem_graph_ranking_mode: fixed_bfs
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

Use `amem_fix_*` for the paper-aligned baseline and `amem_test_*` for experiments. `amem_note_level` accepts only `turn` or `session` and defaults to `turn` when omitted. Session mode combines the same normalized speaker turns before the normal `add_note()` pipeline; it does not change note analysis, evolution, typed/temporal metadata, indexing, or retrieval. Session provenance describes the full session rather than an individual turn. Oversized sessions can still create multiple chunk parts. Treat every experimental feature flag as authoritative: similarly named `amem_test2_*` files may intentionally represent different combinations. Build-time temporal transitions use `amem_temporal_transition_min_confidence`; query-time graph filtering uses `amem_relation_min_confidence`. Hybrid weights, graph mode/ranking parameters, chain-selection controls, expansion counts, temporal ordering, provenance injection, keyword use, and retrieval context budgets can change without rebuilding. Most shipped configurations disable chain selection; individual experiment configs may enable it explicitly.

Do not change note level, build flags, the internal AMEM model, embedding/chunk settings, build prompt budget, or temporal transition threshold when loading a snapshot. New snapshots record the note level in their build state; legacy `amem_test` snapshots without it are treated as `turn`. Keep stable memory IDs, audit fields, and source evidence intact because resume, rejudge, and analysis depend on them.

## Verification

The current completed-run comparison is maintained in [A-MEM Completed Run Results](AMEM_COMPLETED_RUNS_RESULTS_20260823.md). Use it together with [A-MEM Chain-Selection Audit](AMEM_CHAIN_SELECTION_AUDIT_20260823.md) when interpreting `typed_ppr`, hybrid retrieval, fixed BFS, or chain-selection results; the reported runs cover 49 queries rather than the full 97-query dataset.

```bash
python -m pytest tests/test_amem_fix_agent.py tests/test_amem_test_agent.py tests/test_amem_staged_memory.py tests/test_build_feature_metrics.py
python main.py -m amem_fix_gemini -d medmemorybench --dry-run
python main.py -m persona_1/amem_fix_gemini -d medmemorybench --stage memory
python main.py --stage query --memory-run YYYYMMDD_HHMMSS
```

Prefer mocked provider tests. Update the A-MEM regression tests when changing note normalization, retrieval expansion, feature validation, serialization, or batch preparation.
