# A-MEM Chain-Selection Audit and Next Experiments

Date: 2026-08-23

## Scope and conclusion

This audit examines the query-only experiment rooted at `outputs/amem_test_gemini-2.5-flash/20260822_115258`: the active `amem`, `amem_fix`, and `amem_test` paths; the typed-memory selector; the MCD judge; and all ten saved child runs and retrieval audits.

For the consolidated index of this family and the related 2026-08-21 and 2026-08-23 run groups, see [`AMEM_COMPLETED_RUNS_RESULTS_20260823.md`](AMEM_COMPLETED_RUNS_RESULTS_20260823.md).

**Conclusion:** `amem_chain_selection` is implemented, invoked, audited, and compared on one fixed snapshot. It does **not** demonstrate an improvement in multi-hop clinical deduction (MCD) or overall quality. Chain-enabled runs increase observed binary MCD correctness from `1/20` to `3/20`, but lower mean MCD composite score from `0.158` to `0.138`. Overall mean accuracy and score also decline slightly. Query-generation, answer-generation, and judging remain stochastic, so these results do not prove harm—but they provide no evidence of benefit.

The main limitation is not an execution failure. The current policy is principally a **30-note relevance reranker with some connected pairs**, not a sparse, question-specific, directed causal-chain selector. It preserves all ten semantic seeds, replaces only `1.68/30` of the top-30 fused candidates on average, and leaves a mean of `6.30` disconnected components despite its preferred maximum of three.

This is benchmark-memory research only. No result here establishes medical correctness, safety, diagnosis quality, treatment quality, or clinical suitability.

## 1. Experimental identity and comparability

### Shared frozen memory

All ten child runs cite the same parent run and snapshot identity:

| Field | Value |
|---|---|
| Source run | `20260822_115258` |
| Child build ID | `ef0ee1bd-e4eb-45e9-a8af-32f0d0f6fe1b` |
| Build config hash | `a063bcd9e00f6d3d` |
| Build | Atomic notes + typed relations; evolution, temporal state, and provenance off |
| Snapshot | 50 Persona 1 sessions, 788 notes |
| Evaluation | 49/49 configured Persona 1 queries: 10 EEM, 10 TLA, 5 SUA, 10 MQ, 10 IG, 4 MCD |

This is a query-policy experiment, not a memory-construction comparison.

### Compared policy groups

No-chain group (five executions):

- `20260822_163906`
- `20260822_165202`
- `20260822_170415`
- `20260822_171549`
- `20260822_173813`

Chain-enabled group (five executions):

- `20260823_113111`
- `20260823_114657`
- `20260823_120523`
- `20260823_121751`
- `20260823_123207`

Saved configurations agree on the upstream query policy:

```text
hybrid retrieval = true
graph ranking = typed_ppr
typed expansion budget = 20
retrieve_num = 10
ordinary-link expansion = false
temporal/provenance retrieval = false
answer model temperature = 0
```

The second group adds this chain configuration only:

```text
amem_chain_selection = true
candidate_count = 50
evidence_count = 30
max_hops = 2
max_groups = 3
weights = relevance 1.0, coverage 1.0, connectivity 0.35,
          path 0.75, temporal 0.5, redundancy 0.25
```

The no-chain artifacts omit the chain keys instead of storing explicit `false` values. `AMemTestAgent` defaults selection to `False`, so they are valid controls. Future artifacts should still serialize defaults explicitly.

### Remaining comparability limit

The memory snapshot is frozen, but each repeat regenerates its keyword rewrite via `AMemFixAgent._generate_retrieval_query()`, then reruns hybrid retrieval, answering, and LLM judging. This estimates full-pipeline variation, not a selector effect under identical candidate evidence.

| Policy | Mean seed-set Jaccard | Mean final-set Jaccard | Questions with more than one exact answer |
|---|---:|---:|---:|
| No chain | 0.786 | 0.816 | 32/49 |
| Chain enabled | 0.671 | 0.728 | 38/49 |

The lower stability in the enabled group is descriptive, not causal proof. The 50-candidate RRF pool and greedy selection can amplify upstream keyword/seed changes.

## 2. Implementation review

The three active adapters are materially distinct:

| Adapter | Memory write path | Query path |
|---|---|---|
| `methods/amem_agent.py` | Raw token-bounded chunks | Direct dense retrieval |
| `methods/amem_fix_agent.py` | Timestamped atomic dialogue turns | Keyword rewrite, dense seeds, optional ordinary-link expansion |
| `methods/amem_test_agent.py` | Atomic notes plus experimental structures | Hybrid seeds, graph policy, optional temporal/provenance context, then chain selection |

For this run, `AMemTestAgent.prepare_batch_query()` executes:

```text
raw question
  -> LLM keyword rewrite
  -> hybrid candidate retrieval / typed PageRank seeds
  -> up to 20 typed PageRank additions
  -> chain candidate union and RRF
  -> greedy chain-preserving evidence selection
  -> relation-aware 30-note prompt
  -> batch final answer
  -> independent MCD judge
```

`TypedRelationMemorySystem.select_chain_preserving_evidence()` in `methods/amem/A-mem/memory_layer_typed.py`:

1. RRF-fuses existing retrieval, hybrid, graph, and dense rankings.
2. Limits the fused pool to 50 notes.
3. Builds an **undirected** evidence graph from permitted typed/ordinary links.
4. Derives lexical facets plus limited date/state/relation-intent facets from the raw question.
5. Greedily accepts one memory or a complete path of at most two hops, using exact formatted-prompt token cost.
6. Stops at 30 notes and persists candidates, paths, utility terms, graph edges, selected tokens, and rejection reasons.

Implementation strengths:

- Exact relation-aware prompt formatting is used in the token callback.
- Complete paths are accepted atomically; path endpoints are not split by a budget boundary.
- The audit is comprehensive under `extra.chain_selection`.
- Tests cover token budgets, path atomicity, candidate fusion, target reservation, temporal gating, and pipeline wiring in `tests/test_amem_test_agent.py`.

Therefore, the current outcome is about the objective and experiment design—not an unimplemented or bypassed feature.

## 3. Observed outcomes

Values are mean ± sample SD across five full query executions. MCD score is the judge's composite score, not binary correctness.

| Query type | No chain accuracy | Chain accuracy | No chain score | Chain score |
|---|---:|---:|---:|---:|
| Overall | **0.657 ± 0.017** | 0.641 ± 0.049 | **0.678 ± 0.010** | 0.648 ± 0.045 |
| Entity exact match | **1.000 ± 0.000** | 0.920 ± 0.045 | **1.000 ± 0.000** | 0.920 ± 0.045 |
| Temporal localization | **0.840 ± 0.055** | 0.820 ± 0.110 | **0.840 ± 0.055** | 0.820 ± 0.110 |
| State update | **0.400 ± 0.000** | 0.360 ± 0.089 | **0.400 ± 0.000** | 0.360 ± 0.089 |
| Multiple choice | 0.780 ± 0.084 | **0.820 ± 0.084** | 0.840 ± 0.042 | **0.860 ± 0.042** |
| Inference generation | **0.380 ± 0.084** | 0.340 ± 0.114 | **0.380 ± 0.084** | 0.340 ± 0.114 |
| Multi-hop clinical deduction | 0.050 ± 0.112 | **0.150 ± 0.137** | **0.158 ± 0.066** | 0.138 ± 0.081 |

The selected retrieval count is exactly 30 for both policies. Chain selection neither reduces prompt size nor compares a sparse chain against a same-budget ranked bag; it changes the composition of a broad 30-note context.

### Overall descriptive contrast

Question-level mean deltas, chain minus no-chain:

| Endpoint | Delta | Descriptive 95% question-resampling interval | Question wins / ties / losses |
|---|---:|---:|---:|
| Binary accuracy | -0.016 | [-0.086, +0.049] | 8 / 35 / 6 |
| Average score | -0.030 | [-0.096, +0.030] | 9 / 32 / 8 |

Both intervals include zero. They condition on this snapshot and these executions, so they are not construction-level causal intervals. The defensible conclusion is **no demonstrated overall effect**.

### MCD question analysis

| MCD query | No chain correct / 5 | Chain correct / 5 | No-chain mean score | Chain mean score | Finding |
|---|---:|---:|---:|---:|---|
| `session_20_mcd_1` | 1 | 3 | 0.393 | 0.405 | Modest descriptive improvement, but unstable. |
| `session_20_mcd_2` | 0 | 0 | 0.000 | 0.006 | No meaningful change. |
| `session_40_mcd_1` | 0 | 0 | 0.045 | 0.000 | Worse descriptively. |
| `session_40_mcd_2` | 0 | 0 | 0.192 | 0.140 | Worse descriptively. |

MCD node-level judge fields agree with the composite outcome:

| Aggregate over 20 MCD executions | No chain | Chain enabled |
|---|---:|---:|
| Mean NCR | 0.200 | 0.175 |
| Mean CRC | 0.225 | 0.188 |
| Mean CC | 0.250 | 0.188 |
| Required-node mention rate | 0.450 | 0.500 |
| Required-node causal-link rate | 0.225 | 0.213 |
| Answers judged patient-specific | 0.400 | 0.200 |

The selector slightly increases mentioning required content but does not increase correctly linked causal content. The reduction in patient-specific answers matters because `LLMJudgeMCDMetric` explicitly penalizes generic answers.

Persistent failures are substantive:

- `session_20_mcd_2` never becomes correct. The expected chain requires SGLT2-related insulin/glucagon physiology, missed prandial insulin, a ketone-prone low-insulin state, and symptoms despite non-high glucose. Outputs remain generic.
- `session_40_mcd_1` never becomes correct. The expected chain includes late-night high-GI food plus milk coffee, sympathetic/HPA effects, high morning glucose, and visual/cognitive symptoms. The inspected enabled-run judge found no specific values or correct node-level links.
- `session_40_mcd_2` never becomes correct. The expected chain needs long-term SGLT2 use, declining endogenous insulin, missed prandial insulin, ketone risk, and atypical symptoms with non-high glucose. Outputs did not bind these patient facts together.

## 4. Why current selection misses the MCD mechanism

### It is a mild 30-note reranker

Across all 245 enabled query records, the selector:

- keeps all 10 semantic seeds in every query;
- keeps a mean of 28.32 top-30 fused candidates;
- adds only 1.68 notes from ranks 31–50;
- has mean selected rank 15.89; and
- reaches the 30-note target in all 245 queries.

Relevance and lexical-coverage terms remain rewarded until a large target is filled. This cannot strongly choose one minimal chain over a large evidence bag.

### Connectivity is only a soft preference

`amem_chain_max_groups=3` is not a constraint. Selected evidence has a mean of 6.30 disconnected components, range 2–14. Mean component counts for the four MCD questions are 8.0, 5.8, 7.4, and 8.6.

`selected_paths` records every valid short path wholly contained in the final 30-note set. It is not a single chosen explanatory chain. Path presence in the audit must not be interpreted as a coherent causal chain delivered to the answer model.

### Intent detection misses three MCD questions

`detect_graph_query_intents()` recognizes causal intent only through narrow terms such as `why`, `cause`, `because`, `led to`, `resulted in`, and `reason`. Of the four MCD questions, only `session_20_mcd_2` is marked causal. The other three use wording such as “could this be related” or ask whether one event could be an effect of another.

Consequently, three MCD cases optimize mostly lexical facets rather than a source-event to outcome mechanism. Generic terms such as `recent`, `could`, `related`, and `symptoms` do not identify the required medical sequence.

### The graph does not represent claim-level causality

Typed edges are sparse insertion-time LLM hypotheses from at most five dense candidate neighbors. `SUPPORT`, `REFINE`, `SUPERSEDE`, `CONFLICT`, and `RELATED` are broad note relationships, not normalized claims such as “SGLT2 use increases glucagon.” The selector also converts these edges to an undirected packing graph, which loses explanatory direction. A selected connected pair is not necessarily an adequate causal explanation.

### Answer generation has no evidence-to-claim plan

The final prompt contains notes and relations but has no required intermediate plan:

```text
claim -> source evidence IDs -> intermediate links -> conclusion
```

The MCD judge rewards patient-specific node facts and causal links. A broad, unstructured prompt permits plausible generic medical prose without showing that each causal statement is grounded in a selected note.

## 5. Next experiments

Run these in sequence; all initial work is query-only and should reuse the existing snapshot.

### Experiment 1: stage-separated replay

1. Generate and save exactly one keyword query per question.
2. Save dense scores, hybrid rankings, PageRank scores, candidate IDs, graph edges, exact prompt text/hash, and prompt tokens.
3. Freeze the same 50-candidate pool for all policies.
4. Compare policies on identical candidates and exact context-token budgets.
5. Repeat answer generation from identical prompts, then rejudge saved answers separately.

Report candidate and final evidence Jaccard, answer flip rate conditional on identical context, judge flip rate conditional on identical answer, MCD NCR/CRC/CC, tokens, and elapsed time. A selection claim must first survive this test.

### Experiment 2: test genuinely sparse chain packing

Compare at matched exact prompt budgets such as 4k, 8k, and 12k tokens:

| Policy | Evidence structure |
|---|---|
| Ranked bag | Current top-30 no-chain baseline |
| Small ranked bag | Top 6–10 notes |
| Current selector | Current 30-note policy |
| Sparse connected selector | 4–8 notes, one or two paths, hard component limit |
| Chain plus fillers | Required path then nonredundant rank-based support notes |

Measure connectivity, complete-path presence, endpoint coverage, evidence displacement, and MCD quality. The current 200k maximum does not create meaningful selection pressure.

### Experiment 3: causal-question routing

Use a deterministic structured intent schema that recognizes:

```text
could this be related to ...
could ... be an effect of ...
does X explain Y
what mechanism connects X and Y
why / cause / result / led to
```

Extract source event, outcome/symptom, dates, medication names, values, and requested relation. Use this only to choose anchors and retrieval policy. Do not pass benchmark `query_type`, reference answers, reasoning-chain metadata, or node IDs to retrieval. Test routing accuracy independently.

### Experiment 4: directed anchor-constrained paths

For causal questions, retrieve source-event and outcome anchors separately. Search directed paths that preserve edge type, confidence, source time, and event order. Score endpoint relevance, edge compatibility, intermediate claim coverage, and temporal consistency. Select one primary path plus limited support, and record discarded alternatives and missing endpoints.

Start with high-confidence `SUPPORT` edges and explicitly compatible `REFINE`/`SUPERSEDE` edges. Compare against the current undirected connected-set policy.

### Experiment 5: evidence-to-claim answer planning

Before prose generation, require a plan assembled only from selected evidence:

```text
question claim
-> patient-specific source event(s), with M-ID/date/value
-> intermediate mechanism/transition(s), with M-ID
-> outcome evidence, with M-ID/date/value
-> unsupported-link or uncertainty flag
-> concise answer
```

Require every causal assertion to cite selected M-IDs. Save the plan and citations. This separates retrieval failures from failures to express retrieved evidence and directly targets generic-answer penalties.

### Experiment 6: supporting-evidence annotation

Start with the four MCD questions, then annotate all 49 questions with acceptable answers, source turns, minimal evidence sets, causal/temporal links, distractors, and whether external knowledge is necessary. Use independent annotation plus adjudication.

Then measure candidate recall@50, endpoint recall, minimal-chain recall, selected-chain precision, unsupported-claim rate, and answer quality conditional on sufficient evidence. Final-score changes alone cannot validate chain selection.

### Experiment 7: claim/state representation ablation

Only after controlling retrieval, compare the note graph with an append-only claim layer:

```text
(subject, attribute/relation, value, event time, valid interval,
 source note/evidence ID, confidence)
```

This may support explicit joins from source event to mechanism to outcome while retaining immutable note provenance. It is a hypothesis, not an improvement supported by the current results.

## 6. Prioritized changes

### P0: experimental correctness and observability

1. Add replay support for saved keyword queries, candidate IDs, and optional exact prompt text/hash.
2. Persist every resolved retrieval default in child `run_config.json`.
3. Store one normalized retrieval record per query with selected order, prompt tokens, path endpoints, component count, answer hash, and judge hash.
4. Add an analysis script for run matrix, policy deltas, evidence Jaccard, components, endpoint coverage, and MCD NCR/CRC/CC.

### P1: make selection chain-specific

1. Expand causal/event-effect intent recognition and test all four current MCD phrasings.
2. Add a causal mode with a hard maximum component count, with a documented ranked-bag fallback when no valid path exists.
3. Separate causal-path budget from filler-note budget; do not always fill 30 notes.
4. Preserve edge direction and record source, target, and intermediate-facet coverage.
5. De-emphasize generic lexical facets in causal mode; prioritize entities, medications, values, symptoms, dates, and extracted event roles.

### P2: ground the final answer

1. Add M-ID evidence-to-claim planning and a support validator.
2. Audit which selected notes are actually cited.
3. Permit an explicit unsupported-link response and measure abstention separately from benchmark correctness.

## Decision rule

Keep `amem_chain_selection` as an experimental baseline and audit mechanism, but do not describe it as improving MCD or overall accuracy.

Advance a new selector only if, on frozen candidates and matched prompt budgets, it improves annotated minimal-chain recall and MCD NCR/CRC/CC without unacceptable overall loss. Then evaluate all 97 Persona 1 queries and at least three independent memory builds before making a build-level claim.

## Source artifacts

- Root and children: `outputs/amem_test_gemini-2.5-flash/20260822_115258`
- Experimental adapter: `methods/amem_test_agent.py`
- Selector: `methods/amem/A-mem/memory_layer_typed.py`
- Baseline adapter: `methods/amem_fix_agent.py`
- Raw adapter: `methods/amem_agent.py`
- MCD metric: `metrics/llm_judge.py`
- Selector tests: `tests/test_amem_test_agent.py`
- Broader audit: `docs/AMEM_INDEPENDENT_BUILD_AUDIT_AND_RESEARCH_ROADMAP_20260822.md`
