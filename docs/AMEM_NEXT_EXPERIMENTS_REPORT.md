# AMEM Implementation Review and Next Experiments

Date: 2026-08-18

Scope: the repository adapters named `amem`, `amem_fix`, and `amem_test`; their vendored A-MEM layers; MedMemoryBench memory/query artifacts; feature-flag runs; and the active plus standalone LoCoMo integrations.

## Executive conclusion

The implementation is a useful experimental platform, but the current evidence does not establish that typed relations, temporal state, or provenance improve answer accuracy. The strongest conclusions are:

1. `amem` is a coarse, low-call-count baseline: it stores large token-bounded chunks and performs direct semantic retrieval. Its historical complete run scored `43/97`, but it used an older configuration and is not a clean comparison to the newer variants.
2. `amem_fix` is the most faithful practical baseline for the current benchmark integration. It stores one timestamped atomic note per dialogue turn, generates an optional keyword query, and expands untyped links. It is substantially more expensive and can return a large context.
3. `amem_test` makes construction cheaper when original evolution is disabled, then adds a typed graph, optional temporal state, and optional content-addressed evidence. The graph and audit data are structurally useful, but retrieval is currently a bounded breadth-first walk from semantic seeds, not a query-aware graph ranker.
4. The apparent `31/49` typed-only results are not yet causal evidence. The same typed-only memory source produced `24/49` and `28/49` in two query runs, and rejudging unchanged answers changed a result from `28/49` to `26/49`.
5. The immediate priority is experimental control and observability: run the full query set, freeze one memory snapshot, cache retrieval keywords, record retrieval scores and prompts, repeat final generation and judging, and separate build features from query-time features.

The recommended next research direction is a typed-only memory snapshot with a query-aware, token-budgeted evidence selector. Temporal retrieval should be redesigned around direct normalized event timestamps and entities, with graph expansion used as supporting context rather than the primary temporal index.

## Scope and artifact inventory

### Source reviewed

- `methods/amem_agent.py`: shared adapter, provider setup, chunking, direct retrieval, answer generation, and snapshot serialization.
- `methods/amem_fix_agent.py`: atomic note normalization, keyword generation, semantic retrieval, and untyped link expansion.
- `methods/amem_test_agent.py`: experimental snapshot fields, provenance ingestion, typed/temporal/provenance retrieval context, and audit records.
- `methods/amem/A-mem/memory_layer_robust.py`: metadata extraction, original A-MEM evolution, embedding retriever, and consolidation.
- `methods/amem/A-mem/memory_layer_typed.py`: typed edge inference/storage, temporal transitions, provenance hashing, graph expansion, and temporal selection.
- `benchmarks/medmemorybench/dataset.py`, `benchmarks/medmemorybench/evaluator.py`, and `src/result.py`: evaluation-unit construction, batch query preparation, timing, and persisted output fields.
- `benchmarks/locomo/evaluator.py`, `utils/templates.py`, and `methods/amem/A-mem/test_advanced_robust.py`: active and standalone LoCoMo paths.
- `tests/test_amem_fix_agent.py`, `tests/test_amem_test_agent.py`, and `tests/test_amem_staged_memory.py`: current regression coverage.

The repository was already dirty before this report was created. Existing changes, generated outputs, configs, and vendored code were left untouched.

### Main run artifacts

The table below is an artifact inventory, not a causal benchmark table. The runs differ in coverage, build snapshot, provider/backend, temperatures, expansion budgets, and sometimes judge state.

| Run | Main configuration | Score | Build time | LLM calls | Average final memories |
|---|---|---:|---:|---:|---:|
| `outputs/amem_gemini-2.5-flash/20260802_150616` | Raw `amem`, full historical run | `43/97` | 13,873 s | 922 | 3.76 |
| `outputs/amem_fix_gemini-2.5-flash/20260812_112716` | `amem_fix`, evolution + keyword query + links | `26/49` | 25,517 s | 2,801 | 30.9 |
| `outputs/amem_test_gemini-2.5-flash/20260816_014759` | Typed only, no original evolution, expansion 10 | `31/49` | 15,394 s | 1,702 | 20.0 |
| `outputs/amem_test_gemini-2.5-flash/20260817_011704` | Typed only, no original evolution, expansion 20 | `31/49` | 13,932 s | 1,702 | 30.0 |
| `outputs/amem_test_gemini-2.5-flash/20260816_111258` | Typed + temporal state | `30/49` | 16,579 s | 1,702 | 23.1 |
| `outputs/amem_test_gemini-2.5-flash/20260816_111502` | Typed + provenance, raw evidence off | `28/49` | 16,659 s | 1,702 | 19.9 |
| `outputs/amem_test_gemini-2.5-flash/20260817_022709` | Typed + provenance, raw evidence on | `28/49` | 13,248 s | 1,702 | 20.0 |
| `outputs/amem_test_gemini-2.5-flash/20260816_014953` | Original evolution + untyped links, typed off | `27/49` | 24,706 s | 2,826 | 32.1 |

The two runs under `20260813_181118` and `20260813_203724` used the same typed-only memory source, `20260813_143538`, with the same expansion-20 configuration. They scored `24/49` and `28/49`. This is the most important reproducibility artifact in the repository.

The current reports also contain a rejudge artifact under `outputs/amem_test_gemini-2.5-flash/20260813_223103`; the saved answers originally scored `28/49` and scored `26/49` when judged again. This demonstrates judge variance independently of memory construction and answer generation.

## End-to-end architecture

### Variant selection and context isolation

`src/agent.py` routes the three method names to separate classes. Each persona/context receives an independent memory system. `AMemAgent` creates a SentenceTransformer-backed retriever and an internal A-MEM LLM controller in `methods/amem_agent.py:83-157`.

The internal A-MEM API base selection is:

```text
explicit base_url -> BIGMODEL_BASE_URL -> OPENAI_BASE_URL -> BIGMODEL_BASE_URL default
```

This is safe for the Gemini controller, which constructs its own provider client, but it is surprising for an OpenAI-backed internal memory controller: absent an explicit base URL, an OpenAI memory call can inherit the BigModel fallback. This should be made provider-specific and logged in the run configuration.

### Raw `amem`

The raw adapter follows the original coarse flow in `methods/amem_agent.py:328-491`:

1. Count the input with the query LLM client's tokenizer.
2. Split oversized input at paragraph and sentence boundaries; a sentence that is still too long is split by a conservative character estimate.
3. Send each chunk directly to `RobustAgenticMemorySystem.add_note()`.
4. At query time call `find_related_memories(question, k=retrieve_num)`.
5. Format the retrieved notes into the final answer prompt.

The robust layer extracts metadata for each note and, when neighbors exist, may run original evolution. Raw `amem` does not normalize dialogue turns, generate a query keyword string, or expand the graph at query time. It therefore has fewer and larger memories, less temporal granularity, and less provenance than the newer adapters.

The complete historical artifact contains 209 stored passages for 100 reported sessions and 97 queries. Its result is useful as a historical baseline, but its `BAAI/bge-small-zh-v1.5` embedding configuration, old backend settings, chunk size, and model setup differ from the newer `all-MiniLM-L6-v2` experiments.

### `amem_fix`

`AMemFixAgent` changes the logical unit of memory in `methods/amem_fix_agent.py:118-180`:

1. Prefer structured `memory_items` supplied by the evaluator.
2. Map roles to labels such as `Patient` and `Doctor`.
3. Preserve each source timestamp.
4. Add image captions to the stored note text when supplied.
5. Store one note per dialogue turn, splitting only if a turn exceeds the chunk budget.

If structured turns are unavailable, the fallback parser strips the evaluator wrapper and recognizes common `Speaker: content` turn formats (`methods/amem_fix_agent.py:78-128`). The fallback is useful, but it is necessarily less reliable than the structured path for arbitrary speaker names and nested dialogue text.

At query time (`methods/amem_fix_agent.py:196-339`):

1. Optionally send the raw question to an internal LLM to generate comma-separated keywords.
2. Search the semantic retriever with that keyword string.
3. Keep the top `retrieve_num` direct results.
4. For every direct result, add up to `retrieve_num` entries from its untyped `links` list.
5. Preserve insertion order and format every selected note into the final prompt.

The link expansion has no global expansion budget beyond the per-seed slice and no relevance reranking after expansion. It can therefore create a much larger prompt than `retrieve_num` suggests. The selected `amem_fix` artifact averaged 30.9 final memories per query.

### `amem_test`

`AMemTestAgent` loads `TypedRelationMemorySystem` instead of the robust class (`methods/amem_test_agent.py:77-158`). Its build flags are:

- `amem_original_evolution`: preserve the original neighbor-evolution behavior.
- `amem_typed_relations`: infer and store typed edges.
- `amem_temporal_state`: derive validity/currentness from typed transitions.
- `amem_provenance`: store immutable source evidence.

The constructor correctly rejects temporal state without typed relations (`methods/amem_test_agent.py:44-53`). Snapshot import also checks the experimental flags (`methods/amem_test_agent.py:160-230`). This protects state integrity, but it currently makes query-time ablations harder because build-time and query-time flags are treated as one compatibility contract.

When provenance is enabled, each structured turn carries source session, event, turn, timestamp, speaker, role, image caption, and exact raw text into `add_note()` (`methods/amem_test_agent.py:262-397`). When provenance is disabled, `amem_test` reuses the `amem_fix` atomic-note path.

The experimental query path is in `methods/amem_test_agent.py:626-873`:

```text
raw question
  -> optional keyword generation
  -> semantic seeds
  -> optional amem_fix untyped links
  -> typed graph expansion
  -> optional temporal expansion and ordering
  -> optional provenance/evidence records
  -> one relation-aware context string
  -> final answer LLM
```

The final answer is still generated through the normal agent/batch path. Vertex batching covers the final answer requests, not the internal metadata, evolution, relation, temporal, or keyword calls.

## Memory construction in detail

### Metadata extraction

`RobustMemoryNote` runs an LLM metadata analysis whenever keywords, context, category, or tags are missing (`methods/amem/A-mem/memory_layer_robust.py:381-468`). The initial call parses a plain-text response; empty keywords trigger a focused retry. Non-provider parsing failures fall back to heuristic keywords/context, while provider failures are re-raised.

This design is robust to imperfect output, but it has three consequences:

- Every atomic note can cost at least one remote LLM request.
- Metadata quality is part of the retrieval index, so a parsing error affects both initial retrieval and relation candidate selection.
- The metadata prompt sees the note content, not a source-level structured object. Long notes are truncated before analysis.

### Original evolution

When original evolution is enabled, `process_memory()` first retrieves five semantic neighbors and may issue up to three sequential calls (`methods/amem/A-mem/memory_layer_robust.py:603-693`):

1. Decide `NO_EVOLUTION`, `STRENGTHEN`, `UPDATE_NEIGHBOR`, or both.
2. Optionally strengthen the new note and add connections/tags.
3. Optionally update neighbor tags and contexts.

The selected `amem_fix` run spent 25,517 seconds and 2,801 calls, compared with roughly 15,394 seconds and 1,702 calls for the typed-only expansion-10 run. The main cost saving in typed-only mode is disabling original evolution; typed relation inference still costs approximately one additional LLM call for almost every note.

There is a state-consistency limitation: original evolution mutates old note metadata in `self.memories`, but the retriever corpus for those old notes is not rebuilt immediately. Consolidation happens only after the evolution counter reaches `amem_evo_threshold` (`methods/amem/A-mem/memory_layer_robust.py:510-543` and `:645-685`). Until then, search can use stale context/tags/keywords for previously stored notes.

### Typed relation construction

Typed construction works as follows (`methods/amem/A-mem/memory_layer_typed.py:757-960`):

1. Search the current embedding retriever for `amem_relation_candidate_count` neighbors.
2. Send the new note and truncated candidate records to a relation prompt.
3. Parse labels from `SUPPORT`, `REFINE`, `SUPERSEDE`, `CONFLICT`, and `RELATED`.
4. Map candidate positions back to stable memory IDs.
5. Store the edge in both source and target note lists and in the global edge list.
6. Save candidate and raw-response audit data.

The candidate recall ceiling is structural: a related note that is not among the top candidate embeddings is never considered. There is no later global graph pass or missed-edge backfill. The relation prompt also sees truncated candidate content, so a decisive statement outside the retained prefix cannot affect the label.

Representative full typed snapshots contained 788 memories and about 811-825 edges, depending on independent construction. One `20260817_011704` snapshot contained 814 edges with this mix:

| Relation | Count in representative snapshot |
|---|---:|
| `SUPPORT` | 431 |
| `REFINE` | 185 |
| `RELATED` | 169 |
| `CONFLICT` | 13 |
| `SUPERSEDE` | 16 |

All 788 relation audits had approximately five candidates on average and approximately one stored prediction per note. Several zero-confidence edges were retained in storage/audit; the default retrieval threshold of `0.5` excludes them from expansion. Retaining them is useful for audit, but the distinction between predicted, accepted, and retrieval-eligible edges should be explicit in the schema.

The edge deduplication key is only `(source_id, target_id)` (`memory_layer_typed.py:811-834`). A second label for the same pair is discarded, even if the model predicts both `REFINE` and `CONFLICT` or if independent evidence supports a second relation. The tests intentionally document this behavior in `tests/test_amem_test_agent.py:154-175`; it is a limitation for contradictory or multi-faceted medical facts, not merely a test detail.

### Temporal state construction

Temporal state starts every note as current with an open validity interval and applies `SUPERSEDE`, `REFINE`, and `CONFLICT` transitions (`memory_layer_typed.py:478-652`). Invalid timestamp order for a `SUPERSEDE` transition is rejected and audited. In the inspected temporal snapshot:

- 224 temporal transition records were stored.
- 218 transitions were applied.
- 6 transitions were rejected.
- 9 memories ended in a superseded state.

This demonstrates that the temporal state machine is doing real work, but it also shows the limited amount of explicit supersession in the generated graph. Most information remains marked current, and `REFINE` does not make the older note historical. That is appropriate for a refinement relation, but it means current-vs-historical reasoning depends heavily on relation-label quality.

### Provenance construction

Provenance uses a canonical JSON representation of source evidence and hashes it with SHA-256 to create stable `ev_...` IDs (`memory_layer_typed.py:654-681`). Evidence is immutable and deduplicated by content. The inspected provenance snapshots contained:

- 788 evidence records for 788 memories.
- 788 unique evidence IDs.
- 0 provenance audit errors.

This is a strong audit feature. However, chunked notes reuse the same source-turn evidence record and identify only a `part_index`; they do not carry exact character or token spans. Provenance is therefore source-turn-level, not exact-span-level.

## Retrieval and query answering

### Semantic retrieval

The shared retriever encodes the query and returns cosine-similarity top-k indices (`methods/amem/A-mem/memory_layer.py:564-620`). It does not return scores in the adapter audit, only indices. The corpus text combines note content, context, keywords, and tags. This is convenient, but metadata errors can shift semantic retrieval even when the raw note contains the answer.

Keyword generation is an extra LLM request per question (`amem_fix_agent.py:196-214`). It is not cached. Consequently, the same raw question can produce a different semantic query, which can change seeds before any graph feature is applied. The evaluator does pass both the formatted question and the raw question, and the adapters correctly use the raw question for keyword generation and temporal parsing (`benchmarks/medmemorybench/evaluator.py:1762-1774`).

### Link and typed expansion

`amem_fix` performs one-hop untyped expansion from every direct seed. `amem_test` first reuses that path, then calls `expand_typed_relations()` when typed relations are enabled (`amem_test_agent.py:681-725`; `memory_layer_typed.py:981-1056`). Typed expansion is a bidirectional breadth-first walk:

- Relation priority is `SUPERSEDE`, `CONFLICT`, `REFINE`, `SUPPORT`, then `RELATED`.
- `RELATED` is excluded unless explicitly enabled.
- Edges below the confidence threshold are ignored.
- The expansion budget limits added memories, not total prompt tokens.
- The selected IDs are not reranked by query similarity after expansion.

This is a reasonable first graph baseline, but it can add a highly related note that is irrelevant to the current question while omitting a semantically weaker but decisive fact. It also makes expansion count a prompt-size control only indirectly. In the artifacts, typed expansion 10 normally produced exactly 20 final memories and expansion 20 exactly 30 when there were no duplicates.

### Temporal retrieval

Temporal query detection is regex and heuristic based (`memory_layer_typed.py:130-263`). It recognizes ISO dates, English month names, Chinese month forms, years, relative phrases, and intents such as current, historical, time-specific, and change. The selector then traverses temporal neighbors, orders memories by state, and formats the state into the prompt (`memory_layer_typed.py:1190-1324`).

Important limitations:

- Parsing scans the full question string. For multiple-choice questions, a date in a distractor can become the target date.
- Relative dates are anchored to the latest timestamp among selected memories, not to a known conversation clock or an explicitly resolved event.
- Month-only questions without a year infer a year from the selected timeline only when exactly one year is known.
- There is no direct timestamp/entity index; temporal recall still depends on semantic seeds and graph edges.
- Natural event time and source/session time are not modeled as separate first-class fields.

The temporal feature therefore changes ordering and adds state context, but it does not guarantee that the correct timestamp enters the candidate set.

### Provenance retrieval and context assembly

`amem_test` collects up to `amem_provenance_max_evidence` evidence records after the final memory set is selected. Raw source text is injected only when configured or when it differs from the structured note (`amem_test_agent.py:574-624`). In the raw-off run, evidence records were present but no additional raw source text was injected; the structured note already contained the turn. Raw-on increases grounding material and prompt cost, but it is not an independent memory-recall feature.

The final relation-aware context contains, in order:

1. typed relation lines and confidence/reason text;
2. temporal state and query intent, when enabled;
3. immutable evidence labels and optional raw source conversations;
4. all selected A-MEM notes with IDs and origin labels.

The entire context is truncated as one large string against the remaining token budget (`amem_test_agent.py:731-756`). This can cut a relation line, evidence record, or note in the middle and does not guarantee that every selected seed survives. A token-aware selector should choose complete evidence blocks before formatting.

### Answer generation and persisted result limitations

The evaluator prepares retrieval locally and batches only final answer generation (`benchmarks/medmemorybench/evaluator.py:1723-1869`). Internal keyword, metadata, evolution, relation, and temporal calls remain synchronous. The result writer records `query_time` from the response, but batch finalization does not populate a measured query time, so the query JSON reports `total_query_time: 0.0` (`src/result.py:315-375`).

The experimental adapter places the full retrieval audit in `AgentResponse.extra`, but the query answer serializer does not persist that `extra` object. It stores the audit only inside the first retrieved-memory record (`amem_test_agent.py:807-873`). This means the saved artifact often lacks the generated keyword string, embedding scores, full answer-stage metadata, and a clean query-level audit unless the first record is inspected. This is a major obstacle to failure attribution.

## MedMemoryBench result interpretation

### Coverage caveat

The dataset inventory contains 101 sessions and 97 queries. Most recent AMEM experiments use Persona 1 with `max_sessions_per_persona: 50` and `evaluation_interval: 10`. The evaluator emits checkpoints at sessions 10, 20, 30, 40, and 50, so these runs score only 49 queries:

| Query type | Scored in 50-session runs |
|---|---:|
| Entity exact match | 10 |
| Multiple choice | 10 |
| Temporal localization | 10 |
| State update | 5 |
| Inference generation | 10 |
| Multi-hop clinical deduction | 4 |
| Total | 49 |

The remaining 48 queries are from later checkpoints. A `31/49` result is therefore not a full benchmark score and should not be compared directly with the historical `43/97` raw `amem` result.

### Feature-flag observations

The following observations are supported by the stored results, but should be treated as hypotheses until repeated on a fixed snapshot:

- Typed-only construction is much cheaper than original evolution and still produces a connected graph. This is the clearest efficiency win in the current implementation.
- Increasing typed expansion from 10 to 20 increased the final context from about 20 to 30 memories, but did not improve the observed score in the selected runs (`31/49` in both). This suggests the extra neighbors are not being selected by relevance.
- Temporal state produced a `30/49` result in one zero-temperature run and lower results in earlier artifacts. It is promising for state-update and temporal questions, but current graph and date parsing errors prevent a causal conclusion.
- Provenance did not demonstrate an accuracy gain. Its main value is debugging and grounding auditability. Raw source injection should be evaluated separately because it changes prompt length and duplicates information already present in the note.
- Original evolution plus untyped links remains a useful reference, but it is slower and returns more context. The `27/49` result does not establish that evolution is harmful because the build and query conditions are not identical to typed-only runs.
- Query type results are unstable. For example, typed-only expansion-20 runs on the same memory source differed in entity, state, inference, and temporal judgments. Do not infer a feature win from one or two answers.

## LoCoMo compatibility

### Active evaluator

The active LoCoMo evaluator supports `amem_test` and forwards timestamps, image captions, session IDs, event IDs, and turn IDs. This is compatible with the provenance path (`benchmarks/locomo/evaluator.py:238-325`). Natural timestamp coverage exists in `tests/test_amem_test_agent.py:284-296`.

However:

- No saved active `amem_test` LoCoMo run was found under `outputs/`, so cross-dataset behavior remains unverified.
- Separated memory/query execution is not implemented for LoCoMo; `execution_stage` must be `all` (`benchmarks/locomo/evaluator.py:830-846`). This prevents fixed-snapshot query ablations through the normal LoCoMo CLI.
- `utils/templates.py:17-32` explicitly maps `amem_fix` to `agentic`, but has no explicit `amem` or `amem_test` mapping. They fall back to `rag`, which changes source descriptions and may change the QA prompt semantics.

### Standalone harness

`methods/amem/A-mem/test_advanced_robust.py` is not an `amem_test` evaluation. It uses the robust untyped memory layer, generates keyword queries, retrieves raw untyped memories, and calls an answer LLM. It also randomizes the order of the category-5 answer choices (`test_advanced_robust.py:109-153`). Results from that harness should not be used as evidence for typed relations, temporal state, or provenance, and its randomized answer ordering must be disabled for reproducible experiments.

## Prioritized limitations

### P0: experimental validity

1. **Incomplete and mismatched coverage.** Most feature runs score 49/97, while the historical raw baseline scores 97 queries.
2. **Multiple stochastic stages.** Keyword generation, relation inference, final answer generation, and LLM judging can all vary. Rejudging unchanged answers already changes four labels in one artifact comparison.
3. **Insufficient failure attribution.** Saved results do not cleanly preserve keyword strings, embedding scores, context token counts, truncation boundaries, and all `AgentResponse.extra` fields.
4. **Timing is incomplete.** Batch query result files report zero query time, so memory build cost and answer cost cannot be compared fairly.

### P1: retrieval quality

5. **Graph expansion is not query-aware.** BFS order and relation priority are fixed; expanded notes are not reranked with the question.
6. **Candidate recall is bounded by embedding top-k.** Missed relation candidates are never reconsidered.
7. **Temporal recall is indirect.** A timestamp can be correct in the memory but never retrieved because there is no direct temporal/entity index.
8. **Contexts can be too large.** Link expansion and typed expansion are memory-count bounded rather than token-budget bounded; final truncation can cut records.
9. **No score-aware retrieval audit.** Search returns indices but the adapter does not persist cosine scores or rank margins.

### P1: memory construction and semantics

10. **Original evolution is expensive and serial.** Up to three sequential calls per note are possible, and every metadata/relation call is also synchronous.
11. **Retriever metadata can be stale after evolution.** Old note metadata changes are not immediately reflected in the embedding corpus.
12. **Typed edge semantics are lossy.** Pair-only deduplication prevents multiple relation labels between the same two memories.
13. **Confidence is not calibrated.** Zero-confidence predictions are stored, and a fixed `0.5` threshold is used without precision/recall validation.
14. **Generated summaries are repetitive.** Dialogue turns often contain patient statements, doctor restatements, and later refinements. Atomic storage preserves recall but creates many near-duplicate notes and relation edges.

### P2: evidence, persistence, and portability

15. **Provenance is source-turn-level.** Chunk parts do not retain exact source spans.
16. **Evidence is capped by count, not tokens.** Ten long records can cost more than the remaining context budget.
17. **Snapshot compatibility is too coarse for ablations.** Query-only feature flags are entangled with build flags, forcing rebuilds or awkward exact-config reuse.
18. **Snapshots are verbose.** Full note attributes, relation audits, raw relation responses, temporal audits, evidence, and separate embedding sidecars create large JSON artifacts.
19. **LoCoMo lacks a fixed-snapshot path and has prompt/randomness inconsistencies.** This prevents a clean cross-dataset comparison today.

## Recommended fixes before the next large run

These changes improve validity and efficiency without changing the intended memory semantics:

1. **Add a query audit record as a first-class persisted object.** Persist raw question, formatted question hash, generated retrieval query, semantic seed IDs and scores, link IDs, typed expansion IDs and edge types, temporal parse, evidence IDs, context token counts, truncation status, model settings, and measured local/remote timings.
2. **Separate build configuration from retrieval configuration.** Snapshot compatibility should validate embedding/chunk/build semantics, while query experiments should be able to vary keyword use, expansion budgets, confidence thresholds, temporal ordering, provenance injection, and context budgets without rebuilding unchanged memory.
3. **Cache keyword generation by `(question, keyword-prompt-version, model, temperature)`.** Store both the cache key and generated string. Add a raw-question retrieval mode for a direct ablation.
4. **Return semantic scores and rerank final evidence.** Keep the semantic seed score, edge confidence, timestamp match, lexical/entity match, and origin in a single retrieval score. Select complete evidence blocks under a token budget.
5. **Fix temporal parsing inputs.** Parse the question stem separately from multiple-choice options, normalize timestamps at ingestion, preserve event time versus source/session time, and use a direct date/entity index before graph expansion.
6. **Make relation storage multi-label.** Deduplicate by `(source_id, target_id, relation_type)` or store a set of typed observations with confidence and provenance. Keep rejected/zero-confidence predictions in audit-only storage.
7. **Add deterministic structural relation rules.** For explicit date-ordered updates, medication changes, and repeated state values, generate candidate `SUPERSEDE`/`REFINE` edges from rules and let the LLM classify only ambiguous cases.
8. **Add operation-level caching and concurrency.** Cache metadata by source content hash; batch or safely parallelize independent metadata/relation calls; skip relation inference when there are no candidates; and consolidate retriever metadata incrementally after mutations.
9. **Fix output timing.** Measure query preparation, batch waiting, finalization, judging, and total wall time separately. Never use `0.0` for an unmeasured duration.
10. **Add explicit LoCoMo method templates and remove random answer ordering.** Map `amem`/`amem_test` deliberately and make category-5 option order deterministic.

## Controlled experiment protocol

### Phase 0: establish a valid reference

1. Use a dedicated Persona 1 configuration whose explicit flags match the intended combination and whose dataset override has `max_sessions_per_persona: null`. The current `persona_1/amem_test2_gemini.yaml` enables original evolution, typed relations, temporal state, and provenance, so it is not a typed-only configuration.
2. Run all 101 sessions and score all 97 queries. Record the exact dataset file hashes, config hash, model IDs, provider/backend, temperatures, and judge model.
3. Build one typed-only snapshot with original evolution disabled, typed relations enabled, candidate count 5, relation temperature 0, and no untyped links.
4. Verify that every query has a prepared prompt and that the memory manifest is complete before comparing scores.

Example starting commands:

```bash
python main.py -m persona_1/amem_test2_gemini-35flashlite -d medmemorybench --stage memory
python main.py --stage query --memory-run YYYYMMDD_HHMMSS
```

Use the actual published run directory from the memory-stage output. For final generation, use the same answer model and temperature for every condition.

### Phase 1: query-only ablations on one frozen snapshot

Keep memory IDs, metadata, embeddings, relation edges, evidence, questions, prompts, answer model, and judge fixed. Compare:

| Condition | Keyword query | Typed expansion | Temporal | Provenance | Purpose |
|---|---|---:|---:|---:|---|
| A | cached | 0 | off | off | Semantic-seed floor |
| B | cached | 5 | off | off | Small graph budget |
| C | cached | 10 | off | off | Current default range |
| D | cached | 20 | off | off | Larger graph budget |
| E | cached | 40 | off | off | Saturation test |
| F | cached | 10 | annotation only | off | State prompt value without extra traversal |
| G | cached | 10 | annotation + expansion | off | Temporal retrieval value |
| H | cached | 10 | off | metadata only/raw off | Audit-only grounding |
| I | cached | 10 | off | raw on | Raw evidence grounding cost/value |
| J | raw question | 10 | off | off | Keyword-generation value |

Run at least three final-answer repeats per condition at temperature 0. If the provider remains nondeterministic at temperature 0, report answer hashes and per-query flip rates. Reuse the exact final prompts for a separate judge-repeat study.

### Phase 2: construction ablations

After query behavior is understood, rebuild only the following controlled factors:

- original evolution: off versus on;
- relation candidate count: 3, 5, 10;
- relation temperature: 0 versus the current nonzero default;
- relation confidence threshold: 0, 0.5, 0.7, 0.9;
- candidate generation: semantic top-k versus semantic plus lexical/entity candidates;
- note representation: atomic turns versus deduplicated/refinement-aware notes.

For every build, report relation precision/utility on a manually labeled sample, not just edge count. A larger graph is not evidence of a better graph.

### Phase 3: retrieval algorithm experiments

Compare the current BFS against:

1. semantic-only top-k;
2. semantic seed plus query-aware edge reranking;
3. semantic plus lexical/entity/timestamp retrieval;
4. hybrid candidates followed by token-budgeted MMR or evidence selection.

The primary retrieval metric should be supporting-memory recall: whether the memories containing the benchmark's source key points or reasoning-chain nodes enter the final context. Answer accuracy is secondary because it includes answer-model and judge variance.

### Phase 4: cross-dataset validation

Run the same frozen-snapshot protocol through the active LoCoMo evaluator. First make the evaluator deterministic and add separated memory/query execution. Do not mix results from the standalone robust harness with `amem_test` results.

## Metrics to record

### Build metrics

- number of source sessions, source turns, notes, chunks, and duplicate/near-duplicate notes;
- metadata calls, retries, fallback parses, input/output tokens, latency, and errors;
- evolution decisions and calls by type;
- relation candidate count, prediction count, labels, confidence distribution, malformed outputs, and candidate recall sample;
- temporal transitions applied/rejected by reason and status distribution;
- provenance records, unique evidence IDs, missing source IDs, and exact-span availability;
- wall time, serialized snapshot size, embedding sidecar size, and memory entries per session.

### Query metrics

- raw question and generated retrieval query;
- semantic seed IDs, ranks, scores, and score margins;
- link and typed expansion IDs, relation types, confidence, direction, and origin;
- temporal intent, parsed target, anchor, direct timestamp hits, and temporal expansions;
- evidence IDs, raw injection flags, selected source tokens, and duplicate suppression;
- context token count before/after formatting, truncation amount, and which records were cut;
- keyword, answer, and judge calls with latency and token usage.

### Quality metrics

- overall and per-query-type accuracy on the same query set;
- exact supporting-memory recall and final-context recall;
- answer repeat agreement and judge repeat agreement;
- per-query score variance, bootstrap confidence intervals, and paired condition deltas;
- multi-hop NCR, CRC, CC, node mention rate, and node causal rate;
- accuracy per added context token and per build/query LLM call.

## Proposed next-experiment matrix

| Priority | Experiment | Main question | Success criterion |
|---:|---|---|---|
| 0 | Full 97-query typed-only reference | What is the real baseline? | Complete coverage and reproducible manifest |
| 1 | Fixed-snapshot expansion `0/5/10/20/40` | Does graph expansion help after seed retrieval? | Paired gain in supporting-memory recall and accuracy without excessive tokens |
| 2 | Cached keyword versus raw question | Does the keyword LLM help? | Higher seed recall with lower variance and measurable cost justification |
| 3 | Three answer repeats plus judge repeats | How much variance is external to memory? | Confidence intervals and per-query flip table |
| 4 | Temporal direct index versus graph traversal | Does timestamp indexing fix temporal/state queries? | Higher temporal supporting-memory recall and fewer distractor-date parses |
| 5 | Query-aware reranking versus BFS | Are graph neighbors useful when ranked by the question? | Better recall per context token |
| 6 | Candidate count `3/5/10` and threshold `0/.5/.7/.9` | What graph precision/recall tradeoff is useful? | Labeled edge utility and stable answer gains |
| 7 | Provenance metadata/raw evidence/token budget | Is raw source grounding worth its prompt cost? | Better evidence faithfulness at bounded token cost |
| 8 | Cached/batched metadata and relation calls | How much build time can be removed safely? | Same snapshot quality with lower wall time/call count |
| 9 | Fixed-snapshot LoCoMo run | Does the design transfer beyond MedMemoryBench? | Deterministic category-level comparison using the active evaluator |

## Recommended target architecture

The most promising near-term design is:

```text
atomic source notes
  -> metadata cache and exact source provenance
  -> semantic + lexical/entity + normalized-time candidate retrieval
  -> typed edge lookup for candidates only
  -> query-aware edge/recency/state reranking
  -> complete evidence blocks under a token budget
  -> final answer generation with persisted prompt and audit
```

In this design, typed relations explain and connect already plausible candidates. They do not carry the full burden of recall. Temporal state supplies constraints and ordering, while direct timestamp/entity indexes guarantee that explicit dates, medications, lab values, and named conditions have a retrieval path. Provenance remains attached to every selected fact and is injected selectively rather than wholesale.

## Source and artifact references

- Guides: `docs/AMEM_GUIDE.md`, `docs/AMEM_IMPLEMENTATION.md`, `docs/AMEM_EXPERIMENT_ANALYSIS.md`.
- Shared adapter: `methods/amem_agent.py:83-151`, `:185-325`, `:328-438`, `:440-569`.
- Paper-aligned adapter: `methods/amem_fix_agent.py:118-180`, `:196-291`, `:293-339`.
- Experimental adapter: `methods/amem_test_agent.py:77-260`, `:262-460`, `:462-624`, `:626-873`.
- Robust layer: `methods/amem/A-mem/memory_layer_robust.py:381-468`, `:510-562`, `:603-693`.
- Typed layer: `methods/amem/A-mem/memory_layer_typed.py:130-263`, `:430-681`, `:757-960`, `:981-1072`, `:1190-1324`.
- Coverage/evaluation: `benchmarks/medmemorybench/dataset.py:229-330`, `benchmarks/medmemorybench/evaluator.py:1723-1869`, `src/result.py:315-375`.
- LoCoMo: `benchmarks/locomo/evaluator.py:238-325`, `:645-723`, `:830-846`; `utils/templates.py:17-32`; `methods/amem/A-mem/test_advanced_robust.py:53-153`.
- Representative baseline: `outputs/amem_gemini-2.5-flash/20260802_150616`.
- Representative `amem_fix`: `outputs/amem_fix_gemini-2.5-flash/20260812_112716`.
- Fixed-snapshot typed repeats: `outputs/amem_test_gemini-2.5-flash/20260813_143538`, `20260813_181118`, and `20260813_203724`.
- Zero-temperature typed/feature runs: `outputs/amem_test_gemini-2.5-flash/20260816_014759`, `20260816_111258`, `20260816_111502`, `20260817_011704`, and `20260817_022709`.
