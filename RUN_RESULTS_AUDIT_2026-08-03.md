# MedMemoryBench Gemini Results: Research-Level Post-Deduplication Audit

Audit date: 2026-08-07 (updated from the 2026-08-03 snapshot)

## Abstract

This report audits the completed English MedMemoryBench Gemini 2.5 Flash runs after duplicate query rows were removed from historical checkpoint-resume artifacts. The evidence supports four conclusions.

1. **A-Mem is the strongest completed method in logical score** at 43/97 (44.3%), especially on multiple choice, inference generation, and the most demanding patient-specific causal questions. Its result is recovered from a historical duplicated run, so a clean rerun is still required before publication.
2. **Fresh Long Context (32/97) and Embedding RAG (31/97) are indistinguishable in this single-run comparison.** Their paired outcomes differ by only one net query; they excel on different question families.
3. **Fresh LightMem (19/97) is strongly concentrated in exact multiple-choice questions (13/19), rather than showing broad patient-memory grounding.** Its fixed 20-item retrieval succeeds on no query that the other seven methods all miss and reaches neither state-update nor multi-hop clinical-deduction correctness.
4. **The benchmark/result pipeline has two material validity limitations:** a known historical resume-duplication defect and a newly identified false-negative metric case for the punctuation-only gold answer `++`. Neither limitation changes the broad ranking, but both matter for publication-quality claims.

The fresh Mem0 replacement and the completed fresh LightMem run are now included. The active fresh GraphRAG job remains excluded; its checkpoints, local Vertex manifests, and output files were not altered.

## 1. Scope, Inputs, And Treatment Of Artifacts

### 1.1 Evaluated scope

- Dataset: English MedMemoryBench (`data/MedMemoryBench_EN`)
- Persona: 1 only
- History: 100 sessions; evaluation at sessions 10 through 100
- Benchmark queries: 97 unique IDs
- Generator model: Gemini 2.5 Flash
- Completed methods analyzed: A-Mem, Long Context, Embedding RAG, BM25 RAG, GraphRAG, LightMem, MemOS, and Mem0

This is a *within-benchmark, single-run* comparison. It is not a replicated experiment, an estimate of clinical utility, or a claim of statistical generalization across personas, datasets, seeds, prompts, or models.

### 1.2 Source selection

Read this table as the provenance of every later score. Each row names one method, the exact timestamped artifact selected for it, and why that artifact is considered clean, recovered, or provisional. “Recovered” means duplicate rows were removed; it does not mean the method was rerun.

| Method | Artifact used in this report | Reason |
| --- | --- | --- |
| A-Mem | `amem_gemini-2.5-flash_20260802_192343` | Only available completed result; logically recovered from duplicate rows |
| Long Context | `long_context_gemini-2.5-flash_20260802_234838` | New clean completed run; 97 unique rows |
| Embedding RAG | `embedding_rag_gemini-2.5-flash_20260802_104618` | Clean completed run |
| BM25 RAG | `bm25_rag_gemini-2.5-flash_20260802_104515` | Clean completed run |
| GraphRAG | `graph_rag_gemini-2.5-flash_20260802_181710` | Recovered historical diagnostic only; fresh replacement remains active |
| LightMem | `lightmem_gemini-2.5-flash_20260807_011640` | Fresh completed persona-1 run; 97 unique rows |
| MemOS | `memos_gemini-2.5-flash_20260802_225721` | Completed 97-row artifact |
| Mem0 | `mem0_gemini-2.5-flash_20260803_020004` | Fresh completed persona-1 run; 97 unique rows |

The other LightMem artifact, `lightmem_gemini-2.5-flash_20260806_201708`, is explicitly excluded. It is a `--dry-run` artifact over the default ten-persona configuration: it contains 992 rows but only 100 non-namespaced query IDs, every answer is `[DRY RUN]`, and its recorded LLM usage is zero. It is not a completed benchmark result and must not be compared with the selected persona-1 run.

At the initial audit snapshot, this GraphRAG command remained active and is intentionally out of scope:

```bash
python main.py -m persona_1/graph_rag_gemini -d medmemorybench --batch-api --batch-wait
```

### 1.3 Duplicate-row repair

The old checkpoint-resume failure re-evaluated already saved query IDs. For each affected historical `query_answer.json`, this audit retained the first occurrence of every `query_id`, then rebuilt both the query-answer summary/context list and the paired result summary.

This table quantifies that repair. “Rows before” is the number of saved query records in the original artifact, “removed duplicate rows” is the excess beyond the 97 benchmark IDs, “unique rows after” is the corrected logical benchmark size, and “correct after recovery” is the corrected count of rows marked correct.

| Historical artifact | Rows before | Removed duplicate rows | Unique rows after | Correct after recovery |
| --- | ---: | ---: | ---: | ---: |
| Long Context, 09:45 | 125 | 28 | 97 | 33 |
| A-Mem, 19:23 | 121 | 24 | 97 | 43 |
| GraphRAG, 18:17 | 131 | 34 | 97 | 21 |
| Mem0, 20:51 (superseded historical artifact) | 133 | 36 | 97 | 9 |

The original files are preserved, with SHA-256 hashes, in `outputs/deduplication_backups/2026-08-03/manifest.json`. The correction affects logical query rows and score summaries only. Historical `llm_usage` remains the real API work performed by the old run, including repeated/retried calls, and must not be compared as a clean cost measurement.

The first-occurrence policy is exact for Long Context, A-Mem, and the superseded historical Mem0 artifact: each repeated ID had the same score and correctness. GraphRAG is less trustworthy: 20 duplicate IDs had changed later answers. Its recovered score is a diagnostic baseline, not a final GraphRAG result. The selected Mem0 result below is the later fresh artifact and did not need this repair.

## 2. Aggregate Results

### 2.1 Official saved-score view

The confidence intervals below are Wilson 95% intervals over the 97 evaluated rows. They are descriptive only: rows share one persona and correlated history, and some categories use an LLM judge, so the independent-Bernoulli assumption is imperfect.

Each row is one method's selected run. “Correct / 97” is the raw number of benchmark questions scored correct after deduplication; “saved accuracy” is that count divided by 97; the interval describes uncertainty under a simple binomial model; and “result quality” says whether the score is a clean run or a recovered/provisional one. Do not compare the recovered rows as final publication results.

| Method | Correct / 97 | Saved accuracy | Wilson 95% interval | Result quality |
| --- | ---: | ---: | ---: | --- |
| A-Mem | 43 | **44.3%** | 34.8%–54.2% | Recovered historical score |
| Long Context (fresh) | 32 | 33.0% | 24.4%–42.8% | Clean |
| Embedding RAG | 31 | 32.0% | 23.5%–41.8% | Clean |
| BM25 RAG | 29 | 29.9% | 21.7%–39.6% | Clean |
| GraphRAG (recovered) | 21 | 21.6% | 14.6%–30.8% | Recovered; replace with active rerun |
| LightMem (fresh) | 19 | 19.6% | 12.9%–28.6% | Clean 97-row persona-1 run |
| MemOS | 11 | 11.3% | 6.5%–19.2% | Clean rows, low score |
| Mem0 (fresh) | 14 | 14.4% | 9.1%–22.5% | Clean 97-row batch run |

The intervals overlap for Long Context, Embedding RAG, and BM25 RAG. A rank order among those three from one run would be over-interpreting a small difference.

### 2.2 Paired query comparison

Because every method answered the same 97 questions, paired disagreement is more informative than comparing percentages alone. `A-only` means the first method was correct while the second was wrong; `B-only` is the converse. The final column is a two-sided exact McNemar/binomial calculation on discordant pairs, included as a descriptive check rather than a definitive significance claim.

For example, in the first row, A-Mem alone solves 17 questions and Long Context alone solves six; both solve 26 and both miss 48. The p value asks whether the imbalance between the two exclusive-success columns is large under a paired binary model. It does **not** compensate for one-persona scope, one run, or judge variability.

| Pair | A-only | B-only | Both correct | Both wrong | Exact paired p |
| --- | ---: | ---: | ---: | ---: | ---: |
| A-Mem vs Long Context | 17 | 6 | 26 | 48 | 0.0347 |
| Long Context vs Embedding RAG | 14 | 13 | 18 | 52 | 1.0000 |
| Embedding RAG vs BM25 RAG | 11 | 9 | 20 | 57 | 0.8238 |
| A-Mem vs BM25 RAG | 16 | 2 | 27 | 52 | 0.0013 |
| BM25 RAG vs MemOS | 19 | 1 | 10 | 67 | <0.0001 |
| MemOS vs Mem0 | 7 | 10 | 4 | 76 | 0.6291 |
| A-Mem vs LightMem | 25 | 1 | 18 | 53 | <0.0001 |
| Long Context vs LightMem | 19 | 6 | 13 | 59 | 0.0146 |
| Embedding RAG vs LightMem | 18 | 6 | 13 | 60 | 0.0227 |
| BM25 RAG vs LightMem | 15 | 5 | 14 | 63 | 0.0414 |
| GraphRAG vs LightMem | 11 | 9 | 10 | 67 | 0.8238 |
| MemOS vs LightMem | 5 | 13 | 6 | 73 | 0.0963 |
| Mem0 vs LightMem | 3 | 8 | 11 | 75 | 0.2266 |

Interpretation:

- A-Mem's advantage over Long Context is spread over 17 exclusive successes versus six Long Context-exclusive successes; that is the only top-method gap that looks substantial within these rows.
- Long Context and Embedding RAG have nearly symmetric disagreement (14 versus 13). The one-query aggregate gap is not meaningful evidence that either is generally superior.
- BM25 RAG is not clearly worse than Embedding RAG on this one benchmark slice despite its lower total score. Their errors differ, which is useful for an ensemble or follow-up analysis.
- LightMem has markedly fewer exclusive successes than A-Mem, Long Context, Embedding RAG, and BM25 RAG, but its paired result is indistinguishable from the recovered GraphRAG artifact. Its apparent advantage over MemOS and Mem0 is also not decisive on this single persona.

### 2.3 Outcome concentration and difficulty

This is a question-difficulty distribution, not a method ranking. The left column is the number of the eight methods that answered a given query correctly; the right column is how many of the 97 queries fall in that bucket. For example, 42 rows in the `0` bucket means every method missed those 42 queries.

| Number of methods correct on a query | Query count |
| ---: | ---: |
| 0 | 42 |
| 1 | 11 |
| 2 | 8 |
| 3 | 9 |
| 4 | 8 |
| 5 | 7 |
| 6 | 8 |
| 7 | 1 |
| 8 | 3 |

Forty-two queries are missed by every method. This is the clearest evidence that the benchmark is not being solved merely by choosing a different memory architecture: difficult temporal, clinical-advice, and multi-hop questions fail across all of them.

## 3. Performance By Query Family

EEM = entity exact match; MQ = multiple choice; TLA = temporal localization; SUA = state update; IG = inference generation; MCD = multi-hop clinical deduction.

Each row is a method and every score cell is `correct / number of questions in that family`; the number in parentheses in a header is that denominator. “Overall” is `correct / 97`. Bold text marks the highest saved count in a column, including ties; it is not a claim of statistical significance.

| Method | Overall | EEM (20) | MQ (19) | TLA (20) | SUA (10) | IG (20) | MCD (8) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A-Mem | **43 / 97** | **12 / 20** | **15 / 19** | **6 / 20** | 5 / 10 | **4 / 20** | **1 / 8** |
| Long Context (fresh) | 32 / 97 | **12 / 20** | 8 / 19 | 4 / 20 | **6 / 10** | 1 / 20 | **1 / 8** |
| Embedding RAG | 31 / 97 | 8 / 20 | 11 / 19 | 5 / 20 | 3 / 10 | 3 / 20 | **1 / 8** |
| BM25 RAG | 29 / 97 | 8 / 20 | 14 / 19 | 2 / 20 | 4 / 10 | 1 / 20 | 0 / 8 |
| GraphRAG (recovered) | 21 / 97 | 9 / 20 | 7 / 19 | 3 / 20 | 2 / 10 | 0 / 20 | 0 / 8 |
| LightMem (fresh) | 19 / 97 | 3 / 20 | 13 / 19 | 2 / 20 | 0 / 10 | 1 / 20 | 0 / 8 |
| MemOS | 11 / 97 | 1 / 20 | 7 / 19 | 0 / 20 | 1 / 10 | 2 / 20 | 0 / 8 |
| Mem0 (fresh) | 14 / 97 | 1 / 20 | 11 / 19 | 0 / 20 | 1 / 10 | 1 / 20 | 0 / 8 |

### 3.1 Category-level conclusions

- **A-Mem is broad rather than narrowly specialized.** It is top or tied top in five of six families and has the clearest MQ lead (15/19). It does not lead SUA, where Long Context reaches 6/10.
- **Long Context favors direct fact/state access.** It ties A-Mem at EEM and leads SUA, but its IG performance is only 1/20 and it does not consistently meet the benchmark's strict causal-chain requirements.
- **Embedding RAG has the most balanced retrieval baseline.** It is second or close to second on MQ, TLA, and IG, and provides the only non-A-Mem unique wins in state update and one MCD query.
- **BM25 RAG is unusually good on constrained option selection (14/19 MQ) but does not carry that strength into TLA or MCD.** This pattern suggests its lexical retrieval is sufficient for some local decision questions but not for locating precise dates or assembling causal chains.
- **Fresh LightMem is similarly option-selection-heavy (13/19 MQ), but not broadly grounded.** It reaches only 3/20 EEM, 2/20 TLA, 0/10 SUA, 1/20 IG, and 0/8 MCD, so its 19 correct rows do not raise the temporal or causal ceiling.
- **Fresh Mem0 is also multiple-choice-skewed (11/19 MQ).** Its clean result replaces the 9/97 recovered diagnostic artifact, but it still has no correct TLA or MCD row despite returning five memories on every query. The new score therefore strengthens the diagnosis of a temporal/causal memory-grounding limitation rather than a resume artifact.
- **MCD is the universal failure mode.** Every method reaches at most 1/8 binary-correct. Partial MCD scores show more nuance, but they are far below a reliable reasoning threshold.

### 3.2 Multi-hop clinical deduction detail

The MCD judge records node coverage rate (NCR), causal-relation correctness (CRC), and chain completeness (CC). The reported score combines these with retrieval-quality adjustments, so a nonzero average score is not an accuracy claim.

Each row summarizes that method's eight MCD answers. “Correct / 8” is the strict binary result. The next four columns are averages on a 0–1 scale: the stored MCD score, followed by node coverage (NCR), causal-link correctness (CRC), and chain completeness (CC). “Rows judged patient-specific” counts how many of the eight rows the judge explicitly said used patient-specific memory; it is not a retrieval count.

| Method | Correct / 8 | Mean MCD score | Mean NCR | Mean CRC | Mean CC | Rows judged patient-specific |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A-Mem | 1 | **0.187** | **0.250** | **0.188** | **0.250** | 3 |
| Long Context | 1 | 0.162 | 0.188 | 0.156 | 0.194 | **4** |
| Embedding RAG | 1 | 0.091 | 0.125 | 0.125 | 0.125 | 3 |
| BM25 RAG | 0 | 0.013 | 0.031 | 0.031 | 0.063 | 2 |
| GraphRAG | 0 | 0.000 | 0.000 | 0.000 | 0.000 | 0 |
| LightMem (fresh) | 0 | 0.004 | 0.031 | 0.000 | 0.031 | 0 |
| MemOS | 0 | 0.020 | 0.063 | 0.031 | 0.063 | 1 |
| Mem0 (fresh) | 0 | 0.000 | 0.000 | 0.000 | 0.000 | 0 |

The judge demands patient-specific values, dates, mechanisms, and explicit causal links. Plausible generic medical advice is therefore insufficient. This makes MCD a useful memory-faithfulness test, but also makes it especially sensitive to the judge prompt and its interpretation of completeness.

## 4. Performance As History Grows

Each column is the evaluation made after the named number of sessions have entered the persona's memory: `S10` means after session 10, `S20` after session 20, and so on. Each cell is `correct / queries at that phase`; denominators vary because the dataset attaches different numbers of queries to different sessions. Read across a row to see whether a method degrades as the patient history grows.

| Method | S10 | S20 | S30 | S40 | S50 | S60 | S70 | S80 | S90 | S100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A-Mem | 5/9 | 7/11 | 5/9 | 4/11 | 7/9 | 3/11 | 2/9 | 4/10 | 4/9 | 2/9 |
| Long Context (fresh) | 5/9 | 7/11 | 5/9 | 1/11 | 3/9 | 3/11 | 1/9 | 2/10 | 3/9 | 2/9 |
| Embedding RAG | 4/9 | 4/11 | 5/9 | 5/11 | 5/9 | 1/11 | 1/9 | 2/10 | 3/9 | 1/9 |
| BM25 RAG | 4/9 | 2/11 | 5/9 | 3/11 | 3/9 | 3/11 | 3/9 | 2/10 | 2/9 | 2/9 |
| GraphRAG (recovered) | 1/9 | 3/11 | 3/9 | 3/11 | 3/9 | 1/11 | 2/9 | 1/10 | 2/9 | 2/9 |
| LightMem (fresh) | 3/9 | 4/11 | 3/9 | 2/11 | 1/9 | 1/11 | 1/9 | 2/10 | 2/9 | 0/9 |
| MemOS | 1/9 | 2/11 | 2/9 | 1/11 | 1/9 | 2/11 | 0/9 | 2/10 | 0/9 | 0/9 |
| Mem0 (fresh) | 3/9 | 2/11 | 2/9 | 0/11 | 1/9 | 2/11 | 0/9 | 1/10 | 3/9 | 0/9 |

The preceding phase table is valid GitHub-flavored Markdown: each header, separator, and row is on a separate line.

This second phase table collapses the first five phases (S10–S50, 49 queries) and the last five phases (S60–S100, 48 queries). The first two numeric columns show `correct / phase-group queries` and percentage; “change in correct rows” is the later correct count minus the earlier correct count, so a negative number means fewer successes later in the history.

| Method | S10-S50 | S60-S100 | Change in correct rows |
| --- | ---: | ---: | ---: |
| A-Mem | 28/49 (57.1%) | 15/48 (31.3%) | -13 |
| Long Context | 21/49 (42.9%) | 11/48 (22.9%) | -10 |
| Embedding RAG | 23/49 (46.9%) | 8/48 (16.7%) | -15 |
| BM25 RAG | 17/49 (34.7%) | 12/48 (25.0%) | -5 |
| GraphRAG | 13/49 (26.5%) | 8/48 (16.7%) | -5 |
| LightMem (fresh) | 13/49 (26.5%) | 6/48 (12.5%) | -7 |
| MemOS | 7/49 (14.3%) | 4/48 (8.3%) | -3 |
| Mem0 (fresh) | 8/49 (16.3%) | 6/48 (12.5%) | -2 |

The decline is not explained by one method alone. Even A-Mem loses 26 percentage points across the two halves. Embedding RAG loses 30 points despite returning exactly three memories for almost every query; LightMem loses 14 points despite returning 20 items on every query. This is consistent with a retrieval-selection and temporal-disambiguation bottleneck as the history becomes denser, but the single-persona design cannot establish the cause.

## 5. Retrieval Diagnostics

`retrieved_count` is method-specific telemetry, not a common quality scale: Long Context supplies the full history and reports zero retrieved items; A-Mem can use learned memories; RAG methods return passages; GraphRAG can return empty context despite reporting a five-item retrieval attempt.

All means in this table are across the 97 saved query rows. “Nonzero retrieval rows” says how often a method reported at least one retrieval item; “mean count, correct/incorrect rows” splits that same telemetry by outcome. The final column is an interpretation of the *count only*; it cannot tell whether a passage was relevant, factual, or used by the answer model.

| Method | Mean retrieved count | Nonzero retrieval rows | Mean count, correct rows | Mean count, incorrect rows | Interpretation |
| --- | ---: | ---: | ---: | ---: | --- |
| A-Mem | 3.76 | 73/97 | 2.67 | 4.63 | More returned memories did not imply correctness; selection/answer grounding matters. |
| Long Context | 0.00 | 0/97 | 0.00 | 0.00 | Full-context baseline; not comparable to retrieved-item methods. |
| Embedding RAG | 3.01 | 97/97 | 3.00 | 3.02 | Fixed top-k=3; count cannot explain its successes or failures. |
| BM25 RAG | 5.00 | 97/97 | 5.00 | 5.00 | Fixed top-k=5; lexical ranking/relevance is the relevant variable. |
| GraphRAG | 3.25 | 63/97 | 1.67 | 3.68 | Empty/failed context is common; count is inversely associated with correctness here. |
| LightMem (fresh) | 20.00 | 97/97 | 20.00 | 20.00 | Fixed top-k=20; count cannot distinguish evidence coverage from distractor overload. |
| MemOS | 2.47 | 48/97 | 1.82 | 2.56 | Partial retrieval availability, but low grounding quality. |
| Mem0 (fresh) | 5.00 | 97/97 | 5.00 | 5.00 | Fixed top-k=5; the cleaner run improves MQ but still lacks temporal and causal grounding. |

These are descriptive correlations, not causal effects. They do, however, rule out the simple explanation that “more memories retrieved” drove accuracy. Two illustrative patterns are repeated throughout the data:

- A-Mem retrieved five entries for `session_90_tla_1` and found the exact June 22 plantar-numbness event; Embedding RAG, BM25, MemOS, and Mem0 retrieved other plausible but temporally wrong events and all answered that the record was unavailable.
- BM25 RAG returned five passages for every question, including `session_40_sua_1`, yet its answer relied on generic HbA1c reasoning instead of the required two-week comparison. Embedding RAG's three passages supported the exact March worsening and was judged correct.
- LightMem returned 20 memories for both `session_20_tla_1` and `session_90_tla_1`, but still stated that the requested record was absent. A large fixed retrieval set therefore did not guarantee that the answer agent could identify the required temporally specific evidence.

## 6. Evidence From Representative Queries

The cases below are selected because they isolate a method advantage, a systematic failure, or a metric limitation. Output excerpts are shortened only for length; scores and outcome labels come from the saved `query_answer.json` rows.

### Case A: A-Mem wins an exact multi-hop causal chain

**Query:** `session_20_mcd_2` asks whether late-night high-carbohydrate delivery food could explain dry-mouth awakenings and elevated morning glucose.

**Required evidence:** approximately 11 mmol/L nocturnal rise, symptoms, late-night timing, stress/HPA-axis mechanism, several-day pattern, dawn-phenomenon/insulin-resistance implication, and monitoring advice.

This case table compares the same MCD question across methods. “Result” is the saved binary label plus its continuous MCD score, while “evidence in output/judge” identifies the decisive information the judge credited or found missing.

| Method | Result | Evidence in output/judge |
| --- | --- | --- |
| A-Mem | **Correct, 1.000** | Retrieves five indexed memories and explicitly connects late-night high-carbohydrate food, about 11 mmol/L glucose, thirst/dry mouth, and monitoring. The MCD judge gives NCR=1.00, CRC=1.00, CC=1.00. |
| Long Context | Incorrect, 0.124 | Reaches the right broad conclusion, but the judge finds NCR=0.25, CRC=0.00, CC=0.30 because the required HPA/cortisol mechanism and node-by-node chain are missing. |
| Embedding RAG | Incorrect, 0.000 | Retrieves relevant-looking dialogue but omits the required timing/physiology detail; all three MCD sub-scores are zero. |
| BM25, GraphRAG, LightMem, MemOS, Mem0 | Incorrect | Each fails to supply the complete patient-specific reference chain. |

This is the best evidence for A-Mem's advantage: it is not merely verbose. The judged distinction is the use of patient-specific numeric and temporal facts in the complete causal chain.

### Case B: Embedding RAG uniquely resolves medication-efficacy reasoning

**Query:** `session_20_mcd_1` asks whether post-lunch fatigue and head fullness mean oral treatment can no longer control glucose.

**Expected chain:** initial metformin + DPP-4 inhibitor response (HbA1c 9.2% to 8.1%), continued autoimmune beta-cell loss/C-peptide decline, weakening incretin effect, then worsening postprandial control.

This table again compares one shared MCD query. The “concrete difference” column is not a general method description: it names the particular evidence or causal link that caused that row to pass or fail.

| Method | Result | Concrete difference |
| --- | --- | --- |
| Embedding RAG | **Correct, 0.675** | Uses medication, HbA1c, and symptom timing; the judge credits node 1 and 3 and gives NCR=CRC=CC=0.75. |
| A-Mem | Incorrect, 0.289 | Mentions HbA1c and current symptoms but attributes the issue to insulin-adherence difficulty, missing the required beta-cell/C-peptide mechanism. |
| Long Context | Incorrect, 0.175 | Includes some numbers but misses the precise timeline and causal links. |
| BM25, GraphRAG, LightMem, MemOS, Mem0 | Incorrect, 0.000 | Mostly generic diabetes reasoning or unsupported patient facts. |

This is an important counterexample to the aggregate ranking: Embedding RAG can surface the right distributed evidence when the query matches its passages, while A-Mem can retrieve facts but assemble the wrong causal explanation.

### Case C: Long Context wins a precise temporal relation

**Query:** `session_20_tla_1` asks when milk coffee is consumed in the 2024-02-20 record.

**Gold:** immediately or within 1–2 minutes after finishing late-night takeaway.

Here “saved answer behavior” is a short paraphrase/quote of the actual answer. “Result” is the judge's binary outcome for this one temporal question, so the table shows why superficially related clock times were insufficient.

| Method | Saved answer behavior | Result |
| --- | --- | --- |
| Long Context | “while eating ... or immediately after finishing” | **Correct** |
| A-Mem | “around one or two in the morning” | Incorrect: gives clock time, not the required relation to the meal. |
| Embedding RAG | Says one or two sugary coffees during overtime | Incorrect: date/topic overlap but not the relation. |
| BM25 RAG | “late at night” | Incorrect: too coarse. |
| GraphRAG, Mem0 | Says relevant context is absent | Incorrect. |
| LightMem | “There is no record dated 2024-02-20” | Incorrect: its 20 retrieved memories still omit the requested relation. |
| MemOS | Hallucinates “around 10:30 AM” | Incorrect. |

The evidence favors full context for this question because the answer is a small relational phrase that can be lost when a retriever returns topic-relevant but insufficiently localized text.

### Case D: A-Mem retrieves a late, exact date that other methods miss

**Query:** `session_90_tla_1` asks for the date of a plantar-numbness event that lasted over ten minutes, improved after standing/walking, and resolved the next day.

**Gold:** 2024-06-22.

For this date-retrieval case, “evidence” identifies whether the method actually selected the June 22 record or instead selected a thematically related but wrong record. The table is about one query, not average retrieval quality.

| Method | Result | Evidence |
| --- | --- | --- |
| A-Mem | **Correct** | Its five retrieved memories include the 2024-06-22 dialogue and it returns `2024-06-22`. |
| Long Context | Incorrect | Says the exact event is not explicit and cites unrelated later context. |
| Embedding RAG / BM25 / MemOS / Mem0 | Incorrect | Each retrieves plausible foot-symptom or later-event text but concludes that the exact event is unavailable. |
| LightMem | Incorrect | Its 20 retrieved memories do not contain the requested event; it reports the closest related June 10 entries instead. |
| GraphRAG | Incorrect | Returns an empty context. |

This is a strong example of retrieval *precision*, not simply recall: all methods have relevant medical vocabulary, but only A-Mem selects the exact dated event.

### Case E: Embedding RAG handles a state comparison, while other answers drift

**Query:** `session_40_sua_1` asks how March brain fog changed relative to two weeks earlier.

**Gold:** it worsened, especially slower responses after waking.

The “failure or success mode” column explains the comparison anchor each answer used. A result is correct only when it answers the requested change relative to two weeks earlier, not merely when it mentions brain fog.

| Method | Result | Failure or success mode |
| --- | --- | --- |
| Embedding RAG | **Correct** | States that the patient was mentally foggy/out of it with slow reactions and explicitly concludes worsening. |
| A-Mem | Incorrect | Gives a detailed March 18 narrative but the judge considers it over-broad rather than the requested two-week comparison. |
| Long Context | Incorrect | Compares to an early-DKA episode instead of the stated two-week baseline. |
| BM25 RAG | Incorrect | Uses generic HbA1c/hyperglycemia language. |
| GraphRAG | Incorrect | States that symptoms are stable or eased, contradicting the target comparison. |
| LightMem | Incorrect | Describes a broad early-March decline but contrasts it with January rather than the required two-week baseline. |
| Mem0 (fresh) | Incorrect | Says symptoms recently worsened, but omits the required two-week comparison and slower waking; the judge rejects it as generic. |
| MemOS | Incorrect | States it lacks the dialogue records. |

The case shows that query framing matters: a response can be medically plausible and richly sourced yet still fail a state-transition question when it changes the comparison anchor.

### Case F: GraphRAG and LightMem solve a narrow option-set answer

**Query:** `session_40_mq_2` asks what to do when glucose rises; the gold option set is `B,D`.

This is an exact-option-set example. “Output” is the set of option letters emitted by the method, and “result” is correct only when that set equals the gold `B,D` exactly; omitted or extra letters both fail.

| Method | Output | Result |
| --- | --- | --- |
| GraphRAG | `B,D` | **Correct** |
| LightMem | `B,D` | **Correct** |
| A-Mem, Long Context, Embedding RAG | `D` | Incorrect: omits B. |
| BM25 RAG, MemOS | `C,D` | Incorrect: includes C and omits B. |
| Mem0 (fresh) | `C,D` | Incorrect: includes C and omits B. |

This is no longer a GraphRAG-only success: LightMem also returns the exact option set. It does not overturn either method's broader limitations, but it shows that both can satisfy strict option serialization on a suitably retrieved/local question.

### Case G: A universal multiple-choice failure reveals strict option-set scoring

**Query:** `session_20_mq_2` asks for a safer late-night takeaway strategy. The gold is `D` only.

All eight methods are scored wrong: A-Mem, Long Context, and Embedding RAG answer `A,B`; BM25 answers `A,B,D`; GraphRAG, LightMem, MemOS, and fresh Mem0 answer `A,B,C,D`.

The option metric requires an exact selected-option set, not merely inclusion of the correct option. This is a reasonable benchmark rule, but it means models that hedge by listing additional sensible actions receive zero. The output pattern suggests an instruction-following/calibration problem as much as a memory failure.

### Case H: The current EEM metric produces a documented false negative

**Query:** `session_40_eem_2` asks for the urine-ketone result on 2024-03-20. Gold answer: `++`.

This table is evidence of a scoring bug. “Saved output” is the literal answer text, and “saved score” is the repository's current EEM score. The point is that outputs visibly containing `++` still have score zero because the normalizer removes the symbol before matching.

| Method | Saved output | Saved score |
| --- | --- | ---: |
| A-Mem | `Urine ketones ++` | 0 |
| Long Context | `++` | 0 |
| Embedding RAG | `++` | 0 |
| BM25 RAG | `++` | 0 |
| GraphRAG | `(++)` | 0 |
| LightMem | `++` | 0 |
| MemOS | `Trace` | 0 |
| Mem0 (fresh) | `(++)` | 0 |

`metrics/string_match.py` removes ASCII punctuation before comparison. Both the output and gold `++` normalize to an empty string, and `StringContainMetric` intentionally refuses empty normalized answers. Consequently, seven literal/semantically exact answers are impossible to score correct under the present implementation.

**Sensitivity, not a saved-score change:** If this one gold answer is assessed as intended, A-Mem would be 44/97, Long Context 33/97, Embedding RAG 32/97, BM25 RAG 30/97, GraphRAG 22/97, LightMem 20/97, and fresh Mem0 15/97. This leaves the broad ranking unchanged but makes fresh Long Context and Embedding RAG exactly tied at 33/97.

### Case I: GraphRAG has real API-failure rows

The recovered GraphRAG result contains two literal Gemini `RESOURCE_EXHAUSTED`/429 outputs:

- `session_20_ig_1`
- `session_80_mq_2`

It also contains a blank response at `session_10_mq_1`. These are execution failures, not model-quality observations. They should be reported separately or retried; including them in a final model ranking biases GraphRAG downward.

## 7. Common Failure Analysis

All eight methods miss 42 queries. Their family distribution is revealing.

This table groups only the 42 universally missed queries. “All-method misses” is their count within each family; “fraction of family” uses that family's total question count as denominator; and “representative issue” gives the common failure pattern. It does not say that every question in a family has the same cause.

| Query family | All-method misses | Fraction of family | Representative issue |
| --- | ---: | ---: | --- |
| Inference generation | 14 | 14/20 | Patient-specific safe advice needs historical constraints and calibrated recommendations. |
| Temporal localization | 11 | 11/20 | Exact dates and within-event relations are often not retrieved or are replaced by a nearby event. |
| Entity exact match | 7 | 7/20 | Precise labs, medication names, and symptoms are frequently unavailable; one row is a metric bug. |
| MCD | 5 | 5/8 | The reference demands a full, explicitly grounded causal chain. |
| Multiple choice | 3 | 3/19 | Exact-option-set discipline is poor when several options sound reasonable. |
| State update | 2 | 2/10 | Best retained family, but comparison baselines still drift. |

Representative common misses include:

- `session_70_mq_1`: gold `B` (“take a short rest”), but systems over-medicalize the answer with glucose checks or dietary/insulin changes.
- `session_100_mq_2`: gold `A,B,C`; every system provides an incomplete or contradictory subset.
- `session_60_eem_1`: no method returns the vibration-perception threshold range.
- `session_40_mcd_1`: no method reproduces the required late-night intake → sympathetic/HPA activation → cortisol rhythm → hyperglycemia causal chain.

## 8. Method-Specific Interpretation

### A-Mem

Strength: best overall score, the only system with multiple exact patient-specific MCD/IG wins, and the strongest MQ result. Its success on `session_20_mcd_2` and `session_90_tla_1` shows it can retain/use a specific value or date from a long history.

Limitation: the average retrieved count is *higher on its wrong rows* (4.63) than its correct rows (2.67). That does not establish a causal defect, but it argues against interpreting A-Mem's advantage as simple “more memories.” Its `session_40_sua_1` response also shows that detailed recall can drift from the requested comparison frame. Its score remains recovered, not fresh.

### Long Context

Strength: excellent direct fact/state access without a retrieval cutoff; it wins the meal-timing relation and exact 15.1 mmol/L question, ties for best EEM, and leads state updates.

Limitation: sending the full history does not satisfy the strict causal-chain judge automatically. It misses the targeted late June event in `session_90_tla_1` and performs poorly on IG (1/20). The fresh result differs from the historical recovered Long Context result on 11 correctness labels (six old-only correct, five fresh-only correct), so a one-query difference should never be treated as a stable method effect.

### Embedding RAG

Strength: strongest non-A-Mem balance across retrieval-style methods. It uniquely solves medication-efficacy MCD reasoning and a state-update question, and has fixed top-k=3 retrieval across all 97 rows. Its successes demonstrate that a small retrieval set can be sufficient when it contains the right evidence.

Limitation: its 30-point early-to-late history drop is the largest among the top four. Fixed retrieval count hides whether the selected passages are temporally/causally appropriate; `session_90_tla_1` retrieves later foot-symptom discussions rather than the 2024-06-22 event.

### BM25 RAG

Strength: near-best multiple-choice performance (14/19) with much lower prompt-token usage than Long Context in the clean artifacts. This is a credible lexical baseline, not a trivial failure.

Limitation: fixed top-k=5 does not give it temporal precision or causal composition. Its response to `session_40_sua_1` substitutes generic glucose/HbA1c discussion for the required patient-specific change, and it has no correct MCD row.

### GraphRAG

Strength: it does produce the exact option set on `session_40_mq_2` (also solved by LightMem).

Limitation: the historical run is unusable for a final ranking due to duplicate-row recovery, two 429s, one blank answer, and many “context is empty” outputs. It has zero MCD score components across all eight MCD queries. Wait for the active fresh rerun before interpreting the method.

### LightMem

Strength: the fresh artifact has 97 unique rows and reaches 19/97, mainly through 13/19 exact multiple-choice answers. It also returns the exact `B,D` set on `session_40_mq_2`, and has two temporal-localization successes.

Limitation: it has no unique correct query, no state-update or MCD success, and reduces none of the 42 queries missed by all eight methods. It retrieves exactly 20 memories for every question, yet misses both representative temporal records and declines from 13/49 early-history successes to 6/48 late-history successes. The selected persona-1 configuration uses `pre_compress: true`, topic segmentation, and `BAAI/bge-small-zh-v1.5` embeddings on the English dataset; no embedding or retrieval-budget ablation isolates which component drives these outcomes.

### MemOS

Strength: it has a few correct MQ/IG rows and should not be treated as an execution failure; its artifact has 97 unique answers.

Limitation: low accuracy and repeated claims that no source record is available, even when the needed dialogue exists. In MCD case `session_20_mcd_1`, it explicitly substitutes hypothetical generic medical data; the judge correctly rejects this as not patient-specific.

### Mem0

Strength: the fresh run improves from the recovered 9/97 diagnostic result to 14/97, driven by 11/19 exact multiple-choice rows. It also has one patient-specific state-update win (`session_20_sua_1`) and one inference-generation win (`session_90_ig_2`). Unlike the historical artifact, it has exactly one answer row per query ID.

Limitation: every fresh row reports five retrieved memories, yet it has 0/20 temporal-localization and 0/8 MCD correctness. In `session_90_tla_1` it still cannot retrieve the exact June 22 event, and both representative MCD outputs substitute generic metabolic reasoning for required patient-specific values and causal links. The clean artifact establishes the low temporal/causal score as a method result for this configuration rather than a resume-duplication artifact.

## 9. Efficiency Evidence And Its Limits

The following figures are saved telemetry, not normalized pricing measurements. A-Mem and GraphRAG historical usage contains duplicate/retry overhead; GraphRAG also has concept-extraction work. The fresh Mem0 and LightMem artifacts have unique logical rows, but their Vertex batch final-query timing is recorded as zero, so they are not complete latency comparisons. The clean Long Context, Embedding RAG, and BM25 values are more directly comparable, but each remains a single run.

Each row reports the API usage recorded by that artifact: input and output token totals, total LLM call count, and logical correct rows after deduplication. The “caution” column is essential: token/call totals from recovered runs include extra historical work and cannot be treated as a fair efficiency comparison.

| Method | Input tokens | Output tokens | Calls | Logical correct rows | Caution |
| --- | ---: | ---: | ---: | ---: | --- |
| Long Context (fresh) | 9.69M | 31.0K | 155 | 32 | Full-history prompts dominate input use. |
| Embedding RAG | 1.08M | 27.6K | 155 | 31 | Similar score to Long Context with far fewer input tokens. |
| BM25 RAG | 0.34M | 25.1K | 155 | 29 | Lowest clean baseline input use. |
| A-Mem (recovered) | 18.14M | 241.0K | 922 | 43 | Not clean-cost comparable. |
| GraphRAG (recovered) | 64.47M | 1.14M | 3,235 | 21 | Includes extraction and failed calls; not comparable. |
| LightMem (fresh) | 1.14M | 219.1K | 350 | 19 | Clean rows; 4,490 s memory construction / 195 entries; batch final-query time is recorded as zero. |
| MemOS | 1.13M | 47.4K | 255 | 11 | Unique rows but low correctness. |
| Mem0 (fresh) | 1.43M | 151.3K | 355 | 14 | Clean rows; batch final-query wall time is not recorded. |

For the three clean baseline artifacts, the main cost finding is robust enough to state descriptively: Embedding RAG achieves 31 correct versus Long Context's 32 with roughly one ninth of the recorded input tokens, while BM25 RAG reaches 29 with roughly one thirtyth. LightMem's 1.14M input tokens are similar to Embedding RAG's, but its 19 correct rows require an additional 4,490 seconds of recorded memory construction. This is not a price comparison because Vertex pricing/model settings and batch overhead need to be accounted for separately.

## 10. Full Benchmark Evaluation Flow And LLM-Judge Contract

This section documents what the repository actually does for MedMemoryBench. It is included because interpretation of a score requires knowing what the answer model saw, what the judge saw, and which stages can fail independently.

### 10.1 End-to-end flow

```text
method YAML + dataset YAML + .env
        |
        v
load persona sessions and query metadata
        |
        v
form an evaluation unit every N regular sessions (N=10 here)
        |
        v
incrementally send only the new sessions to the method's memory builder
        |
        v
format the benchmark question and ask the method's answer model
        |
        v
route the answer to a deterministic metric or LLM judge by query type
        |
        v
aggregate scores and write result, memory-build, and query-answer JSON files
```

For the evaluated persona-1 configuration, this produces ten units at sessions 10, 20, ..., 100. The units contain 9, 11, 9, 11, 9, 11, 9, 10, 9, and 9 queries respectively, totaling 97. The current dataset configuration has `inject_noise: false`, so these are regular dialogue sessions rather than noise-augmented sessions.

### 10.2 Configuration resolution and data scope

1. `main.py` loads a method YAML and `configs/dataset_config/medmemorybench.yaml`. Method-level `dataset_overrides` are then applied before construction of the dataset; this is why the persona-1 method YAML can override the base dataset YAML's persona list.
2. The dataset loader reads `persona_{id}/eval/generated_dialogues.json` when noise is disabled, plus `generated_queries.json`. A dialogue session becomes a `MedSession` containing its date, Patient/Doctor messages, event metadata, and original session order. A query becomes a `MedQuery` containing its ID, question, type, session ID, answer options, correct answer(s), explanation, and metadata.
3. In independent mode, each persona receives a separate agent/memory context. Persona 1 cannot supply memory to persona 2. The evaluator creates a new agent when it switches personas, so scores from different personas are independent at the memory-state level.
4. Sessions are not re-injected from the beginning at every phase. The dataset accumulates a `pending_sessions` list until the tenth *regular* session, emits the unit, then clears the pending list. The agent itself retains the prior memory state. Thus S20 injects the next block of sessions into the already built persona-1 memory, rather than rebuilding S1-S20 from scratch.

### 10.3 Memory-construction stage

For every session in a unit, the evaluator first formats it as a dated medical dialogue with `Patient:` and `Doctor:` turns, then wraps it with the English memorization prompt: “Please read it carefully and memorize the key information.” It calls `AgentManager.send_message(..., memorizing=True)` once per session and stores the method's `MemoryBuildResult`.

The common evaluation contract is incremental, but the implementation differs by method:

This is an architecture map, not a performance table. The middle column says what state a method stores after a new session; the right column says what evidence is placed in front of the answer model for a later query. It explains why “memory” has a different technical meaning for Long Context, RAG, and agentic systems.

| Method family | What the method retains after each session | What the later answer prompt receives |
| --- | --- | --- |
| Long Context | Appended dialogue text, subject to `max_context_tokens` and `oldest_first` truncation | Entire retained context followed by the formatted question |
| Embedding RAG | Token chunks plus a rebuilt FAISS vector index | Local top-k dense-retrieved chunks followed by the formatted question |
| BM25 RAG | Token chunks plus a rebuilt BM25 index | Local top-k lexical chunks followed by the formatted question |
| Memory/agent methods | Method-specific notes, graph entries, vectors, or external memory state | Method-specific retrieved memory or internal agent context followed by the formatted question |

In the selected Long Context YAML, `max_context_tokens` is 100,000 and the truncation strategy is `oldest_first`. This means “Long Context” is not literally all 100 sessions if the accumulated dialogue exceeds that cap. A RAG method's `retrieved_count` is output telemetry; it does not prove that the retrieved material was relevant or used correctly.

### 10.4 Answer-generation stage

The answer model is given a question-specific English prompt and the memory context selected by the method. It does **not** receive the reference answer, correct option labels, answer explanation, reasoning-chain nodes, or judge metadata. This separation prevents direct label leakage from the evaluator into the answer model.

The prompt requirement changes by query type:

This table describes instructions given to the **answer model**, before scoring. The first column is the benchmark family, the second is the behavioral/format instruction, and the third says what kind of answer the later metric expects. It does not show the hidden reference answer, which the answer model never receives.

| Query type | Answer-model instruction | Expected answer style |
| --- | --- | --- |
| EEM | Return only the entity term(s). | Direct entity/fact |
| MQ | Return only all option letters, for example `B` or `B,D`. | Exact set of correct options |
| TLA | Return a date in `YYYY-MM-DD` when asked for time, or the event content when asked what happened at a time. | Exact time/event relation |
| SUA | Describe the patient's latest status and changes over time. | Patient-specific state transition |
| IG | Give concise patient-specific reasoning/advice rather than generic medical advice. | Grounded inference/recommendation |
| MCD | List remembered evidence, give a multi-step reasoning path, then a final judgment. | Patient-specific multi-hop causal explanation |

The answer response is saved with `query_id`, question, expected answer, model output, retrieval snippets/count, score, and evaluation details in `*_query_answer.json`. The saved expected answer is for auditability; it was not included in the answer-generation request.

### 10.5 Metric routing

The effective MedMemoryBench mapping in `MetricsCalculator` is fixed in code:

This routing table explains which component scores each family. “Count in persona 1” is the number of benchmark rows of that type; “LLM judge?” says whether a separate judge model, rather than local string/option logic, determines correctness for those rows.

| Query type | Count in persona 1 | Metric | LLM judge? |
| --- | ---: | --- | --- |
| EEM | 20 | `StringContainMetric` | No |
| MQ | 19 | `OptionMatchMetric` | No |
| TLA | 20 | `LLMJudgeMetric` | Yes |
| SUA | 10 | `LLMJudgeMetric` | Yes |
| IG | 20 | `LLMJudgeMetric` | Yes |
| MCD | 8 | `LLMJudgeMCDMetric` | Yes |

Therefore, 39/97 rows are locally scored and 58/97 rows require an LLM judge. The query-type `metric` fields in the dataset YAML describe the intended mapping, but the evaluator currently uses the calculator's built-in mapping shown above.

**Local metrics:**

- EEM lowercases, strips whitespace and punctuation, then checks whether every normalized expected answer occurs in the normalized output. This is the source of the documented `++` bug: a punctuation-only answer becomes empty and cannot match.
- MQ extracts option letters from the model output and requires the set to exactly equal the set of correct options. Extra options are wrong even when the correct option is present.

### 10.6 LLM judge: configuration and invocation

`LLMJudge` lazily creates a separate client when it first needs to judge a TLA, SUA, IG, or MCD row. Its configuration comes from the environment, not the method YAML:

The left column is the environment variable a user can set; the right column says the fallback used if it is blank. This table explains which model/provider is acting as judge, not which model generated the benchmark answer.

| Environment value | Resolution rule |
| --- | --- |
| `JUDGE_PROVIDER` | Provider; defaults to `openai` unless explicitly configured, for example `gemini`/Vertex |
| `JUDGE_MODEL` | Judge model; falls back to `DEFAULT_LLM_MODEL` when empty |
| `JUDGE_API_KEY` | Falls back to the normal provider API credential when empty |
| `JUDGE_BASE_URL` | Falls back to the normal provider base URL when empty |

The judge client is created with temperature 1.0. A real-time TLA/SUA/IG judge request has a 500-token response budget; MCD has 2,000 tokens. The judge sends a single user message containing the full evaluation prompt. For Gemini/Vertex providers it requests a JSON-object response MIME type. The code first attempts direct JSON parsing and then bracket-matching extraction; an unparseable or failed judge response becomes an incorrect result with reason `Judge failed`.

For reproducibility, note that `*_result.json` records the answer model configuration but does not embed the resolved `JUDGE_PROVIDER` or `JUDGE_MODEL`. In Vertex batch runs, the judge model is recorded in the local judge-batch manifest. Preserve the `.env` values and the manifest alongside any publishable result.

### 10.7 Exactly what the LLM judge sees

The judge sees the model's answer *and* hidden reference/evaluation material. This is correct for an evaluator but is fundamentally different from the answer-model prompt.

Each row is one judge prompt family. “Input fields” lists the information concatenated into that judge request; “key instruction” states the rule the judge is asked to enforce. This table is central for interpreting strict TLA/SUA/IG/MCD failures: the judge has more information than the answer model by design.

| Judge family | Input fields inserted into the judge prompt | Key instruction |
| --- | --- | --- |
| TLA | Question; reference answer; explanation attached to the correct answer; model answer | Strictly check the requested time point or event, while allowing equivalent date formats. |
| SUA | Question; reference answer; correct-answer explanation; model answer | Require the latest patient state and evidence that it comes from memory; generic lucky guesses are incorrect. |
| IG | TLA/SUA fields plus query metadata when present: inference type, trap mechanism, required patient information, common wrong answer, and why it is wrong | Require patient-specific reasoning and reject generic/common-wrong advice even if its conclusion is superficially plausible. |
| MCD | Question; reference answer; correct-answer explanation; full reasoning-chain nodes (node ID, source session, role, content/source); required-memory nodes; hop count; reasoning pattern; model answer | Validate each required node's specific data and causal link, then judge chain completeness and memory grounding. |

The answer explanation is selected from the first answer option marked `is_correct`. The MCD chain fields come from query metadata. If an answer is empty, the evaluator emits an immediate incorrect result and does not call the judge.

### 10.8 Judge output and score conversion

For TLA, SUA, and IG, the requested JSON is:

```json
{"is_correct": true, "reason": "brief justification"}
```

The evaluator ignores any model-supplied numeric score for these types and converts `is_correct` to a binary stored score: `true -> 1.0`, `false -> 0.0`. It saves the judge reason, correct-answer explanation, and metric name under `evaluation_details`.

MCD has a richer required JSON shape:

```json
{
  "node_validations": [
    {
      "node_id": 1,
      "mentioned": true,
      "specific_data_matched": true,
      "causal_link_correct": true,
      "note": "why this node does or does not match"
    }
  ],
  "ncr_score": 0.0,
  "crc_score": 0.0,
  "cc_score": 0.0,
  "memory_retrieval_quality": "excellent|good|partial|poor|none",
  "uses_patient_specific_info": true,
  "is_correct": false,
  "reason": "comprehensive justification"
}
```

The MCD binary label is strict: the judge template says a correct answer needs NCR >= 0.75, CRC >= 0.75, CC >= 0.7, and a reference-consistent conclusion. Independently of that label, the evaluator computes a continuous diagnostic score:

```text
base = 0.35 * NCR + 0.35 * CRC + 0.30 * CC
if not uses_patient_specific_info: base *= 0.5
score = base * retrieval_quality_multiplier
```

The retrieval-quality multiplier is 1.0 for `excellent`, 0.9 for `good`, 0.7 for `partial`, 0.4 for `poor`, 0.1 for `none`, and 0.5 for an unknown label. This explains why an MCD row can have a nonzero score while `is_correct` is false. All MCD fields, including node validations, are saved in `evaluation_details`.

### 10.9 Vertex batch behavior

With `--batch-api`, memory construction and local retrieval still occur in the local process because they are stateful/dependent. A method that supports batch final queries prepares the already-grounded final answer prompt locally, stores an immutable request snapshot, and submits only that final LLM call to Vertex. When the response returns, the evaluator applies the same scoring path described above.

For a Gemini/Vertex judge, the 58 judgeable rows are deferred and submitted as a final `judge-final` batch stage. Each request contains the exact judge prompt as one user message, uses temperature 1.0, sets JSON-object response format, and is correlated to `method:persona:query_id`. Local manifests persist the prepared judge inputs before submission so a valid `--resume` can collect the same job without regenerating model answers. If the configured judge cannot expose a Gemini batch client, the evaluator logs a warning and falls back to real-time judging.

### 10.10 Aggregation and output artifacts

After all rows are scored, `MetricsAggregator` calculates overall totals, correctness, mean score, query-type breakdowns, and timing. `ResultCollector` writes three timestamped files in the method/model output directory:

The filename suffix in the left column identifies the artifact produced for each run. “Contents” lists what is stored there, and “primary audit use” tells you which file to open for a particular question: headline score, memory-building trace, or individual model answers.

| File | Contents | Primary audit use |
| --- | --- | --- |
| `*_result.json` | Aggregate scores, query-type summary, efficiency, memory-build summary, token usage, saved method/dataset config | Headline numbers and run metadata |
| `*_memory_build.json` | Per-unit/per-session memory construction action, stored content, passages, entries, and errors | Diagnose what the memory method actually retained/built |
| `*_query_answer.json` | One row per evaluated query: question, expected answer, model output, retrieval trace/count, score, correctness, and evaluation details | Reproduce score analysis and inspect individual successes/failures |

Checkpoint state is written only for independent mode. It stores completed results and query IDs for recovery, but the historical resume-duplication issue documented in this report means a resumed result must be checked for exactly one row per `(persona_id, query_id)` before it is analyzed.

## 11. Validity Threats And Required Follow-Up

1. **Resume duplication remains a code-path risk.** The saved artifacts have been repaired, but this does not repair the checkpoint completion lookup. Avoid `--resume` when possible until a regression test proves that a resumed run keeps exactly 97 distinct query IDs per persona.
2. **Fix the punctuation-only EEM scorer before publication.** `normalize_text("++")` becomes an empty string. Preserve clinically meaningful symbols or special-case answers whose normalized form is empty; then rerun/re-score the affected EEM row transparently.
3. **Separate API failures from method failures.** The fresh GraphRAG run should retry/finish cleanly before score comparison. A 429 string and a blank answer should be counted as infrastructure failures in the audit, even if the benchmark summary records them as incorrect.
4. **Repeat with independent runs and more personas.** One persona and one generation/judge realization cannot support fine-grained claims such as “Long Context is better than Embedding RAG.” At minimum, run multiple seeds/configuration-stable repetitions and report mean, dispersion, and per-persona effects.
5. **Inspect judge calibration.** TLA, SUA, IG, and MCD rely on LLM judging. The cases above show it rejects broad but plausible advice when it lacks required patient-specific details; this may be intentional, but should be verified with blinded human adjudication on a stratified sample.
6. **Report logical score and actual cost separately.** The corrected 97-row summaries describe benchmark quality. Existing token/call logs describe what the historical process spent, including duplicated work. Combining the two produces misleading efficiency claims.

## 12. Observed Method Limitations And Core Directions

This section concerns limitations of the **methods as configured and observed in these runs**, not limitations of the experimental design. The run/evaluator caveats remain in section 11. These findings should not be generalized beyond the tested configurations, but they identify concrete failure modes an agent-memory system must address.

### 12.1 Method limitations observed in the results

1. **Methods do not reliably maintain a time-resolved patient state.** Temporal localization is weak across the board: the best observed score is A-Mem's 6/20, while fresh Mem0 and MemOS score 0/20. On `session_90_tla_1`, several methods retrieve medically related material but cannot surface the exact June 22 event; on `session_20_tla_1`, the required relation to the meal is replaced by a broad clock-time or “not available” answer. The observed memories behave more like topical records than versioned patient state.
2. **They fail to compose patient-specific causal memory into an explanation.** No method exceeds 1/8 strict MCD correctness. Fresh Mem0 has zero NCR, CRC, and CC across all eight rows even when it retrieves five memories per query; fresh LightMem has mean NCR/CC of only 0.031 and CRC of 0 despite retrieving 20. The representative MCD answers often offer sound generic diabetes advice but omit the required patient values, dates, mechanisms, and links. The limitation is not merely missing medical knowledge; it is failure to retrieve and connect a chain of patient-specific memory.
3. **Memory writing appears lossy for precise clinical qualifiers.** The failed EEM/TLA/MCD rows repeatedly depend on details such as a date, value, symptom duration, negation, comparison anchor, or medication-response trajectory. Extracted-memory methods (Mem0, MemOS, graph methods, and others) frequently return broad summaries or related facts instead. The artifacts cannot always isolate whether the detail was lost during writing or missed during recall, but either outcome means the current memory representation does not reliably preserve query-critical qualifiers with usable provenance.
4. **Selecting more memories does not solve the evidence-selection problem.** Fresh Mem0 retrieves exactly five memories on all 97 queries yet has 0/20 TLA and 0/8 MCD; LightMem retrieves exactly 20 and still has only 2/20 TLA and 0/8 MCD. BM25 likewise returns five passages on every row without temporal precision, while A-Mem's wrong answers contain more returned memories on average than its correct answers (4.63 versus 2.67). Fixed-size or relevance-only selection admits distractors and does not ensure coverage of the necessary event, state transition, or causal node.
5. **State updates are vulnerable to stale or misanchored recall.** Even answers that recognize the current symptom can fail to compare it to the requested historical baseline. In `session_40_sua_1`, Long Context compares with an unrelated early episode; A-Mem gives over-broad detail; fresh Mem0 says symptoms worsened but omits the required two-week comparison and slower waking. The systems lack an explicit update/supersession mechanism that makes “current,” “two weeks earlier,” and “historical but no longer current” distinct memory states.
6. **The final agent does not consistently convert retrieved evidence into the required answer form.** Exact-option MQ rows fail when a model adds medically plausible but unrequested options; `session_20_mq_2` is wrong for every method because all return a superset of the gold `D`. Conversely, fresh Mem0 reaches 11/19 MQ but still misses dates and causal tasks, showing that format discipline alone is insufficient. The agent needs separate control of evidence reasoning and final answer serialization.
7. **Complex agent-memory pipelines can be operationally fragile and expensive without corresponding gains.** The historical GraphRAG run has 429s, blank answers, and empty contexts. Fresh Mem0 completes cleanly, but its 1.43M input tokens and 355 calls yield 14/97; fresh LightMem spends 1.14M input tokens, 350 calls, and 4,490 seconds in memory construction for 19/97. Both trail the simpler clean baselines. More write-time extraction, graph construction, or memory management is not automatically useful; each added stage can introduce extraction error, latency, cost, and another point where patient evidence is lost.

### 12.2 Core Directions For An Agent-Memory System

The following are the few directions most likely to improve an *agent-memory* system on this benchmark. They focus on what the agent writes, maintains, selects, and reasons over--not on replacing it with a better RAG pipeline.

1. **Write faithful, provenance-preserving patient memories.** Convert each session into compact atomic memories carrying patient identity, source-session ID, event time, status/negation, uncertainty, quantities, and a pointer to the supporting text. Validate an extracted memory against its source before it is committed. This addresses the highest-leverage failure: once an agentic system drops a medication change, a negative finding, or a date during writing, no later retrieval policy can recover it.
2. **Maintain a temporal patient state, not an unordered memory collection.** Store updates as explicit state transitions: what changed, when it changed, what it supersedes, and what remains current. At query time, a time-specific question should resolve the state at that time; a latest-status question should resolve the newest non-superseded state while retaining earlier evidence needed to explain the change. This directly targets TLA/SUA errors caused by retrieving a true fact from the wrong phase of the patient history.
3. **Use typed memories and learned utility for selective recall.** Separate episodic events, durable clinical facts, procedures/preferences, and causal relations; score them by query relevance, recency, importance, and prior usefulness rather than only embedding similarity or fixed top-k. Then consolidate redundant memories while protecting rare, clinically consequential, and temporally specific facts. This is the core agent-memory hypothesis represented in methods such as A-Mem, MIRIX, MemRL, and MemOS.
4. **Make the answer agent reason over an explicit evidence set.** Retrieve a small, provenance-linked set of memories, assemble it into a patient timeline or causal path, and require the agent to check that every answer claim is supported before responding. For MCD, require the agent to connect the relevant events before drawing the conclusion; for EEM/MQ/TLA, enforce the public answer format. The check must see only the method's memories and question--never the gold answer, explanation, or judge metadata.
5. **Improve and tune the memory policy with paired error analysis.** On held-out personas, ablate memory granularity, retention/consolidation thresholds, memory-type budgets, temporal decay, and utility weights. For the same query IDs, label whether the failure was missing write, stale/conflicting state, wrong memory selection, or unsupported final reasoning. Optimize the policy that repairs the dominant failure class and verify it does not harm earlier-history or multi-hop rows.

Two prerequisites prevent false conclusions about these directions: repair the punctuation-only EEM scorer (`++`) and treat exhausted Gemini retries/blank outputs as separately rerunnable infrastructure failures. Those changes recover or correctly score existing rows, but they should be reported separately from genuine agent-memory improvements.

## 13. Bottom Line

For the completed evidence available now, the most defensible claims are:

- A-Mem has the best recovered logical score and the strongest evidence of patient-specific multi-hop reasoning, but needs a clean rerun.
- Fresh Long Context and Embedding RAG are a practical tie overall, with complementary strengths: Long Context handles direct fact/state access; Embedding RAG handles several targeted retrieval/state and causal cases more effectively.
- BM25 RAG is a strong low-token lexical baseline for multiple choice but weak on long-range temporal and causal tasks.
- Fresh LightMem reaches 19/97 through 13/19 multiple-choice answers, but its fixed 20-memory retrieval produces no unique wins, no state-update/MCD success, and a pronounced late-history decline.
- Fresh Mem0 replaces its recovered diagnostic score: it improves to 14/97, primarily through multiple choice, but remains at 0/20 TLA and 0/8 MCD. GraphRAG alone still needs its active fresh rerun before publication.
- The benchmark currently has at least one demonstrable EEM false negative (`++`), and the resume path has a known duplication hazard. Those issues should be fixed or explicitly disclosed before external reporting.
