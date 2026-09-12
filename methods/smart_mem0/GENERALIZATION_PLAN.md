# Generalization and Evaluation

## Design Target

MedMemoryBench and LoCoMo are regression suites, not design oracles. Use equivalent
invariants across personal memory, software/configuration, project state, preferences,
travel, accounts, documents, multiple owners and multilingual records.

## Current READ Gate

Offline tests exercise real Unicode tokenization, memory text, BM25/hybrid retrieval,
state resolution and certificates with a deterministic embedding stub. Include English,
Vietnamese, Japanese and Arabic predicates and values. Stubbed embeddings validate
control/lexical behavior only, not cross-language semantic retrieval quality.

Check: no rarity-as-proof; no span-count veto; no fabricated owner in shared identity;
invalid selector preserves evidence; four hints stay bounded and additive; structured
candidate labels survive; context has roles rather than quota fill; active READ cannot
import legacy execution/planning; at most two method LLM calls.

## Experiment Sequence

1. Offline current READ, write lifecycle and snapshot tests.
2. Query-only paired ablation on the same compatible frozen memory at each unit.
3. Report answer-bearing memory existence, retrieval recall and final-context survival
   separately from answer-model correctness, including previously correct questions.
4. Compare total/mean/median/P90/P95/max tokens, observed calls and latency.
5. Evaluate each supported embedding backend on same-language and cross-language
   queries before claiming multilingual dense retrieval. Do not silently swap models.

## Separate WRITE Phase

Do not combine capture/consolidation changes into this READ patch. Shared normalization
uses schema 11 and requires new snapshot fingerprints. Next, audit owner extraction,
recap handling, qualifier preservation and source attribution across domains; then
revise WRITE prompts/heuristics in a separate measured commit and rebuild memories.
Old schema memories must never be stamped compatible without migration/rebuild.

## Legacy

Historical architecture tests and reports remain useful records, not the current
specification. The active facade and direct import boundary are tested explicitly.
No performance milestone is considered achieved merely because unit tests pass.
