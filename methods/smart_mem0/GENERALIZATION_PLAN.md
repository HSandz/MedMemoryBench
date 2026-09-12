# Generalization and Evaluation

## Design Target

SmartMem0's READ path must be domain-neutral, benchmark-neutral and language-neutral at
the control level. MedMemoryBench, LoCoMo and other suites are regression/evaluation
sources, not routing taxonomies.

The core accepts natural-language question text plus optional hard caller metadata.
There are no benchmark query types, multiple-choice fields, medical routes, per-language
routers or answer-type-specific planners.

## Current READ Gate

The active query lifecycle is:

1. original-question lexical+dense+literal BaseWorld;
2. one grounded ANSWER-or-SEARCH LLM;
3. mechanical support-receipt validation for one-call answers;
4. bounded additive search when evidence is missing;
5. one grounded fallback answer call when needed.

The important separation is:

- lexical/dense/exact matching = retrieval/ranking;
- LLM = semantic interpretation, paraphrase understanding and grounded reasoning;
- code = budgets, provenance, hard metadata, relation validity and execution control.

Code must never promote lexical rarity, token overlap, regexes or embedding similarity
into semantic proof.

## Evaluation Sequence

1. Run offline active READ contracts.
2. Run query-only paired ablations on the same compatible frozen memory snapshot.
3. Measure BaseWorld and CandidateWorld answer-bearing recall separately from answer
   correctness.
4. Measure early-answer coverage and, more importantly, early-answer precision /
   false-ANSWER rate.
5. Measure false-SEARCH rate, calls/query, controller and answer token percentiles,
   latency and CandidateWorld expansion.
6. Repeat on multiple domains and languages. Benchmark labels may be reported as slices
   but may not change the pipeline.

## Required Regression Cases

Include at least:

- direct factual paraphrase;
- temporal/state questions;
- comparison/inference with all premises already in BaseWorld;
- missing-premise search expansion;
- conflicting stored evidence;
- raw visible alternatives without a special query type;
- vocatives/names that must not become owner constraints;
- explicit hard owner metadata that must be enforced;
- English, Vietnamese and non-Latin-script predicates/values.

## Separate WRITE Phase

Do not attribute READ gains or regressions to WRITE unless the snapshot changes.
Capture/consolidation, owner extraction, qualifier retention and state identity require
their own measured commit and snapshot generation. A query-phase experiment should
reuse a compatible frozen snapshot whenever possible.

## Acceptance Rule

Reject a patch even if benchmark accuracy rises when the improvement depends on:

- benchmark labels or hidden evaluator metadata;
- medical/domain-specific branches;
- English-only semantic cue lists;
- extra online LLM stages beyond the two-call ceiling;
- LLM-proposed memory IDs that were never retrieved;
- lexical/embedding similarity used as truth;
- pruning question-owned BaseWorld because of controller output.

Unit tests establish architecture invariants only. Benchmark accuracy and controller
calibration must be measured separately.
