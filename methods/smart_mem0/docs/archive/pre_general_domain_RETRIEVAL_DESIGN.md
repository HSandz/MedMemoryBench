# SmartMem0 Progressive Evidence Retrieval

This READ-only architecture follows one cross-benchmark rule:

> The controller understands the question once. Deterministic retrieval may inspect
> multiple authorized evidence views, build a bounded candidate world, then hand a
> small selected context to `ProofContext`.

The core is benchmark agnostic. Evaluation category names never route retrieval. The same
primitives apply across long-term-memory tasks: entity/subject, event, state, time, relation,
proposition, and provenance.

## 1. Premise-complete semantic control

The single controller still sees only the question and Top-3 planning seeds. It emits at most
four participant-memory requirements plus answer-relevant bridges.

For inference, explanation, comparison, decision, or cross-memory synthesis, the controller
must be **premise-complete but minimal**: it should include unmentioned participant-specific
variables whose values could materially change the conclusion, instead of stopping at only
the two endpoints named in the question.

General-domain mechanisms are never turned into fake memories. They remain `INFER` or
`POSSIBLE_CAUSE` bridges for the optional final answer model after participant-specific
premises have been grounded.

## 2. Acquisition versus answer context

Rank relevance and answer-bearing acquisition are different signals.

A weak lexical or dense match may enter the candidate pool, but it does not close acquisition.
The progressive rank gate still controls cheap first-pass breadth. On top of that, a direct
semantic requirement with structurally viable candidates but no conservative target proof is
kept open for bounded deterministic recovery.

On a target-proof miss, retrieval may:

- open up to two already-authorized auxiliary requirement views;
- add at most two novel candidates per view;
- inspect COLD evidence even if HOT was topically rank-adequate;
- remain under a hard candidate-world cap of 16.

This fixes the important distinction:

`topically strong candidate != answer-bearing target evidence`

and keeps two separate budgets:

- **candidate acquisition budget**: enough breadth to ground the required participant facts;
- **ProofContext budget**: a later, smaller evidence set shown to the answer model.

`ProofContext` remains the sole final-context owner and provenance boundary.

## 3. Retrieval closure is proof-gated, not truth-classified

For direct semantic `REQUIREMENT` / `COMPARAND` slots, structural `FOUND` is not allowed to
close deterministic acquisition when none of the selected supports passes conservative target
proof. Such a slot is returned to the existing deterministic recovery path as acquisition-
`EMPTY`.

This is **not** a SUPPORTS/CONTRADICTS truth classifier. Target proof only asks whether an
answer-bearing requirement is actually anchored to retrieved memory. CandidateSet lanes keep
`UNKNOWN_NOT_FALSE` semantics, and selector-driven TEMPORAL / CURRENT_STATE requirements keep
their own deterministic selector contracts rather than being replaced by lexical proof.

## 4. Source-neighborhood context envelope

When a requirement participates in an inference/causal/comparison/decision bridge, SmartMem0
may retain up to two provenance-local neighboring memories from the same capsule, falling back
to the same source session. Neighbors are owner/visibility checked and ordered by source
proximity.

This is contextual adjacency, not graph traversal. Stored relation expansion remains at most
one provenance-valid hop.

The goal is to preserve nearby participant premises that atomic extraction separated into
different memory objects, without broadening the final prompt indiscriminately.

## 5. HOT/COLD semantics

HOT is a storage/retrieval priority, not a relevance certificate. The normal progressive layer
opens COLD when HOT is rank-inadequate. The proof-gated layer may additionally inspect COLD on
a direct target-proof miss, which prevents a strong-but-wrong HOT decoy from permanently
hiding answer-bearing historical evidence.

## 6. Final reasoning synthesis

The architecture still uses at most one final answer-model call. For reasoning-sensitive plans,
the final instruction explicitly separates:

- participant-specific nodes, which must be grounded in retrieved memory; and
- general-domain bridge nodes, which may be supplied only when the controller authorized the
  bridge.

The answer model is told to use every material grounded participant premise, preserve relevant
chronology, and never replace a missing participant premise with generic knowledge.

## Locked architecture invariants

- no benchmark/query-type/medical/language-specific routing;
- one semantic controller call plus at most one final answer call;
- Top-3 planning seeds remain unchanged;
- no planner/replanner/critic/slot-validation LLM stage;
- CandidateSet remains proposition-oriented and no-hit means UNKNOWN;
- temporal/state selectors remain separate from family retrieval;
- stored relation expansion remains provenance-valid and at most one hop;
- EFF and MIX share semantic behavior; only the authorized source pool may differ;
- candidate acquisition may be wider than final ProofContext;
- no WRITE-path change in this layer.

## Evaluation order

First rerun the same frozen memory snapshot against the previous READ commit, then evaluate the
same new commit under the alternate authorized evidence pool. Inspect both accuracy and the new
telemetry: proof-gate misses, proof-gated requirements, auxiliary openings/additions, COLD proof
expansions, source-neighbor additions, candidate-world peak, and final LLM-call count.

Category metrics are diagnostics only; they must never become routing conditions or special-case
retrieval logic.
