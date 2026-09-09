# SmartMem0 Progressive Evidence Retrieval

This READ-only layer implements one cross-benchmark retrieval rule:

> The controller understands the question once. Deterministic retrieval may inspect
> multiple authorized evidence views, build a bounded candidate world, then hand a
> small selected context to `ProofContext`.

The core is benchmark agnostic. LoCoMo and MedMemoryBench category names are evaluation
labels only and never route retrieval. The same primitives apply to both and to future
long-term-memory benchmarks: entity/subject, event, state, time, relation, proposition,
and provenance.

## Acquisition versus answer context

A weak lexical or dense match may enter the candidate pool, but it no longer closes
candidate acquisition. The obligation view is checked with deterministic retrieval
adequacy. If adequate, family/key/question views are restricted reranks over the same
IDs. If weak, auxiliary views may add at most two novel IDs each, within a bounded
candidate-world cap, and expansion stops as soon as the pool becomes adequate.

This separates two budgets:

- **candidate acquisition budget**: enough local breadth to avoid missing the relevant
  evidence;
- **ProofContext budget**: a later, much smaller evidence set shown to the answer model.

`ProofContext` remains the sole final-context owner and provenance boundary.

## HOT/COLD semantics

HOT is a storage/retrieval priority, not a relevance certificate. A merely related HOT
hit cannot block historical COLD evidence. COLD opens only when the HOT pool is not
retrieval-adequate, then HOT+COLD candidates are reranked together.

## Adequacy is not truth

The adequacy gate uses only generic retrieval signals: stronger lexical overlap, stronger
dense similarity, dense-score separation, or lexical+dense agreement. It never marks an
answer true, never turns a CandidateSet miss into false, and never substitutes for
structural requirement coverage or final semantic reasoning.

## Locked architecture invariants

- no benchmark/query-type/medical/language-specific routing;
- one semantic controller call plus at most one final answer call;
- Top-3 planning seeds remain unchanged;
- temporal/state selectors remain separate from family retrieval;
- stored relation expansion remains provenance-valid and at most one hop;
- EFF and MIX share semantic behavior; only the authorized source pool may differ;
- no WRITE-path change in this layer.

## Evaluation order

First rerun the same frozen EFF memory snapshot, then run the exact same commit in MIX.
Run LoCoMo with its categories only as diagnostics. A category-specific gain is useful
for analysis but must never become a retrieval route.
