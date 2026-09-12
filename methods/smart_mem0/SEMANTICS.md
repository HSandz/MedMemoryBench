# SmartMem0 Semantics

## Constitution: General-Domain First

Architectural decisions are domain-neutral, benchmark-neutral and language-neutral at
the control level. The READ core has no MedMemoryBench, LoCoMo, multiple-choice,
medical, or language-specific route. Visible alternatives and numbered lists remain
ordinary question text. Hard caller metadata is explicit and separate from natural
language.

## Durable Records

MemoryAtom contains a stored assertion, owner, predicate, object, value, stance,
modality, temporal axes and linked EvidenceRecords. Source speaker is not fact owner.
StateIdentity is the normalized tuple (owner, scope, predicate, object). Missing
identity fields remain unknown. Typed relations remain ledger data. CAUSES requires
valid stored endpoints/provenance; time order alone is not causality.

## READ Semantic Boundary

The first LLM call is a grounded ANSWER-or-SEARCH controller. It receives the original
question plus BaseWorld memories and valid stored relations.

It owns semantic interpretation: paraphrases, reference resolution, comparison,
ordinary logical composition and the judgment of whether the displayed evidence is
sufficient. It may either:

- return ANSWER with the final response and exact support receipts; or
- return SEARCH with at most four concise probes for missing evidence.

The controller may not control physical retrieval, budgets, candidate eligibility,
stored relations, provenance, or memory mutation. SEARCH probes are additive retrieval
hints only.

Code does not attempt natural-language entailment with regexes, token overlap,
frequency, synonym tables, embedding thresholds, or benchmark-specific rules.
Lexical/dense scores are retrieval signals, never truth conditions.

## Grounding Integrity

There is no active semantic EvidenceCertificate in READ. Early stopping is permitted
only when the controller returns ANSWER and GroundingGuard mechanically validates:

- every cited memory id was in BaseWorld;
- every cited quote is an exact displayed memory span;
- every cited memory has linked stored provenance; and
- any caller-supplied hard owner constraint is respected.

GroundingGuard does not decide whether an answer is semantically correct. It cannot
retrieve, rerank, prune, mutate, or infer. Failure to pass the guard means only that
the one-call shortcut is unavailable; it is not evidence absence.

## Acquisition

The original question owns BaseWorld through independent lexical, dense and literal
retrieval rails. BaseWorld is created before LLM output and is never evicted by it.

If the controller returns SEARCH, code executes the bounded probes and unions novel
results with BaseWorld. CandidateWorld is at most 16 and BaseWorld is always a subset.
A single zero-hit recovery may rerun the original question when the world is empty.

The fallback reader receives acquired evidence after only deterministic duplicate
removal. Code does not perform semantic pruning.

## LLM Call Policy

There are at most two READ LLM calls:

1. grounded ANSWER-or-SEARCH controller;
2. final grounded reader only when the first call requests more evidence or its
   grounding receipt fails mechanical validation.

Any query may finish after one call, including inference or multi-premise questions,
when BaseWorld already contains sufficient grounded evidence. The system optimizes
precision of early ANSWER decisions before early-answer coverage.

## WRITE Scope

This revision changes the READ contract only. Capture/consolidation and snapshot
semantics remain a separate concern; READ changes must not claim WRITE generalization
that has not been implemented and evaluated.
