# SmartMem0 Universal Semantics and Architecture Contract

This document defines the public concepts of SmartMem0 independently of any benchmark,
domain, language, model vendor, or answer format. Benchmark-specific labels are evaluation
metadata only and must never become routing rules.

## Design objective

SmartMem0 is an **evidence-grounded long-term memory layer**. Its online READ control plane
must stay small: one semantic normalization call plus, only when needed, one answer call.
Accuracy should come primarily from durable representation, multi-signal retrieval,
selector correctness, structural evidence coverage, and an explicit answer context—not from
online planner/replanner loops.

The architecture is evaluated against four simultaneous requirements:

1. **Benchmark generality** — factual recall, temporal localization, updates, implicit
   constraints, alternatives, multi-hop reasoning, source verification, and abstention.
2. **Language generality** — semantic decisions come from meaning, Unicode-normalized text,
   embeddings and the configured LLM; deterministic code must not maintain per-language
   cue-word routing tables.
3. **Production generality** — bounded online calls, deterministic provenance, explicit
   authorization boundaries, incremental updates, multi-user ownership, stable telemetry,
   and graceful retrieval failure.
4. **Representation fidelity** — preserve source evidence and atomic memories so later
   queries can reinterpret earlier information without irreversible task-specific summaries.

## Canonical memory-side concepts

### EvidenceRecord
An immutable provenance-bearing source unit (turn/span/document fragment). It is the source
of truth for wording, attribution and audit. EvidenceRecord is not injected into every query;
it is dereferenced when source wording or verification is required.

### MemoryAtom
An atomic normalized participant/world assertion derived from EvidenceRecord. A MemoryAtom
may be a FACT, EVENT, or STATE and carries participants, value, provenance, temporal fields,
stance/modality, and optional semantic role. It is the primary retrieval unit.

### EvidenceFamily
A **selector-neutral semantic recall neighborhood** for MemoryAtoms that express the same
answer-relevant variable or closely related variable. It is a soft retrieval hint—not a
fact, proof, durable state identity, query class, or temporal selector.

### StateIdentity / StateVersion / StateHead
StateIdentity is the durable identity of one versionable attribute owned by a subject and,
when needed, an object/scope. StateVersion is one chronological assertion about that
identity. StateHead is the active/current version determined from explicit chronology and
conflict rules. FACT/EVENT memories are not silently coerced into state versions.

### MemoryTier
HOT and COLD are retrieval/storage-priority tiers. Tier is never truth, confidence, or
semantic relevance. A COLD record may still be the correct answer and must remain reachable.

### StoredRelation
A provenance-backed relation between stored MemoryAtoms. Current durable relations include
SUPPORT, REFINE, SUPERSEDE, CONFLICT, RELATED, and CAUSES. Stored `CAUSES` is stronger than a
reasoning hypothesis: it must come from explicit stored evidence/provenance.

## Canonical query-side concepts

### Projection
The semantic shape of the requested result: ENTITY, VALUE, DATE, RELATIVE_TIME, OPTION_SET,
or TEXT. Projection controls projection/formatting only; it is not a query class, difficulty
label, retrieval budget, or reasoning route.

### EvidenceRequirement
One atomic participant-specific variable whose retrieved value can change the answer.
`QUESTION` requirements are explicitly requested by the question. `DERIVED` requirements
are additional participant-memory variables needed for the answer. General-domain rules and
mechanisms are never fake participant requirements.

### Constraint
A semantic restriction on an EvidenceRequirement (for example an attribute, condition,
participant or known temporal bound). Constraint narrows recall/selection but is not proof.

### Selector
A deterministic operator that chooses the answer-bearing member(s) from an already-retrieved
family. Temporal selectors include LOCATE, EXACT, EARLIEST, LATEST, BEFORE, AFTER and BETWEEN
on an explicit temporal axis. State resolution and exact source verification are also
selection concerns. **Family retrieval and selection are orthogonal.**

### CandidateSet
A set of alternative propositions evaluated under one shared question-owned predicate.
Candidates are not memory facts and are never classified true/false by retrieval proximity.
Each candidate may have one or more EvidenceViews. Empty evidence means UNKNOWN, never false.
Visible multiple-choice options are only one adapter into CandidateSet.

### ReasoningBridge
A relation that remains to be evaluated after its participant-specific endpoints are
grounded:

- **COMPARE** — compare grounded values; no causal claim.
- **DEPENDS_ON** — one answer-relevant variable depends on another; no causal claim.
- **TEMPORAL_ORDER** — order grounded endpoints; never implies causality.
- **CAUSES** — requires an explicit stored causal relation.
- **POSSIBLE_CAUSE** — grounded endpoints may be connected using standard domain knowledge.
- **INFER** — authorizes a standard-domain rule needed to derive the requested answer after
  grounded participant facts are available.

`CURRENT` and `VERIFY_SOURCE` remain compatibility encodings in the controller schema, but
conceptually they are unary selection/verification modifiers rather than reasoning edges.

### RetrievalView
A query-local, bounded set of MemoryAtoms returned by one authorized retrieval operation.
It records lineage (which requirement/candidate and retrieval round produced it). A
RetrievalView is a candidate evidence surface, not a semantic certificate.

### Coverage
A requirement/candidate has coverage when at least one structurally viable authorized
retrieval view exists. Coverage answers "do we have evidence to consider?" and never means
"the answer has been proved".

### Certificate
A conservative, exact structured proof that one stored value uniquely determines the
requested projection. Certificates may authorize deterministic terminal answers. Text
similarity, family labels, resolver keys, candidate coverage and confidence scores are not
certificates.

### ProofContext / AnswerContext
The one bounded authorized evidence bundle delivered to the answer model. It preserves
requirement lineage, selector winners, CandidateSet breadth, relevant constraints and
ReasoningBridge obligations. Final context IDs must be a subset of authorized Top-3 seeds or
explicit retrieval-operation outputs.

### Answerability
A deterministic state transition:
- `RECOVER` — a required evidence obligation has zero viable authorized candidates.
- `TERMINAL` — one strict certificate uniquely determines the answer.
- `SYNTHESIZE` — viable evidence exists but semantic synthesis remains for LLM #2.

### DirectCandidate
An optional one-seed fast path: exactly one Top-3 seed already contains the complete atomic
answer and independently passes structural authorization. Failure to authorize a candidate
always returns to the normal requirement/retrieval path; it never forces the query to stay
"direct".

### ComputeBudget vs CoverageFloor
ComputeBudget bounds physical retrieval work and is derived from evidence obligations.
CoverageFloor is the minimum number of authorized evidence views that must survive context
assembly for the structure already present (requirements, alternatives, bridge endpoints).
Candidate count may raise a coverage floor without becoming a query-difficulty class.

## Concepts that are implementation detail, not public semantics

Legacy `query_mode`, `reasoning_type`, `slot_type`, `planning_tag`, `evidence_role`,
SMALL/MEDIUM/LARGE labels, and legacy physical operation names are compiler/runtime details.
They may remain temporarily for compatibility but must not influence LLM #1 semantics or
become benchmark-specific routing branches.

The public READ vocabulary should converge on:

`Projection -> EvidenceRequirement -> Constraint/Selector -> RetrievalView -> Coverage ->`
`CandidateSet/ReasoningBridge -> Certificate (optional) -> ProofContext -> Answerability`.

## Comparison with representative systems

### Mem0 (production-oriented)
The April 2026 Mem0 architecture emphasizes single-pass ADD-only extraction, entity linking,
parallel semantic/BM25/entity signals, temporal ranking, and single-pass retrieval rather
than agentic retrieval loops. Its managed-platform benchmark numbers are not directly
comparable to this repository, but the architecture supports the principle that online
control flow should be simple while representation/retrieval is rich.

### EMem / EMem-G
EMem uses enriched event-like elementary discourse units with normalized participants,
time cues and source attribution. The simple variant is dense retrieval plus LLM filtering;
EMem-G adds graph propagation. The important lesson is that strong long-memory QA does not
require a large query taxonomy when the retrieval unit is semantically complete and evidence
selection is effective.

### LightMem
LightMem separates online retrieval from offline consolidation and uses coarse vector recall
followed by semantic-consistency reranking under a bounded budget. This motivates SmartMem0
to spend complexity on candidate quality/coverage instead of adding middle LLM planners.

### HippoRAG
HippoRAG demonstrates that graph propagation can improve associative multi-hop retrieval at
lower online cost than iterative retrieval loops. SmartMem0 should therefore keep bounded
stored-relation expansion as an optional primitive, not make every query a graph traversal.

### A-MEM, MIRIX and MemRL
These systems explore dynamic note linking/evolution, multiple memory types/managers, or
learned retrieval utility. They are valuable when self-evolution or heterogeneous agent
memory is the target, but copying their full control machinery into SmartMem0 would increase
online complexity before the current evidence-selection bottleneck is solved.

## Cross-benchmark implications

- **LoCoMo / LongMemEval:** preserve atomic facts, temporal axes, updates and multi-session
  evidence; use query/predicate expansion rather than only narrow family labels.
- **LoCoMo-Plus:** implicit user constraints may be semantically disconnected from the later
  cue; evidence selection must retain decision-changing context without task-type prompting.
- **GroupMemBench:** retain speaker/owner provenance and lexical evidence; ingestion must not
  erase surface terms that BM25 or attribution needs.
- **Multi-hop QA:** graph/bridge structure should connect already-grounded endpoints; general
  mechanisms belong to reasoning, not fabricated participant memories.
- **Multilingual:** keep original-language question predicates in retrieval queries; use
  Unicode normalization and model/embedding semantics rather than English cue routing.
- **Production:** enforce two READ LLM calls maximum, deterministic evidence boundaries,
  bounded coverage, immutable provenance, and inspectable telemetry.

## Architecture decision after the Phase-1..6 audit

SmartMem0 is currently **over-complex in vocabulary/control compatibility** and
**under-powered in evidence selection**. It should not restore planner/replanner LLMs. The
next correction is to keep the two-stage architecture but strengthen the evidence plane:

1. retain the original question predicate in semantic recall;
2. condition CandidateSet recall on `shared predicate + proposition`;
3. separate compute budget from structural context coverage;
4. prioritize selector outputs over their unfiltered family roots;
5. materialize ReasoningBridge endpoints/goals for LLM #2;
6. forbid arbitration from adding memory IDs outside the authorization boundary.

No rule above depends on MedMemoryBench query-type labels or on English keywords.

## Evaluation gates

The development target for the first 49 MedMemoryBench queries is **>= 75% accuracy
(>= 37/49)**, but that number is a gate, not permission to overfit. A patch is only considered
architecturally successful when it also preserves the following:

- no `query_type`/benchmark-specific READ branches;
- no per-language cue-word routing tables;
- at most one semantic-controller call plus one answer call;
- zero middle planner/replanner/validator LLM calls;
- zero final-memory authorization-boundary violations;
- no-hit CandidateSet evidence remains UNKNOWN;
- temporal selectors use the requested axis and selector winner;
- participant-specific claims remain provenance-grounded;
- changes are evaluated on at least one additional long-memory benchmark before being
  treated as a production default.
