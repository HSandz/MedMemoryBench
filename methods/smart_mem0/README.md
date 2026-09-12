# SmartMem0

SmartMem0 preserves durable memory and performs bounded, question-owned retrieval.
Domain vocabulary is data, never a routing rule. MedMemoryBench and LoCoMo are
evaluation adapters, not semantic authorities.

## Active READ

QuestionInput or raw question -> BaseWorld -> one LLM advisor -> additive retrieval
-> CandidateWorld (at most 16) -> read-only certificate -> projected answer OR
role-based context (at most 8) -> one answer LLM.

The advisor sees both question and BaseWorld. It cannot emit operations, budgets,
proof, or required slots. Up to two hints serve atomic queries; up to four distinct
premise hints serve synthesis. An optional atomic hypothesis is only a search hint.
BaseWorld is never evicted. BM25/dense ranks and lexical rarity are not proof.

## API

Use QuestionInput(text=..., candidates={label: proposition}, owner_id=...) when
candidate labels or ownership are supplied structurally. Raw strings remain supported
through a generic enumeration parser. Labels and query language are preserved.
selector_required is optional caller metadata, not an intent classifier.

## Code Owners

- read_advisory.py: question-owned acquisition and advisory normalization.
- read_evidence_certificate.py: stored-predicate binding, selectors and terminal.
- read_question_runtime.py: immutable-world check and role-based context.
- read_query_orchestrator.py: mandatory first call, conditional second call.
- read_structural_resolution.py: state heads, temporal axes and causal provenance.
- query_rendering.py / query_answer_runtime.py: rendering and answer lifecycle.
- query.py: compatibility facade only.

Historical planner/RequirementGraph modules are not in the active MRO. Current
architecture tests prohibit their direct import by active READ modules. The old
query implementation and previous specifications are under legacy/ and docs/archive/.

## Snapshot Compatibility

Shared identity and normalization now use memory schema 11. This invalidates prior
snapshot fingerprints deliberately; do not relabel an old snapshot as schema 11.
Capture/consolidation algorithms remain unchanged in this READ patch. Full WRITE
owner/prompt generalization is a separate phase. Existing memory may already have
lost qualifiers or acquired a fabricated owner; READ cannot repair that history.

## Verification

```sh
.venv/bin/python -m pytest -q tests/test_smart_mem0_question_runtime.py tests/test_smart_mem0_general_read.py
```

The local write lifecycle and snapshot suites were also checked. They currently
depend on additional local/ignored test helpers and are not the clean-checkout gate.

Offline tests establish contracts, not benchmark accuracy or multilingual embedding
quality. Embeddings remain configurable; no model default was changed. See
GENERALIZATION_PLAN.md for the evaluation protocol.
