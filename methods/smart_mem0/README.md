# SmartMem0 Module Map

## Active READ: Question-Authoritative Advisory Runtime

`read_question_runtime.py` is the sole active preparation owner, immediately after
the two-call auditor in the facade MRO. The older RequirementGraph mixins have
been removed from the facade MRO. Their standalone modules are retained only for
historical utility tests, not as public READ routes or certificate dependencies.

1. `read_advisory.py`: raw-question BM25 and dense rails, literal surface matches,
   and visibly enumerated proposition rails build BaseWorld **before** any LLM.
   Reserve two lexical and two dense hits plus proposition coverage, then fill.
   BaseWorld has at most 10 memories normally (12 for large explicit option sets).
2. One advisor call sees the question and BaseWorld. Only projection, hypothesis,
   exact question spans, selector and relation hints are accepted. All expansion
   hints share a two-hint budget, at most two novel candidates each. A short atomic
   ENTITY/VALUE/DATE hypothesis may add at most two candidates through one further
   search; all additions share the same world cap and never replace base evidence.
3. BaseWorld is retained unconditionally; CandidateWorld is capped at 16. Only an
   empty acquired world permits one deterministic recovery. A failed certificate
   never initiates retrieval. Relation hints do not traverse stored edges.
4. `read_evidence_certificate.py` inspects the frozen world once. Its independent
   identity contract binds exact question predicate/qualifier spans to stored
   assertions or durable predicates, not whole-question similarity. Names and
   values alone cannot certify a predicate. Unknown semantic paraphrases remain
   synthesis work; allergy is not silently rewritten as an avoidance instruction.
   Provenance, explicit temporal axis and durable state checks remain. It groups stored projections rather
   than trusting the advisor hypothesis. It returns supported-unique,
   supported-competing or insufficient; an unsupported hypothesis is reported
   separately. This is memory support, **not** proof of real-world truth.
5. A unique atomic supported projection can terminate without an answer call.
   Options, inference, comparison and ambiguity use one synthesis call with
   structured evidence, local evidence labels and valid stored relation topology.
   Relation hints alone cannot veto a unique atomic answer. Context limits are
   three for terminal atomic, five for competing atomic and eight for synthesis,
   increased up to eight when needed to retain distinct supported competitors.
   Nonterminal packing reserves one candidate per hint alongside raw rails.
   Named exact surfaces require corpus DF <= 20%; quotes and numeric surfaces
   are exempt. No language-specific stopword list is used for this policy.
   Hints are not facts, and
   option associations are not precomputed support/contradiction verdicts.

No WRITE schema or snapshot revision changes are introduced by this READ patch.
Reuse existing compatible memory snapshots, but start a **new query run**: old
prepared requests and old controller responses do not exercise this architecture.

Telemetry includes base/hint/final IDs, introduction provenance, retained-base
rate, certificate/terminal status, actual second-call use, tokens and boundary
audits. `selector_status` is NOT_REQUESTED, VALID or INVALID. An empty selector
is not invalid. `candidate_world_nonempty` replaces the misleading
`retrieval_complete = bool(world)`. `read_usage_contract.py` has one idempotent
accounting function shared by preparation, synchronous completion and external
batch completion: planned answer calls are never counted as executed calls.
The two-call ceiling concerns method invocations, not transport retries or judges.

Validation: the new offline suite covers the real facade and hybrid index with
a deterministic embedder, plus mocked-advisor terminal, ambiguity, provenance,
temporal-axis, multilingual rendering, recovery and monotonicity cases. It does
not establish benchmark accuracy, latency gains or terminal precision. The
predicate binding is intentionally conservative and exact; language-independent
control flow is not a claim of equal semantic recall across languages.

Historical certificate tests that expect controller `proof_spec`, mid-operation
STOP and the old retrieval fixtures are not the active READ contract. At the
pre-patch HEAD, 19 tests in `test_smart_mem0_read_certificates.py` already failed;
the old full-query fixture must also be migrated for the new preparation owner.
The replacement end-to-end call-budget and restored-terminal cases are in
`tests/test_smart_mem0_question_runtime.py`.

Removed obsolete files: `controller.py` (unused former controller), `router.py`
and `p1a_execution.py` (retired zero-call/English-regex routing), plus their retired
root-level `test_p1a_routing.py`. Historical experiment reports, frozen memories,
and compatibility modules still imported by other tests are not disposable data.

## Lean Migration: Capture Retention

Write schema 10 retains every provenance-valid captured atom in the HOT semantic
ledger, including recalled and temporary observations. Exact turns remain in the
COLD evidence archive. `facets` preserve timing, conditions and modality without
changing state identity; `atom_id` links capture to durable `atom_dispositions`.
Rejected capture items have an explicit reason rather than silently disappearing.
This guarantees retention of valid extracted atoms, not lossless LLM extraction.

Capture is the only write-model call per bounded window. Reconciliation no longer
calls a second LLM, guesses recap status from English source phrases, inherits
families by similarity, or suppresses recalled atoms. Only matching state identities
and facets can be versioned, and SUPERSEDE requires explicit event chronology.
Unknown chronology preserves competing values instead of substituting document time.
Explicit captured causal links retain the existing source-pointer validation.

Schema 9 snapshots are not a fresh-build evaluation of this contract. Schema 10
automatically creates a new snapshot generation. Reuse those same new snapshots
for subsequent query-only comparisons. Disposition auditing is transactional and
does not participate in query retrieval.

The lean roadmap is gated: compare identical query/session sets and models before
claiming Pareto improvements; evaluate retention and rebuild variance before removing
further compatibility layers. Current offline tests do not establish accuracy,
latency or stochastic rebuild-variance gains. The remaining controller, CandidateSet,
viability/terminal and context-owner migrations must be evaluated separately.

`methods.smart_mem0_agent` remains the compatibility import used by the benchmark.

## Evaluation memory snapshots

SmartMem0 evaluations persist the complete memory state after every evaluation
unit under `outputs/memory_snapshots/`. A normal rerun automatically restores the
exact snapshot matching dataset, method, model, config, context, unit content,
and parent-unit lineage before evaluating that unit's queries.

For MedMemoryBench, each later unit snapshot is cumulative for its persona even
though the evaluator only injects the new sessions at that boundary. LoCoMo uses
one isolated snapshot per sample. Snapshots are never shared across personas or
samples.

Capture/consolidation prompt hashes and `MEMORY_WRITE_SCHEMA_VERSION` are part of
the cache key, so a write-contract change automatically creates a new snapshot
generation. Use `--rebuild-memory` only to force a fresh stochastic rebuild under
the same contract. Query-only retrieval experiments omit the flag and reuse the
frozen unit memories for a fair ablation. `memory_snapshot_revision` remains a
manual escape hatch for an intentional cache generation change.
The implementation lives in this package and is split by runtime responsibility.

### Historical Module Map And Contracts

The following tables and READ descriptions document the pre-advisory architecture;
they are not the active READ contract. The active owners are specified above.

| Module | Owns |
| --- | --- |
| `agent.py` | Public `SmartMem0Agent` facade only |
| `contracts.py` | Dataclasses, enums, and validation constants |
| `prompts.py` | LLM input/output contracts |
| `core.py` | Runtime state, normalization, evidence staging, BM25/dense index |
| `capture.py` | Turn windows, MWC, and LLM memory extraction |
| `consolidation.py` | ADD-only commit, state heads, and typed relations |
| `write.py` | Session transaction, rollback, frozen-store lifecycle |
| `planning.py` | Hard constraints, initial recall helpers, and structural seed authorization |
| `retrieval.py` | Deterministic temporal, state, semantic, and causal operations |
| `execution.py` | Bounded operation execution, retrieval status, and pure arbitration |
| `query.py` | Final support boundary, context packing, telemetry, and answer request |
| `read_controller.py` | One-call seed-conditioned minimal semantic IR |
| `read_plan_contract.py` | Deterministic requirement-graph compiler |
| `read_execution_contract.py` | Requirement coverage and recovery rules |
| `read_option_contract.py` | Shared physical retrieval for visible options |
| `read_temporal_contract.py` | Explicit temporal parsing and axis-preserving filters |
| `read_usage_contract.py` | Method-local two-call accounting |

## Runtime Paths

```text
WRITE: capture -> consolidation -> transactional commit -> index refresh
READ: Dense + BM25 -> RRF Top-8 -> Top-3 seeds -> semantic controller
  candidate authorized -> validate exactly one atomic seed -> return grounded value
  otherwise -> compile requirement graph -> execute all bounded operations
            -> FOUND/EMPTY + PROVEN/UNPROVEN status
            -> pure arbitration -> evidence on demand -> context -> answer
```

The active semantic controller uses the same configured read client
(`gemini-3.5-flash-lite` in `configs/method_config/smart_mem0.yaml`) for one compact
JSON call. Its inputs are the original question, visible answer options, deterministic
syntax hints, and the Top-3 seed payloads. It returns only `answer_type`, question-owned
`focus_span` requirements, soft seed-conditioned `retrieval_hint` values, generic
relations between requirements, and an optional atomic candidate. It never emits a
route, query class, operator, budget, internal subject ID, store identity, evidence
role, or retrieval operation.

Relative chronology is represented by one generic requirement edge:
`TEMPORAL_ORDER(source, target, BEFORE|AFTER|OVERLAPS)`. The compiler resolves the
target event first and filters the source on `event_time` without an implicit fallback
axis. This edge expresses order only and can never create a `CAUSES` relation.

The question owns what must be answered. Seeds may suggest where evidence lives through
`retrieval_hint`, but that hint is never a hard filter or a fact. Code derives the route,
typed slots, budget tier, inference permission, and physical operations. It resolves
the participant identity, validates `$seed0..2`, hard query constraints, active/current
state, temporal axes, stored causal paths, and conflicts. Dense/BM25 scores and lexical
normalization are ranking aids only; they never prove semantic answerability. The active baseline
calls no planner LLM, semantic slot validator, repair LLM, or LLM replan.

Planned queries execute every operation in their bounded plan. Semantic requirements report
only `FOUND` or `EMPTY`; structurally decidable relations (`CAUSES`, `COMPARE`, and
`TEMPORAL_ORDER`) report `PROVEN` or `UNPROVEN`. `retrieval_complete` means every
requirement was found and every required structural relation was proven. It does not claim
that the final answer is correct. The final answer model remains the semantic authority.
`sufficient` is emitted only as a deprecated compatibility alias for
`retrieval_complete` and never controls execution or context enrichment.

Raw source turns are dereferenced only when `VERIFY_SOURCE` requests exact evidence, an
executed verification returns pointers, or a conflict remains unresolved. The final
answer may use a standard-domain bridge only when the requirement graph contains
`POSSIBLE_CAUSE` or `INFER`; that bridge may connect grounded participant facts but
cannot create participant history.

Only the semantic controller requests `VERIFY_SOURCE`; selecting `document_time`
does not automatically request raw turns. Intermediate `LOCATE` requirements remain
available as temporal anchors for other requirements, even for entity/value answers.
An unresolved question subject stays unconstrained; seed owners never become a hard
subject filter. Causal endpoints use generic `REQUIREMENT` slots, with stored causal
proof evaluated separately in relation status.

Supplementary candidates must come from trace entries whose `produces` includes the
same requirement. Their provenance records both the requirement and retrieval round,
including recovery rounds that reuse an operation index.

The baseline capture settings are `write_context_mode: none` and
`max_new_memories: 12`. Measurement families prefer the discriminative predicate
before a generic object, keeping fasting and postprandial readings separate.
Write schema 9 invalidates snapshots built with the earlier measurement identity.
Retained write fixes mean a fresh build is not a query-only ablation against old
results; a query-only comparison must use the same frozen unit memories.

Visible options are detected deterministically and use one `SHARED_OPTIONS` physical
operation with an independent probe per option. They are propositions, not semantic
memory requirements. Stored `CAUSES` traversal remains strict and requires relation
provenance; `POSSIBLE_CAUSE` instead retrieves grounded endpoints and explicitly
authorizes only the final general-domain bridge.

The default read path separates semantic authority from deterministic execution.
An authorized `candidate` is the atomic fast path; otherwise the compiler turns the
requirement graph directly into bounded operations. Deterministic recovery is allowed only
for `EMPTY` requirements or `UNPROVEN` structural relations; it removes soft resolver hints,
excludes already returned candidates, and never changes the question-owned requirement.
The normal upper bound is two method LLM calls per query:
controller plus answer, or just the controller when its atomic candidate passes
structural and grounding validation. Candidate grounding requires the answer to occur
in an atomic `value`, `verbatim_value`, or `object_anchor`; an entity appearing only
incidentally elsewhere in a rich seed cannot authorize the fast path.

The split follows the useful boundaries visible in the bundled implementations:
Mem0 separates memory/index backends, MemoRAG separates prompts and retrieval, and
LightMem separates construction from search. SmartMem0 keeps one stable benchmark
facade while applying those boundaries to its evidence and typed-slot architecture.

## Dependency Rule

Pipeline modules may call methods from modules to their right in `agent.py`'s mixin
order. They must not import the public facade. This keeps the compatibility layer
free of algorithmic behavior and avoids circular imports.
