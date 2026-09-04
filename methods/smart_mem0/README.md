# SmartMem0 Module Map

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
