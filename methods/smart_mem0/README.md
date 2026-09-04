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
| `planning.py` | One-seed answerability gate, bounded planning context map, and validated retrieval plans |
| `retrieval.py` | Deterministic temporal, state, semantic, and causal operations |
| `execution.py` | Deterministic typed coverage, stopping, and pure arbitration |
| `query.py` | Final support boundary, context packing, telemetry, and answer request |
| `read_controller.py` | One-call semantic requirement and seed-coverage contract |
| `read_plan_contract.py` | Deterministic compilation of missing requirements into operations |
| `read_execution_contract.py` | Structural proof of controller-authorized seed coverage |

## Runtime Paths

```text
WRITE: capture -> consolidation -> transactional commit -> index refresh
READ: Dense + BM25 -> RRF Top-8 -> Top-3 seeds -> semantic controller
  ANSWER -> validate exactly one atomic seed -> return grounded value
  PLAN   -> validate covered requirements -> retrieve only missing requirements
         -> pure arbitration -> evidence on demand -> context -> answer
```

The active semantic controller uses the same configured read client
(`gemini-3.5-flash-lite` in `configs/method_config/smart_mem0.yaml`) for one compact
JSON call. Its inputs are the original question, visible answer options, deterministic
syntax hints, and the Top-3 seed payloads. It returns semantic fields only: route,
operator, answer slot, temporal contract, immutable evidence requirements, each
requirement's `COVERED|MISSING` state, seed references, raw-evidence need, and whether a
general-domain reasoning bridge is authorized. It never emits store identities or
retrieval operations.

Code is the deterministic authority. It validates `$seed0..2`, subject and hard query
constraints, active/current state, typed temporal axes, transition/causal relations,
and conflicts. Only validated covered requirements enter `seed_coverage`; only missing
requirements are compiled into local retrieval operations. A PLAN with complete seed
coverage and `operations=[]` is valid. Dense/BM25 scores remain ranking telemetry and
never prove sufficiency. The baseline calls no semantic slot validator, repair LLM, or
LLM replan.

Raw source turns are dereferenced only when the controller explicitly requests exact
evidence, an executed verification returns pointers, or a conflict remains unresolved.
The final answer may use standard domain knowledge only when
`world_knowledge_bridge=true`; that bridge can connect grounded participant facts but
cannot create new participant history.

The planning context map is built from compact write-time tags and salience metadata.
It gives the planner a bounded view of durable identity, trajectory, risk, constraint,
preference, resource, and plan anchors. Map entries are routing hints only: they have
no valid `$` reference and cannot enter final answer context without an explicit
retrieval operation.

Decision plans retrieve only roles that can change the answer. Enumerated-option
plans retrieve one shared evidence bundle of current facts, active guidance,
constraints, contraindications, preferences, and alternatives; distractor text is
never treated as a memory fact that must be retrieved. Causal plans retrieve grounded
endpoints and use stored `CAUSES` only when an explicit participant-specific edge
exists.

The default read path separates semantic authority from execution authority inside one
controller contract. `ANSWER` is the atomic fast path; `PLAN` is the optional retrieval
path. Deterministic recovery may broaden an uncovered target through explicit traced
operations without another LLM call. The normal upper bound is therefore two method
LLM calls per query: controller plus answer, or just the controller when its atomic
answer passes grounding validation.

The split follows the useful boundaries visible in the bundled implementations:
Mem0 separates memory/index backends, MemoRAG separates prompts and retrieval, and
LightMem separates construction from search. SmartMem0 keeps one stable benchmark
facade while applying those boundaries to its evidence and typed-slot architecture.

## Dependency Rule

Pipeline modules may call methods from modules to their right in `agent.py`'s mixin
order. They must not import the public facade. This keeps the compatibility layer
free of algorithmic behavior and avoids circular imports.
