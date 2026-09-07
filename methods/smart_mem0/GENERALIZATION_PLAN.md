# SmartMem0 Generalization Plan

## Goal and non-goals

Primary development gate: reach at least **37/49 (75.5%)** on the first 49
MedMemoryBench queries while using the same generic READ architecture on other datasets.
The 49-query target is an acceptance test, not a routing taxonomy.

Non-goals:
- no `query_type` or MedMemoryBench/LoCoMo-specific branches;
- no per-language keyword routers;
- no return of planner/replanner/semantic-validator LLM stages;
- no retrieval-proximity truth labels for CandidateSet alternatives;
- no hidden graph expansion during final arbitration.

## Evidence from the Phase-1..6 results

The frozen-memory runs declined from 24/49 to 21/49 while the memory snapshot stayed fixed.
The largest persistent failures are inference, alternative selection and multi-hop clinical
deduction; entity lookup and temporal/state tasks remain materially stronger. Average final
retrieval breadth also fell after the lean retrieval migration. Therefore the next work
should improve evidence coverage and synthesis interfaces, not rebuild WRITE or add online
planning calls.

## Research-derived design principles

1. **Rich retrieval units, simple online control.** EMem and current Mem0 both support this
   direction: retain atomic, provenance-bearing units and keep online retrieval simple.
2. **Coarse recall needs a consistency/selection stage.** LightMem's two-stage retrieval and
   LongMemEval findings motivate query expansion and candidate filtering/reranking rather
   than blind top-k compression.
3. **Graphs are optional associativity tools.** HippoRAG supports bounded graph propagation
   for multi-hop recall, not universal graph traversal.
4. **Implicit constraints matter.** LoCoMo-Plus shows that later cues may be semantically
   disconnected from stored user constraints; narrow query-family strings are insufficient.
5. **Do not erase lexical/speaker structure.** GroupMemBench shows that even BM25 can beat
   sophisticated memory systems when ingestion loses lexical or speaker-grounded features.
6. **Production favors bounded, inspectable passes.** Keep one controller plus one optional
   answer call and make every final memory traceable to an authorized retrieval surface.

## Implementation stages

### Stage A — ontology lock (implemented in this change)
Define one public vocabulary: EvidenceRecord, MemoryAtom, EvidenceFamily, StateIdentity,
Projection, EvidenceRequirement, Constraint, Selector, CandidateSet, ReasoningBridge,
RetrievalView, Coverage, Certificate, ProofContext and Answerability. Legacy query modes,
slot types and reasoning labels remain compatibility internals only.

### Stage B — generalized evidence policy (implemented in this change)
- preserve the original-language question predicate in semantic retrieval;
- condition CandidateSet search on shared predicate + candidate;
- separate compute tier from structural context coverage floor;
- treat non-empty typed authorized support as retrieval coverage, not proof;
- prioritize temporal selector outputs in final context;
- materialize reasoning bridge endpoints/goals/permissions for LLM #2;
- prevent final arbitration from introducing new memory IDs.

### Stage C — controller under-decomposition audit (next only if telemetry proves necessary)
Measure how often answer-changing participant variables are absent from the controller IR.
If this remains a dominant error, refine the single controller contract so DERIVED
requirements encode generic decision-changing participant constraints. Do not add a second
semantic call and do not infer new semantic edges deterministically.

### Stage D — deterministic evidence consistency rerank
If Stage B recall is broad but noisy, add a bounded deterministic multi-signal rerank over
already authorized candidates using existing dense score, BM25, entity/owner match,
selector validity, provenance and state status. This must not make semantic truth claims and
must not add an LLM call.

### Stage E — representation improvements only if retrieval diagnostics require them
Audit whether missing facts are absent from MemoryAtoms versus merely not retrieved. Only
then consider richer event/entity linking or additional lexical aliases at WRITE time.
Preserve the one-call ADD-only capture/reconciliation direction.

## Evaluation matrix

Every candidate default should be evaluated on:

1. **MedMemoryBench frozen 49-query gate:** >=37/49, with per-family diagnostics.
2. **Fresh memory rebuild:** verifies the improvement is not snapshot-specific.
3. **LoCoMo:** factual, temporal, multi-hop and causal long conversation recall.
4. **LongMemEval:** extraction, multi-session reasoning, temporal reasoning, updates and
   abstention.
5. **Multilingual parity suite:** semantically equivalent queries in at least English,
   Vietnamese and one non-Latin-script language; compare retrieval IDs and answer quality.
6. **Multi-user/speaker suite:** ensure owner/speaker grounding survives ambiguous lexical
   overlap.

## Runtime gates

- method READ LLM calls <= 2/query;
- middle LLM calls = 0;
- `boundary_violation = false` for every query;
- CandidateSet empty evidence = UNKNOWN;
- stored `CAUSES` never inferred from temporal order or similarity;
- selector winner survives ProofContext when the selector is answer-bearing;
- final context is normally 4–7 memories for structured synthesis and may stay smaller for
  a validated atomic terminal answer;
- controller and answer token distributions are tracked separately.

## Rollback/iteration rule

A patch that increases the 49-query score by exploiting benchmark labels, English-only cue
rules, extra online LLM calls, or unauthorized context is rejected even if it crosses 75%.
If a generic change harms EEM/temporal/state substantially, revert or narrow it using only
semantic structure already present in the universal ontology.
