# SmartMem0 frozen EFF run — 2026-09-08 02:15

## Result

Commit under test: `a9a72c874b6b5fdb392b13ba9dbbcaa345dd91fa`
(`Make SmartMem0 controller answer-sensitive and CandidateSet contrastive`).

The run reused the same frozen memory snapshots for all five units, so score movement is
attributable to READ behavior rather than WRITE/rebuild variance.

- Overall: **24/49 = 48.98%** (previous run: 22/49 = 44.90%)
- EEM: 9/10
- Temporal localization: 7/10
- State update: 4/5
- Inference generation: 2/10
- Multiple choice: 2/10
- Multi-hop clinical deduction: 0/4
- MCD node causal rate: 6.25% (previously 0%)
- Average retrieved memories: 4.18/query
- Average query time: 3.07 s
- Method READ calls remain controller + optional answer; middle LLM stages remain zero.

## Query-level delta versus the previous frozen EFF run

Newly correct:
- `session_10_tla_2`
- `session_10_ig_2`
- `session_20_ig_2`
- `session_50_mq_1`

Regressed:
- `session_10_eem_2`
- `session_20_tla_2`

Net: +2 correct.

## Diagnosis

### 1. Answer-sensitive decomposition is directionally correct

All inference-generation queries now receive at least one reasoning bridge and two inference
queries became correct. CandidateSet contrastive lanes also recovered `session_50_mq_1`.
MCD causal coverage moved above zero for the first time.

### 2. Selector overuse is now the largest controller regression

The controller attached temporal selectors to nearly every difficult requirement:
- IG: 17/18 requirements used `LATEST`
- MC: 12/14 requirements used `LATEST`
- MCD: 6/8 requirements used `LATEST`

`session_10_eem_2` demonstrates the failure. The atomic historical antibiotic-avoidance lookup
was rewritten as a `LATEST` medication-avoidance search, which displaced cefuroxime evidence
with newer NSAID/medication instructions.

Rule: selector default must be empty. `past`, `recent`, `previous` and `ongoing` are not
automatic extrema. Current-state identity belongs to `CURRENT`, while EARLIEST/LATEST must
reflect an answer-bearing temporal ordering requested by the question.

### 3. Final context ordering is wrong for non-temporal synthesis

`QueryMixin._context_time_axis()` currently returns `event_time` even when no temporal slot
exists. Therefore relevance/role ordering chosen by ProofContext is overwritten by chronology
before LLM #2. This is harmful for decisions, inference and MCD because older guidance can
appear before newer answer-changing evidence.

Rule: chronology is a requested projection, not a default context order. Preserve relevance
order unless an explicit temporal slot supplies exactly one requested axis.

### 4. Candidate-local probes became too context-free

Separating shared and candidate-local lanes fixed evidence aliasing, but proposition-only
queries are weak for short options such as drug names. Candidate probes should use
`shared predicate + proposition`; duplicate memories across probes are then classified as
shared rather than candidate-local. This restores contextual recall without reintroducing
truth labels.

### 5. Numeric temporal aliases still split across atom identities

`session_30_tla_2` asks when metformin 1500 mg/day started. The exact Jan-6 start atom and later
adherence atoms have different object/state identities. Family-only extremum recall can miss
the true earliest atom despite the shared numeric dose.

Rule: for temporal extrema with explicit numeric anchors, augment the family pool with a
bounded semantic numeric-alias pool. This expands the selector's candidate pool, not final
answer context.

## Next patch

Implement one Stage-D precision layer with no additional LLM calls:

1. selector-disciplined answer-sensitive controller policy;
2. relevance order for non-temporal answer context;
3. `shared predicate + proposition` candidate probes with existing shared/local dedupe;
4. numeric temporal alias backup for EARLIEST/LATEST;
5. compact Evidence Obligation Map for LLM #2, including exact selector semantics.

The same policy is mode-neutral. EFF and MIX must share the same controller, selector,
CandidateSet, bridge, ProofContext and answerability contracts; MIX may only widen the
authorized evidence pool/source composition.
