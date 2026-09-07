# SmartMem0 Evaluation Note — 2026-09-08

## Frozen EFF run after general-evidence-policy-v1

Run: `medmemorybench_smart_mem0_gemini-3.5-flash-lite_20260908_014025`

- 22/49 correct (44.9%), up only one question from the preceding 21/49 lean run.
- EEM: 10/10; temporal localization: 7/10; state update: 4/5.
- Inference generation: 0/10; multiple choice: 1/10; MCD: 0/4.
- Average final retrieved memories increased to 3.94/query.
- The two-stage invariant remained intact: one controller call and at most one answer call,
  with no middle planner/validator/replan LLM stages.

The breadth change therefore helped evidence availability but did not move the dominant hard
families.  The next bottleneck is semantic targeting and evidence discrimination, not another
blind context increase.

## Failure taxonomy

### 1. Controller under-decomposition

Repeated inference failures turn the requested conclusion into the memory lookup itself, for
example generic `medical_advice_or_action`, `clinical_assessment`, or monitoring-advice families.
For a decision query this retrieves memories that sound like advice while missing the concrete
participant variables that should control the decision: current trajectory, contraindications,
current regimen/prior outcome, or a relevant prior policy.

Causal/MCD queries show the same collapse when a cause/effect question becomes one generic
assessment requirement with no `POSSIBLE_CAUSE` or `INFER` bridge.  Critical longitudinal facts
are present in the frozen ledger, so WRITE is not the first component to change.

### 2. CandidateSet evidence aliasing

The v1 CandidateSet expansion conditioned every candidate search on the full shared predicate.
This makes global constraints or highly salient participant facts appear in several candidate
views simultaneously.  Candidate evidence then loses contrast: a contraindication can be shown
beside both the safe and unsafe alternatives even though retrieval proximity does not say which
one it supports.

The correct abstraction is two lanes:

- **shared predicate evidence** — participant facts relevant to evaluating every alternative;
- **candidate-local recall** — evidence retrieved specifically near one proposition.

A memory seen in several candidate probes is shared, not local.  Neither lane is a truth label;
the final model still evaluates each proposition under the question polarity.

### 3. Final mechanism under-specification

Even when the controller already produced grounded endpoints and an authorized `INFER` or
`POSSIBLE_CAUSE` bridge, MCD answers often stopped at vague labels such as stress, lifestyle,
adjustment, or metabolic changes.  The final bridge interface must require the explicit
intermediate cause/rule chain while preserving the invariant that participant-specific facts
come only from memory.

## Implemented next stage

### Answer-sensitive controller policy

Requirement-vNext-2 keeps the existing schema but tightens the semantic contract:

1. every requirement must be **MEMORY-VALUED** and **ANSWER-SENSITIVE**;
2. a requested decision/conclusion is the ANSWER, not a memory variable;
3. action/recommendation/monitoring questions retrieve the minimal decision-changing participant
   variables instead of a generic advice family;
4. causal/explanatory questions represent cause-side and effect-side participant evidence as
   separate requirements and use `POSSIBLE_CAUSE` unless an explicit stored causal edge is the
   object of the question;
5. longitudinal reasoning retrieves a prior baseline when comparison with current state changes
   the answer;
6. CandidateSet questions request shared/discriminative evidence variables, never one semantic
   requirement per visible option merely because it is visible.

No additional model call is introduced.

### Contrastive CandidateSet lanes

Candidate-local recall now uses proposition text without copying the full shared predicate into
every probe.  The existing SHARED_OPTIONS primitive retains a separate shared search.  Runtime
lineage classifies memories appearing in multiple candidate probes as shared and reserves local
representatives before shared noise consumes the final context budget.

LLM #2 receives claim-level shared/local evidence lanes rather than only memory IDs.  Empty local
recall is explicitly UNKNOWN, not false, and the prompt preserves question polarity (for example,
a contraindication excludes an option for a *safe/recommended* question but may identify it for
an *unsafe/forbidden* question).

### Bridge synthesis

For authorized `POSSIBLE_CAUSE`/`INFER` bridges, LLM #2 is instructed to name necessary
intermediate mechanisms and build the explicit chain from grounded source to grounded target or
ANSWER.  It may not invent participant history, and `CAUSES` remains restricted to explicit
stored causal relations.

## EFF / MIX mode invariant

EFF and MIX must not become separate semantic architectures.  Both use the same:

`Projection -> EvidenceRequirement -> Selector/CandidateSet -> ReasoningBridge -> RetrievalView
-> Coverage -> Certificate -> ProofContext -> Answerability`.

A mode may change the **eligible evidence pool or source composition**, but not the meaning of a
requirement, selector, bridge, candidate, proof certificate, or final-context boundary.  MIX may
augment EFF evidence with additional authorized source views, but it may not weaken provenance,
owner/speaker grounding, temporal-axis selection, state-head rules, or the rule that final context
contains only authorized seed/operation outputs.

There must be no controller prompt branch, CandidateSet truth rule, or retrieval operation chosen
because a mode label is `eff` or `mix`.

## Next measurement gate

Run the same frozen 49 EFF queries first.  Inspect:

- controller requirement count and bridge count on IG/MCD/MC;
- whether known critical ledger atoms now enter operation outputs and final context;
- CandidateSet local/shared overlap and option accuracy;
- MCD NCR/CRC/CC and explicit mechanism coverage;
- EEM/TLA/SUA regressions;
- two-stage call/token distribution and boundary violations.

Only after the EFF semantic ablation is understood should the same commit be evaluated in MIX.
If controller decomposition is now correct but relevant atoms still rank below the final context,
finish Stage D with a deterministic multi-signal rerank over already-authorized candidates.  If
relevant atoms reach ProofContext but the final answer still misses the chain, improve the bridge
synthesis interface/model compatibility rather than restoring a planner.
