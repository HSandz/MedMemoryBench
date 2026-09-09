# SmartMem0 retrieval-fusion synthesis (2026-09-09)

## What the accumulated evaluations say

The frozen READ sequence moved from 22/49 to 24/49 and then to 27/49 after selector discipline,
non-temporal context-order repair, CandidateSet context conditioning, semantic closure, and
Evidence Resolve.  The latest frozen Stage-D result is 27/49 (EEM 9/10, TLA 8/10, SUA 4/5,
IG 3/10, MC 3/10, MCD 0/4).  Temporal/state behavior is therefore a capability to preserve,
not the next place to add complexity.

The same-backbone cross-method table adds a second signal: simple lexical retrieval remains
competitive on inference and substantially stronger on multi-option evidence discrimination,
while Event State has the strongest total score.  A-Mem is stronger on multi-option and
multi-hop but spends far more model work.  The combined conclusion is that SmartMem0 already
has enough semantic control; its main quality bottleneck is evidence placement and fusion.

## Highest-leverage failures

1. Evidence Resolve v1 treats obligation, family, resolved keys and whole-question views as
   symmetric candidate producers. Generic family/key hits can therefore enter the union and
   gain RRF score merely by appearing in several broad views.
2. CandidateSet has useful per-proposition probes, but the final local ordering can still favor
   a generic semantic neighbor over a literal proposition match.
3. Stored CAUSES relations are written and validated, yet POSSIBLE_CAUSE synthesis usually does
   not consume them. Final reasoning then falls back to a generic world-knowledge mechanism.
4. Complex ProofContext is often too lean, but increasing top-k globally would regress the
   efficient atomic/temporal paths. Extra context should come only from unmet evidence
   obligations or one authorized bridge neighbor.
5. WRITE/state simplification is still warranted later, but changing WRITE now would destroy
   attribution for the current READ bottleneck.

## Patch chosen for this commit

`ReadRetrievalFusionMixin` is an implementation-only refinement of Evidence Resolve. It does
not create a new semantic ontology, planner, route, or model stage.

- For non-temporal semantic SEARCH_FAMILY, the normalized participant-memory target is the
  primary candidate producer. Family, resolved-key and question/focus views become deterministic
  ranking signals over that same universe. A zero/semantically-weak primary may open exactly one
  deterministic rescue view; broad views never union extra rows into an already viable primary pool.
- Temporal/anchor retrieval is untouched.
- CandidateSet keeps its current bounded shared/proposition probes, then deterministically
  reserves the strongest proposition-affine local representative for each candidate.
- POSSIBLE_CAUSE may use one spare operation to follow one stored, provenance-valid CAUSES edge
  from an already grounded endpoint. The neighbor is admitted only after endpoint grounding,
  and the edge is materialized only when both endpoints survive final ProofContext.
- Certified Direct now falls back to Evidence Resolve when the chosen seed does not close the
  requested predicate, even if the candidate answer surface itself occurs in the same record.
- Final memory limits, provenance boundary, no-hit UNKNOWN semantics and the two-LLM-call lock
  remain unchanged.

## Acceptance telemetry for the next frozen run

- `retrieval_fusion.version = primary-recall-aux-rank-v1`.
- `auxiliary_searches_avoided > 0` on ordinary multi-view semantic requirements.
- `zero_hit_rescue_searches` should remain uncommon and bounded to one rescue per weak/empty primary requirement.
- TLA should not regress because temporal/anchor execution is unchanged.
- CandidateSet local order should expose literal/discriminative proposition evidence more often.
- `bridge_relation_count` and `stored_bridge_relation_materialized` should become non-zero only
  when a valid stored CAUSES edge is reachable and both endpoints survive ProofContext.
- Controller=1, middle stages=0, answer<=1; no WRITE change; no EFF/MIX semantic branch;
  boundary violations remain zero.

Do not change WRITE until this READ-only patch is rerun on the frozen EFF snapshot. After EFF
stabilizes, run the exact same commit in MIX; semantic behavior must remain identical.
