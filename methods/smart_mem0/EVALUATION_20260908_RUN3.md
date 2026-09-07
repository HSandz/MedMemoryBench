# SmartMem0 EFF post-Stage-D evaluation (2026-09-08 02:45)

## Result

- 27/49 correct = 55.10%.
- EEM 9/10, TLA 8/10, SUA 4/5, IG 3/10, MC 3/10, MCD 0/4.
- MCD node causal rate increased to 12.5%.
- All five units were memory-snapshot hits, so this run isolates READ changes from WRITE rebuild variance.

## Delta versus previous frozen-snapshot run

Correctness gains with no correctness regressions:
- session_20_tla_2: 2024-03-02 -> 2024-03-01.
- session_20_ig_1: incorrect -> correct.
- session_40_mq_2: B -> B,D.

MCD session_20_mcd_2 partial score also improved from 0.1855 to 0.35.

## Remaining architectural failures

1. A one-obligation EARLIEST/LATEST plan can be compiled as SMALL and truncated to SEARCH_FAMILY only. SELECT is a deterministic transform, not another semantic retrieval decision, so this violates the locked ontology.
2. VERIFY_SOURCE can likewise be dropped by the same physical-operation cap.
3. Source-date filters may still be placed on event_time even when the controller emitted VERIFY_SOURCE; an exact source occurrence filter should use document_time when the date is a constraint rather than the requested DATE answer.
4. CandidateSet packet recall excludes Top-3 seeds to avoid duplication. This can discard an exact proposition-matching memory from final candidate evidence even though seeds are already inside the authorized evidence boundary.
5. Multi-hop answers are beginning to expose causal nodes but still stop at generic mechanisms; final synthesis needs a compact closure invariant, not another reasoning LLM.

## Patch acceptance invariants

- No WRITE change.
- No benchmark/query-type/mode branch.
- No extra LLM call.
- Keep semantic budget tier unchanged when only SELECT or VERIFY_SOURCE is appended.
- Exact candidate-seed association is retrieval lineage only, never SUPPORTS/CONTRADICTS truth.
- EFF and MIX share the same closure policy.
