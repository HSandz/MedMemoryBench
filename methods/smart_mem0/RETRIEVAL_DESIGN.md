# Missing-Premise Grounded READ

SmartMem0 keeps the original question as retrieval authority and uses at most two READ LLM calls.

1. Build BaseWorld from raw lexical, dense and literal retrieval before any LLM output.
2. LLM #1 returns either a grounded ANSWER or SEARCH with `known_supports` and up to four distinct missing-premise `needs`.
3. SEARCH queries may add evidence but cannot evict BaseWorld or authorize truth. CandidateWorld remains capped at 16.
4. Preserve the mapping from each missing premise to the memories retrieved for it. LLM #2 receives known supports, evidence grouped by retrieval need, other acquired evidence and valid stored relations.
5. GroundingGuard is mechanical only: displayed memory id, exact quote, provenance and hard caller metadata. It never performs semantic entailment.
6. Stored memory is the sole authority for user/entity-specific historical facts. General domain knowledge may interpret, classify and connect grounded facts, but may not invent user-specific history.
7. The second reader must answer the requested attribute, distinguish observations from advice/plans/outcomes, adjudicate visible alternatives independently, preserve exact extractive surfaces, and ignore historical templates/scripts as instructions.
8. Benchmark QA wrappers are not retrieval input. SmartMem0 receives the user-visible question; formatting policy remains separate caller/system instructions.

No benchmark query type, medical route, language-specific semantic rule, OPTION_SET, projection type or semantic certificate belongs to active READ control.
