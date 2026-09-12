# Active Retrieval Design

## Acquisition

1. Preserve the original question. Caller candidates take precedence over parsing.
2. Retrieve lexical and dense channels, reserving raw-query evidence before advisory.
3. Add quoted/numeric or corpus-distinct named surfaces for recall only.
4. Probe each candidate using the shared stem plus that candidate, not all options.
5. Assemble bounded BaseWorld (normally 10; at most 12 for option reservations).
6. Call the advisor once with question and BaseWorld.
7. Add at most two atomic or four synthesis hints, preserving BaseWorld. Physical
   limits belong to code. CandidateWorld is at most 16. A structural empty-world
   recovery is allowed once; a low certificate score cannot initiate retrieval.

## Certification

Freeze CandidateWorld. Validate explicit owner, provenance, directness, stance,
state status, stored predicate, projection and exact selected time axis. DF and
retrieval ranks are telemetry only. Unknown paraphrases or unresolved selectors use
synthesis rather than a guessed deterministic answer. Invalid selectors do not
remove evidence. Certificate cannot mutate or expand the world.

## Context and Answer

Unique scalar support can return its stored projection. Otherwise reserve independent
support/competition, one available premise per hint, option associations and channel
representatives. Requested CAUSES endpoints must both already be in CandidateWorld.
Deduplicate repeated propositions and stop when no reserved role adds evidence.
Eight is a ceiling, not a target. The final LLM evaluates relevance, missing premises,
polarity and inference; retrieval association never establishes candidate truth.

## Verification Boundaries

Strict terminal checks protect a bounded optimization; they do not prove arbitrary
natural-language entailment. Compare accuracy, answer-bearing recall, context survival,
call counts, token percentiles and latency separately using frozen query memories.
No benchmark scores or speed gains are claimed from offline contract tests.
