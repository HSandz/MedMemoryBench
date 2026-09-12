# Active Retrieval Design

## 1. Question-Owned BaseWorld

The original user-visible question is the only mandatory acquisition input.

1. Run lexical retrieval.
2. Run dense retrieval.
3. Add strong literal surfaces such as quoted spans, numbers and corpus-distinct
   named anchors.
4. Fuse by deterministic reservation/round-robin.
5. Freeze BaseWorld at at most 10 memories.

BaseWorld exists before any LLM output. No LLM decision may evict it.

## 2. First LLM: Grounded ANSWER-or-SEARCH

The first LLM receives QUESTION, BaseWorld, valid stored relations among BaseWorld,
optional hard caller metadata, and non-factual caller formatting instructions.

It returns exactly one of:

- `ANSWER`: a complete final answer plus exact memory receipts
  `{memory_id, quote}`;
- `SEARCH`: at most four concise probes targeting missing evidence.

The controller performs semantic interpretation. It may understand paraphrases and
compose displayed premises. It may not invent factual premises from outside the
displayed evidence.

There is no `projection`, `OPTION_SET`, benchmark query type, answer hypothesis,
confidence threshold, planner, proof graph or semantic certificate in the active READ
contract.

## 3. Mechanical Grounding Guard

For `ANSWER`, code checks only facts it can verify exactly:

- memory id belongs to BaseWorld;
- quoted support occurs in the displayed memory;
- cited memory has linked EvidenceRecord provenance;
- explicit hard owner metadata, if supplied, matches.

The guard does not test semantic entailment and never uses lexical/embedding similarity
as proof. A failed guard falls back to the second reader; it does not mutate retrieval.

## 4. Bounded Expansion

For `SEARCH`, code executes each probe against the same lexical+dense retrieval stack.
Each probe may add at most a small number of novel memories. BaseWorld is preserved and
CandidateWorld is capped at 16.

If CandidateWorld is structurally empty, one recovery using the original question is
allowed. Low confidence, answer difficulty or a guard failure cannot trigger recursive
retrieval.

## 5. Fallback Context

CandidateWorld is passed to the second reader after deterministic duplicate removal
only. Code does not attempt semantic pruning with regexes, predicate overlap or
embedding thresholds. Valid stored relations among selected memories are rendered as
data.

The second reader receives a strict instruction to use retrieved memories and stored
relations as factual premises, while allowing normal language understanding, arithmetic
and grounded logical composition. If material evidence is still missing it must report
the gap rather than invent entity-specific facts.

## 6. Call Budget

Every query uses exactly one controller call. A second answer call is used only when:

- the controller returned `SEARCH`; or
- the controller returned `ANSWER` but its grounding receipt failed mechanical
  integrity checks.

Maximum READ LLM calls per query: 2.

## 7. Evaluation

Evaluate these separately:

- answer accuracy;
- early-answer coverage;
- early-answer precision / false-ANSWER rate;
- false-SEARCH rate;
- BaseWorld recall;
- CandidateWorld answer-bearing recall;
- BaseWorld retention (must be 100%);
- CandidateWorld size and expansion ratio;
- LLM calls/query;
- input/output tokens and latency.

Benchmark labels may be used for reporting slices only. They may not change the active
query pipeline.
