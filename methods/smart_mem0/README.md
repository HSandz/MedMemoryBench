# SmartMem0

SmartMem0 preserves durable memory and performs bounded, question-owned retrieval.
Domain vocabulary and benchmark labels are data, never routing rules. MedMemoryBench,
LoCoMo and other suites are evaluation adapters, not semantic authorities.

## Active READ

```text
question
  -> lexical + dense + literal BaseWorld (<=10)
  -> LLM #1 grounded ANSWER-or-SEARCH
       -> ANSWER + exact support receipts
            -> mechanical GroundingGuard
            -> return immediately when valid
       -> SEARCH (<=4 probes), or invalid answer receipt
            -> additive retrieval
            -> CandidateWorld (<=16)
            -> deterministic duplicate removal only
            -> LLM #2 grounded final answer
```

The first LLM is both the semantic reader and the adaptive retrieval controller. It may
interpret paraphrases, compare memories and combine displayed premises. It may not
control retrieval budgets, memory eligibility, provenance, stored relations or memory
mutation.

Code deliberately does not attempt natural-language entailment. Regexes, lexical
overlap, corpus frequency and embedding similarity may support retrieval/ranking but
never prove an answer.

## Query Contract

The primary input is natural-language text. `QuestionInput` adds only optional hard
caller metadata:

```python
QuestionInput(
    text="Where is Alice based?",
    hard_metadata={"owner_id": "alice"},
)
```

There is no core field for multiple-choice candidates, benchmark category, projection,
temporal query type or reasoning type. If alternatives are visible to the user, they
remain ordinary question text and are interpreted by the same controller.

Hard metadata is authoritative only when the caller genuinely knows it. A name or
vocative appearing in natural language (for example, `Doctor, ...`) is not converted
into an owner constraint by regex.

## GroundingGuard

`GroundingGuard` is an integrity check, not a semantic certificate. For a one-call
answer it verifies only mechanically checkable facts:

- cited memory ids were in BaseWorld;
- cited quotes occur in those displayed memories;
- cited memories have linked evidence provenance;
- explicit hard owner metadata is respected.

If the guard fails, SmartMem0 simply loses the one-call optimization and uses the
fallback reader. Guard failure does not mean the answer is semantically false or that
evidence is absent.

## Code Owners

- `read_advisory.py`: BaseWorld acquisition, ANSWER-or-SEARCH controller, mechanical
  grounding receipt validation and bounded additive search.
- `read_question_runtime.py`: active READ lifecycle, CandidateWorld invariants,
  fallback evidence rendering and telemetry.
- `read_query_orchestrator.py`: maximum-two-call enforcement.
- `read_structural_resolution.py`: structural state/time/relation utilities.
- `query_rendering.py` / `query_answer_runtime.py`: evidence rendering and answer
  lifecycle.
- `question_input.py`: text plus optional hard caller metadata.
- `query.py`: compatibility facade only.

`read_evidence_certificate.py` is not imported by the active READ runtime. It remains
only as historical compatibility code until all external consumers have migrated.

## Snapshot Compatibility

This READ revision does not intentionally change the WRITE schema or rebuild policy.
Capture/consolidation quality remains a separate evaluation axis. READ experiments
should reuse a compatible frozen snapshot when measuring query-phase changes.

## Verification

```sh
.venv/bin/python -m pytest -q tests/test_smart_mem0_question_runtime.py tests/test_smart_mem0_general_read.py
```

Offline tests establish architecture and grounding invariants, not benchmark accuracy.
For benchmark runs, report overall accuracy together with early-answer precision,
early-answer coverage, false-ANSWER rate, false-SEARCH rate, answer-bearing retrieval
recall, calls/query, token use and latency.
