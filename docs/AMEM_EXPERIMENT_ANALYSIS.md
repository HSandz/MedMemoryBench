# A-MEM Experiment Analysis

## Bottom Line

Current experiments do not prove a repeatable accuracy gain from Typed Relations, Temporal State, or Provenance. Provenance is reliable as an audit feature; Temporal State is promising but noisy; typed retrieval remains sensitive to stochastic construction and query generation.

## Key Results

The main comparison uses Persona 1, 50 sessions, 788 notes, and 49 scored queries. These runs are not fully causal because providers, independently rebuilt memories, expansion budgets, and generation temperatures differ.

| Variant | Score |
|---|---:|
| Original A-MEM (`amem_fix`) | 26/49 (53.06%) |
| Typed-only, same snapshot, query A | 24/49 (48.98%) |
| Typed-only, same snapshot, query B | 28/49 (57.14%) |
| Same answers rejudged | 26/49 (53.06%) |
| Typed + Provenance | 28/49 (57.14%) |
| Typed + Temporal | 26/49 (53.06%) |
| Typed + Temporal + Provenance | 25/49 (51.02%) |
| Original evolution + Provenance | 24/49 (48.98%) |

The same memory snapshot varied by four answers across query runs, and rejudging unchanged answers changed four labels. Small one- or two-answer differences are therefore not strong evidence of a method improvement.

## Feature Findings

### Typed Relations

The graph is structurally valid and usually connects a new note to its nearest candidate, but relation labels vary across independent builds. Expansion adds bounded graph neighbors; it does not replace semantic retrieval or guarantee query-relevant ordering. Reject malformed/confidence-zero edges rather than retaining them silently.

### Temporal State

Temporal state tracks current, historical, superseded, refined, and conflicting information. It activated on 32/49 queries in the initial analysis and added roughly 150 notes, but it missed 6/10 temporal-localization questions and sometimes parsed dates or multiple-choice distractors incorrectly. It currently relies on graph traversal rather than a direct timestamp index.

### Provenance

Provenance produced 788 evidence records for 788 notes with zero recorded errors and stable evidence IDs across independent builds. It improves traceability and debugging, not demonstrated answer accuracy. Raw-off runs injected no new raw evidence because the notes already contained the source turns; raw-on runs increase prompt cost and should be treated as a separate grounding experiment.

### Original Evolution and Links

Original A-MEM evolution is more expensive and retrieves a larger context through untyped links. It remains the strongest practical reference for `amem_test`, but provider and construction differences prevent a definitive causal ranking.

## Recommended Next Experiment

1. Build one typed-only snapshot and reuse it for every feature mode.
2. Cache one retrieval-keyword result per question, or disable keyword generation.
3. Compare typed-only, temporal annotation, temporal expansion, and provenance on identical semantic seeds.
4. Set answer, judge, and relation temperatures to 0 where supported.
5. Run at least three query repeats and report means, standard deviations, and per-query flips.
6. Fix question-stem-only temporal parsing, add direct timestamp retrieval, filter irrelevant expansions, and validate `SUPERSEDE` transitions more strictly.

## Source Artifacts

- Reference `amem_fix`: `outputs/amem_fix_gemini-2.5-flash/20260812_112716`
- Typed-only fixed snapshot: `outputs/amem_test_gemini-2.5-flash/20260813_143538`
- Combined Temporal + Provenance: `outputs/amem_test_gemini-2.5-flash/20260814_093118`
- Zero-temperature reruns: `outputs/amem_test_gemini-2.5-flash/20260816_014759` and later timestamped runs
- Implementations: `methods/amem_test_agent.py`, `methods/amem/A-mem/memory_layer_typed.py`
