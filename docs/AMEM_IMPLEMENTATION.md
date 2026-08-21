# A-MEM Implementation Reference

This is the concise, code-oriented guide to the A-MEM integrations in MedMemoryBench. The active adapter files are outside this directory; the vendored implementation is under `methods/amem/A-mem/`.

## Components

| Component | Role |
|---|---|
| `methods/amem_agent.py` | Shared adapter, embedding retriever setup, query answering, and snapshot serialization. |
| `methods/amem_fix_agent.py` | Paper-aligned atomic-note and linked-retrieval flow (`amem_fix`). |
| `methods/amem_test_agent.py` | Experimental typed-relation, temporal-state, and provenance flow (`amem_test`). |
| `methods/amem/A-mem/memory_layer_robust.py` | Active plain-text LLM memory layer used by `amem` and `amem_fix`. |
| `methods/amem/A-mem/memory_layer_typed.py` | Experimental layer used by `amem_test`. |
| `configs/method_config/*amem*.yaml` | Method and persona-specific configurations. |
| `tests/test_amem_*.py` | Adapter, feature, and snapshot regression coverage. |

The older `memory_layer.py` and standalone scripts in `methods/amem/A-mem/` support upstream/LoCoMo experiments; they are not the normal MedMemoryBench entry point.

## Runtime Pipeline

1. `AgentManager` selects `amem`, `amem_fix`, or `amem_test` from the YAML `method_name` and forwards `agent_params`.
2. Each persona/context gets an independent A-MEM system with a SentenceTransformer retriever and an internal metadata/evolution LLM.
3. During memorization, input is split at `amem_chunk_size_tokens` and passed to `add_note()`.
4. A-MEM analyzes each note, extracts metadata, and may evolve neighboring notes. The robust layer can make up to three conditional calls: evolution decision, strengthening, and neighbor update. `amem_evo_threshold` controls retriever consolidation.
5. At query time, the adapter retrieves semantic seeds, formats the notes into the answer prompt, and records retrieval metadata. `amem_fix` optionally generates keyword queries and expands untyped links.
6. `amem_test` can expand typed relations, select temporal states, and attach provenance evidence before final prompt construction. Typed-relation inference occurs during memory construction; query-time retrieval is local after optional keyword generation. Vertex batch applies to eligible final-answer generation.

Memory snapshots contain notes, retriever corpus/embeddings, IDs, configuration, and feature audits. Query-stage runs can reuse a completed memory snapshot with `--memory-run`; imports reject incompatible state.

## Snapshot and Append Lifecycle

Normal A-MEM memory builds publish one snapshot per evaluation unit under `<run>/memory/`. The manifest remains `building` until every expected unit has a snapshot, then becomes `complete`. Each snapshot records the source context, unit, build ID, config hash, memory construction time, and an integrity hash; embedding arrays are stored in adjacent `.embeddings.npy` sidecars.

Build telemetry is captured as a usage delta around each injected session and aggregated per unit and run. The tracker uses context-local phase and operation scopes so concurrent work cannot relabel another call. AMEM scopes distinguish note analysis, original evolution, chunking, embedding/indexing, typed candidate search and inference, temporal initialization/transitions, and provenance preprocessing/storage. Reports keep successful calls separate from attempts, failures, and retries. Snapshot publication also measures the compact serialized memory-state JSON and exact embedding-sidecar bytes. Unit metrics include cumulative size and signed growth; the overall footprint sums the latest unit for each context instead of all cumulative snapshots. Unit metrics are embedded in snapshots and summarized in the completed manifest, allowing memory-only, resume, append, and query-only workflows to expose the same source-build measurements. See [A-MEM Build Metrics](AMEM_BUILD_METRICS.md) for the serialized schema and reporting workflow.

An append creates a child run under `<source-run>/query_runs/<child-run>/`. It copies the source's contiguous snapshot prefix into the child, rewrites the snapshots to the child build ID, preserves source provenance, and builds only missing units through the inclusive `(persona, unit)` target. `--stage memory` stops after this publication. The default/all append restores both copied and newly built snapshots and evaluates the complete prefix, writing reports directly in the child run. A child manifest may contain a prefix of the full dataset and is therefore accepted by normal `--stage query` loading.

Append requires independent evaluation mode and an A-MEM method. The target uses the dataset's global `unit_id` plus its `context_id`/persona for validation. Source manifests may be `building` or `complete`; a building source must still contain a contiguous snapshot prefix. `--resume` reuses the child run and its checkpoint, including the small window between manifest publication and checkpoint creation.

## Variants and Features

| Variant | Write path | Query path |
|---|---|---|
| `amem` | Raw token-bounded chunks passed directly to `add_note()`. | Direct semantic retrieval. |
| `amem_fix` | One timestamped atomic note per dialogue turn; structured `memory_items` are preferred. | Keyword query -> semantic seeds -> one-hop untyped link expansion. |
| `amem_test` | Atomic notes plus optional typed relation/provenance metadata; original evolution is configurable. | `amem_fix` retrieval, then typed expansion and optional temporal/provenance context. |

Typed relation labels are `SUPPORT`, `REFINE`, `SUPERSEDE`, `CONFLICT`, and `RELATED`. Temporal state is derived from typed transitions and therefore requires `amem_typed_relations: true`. Provenance creates stable source evidence records; `amem_provenance_inject_raw_text` optionally adds selected source turns to the answer prompt.

## Configuration

Common fields:

```yaml
model:
  provider: gemini
  name: gemini-2.5-flash
  temperature: 0.0
  max_completion_tokens: 2000

build_config:
  amem_backend: vertex
  amem_model: gemini-2.5-flash
  amem_embedding_model: all-MiniLM-L6-v2
  amem_evo_threshold: 100
  amem_max_tokens: 1000
  amem_chunk_size_tokens: 10240
  amem_build_max_context_tokens: 200000

retrieval_config:
  retrieve_num: 10
  amem_max_context_tokens: 200000
```

Use `amem_fix_*` for the paper-aligned baseline and `amem_test_*` for experiments. Treat every experimental feature flag as authoritative: similarly named `amem_test2_*` files may intentionally represent different combinations. Build-time temporal transitions use `amem_temporal_transition_min_confidence`; query-time graph filtering uses `amem_relation_min_confidence`. Expansion counts, temporal ordering, provenance injection, keyword use, and retrieval context budgets can change without rebuilding.

Do not change build flags, the internal AMEM model, embedding/chunk settings, build prompt budget, or temporal transition threshold when loading a snapshot. Keep stable memory IDs, audit fields, and source evidence intact because resume, rejudge, and analysis depend on them.

## Verification

```bash
python -m pytest tests/test_amem_fix_agent.py tests/test_amem_test_agent.py tests/test_amem_staged_memory.py tests/test_build_feature_metrics.py
python main.py -m amem_fix_gemini -d medmemorybench --dry-run
python main.py -m persona_1/amem_fix_gemini -d medmemorybench --stage memory
python main.py --stage query --memory-run YYYYMMDD_HHMMSS
```

Prefer mocked provider tests. Update the A-MEM regression tests when changing note normalization, retrieval expansion, feature validation, serialization, or batch preparation.
