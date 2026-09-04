# MedMemoryBench Method Reference

This is a compact availability and comparison guide for the methods registered by `AgentManager`. It describes this checkout, not proof of upstream reproduction or performance.

## Common Evaluation Contract

```text
dialogue -> memorize() -> method memory state
question -> retrieval/tool loop -> answer prompt -> answer
answer + reference data -> local metric or LLM judge
```

The method controls the write path and read path. The answer model and judge model are separate configurations. `prepare_batch_query()`/`finalize_batch_query()` make only the immutable final answer stage batchable; stateful memory construction and retrieval remain local.

## Registry Status

| Method | Family | Adapter | Final-answer batch | Notes |
|---|---|---|---|---|
| `long_context` | Baseline | Available | Yes | Full retained history with truncation. |
| `embedding_rag` | Dense RAG | Available | Yes | FAISS/vector retrieval. |
| `event_state` | Agentic memory | Available | Yes | Immutable episodes plus versioned semantic claims; see `event_state_method.md`. |
| `bm25_rag` | Sparse RAG | Available | Yes | Lexical BM25 retrieval. |
| `graph_rag` | Graph RAG | Available | No | May batch internal concept extraction. |
| `amem` / `amem_fix` / `amem_test` | Agentic memory | Available | Yes | See `AMEM_IMPLEMENTATION.md`. |
| `letta` | Agentic memory | Available | No | Stateful tool/agent loop. |
| `memos` | Agentic memory | Available | Yes | Vendored memory backend. |
| `mirix` | Agentic memory | Available | Conditional | Native query mode is not batchable. |
| `mem0` | Agentic memory | Available | Yes | Extracted semantic memory. |
| `memrl` | Agentic/RL memory | Available | Yes | RL/value configuration affects behavior. |
| `zep` | Managed graph memory | Available | Yes | Requires Zep Cloud credentials. |
| `lightmem` | Agentic memory | Available | Yes | Extraction and index controls. |
| `remem` | Episodic graph | Available | No | Internal extraction can be batched. |
| `hipporag` | Graph RAG | Available | No | OpenIE, linking, and PPR retrieval. |
| `mem1` | Compact memory | Not available | No | Registry entry points to a missing adapter. |
| `q2q` | Q2Q benchmark | Not available | No | Registry-only placeholder with no YAML. |

Check the live registry with:

```bash
python main.py --list-agents
python main.py --list-methods
```

## Configuration and Fair Comparisons

Configs live under `configs/method_config/`. Top-level files select method/model combinations; `persona_1/` copies add `dataset_overrides.persona_ids: [1]`. Use the full relative name, for example:

```bash
python main.py -m persona_1/embedding_rag_gemini -d medmemorybench
```

For a fair comparison, pin the complete YAML, answer model, judge provider/model/settings, embedding model, dataset revision, persona selection, and session limit. Changing only the model name can also change chunking, retrieval count, internal extraction, or context limits.

## Method Selection Notes

- **Long context:** simplest control; retains history until its context policy removes older material.
- **Embedding RAG:** bounded dense retrieval; good for paraphrases but can miss exact terms or negations.
- **Event-State Hybrid:** preserves source episodes while compressing recurring/evolving claims; dense hybrid retrieval and optional query-only PPR.
- **Mixed MedMemoryBench runs:** methods receive opaque source UIDs and generic conversation scope only. The evaluator privately maps clean source UIDs to gold benchmark sessions; distractor UIDs never count as retrieval gold.
- **BM25 RAG:** deterministic lexical baseline; good for rare names, dates, and medications.
- **Graph methods:** add extraction, linking, graph traversal, or PPR; expect higher setup cost and more failure points.
- **Agentic memory:** rewrites or manages memory during ingestion and may use tools at query time.
- **Managed services:** Zep depends on remote state, credentials, service version, cost, and privacy settings.

Do not treat historical outputs as rankings. Inspect `retrieved_memories`, method logs, `run_config.json`, and judge configuration before attributing an error to retrieval or answer generation.
Mixed artifacts created before source-identity schema version 1 may contain
positional source-ID collisions and are not valid noise-robustness comparisons.

## Outputs and Evidence

Evaluation artifacts are written under `outputs/` and normally include run configuration, logs, memory-build records, query answers, retrieval traces, checkpoints, and optional batch manifests. `outputs/`, `logs/`, `results/`, and credentials are ignored by Git. Keep secrets and private data out of commits.
