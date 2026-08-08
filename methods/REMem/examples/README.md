# ReMem Examples

This directory contains benchmark scripts and analysis tools for evaluating ReMEM on various long-context QA datasets.

## Directory Structure

```
examples/
├── Benchmark Scripts      # Main evaluation scripts for different datasets
│   ├── locomo.py          # LoCoMo dataset evaluation
│   ├── longmemeval.py     # LongMemEval dataset evaluation  
│   ├── semantic_qa.py     # Semantic QA benchmarks
│   ├── realtalk.py        # RealTalk conversation dataset
│   ├── complex_tr.py      # Complex temporal reasoning
│   ├── menatqa.py         # MenatQA evaluation
│   ├── timeqa.py          # TimeQA temporal queries
│   └── tot_semantic.py    # Tree-of-thought semantic reasoning
│
├── Analysis Tools         # Post-processing and analysis
│   ├── *_overall_eval.py  # Aggregate evaluation scripts
│   ├── *_questions.py     # Question extraction utilities
│   ├── analyze_*.py       # Result analysis tools
│   └── igraph_graph_*.py  # Graph visualization tools
│
└── openie/                # OpenIE preprocessing utilities
    └── longmemeval_*.py   # Dataset-specific extraction scripts
```

## Running Benchmarks

### Basic Usage

Most benchmark scripts follow a similar pattern. Example for LoCoMo:

```bash
python examples/locomo.py \
    --llm_name gpt-4o-mini \
    --embedding_name nvidia/NV-Embed-v2 \
    --extract_method episodic_gist
```

### Common Arguments

- `--llm_name`: LLM model for QA (default: `gpt-4o-mini`)
- `--embedding_name`: Embedding model (default: `nvidia/NV-Embed-v2`)
- `--extract_method`: Extraction strategy (`openie`, `episodic`, `episodic_gist`, `temporal`)
- `--llm_base_url`: Custom API endpoint (optional)
- `--force_index_from_scratch`: Rebuild index from scratch (default: `false`)
- `--force_openie_from_scratch`: Rerun information extraction (default: `false`)

### Dataset-Specific Examples

**LongMemEval:**
```bash
python examples/longmemeval.py --llm_name gpt-4o-mini --embedding_name nvidia/NV-Embed-v2
```

**Semantic QA (Multi-threaded):**
```bash
python examples/semantic_qa.py --llm_name gpt-4o-mini --num_threads 5
```

**TimeQA (Temporal reasoning):**
```bash
python examples/timeqa.py --llm_name gpt-4o-mini --extract_method temporal
```

## Analysis Tools

### Overall Evaluation

After running benchmarks, aggregate results:

```bash
python examples/locomo_overall_eval.py --working_dir outputs/locomo/
```

### Graph Visualization

Visualize the memory graph structure:

```bash
python examples/igraph_graph_visualization.py --graph_path outputs/sample/graph.pkl
```

## OpenIE Preprocessing

For batch information extraction on large corpora:

```bash
python examples/openie/longmemeval_openai_batch_ner_submission.py
python examples/openie/longmemeval_openai_batch_triplets_submission.py
```

These prepare OpenAI batch API requests for efficient extraction.

## Output Files

Benchmarks save results to `outputs/{dataset}/`:
- `results.json` - QA predictions and metrics
- `retrieval_results.json` - Retrieved passages per query
- `vdb_*.pkl` - Cached embeddings
- `graph.pkl` - Memory graph structure

## Notes

- First run will build index and extract information (slow)
- Subsequent runs reuse cached data unless `--force_*_from_scratch` flags are set
- Set `OPENAI_API_KEY` environment variable for OpenAI models
- For custom embedding servers, see `src/remem/embedding_model/*_server.py`