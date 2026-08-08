# REMem

## What is REMem?

REMem (**Re**asoning with **E**pisodic **Mem**ory) is a retrieval-augmented generation system that organizes documents into a hybrid memory graph of entities, facts, and episodic gist traces. It combines dense retrieval with graph-based exploration to answer complex, multi-hop, and temporal questions over long-form text. See our [ICLR 2026 paper](https://www.arxiv.org/pdf/2602.13530) for details.

## Setup

### Installation

Install the package in editable mode with dependencies:

```bash
# Clone the repository
git clone https://github.com/intuit-ai-research/REMem.git
cd REMem

# Install in editable mode
pip install -e .
```

Or install from `requirements.txt`:

```bash
pip install -r requirements.txt
pip install -e .
```

Requires Python 3.10+.

## Quick start

```python
from remem.remem import ReMem
from remem.utils.config_utils import BaseConfig

config = BaseConfig(
    dataset="sample",
    extract_method="episodic_gist",
    llm_name="gpt-4o-mini",
    embedding_model_name="nvidia/NV-Embed-v2",
)

rag = ReMem(global_config=config)

docs = [
    "Alan Turing proposed the Turing Test in 1950.",
    "Grace Hopper pioneered COBOL and popularized the term 'debugging'.",
]
rag.index(docs)

solutions, responses, meta = rag.rag_for_qa(
    ["Who proposed the Turing Test?", "Who worked on COBOL?"]
)
for s in solutions:
    print(s.question, "->", s.answer)
```

## Running benchmarks

```bash
python main.py --dataset musique --llm_name gpt-4o-mini --embedding_name nvidia/NV-Embed-v2
```

See `examples/` for dataset-specific scripts (MuSiQue, LoCoMo, LongMemEval, etc.) and `baselines/` for comparison methods.

## Architecture

| Layer | What It Does | Key Entry Points |
|-------|--------------|------------------|
| Orchestrator | Coordinates everything | `remem/remem.py` (`ReMem` class) |
| Preprocessing | Chunking & text cleanup | `graph/preprocessing/` |
| Extraction | Text → entities / facts / episodic traces | `information_extraction/` |
| Embeddings | Encode passages, entities, facts, summaries | `embedding_model/`, `embedding_store.py` |
| Graph Memory | Build + persist hybrid graph | `ReMem` (`add_*`, `augment_graph`) |
| Retrieval | Combines dense, fact, and graph signals | `rag_strategies/` |
| Reranking | Optional DSPy filtering of facts | `rerank.py` |
| Prompting | Structured prompt templates per dataset | `prompts/` |
| QA / Evaluation | Generate answers + metrics | `remem.py` + `evaluation/` |
| Agent | Multi-step tool reasoning variants | `agent/` |

### Processing pipeline

1. **Ingestion & Chunking** — documents → normalized chunks
2. **Embedding Storage** — passages inserted into `EmbeddingStore`
3. **Information Extraction** — produces structured units (entities, triples, episodic traces, gists)
4. **Memory Graph Build**
   - Facts: unit for embeddings and retrieval
   - Entities: unit for graph connections
   - Edges: gists → facts, entities ↔ entities
5. **Retrieval and QA**
   - Initial retrieval via semantic/lexical search for gists and facts
   - Graph exploration: `find_entity_contexts`/`find_gist_contexts`
   - Answer generation
6. **Evaluation**

## Extraction methods

Set `extract_method` in `BaseConfig`:

- `openie` — entity + triple extraction (lean & fast)
- `episodic` — episodic fact extraction
- `episodic_gist` — adds paraphrased gist memories for associative recall
- `temporal` — emphasizes temporal anchors for time-sensitive QA

Each mode influences which embedding stores exist, what graph nodes are created, and how retrieval seeds PageRank.

## Project structure

```
src/remem/
  remem.py                  # orchestrator (index, retrieve, rag_for_qa)
  embedding_store.py        # local embedding store
  embedding_model/          # embedding backends (OpenAI, NV-Embed-v2, GritLM, Qwen3)
  llm/                      # LLM backends (OpenAI API, Azure, vLLM offline)
  information_extraction/   # extraction methods
  rag_strategies/           # retrieval strategies
  graph/                    # preprocessing, graph construction
  prompts/                  # prompt templates
  rerank.py                 # DSPy-based fact filtering (optional)
  evaluation/               # metrics (EM, F1, BLEU, NDCG, LLM-judge)
  agent/                    # multi-step reasoning variants
  utils/                    # config, hashing, datetime, misc
examples/                   # benchmark scripts
baselines/                  # comparison baselines
reproduce/                  # datasets and reproducibility
```

### Key files

| File | Why It Matters |
|------|----------------|
| `remem.py` | Unified API: indexing, retrieval, QA, evaluation |
| `embedding_store.py` | Stores & reuses embeddings; enables incremental runs |
| `rag_strategies/*` | Swap retrieval logic without rewriting orchestration |
| `information_extraction/*` | Controls what memory units get stored |
| `prompts/prompt_template_manager.py` | Centralizes templates for extraction & QA |
| `utils/config_utils.py` | Single config object flows through all components |

## Customization

| You Want To… | Touch |
|--------------|-------|
| Add a new embedding backend | Add class in `embedding_model/` + register in `__init__.py` |
| Change chunking rules | Implement new preprocessor in `graph/preprocessing/` |
| Introduce new extraction style | Add module under `information_extraction/` + update factory in `remem.py` |
| Modify ranking blend | Edit / extend a strategy in `rag_strategies/` |
| Add new prompt template | Drop file in `prompts/templates/` and reference by name |
| Plug in new metric | Implement under `evaluation/` and include in `remem.py:get_evaluators` |
| Use multi-step reasoning | Explore `agent/` strategies (fixed vs flexible tools) |

## Citation

If you use REMem in your research, please cite our paper:

```bibtex
@inproceedings{shu2026remem,
  title={REMem: Reasoning with Episodic Memory in Language Agents},
  author={Shu, Yiheng and Jonnalagedda, Saisri Padmaja and Gao, Xiang and Guti{\'e}rrez, Bernal Jim{\'e}nez and Qi, Weijian and Das, Kamalika and Sun, Huan and Su, Yu},
  booktitle={The Thirteenth International Conference on Learning Representations (ICLR)},
  year={2026}
}
```

Paper: [https://www.arxiv.org/pdf/2602.13530](https://www.arxiv.org/pdf/2602.13530)

## License

Apache License 2.0
