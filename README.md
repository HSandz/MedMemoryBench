# MedMemoryBench: Benchmarking Agent Memory in Personalized Healthcare

<div align="center">
  <img src="figs/4examples.png" alt="MedMemoryBench overview — four key challenges in medical memory evaluation" width="800"/>
</div>

<p align="center">
  <em>Solving the issues of agent memory evaluation in healthcare scenarios.</em>
</p>

<p align="center">
｜🤗 <a href="https://huggingface.co/datasets/Cyan27/MedMemoryBench" target="_blank">HuggingFace Dataset</a> ｜
📄 <a href="https://arxiv.org/abs/2605.11814">Arxiv Preprint</a> ｜
🌐 <a href="README_ZH.md">中文</a> ｜
</p>

<p align="center">
  <img src="https://img.shields.io/badge/version-1.0.0-blue" alt="version"/>
  <img src="https://img.shields.io/badge/python-%3E%3D3.10-blue" alt="python"/>
  <img src="https://img.shields.io/badge/license-Apache%202.0-green" alt="license"/>
</p>

---

**MedMemoryBench** is a benchmark framework for evaluating Agent memory methods, with a focus on memory capability assessment in medical dialogue scenarios. This framework provides unified evaluation interfaces, multiple baseline method implementations, and a flexible configuration management system, while also supporting the import and evaluation of other datasets.

## Table of Contents

- [News](#-news)
- [Features](#-features)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
- [Configuration](#-configuration)
- [Output](#-output)
- [Citations](#-citations)

---

## 📰 News

- **[2026.05]** MedMemoryBench v1.0 is officially released — dataset, evaluation framework, and 14 memory method baselines.
- **[2026.05]** Dataset available on [HuggingFace](https://huggingface.co/datasets/Cyan27/MedMemoryBench).

## ✨ Features

<table>
<tr>
<td width="50%">

**Comprehensive Medical Dataset**
- 20 longitudinal patient personas with background, life events, and trap events
- ~2,020 multi-session doctor–patient dialogue sessions
- ~1,986 evaluation queries across 6 clinically motivated types
- Bilingual support: Chinese (~598 MB) + English (~443 MB)

</td>
<td width="50%">

**Rich Baseline Coverage**
- **3 classic baselines**: Long Context, Embedding RAG, BM25 RAG
- **7 agentic memory systems**: Mem0, Letta, MemOS, A-MEM, MIRIX, MemRL, LightMem
- **4 graph-based systems**: GraphRAG, HippoRAG-v2, ReMem, Zep

</td>
</tr>
<tr>
<td>

**Unified Evaluation Framework**
- Plug-and-play method integration via `BaseAgent`
- Multi-metric evaluation: string match + LLM-as-a-Judge
- Checkpoint & resume for long-running experiments
- Dry-run mode for fast pipeline validation

</td>
<td>

**Flexible Configuration**
- YAML-driven method & dataset configs
- Multi-provider LLM support (OpenAI / BigModel / Azure)
- Local & remote embedding models
- Cross-benchmark evaluation (MedMemoryBench + LoCoMo)

</td>
</tr>
</table>

## 📁 Project Structure

<details>
<summary>Click to expand full directory tree</summary>

```
MedMemoryBench/
├── main.py                       # Evaluation entry point
├── requirements.txt              # Python dependencies
├── LICENSE                       # Apache License 2.0
├── LEGAL.md                      # Comment-language legal notice
├── .env.example                  # Environment variable template
│
├── configs/                      # Configuration files
│   ├── method_config/            # Per-method YAML configs (gpt-5.1 / qwen3 variants)
│   │   ├── long_context_gpt-5.1.yaml
│   │   ├── embedding_rag_gpt-5.1.yaml
│   │   ├── bm25_rag_gpt-5.1.yaml
│   │   ├── graph_rag_gpt-5.1.yaml
│   │   ├── mem0_gpt-5.1.yaml
│   │   ├── memos_gpt-5.1.yaml
│   │   ├── memrl_gpt-5.1.yaml
│   │   ├── amem_gpt-5.1.yaml
│   │   ├── hipporag_gpt-5.1.yaml
│   │   ├── lightmem_gpt-5.1.yaml
│   │   ├── letta_gpt-5.1.yaml
│   │   ├── mirix_gpt-5.1.yaml
│   │   ├── remem_gpt-5.1.yaml
│   │   ├── zep_gpt-5.1-chat.yaml
│   │   └── ...                   # + qwen3 variants
│   └── dataset_config/
│       ├── medmemorybench.yaml
│       └── locomo.yaml
│
├── methods/                      # Memory method implementations
│   ├── base.py                   # BaseAgent abstract class
│   ├── long_context.py           # Long-context baseline
│   ├── embedding_rag.py          # Dense embedding RAG
│   ├── bm25_rag.py               # BM25 sparse RAG
│   ├── graph_rag.py              # Graph-based RAG
│   ├── self_rag.py               # Self-RAG
│   ├── mem0_agent.py             # Mem0 adapter
│   ├── memos_agent.py            # MemOS adapter
│   ├── memrl_agent.py            # MemRL adapter
│   ├── amem_agent.py             # A-MEM adapter
│   ├── hipporag_agent.py         # HippoRAG adapter
│   ├── lightmem_agent.py         # LightMem adapter
│   ├── letta_agent.py            # Letta adapter
│   ├── mirix_agent.py            # MIRIX adapter
│   ├── remem_agent.py            # ReMem adapter
│   ├── zep_agent.py              # Zep Cloud adapter
│   └── <vendored repos>          # mem0/, memOS/, MemRL/, amem/, HippoRAG/,
│                                 # LightMem/, letta/, MIRIX/, REMem/, MEM1/,
│                                 # cognee/, memorag/  (third-party sources)
│
├── benchmarks/                   # Dataset evaluation implementations
│   ├── base.py                   # BaseDataset abstract class
│   ├── medmemorybench/           # MedMemoryBench dataset
│   │   ├── dataset.py
│   │   ├── evaluator.py
│   │   └── checkpoint.py
│   └── locomo/                   # LoCoMo dataset
│       ├── dataset.py
│       └── evaluator.py
│
├── metrics/                      # Evaluation metrics
│   ├── base.py                   # BaseMetric abstract class
│   ├── string_match.py           # String matching metrics
│   ├── llm_judge.py              # LLM-as-a-Judge metrics
│   └── locomo_metrics.py         # LoCoMo-specific metrics
│
├── src/                          # Core orchestration modules
│   ├── config.py                 # Configuration loader
│   ├── agent.py                  # AgentManager
│   ├── evaluator.py              # Evaluation dispatcher
│   └── result.py                 # Result collection & reporting
│
├── utils/                        # Utility modules
│   ├── llm_client.py             # Unified LLM client
│   ├── tokenizer.py              # Tokenizer helpers
│   ├── templates.py              # Prompt templates
│   ├── prompts_qa.py             # QA prompts
│   ├── prompts_judge.py          # Judge prompts
│   ├── prompts_memorize.py       # Memorization prompts
│   ├── langchain_callback.py     # LangChain callback hooks
│   └── logger.py                 # Logger
│
├── docker/                       # Optional service compose files
│   ├── mirix-init.sql
│   └── mirix-services.yml
│
├── scripts/                      # Helper scripts
│   ├── run_eval.sh
│   └── mirix-services.sh
│
├── data/                         # Datasets (Git LFS)
│   ├── MedMemoryBench/           # Chinese, ~598 MB
│   ├── MedMemoryBench_EN/        # English, ~443 MB
│   └── locomo/                   # LoCoMo, ~18 MB
│
├── generation/                   # Dataset generation pipeline (sub-project)
├── outputs/                      # Evaluation outputs (gitignored)
├── exp_results/                  # Curated experiment reports
├── logs/                         # Runtime logs (gitignored)
└── results/                      # Method-side caches (gitignored)
```

</details>

## 🚀 Quick Start

### 1. Clone the Repository

> **Note:** This repository ships datasets via **Git LFS**. Please install it before cloning.

```bash
# Install Git LFS (skip if already installed)
brew install git-lfs                  # macOS
sudo apt-get install git-lfs          # Ubuntu/Debian
# Windows: https://git-lfs.github.com/

git lfs install
git clone https://github.com/AQ-MedAI/MedMemoryBench.git
cd MedMemoryBench
```

### 2. Environment Setup

<details open>
<summary><b>Using uv (recommended)</b></summary>

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh

uv venv
source .venv/bin/activate          # Linux/macOS
# .venv\Scripts\activate           # Windows

uv pip install -r requirements.txt
```

</details>

<details>
<summary><b>Using conda</b></summary>

```bash
conda create -n medmemorybench python=3.10
conda activate medmemorybench
pip install -r requirements.txt
```

</details>

> **Method-specific dependencies:** Some memory methods vendor upstream packages under `methods/` (e.g. `methods/mem0/`, `methods/memOS/`). If a method has its own `requirements.txt` or `README`, follow those instructions to enable it.

> **Embedding models:** Method configs reference local embedding models or API. For the former, please download the embedded model before running.

### 3. Configure Environment Variables

```bash
cp .env.example .env
```

Edit `.env` and fill in the API keys you intend to use:

```env
# BigModel (OpenAI-compatible, primary endpoint used in this project)
BIGMODEL_API_KEY=your_bigmodel_api_key
BIGMODEL_BASE_URL=https://open.bigmodel.cn/api/paas/v4

# OpenAI (optional)
OPENAI_API_KEY=your_openai_api_key
OPENAI_BASE_URL=https://api.openai.com/v1

# Azure OpenAI (optional)
AZURE_OPENAI_API_KEY=your_azure_key
AZURE_OPENAI_ENDPOINT=https://your-endpoint.openai.azure.com/

# Google AI Studio (optional; comma-separated keys enable rotation)
GOOGLE_AI_STUDIO_API_KEYS=your_first_key,your_second_key

# Zep Cloud (optional, only needed for the Zep agent)
ZEP_API_KEY=your_zep_api_key

# Default model selection
DEFAULT_LLM_MODEL=gpt-4o-mini
DEFAULT_EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_PROVIDER=openai

# Optional: isolate Letta local runtime data (defaults to ~/.letta)
LETTA_DIR=.tmp/letta_runtime
```

> **Tips:**
> - For BigModel, set `BIGMODEL_API_KEY` / `BIGMODEL_BASE_URL` first; the framework maps them to OpenAI-compatible settings internally.
> - `LETTA_DIR` is recommended to avoid stale SQLite metadata from previous Letta runs.
> - Gemini has exactly three provider names: `vertex`, `ai_studio`, and `gemini`. Vertex can rotate through multiple service-account JSON files; hybrid `gemini` rotates through those Vertex accounts and then AI Studio keys when one failure type reaches five attempts on a transport. AI Studio ignores Vertex-only batch options; hybrid batch calls always use the rotating Vertex account pool.

### 4. Run Evaluation

**Via shell script:**

```bash
./scripts/run_eval.sh bm25_rag_gpt-5.1 medmemorybench
```

**Via Python:**

```bash
# Standard run
python main.py -m bm25_rag_gpt-5.1 -d medmemorybench

# Dry run (no real LLM/API calls)
python main.py -m embedding_rag_gpt-5.1 -d medmemorybench --dry-run

# Resume from checkpoint
python main.py -m embedding_rag_gpt-5.1 -d medmemorybench --resume

# Re-run only the LLM judge from saved method answers
JUDGE_MODEL=gpt-5.1 python main.py \
  --rejudge outputs/METHOD_MODEL/RUN_query_answer.json
```

`--rejudge` does not rebuild memory or call the evaluated method again. It
reloads the MedMemoryBench ground-truth metadata, reuses every saved
`model_output`, reruns only the LLM-judge query types, and preserves the
existing local-metric results. Each pass is written beside the source file as
`<source_stem>_rejudge_1.json`, then `_rejudge_2.json`, and so on. Configure the
judge with `JUDGE_PROVIDER`, `JUDGE_MODEL`, `JUDGE_API_KEY` or
`JUDGE_API_KEYS`, and `JUDGE_BASE_URL` as usual. Judge generation is also
configurable with `JUDGE_TEMPERATURE`, `JUDGE_CLIENT_MAX_TOKENS`,
`JUDGE_MAX_TOKENS`, and `JUDGE_MCD_MAX_TOKENS`; the same values are used by
real-time evaluation, Vertex batch evaluation, and `--rejudge`.

For a Vertex or hybrid Gemini judge, add `--batch-api`. `--batch-wait` polls
until completion; without it, resume the submitted job with the same command
plus `--resume --batch-api`. Batch rejudge uses the normal
`GOOGLE_BATCH_GCS_URI`/`--batch-gcs-uri` configuration and always uses Vertex.
An `ai_studio` or non-Gemini judge falls back to its real-time client.

<!-- > 💡 **Extending with a new method?** See [`methods/README.md`](methods/README.md) for the step-by-step guide. -->

## 🔧 Configuration

### Method Configuration

Each method is driven by a YAML file under `configs/method_config/`:

```yaml
# configs/method_config/embedding_rag_gpt-5.1.yaml

method_name: "embedding_rag"
method_type: "rag"                  # baseline / rag / agentic_memory
description: "Embedding RAG Agent - Dense vector retrieval based RAG method"

model:
  provider: "openai"
  name: "gpt-5.1"
  temperature: 0.3
  max_completion_tokens: 200000

agent_params:
  top_k: 5                          # Number of documents to retrieve
  chunk_size: 512                   # Text chunk size
  chunk_overlap: 50                 # Chunk overlap

embedding:
  provider: "local"                 # openai / local / huggingface
  model: "/path/to/local/model"
```

Gemini providers are intentionally distinct:

```yaml
# Vertex AI / Google Agent Platform (rotating service accounts; batch eligible)
model:
  provider: "vertex"
  name: "gemini-2.5-flash"

# Google AI Studio / Gemini Developer API (API keys; real-time only)
model:
  provider: "ai_studio"
  name: "gemini-2.5-flash"

# Hybrid rotation: each Vertex account, then each AI Studio key, then Vertex again
model:
  provider: "gemini"
  name: "gemini-2.5-flash"
```

For Vertex, set `GOOGLE_SERVICE_ACCOUNT_FILE` to an ordered comma-separated list such as `service-account.json,service-account-2.json,service-account-3.json`. Relative paths resolve from the repository root. Each file supplies its own authentication and project ID. `GOOGLE_VERTEX_SERVICE_ACCOUNT_RETRIES` controls the per-failure-type threshold before moving to the next account and defaults to `5`; Vertex batch upload, submission, polling, output collection, and direct fallback use the same pool.

For AI Studio and hybrid Gemini, set `GOOGLE_AI_STUDIO_API_KEYS` to an ordered comma-separated list. Every recognized non-critical failure type has an independent retry count and exponential-delay sequence, including retryable HTTP statuses, provider rate-limit/timeout/connection/availability exceptions, empty responses, malformed structured responses, service-account authentication/permission failures, and AI Studio key/quota/restriction failures. A transport rotates when any one failure type reaches its configured threshold. For example, three HTTP 429 failures and one empty response count as `3/5` and `1/5`, not `4/5`. The shared `GEMINI_MAX_RETRIES` limit uses the same per-failure-type accounting for non-rotating shared calls. Critical failures such as invalid requests remain non-retryable. `GOOGLE_AI_STUDIO_API_KEY`, `GOOGLE_API_KEY`, and `GEMINI_API_KEY` are supported for single-key setups.

All final-answer LLM settings belong in the method YAML's `model` block. Method
adapters with extra internal LLM calls expose those settings under
`agent_params`:

```yaml
model:
  temperature: 0.3
  max_completion_tokens: 20000

agent_params:
  # A-MEM metadata/evolution and typed-relation calls
  amem_temperature: 0.7
  amem_retry_temperature: 0.3
  amem_connectivity_temperature: 0.0
  amem_relation_temperature: 0.2
  amem_max_tokens: 1000

  # MemOS extraction
  memos_temperature: 0.0
  memos_max_tokens: 4096

  # LightMem extraction and buffering
  lightmem_temperature: 0.1
  lightmem_max_tokens: 2000
  lightmem_top_p: 0.1
  lightmem_buffer_max_tokens: 4096

  # MemRL keyword, script, reflection, and extractor calls
  memrl_keyword_temperature: 0.0
  memrl_keyword_max_tokens: 100
  memrl_script_temperature: 0.7
  memrl_script_max_tokens: 500
  memrl_reflection_temperature: 0.3
  memrl_reflection_max_tokens: null
  memrl_extractor_temperature: 0.0
  memrl_extractor_max_tokens: 4096
```

These are optional and preserve the previous defaults when omitted. Other
adapter-specific model settings already flow from `model` or their documented
`agent_params`. Shared retry timing can be changed through `LLM_MAX_RETRIES`,
`LLM_RETRY_MIN_DELAY`, `LLM_RETRY_MAX_DELAY`, `GEMINI_MAX_RETRIES`,
`GEMINI_RETRY_INITIAL_DELAY`, and `GEMINI_RETRY_MAX_DELAY`.

Official references: [Gemini API keys](https://ai.google.dev/gemini-api/docs/api-key), [Google Gen AI Python SDK](https://googleapis.github.io/python-genai/), and [Vertex AI batch inference](https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/batch-prediction-from-cloud-storage).

### Dataset Configuration

Dataset configs live under `configs/dataset_config/`:

```yaml
# configs/dataset_config/medmemorybench.yaml

dataset_name: "medmemorybench"
description: "Medical dialogue memory evaluation dataset"
language: "zh"

data:
  root_dir: "data/MedMemoryBench"
  sessions_pattern: "persona_{id}/eval/generated_dialogues.json"
  queries_pattern: "persona_{id}/eval/generated_queries.json"

evaluation:
  mode: "independent"               # independent / merged
  evaluation_interval: 10           # Evaluate every N sessions

query_types:
  - name: "entity_exact_match"
    metric: "string_contain"
  - name: "temporal_localization"
    metric: "llm_judge"
  # ... more types
```

## 📄 Output

Full runs and memory-build runs are isolated by their start timestamp. Query
reruns that explicitly select a memory run are nested under that source:

```
outputs/
└── amem_gemini-2.5-flash/
    └── 20260814_093138/
        ├── run_config.json
        ├── evaluation.log
        ├── memory/
        ├── batch/
        ├── checkpoints/
        ├── *_memory_build.json
        └── query_runs/
            └── 20260814_223919/
                ├── run_config.json
                ├── evaluation.log
                ├── memory_source.json
                ├── batch/
                ├── checkpoints/
                ├── *_result.json
                └── *_query_answer.json
```

`run_config.json` records the effective method, dataset, judge, stage, batch,
and resume settings with secrets redacted. AMem on MedMemoryBench can run as
separate build and query stages:

```bash
python main.py -m persona_1/amem_gemini -d medmemorybench --stage memory
python main.py -m persona_1/amem_gemini -d medmemorybench --stage query
python main.py --stage query --memory-run YYYYMMDD_HHMMSS
```

Query runs select the newest compatible completed memory run by default and
pin it in `memory_source.json`. For an explicit unified-layout memory run,
`-m` and `-d` are optional: the CLI reads the effective method and dataset
configs from that run's `run_config.json`, verifies its completed memory
manifest, and records the inference source in
`query_runs/<query-start-timestamp>/`. Query stages do not reload the current
YAML files, so later config edits do not invalidate an existing memory run.
Repeating the query command creates another child, so logs, batch state, and
reports never overwrite an earlier query. Use `--resume` to continue the same
incomplete query child.
Legacy artifacts cannot infer these values, so they still require `-m` and
`-d`. Legacy artifacts can be audited or migrated with
`scripts/migrate_run_artifacts.py --dry-run` and `--apply`.

## 📝 Citations

If you find MedMemoryBench useful in your research, please consider citing our work:

```bibtex
@article{wang2026medmemorybench,
  title={MedMemoryBench: Benchmarking Agent Memory in Personalized Healthcare},
  author={Yihao Wang and Haoran Xu and Renjie Gu and Yixuan Ye and Xinyi Chen and Xinyu Mu and Yuan Gao and Chunxiao Guo and Peng Wei and Jinjie Gu and Huan Li and Ke Chen and Lidan Shou},
  journal={arXiv preprint arXiv:2605.11814},
  year={2026}
}
```

---

## 📜 License

- **Code** — [Apache License 2.0](LICENSE)
- **Dataset** (`data/MedMemoryBench/`, `data/MedMemoryBench_EN/`) — [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- **Vendored third-party sources** under `methods/` retain their original upstream licenses.
- See [LEGAL.md](LEGAL.md) for the source-comment language clause.
