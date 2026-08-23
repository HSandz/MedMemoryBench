# MedMemoryBench Run Guide

## Setup

Use Python 3.10+ and Git LFS for the benchmark data.

```bash
git lfs install
git lfs pull
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt
# or: pip install -r requirements.txt
cp .env.example .env
```

Set only the credentials required by the selected method. Keep `.env`, `service-account.json`, and private GCS paths out of commits.

## Configure a Run

Method configs are YAML files under `configs/method_config/`; dataset configs are under `configs/dataset_config/`. A persona-specific config uses a path such as `persona_1/amem_fix_gemini`.

Common environment settings include:

```env
OPENAI_API_KEY=...
OPENAI_BASE_URL=https://api.openai.com/v1
GOOGLE_SERVICE_ACCOUNT_FILE=/absolute/path/service-account.json
GOOGLE_CLOUD_LOCATION=global
GOOGLE_AI_STUDIO_API_KEYS=key1,key2
JUDGE_PROVIDER=vertex
JUDGE_MODEL=gemini-2.5-flash
DEFAULT_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_PROVIDER=local
```

Gemini provider names are `vertex`, `ai_studio`, and hybrid `gemini`. Zep additionally needs `ZEP_API_KEY`; MIRIX may need PostgreSQL/Redis services.

## Run Commands

```bash
# Inspect registered components
python main.py --list-agents
python main.py --list-datasets

# Validate configuration without real LLM calls
python main.py -m embedding_rag_gpt-5.1 -d medmemorybench --dry-run

# Full evaluation
python main.py -m bm25_rag_gpt-5.1 -d medmemorybench

# Continue an interrupted run
python main.py -m embedding_rag_gpt-5.1 -d medmemorybench --resume

# Use the helper script
./scripts/run_eval.sh bm25_rag_gpt-5.1 medmemorybench

# Re-run only judge metrics from a saved answer file
JUDGE_MODEL=gpt-5.1 python main.py --rejudge outputs/METHOD_MODEL/RUN_query_answer.json
```

## Staged A-MEM Runs

Build memory once, then answer queries from its snapshot:

```bash
python main.py -m persona_1/amem_fix_gemini -d medmemorybench --stage memory
python main.py --stage query --memory-run YYYYMMDD_HHMMSS
```

The query stage reads the stored effective configuration and writes under the source run's `query_runs/` directory. Use `--resume` for an incomplete query child. Do not change snapshot-incompatible A-MEM flags between stages.

### Incremental A-MEM Append

Append from a completed or interrupted A-MEM memory run when only part of the benchmark has been built:

```bash
# Extend memory through the requested target and export snapshots only
python main.py --memory-run SOURCE_RUN --append --stage memory \
  --persona PERSONA_ID --unit UNIT_ID

# Extend memory, then answer and score every unit in the new prefix
python main.py --memory-run SOURCE_RUN --append \
  --persona PERSONA_ID --unit UNIT_ID
```

`--persona` identifies the persona and `--unit` is the dataset's global evaluation-unit ID. The target is inclusive. The method and dataset configurations are reconstructed from the source run's `run_config.json`; do not supply a different configuration. If the exact target snapshot already exists, the command exits without creating another run. An interrupted append can be continued with the same command plus `--resume`.

Each append is a new run under `SOURCE_RUN/query_runs/`, with its own `run_config.json`, `memory/manifest.json`, snapshots, and (for the default/all form) query reports. A completed append can itself be used as `SOURCE_RUN` for another append, so memory can be extended in multiple increments. A partial append can also be queried with the normal staged command:

```bash
python main.py --stage query --memory-run APPEND_RUN
```

## Vertex Batch API

For eligible Gemini/Vertex stages, provide a private bucket and run:

```bash
python main.py -m METHOD -d DATASET --batch-api --batch-wait
```

Without `--batch-wait`, resume the submitted job with `--resume --batch-api`. Batch mode does not batch stateful memory construction or local retrieval; it only batches supported immutable LLM stages. Use `GOOGLE_BATCH_GCS_URI` or `--batch-gcs-uri`.

## Outputs and Troubleshooting

Runs are organized as `outputs/<method-model>/<timestamp>/`, with `run_config.json`, `evaluation.log`, memory artifacts, checkpoints, and query answers. Explicit query reruns and append runs are nested under the source run's `query_runs/`.

Result JSON files report `duration_seconds` as total evaluation wall time. `true_duration_seconds` subtracts measured failed API-attempt time and retry waits, including failures that later recover and terminal API failures; successful API-call time remains included. The matching `llm_usage` totals expose the measured retry/error time as `failure_duration_seconds`.

- Missing API credentials: check `.env` and the provider named in the method YAML.
- Missing embedding model: use a valid local path or allow the configured Hugging Face model to download.
- Stale or incompatible resume: use the original method/dataset config and inspect `run_config.json`; do not overwrite a completed run.
- Vertex job pending: rerun with `--resume --batch-api`, or use `--batch-wait` initially.
- Method import failure: run `python main.py --list-agents` and verify the adapter and YAML exist.
