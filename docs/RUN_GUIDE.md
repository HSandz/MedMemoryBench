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

### Modal-hosted vLLM

The `modal` provider uses the same OpenAI-compatible client and shared retry and
usage tracking as `openai`. Install Modal separately from the benchmark
environment, then deploy the bundled server:

See [`docs/MODAL_GUIDE.md`](MODAL_GUIDE.md) for the complete deployment guide,
including how to create a Proxy Token and determine `MODAL_API_KEY` and
`MODAL_BASE_URL`. Deployment uses `MODAL_TOKEN_ID` and
`MODAL_TOKEN_SECRET`; no browser or interactive CLI authentication is required.

```bash
uv pip install modal python-dotenv
modal deploy scripts/modal_vllm_server.py
```

The deployment script reads Modal account credentials and `MODAL_VLLM_*`
settings directly from the repository-root `.env`. Existing shell variables
take precedence, so exports are only needed for one-off overrides. Set the
optional `HF_TOKEN` there to authenticate Hugging Face model downloads; the
deployment injects it as a runtime secret rather than baking it into the image.

The deployment defaults to
`ELVISIO/Qwen3-30B-A3B-Instruct-2507-AWQ`, an L40S GPU, and a 32,768-token
model context. Override `MODAL_GPU`, `MODAL_VLLM_MODEL`,
`MODAL_VLLM_SERVED_MODEL`, or `MODAL_VLLM_MAX_MODEL_LEN` in `.env` when the
selected GPU or model requires different settings. The deployment caches
Hugging Face and vLLM artifacts in Modal Volumes. It defaults to one minimum
container during deployment, so `modal deploy` starts the GPU server without a
separate wake-up request. After model readiness, it holds the container for the
300-second idle window and then restores `min_containers=0`, allowing normal
scale-to-zero. Set `MODAL_VLLM_SCALEDOWN_WINDOW_SECONDS` before deployment to
change that post-readiness window.

Configure a method model with `provider: modal`, or set the equivalent
environment variables for a method that accepts provider settings:

```env
MODAL_API_KEY=wk-<id>.ws-<secret>
MODAL_BASE_URL=https://<workspace>--<app>-server.<region>.modal.direct/v1
```

Use the Modal Proxy Token as `MODAL_API_KEY`; do not put it in a committed YAML
file. The base URL must include `/v1`. Modal Servers can return HTTP 503 while
the model is starting or while scaling from zero, so leave the shared retry
settings enabled. Modal Servers use proxy authentication by default; keep
`unauthenticated=True` out of the deployment unless the endpoint is
intentionally public.

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
