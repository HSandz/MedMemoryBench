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
OPENROUTER_API_KEY=...
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
GOOGLE_SERVICE_ACCOUNT_FILE=/absolute/path/service-account.json
GOOGLE_CLOUD_LOCATION=global
GOOGLE_AI_STUDIO_API_KEYS_FILE=secrets/google_ai_studio_api_keys.txt
GOOGLE_AI_STUDIO_KEY_ROTATION_MODE=sequential
GOOGLE_AI_STUDIO_ROUND_ROBIN_CALLS_PER_KEY=1
GOOGLE_AI_STUDIO_MAX_ROTATION_ROUNDS=1
GOOGLE_AI_STUDIO_RESOURCE_EXHAUSTED_RETRIES=3
LLM_MAX_RETRIES=100
LLM_REQUEST_TIMEOUT_SECONDS=180
LLM_TRUNCATION_MAX_TOKENS=32768
JUDGE_PROVIDER=vertex
JUDGE_MODEL=gemini-2.5-flash
DEFAULT_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_PROVIDER=local
```

Generation reasoning can be configured per YAML model block when the selected
model supports it:

```yaml
model:
  provider: openai
  name: gpt-5.1
  reasoning_effort: high
```

The setting is translated to each provider's documented request shape:
OpenAI, Azure OpenAI, and Modal use `reasoning_effort`; OpenRouter uses
`reasoning: {effort: ...}`; Anthropic uses `output_config.effort`; and Gemini 3
uses `thinkingConfig.thinkingLevel` (`minimal`, `low`, `medium`, or `high`).
Gemini 2.5 requires an integer `reasoning_effort` interpreted as
`thinkingConfig.thinkingBudget` (for example, `1024`); invalid Gemini
model/value combinations fail before submission. Other providers pass the
documented value through for the provider API to validate.
The optional `memorize_model.reasoning_effort` applies independently to A-MEM
build calls. Judge calls use `JUDGE_REASONING_EFFORT` in `.env`.

For `NIM/nvidia/nemotron-*` models served through Bifrost, configure the
NVIDIA reasoning controls with the model's `nim` block:

```yaml
model:
  provider: openai
  name: NIM/nvidia/nemotron-3.5-lightning-30b-a3b
  nim:
    enable_thinking: false
    reasoning_budget: 0
```

The OpenAI-compatible client sends `chat_template_kwargs.enable_thinking` and
`reasoning_budget` at the request top level through Bifrost, with
`x-bf-passthrough-extra-params: true`. `reasoning_budget` must be an integer
from `-1` through `32768`. Set `enable_thinking: true` and a positive budget
only when visible reasoning is intended; benchmark answer runs should normally
disable it.

For `--stage query --memory-run`, compatibility is checked against snapshot
invariants only: the adapter, query-time embedding/index identity, serialized
snapshot feature flags, and dataset/query-unit identity. Retrieval options are
query-time controls and may be changed in the YAML; build-only settings such as
the memorization model, chunking, evolution thresholds, and build token budgets
are also ignored. The query LLM may therefore be changed while reusing an
existing memory snapshot.

Authoritative references: [OpenAI reasoning](https://developers.openai.com/api/docs/guides/reasoning),
[Gemini thinking](https://ai.google.dev/gemini-api/docs/thinking), and
[Anthropic effort](https://platform.claude.com/docs/en/build-with-claude/effort).

Gemini provider names are `vertex`, `ai_studio`, and hybrid `gemini`. Zep additionally needs `ZEP_API_KEY`; MIRIX may need PostgreSQL/Redis services.

For AI Studio and hybrid Gemini, put one API key on each non-empty line of
`secrets/google_ai_studio_api_keys.txt` and set
`GOOGLE_AI_STUDIO_API_KEYS_FILE` to that repository-relative path. Lines that
start with `#` are ignored. The key file takes precedence over inline AI Studio
key variables and the `secrets/` directory is Git-ignored. Inline
`GOOGLE_AI_STUDIO_API_KEYS` remains available for existing configurations.
When an AI Studio key reaches its retry threshold, the client logs a bounded,
single-line error summary before trying the next key. Set
`GOOGLE_AI_STUDIO_MAX_ROTATION_ROUNDS` to a positive number of full key-pool
passes per `ai_studio` LLM call, or `-1` to keep rotating indefinitely. A 429
with `You exceeded your current quota` rotates immediately, while a generic
`Resource has been exhausted` 429 retries the current key the configured number
of times first; `GOOGLE_AI_STUDIO_RESOURCE_EXHAUSTED_RETRIES` defaults to `3`.
For `ai_studio`, explicit 401, 403, or 404 messages identifying a deleted,
disabled, suspended, or revoked API key, bound service account, or project
retire the key immediately. When using `GOOGLE_AI_STUDIO_API_KEYS_FILE`, the
matching key line is removed atomically so later runs do not use it.

For `provider: ai_studio`, `GOOGLE_AI_STUDIO_KEY_ROTATION_MODE` defaults to
`sequential`: a key changes only after its configured failure threshold is
reached. Set it to `round_robin` to distribute successful top-level LLM calls
across keys. `GOOGLE_AI_STUDIO_ROUND_ROBIN_CALLS_PER_KEY` is the positive number
of successful calls served by each key before advancing to the next key and
defaults to `1`. Failure-triggered rotation and permanent-key retirement still
apply in both modes. Hybrid `gemini` continues to use its existing
failure-driven Vertex/AI Studio transport rotation.

`LLM_MAX_RETRIES` is the single per-failure-type retry budget for every
first-party LLM provider, including OpenAI-compatible APIs, Azure OpenAI,
Anthropic, Vertex, AI Studio, hybrid Gemini, and OpenRouter Batch. For providers
with multiple service accounts or API keys, the same value is also the threshold
before rotating to the next transport. Delays use exponential backoff starting
at `LLM_RETRY_MIN_DELAY=1` second and are capped at 100 seconds;
`LLM_RETRY_MAX_DELAY` may lower that cap but cannot raise it above 100. The old
`GOOGLE_VERTEX_SERVICE_ACCOUNT_RETRIES` setting is accepted only as a fallback
when `LLM_MAX_RETRIES` is absent; migrate to the general setting.

`LLM_REQUEST_TIMEOUT_SECONDS` controls the client-side timeout for
OpenAI-compatible realtime requests, including OpenAI, OpenRouter, and Modal.
It defaults to `180` seconds. The connection and write timeouts remain fixed at
30 seconds. A timeout raises a retryable SDK exception and is handled by the
shared retry policy.

Experiment error messages are limited to 1,000 characters wherever they are
printed, logged, or written to run metadata and API-failure artifacts. Longer
messages are cut at the limit and end with `...`; the full exception remains
available to the in-process retry and classification logic.

OpenRouter uses the same OpenAI-compatible request, retry, failure handling,
and usage tracking as the `openai` provider. Set `OPENROUTER_API_KEY`; the base
URL defaults to `https://openrouter.ai/api/v1`. Select it in a method config:

```yaml
model:
  provider: openrouter
  name: openai/gpt-5.1
  openrouter:
    provider:
      order: [openai, azure]
      allow_fallbacks: false
    service_tier: priority
```

Use OpenRouter model IDs, including the provider prefix. A model-level
`api_key` or `base_url` in YAML overrides the environment setting, but secrets
should remain in `.env`. The judge also accepts `JUDGE_PROVIDER=openrouter` and
uses the OpenRouter defaults unless `JUDGE_API_KEY` or `JUDGE_BASE_URL` is set.

The optional `model.openrouter.provider` mapping is passed unchanged as
OpenRouter's provider-routing object. It can use documented fields such as
`order`, `only`, `ignore`, `allow_fallbacks`, `require_parameters`, `sort`,
`data_collection`, `zdr`, `quantizations`, `preferred_min_throughput`,
`preferred_max_latency`, and `max_price`. This allows separate YAML configs to
run the same model through different upstream providers without changing the
model ID.

The optional `model.openrouter.service_tier` selects an OpenRouter service tier.
Current opt-in values are `flex` and `priority`; `fast` is an alias for
`priority`. Omit the field for normal default routing. Priority requests prefer
matching priority endpoints and may fall back, while flex requests stay on
flex-capable endpoints. Tier availability depends on the selected model and
upstream provider.

A-MEM methods can configure build calls independently with `memorize_model`,
using the same model/OpenRouter shape. This allows, for example, flex build
calls while the top-level query `model` omits `service_tier` and uses Batch API.
See [A-MEM Method Guide](AMEM_GUIDE.md#configuration).

Official references: [provider routing](https://openrouter.ai/docs/provider-routing)
and [service tiers](https://openrouter.ai/docs/guides/features/service-tiers).

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

## Staged Memory Runs

Build memory once, then answer queries from its snapshot:

```bash
python main.py -m persona_1/amem_fix_gemini -d medmemorybench --stage memory
python main.py --stage query --memory-run YYYYMMDD_HHMMSS
```

The query stage reads the stored effective configuration and writes under the source run's `query_runs/` directory. Use `--resume` for an incomplete query child. Do not change snapshot-incompatible A-MEM flags between stages.

Event-State supports the same staged workflow for LoCoMo, with a snapshot per
complete conversation sample rather than a persona/evaluation unit:

```bash
python main.py -m event_state_gemini -d locomo --stage memory --workers 4
python main.py --stage query --memory-run YYYYMMDD_HHMMSS --workers 4
```

LoCoMo snapshots retain original session and turn provenance. Query-stage
workers restore independent Event-State instances; `--resume` reuses completed
sample snapshots, rebuilds only a sample without a valid snapshot, and skips
answers recorded in the query checkpoint.

### LoCoMo Reporting

LoCoMo reports official token/stem `mean_f1` as the primary score. The
`queries_f1_ge_0_5` and `fraction_f1_ge_0_5` fields are debugging thresholds,
not benchmark accuracy. Any `enhanced_f1` is an explicitly labelled diagnostic
and never changes official F1.

When LoCoMo evidence annotations are present, query artifacts include
evaluator-only diagnostics for selected-memory source sessions and exact source
turns rendered into final answer context. Gold evidence is not sent to the
memory method or answer model. Batch runs mark per-request answer latency as
unavailable and report stage and batch wall times instead.

LoCoMo session timestamps are normalized by the dataset adapter before they
enter generic Event-State temporal handling; the raw timestamp is retained for
audit. With `include_images: true`, LoCoMo provides `blip_caption` text only:
Event-State does not consume image pixels. A run configured with
`max_samples: 1` is a one-conversation development run, not a full benchmark.

When `-d` is omitted, query stage restores the exact stored dataset config. To
apply a revised query selection from the same named config (for example,
`locomo_1` with `evaluation.category_filter: [1, 2, 3, 4]`), pass it explicitly:

```bash
python main.py -m event_state_gemini -d locomo_1 --stage query \
  --memory-run YYYYMMDD_HHMMSS --batch-api
```

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

## Batch API

For eligible Vertex Gemini or OpenRouter stages, run:

```bash
python main.py -m METHOD -d DATASET --batch-api
```

Batch jobs are waited on by default. Use `--no-batch-wait` to submit and exit,
then resume a submitted job with `--resume --batch-api`.
Batch mode does not batch stateful memory construction or local retrieval; it
only batches supported immutable LLM stages and LLM-judge requests.
Batch eligibility is evaluated independently for query answering and judging:
each supported non-empty stage uses its provider's Batch API regardless of
request count. An unsupported stage logs the fallback and remains real-time
without disabling batch execution for other supported stages.

Result `llm_usage` keeps answer/query calls and evaluator judge calls in separate
phase objects: `query_phase` contains retrieval preparation and final-answer
generation, `judge_phase` contains LLM-based scoring, and `total` combines both
with `memorize_phase`. The per-operation breakdown uses matching `query` and
`judge` operation buckets.

Vertex Gemini uses Cloud Storage JSONL staging and requires
`GOOGLE_BATCH_GCS_URI` or `--batch-gcs-uri`. OpenRouter submits inline requests
to its Batch API and ignores the GCS argument. It preserves
`model.openrouter.provider` routing when the selected upstream has a batch
endpoint. Before submission, the client checks OpenRouter's `:batch` model
variant against the configured provider routing and service tier. If no batch
endpoint matches, or support cannot be confirmed, it logs the reason and uses
the existing real-time OpenRouter client with the same retry, failure, and
usage-tracking behavior.

OpenRouter Batch is asynchronous and text-only. The current implementation uses
its chat-completions shape and correlates results by `custom_id`. See the
[OpenRouter Batch API quickstart](https://openrouter.ai/docs/batch-quickstart).

## Parallel Query Workers

Use `--workers N` to run up to `N` independent real-time query evaluations at
once. The default is `1`, which preserves sequential execution:

```bash
python main.py -m METHOD -d DATASET --workers 4
```

Each worker owns one question until its answer and real-time metric scoring are
complete, then takes another question. This keeps a question's answer and
judge calls together while bounding concurrent provider requests. Memory
construction, result collection, and checkpoint writes remain coordinated by the
main evaluator; completed reports retain dataset order. In Batch API mode,
query rewriting and read-only retrieval preparation are also worker-bounded.

The evaluator displays one run-wide `Query progress` bar only when query-answer
work starts, and closes it before a separate LLM-judge batch begins. It advances
as each question reaches an answer or terminal API failure, regardless of
completion order, and therefore works for both serial and worker-parallel
execution. Event-State memory preparation also displays a per-unit
`Memory build` bar when `--workers` enables staged preparation. The bar counts
both preparation and ordered stateful commit steps, so completion means the
entire memory build is finished.

When running `--stage query` against completed A-MEM or LoCoMo Event-State snapshots, independent
unit contexts can initialize without waiting for earlier units. Each unit uses
an isolated agent/memory context, while `N` remains a single global cap across
all real-time queries, not a per-unit cap. Result, batch, deferred-judge, and
checkpoint commits retain dataset order. This optimization requires query-stage
snapshots and does not change serial behavior for `N = 1`.

`--batch-api` takes precedence for every eligible query-answer or judge stage.
When both options are supplied, eligible stages use their provider's Batch API;
real-time stages use the configured workers instead. For example, this uses
workers for an unsupported answer provider while still batching an eligible
judge provider:

```bash
python main.py -m METHOD -d DATASET --batch-api --workers 4
```

Choose `N` within the provider's rate and concurrency limits. `N` must be at
least `1`.

## Outputs and Troubleshooting

Runs are organized as `outputs/<method-model>/<timestamp>/`, with `run_config.json`, `evaluation.log`, memory artifacts, checkpoints, and query answers. Explicit query reruns and append runs are nested under the source run's `query_runs/`.

Result JSON files report `duration_seconds` as total evaluation wall time. `true_duration_seconds` subtracts measured failed API-attempt time and retry waits, including failures that later recover and terminal API failures; successful API-call time remains included. The matching `llm_usage` totals expose the measured retry/error time as `failure_duration_seconds`.

Runs with failed API attempts also write a separate `*_api_failures.json` file.
It lists terminal failures that affected the run and includes aggregate counts
for failed attempts and retries by phase, including failures that recovered
within the retry policy.

Usage entries preserve the provider-reported total in `output_tokens` and add
`visible_output_tokens` and `thinking_tokens`. OpenAI-compatible APIs and
Gemini populate the split when their response usage contains a reasoning or
thought-token count. Providers that expose only aggregate output usage report
that count as visible output with `thinking_tokens: 0`.

OpenAI-compatible reasoning models may return `finish_reason: "length"` when
the output budget is consumed by reasoning before a final answer is produced.
The client logs `finish_reason`, bounded previews of `message.reasoning`, and
`message.reasoning_content`, then retries with a doubled `max_tokens` budget.
Retries stop at `LLM_TRUNCATION_MAX_TOKENS` (default `32768`); raise that value
or the model's configured output budget when truncation persists.

- Missing API credentials: check `.env` and the provider named in the method YAML.
- Missing embedding model: use a valid local path or allow the configured Hugging Face model to download.
- Stale or incompatible resume: use the original method/dataset config and inspect `run_config.json`; do not overwrite a completed run.
- Batch job pending: rerun with `--resume --batch-api`; use `--no-batch-wait`
  only when submission should exit immediately.
- Method import failure: run `python main.py --list-agents` and verify the adapter and YAML exist.
