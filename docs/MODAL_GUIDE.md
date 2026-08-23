# Modal vLLM Guide

This guide explains how to deploy the bundled vLLM server on Modal and connect
MedMemoryBench to it through the OpenAI-compatible `modal` provider.

The benchmark deployment defaults to:

- Model: `ELVISIO/Qwen3-30B-A3B-Instruct-2507-AWQ`
- GPU: `L40S`
- Maximum model context: `32768` tokens
- vLLM HTTP port: `8000`

You can override these defaults with the environment variables described below.

## Prerequisites

1. Create or access a [Modal account](https://modal.com/).
2. Create a Modal account token in the dashboard. You need the token ID and
  token secret to deploy the server.
3. Install the Modal CLI in the environment used for deployment. Modal is a
  deployment dependency and does not need to be installed in the benchmark's
  runtime environment.
4. Clone this repository and complete the normal benchmark setup.
5. Make sure the selected Modal workspace has access to the GPU type required by
   the model.

From the repository root:

```bash
uv pip install modal
```

Set the Modal account token as environment variables before running any Modal
deployment command. These variables are used by the Modal CLI without browser
login or interactive authentication:

```bash
export MODAL_TOKEN_ID=ak-<token-id>
export MODAL_TOKEN_SECRET=as-<token-secret>
```

The Modal CLI does not automatically read the repository's `.env` file. Export
only the two deployment variables explicitly:

```bash
export MODAL_TOKEN_ID="$(grep '^MODAL_TOKEN_ID=' .env | cut -d= -f2-)"
export MODAL_TOKEN_SECRET="$(grep '^MODAL_TOKEN_SECRET=' .env | cut -d= -f2-)"
modal deploy scripts/modal_vllm_server.py
```

Do not use `source .env` if `MODAL_BASE_URL` still contains placeholders such
as `<workspace>` or `<app>`; angle brackets are interpreted by Bash. Replace
those placeholders with the real deployed URL after the first successful
deployment.

The deployment token must be present in the shell that runs `modal deploy`.
`MODAL_API_KEY` is not a substitute: it is the inference Proxy Token used by
clients after deployment.

Modal account tokens use the `ak-` and `as-` prefixes. They are different from
the `wk-` / `ws-` Proxy Token used later by benchmark requests. Do not confuse
the two token types.

## Deploy the vLLM server

Deploy the bundled server from the repository root:

```bash
modal deploy scripts/modal_vllm_server.py
```

The first deployment may take several minutes because Modal builds the image and
vLLM downloads the model weights. Hugging Face and vLLM caches are stored in
Modal Volumes so later container starts do not need to download everything
again.

The Server is configured with a five-minute idle `scaledown_window`. When no
requests arrive for approximately five minutes, Modal scales the GPU container
down; the next request starts a new container and may take longer while vLLM
loads the model.

The deploy command prints a Server URL similar to:

```text
https://your-workspace--medmemorybench-vllm-server.us-east.modal.direct
```

Save this URL. It is the basis of `MODAL_BASE_URL`, but the benchmark requires
the OpenAI API path suffix `/v1`:

```text
https://your-workspace--medmemorybench-vllm-server.us-east.modal.direct/v1
```

Do not use a URL ending in `/v1/chat/completions` as `MODAL_BASE_URL`; the
OpenAI client appends the API path itself.

### Optional deployment overrides

Set overrides before deploying:

```bash
export MODAL_APP_NAME=medmemorybench-vllm
export MODAL_GPU=L40S
export MODAL_VLLM_MODEL=RedHatAI/Qwen3.8-27B-INT4
export MODAL_VLLM_SERVED_MODEL=RedHatAI/Qwen3.8-27B-INT4
export MODAL_VLLM_MAX_MODEL_LEN=32768
modal deploy scripts/modal_vllm_server.py
```

These values are baked into the deployed container image. Always set them in
the same shell immediately before `modal deploy`; changing them later does not
change an already deployed App. After deployment, verify the exact model ID
advertised by the Server:

```bash
curl --fail-with-body \
  -H "Authorization: Bearer ${MODAL_API_KEY}" \
  "${MODAL_BASE_URL}/models"
```

Use the returned `data[].id` value exactly in the method YAML's `model.name` and
`build_config.amem_model`. A successful `modal deploy` only confirms that the
App was registered; the vLLM model is loaded when the Server starts.

Use a shorter context when the selected model or GPU does not have enough VRAM.
Changing the model can also require model-specific vLLM arguments; the bundled
script is configured for the default model.

## Create the API key

The deployed object is a private Modal Server by default. MedMemoryBench uses a
Modal **Proxy Token** as its API key. A Proxy Token consists of a token ID and a
secret:

- Token ID: starts with `wk-`
- Token secret: starts with `ws-`

### Dashboard method

1. Open the Modal dashboard and create an **Account Token** for the workspace.
  Use the dashboard's token or API-token settings; the exact dashboard path
  can vary by workspace UI version.
2. Copy the account token ID and token secret when Modal displays them.
3. Set them for deployment as `MODAL_TOKEN_ID=ak-...` and
  `MODAL_TOKEN_SECRET=as-...`.
4. If the workspace uses environments or RBAC, ensure the account token can
  deploy to the target environment.

The benchmark endpoint itself uses a separate **Proxy Token**. Create it after
the Server is deployed:

1. Open the Modal [Proxy Auth Tokens settings](https://modal.com/settings/proxy-auth-tokens).
2. Choose **Create proxy token**.
3. Give the token a descriptive name, such as `medmemorybench-local`.
4. Copy the Proxy Token ID and secret when Modal displays them.
5. If the workspace uses environments or RBAC, allow the Proxy Token to access the
   environment where the vLLM App is deployed.

Combine the two values with a period to form the API key:

```text
MODAL_API_KEY=wk-<token-id>.ws-<token-secret>
```

For example, if the dashboard shows `wk-1234` and `ws-5678`, the combined value
is `wk-1234.ws-5678`. Keep this value private. Do not commit it to YAML, shell
scripts, or the repository.

## Configure MedMemoryBench

Add the following inference values to the repository `.env` file. The
`.env.example` file contains the account-token and Proxy-Token variable names
as commented examples. The account token is for deployment; the Proxy Token is
for requests sent to the deployed Server.

```env
MODAL_API_KEY=wk-<token-id>.ws-<token-secret>
MODAL_BASE_URL=https://your-workspace--medmemorybench-vllm-server.us-east.modal.direct/v1
```

`MODAL_PROXY_TOKEN` may be used instead of `MODAL_API_KEY`; the application
accepts it as a compatibility fallback. `MODAL_API_KEY` is recommended because
it describes the value's use by the OpenAI-compatible client.

Configure a method YAML with `provider: modal` and the model name served by
vLLM. For example:

```yaml
model:
  provider: modal
  name: ELVISIO/Qwen3-30B-A3B-Instruct-2507-AWQ
  temperature: 0.0
  max_completion_tokens: 4096
```

For A-MEM configurations, also set the internal memory-building backend to
`modal`; this routes A-MEM's metadata and evolution prompts to the same
OpenAI-compatible Modal Server:

```yaml
build_config:
  amem_backend: modal
  amem_model: ELVISIO/Qwen3-30B-A3B-Instruct-2507-AWQ
```

The top-level `provider: modal` controls the agent's answer-generation client,
while `amem_backend: modal` controls A-MEM's internal LLM calls.

You can also put `api_key` and `base_url` directly in the model YAML, but using
`.env` is safer and keeps credentials out of committed configuration files.

Run a configuration check before starting a long evaluation:

```bash
python main.py -m METHOD -d DATASET --dry-run
```

Replace `METHOD` and `DATASET` with the method and dataset you want to run.

## Test the endpoint

Use the base URL without `/v1` for the first health check, and the `/v1` URL for
OpenAI-compatible API calls. The `/v1/models` request verifies both the URL and
the Proxy Token:

```bash
curl --fail-with-body \
  -H "Authorization: Bearer ${MODAL_API_KEY}" \
  "${MODAL_BASE_URL}/models"
```

A successful response contains a JSON list with the served model name. To test
chat completion directly:

```bash
curl --fail-with-body \
  -H "Authorization: Bearer ${MODAL_API_KEY}" \
  -H "Content-Type: application/json" \
  "${MODAL_BASE_URL}/chat/completions" \
  -d '{
    "model": "ELVISIO/Qwen3-30B-A3B-Instruct-2507-AWQ",
    "messages": [{"role": "user", "content": "Reply with OK."}],
    "max_tokens": 16,
    "temperature": 0
  }'
```

The first request can return HTTP 503 while Modal starts a container and vLLM
loads the model. Retry after the container becomes ready. MedMemoryBench's
shared retry layer already retries transient 503 responses.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| `401 Unauthorized` | Verify that `MODAL_API_KEY` is exactly `wk-<id>.ws-<secret>` and that the token has access to the deployment environment. |
| `403 Forbidden` | The workspace likely uses RBAC; allow the Proxy Token for the environment containing the App. |
| `404 Not Found` | Confirm that `MODAL_BASE_URL` ends with `/v1` and does not include `/models` or `/chat/completions`. |
| `503 Service Unavailable` | The Server is scaling from zero or vLLM is still loading. Retry and check the Modal App logs. |
| vLLM startup timeout | Use a compatible GPU, reduce `MODAL_VLLM_MAX_MODEL_LEN`, or increase the deployment's startup timeout if the model needs more time. |
| Model not found | Set the YAML `model.name` to the value passed as `MODAL_VLLM_SERVED_MODEL`. |

Do not set `unauthenticated=True` merely to work around authentication errors.
That would make the inference endpoint publicly accessible. Instead, create a
new Proxy Token or correct its environment permissions.

## Useful Modal commands

```bash
modal app list
modal app logs medmemorybench-vllm
modal workspace proxy-tokens list
modal volume list
```

For detailed Modal behavior, see the official documentation for
[Servers](https://modal.com/docs/guide/servers),
[Proxy Tokens](https://modal.com/docs/guide/webhook-proxy-auth), and
[OpenAI-compatible vLLM inference](https://modal.com/docs/examples/vllm_inference).
