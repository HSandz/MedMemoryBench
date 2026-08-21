# MedMemoryBench Repository Audit

Audit date: 2026-08-01. Scope: the evaluation application (`main.py`, `src/`, `benchmarks/`, `metrics/`, `utils/`, method adapters, configuration, and repository hygiene). Vendored projects under `methods/` were sampled for security-sensitive integration paths, not exhaustively audited as independent products.

## Findings

### High: a long-lived service-account key is used from the working tree

`service-account.json` is present as an untracked file. It is now ignored, but its contents were intentionally not read. The Gemini provider loads this user-requested local credential, which places a long-lived private key beside the source checkout. Restrict file permissions, do not copy it into artifacts or backups, rotate it if exposed, and prefer `GOOGLE_SERVICE_ACCOUNT_FILE` pointing outside the repository for shared development or production.

### High: `eval()` parses tool-call text in vendored runtime paths

`methods/letta/interface.py:199`, `methods/letta/interface.py:225`, `methods/MIRIX/mirix/interface.py:194`, and `methods/MIRIX/mirix/interface.py:223` evaluate string arguments extracted from tool-status messages. LLM-generated or remote text reaching these paths can execute arbitrary Python in the evaluator process. Replace `eval()` with `json.loads()` for JSON tool arguments (or `ast.literal_eval()` only where Python literals are an unavoidable compatibility requirement), validate the expected schema, and test malicious payloads. Treat Letta and MIRIX runs as trusted-input-only until this is fixed.

### Medium: persisted pickle data is loaded without a trust boundary

Examples include `methods/amem/A-mem/memory_layer.py:461` and `methods/mem0/vector_stores/faiss.py:85`. Python pickle deserialization can execute code; caches and index files must therefore be treated as executable artifacts. Keep `results/`, cache directories, and restored experiment outputs private to trusted users; prefer a JSON/NumPy representation where practical, and add provenance or integrity checks before loading shared artifacts.

### Medium: dataset errors can silently turn into empty or partial evaluations

`benchmarks/medmemorybench/dataset.py:130` and `benchmarks/medmemorybench/dataset.py:186` quietly skip a missing dialogue or query file. A missing root directory raises a generic filesystem exception at `:100`, but incomplete persona data can produce an apparently successful run with zero sessions or queries. Fail early when required files are absent and reject runs with zero evaluation units unless the caller explicitly opts in.

### Medium: provider configuration is not fully provider-specific across all methods

`src/agent.py:79`-`:86` passes OpenAI API key/base URL defaults to every method. The Gemini Enterprise client safely ignores them and uses the service-account credential, but some adapters bypass the shared client: GraphRAG chooses `ChatGoogleGenerativeAI` from the model name in `methods/graph_rag.py:87` rather than the configured provider. Keep Gemini Enterprise runs to adapters that use `utils.llm_client` (for example `long_context`, BM25 RAG, and embedding RAG with a supported embedding provider) until each bypassing adapter has an explicit managed-Google integration.

### Medium: one listed method and one support script are not runnable from the checkout

`mem1_gpt-5.1` is listed by the CLI, but `src/agent.py` imports `methods.mem1_agent`, which does not exist. In addition, `scripts/mirix-services.sh` references `docker/mirix-services.yml`, which is absent. Users receive a runtime import or compose-file failure rather than an actionable preflight error. Remove or repair the Mem1 configuration and include the MIRIX compose file (or fail early with the exact prerequisite).

### Low: dependency resolution is not reproducible

`requirements.txt` primarily uses unbounded lower limits and does not ship a lock or constraints file. A clean install can resolve materially different versions over time, affecting evaluation behavior and possibly breaking vendored integrations. Publish a tested lock/constraints file per supported Python version and use it in CI and result-producing runs.

### Low: core run reproducibility is incomplete

The main evaluation CLI exposes no common seed or captured dependency/model metadata. Some adapters have their own seeds, while the shared path does not establish deterministic settings. Record the resolved dependency set, method/dataset YAML hashes, model revision, provider region, and sampling parameters with each output. Where model APIs allow it, expose a shared seed parameter, while documenting that managed model outputs are not guaranteed deterministic.

## Architecture and operational notes

- `main.py` loads method and dataset YAML, then `src/agent.py` dynamically imports the adapter selected by method name.
- Most core adapters call `utils.llm_client.create_llm_client`; this is the integration point for OpenAI, Azure, Anthropic, and Gemini Enterprise.
- `metrics/llm_judge.py` uses `JUDGE_PROVIDER`; `JUDGE_PROVIDER=gemini` uses the same managed-Gemini service-account path for MedMemoryBench's judge metrics.
- The shipped method configurations target model names such as `gpt-5.1` and `qwen3`; availability, costs, and compatible endpoints remain operator responsibilities.
- The repository includes many vendored and optional method projects. Install and expose only the method components needed for a given experiment.

## Recommended order

1. Secure the local service-account key and move it outside the checkout where practical.
2. Replace the four `eval()` calls in the Letta/MIRIX runtime interfaces and add regression tests.
3. Make incomplete datasets fail fast and add a small fixture-based integration test for each benchmark.
4. Add CI with a pinned dependency set, a core dry-run smoke test, and mocked provider-client tests.
5. Separate provider credentials and configuration per adapter, then add managed-Gemini coverage for any adapters that bypass `utils.llm_client`.
