"""Configuration loading module - method config, dataset config, and environment variables."""

import os
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field

import yaml

try:
    from dotenv import load_dotenv
    HAS_DOTENV = True
except ImportError:
    HAS_DOTENV = False


# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

AMEM_BUILD_CONFIG_KEYS = {
    "amem_backend", "amem_model", "amem_embedding_model", "amem_evo_threshold",
    "amem_max_tokens", "amem_chunk_size_tokens", "amem_original_evolution",
    "amem_typed_relations", "amem_relation_candidate_count",
    "amem_relation_temperature", "amem_temporal_state", "amem_provenance",
    "amem_temperature", "amem_retry_temperature", "amem_connectivity_temperature",
    "amem_build_max_context_tokens", "amem_temporal_transition_min_confidence",
}

AMEM_RETRIEVAL_CONFIG_KEYS = {
    "retrieve_num", "amem_max_context_tokens", "amem_query_keywords",
    "amem_expand_links", "amem_typed_retrieval", "amem_typed_expansion_count",
    "amem_expand_related", "amem_relation_min_confidence",
    "amem_temporal_retrieval", "amem_temporal_ordering",
    "amem_temporal_expansion_count", "amem_provenance_retrieval",
    "amem_provenance_max_evidence", "amem_provenance_inject_raw_text",
    "amem_hybrid_retrieval", "amem_hybrid_candidate_count",
    "amem_hybrid_rrf_k", "amem_hybrid_dense_weight",
    "amem_hybrid_bm25_weight", "amem_hybrid_entity_weight",
    "amem_hybrid_timestamp_weight", "amem_hybrid_state_weight",
    "amem_hybrid_graph_weight", "amem_graph_ranking_mode",
    "amem_graph_alpha", "amem_graph_iterations", "amem_graph_tolerance",
    "amem_graph_supersede_weight", "amem_graph_conflict_weight",
    "amem_graph_refine_weight", "amem_graph_support_weight",
    "amem_graph_related_weight",
    "amem_chain_selection", "amem_chain_candidate_count",
    "amem_chain_evidence_count",
    "amem_chain_max_hops", "amem_chain_max_groups",
    "amem_chain_relevance_weight", "amem_chain_coverage_weight",
    "amem_chain_connectivity_weight", "amem_chain_path_weight",
    "amem_chain_temporal_weight", "amem_chain_redundancy_weight",
}


def _split_legacy_amem_params(
    params: Dict[str, Any],
    *,
    include_temporal_state: bool,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Preserve legacy AMEM behavior while assigning each setting an owner."""
    build_config = {
        key: value for key, value in params.items() if key in AMEM_BUILD_CONFIG_KEYS
    }
    retrieval_config = {
        key: value for key, value in params.items() if key not in AMEM_BUILD_CONFIG_KEYS
    }
    if "amem_max_context_tokens" in params:
        build_config.setdefault(
            "amem_build_max_context_tokens",
            params["amem_max_context_tokens"],
        )
    if include_temporal_state and "amem_relation_min_confidence" in params:
        build_config.setdefault(
            "amem_temporal_transition_min_confidence",
            params["amem_relation_min_confidence"],
        )
    return build_config, retrieval_config


@dataclass
class APIConfig:
    """API configuration."""
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"

    modal_api_key: str = ""
    modal_base_url: str = ""

    bigmodel_api_key: str = ""
    bigmodel_base_url: str = "https://open.bigmodel.cn/api/paas/v4"

    azure_api_key: str = ""
    azure_endpoint: str = ""
    azure_api_version: str = "2024-02-01"
    azure_deployment: str = ""

    anthropic_api_key: str = ""
    google_ai_studio_api_keys: str = ""

    default_llm_model: str = "gpt-4o-mini"
    default_embedding_model: str = "text-embedding-3-small"
    embedding_provider: str = "openai"

    judge_model: str = ""
    judge_provider: str = "openai"
    judge_api_key: str = ""
    judge_api_keys: str = ""
    judge_base_url: str = ""
    judge_temperature: float = 1.0
    judge_client_max_tokens: int = 10000
    judge_max_tokens: int = 500
    judge_mcd_max_tokens: int = 2000

    @property
    def use_azure(self) -> bool:
        return bool(self.azure_api_key and self.azure_endpoint)

    @property
    def is_configured(self) -> bool:
        if self.use_azure:
            return True
        return bool(self.openai_api_key or self.modal_api_key)

    def get_judge_model(self) -> str:
        return self.judge_model or self.default_llm_model

    def get_judge_provider(self) -> str:
        return self.judge_provider

    def get_judge_api_key(self) -> str:
        provider = self.get_judge_provider().lower()
        if provider in {"ai_studio", "gemini"}:
            return self.judge_api_keys or self.judge_api_key or self.google_ai_studio_api_keys
        if provider == "vertex":
            return self.judge_api_key
        if provider == "modal":
            return self.judge_api_key or self.modal_api_key
        return self.judge_api_key or self.openai_api_key

    def get_judge_base_url(self) -> str:
        if self.get_judge_provider().lower() == "modal":
            return self.judge_base_url or self.modal_base_url
        return self.judge_base_url or self.openai_base_url

    def get_judge_temperature(self) -> float:
        return self.judge_temperature

    def get_judge_client_max_tokens(self) -> int:
        return self.judge_client_max_tokens

    def get_judge_max_tokens(self, query_type: Optional[str] = None) -> int:
        if query_type == "multi_hop_clinical_deduction":
            return self.judge_mcd_max_tokens
        return self.judge_max_tokens


def load_env_config(env_path: Optional[Path] = None) -> APIConfig:
    """Load environment configuration."""
    if env_path is None:
        env_path = PROJECT_ROOT / ".env"

    if HAS_DOTENV and env_path.exists():
        load_dotenv(env_path)

    openai_api_key = os.getenv("OPENAI_API_KEY", "")
    openai_base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    modal_api_key = os.getenv("MODAL_API_KEY", "") or os.getenv("MODAL_PROXY_TOKEN", "")
    modal_base_url = os.getenv("MODAL_BASE_URL", "")
    bigmodel_api_key = os.getenv("BIGMODEL_API_KEY", "")
    bigmodel_base_url = os.getenv("BIGMODEL_BASE_URL", "https://open.bigmodel.cn/api/paas/v4")

    # Only use BigModel config when BIGMODEL_API_KEY is explicitly set
    if bigmodel_api_key:
        openai_api_key = bigmodel_api_key
        # Only override base_url if BIGMODEL_BASE_URL is explicitly set in env
        env_bigmodel_base_url = os.getenv("BIGMODEL_BASE_URL")
        if env_bigmodel_base_url:
            openai_base_url = env_bigmodel_base_url

    return APIConfig(
        openai_api_key=openai_api_key,
        openai_base_url=openai_base_url,
        modal_api_key=modal_api_key,
        modal_base_url=modal_base_url,
        bigmodel_api_key=bigmodel_api_key,
        bigmodel_base_url=bigmodel_base_url,
        azure_api_key=os.getenv("AZURE_OPENAI_API_KEY", ""),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
        azure_api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT", ""),
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
        google_ai_studio_api_keys=(
            os.getenv("GOOGLE_AI_STUDIO_API_KEYS", "")
            or os.getenv("GOOGLE_AI_STUDIO_API_KEY", "")
            or os.getenv("GOOGLE_API_KEY", "")
            or os.getenv("GEMINI_API_KEY", "")
        ),
        default_llm_model=os.getenv("DEFAULT_LLM_MODEL", "gpt-4o-mini"),
        default_embedding_model=os.getenv("DEFAULT_EMBEDDING_MODEL", "text-embedding-3-small"),
        embedding_provider=os.getenv("EMBEDDING_PROVIDER", "openai"),
        judge_model=os.getenv("JUDGE_MODEL", ""),
        judge_provider=os.getenv("JUDGE_PROVIDER", "openai"),
        judge_api_key=os.getenv("JUDGE_API_KEY", ""),
        judge_api_keys=os.getenv("JUDGE_API_KEYS", ""),
        judge_base_url=os.getenv("JUDGE_BASE_URL", ""),
        judge_temperature=float(os.getenv("JUDGE_TEMPERATURE", "1.0")),
        judge_client_max_tokens=int(os.getenv("JUDGE_CLIENT_MAX_TOKENS", "10000")),
        judge_max_tokens=int(os.getenv("JUDGE_MAX_TOKENS", "500")),
        judge_mcd_max_tokens=int(os.getenv("JUDGE_MCD_MAX_TOKENS", "2000")),
    )


@dataclass
class ModelConfig:
    """Model configuration."""
    provider: str = "openai"
    name: str = "gpt-4o-mini"
    temperature: float = 1.0
    max_tokens: int = 2000
    max_completion_tokens: Optional[int] = None  # For new models (gpt-5.x, o1-*, o3-*)
    api_key: Optional[str] = None
    base_url: Optional[str] = None


@dataclass
class EmbeddingConfig:
    """Embedding configuration."""
    provider: str = "openai"
    model: str = "text-embedding-3-small"
    model_path: Optional[str] = None  # Local model path
    dim: Optional[int] = None  # Embedding dimension
    api_key: Optional[str] = None  # API key for embedding service
    base_url: Optional[str] = None  # Base URL for embedding service


@dataclass
class MethodConfig:
    """Method configuration."""
    method_name: str
    method_type: str  # baseline / rag / agentic_memory
    description: str = ""

    model: ModelConfig = field(default_factory=ModelConfig)
    embedding: Optional[EmbeddingConfig] = None
    memorize_model: Optional[ModelConfig] = None  # Optional separate model for memorize phase
    build_config: Dict[str, Any] = field(default_factory=dict)
    retrieval_config: Dict[str, Any] = field(default_factory=dict)
    agent_params: Dict[str, Any] = field(default_factory=dict)

    # Raw config (preserve all fields)
    raw_config: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MethodConfig":
        """Create config from dict."""
        model_data = data.get("model", {})
        model_config = ModelConfig(
            provider=model_data.get("provider", "openai"),
            name=model_data.get("name", "gpt-4o-mini"),
            temperature=model_data.get("temperature", 1.0),
            max_tokens=model_data.get("max_tokens", 2000),
            max_completion_tokens=model_data.get("max_completion_tokens"),
            api_key=model_data.get("api_key"),
            base_url=model_data.get("base_url"),
        )

        embedding_config = None
        if "embedding" in data:
            emb_data = data["embedding"]
            configured_model = emb_data.get("model", "text-embedding-3-small")
            env_embedding_model = os.getenv("DEFAULT_EMBEDDING_MODEL")
            env_embedding_provider = os.getenv("EMBEDDING_PROVIDER", "").lower()
            raw_model_path = emb_data.get("model_path")
            if raw_model_path is None and configured_model.startswith(("models/", "./", "/")):
                raw_model_path = configured_model
            # Resolve relative model paths against PROJECT_ROOT
            if raw_model_path and not os.path.isabs(raw_model_path):
                raw_model_path = str(PROJECT_ROOT / raw_model_path)
            missing_local_model = bool(
                raw_model_path
                and emb_data.get("provider", "openai").lower() in ("local", "huggingface")
                and not Path(raw_model_path).exists()
            )
            use_env_embedding = bool(
                env_embedding_model
                and env_embedding_provider in ("local", "huggingface")
                and missing_local_model
            )
            embedding_config = EmbeddingConfig(
                provider=env_embedding_provider if use_env_embedding else emb_data.get("provider", "openai"),
                model=env_embedding_model if use_env_embedding else configured_model,
                # A missing local path falls back to the configured Hugging Face ID.
                model_path=None if use_env_embedding or missing_local_model else raw_model_path,
                dim=emb_data.get("dim"),
                api_key=emb_data.get("api_key"),
                base_url=emb_data.get("base_url"),
            )

        memorize_model_config = None
        if "memorize_model" in data:
            mem_data = data["memorize_model"]
            memorize_model_config = ModelConfig(
                provider=mem_data.get("provider", "openai"),
                name=mem_data.get("name", model_config.name),
                temperature=mem_data.get("temperature", 0.7),
                max_tokens=mem_data.get("max_tokens", 1000),
                max_completion_tokens=mem_data.get("max_completion_tokens"),
                api_key=mem_data.get("api_key"),
                base_url=mem_data.get("base_url"),
            )

        # New configs separate snapshot/build semantics from query behavior.
        # Legacy agent_params remain accepted and are treated as one combined view.
        raw_build_config = data.get("build_config", {})
        raw_retrieval_config = data.get("retrieval_config", {})
        legacy_agent_params = data.get("agent_params", {})
        if not isinstance(raw_build_config, dict):
            raise ValueError("build_config must be a mapping")
        if not isinstance(raw_retrieval_config, dict):
            raise ValueError("retrieval_config must be a mapping")
        if not isinstance(legacy_agent_params, dict):
            raise ValueError("agent_params must be a mapping")

        is_amem = str(data.get("method_name", "")).lower().startswith("amem")
        legacy_amem_only = bool(legacy_agent_params) and not (
            "build_config" in data or "retrieval_config" in data
        )
        if is_amem:
            misplaced_build = sorted(set(raw_build_config) & AMEM_RETRIEVAL_CONFIG_KEYS)
            misplaced_retrieval = sorted(
                set(raw_retrieval_config) & AMEM_BUILD_CONFIG_KEYS
            )
            duplicated = sorted(set(raw_build_config) & set(raw_retrieval_config))
            if misplaced_build or misplaced_retrieval or duplicated:
                problems = []
                if misplaced_build:
                    problems.append(
                        "retrieval settings in build_config: "
                        + ", ".join(misplaced_build)
                    )
                if misplaced_retrieval:
                    problems.append(
                        "build settings in retrieval_config: "
                        + ", ".join(misplaced_retrieval)
                    )
                if duplicated:
                    problems.append(
                        "settings declared in both sections: " + ", ".join(duplicated)
                    )
                raise ValueError("Invalid AMEM configuration: " + "; ".join(problems))

            legacy_build, legacy_retrieval = _split_legacy_amem_params(
                legacy_agent_params,
                include_temporal_state=(
                    str(data.get("method_name", "")).lower() == "amem_test"
                ),
            )
            build_config = {**legacy_build, **raw_build_config}
            retrieval_config = {**legacy_retrieval, **raw_retrieval_config}
            if legacy_amem_only and model_config.provider.lower() in {
                "gemini", "vertex", "ai_studio"
            }:
                # Historical adapters replaced AMEM's configured controller with
                # the top-level Gemini transport/model during construction.
                build_config["amem_backend"] = model_config.provider.lower()
                build_config["amem_model"] = model_config.name
        elif legacy_agent_params and not (
            "build_config" in data or "retrieval_config" in data
        ):
            build_config = dict(legacy_agent_params)
            retrieval_config = {}
        else:
            build_config = {**legacy_agent_params, **raw_build_config}
            retrieval_config = dict(raw_retrieval_config)
        merged_agent_params = {**build_config, **retrieval_config}

        # Resolve relative paths against PROJECT_ROOT in both config sections.
        _path_keys = {
            "embedding_model_path", "model_path",
            "working_dir", "q2q_project_path",
        }
        def resolve_paths(params: Dict[str, Any]) -> Dict[str, Any]:
            resolved = {}
            for k, v in params.items():
                is_explicit_amem_path = (
                    k == "amem_embedding_model"
                    and isinstance(v, str)
                    and v.startswith(("./", "models/"))
                )
                if (
                    (k in _path_keys or is_explicit_amem_path)
                    and isinstance(v, str)
                    and v
                    and not os.path.isabs(v)
                ):
                    resolved[k] = str(PROJECT_ROOT / v)
                else:
                    resolved[k] = v
            return resolved

        build_config = resolve_paths(build_config)
        retrieval_config = resolve_paths(retrieval_config)
        merged_agent_params = {**build_config, **retrieval_config}

        return cls(
            method_name=data.get("method_name", "unknown"),
            method_type=data.get("method_type", "baseline"),
            description=data.get("description", ""),
            model=model_config,
            embedding=embedding_config,
            memorize_model=memorize_model_config,
            build_config=build_config,
            retrieval_config=retrieval_config,
            agent_params=merged_agent_params,
            raw_config=data,
        )

    def snapshot_build_config(self) -> Dict[str, Any]:
        """Return only method settings that affect a serialized memory snapshot."""
        if "build_config" in self.raw_config or self.build_config:
            return dict(self.build_config)
        return dict(self.agent_params)

    def query_config(self) -> Dict[str, Any]:
        """Return only method settings that affect retrieval/query execution."""
        if self.retrieval_config:
            return dict(self.retrieval_config)
        return {}


@dataclass
class QueryTypeConfig:
    """Query type configuration."""
    name: str
    abbr: str = ""
    metric: str = "exact_match"
    description: str = ""


@dataclass
class DatasetConfig:
    """Dataset configuration."""
    dataset_name: str
    description: str = ""
    language: str = "zh"

    # Data paths
    data_root_dir: str = ""
    data_files: Dict[str, Any] = field(default_factory=dict)

    # Evaluation config
    evaluation_mode: str = "independent"
    persona_ids: Optional[List[int]] = None
    max_personas: Optional[int] = None
    max_sessions_per_persona: Optional[int] = None
    evaluation_interval: int = 10
    inject_noise: bool = True

    # Query types
    query_types: List[QueryTypeConfig] = field(default_factory=list)

    # Output config
    save_intermediate: bool = True
    save_retrieved_context: bool = True

    # Raw config
    raw_config: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DatasetConfig":
        """Create config from dict."""
        data_cfg = data.get("data", {})
        eval_cfg = data.get("evaluation", {})
        output_cfg = data.get("output", {})

        query_types = []
        for qt in data.get("query_types", []):
            query_types.append(QueryTypeConfig(
                name=qt.get("name", ""),
                abbr=qt.get("abbr", ""),
                metric=qt.get("metric", "exact_match"),
                description=qt.get("description", ""),
            ))

        return cls(
            dataset_name=data.get("dataset_name", "unknown"),
            description=data.get("description", ""),
            language=data.get("language", "zh"),
            data_root_dir=data_cfg.get("root_dir", ""),
            data_files=data_cfg,
            evaluation_mode=eval_cfg.get("mode", "independent"),
            persona_ids=eval_cfg.get("persona_ids"),
            max_personas=eval_cfg.get("max_personas"),
            max_sessions_per_persona=eval_cfg.get("max_sessions_per_persona"),
            evaluation_interval=eval_cfg.get("evaluation_interval", 10),
            inject_noise=eval_cfg.get("inject_noise", True),
            query_types=query_types,
            save_intermediate=output_cfg.get("save_intermediate", True),
            save_retrieved_context=output_cfg.get("save_retrieved_context", True),
            raw_config=data,
        )


def method_config_from_snapshot(snapshot: Dict[str, Any]) -> MethodConfig:
    """Reconstruct a method config from a run_config.json snapshot."""
    if not isinstance(snapshot, dict):
        raise ValueError("Stored method configuration must be a JSON object")

    raw_config = snapshot.get("raw_config")
    if not isinstance(raw_config, dict):
        raise ValueError(
            "Stored method configuration is missing raw_config; "
            "this run cannot be queried without its original configuration"
        )

    config = MethodConfig.from_dict(raw_config)
    # Preserve resolved paths and effective values captured at run startup.
    if isinstance(snapshot.get("build_config"), dict):
        config.build_config = dict(snapshot["build_config"])
    if isinstance(snapshot.get("retrieval_config"), dict):
        config.retrieval_config = dict(snapshot["retrieval_config"])
    if config.method_name.lower().startswith("amem"):
        legacy_query_values = {
            **(
                snapshot.get("agent_params", {})
                if isinstance(snapshot.get("agent_params"), dict)
                else {}
            ),
            **config.retrieval_config,
        }
        if "amem_build_max_context_tokens" not in config.build_config:
            context_budget = legacy_query_values.get("amem_max_context_tokens")
            if context_budget is not None:
                config.build_config["amem_build_max_context_tokens"] = context_budget
        if (
            config.method_name.lower() == "amem_test"
            and "amem_temporal_transition_min_confidence" not in config.build_config
        ):
            transition_threshold = legacy_query_values.get(
                "amem_relation_min_confidence"
            )
            if transition_threshold is not None:
                config.build_config[
                    "amem_temporal_transition_min_confidence"
                ] = transition_threshold
    if isinstance(snapshot.get("agent_params"), dict):
        config.agent_params = dict(snapshot["agent_params"])
    else:
        config.agent_params = {**config.build_config, **config.retrieval_config}
    return config


def dataset_config_from_snapshot(snapshot: Dict[str, Any]) -> DatasetConfig:
    """Reconstruct an effective dataset config from a run_config.json snapshot."""
    if not isinstance(snapshot, dict):
        raise ValueError("Stored dataset configuration must be a JSON object")

    raw_config = snapshot.get("raw_config")
    if not isinstance(raw_config, dict):
        raise ValueError(
            "Stored dataset configuration is missing raw_config; "
            "this run cannot be queried without its original configuration"
        )

    config = DatasetConfig.from_dict(raw_config)
    for field_name in (
        "dataset_name",
        "description",
        "language",
        "data_root_dir",
        "data_files",
        "evaluation_mode",
        "persona_ids",
        "max_personas",
        "max_sessions_per_persona",
        "evaluation_interval",
        "inject_noise",
        "save_intermediate",
        "save_retrieved_context",
    ):
        if field_name in snapshot:
            setattr(config, field_name, snapshot[field_name])

    stored_query_types = snapshot.get("query_types")
    if isinstance(stored_query_types, list):
        config.query_types = [
            QueryTypeConfig(
                name=item.get("name", ""),
                abbr=item.get("abbr", ""),
                metric=item.get("metric", "exact_match"),
                description=item.get("description", ""),
            )
            for item in stored_query_types
            if isinstance(item, dict)
        ]
    return config


class ConfigLoader:
    """Configuration loader."""

    def __init__(self, project_root: Optional[Path] = None):
        self.project_root = project_root or PROJECT_ROOT
        self.configs_dir = self.project_root / "configs"
        self.method_config_dir = self.configs_dir / "method_config"
        self.dataset_config_dir = self.configs_dir / "dataset_config"

        # Cache
        self._api_config: Optional[APIConfig] = None
        self._method_configs: Dict[str, MethodConfig] = {}
        self._dataset_configs: Dict[str, DatasetConfig] = {}

    @property
    def api_config(self) -> APIConfig:
        """Get API config (lazy load)."""
        if self._api_config is None:
            self._api_config = load_env_config(self.project_root / ".env")
        return self._api_config

    def load_method_config(self, config_name: str) -> MethodConfig:
        """Load method config."""
        if config_name in self._method_configs:
            return self._method_configs[config_name]

        requested_path = Path(config_name)
        if requested_path.exists() and requested_path.is_file():
            config_path = requested_path.resolve()
        else:
            config_path = self.method_config_dir / f"{config_name}.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"Method config file not found: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        config = MethodConfig.from_dict(data)
        self._method_configs[config_name] = config
        return config

    def load_dataset_config(self, dataset_name: str) -> DatasetConfig:
        """Load dataset config."""
        if dataset_name in self._dataset_configs:
            return self._dataset_configs[dataset_name]

        config_path = self.dataset_config_dir / f"{dataset_name}.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"Dataset config file not found: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        config = DatasetConfig.from_dict(data)
        self._dataset_configs[dataset_name] = config
        return config

    def list_method_configs(self) -> List[str]:
        """List all available method configs."""
        if not self.method_config_dir.exists():
            return []
        return [f.stem for f in self.method_config_dir.glob("*.yaml")]

    def list_dataset_configs(self) -> List[str]:
        """List all available dataset configs."""
        if not self.dataset_config_dir.exists():
            return []
        return [f.stem for f in self.dataset_config_dir.glob("*.yaml")]

_config_loader: Optional[ConfigLoader] = None


def get_config_loader() -> ConfigLoader:
    """Get global config loader."""
    global _config_loader
    if _config_loader is None:
        _config_loader = ConfigLoader()
    return _config_loader


def get_api_config() -> APIConfig:
    """Get API config."""
    return get_config_loader().api_config


if __name__ == "__main__":
    loader = ConfigLoader()

    print("=== API Config ===")
    api_cfg = loader.api_config
    print(f"  OpenAI Key: {'configured' if api_cfg.openai_api_key else 'not set'}")
    print(f"  Azure: {'configured' if api_cfg.use_azure else 'not set'}")

    print("\n=== Available Method Configs ===")
    for name in loader.list_method_configs():
        print(f"  - {name}")

    print("\n=== Available Dataset Configs ===")
    for name in loader.list_dataset_configs():
        print(f"  - {name}")
