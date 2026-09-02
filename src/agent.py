import os
import time
from pathlib import Path
from typing import Callable, Dict, Any, Optional, List

from src.config import MethodConfig, DatasetConfig, get_api_config
from methods.base import MemoryBuildResult, AgentResponse
from utils.llm_client import get_usage_tracker


class AgentManager:

    SUPPORTED_METHODS = {
        "long_context": ("methods.long_context", "LongContextAgent"),
        "embedding_rag": ("methods.embedding_rag", "EmbeddingRAGAgent"),
        "bm25_rag": ("methods.bm25_rag", "BM25RAGAgent"),
        "amem": ("methods.amem_agent", "AMemAgent"),
        "letta": ("methods.letta_agent", "LettaAgent"),
        "memos": ("methods.memos_agent", "MemOSAgent"),
        "mirix": ("methods.mirix_agent", "MIRIXAgent"),
        "mem0": ("methods.mem0_agent", "Mem0Agent"),
        "smart_mem0": ("methods.smart_mem0_agent", "SmartMem0Agent"),
        "mem1": ("methods.mem1_agent", "Mem1Agent"),
        "memrl": ("methods.memrl_agent", "MemRLAgent"),
        "zep": ("methods.zep_agent", "ZepAgent"),
        "graph_rag": ("methods.graph_rag", "GraphRAGAgent"),
        "lightmem": ("methods.lightmem_agent", "LightMemAgent"),
        "remem": ("methods.remem_agent", "RememAgent"),
        "hipporag": ("methods.hipporag_agent", "HippoRAGAgent"),
        "q2q": ("methods.q2q_agent", "Q2QBenchAgent"),
    }

    def __init__(
        self,
        method_config: MethodConfig,
        dataset_config: DatasetConfig,
        *,
        batch_api: bool = False,
        batch_gcs_uri: Optional[str] = None,
        batch_wait: bool = False,
        batch_manifest_dir: Optional[Path] = None,
        batch_config_hash: str = "",
        batch_progress_callback: Optional[Callable[[str], None]] = None,
    ):
        self.method_config = method_config
        self.dataset_config = dataset_config
        self.method_name = method_config.method_name.lower()
        self._batch_api = batch_api
        self._batch_gcs_uri = batch_gcs_uri
        self._batch_wait = batch_wait
        self._batch_manifest_dir = batch_manifest_dir
        self._batch_config_hash = batch_config_hash
        self._batch_progress_callback = batch_progress_callback

        self._api_config = get_api_config()

        self._agent = None
        self._context_id: Optional[int] = None
        self._agent_start_time = time.time()

        self._initialize_agent()

    def _initialize_agent(self) -> None:
        matched_method = None
        if self.method_name in self.SUPPORTED_METHODS:
            matched_method = self.method_name
        else:
            for method_key in self.SUPPORTED_METHODS:
                if method_key in self.method_name:
                    matched_method = method_key
                    break

        if matched_method is None:
            matched_method = "long_context"

        module_path, class_name = self.SUPPORTED_METHODS[matched_method]
        self._agent = self._create_agent_instance(module_path, class_name, matched_method)

    def _create_agent_instance(self, module_path: str, class_name: str, method_key: str):
        import importlib
        module = importlib.import_module(module_path)
        agent_class = getattr(module, class_name)

        init_params = self._build_agent_params(method_key)
        return agent_class(**init_params)

    def _build_agent_params(self, method_key: str) -> Dict[str, Any]:
        model_config = self.method_config.model
        agent_params = self.method_config.agent_params or {}
        env_embedding_model = os.environ.get("DEFAULT_EMBEDDING_MODEL")
        env_embedding_provider = os.environ.get("EMBEDDING_PROVIDER", "").lower()
        use_env_embedding = bool(
            env_embedding_model
            and env_embedding_provider in ("local", "huggingface")
        )
        amem_embedding_model = agent_params.get("amem_embedding_model", "all-MiniLM-L6-v2")
        use_env_amem_embedding = (
            use_env_embedding
            and amem_embedding_model.startswith(("models/", "./", "/"))
            and not Path(amem_embedding_model).exists()
        )

        effective_max_tokens = model_config.max_completion_tokens or model_config.max_tokens

        params = {
            "model": model_config.name,
            "temperature": model_config.temperature,
            "max_tokens": effective_max_tokens,
            "provider": model_config.provider,
            "api_key": (
                model_config.api_key
                or self._api_config.openai_api_key
            ),
            "base_url": (
                model_config.base_url
                or self._api_config.openai_base_url
            ),
        }

        if (
            self._batch_api
            and model_config.provider.lower() in {"gemini", "vertex", "vertex_ai"}
            and method_key in {"lightmem", "remem", "graph_rag", "memrl"}
        ):
            params.update({
                "vertex_batch_enabled": True,
                "vertex_batch_gcs_uri": self._batch_gcs_uri,
                "vertex_batch_wait": self._batch_wait,
                "vertex_batch_manifest_dir": str(self._batch_manifest_dir) if self._batch_manifest_dir else None,
                "vertex_batch_config_hash": self._batch_config_hash,
                "vertex_batch_progress_callback": self._batch_progress_callback,
            })

        if method_key == "long_context":
            params.update({
                "max_context_tokens": agent_params.get("max_context_tokens", 100000),
                "truncation_strategy": agent_params.get("truncation_strategy", "oldest_first"),
            })

        elif method_key == "embedding_rag":
            params.update({
                "top_k": agent_params.get("top_k", 5),
                "chunk_size": agent_params.get("chunk_size", 512),
                "chunk_overlap": agent_params.get("chunk_overlap", 50),
                "max_context_tokens": agent_params.get("max_context_tokens", 120000),
            })
            if self.method_config.embedding:
                params["embedding_model"] = self.method_config.embedding.model
                params["embedding_provider"] = self.method_config.embedding.provider
                if self.method_config.embedding.model_path:
                    params["embedding_model_path"] = self.method_config.embedding.model_path

        elif method_key == "bm25_rag":
            params.update({
                "top_k": agent_params.get("top_k", 5),
                "k1": agent_params.get("k1", 1.5),
                "b": agent_params.get("b", 0.75),
                "language": agent_params.get("language", "auto"),
                "chunk_size": agent_params.get("chunk_size", 512),
                "chunk_overlap": agent_params.get("chunk_overlap", 50),
                "max_context_tokens": agent_params.get("max_context_tokens", 120000),
            })

        elif method_key == "mem0":
            params.update({
                "retrieve_num": agent_params.get("retrieve_num", 5),
                "chunk_size_tokens": agent_params.get("chunk_size_tokens"),
                "max_context_tokens": agent_params.get("max_context_tokens"),
                "turn_by_turn": agent_params.get("turn_by_turn", False),
                "max_recent_turns": agent_params.get("max_recent_turns", 10),
            })
            if self.method_config.embedding:
                params["embedding_model"] = self.method_config.embedding.model
                params["embedding_provider"] = self.method_config.embedding.provider
                if self.method_config.embedding.model_path:
                    params["embedding_model_path"] = self.method_config.embedding.model_path

        elif method_key == "smart_mem0":
            params.update({
                "retrieve_num": agent_params.get("retrieve_num", 8),
                "max_context_tokens": agent_params.get("max_context_tokens", 32768),
                "max_question_tokens": agent_params.get("max_question_tokens", 1200),
                "enable_memory_write": agent_params.get("enable_memory_write", True),
                "enable_planner": agent_params.get("enable_planner", True),
                "enable_unified_controller": agent_params.get(
                    "enable_unified_controller", True
                ),
                "enable_slot_support_validation": agent_params.get(
                    "enable_slot_support_validation", False
                ),
                "enable_replan": agent_params.get("enable_replan", False),
                "enable_zero_result_recovery": agent_params.get(
                    "enable_zero_result_recovery", True
                ),
                "enable_planner_repair": agent_params.get(
                    "enable_planner_repair", False
                ),
                "frozen_memory_path": agent_params.get("frozen_memory_path"),
                "write_window_max_turns": agent_params.get("write_window_max_turns", 64),
                "write_window_max_tokens": agent_params.get("write_window_max_tokens", 3000),
                "write_context_mode": agent_params.get("write_context_mode", "full"),
                "write_prior_belief_limit": agent_params.get(
                    "write_prior_belief_limit", 8
                ),
                "write_provisional_limit": agent_params.get(
                    "write_provisional_limit", 4
                ),
            })
            if self.method_config.embedding:
                params["embedding_model"] = self.method_config.embedding.model
                params["embedding_provider"] = self.method_config.embedding.provider
                if self.method_config.embedding.model_path:
                    params["embedding_model_path"] = self.method_config.embedding.model_path

        elif method_key == "amem":
            params.update({
                "retrieve_num": agent_params.get("retrieve_num", 5),
                "amem_backend": agent_params.get("amem_backend", "openai"),
                "amem_model": agent_params.get("amem_model", model_config.name),
                "amem_embedding_model": (
                    env_embedding_model
                    if use_env_amem_embedding
                    else amem_embedding_model
                ),
                "amem_evo_threshold": agent_params.get("amem_evo_threshold", 100),
                "amem_max_tokens": agent_params.get("amem_max_tokens"),
                "amem_max_context_tokens": agent_params.get("amem_max_context_tokens", 200000),
                "amem_chunk_size_tokens": agent_params.get("amem_chunk_size_tokens"),
            })

        elif method_key == "letta":
            params.update({
                "retrieve_num": agent_params.get("retrieve_num", 5),
                "context_window": agent_params.get("context_window", 128000),
                "embedding_model": agent_params.get("embedding_model", "embedding-3"),
                "embedding_provider": agent_params.get("embedding_provider", "openai"),
                "embedding_dim": agent_params.get("embedding_dim", 2048),
                "embedding_chunk_size": agent_params.get("embedding_chunk_size", 300),
                "memory_persona": agent_params.get(
                    "memory_persona",
                    "I am an assistant helping with medical QA while preserving long-term memory.",
                ),
                "memory_human": agent_params.get(
                    "memory_human",
                    "The user is a patient in a longitudinal medical dialogue setting.",
                ),
                # Memorize chunking parameters
                "memorize_chunk_tokens": agent_params.get("memorize_chunk_tokens", 6000),
                "memorize_chunk_overlap_tokens": agent_params.get("memorize_chunk_overlap_tokens", 200),
                # Query truncation parameters
                "max_input_tokens": agent_params.get("max_input_tokens", 8000),
                "max_question_tokens": agent_params.get("max_question_tokens", 4096),
                "max_context_tokens": agent_params.get("max_context_tokens", 120000),
            })
            if self.method_config.embedding:
                params["embedding_model"] = self.method_config.embedding.model
                params["embedding_provider"] = self.method_config.embedding.provider
                if self.method_config.embedding.model_path:
                    params["embedding_model_path"] = self.method_config.embedding.model_path
                if self.method_config.embedding.dim:
                    params["embedding_dim"] = self.method_config.embedding.dim

        elif method_key == "memos":
            params.update({
                "retrieve_num": agent_params.get("retrieve_num", 5),
                "memos_backend": agent_params.get("memos_backend", "openai"),
                "memos_model": agent_params.get("memos_model", model_config.name),
                "text_mem_type": agent_params.get("text_mem_type", "general_text"),
                "embedding_dim": agent_params.get("embedding_dim"),
                "observability_search": agent_params.get("observability_search", True),
                "max_input_tokens": agent_params.get("max_input_tokens", 8000),
                "max_question_tokens": agent_params.get("max_question_tokens", 4096),
                "max_context_tokens": agent_params.get("max_context_tokens", 120000),
            })
            if self.method_config.embedding:
                params["embedding_model"] = self.method_config.embedding.model
                params["embedding_provider"] = self.method_config.embedding.provider
                if self.method_config.embedding.model_path:
                    params["embedding_model_path"] = self.method_config.embedding.model_path

        elif method_key == "mirix":
            params.update({
                "retrieve_num": agent_params.get("retrieve_num", 5),
                "memorize_chunk_tokens": agent_params.get("memorize_chunk_tokens", 2500),
                "memorize_chunk_overlap_tokens": agent_params.get("memorize_chunk_overlap_tokens", 200),
                "query_memory_item_tokens": agent_params.get("query_memory_item_tokens", 300),
                "query_memory_context_tokens": agent_params.get("query_memory_context_tokens", 1800),
                "max_input_tokens": agent_params.get("max_input_tokens", 8000),
                "max_question_tokens": agent_params.get("max_question_tokens", 4096),
                "max_context_tokens": agent_params.get("max_context_tokens", 120000),
                # Query mode: use MIRIX native send_message for memory-aware responses
                "use_native_query": agent_params.get("use_native_query", True),
            })
            if self.method_config.embedding:
                params["embedding_model"] = self.method_config.embedding.model
                params["embedding_provider"] = self.method_config.embedding.provider
                if self.method_config.embedding.model_path:
                    params["embedding_model_path"] = self.method_config.embedding.model_path
                if self.method_config.embedding.dim:
                    params["embedding_dim"] = self.method_config.embedding.dim
                # For local embedding: device specification (cuda, cpu, mps)
                if self.method_config.embedding.provider == "local":
                    params["embedding_device"] = agent_params.get("embedding_device", None)

        elif method_key == "mem1":
            # MEM1 uses vLLM for model serving
            params.update({
                "vllm_url": agent_params.get("vllm_url", "http://localhost:8014"),
                "model_path": agent_params.get("model_path", model_config.name),
                "max_state_tokens": agent_params.get("max_state_tokens", 4096),
                "max_input_tokens": agent_params.get("max_input_tokens", 4096),
                "max_context_tokens": agent_params.get("max_context_tokens", 8192),
                "use_chat_api": agent_params.get("use_chat_api", True),
            })

        elif method_key == "memrl":
            params.update({
                # Retrieval configuration
                "retrieve_num": agent_params.get("retrieve_num", 5),
                "candidate_top_k": agent_params.get("candidate_top_k", 12),
                "similarity_threshold": agent_params.get("similarity_threshold", 0.2),
                # Memory building configuration
                "max_new_items_per_memorize": agent_params.get("max_new_items_per_memorize", 8),
                "max_experience_tokens": agent_params.get("max_experience_tokens", 700),
                "memorize_chunk_tokens": agent_params.get("memorize_chunk_tokens", 6000),
                "memorize_chunk_overlap_tokens": agent_params.get("memorize_chunk_overlap_tokens", 200),
                "query_memory_item_tokens": agent_params.get("query_memory_item_tokens", 300),
                "query_memory_context_tokens": agent_params.get("query_memory_context_tokens", 1800),
                "max_concurrent_api_calls": agent_params.get("max_concurrent_api_calls", 4),
                # Strategy configuration
                "build_strategy": agent_params.get("build_strategy", "proceduralization"),
                "retrieve_strategy": agent_params.get("retrieve_strategy", "query"),
                "update_strategy": agent_params.get("update_strategy", "adjustment"),
                # Q-learning / RL configuration
                "epsilon": agent_params.get("epsilon", 0.1),
                "gamma": agent_params.get("gamma", 0.0),
                "learning_rate": agent_params.get("learning_rate", 0.2),
                "initial_q": agent_params.get("initial_q", 0.5),
                "q_init_pos": agent_params.get("q_init_pos", 0.5),
                "q_init_neg": agent_params.get("q_init_neg", 0.0),
                "success_reward": agent_params.get("success_reward", 1.0),
                "failure_reward": agent_params.get("failure_reward", -1.0),
                "weight_sim": agent_params.get("weight_sim", 0.5),
                "weight_q": agent_params.get("weight_q", 0.5),
                "utility_mix_lambda": agent_params.get("utility_mix_lambda", 0.6),
                # Token limits
                "max_input_tokens": agent_params.get("max_input_tokens", 8000),
                "max_context_tokens": agent_params.get("max_context_tokens", 120000),
                "max_question_tokens": agent_params.get("max_question_tokens", 4096),
                # Embedding (default, may be overridden below)
                "embedding_model": agent_params.get("embedding_model", "all-MiniLM-L6-v2"),
                "embedding_provider": agent_params.get("embedding_provider", "local"),
            })
            if self.method_config.embedding:
                params["embedding_model"] = self.method_config.embedding.model
                params["embedding_provider"] = self.method_config.embedding.provider
                if self.method_config.embedding.model_path:
                    params["embedding_model_path"] = self.method_config.embedding.model_path

        elif method_key == "zep":
            params.update({
                "retrieve_num": agent_params.get("retrieve_num", 5),
                "chunk_size": agent_params.get("chunk_size", 512),
            })
            # Zep API Key
            zep_api_key = agent_params.get("zep_api_key") or os.environ.get("ZEP_API_KEY")
            if zep_api_key:
                params["zep_api_key"] = zep_api_key

            # Azure OpenAI
            if self._api_config.use_azure:
                params.update({
                    "use_azure": True,
                    "azure_endpoint": self._api_config.azure_endpoint,
                    "azure_api_key": self._api_config.azure_api_key,
                    "azure_api_version": self._api_config.azure_api_version,
                })

        elif method_key == "graph_rag":
            params.update({
                "top_k": agent_params.get("top_k", 5),
                "chunk_size": agent_params.get("chunk_size", 4096),
                "chunk_overlap": agent_params.get("chunk_overlap", 200),
                "edges_threshold": agent_params.get("edges_threshold", 0.8),
            })
            if self.method_config.embedding:
                params["embedding_model"] = self.method_config.embedding.model
                params["embedding_provider"] = self.method_config.embedding.provider
                if self.method_config.embedding.model_path:
                    params["embedding_model_path"] = self.method_config.embedding.model_path

        elif method_key == "lightmem":
            params.update({
                # Retrieval configuration
                "retrieve_num": agent_params.get("retrieve_num", 5),
                # LightMem core feature switches
                "pre_compress": agent_params.get("pre_compress", False),
                "topic_segment": agent_params.get("topic_segment", True),
                # Strategy configuration
                "index_strategy": agent_params.get("index_strategy", "embedding"),
                "retrieve_strategy": agent_params.get("retrieve_strategy", "embedding"),
                "update_mode": agent_params.get("update_mode", "offline"),
                "extraction_mode": agent_params.get("extraction_mode", "flat"),
                "messages_use": agent_params.get("messages_use", "user_only"),
                # LightMem internal LLM settings
                "lightmem_temperature": agent_params.get("lightmem_temperature", 0.1),
                "lightmem_max_tokens": agent_params.get("lightmem_max_tokens", 2000),
                "lightmem_top_p": agent_params.get("lightmem_top_p", 0.1),
                # Token limits
                "max_context_tokens": agent_params.get("max_context_tokens", 120000),
            })
            # Embedding configuration
            if self.method_config.embedding:
                params["embedding_model"] = self.method_config.embedding.model
                params["embedding_provider"] = self.method_config.embedding.provider
                if self.method_config.embedding.model_path:
                    params["embedding_model_path"] = self.method_config.embedding.model_path
                if self.method_config.embedding.dim:
                    params["embedding_dim"] = self.method_config.embedding.dim
                # Embedding API configuration (for openai/huggingface_api providers)
                if self.method_config.embedding.api_key:
                    params["embedding_api_key"] = self.method_config.embedding.api_key
                if self.method_config.embedding.base_url:
                    params["embedding_base_url"] = self.method_config.embedding.base_url

        elif method_key == "remem":
            # ReMem: Reasoning with Episodic Memory
            params.update({
                # Information extraction method
                "extract_method": agent_params.get("extract_method", "episodic_gist"),
                # Graph configuration
                "is_directed_graph": agent_params.get("is_directed_graph", False),
                "synonymy_edge_sim_threshold": agent_params.get("synonymy_edge_sim_threshold", 0.8),
                "synonymy_edge_topk": agent_params.get("synonymy_edge_topk", 10),
                # Retrieval configuration
                "retrieval_top_k": agent_params.get("retrieval_top_k", 20),
                "qa_top_k": agent_params.get("qa_top_k", 10),
                "linking_top_k": agent_params.get("linking_top_k", 5),
                "damping": agent_params.get("damping", 0.5),
                "passage_node_weight": agent_params.get("passage_node_weight", 0.05),
                # Agent configuration
                "use_agent": agent_params.get("use_agent", False),
                "agent_fixed_tools": agent_params.get("agent_fixed_tools", False),
                "agent_max_steps": agent_params.get("agent_max_steps", 5),
                # Cache configuration
                "use_cache": agent_params.get("use_cache", True),
                "force_index_from_scratch": agent_params.get("force_index_from_scratch", False),
                "save_openie": agent_params.get("save_openie", True),
                # API concurrency configuration
                "extraction_max_workers": agent_params.get("extraction_max_workers", 5),
                # Text preprocessing
                "text_preprocessor_class_name": agent_params.get(
                    "text_preprocessor_class_name", "SentenceWindowPreprocessor"
                ),
                # Chunking configuration
                "chunk_size_tokens": agent_params.get("chunk_size_tokens", 8000),
                "chunk_overlap_tokens": agent_params.get("chunk_overlap_tokens", 200),
                # Embedding settings
                "embedding_batch_size": agent_params.get("embedding_batch_size", 16),
                "embedding_max_seq_len": agent_params.get("embedding_max_seq_len", 512),
                # Token limits
                "max_input_tokens": agent_params.get("max_input_tokens", 8000),
                "max_context_tokens": agent_params.get("max_context_tokens", 120000),
                # Working directory (optional)
                "working_dir": agent_params.get("working_dir", None),
            })
            # Embedding configuration
            if self.method_config.embedding:
                params["embedding_model"] = self.method_config.embedding.model
                params["embedding_provider"] = self.method_config.embedding.provider
                if self.method_config.embedding.model_path:
                    params["embedding_model_path"] = self.method_config.embedding.model_path
                if self.method_config.embedding.dim:
                    params["embedding_dim"] = self.method_config.embedding.dim
                # Embedding API configuration (for openai/api providers)
                if self.method_config.embedding.api_key:
                    params["embedding_api_key"] = self.method_config.embedding.api_key
                if self.method_config.embedding.base_url:
                    params["embedding_base_url"] = self.method_config.embedding.base_url

        elif method_key == "hipporag":
            # HippoRAG 2: Graph-Based RAG Framework
            params.update({
                # OpenIE mode
                "openie_mode": agent_params.get("openie_mode", "online"),
                # Graph configuration
                "is_directed_graph": agent_params.get("is_directed_graph", False),
                "synonymy_edge_sim_threshold": agent_params.get("synonymy_edge_sim_threshold", 0.8),
                "synonymy_edge_topk": agent_params.get("synonymy_edge_topk", 2047),
                # Retrieval configuration
                "linking_top_k": agent_params.get("linking_top_k", 5),
                "retrieval_top_k": agent_params.get("retrieval_top_k", 200),
                "qa_top_k": agent_params.get("qa_top_k", 5),
                "damping": agent_params.get("damping", 0.5),
                "passage_node_weight": agent_params.get("passage_node_weight", 0.05),
                # Cache configuration
                "force_index_from_scratch": agent_params.get("force_index_from_scratch", False),
                "force_openie_from_scratch": agent_params.get("force_openie_from_scratch", False),
                "save_openie": agent_params.get("save_openie", True),
                # Chunking configuration
                "chunk_size_tokens": agent_params.get("chunk_size_tokens", 8000),
                "chunk_overlap_tokens": agent_params.get("chunk_overlap_tokens", 200),
                # Embedding settings
                "embedding_batch_size": agent_params.get("embedding_batch_size", 16),
                "embedding_max_seq_len": agent_params.get("embedding_max_seq_len", 512),
                # Token limits
                "max_input_tokens": agent_params.get("max_input_tokens", 8000),
                "max_context_tokens": agent_params.get("max_context_tokens", 120000),
                # Working directory (optional)
                "working_dir": agent_params.get("working_dir", None),
            })
            # Embedding configuration
            if self.method_config.embedding:
                params["embedding_model"] = self.method_config.embedding.model
                params["embedding_provider"] = self.method_config.embedding.provider
                if self.method_config.embedding.model_path:
                    params["embedding_model_path"] = self.method_config.embedding.model_path
                if self.method_config.embedding.dim:
                    params["embedding_dim"] = self.method_config.embedding.dim
                # Embedding API configuration (for openai/api providers)
                if self.method_config.embedding.api_key:
                    params["embedding_api_key"] = self.method_config.embedding.api_key
                if self.method_config.embedding.base_url:
                    params["embedding_base_url"] = self.method_config.embedding.base_url

        elif method_key == "q2q":
            params.update({
                "q2q_project_path": agent_params.get("q2q_project_path", ""),
                "alpha": agent_params.get("alpha", 0.7),
                "top_k_per_sub": agent_params.get("top_k_per_sub", 20),
                "top_n": agent_params.get("top_n", 5),
                "top_k_q2c": agent_params.get("top_k_q2c", 3),
                "num_fake_queries": agent_params.get("num_fake_queries", 10),
                "storage_backend": agent_params.get("storage_backend", "chromadb"),
                "language": agent_params.get("language", "zh"),
                "max_context_tokens": agent_params.get("max_context_tokens", 120000),
                "embedding_device": agent_params.get("embedding_device", "cpu"),
                "version_threshold": agent_params.get("version_threshold", 0.80),
                "version_chain_depth": agent_params.get("version_chain_depth", 3),
                "fq_confidence_threshold": agent_params.get("fq_confidence_threshold", 0.80),
            })
            if self.method_config.embedding:
                params["embedding_model"] = self.method_config.embedding.model
                params["embedding_provider"] = self.method_config.embedding.provider
                if self.method_config.embedding.model_path:
                    params["embedding_model_path"] = self.method_config.embedding.model_path

        return params

    def send_message(
        self,
        message: str,
        memorizing: bool = False,
        context_id: Optional[int] = None,
        is_last_session: bool = False,
        **kwargs,
    ) -> Any:
        if context_id is not None and context_id != self._context_id:
            self._context_id = context_id
            self._agent.set_context_id(context_id)

        if memorizing:
            return self._handle_memorize(message, is_last_session=is_last_session, **kwargs)
        else:
            return self._handle_query(message, **kwargs)

    def supports_batch_queries(self) -> bool:
        """Return whether this adapter can separate retrieval from final generation."""
        adapter_support = getattr(self._agent, "supports_batch_queries", None)
        if callable(adapter_support) and not adapter_support():
            return False
        return bool(
            self.method_config.model.provider.lower() in {"gemini", "vertex", "vertex_ai"}
            and hasattr(self._agent, "prepare_batch_query")
            and hasattr(self._agent, "finalize_batch_query")
        )

    def get_batch_llm_client(self):
        """Expose the managed client only for the evaluator's batch transport."""
        if not self.supports_batch_queries():
            return None
        return getattr(self._agent, "_llm_client", None)

    def prepare_batch_query(self, message: str, **kwargs) -> Dict[str, Any]:
        """Prepare local retrieval and an immutable final-answer request."""
        if not self.supports_batch_queries():
            raise RuntimeError(f"{self.method_name} does not support Vertex batch queries")
        context_id = kwargs.pop("context_id", None)
        if context_id is not None and context_id != self._context_id:
            self._context_id = context_id
            self._agent.set_context_id(context_id)
        get_usage_tracker().set_phase("query")
        return self._agent.prepare_batch_query(message, **kwargs)

    def finalize_batch_query(
        self,
        prepared: Dict[str, Any],
        content: str,
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
    ) -> AgentResponse:
        """Convert one Vertex output row to the method's normal response shape."""
        response = self._agent.finalize_batch_query(prepared, content)
        record_usage = getattr(self._agent, "record_batch_query_usage", None)
        if callable(record_usage) and input_tokens is not None and output_tokens is not None:
            record_usage(response, input_tokens, output_tokens)
        return response

    def supports_memory_snapshots(self) -> bool:
        """Return whether this agent exposes a complete durable memory state."""
        return bool(
            self._agent
            and callable(getattr(self._agent, "export_memory_state", None))
            and callable(getattr(self._agent, "import_memory_state", None))
        )

    def export_memory_state(self) -> Dict[str, Any]:
        if not self.supports_memory_snapshots():
            raise RuntimeError(f"{self.method_name} does not support memory snapshots")
        return self._agent.export_memory_state()

    def import_memory_state(self, state: Dict[str, Any]) -> None:
        if not self.supports_memory_snapshots():
            raise RuntimeError(f"{self.method_name} does not support memory snapshots")
        self._agent.import_memory_state(state)
        if self._context_id is not None:
            self._agent.set_context_id(self._context_id)

    def _handle_memorize(self, message: str, is_last_session: bool = False, **kwargs) -> MemoryBuildResult:
        get_usage_tracker().set_phase("memorize")
        start_time = time.time()

        result = self._agent.memorize(message, is_last_session=is_last_session, **kwargs)

        memory_time = time.time() - start_time

        if isinstance(result, MemoryBuildResult):
            result.time_cost = memory_time
            return result

        if isinstance(result, dict):
            return MemoryBuildResult(
                success=result.get("success", True),
                method=result.get("method", self.method_name),
                action=result.get("action", "add_to_memory"),
                input_content=message,
                stored_content=result.get("stored_content", message),
                memory_entries=result.get("memory_entries", []),
                chunk_count=result.get("chunk_count", self._agent.memory_size),
                time_cost=memory_time,
                extra={}
            )

        return MemoryBuildResult(
            success=True,
            method=self.method_name,
            action="add_to_memory",
            input_content=message,
            stored_content=message,
            memory_entries=[],
            chunk_count=self._agent.memory_size,
            time_cost=memory_time,
        )

    def _handle_query(self, message: str, **kwargs) -> Dict[str, Any]:
        get_usage_tracker().set_phase("query")
        from utils.templates import get_template_manager
        template_manager = get_template_manager(self.dataset_config.dataset_name)
        system_message = template_manager.get_system_message()

        start_time = time.time()

        response = self._agent.query(message, system_message=system_message, **kwargs)

        query_time = time.time() - start_time

        return {
            "output": response.output,
            "query_time": query_time,
            "retrieved_count": response.retrieved_count,
            "retrieved_memories": response.retrieved_memories, 
            "extra": response.extra,
        }

    def reset(self) -> None:
        if self._agent:
            self._agent.reset()
        self._context_id = None
        self._agent_start_time = time.time()

    def set_context_id(self, context_id: int) -> None:
        self._context_id = context_id
        if self._agent:
            self._agent.set_context_id(context_id)

    def get_info(self) -> Dict[str, Any]:
        info = {
            "method_name": self.method_name,
            "context_id": self._context_id,
        }
        if self._agent:
            info.update(self._agent.get_info())
        return info


def create_agent_manager(
    method_config: MethodConfig,
    dataset_config: DatasetConfig,
) -> AgentManager:
    return AgentManager(method_config, dataset_config)


def list_available_methods() -> List[str]:
    return list(AgentManager.SUPPORTED_METHODS.keys())
