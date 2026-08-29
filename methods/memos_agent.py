"""MemOS agent adapter for MedMemoryBench."""

from __future__ import annotations

import importlib
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import BaseAgent, MemoryBuildResult, AgentResponse
from utils.llm_client import (
    create_llm_client,
    format_messages,
    BaseLLMClient,
    is_google_ai_studio_provider,
    is_gemini_provider,
    is_hybrid_gemini_provider,
    is_vertex_gemini_provider,
)

logger = logging.getLogger(__name__)


class MemOSAgent(BaseAgent):
    """Adapter that uses MemOS's official memory API.

    This implementation uses MemOS's native memory system:
    - memorize(): uses MemOS extract() + add() for memory storage
    - query(): uses MemOS search() for retrieval, then LLM for response
    """

    METHOD_TYPE = "agentic_memory"

    BIGMODEL_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"

    DEFAULT_EXTRACTOR_TEMPERATURE = 0
    DEFAULT_EXTRACTOR_MAX_TOKENS = 4096

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 1.0,
        max_tokens: int = 2000,
        provider: str = "openai",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        retrieve_num: int = 5,
        memos_backend: str = "openai",
        memos_model: Optional[str] = None,
        memos_temperature: float = DEFAULT_EXTRACTOR_TEMPERATURE,
        memos_max_tokens: int = DEFAULT_EXTRACTOR_MAX_TOKENS,
        text_mem_type: str = "naive_text",
        embedding_model: Optional[str] = None,
        embedding_model_path: Optional[str] = None,
        embedding_dim: int = 512,
        embedding_provider: str = "local",
        **kwargs,
    ):
        super().__init__(model, temperature, max_tokens, **kwargs)

        self.retrieve_num = retrieve_num
        self._provider = provider.lower()
        self._llm_client_kwargs = dict(kwargs.get("llm_client_kwargs", {}))
        self.memos_backend = memos_backend
        self.memos_model = memos_model or model
        self.memos_temperature = memos_temperature
        self.memos_max_tokens = memos_max_tokens
        self.text_mem_type = text_mem_type

        # Token limits configuration
        self.max_input_tokens = int(kwargs.get("max_input_tokens", 8000))
        self.max_context_tokens = int(kwargs.get("max_context_tokens", 120000))
        self.max_question_tokens = int(kwargs.get("max_question_tokens", 4096))

        # Calculate max_memory_tokens with floor protection
        default_memory_tokens = self.max_context_tokens - self.max_question_tokens - max_tokens - 500
        self.max_memory_tokens = max(0, int(kwargs.get("max_memory_tokens", default_memory_tokens)))

        # Embedding configuration
        # embedding_provider controls backend: "local" -> sentence_transformer, others -> universal_api
        self.embedding_model = embedding_model
        self.embedding_model_path = embedding_model_path
        self.embedding_dim = embedding_dim
        self.embedding_provider = embedding_provider

        # API configuration
        self._memos_api_key = api_key
        if not is_gemini_provider(provider):
            self._memos_api_key = (
                self._memos_api_key
                or os.getenv("BIGMODEL_API_KEY")
                or os.getenv("OPENAI_API_KEY", "")
            )
        self._memos_api_base = (
            base_url
            or os.getenv("BIGMODEL_BASE_URL")
            or os.getenv("OPENAI_BASE_URL")
            or self.BIGMODEL_BASE_URL
        )

        # Create llm_client for response generation (ensures token tracking)
        self._llm_client: BaseLLMClient = create_llm_client(
            provider=provider,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
            base_url=base_url,
            **kwargs.get("llm_client_kwargs", {}),
        )

        # Memory system instances per context
        self._memory_systems: Dict[int, Any] = {}

        # Counter for actual stored memory items (not raw chunks)
        self._stored_memory_count: int = 0

        # Load MemOS classes
        self._MemoryFactory, self._MemoryConfigFactory = self._load_memos_classes()

    def _truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """Truncate text to max_tokens."""
        if not text or max_tokens <= 0:
            return ""
        tokens = self._tokenizer.encode(text)
        if len(tokens) <= max_tokens:
            return text
        return self._tokenizer.decode(tokens[:max_tokens])

    def _load_memos_classes(self) -> tuple:
        """Load vendored memOS package from methods/memOS/MemOS/src.

        Returns:
            Tuple of (MemoryFactory, MemoryConfigFactory) classes.
        """
        memos_src = Path(__file__).resolve().parent / "memOS" / "MemOS" / "src"
        if not memos_src.exists():
            raise ImportError("MemOS source folder not found at methods/memOS/MemOS/src")

        memos_src_str = str(memos_src)
        if memos_src_str not in sys.path:
            sys.path.insert(0, memos_src_str)

        memory_factory_module = importlib.import_module("memos.memories.factory")
        memory_config_module = importlib.import_module("memos.configs.memory")
        return (
            getattr(memory_factory_module, "MemoryFactory"),
            getattr(memory_config_module, "MemoryConfigFactory"),
        )

    def _get_context_id(self) -> int:
        """Get current context ID, defaulting to 0 if not set."""
        return self._context_id if self._context_id is not None else 0

    def _build_memory_config(self, context_id: int) -> Any:
        """Build memory configuration based on text_mem_type.

        Args:
            context_id: Context identifier for collection naming.

        Returns:
            MemoryConfigFactory instance with configured backend.
        """
        use_vertex_gemini = is_vertex_gemini_provider(self._provider)
        use_ai_studio = is_google_ai_studio_provider(self._provider)
        use_hybrid_gemini = is_hybrid_gemini_provider(self._provider)
        use_gemini = use_vertex_gemini or use_ai_studio or use_hybrid_gemini
        extractor_config: Dict[str, Any] = {
            "model_name_or_path": self.model if use_gemini else self.memos_model,
            "temperature": getattr(
                self, "memos_temperature", self.DEFAULT_EXTRACTOR_TEMPERATURE
            ),
            "max_tokens": getattr(
                self, "memos_max_tokens", self.DEFAULT_EXTRACTOR_MAX_TOKENS
            ),
        }
        if self._provider == "openrouter":
            extractor_config["extra_body"] = {
                key: value
                for key, value in (
                    ("provider", self._llm_client_kwargs.get("provider_routing")),
                    ("service_tier", self._llm_client_kwargs.get("service_tier")),
                )
                if value is not None
            } or None
        if use_ai_studio or use_hybrid_gemini:
            extractor_config.update({
                "gemini_provider": self._provider,
                "api_key": self._memos_api_key,
            })
        elif not use_vertex_gemini:
            extractor_config.update({
                "api_key": self._memos_api_key,
                "api_base": self._memos_api_base,
            })

        base_config: Dict[str, Any] = {
            "extractor_llm": {
                "backend": (
                    self._provider
                    if use_vertex_gemini
                    else "gemini" if use_ai_studio or use_hybrid_gemini else self.memos_backend
                ),
                "config": extractor_config,
            }
        }

        # Local model IDs are valid SentenceTransformer inputs and download from
        # Hugging Face when no on-disk model path is available.
        if self.text_mem_type in ["general_text", "tree_text", "simple_tree_text", "pref_text"]:
            if self.embedding_provider.lower() in {"local", "huggingface"}:
                base_config["embedder"] = {
                    "backend": "sentence_transformer",
                    "config": {
                        "model_name_or_path": self.embedding_model_path or self.embedding_model,
                        "embedding_dims": self.embedding_dim,
                        "trust_remote_code": True,
                    },
                }
            else:
                base_config["embedder"] = {
                    "backend": "universal_api",
                    "config": {
                        "provider": "openai",
                        "model_name_or_path": self.embedding_model or "text-embedding-3-small",
                        "api_key": self._memos_api_key,
                        "base_url": self._memos_api_base,
                        "embedding_dims": self.embedding_dim,
                    },
                }

            # Vector database configuration (required for general_text)
            # Use Qdrant with in-memory storage for each context
            base_config["vector_db"] = {
                "backend": "qdrant",
                "config": {
                    "collection_name": f"memos_ctx_{context_id}",
                    "vector_dimension": self.embedding_dim,
                    "distance_metric": "cosine",
                },
            }

        return self._MemoryConfigFactory(
            backend=self.text_mem_type,
            config=base_config,
        )

    def _create_memory_system(self, context_id: int) -> Any:
        """Create a MemOS memory system for the given context.

        Args:
            context_id: Context identifier.

        Returns:
            MemOS memory system instance.
        """
        config = self._build_memory_config(context_id)
        memory = self._MemoryFactory.from_config(config)
        self._memory_systems[context_id] = memory
        return memory

    def _get_memory_system(self, context_id: int) -> Any:
        """Get or create memory system for the given context.

        Args:
            context_id: Context identifier.

        Returns:
            MemOS memory system instance.
        """
        system = self._memory_systems.get(context_id)
        if system is None:
            system = self._create_memory_system(context_id)
        return system

    def memorize(self, text: str, **kwargs) -> MemoryBuildResult:
        """Store memory using MemOS's official extract() and add() methods.

        This uses MemOS's native LLM-based memory extraction pipeline.

        Args:
            text: Input text to memorize.
            **kwargs: Additional arguments (unused).

        Returns:
            MemoryBuildResult with extraction and storage details.
        """
        context_id = self._get_context_id()
        memory_system = self._get_memory_system(context_id)

        # Truncate input text
        bounded_text = self._truncate_to_tokens(text, self.max_input_tokens)

        # Convert text to message format for MemOS extract()
        messages = [{"role": "user", "content": bounded_text}]

        start_time = time.time()
        all_extraction_results: List[Dict[str, Any]] = []
        all_stored_memories: List[Dict[str, Any]] = []
        extraction_time = 0.0
        add_time = 0.0

        try:
            # Use MemOS's official extract() method
            # This calls MemOS's LLM-based extractor internally
            extracted_items = memory_system.extract(messages)
            extraction_time = time.time() - start_time

            # Record extraction results
            for item in extracted_items:
                memory_text = getattr(item, "memory", str(item))
                metadata = getattr(item, "metadata", {})
                if hasattr(metadata, "model_dump"):
                    metadata = metadata.model_dump()

                all_extraction_results.append({
                    "memory": memory_text,
                    "metadata": metadata,
                })

            # Use MemOS's official add() method to store extracted memories
            if extracted_items:
                add_start = time.time()
                memory_system.add(extracted_items)
                add_time = time.time() - add_start

                for item in extracted_items:
                    memory_text = getattr(item, "memory", str(item))
                    metadata = getattr(item, "metadata", {})
                    if hasattr(metadata, "model_dump"):
                        metadata = metadata.model_dump()

                    all_stored_memories.append({
                        "memory": memory_text,
                        "metadata": metadata,
                    })

                # Update stored memory count
                self._stored_memory_count += len(extracted_items)

            total_time = time.time() - start_time

        except Exception as e:
            logger.error(f"MemOS extract/add failed: {e}", exc_info=True)
            return MemoryBuildResult(
                success=False,
                method="memos",
                action="extract_and_add",
                input_content=text,
                stored_content="",
                extraction_result=f"[Error: {e}]",
                all_passages=[],
                memory_entries=[],
                chunk_count=0,
                extra={"context_id": context_id, "error": str(e)},
            )

        # Track raw input chunks for compatibility
        self._memory_chunks.append(text)
        self._is_initialized = True

        # Create detailed memory entries for display (limited to first 10)
        memory_entries = [
            {
                "event": "ADD",
                "memory": mem.get("memory", "")[:200],
                "metadata": mem.get("metadata", {}),
            }
            for mem in all_stored_memories[:10]
        ]

        # Build extraction result summary
        extraction_summary = f"Extracted {len(extracted_items)} memories using MemOS extract()\n"
        for i, item in enumerate(all_extraction_results[:5]):
            extraction_summary += f"  [{i+1}] {item['memory'][:100]}...\n"

        return MemoryBuildResult(
            success=True,
            method="memos",
            action="extract_and_add",
            input_content=text,
            stored_content=bounded_text,
            extraction_result=extraction_summary,
            all_passages=all_stored_memories,
            memory_entries=memory_entries,
            chunk_count=self._stored_memory_count,
            time_cost=total_time,
            extra={
                "context_id": context_id,
                "retrieve_num": self.retrieve_num,
                "extracted_count": len(extracted_items),
                "text_mem_type": self.text_mem_type,
                "extraction_time": extraction_time,
                "add_time": add_time,
            },
        )

    def query(
        self,
        question: str,
        system_message: Optional[str] = None,
        **kwargs,
    ) -> AgentResponse:
        """Query using MemOS's official search() method.

        Uses MemOS's native vector search for retrieval, then generates response.

        Args:
            question: User question to answer.
            system_message: Optional system prompt.
            **kwargs: Additional arguments (unused).

        Returns:
            AgentResponse with answer and retrieval details.
        """
        prepared = self.prepare_batch_query(question, system_message=system_message, **kwargs)
        response = self._llm_client.chat(prepared["messages"])
        result = self.finalize_batch_query(prepared, response.content)
        # The original timer starts after question truncation, immediately
        # before MemOS search. Preserve that reporting boundary in real-time.
        result.query_time = time.time() - prepared["query_start"]
        result.extra["tokens_used"] = {
            "input": response.input_tokens,
            "output": response.output_tokens,
        }
        return result

    def prepare_batch_query(
        self,
        question: str,
        system_message: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Perform read-only MemOS retrieval before a final batchable answer."""
        context_id = self._get_context_id()
        memory_system = self._get_memory_system(context_id)

        # Truncate question
        bounded_question = self._truncate_to_tokens(question, self.max_question_tokens)

        start_time = time.time()

        # Use MemOS's official search() method
        # Request exactly retrieve_num results (no over-fetching)
        memory_items = memory_system.search(bounded_question, top_k=self.retrieve_num)

        search_time = time.time() - start_time

        # Calculate token budget for memory context
        system_tokens = self._llm_client.count_tokens(system_message) if system_message else 0
        reserved_tokens = self.max_tokens + 500
        available_tokens = max(self.max_context_tokens - reserved_tokens - system_tokens, 0)
        question_tokens = self._llm_client.count_tokens(bounded_question)
        memory_budget = max(available_tokens - question_tokens, 0)
        memory_budget = min(memory_budget, self.max_memory_tokens)

        # Build memory context with truncation
        retrieved_memories: List[Dict[str, Any]] = []
        memory_blocks: List[str] = []
        used_tokens = 0

        for item in memory_items:
            mem_text = str(getattr(item, "memory", "") or "").strip()
            if not mem_text:
                continue

            mem_tokens = self._llm_client.count_tokens(mem_text)
            if used_tokens + mem_tokens > memory_budget:
                # Truncate this memory to fit remaining budget
                remaining = memory_budget - used_tokens
                if remaining > 100:
                    truncated = self._truncate_to_tokens(mem_text, remaining)
                    memory_blocks.append(truncated)
                    retrieved_memories.append({
                        "memory": truncated[:2000],
                        "type": "memos_search",
                        "truncated": True,
                    })
                break

            metadata = getattr(item, "metadata", None)
            memory_blocks.append(mem_text)
            used_tokens += mem_tokens
            retrieved_memories.append({
                "memory": mem_text[:2000],
                "type": "memos_search",
                "metadata": metadata.model_dump() if hasattr(metadata, "model_dump") else {},
            })

        # Build final prompt with retrieved memories
        full_question = bounded_question
        if memory_blocks:
            memory_context = "\n\n".join(
                [f"[Memory {idx + 1}]\n{block}" for idx, block in enumerate(memory_blocks)]
            )
            full_question = f"[Retrieved MemOS Memories]\n{memory_context}\n\n[Question]\n{bounded_question}"

        return {
            "messages": format_messages(full_question, system_message),
            "retrieved_count": len(retrieved_memories),
            "retrieved_memories": retrieved_memories,
            "query_start": start_time,
            "extra": {
                "method": "memos",
                "text_mem_type": self.text_mem_type,
                "search_time": search_time,
            },
        }

    @staticmethod
    def finalize_batch_query(prepared: Dict[str, Any], content: str) -> AgentResponse:
        """Return MemOS's normal response fields after final generation."""
        return AgentResponse(
            output=content,
            query_time=0.0,
            retrieved_count=prepared["retrieved_count"],
            retrieved_memories=prepared["retrieved_memories"],
            extra=prepared["extra"],
        )

    @staticmethod
    def record_batch_query_usage(
        response: AgentResponse,
        input_tokens: int,
        output_tokens: int,
    ) -> None:
        """Keep batch result diagnostics identical to MemOS real-time queries."""
        response.extra["tokens_used"] = {
            "input": input_tokens,
            "output": output_tokens,
        }

    def reset(self) -> None:
        """Reset agent state and clear all memory systems."""
        super().reset()
        self._memory_systems = {}
        self._stored_memory_count = 0

    def set_context_id(self, context_id: int) -> None:
        """Set context ID for distinguishing personas.

        Args:
            context_id: Context identifier.
        """
        super().set_context_id(context_id)

    @property
    def memory_size(self) -> int:
        """Get count of actual stored memory items."""
        return self._stored_memory_count
