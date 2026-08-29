"""A-Mem agent adapter for MedMemoryBench."""

from __future__ import annotations

import importlib
import logging
import os
import re
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np

from .base import BaseAgent, MemoryBuildResult, AgentResponse
from utils.llm_client import (
    create_llm_client,
    format_messages,
    BaseLLMClient,
    get_usage_tracker,
)

logger = logging.getLogger(__name__)

# Reserved tokens for output generation
RESERVED_OUTPUT_TOKENS = 1000


class AMemAgent(BaseAgent):
    """Adapter that bridges A-Mem memory system to BaseAgent interface.

    This implementation follows the official A-Mem logic:
    1. Content is split into chunks if exceeding token limit
    2. Each chunk is directly passed to add_note() for analysis and evolution
    3. No additional preprocessing (like summarization) is applied
    """

    METHOD_TYPE = "agentic_memory"
    BIGMODEL_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"

    # Default chunk size for memorization (in tokens)
    DEFAULT_CHUNK_SIZE_TOKENS = 10240
    MEMORY_STATE_VERSION = 1

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 1.0,
        max_tokens: int = 2000,
        provider: str = "openai",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        retrieve_num: int = 5,
        amem_backend: str = "openai",
        amem_model: Optional[str] = None,
        amem_api_key: Optional[str] = None,
        amem_base_url: Optional[str] = None,
        amem_llm_client_kwargs: Optional[Dict[str, Any]] = None,
        amem_embedding_model: str = "all-MiniLM-L6-v2",
        amem_evo_threshold: int = 100,
        amem_max_tokens: Optional[int] = None,
        amem_temperature: float = 0.7,
        amem_retry_temperature: float = 0.3,
        amem_connectivity_temperature: float = 0.0,
        amem_build_max_context_tokens: int = 200000,
        amem_max_context_tokens: int = 200000,
        amem_chunk_size_tokens: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(model, temperature, max_tokens, **kwargs)

        self.retrieve_num = retrieve_num
        self.amem_backend = amem_backend
        self.amem_model = amem_model or model
        self.amem_embedding_model = amem_embedding_model
        self.amem_evo_threshold = amem_evo_threshold
        self.amem_max_tokens = amem_max_tokens or max_tokens
        self.amem_temperature = amem_temperature
        self.amem_retry_temperature = amem_retry_temperature
        self.amem_connectivity_temperature = amem_connectivity_temperature
        self.amem_build_max_context_tokens = max(
            1, int(amem_build_max_context_tokens)
        )
        self.amem_max_context_tokens = amem_max_context_tokens
        self.amem_chunk_size_tokens = amem_chunk_size_tokens or self.DEFAULT_CHUNK_SIZE_TOKENS
        self._amem_backend = self.amem_backend
        self._amem_model = self.amem_model
        self._llm_client_kwargs = dict(kwargs.get("llm_client_kwargs", {}))
        self._amem_llm_client_kwargs = dict(
            self._llm_client_kwargs
            if amem_llm_client_kwargs is None
            else amem_llm_client_kwargs
        )

        # API configuration for A-Mem internal LLM calls
        self._amem_api_key = (
            amem_api_key
            or api_key
            or os.environ.get("BIGMODEL_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
        )
        self._amem_api_base = (
            amem_base_url
            or base_url
            or os.environ.get("BIGMODEL_BASE_URL")
            or os.environ.get("OPENAI_BASE_URL")
            or self.BIGMODEL_BASE_URL
        )

        # LLM client for query-time response generation
        self._llm_client: BaseLLMClient = create_llm_client(
            provider=provider,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
            base_url=base_url,
            **kwargs.get("llm_client_kwargs", {}),
        )

        # Per-context A-Mem memory systems
        self._amem_systems: Dict[int, Any] = {}
        self._amem_class = self._load_amem_system_class()

    def _load_amem_system_class(self):
        """Load RobustAgenticMemorySystem from methods/amem/A-mem."""
        amem_dir = Path(__file__).resolve().parent / "amem" / "A-mem"
        if not amem_dir.exists():
            raise ImportError(f"A-mem source folder not found at {amem_dir}")

        amem_dir_str = str(amem_dir)
        if amem_dir_str not in sys.path:
            sys.path.insert(0, amem_dir_str)

        module = importlib.import_module("memory_layer_robust")
        return getattr(module, "RobustAgenticMemorySystem")

    def _get_context_id(self) -> int:
        """Get current context ID, defaulting to 0."""
        return self._context_id if self._context_id is not None else 0

    def _get_memory_system(self, context_id: int):
        """Get or create A-Mem system for the given context."""
        system = self._amem_systems.get(context_id)
        if system is not None:
            return system

        # Convert token limit to char limit
        # For Chinese text: ~1.5 chars per token (conservative)
        # For English text: ~4 chars per token
        # Use 1.5 as conservative estimate for mixed content
        max_context_chars = int(
            getattr(self, "amem_build_max_context_tokens", 200000) * 1.5
        )

        system = self._amem_class(
            model_name=self.amem_embedding_model,
            llm_backend=self._amem_backend,
            llm_model=self._amem_model,
            evo_threshold=self.amem_evo_threshold,
            api_key=self._amem_api_key,
            api_base=self._amem_api_base,
            max_tokens=self.amem_max_tokens,
            temperature=getattr(self, "amem_temperature", 0.7),
            retry_temperature=getattr(self, "amem_retry_temperature", 0.3),
            connectivity_temperature=getattr(
                self, "amem_connectivity_temperature", 0.0
            ),
            max_context_chars=max_context_chars,
            check_connection=False,
            usage_tracker=get_usage_tracker(),
            provider_routing=getattr(self, "_amem_llm_client_kwargs", {}).get(
                "provider_routing"
            ),
            service_tier=getattr(self, "_amem_llm_client_kwargs", {}).get(
                "service_tier"
            ),
            reasoning_effort=getattr(self, "_amem_llm_client_kwargs", {}).get(
                "reasoning_effort"
            ),
        )
        self._amem_systems[context_id] = system
        logger.info(
            "Created A-Mem system for context %d: model=%s, evo_threshold=%d, max_context_chars=%d",
            context_id, self.amem_model, self.amem_evo_threshold, max_context_chars
        )
        return system

    @staticmethod
    def _json_safe(value: Any) -> Any:
        """Convert A-MEM state values to lossless JSON-compatible values."""
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {str(key): AMemAgent._json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [AMemAgent._json_safe(item) for item in value]
        return value

    def _memory_state_config(self) -> Dict[str, Any]:
        return {
            "config_version": 2,
            "agent_class": self.__class__.__name__,
            "backend": getattr(self, "_amem_backend", ""),
            "model": getattr(self, "_amem_model", ""),
            "embedding_model": getattr(self, "amem_embedding_model", ""),
            "evo_threshold": getattr(self, "amem_evo_threshold", 0),
            "max_tokens": getattr(self, "amem_max_tokens", 0),
            "temperature": getattr(self, "amem_temperature", 0.7),
            "retry_temperature": getattr(self, "amem_retry_temperature", 0.3),
            "connectivity_temperature": getattr(
                self, "amem_connectivity_temperature", 0.0
            ),
            "build_max_context_tokens": getattr(
                self, "amem_build_max_context_tokens", 200000
            ),
            "chunk_size_tokens": getattr(self, "amem_chunk_size_tokens", 0),
        }

    def supports_memory_snapshots(self) -> bool:
        """A-MEM snapshots contain all state needed for later retrieval."""
        return True

    def export_memory_state(self, context_id: Optional[int] = None) -> Dict[str, Any]:
        """Export one context without serializing model or provider clients."""
        resolved_context_id = self._get_context_id() if context_id is None else int(context_id)
        memory_system = self._get_memory_system(resolved_context_id)
        embeddings = memory_system.retriever.embeddings
        embedding_state = None
        if embeddings is not None:
            embedding_array = np.asarray(embeddings)
            embedding_state = {
                "dtype": str(embedding_array.dtype),
                "shape": list(embedding_array.shape),
                "values": embedding_array.copy(),
            }

        system_state: Dict[str, Any] = {
            "system_class": memory_system.__class__.__name__,
            "evo_cnt": memory_system.evo_cnt,
            "evo_threshold": memory_system.evo_threshold,
            "max_context_chars": memory_system.max_context_chars,
            # Dict insertion order is the retriever's index-to-note mapping.
            "memories": [
                {
                    "memory_id": str(memory_id),
                    "attributes": self._json_safe(vars(note)),
                }
                for memory_id, note in memory_system.memories.items()
            ],
            "retriever": {
                "corpus": self._json_safe(memory_system.retriever.corpus),
                "document_ids": self._json_safe(memory_system.retriever.document_ids),
                "embeddings": embedding_state,
            },
        }

        typed_fields = (
            "original_evolution_enabled",
            "typed_relations_enabled",
            "relation_candidate_count",
            "typed_relations",
            "relation_audit",
            "_relation_audit_by_memory",
        )
        for field_name in typed_fields:
            if hasattr(memory_system, field_name):
                system_state[field_name] = self._json_safe(
                    getattr(memory_system, field_name)
                )
        if hasattr(memory_system, "_typed_edge_keys"):
            system_state["_typed_edge_keys"] = [
                list(edge_key)
                for edge_key in sorted(memory_system._typed_edge_keys)
            ]

        return {
            "format": "medmemorybench.amem.memory_state",
            "version": self.MEMORY_STATE_VERSION,
            "context_id": resolved_context_id,
            "config": self._memory_state_config(),
            "agent_state": {
                "memory_chunks": list(self._memory_chunks),
                "is_initialized": self._is_initialized,
            },
            "system_state": system_state,
        }

    def import_memory_state(
        self,
        state: Dict[str, Any],
        context_id: Optional[int] = None,
    ) -> None:
        """Restore a snapshot without recomputing any stored embedding."""
        if state.get("format") != "medmemorybench.amem.memory_state":
            raise ValueError("Unsupported A-MEM memory snapshot format")
        if state.get("version") != self.MEMORY_STATE_VERSION:
            raise ValueError(
                f"Unsupported A-MEM memory snapshot version: {state.get('version')}"
            )
        stored_config = state.get("config", {})
        current_config = self._memory_state_config()
        # Older snapshots included the query context budget in this state
        # record; it is deliberately ignored now because it is retrieval-only.
        if stored_config.get("config_version") == 2:
            config_matches = stored_config == current_config
        else:
            legacy_current_config = {
                "agent_class": current_config["agent_class"],
                "embedding_model": current_config["embedding_model"],
                "evo_threshold": current_config["evo_threshold"],
                "chunk_size_tokens": current_config["chunk_size_tokens"],
            }
            legacy_config = dict(stored_config) if isinstance(stored_config, dict) else {}
            legacy_config.pop("max_context_tokens", None)
            config_matches = legacy_config == legacy_current_config
        if not config_matches:
            raise ValueError("A-MEM memory snapshot configuration does not match the agent")

        state_context_id = int(state["context_id"])
        resolved_context_id = state_context_id if context_id is None else int(context_id)
        if resolved_context_id != state_context_id:
            raise ValueError(
                f"A-MEM snapshot context {state_context_id} cannot load as {resolved_context_id}"
            )

        memory_system = self._get_memory_system(resolved_context_id)
        system_state = state["system_state"]
        if system_state.get("system_class") != memory_system.__class__.__name__:
            raise ValueError("A-MEM memory-system class does not match the snapshot")

        robust_module = importlib.import_module("memory_layer_robust")
        note_class = getattr(robust_module, "RobustMemoryNote")
        restored_memories: Dict[str, Any] = {}
        for item in system_state.get("memories", []):
            memory_id = str(item["memory_id"])
            attributes = dict(item.get("attributes", {}))
            note = note_class.__new__(note_class)
            note.__dict__.update(attributes)
            note.id = memory_id
            restored_memories[memory_id] = note
        memory_system.memories = restored_memories
        memory_system.evo_cnt = int(system_state.get("evo_cnt", 0))
        memory_system.evo_threshold = int(system_state["evo_threshold"])
        memory_system.max_context_chars = int(system_state["max_context_chars"])

        retriever_state = system_state["retriever"]
        memory_system.retriever.corpus = list(retriever_state.get("corpus", []))
        memory_system.retriever.document_ids = {
            str(document): int(index)
            for document, index in retriever_state.get("document_ids", {}).items()
        }
        embedding_state = retriever_state.get("embeddings")
        if embedding_state is None:
            memory_system.retriever.embeddings = None
        else:
            memory_system.retriever.embeddings = np.asarray(
                embedding_state["values"],
                dtype=np.dtype(embedding_state["dtype"]),
            ).reshape(embedding_state["shape"])

        for field_name in (
            "original_evolution_enabled",
            "typed_relations_enabled",
            "relation_candidate_count",
            "typed_relations",
            "relation_audit",
            "_relation_audit_by_memory",
        ):
            if field_name in system_state:
                setattr(memory_system, field_name, system_state[field_name])
        if "_typed_edge_keys" in system_state:
            memory_system._typed_edge_keys = {
                tuple(edge_key) for edge_key in system_state["_typed_edge_keys"]
            }

        agent_state = state.get("agent_state", {})
        self._memory_chunks = list(agent_state.get("memory_chunks", []))
        self._is_initialized = bool(agent_state.get("is_initialized", False))
        self._amem_systems[resolved_context_id] = memory_system
        self.set_context_id(resolved_context_id)

    def _split_text_into_chunks(self, text: str, max_tokens: int) -> List[str]:
        """Split text into chunks based on token count.

        Tries to split at natural boundaries (paragraphs, sentences) when possible.
        Uses _llm_client.count_tokens for consistent token counting.
        """
        total_tokens = self._llm_client.count_tokens(text)

        if total_tokens <= max_tokens:
            return [text]

        chunks = []
        paragraphs = text.split('\n\n')

        current_chunk = ""
        current_tokens = 0

        for para in paragraphs:
            para_tokens = self._llm_client.count_tokens(para)

            if para_tokens > max_tokens:
                # Save current chunk if not empty
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                    current_tokens = 0

                # Split large paragraph by sentences
                sub_chunks = self._split_large_paragraph(para, max_tokens)
                chunks.extend(sub_chunks)
            elif current_tokens + para_tokens > max_tokens:
                # Current chunk is full, start a new one
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = para
                current_tokens = para_tokens
            else:
                # Add to current chunk
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para
                current_tokens += para_tokens

        # Don't forget the last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks

    def _split_large_paragraph(self, text: str, max_tokens: int) -> List[str]:
        """Split a large paragraph that exceeds max_tokens."""
        chunks = []

        # Try to split by sentences (Chinese and English punctuation)
        sentences = re.split(r'([。！？.!?]+)', text)

        # Recombine sentences with their punctuation
        combined_sentences = []
        for i in range(0, len(sentences) - 1, 2):
            if i + 1 < len(sentences):
                combined_sentences.append(sentences[i] + sentences[i + 1])
            else:
                combined_sentences.append(sentences[i])
        if len(sentences) % 2 == 1 and sentences[-1].strip():
            combined_sentences.append(sentences[-1])

        current_chunk = ""
        current_tokens = 0

        for sent in combined_sentences:
            sent_tokens = self._llm_client.count_tokens(sent)

            if sent_tokens > max_tokens:
                # Sentence itself is too long, split by character count
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                    current_tokens = 0

                # Estimate chars per token (conservative for Chinese)
                chars_per_chunk = int(max_tokens * 1.5)
                for i in range(0, len(sent), chars_per_chunk):
                    chunks.append(sent[i:i + chars_per_chunk])
            elif current_tokens + sent_tokens > max_tokens:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = sent
                current_tokens = sent_tokens
            else:
                current_chunk += sent
                current_tokens += sent_tokens

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks

    def _truncate_to_token_limit(self, text: str, max_tokens: int) -> str:
        """Truncate text to fit within token limit using _llm_client tokenizer."""
        if not text or max_tokens <= 0:
            return ""

        current_tokens = self._llm_client.count_tokens(text)
        if current_tokens <= max_tokens:
            return text

        # Use base class tokenizer for actual truncation (tiktoken-based)
        tokens = self._tokenizer.encode(text)
        truncated = self._tokenizer.decode(tokens[:max_tokens])
        return truncated + "\n... [truncated]"

    def memorize(self, text: str, **kwargs) -> MemoryBuildResult:
        """Store text into A-Mem memory system.

        This method follows the official A-Mem implementation:
        1. Split text into chunks if needed
        2. Pass each chunk directly to add_note() for analysis and evolution
        3. No additional preprocessing is applied
        """
        context_id = self._get_context_id()
        memory_system = self._get_memory_system(context_id)

        # Split text into chunks to avoid context length exceeded
        with get_usage_tracker().scope("amem.base.chunking"):
            chunks = self._split_text_into_chunks(text, self.amem_chunk_size_tokens)
        logger.debug("Split input into %d chunks (chunk_size=%d tokens)", len(chunks), self.amem_chunk_size_tokens)

        note_ids = []
        memory_entries = []

        for i, chunk in enumerate(chunks):
            # Directly pass chunk to A-Mem add_note (following official implementation)
            # A-Mem internally performs: content analysis, metadata extraction, evolution
            note_id = memory_system.add_note(content=chunk)
            note_ids.append(str(note_id))
            self._memory_chunks.append(chunk)

            memory_entries.append({
                "event": "ADD",
                "memory": chunk[:400],
                "id": str(note_id),
                "chunk_index": i,
            })
            logger.debug("Added chunk %d/%d to A-Mem, note_id=%s", i + 1, len(chunks), note_id)

        self._is_initialized = True

        return MemoryBuildResult(
            success=True,
            method="amem",
            action="add_note",
            input_content=text,
            stored_content=text,
            memory_entries=memory_entries,
            all_passages=list(memory_entries),
            chunk_count=len(self._memory_chunks),
            extra={
                "context_id": context_id,
                "retrieve_num": self.retrieve_num,
                "chunks_created": len(chunks),
                "note_ids": note_ids,
                "inserted_count": len(note_ids),
            },
        )

    def query(
        self,
        question: str,
        system_message: Optional[str] = None,
        **kwargs,
    ) -> AgentResponse:
        """Query the A-Mem memory system and generate response.

        Steps:
        1. Retrieve related memories using A-Mem's find_related_memories
        2. Construct context with retrieved memories
        3. Generate response using LLM
        """
        prepared = self.prepare_batch_query(question, system_message=system_message, **kwargs)
        response = self._llm_client.chat(prepared["messages"])
        return self.finalize_batch_query(prepared, response.content)

    def prepare_batch_query(
        self,
        question: str,
        system_message: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Retrieve locally and defer only A-Mem's immutable final answer."""
        context_id = self._get_context_id()
        memory_system = self._get_memory_system(context_id)

        # This retrieval is read-only. Note analysis and evolution remain in
        # ``memorize`` because each note can update its predecessors.
        memory_str, indices = memory_system.find_related_memories(question, k=self.retrieve_num)

        # Calculate available tokens for memory context
        system_tokens = self._llm_client.count_tokens(system_message) if system_message else 0
        question_tokens = self._llm_client.count_tokens(question)
        reserved_tokens = self.amem_max_tokens + RESERVED_OUTPUT_TOKENS
        max_memory_tokens = max(
            self.amem_max_context_tokens - system_tokens - question_tokens - reserved_tokens,
            0,
        )

        # Truncate memory if needed and construct full question
        if memory_str.strip():
            memory_str = self._truncate_to_token_limit(memory_str, max_memory_tokens)
            memory_context = f"[Retrieved A-Mem Notes]\n{memory_str}\n\n"
            full_question = memory_context + question
        else:
            full_question = question

        # Build retrieved memories list for logging
        indices_list = indices.tolist() if hasattr(indices, "tolist") else list(indices)
        retrieved_memories: List[Dict[str, Any]] = []
        if memory_str.strip():
            retrieved_memories.append({
                "memory": memory_str[:2000],
                "type": "amem_retrieval",
                "indices": indices_list,
            })

        return {
            "messages": format_messages(full_question, system_message),
            "retrieved_count": len(indices_list),
            "retrieved_memories": retrieved_memories,
            "extra": {
                "method": "amem",
                "context_id": context_id,
            },
        }

    @staticmethod
    def finalize_batch_query(prepared: Dict[str, Any], content: str) -> AgentResponse:
        """Return the normal A-Mem response after real-time or batch generation."""
        return AgentResponse(
            output=content,
            retrieved_count=prepared["retrieved_count"],
            retrieved_memories=prepared["retrieved_memories"],
            extra=prepared["extra"],
        )

    def reset(self) -> None:
        """Reset agent state including all A-Mem systems."""
        super().reset()
        self._amem_systems = {}
        logger.debug("Reset A-Mem agent state")

    def set_context_id(self, context_id: int) -> None:
        """Set context ID for distinguishing personas."""
        super().set_context_id(context_id)

    def get_info(self) -> Dict[str, Any]:
        """Get agent info including A-Mem specific parameters."""
        info = super().get_info()
        info.update({
            "retrieve_num": self.retrieve_num,
            "amem_backend": self._amem_backend,
            "amem_model": self._amem_model,
            "amem_embedding_model": self.amem_embedding_model,
            "amem_evo_threshold": self.amem_evo_threshold,
            "amem_max_tokens": self.amem_max_tokens,
            "amem_build_max_context_tokens": self.amem_build_max_context_tokens,
            "amem_max_context_tokens": self.amem_max_context_tokens,
            "amem_chunk_size_tokens": self.amem_chunk_size_tokens,
            "active_contexts": list(self._amem_systems.keys()),
        })
        return info
