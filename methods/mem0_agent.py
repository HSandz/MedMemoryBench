"""Mem0 Agent - automatic memory extraction and retrieval using Mem0 library."""

import os
import re
import time
from typing import Optional, List, Tuple, Dict, Any

# Disable Mem0 telemetry (PostHog) to avoid SSL errors
os.environ["MEM0_TELEMETRY"] = "false"

from .base import BaseAgent, MemoryBuildResult, AgentResponse
from utils.llm_client import create_llm_client, format_messages, BaseLLMClient, get_usage_tracker


class Mem0Agent(BaseAgent):
    """Mem0 Agent for intelligent memory management with automatic extraction and semantic retrieval."""

    METHOD_TYPE = "agentic_memory"
    BIGMODEL_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 1.0,
        max_tokens: int = 2000,
        provider: str = "openai",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        embedding_model: str = "text-embedding-3-small",
        embedding_provider: str = "openai",
        embedding_model_path: Optional[str] = None,
        retrieve_num: int = 5,
        **kwargs
    ):
        super().__init__(model, temperature, max_tokens, **kwargs)

        self.embedding_model = embedding_model
        self.embedding_provider = embedding_provider
        self.embedding_model_path = embedding_model_path
        self.retrieve_num = retrieve_num
        # Support both 'chunk_size_tokens' (config file) and 'max_input_tokens' (legacy)
        self.max_input_tokens = int(
            kwargs.get("chunk_size_tokens")
            or kwargs.get("max_input_tokens")
            or 8000
        )
        self.max_context_tokens = int(kwargs.get("max_context_tokens", 120000))
        self.max_question_tokens = int(kwargs.get("max_question_tokens", 4096))
        # Turn-by-turn processing (matches original Mem0 paper design)
        self.turn_by_turn: bool = bool(kwargs.get("turn_by_turn", False))
        self.max_recent_turns: int = int(kwargs.get("max_recent_turns", 10))

        # API config
        api_key_str = api_key or os.environ.get("BIGMODEL_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
        # If multiple keys are provided via comma-separated string, use the first one for Mem0's internal client initially
        self._all_api_keys = [k.strip() for k in api_key_str.split(",") if k.strip()] if api_key_str else []
        self._api_key = self._all_api_keys[0] if self._all_api_keys else ""
        
        self._base_url = (
            base_url
            or os.environ.get("BIGMODEL_BASE_URL")
            or os.environ.get("OPENAI_BASE_URL")
            or self.BIGMODEL_BASE_URL
        )
        self._provider = provider

        # LLM client for Q&A
        self._llm_client: BaseLLMClient = create_llm_client(
            provider=provider,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
            base_url=base_url,
        )

        # Mem0 Memory instance
        self._memory = None
        self._user_id = None
        self._agent_start_time = time.time()
        self._qdrant_path = None  # Will be set in _init_mem0

        # Cache for embedding dimensions
        self._embedding_dims_cache: Dict[str, int] = {}

        # Note: _init_mem0 is also called in _init_memory() after context_id is set
        # so that the Qdrant path uses the correct persona ID.
        # The function is idempotent - it will reinitialize when context_id changes.
        self._init_mem0()

    def _init_memory(self) -> None:
        super()._init_memory()
        # Re-initialize Mem0 now that self._context_id has been set
        self._init_mem0()

    def _get_embedding_dims(self, model_name_or_path: str) -> int:
        """Get embedding dimensions for a model without loading it repeatedly.

        Uses a cache and known model dimensions to avoid expensive model loading.
        """
        # Check cache first
        if model_name_or_path in self._embedding_dims_cache:
            return self._embedding_dims_cache[model_name_or_path]

        # Known model dimensions (common models)
        known_dims = {
            # BGE models
            "bge-small": 512,
            "bge-base": 768,
            "bge-large": 1024,
            "bge-m3": 1024,
            # Other common models
            "all-MiniLM-L6": 384,
            "all-mpnet-base": 768,
            "paraphrase-multilingual": 768,
            "multi-qa-MiniLM": 384,
        }

        # Try to match known models
        model_lower = model_name_or_path.lower()
        for pattern, dims in known_dims.items():
            if pattern.lower() in model_lower:
                self._embedding_dims_cache[model_name_or_path] = dims
                return dims

        # If not a known model, we need to load it once
        # But this should only happen once per unique model
        try:
            from sentence_transformers import SentenceTransformer
            temp_model = SentenceTransformer(model_name_or_path)
            dims = temp_model.get_sentence_embedding_dimension()
            del temp_model
            self._embedding_dims_cache[model_name_or_path] = dims
            return dims
        except Exception as e:
            print(f"[Mem0] Warning: Failed to get embedding dims for {model_name_or_path}: {e}")
            # Default fallback
            return 512

    def _truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        if not text or max_tokens <= 0:
            return ""
        tokens = self._tokenizer.encode(text)
        if len(tokens) <= max_tokens:
            return text
        return self._tokenizer.decode(tokens[:max_tokens])

    def _split_text_into_chunks(self, text: str, max_tokens: int) -> List[str]:
        if not text.strip():
            return []
        tokens = self._tokenizer.encode(text)
        if len(tokens) <= max_tokens:
            return [text]

        chunks: List[str] = []
        start = 0
        while start < len(tokens):
            end = min(start + max_tokens, len(tokens))
            chunks.append(self._tokenizer.decode(tokens[start:end]))
            if end >= len(tokens):
                break
            start = end
        return chunks

    def _init_mem0(self) -> None:
        """Initialize Mem0 Memory instance. Idempotent: reinitializes if context_id changed."""
        # Skip if already initialized with the same context_id
        if self._memory is not None and getattr(self, '_mem0_context_id', None) == self._context_id:
            return
        self._mem0_context_id = self._context_id
        print(f"[Mem0Agent DEBUG] _init_mem0 starting... (context_id={self._context_id})")
        from methods.mem0.memory.main import Memory
        import tempfile
        import os

        # --- MONKEY PATCH FOR API KEY ROTATION ---
        try:
            from mem0.embeddings.gemini import GoogleGenAIEmbedding
            import google.genai.errors
            from google.genai import Client

            if not hasattr(GoogleGenAIEmbedding, "_original_embed_batch"):
                GoogleGenAIEmbedding._original_embed_batch = GoogleGenAIEmbedding.embed_batch
                GoogleGenAIEmbedding._all_api_keys_for_rotation = self._all_api_keys
                GoogleGenAIEmbedding._current_key_idx = 0
                
                def patched_embed_batch(self_emb, texts, memory_action="add"):
                    keys = getattr(GoogleGenAIEmbedding, "_all_api_keys_for_rotation", [])
                    max_attempts = max(1, len(keys) * 10)
                    
                    for attempt in range(max_attempts):
                        try:
                            return self_emb._original_embed_batch(texts, memory_action=memory_action)
                        except google.genai.errors.APIError as e:
                            if "429" in str(e) or "quota" in str(e).lower() or getattr(e, "code", None) == 429:
                                GoogleGenAIEmbedding._current_key_idx = (GoogleGenAIEmbedding._current_key_idx + 1) % len(keys)
                                next_key = keys[GoogleGenAIEmbedding._current_key_idx]
                                self_emb.client = Client(api_key=next_key)
                                
                                if (attempt + 1) % len(keys) == 0:
                                    print(f"\n[Mem0Agent] ALL API Keys limit! Wait 60s...")
                                    time.sleep(60.0)
                                else:
                                    print(f"\n[Mem0Agent] Embedding API limit! Rotating...")
                                    time.sleep(1.0)
                            else:
                                raise
                    raise RuntimeError("All keys exhausted.")
                    
                GoogleGenAIEmbedding.embed_batch = patched_embed_batch

            # Patch OpenAILLM
            from methods.mem0.llms.openai import OpenAILLM
            if not hasattr(OpenAILLM, "_original_generate_response"):
                OpenAILLM._original_generate_response = OpenAILLM.generate_response
                OpenAILLM._all_api_keys_for_rotation = self._all_api_keys
                OpenAILLM._current_key_idx = 0
                
                def patched_generate_response(self_llm, messages, response_format=None, tools=None, tool_choice="auto"):
                    keys = getattr(OpenAILLM, "_all_api_keys_for_rotation", [])
                    max_attempts = max(1, len(keys) * 10)
                    
                    for attempt in range(max_attempts):
                        try:
                            return self_llm._original_generate_response(
                                messages, response_format=response_format, tools=tools, tool_choice=tool_choice
                            )
                        except Exception as e:
                            error_msg = str(e).lower()
                            is_rate_limit = "rate limit" in error_msg or "429" in error_msg
                            is_connection_error = any(keyword in error_msg for keyword in [
                                "connection", "ssl", "eof", "timeout", "reset",
                                "connect", "network", "socket", "refused", "closed"
                            ])
                            if is_rate_limit or is_connection_error:
                                OpenAILLM._current_key_idx = (OpenAILLM._current_key_idx + 1) % len(keys)
                                next_key = keys[OpenAILLM._current_key_idx]
                                self_llm.client.api_key = next_key
                                
                                if (attempt + 1) % len(keys) == 0:
                                    print(f"\n[Mem0Agent] LLM API Keys exhausted! Wait 60s...")
                                    time.sleep(60.0)
                                else:
                                    print(f"\n[Mem0Agent] LLM API Error ({e})! Rotating key...")
                                    time.sleep(1.0)
                            else:
                                raise
                    raise RuntimeError("All keys exhausted.")
                
                OpenAILLM.generate_response = patched_generate_response

        except Exception as e:
            print(f"[Mem0Agent] Failed to patch rotation: {e}")
        # -----------------------------------------

        # Create unique Qdrant path for this agent instance to avoid lock conflicts
        # Use context_id if available, otherwise use a unique timestamp
        if self._context_id is not None:
            qdrant_path = f"./.tmp/qdrant_{self.__class__.__name__}_persona_{self._context_id}"
        else:
            import uuid
            qdrant_path = f"./.tmp/qdrant_{self.__class__.__name__}_pid{os.getpid()}_{uuid.uuid4().hex[:8]}"

        print(f"[Mem0Agent DEBUG] Qdrant path: {qdrant_path}")

        # Build Mem0 config
        use_vertex_gemini = self._provider.lower() in {"gemini", "vertex", "vertex_ai"}
        mem0_config = {
            "history_db_path": f"{qdrant_path}_history.db",
            "llm": {
                "provider": "gemini" if use_vertex_gemini else "openai",
                "config": {
                    "model": self.model,
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                    "api_key": self._api_key,
                    "openai_base_url": self._base_url,
                }
            }
        }

        if self.embedding_provider in ("local", "huggingface"):
            model_name_or_path = self.embedding_model_path or self.embedding_model

            embedding_dims = self._get_embedding_dims(model_name_or_path)
            print(f"[Mem0] Using local Embedding model: {model_name_or_path} (dims: {embedding_dims})")

            mem0_config["embedder"] = {
                "provider": "huggingface",
                "config": {
                    "model": model_name_or_path,
                    "embedding_dims": embedding_dims,
                }
            }
            # Configure vector_store dimension with unique path
            collection_name = f"mem0_local_{self._context_id}" if self._context_id is not None else "mem0_local"
            mem0_config["vector_store"] = {
                "provider": "qdrant",
                "config": {
                    "embedding_model_dims": embedding_dims,
                    "collection_name": collection_name,
                    "path": qdrant_path,
                }
            }
        else:
            embedding_dims = 768 if ("embedding-004" in self.embedding_model or "embedding-001" in self.embedding_model) else (2048 if "embedding-3" in self.embedding_model else 1536)
            
            embedder_config = {
                "model": self.embedding_model,
                "api_key": self._api_key,
                "embedding_dims": embedding_dims,
            }
            if self.embedding_provider == "openai":
                embedder_config["openai_base_url"] = self._base_url
                
            mem0_config["embedder"] = {
                "provider": self.embedding_provider if self.embedding_provider else "openai",
                "config": embedder_config
            }
            # Configure vector_store with unique path
            collection_name = f"mem0_{self.embedding_model.replace('-', '_')}_{self._context_id}" if self._context_id is not None else f"mem0_{self.embedding_model.replace('-', '_')}"
            mem0_config["vector_store"] = {
                "provider": "qdrant",
                "config": {
                    "embedding_model_dims": embedding_dims,
                    "collection_name": collection_name,
                    "path": qdrant_path,
                },
            }

        # Store Qdrant path for cleanup
        self._qdrant_path = qdrant_path

        print(f"[Mem0Agent DEBUG] Creating Memory.from_config...")
        self._memory = Memory.from_config(mem0_config)
        print(f"[Mem0Agent DEBUG] Memory created successfully")

        # Inject usage tracking callback into Mem0's LLM and Embedding
        self._inject_usage_callback()

    def _inject_usage_callback(self) -> None:
        """Inject usage callback into Mem0's internal LLM and Embedding."""
        # LLM usage callback
        def llm_usage_callback(input_tokens: int, output_tokens: int, latency: float) -> None:
            from utils.llm_client import LLMResponse
            print(f"[Mem0Agent] LLM usage: input={input_tokens}, output={output_tokens}, latency={latency:.2f}s")
            response = LLMResponse(
                content="",
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency=latency,
                model=self.model,
            )
            get_usage_tracker().record(response)

        # Embedding usage callback (input_tokens only, no output)
        def embedding_usage_callback(input_tokens: int, latency: float) -> None:
            from utils.llm_client import LLMResponse
            print(f"[Mem0Agent] Embedding usage: tokens={input_tokens}, latency={latency:.2f}s")
            response = LLMResponse(
                content="",
                input_tokens=input_tokens,
                output_tokens=0,
                latency=latency,
                model=self.embedding_model,
            )
            get_usage_tracker().record(response)

        # Inject LLM callback
        if self._memory and hasattr(self._memory, "llm") and self._memory.llm:
            if hasattr(self._memory.llm, "set_usage_callback"):
                self._memory.llm.set_usage_callback(llm_usage_callback)
                print(f"[Mem0Agent] LLM usage callback injected: {type(self._memory.llm).__name__}")
            else:
                print(f"[Mem0Agent] Warning: LLM {type(self._memory.llm).__name__} does not support set_usage_callback")
        else:
            print(f"[Mem0Agent] Warning: Unable to inject LLM usage callback - memory.llm not available")

        # Inject Embedding callback (only for OpenAI embeddings, local embeddings don't have token costs)
        if self._memory and hasattr(self._memory, "embedding_model") and self._memory.embedding_model:
            if hasattr(self._memory.embedding_model, "set_usage_callback"):
                # Only inject if using OpenAI embedding (has API token cost)
                if self.embedding_provider not in ("local", "huggingface"):
                    self._memory.embedding_model.set_usage_callback(embedding_usage_callback)
                    print(f"[Mem0Agent] Embedding usage callback injected: {type(self._memory.embedding_model).__name__}")
                else:
                    print(f"[Mem0Agent] Skipping embedding callback for local model (no token cost)")
            else:
                print(f"[Mem0Agent] Warning: Embedding model does not support set_usage_callback")

    def _get_user_id(self) -> str:
        """Get user ID."""
        if self._user_id:
            return self._user_id
        return f"context_{self._context_id}" if self._context_id else "default_user"

    def _parse_turns_from_text(self, text: str) -> Tuple[List[Tuple[str, str]], str]:
        """Parse Patient/Doctor conversation turns from formatted memory text.

        The MedMemoryBench session text has format:
            [YYYY-MM-DD]

            Patient: <user turn content>

            Doctor: <assistant turn content>

            Patient: ...
            ...

        Returns:
            - List of (patient_content, doctor_content) pairs (complete turn pairs only)
            - Date string extracted from the header (empty string if not found)
        """
        # Extract date/header from text, e.g. "[2026-01-15]" or "[Health consultation record]"
        date_str = ""
        date_match = re.search(r'^\[([^\]]+)\]', text.strip())
        if date_match:
            date_str = date_match.group(1)

        # Split text into segments by role markers
        # Split at newlines followed immediately by "Patient:" or "Doctor:"
        segments = re.split(r'\n(?=Patient:|Doctor:)', text)

        turns: List[Tuple[str, str]] = []
        pending_user: Optional[str] = None

        for seg in segments:
            seg = seg.strip()
            if not seg:
                continue
            if seg.startswith("Patient:"):
                # New user turn — flush any orphan (shouldn't happen in well-formed data)
                pending_user = seg[len("Patient:"):].strip()
            elif seg.startswith("Doctor:"):
                asst_content = seg[len("Doctor:"):].strip()
                if pending_user is not None:
                    turns.append((pending_user, asst_content))
                    pending_user = None
                # else: doctor turn without preceding patient turn — skip

        return turns, date_str

    def memorize(self, text: str, **kwargs) -> MemoryBuildResult:
        """Add text to Mem0 memory.

        When ``turn_by_turn=True`` (set in YAML config), each Patient/Doctor
        exchange is fed individually to Mem0 — matching the original Mem0 paper
        design where memory is built incrementally per conversational turn.

        Falls back to the original chunked approach when turn_by_turn is False
        or when the text cannot be parsed into turn pairs.
        """
        user_id = self._get_user_id()
        memory_entries: List[Dict] = []

        if self.turn_by_turn:
            turns, date_str = self._parse_turns_from_text(text)

            if turns:
                for patient_content, doctor_content in turns:
                    messages = []

                    # Provide date as system context so the LLM knows when this
                    # exchange happened — mirrors the "recent context" Mem0 paper uses.
                    if date_str:
                        messages.append({
                            "role": "system",
                            "content": f"Date of medical consultation: {date_str}"
                        })

                    messages.append({"role": "user",      "content": patient_content})
                    messages.append({"role": "assistant", "content": doctor_content})

                    result = self._memory.add(messages, user_id=user_id)
                    if result and "results" in result:
                        for entry in result.get("results", []):
                            memory_entries.append({
                                "event": entry.get("event", "ADD"),
                                "memory": entry.get("memory", ""),
                                "id": entry.get("id", ""),
                            })

                self._memory_chunks.append(text)
                self._is_initialized = True

                return MemoryBuildResult(
                    success=True,
                    method="mem0",
                    action="add_to_memory",
                    input_content=text,
                    stored_content=text,
                    memory_entries=memory_entries,
                    chunk_count=len(self._memory_chunks),
                    extra={"input_chunks": len(turns), "mode": "turn_by_turn"},
                )

            # Could not parse turns — warn and fall through to chunked mode
            import logging
            logging.warning(
                "[Mem0Agent] turn_by_turn=True but no Patient/Doctor turns found. "
                "Falling back to chunked mode."
            )

        # ── Chunked (original) mode ────────────────────────────────────────────
        chunks = self._split_text_into_chunks(text, self.max_input_tokens)

        for chunk in chunks:
            memory_messages = [
                {"role": "user",      "content": chunk},
                {"role": "assistant", "content": "I'll make sure to remember this information."},
            ]
            result = self._memory.add(memory_messages, user_id=user_id)
            if result and "results" in result:
                for entry in result.get("results", []):
                    memory_entries.append({
                        "event": entry.get("event", "ADD"),
                        "memory": entry.get("memory", ""),
                        "id": entry.get("id", ""),
                    })

        self._memory_chunks.append(text)
        self._is_initialized = True

        return MemoryBuildResult(
            success=True,
            method="mem0",
            action="add_to_memory",
            input_content=text,
            stored_content=text,
            memory_entries=memory_entries,
            chunk_count=len(self._memory_chunks),
            extra={"input_chunks": len(chunks), "mode": "chunked"},
        )

    def _retrieve(self, query: str) -> List[Dict[str, Any]]:
        """Retrieve relevant memories from Mem0."""
        if self._memory is None:
            print("[Mem0Agent] WARNING: _memory is None in _retrieve, returning empty list")
            return []
        user_id = self._get_user_id()
        results = self._memory.search(query=query, user_id=user_id, limit=self.retrieve_num)

        memories = []
        if results and "results" in results:
            for entry in results["results"]:
                memories.append({
                    "memory": entry.get("memory", ""),
                    "score": entry.get("score", 0.0),
                    "id": entry.get("id", ""),
                })

        return memories

    def query(
        self,
        question: str,
        system_message: Optional[str] = None,
        **kwargs
    ) -> AgentResponse:
        """Query the agent."""
        prepared = self.prepare_batch_query(question, system_message=system_message, **kwargs)
        response = self._llm_client.chat(prepared["messages"])
        return self.finalize_batch_query(prepared, response.content)

    def prepare_batch_query(
        self,
        question: str,
        system_message: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Retrieve locally and prepare Mem0's side-effect-free answer call."""
        # Retrieve relevant memories
        retrieved_memories = self._retrieve(question)

        # Bound question tokens first
        request_time = kwargs.get("batch_request_time") or time.strftime("%Y-%m-%d %H:%M:%S")
        full_question = f"{question}\n\nCurrent Time: {request_time}"
        full_question = self._truncate_to_tokens(full_question, self.max_question_tokens)

        base_system = system_message or ""
        reserved_tokens = self.max_tokens + 400
        available_tokens = max(self.max_context_tokens - reserved_tokens, 0)
        question_tokens = self._llm_client.count_tokens(full_question)
        base_system_tokens = self._llm_client.count_tokens(base_system) if base_system else 0
        memory_budget = max(available_tokens - question_tokens - base_system_tokens, 0)

        # Build memory string
        if retrieved_memories:
            memory_lines: List[str] = []
            used_tokens = 0
            for entry in retrieved_memories:
                line = f"- {entry['memory']}"
                line_tokens = self._llm_client.count_tokens(line)
                if used_tokens + line_tokens <= memory_budget:
                    memory_lines.append(line)
                    used_tokens += line_tokens
                else:
                    break
            memories_str = "\n".join(memory_lines)
            # Build system message with retrieved content
            memory_prompt = f"You are a helpful AI. Answer the question based on the following memories:\n{memories_str}\n"
            if system_message:
                full_system = f"{system_message}\n\n{memory_prompt}"
            else:
                full_system = memory_prompt
        else:
            full_system = system_message

        # Build retrieved_memories format
        formatted_memories = [
            {"memory": m["memory"], "type": "mem0_retrieval", "score": m.get("score", 0)}
            for m in retrieved_memories
        ]

        return {
            "messages": format_messages(full_question, full_system),
            "retrieved_count": len(retrieved_memories),
            "retrieved_memories": formatted_memories,
            "extra": {
                "method": "mem0",
                "embedding_model": self.embedding_model,
                "embedding_provider": self.embedding_provider,
            },
        }

    @staticmethod
    def finalize_batch_query(prepared: Dict[str, Any], content: str) -> AgentResponse:
        """Build Mem0's normal result shape from a completed answer request."""
        return AgentResponse(
            output=content,
            query_time=0.0,
            retrieved_count=prepared["retrieved_count"],
            retrieved_memories=prepared["retrieved_memories"],
            extra=prepared["extra"],
        )

    def reset(self) -> None:
        """Reset agent and release all resources."""
        import gc
        import shutil
        import logging

        logger = logging.getLogger(__name__)
        super().reset()

        # Close Mem0 Memory using its close() method
        if self._memory:
            try:
                if hasattr(self._memory, 'close'):
                    self._memory.close()
                    logger.info("[Mem0Agent] Closed Memory instance")
                else:
                    # Fallback: manually close resources
                    if hasattr(self._memory, "llm") and self._memory.llm:
                        if hasattr(self._memory.llm, "close"):
                            self._memory.llm.close()
                    if hasattr(self._memory, "vector_store") and self._memory.vector_store:
                        if hasattr(self._memory.vector_store, "client"):
                            self._memory.vector_store.client.close()
            except Exception as e:
                logger.warning(f"[Mem0Agent] Warning: Failed to close Memory: {e}")

            self._memory = None

        # Close the Q&A LLM client
        if self._llm_client:
            try:
                if hasattr(self._llm_client, "client"):
                    client = self._llm_client.client
                    if hasattr(client, "close"):
                        client.close()
                        logger.info("[Mem0Agent] Closed Q&A LLM client")
            except Exception as e:
                logger.warning(f"[Mem0Agent] Warning: Failed to close Q&A LLM client: {e}")
            self._llm_client = None

        # Clean up Qdrant directory to release file locks
        if self._qdrant_path:
            try:
                import os
                if os.path.exists(self._qdrant_path):
                    shutil.rmtree(self._qdrant_path)
                    logger.info(f"[Mem0Agent] Cleaned up Qdrant path: {self._qdrant_path}")
            except Exception as e:
                logger.warning(f"[Mem0Agent] Warning: Failed to clean up Qdrant path: {e}")
            self._qdrant_path = None

        # Force garbage collection to release resources immediately
        gc.collect()

        self._agent_start_time = time.time()

    def set_context_id(self, context_id: int) -> None:
        """Set context ID."""
        super().set_context_id(context_id)
        # Update user_id
        self._user_id = f"context_{context_id}"
        self._agent_start_time = time.time()

    @property
    def memory_count(self) -> int:
        """Get memory count in Mem0."""
        try:
            user_id = self._get_user_id()
            all_memories = self._memory.get_all(user_id=user_id)
            return len(all_memories.get("results", []))
        except Exception:
            return len(self._memory_chunks)
