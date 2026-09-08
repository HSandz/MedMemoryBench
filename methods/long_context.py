"""Long Context Agent - concatenates all memory into context for LLM."""

from typing import Optional, List

from .base import BaseAgent, MemoryBuildResult, AgentResponse
from utils.llm_client import create_llm_client, format_messages


class LongContextAgent(BaseAgent):
    """Baseline method using LLM's long context capability."""

    METHOD_TYPE = "baseline"
    SNAPSHOT_VERSION = 1

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 1.0,
        max_tokens: int = 2000,
        provider: str = "openai",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_context_tokens: int = 100000,
        truncation_strategy: str = "oldest_first",
        **kwargs
    ):
        super().__init__(model, temperature, max_tokens, **kwargs)

        self.max_context_tokens = max_context_tokens
        self.truncation_strategy = truncation_strategy

        self._llm_client = create_llm_client(
            provider=provider,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
            base_url=base_url,
            **kwargs.get("llm_client_kwargs", {}),
        )

        self._context = ""

    def memorize(self, text: str, **kwargs) -> MemoryBuildResult:
        """Add text to memory context."""
        prev_context_len = len(self._context)

        self._memory_chunks.append(text)

        if self._context:
            self._context += "\n\n" + text
        else:
            self._context = text

        self._truncate_if_needed()
        self._is_initialized = True

        return MemoryBuildResult(
            success=True,
            method="long_context",
            action="append_context",
            input_content=text,
            stored_content=text,
            memory_entries=[],
            chunk_count=len(self._memory_chunks),
            extra={
                "context_length_before": prev_context_len,
                "context_length_after": len(self._context),
            }
        )

    def _truncate_if_needed(self) -> None:
        """Truncate context if exceeds limit."""
        current_tokens = self.count_tokens(self._context)

        if current_tokens <= self.max_context_tokens:
            return

        if self.truncation_strategy == "oldest_first":
            # Retain the tail of a single oversized session rather than
            # accidentally evicting the only stored item.
            if len(self._memory_chunks) == 1:
                encoded = self._tokenizer.encode(self._context)
                self._context = self._tokenizer.decode(encoded[-self.max_context_tokens:])
                self._memory_chunks = [self._context]
                return
            while self._memory_chunks and self.count_tokens(self._context) > self.max_context_tokens:
                self._memory_chunks.pop(0)
                self._context = "\n\n".join(self._memory_chunks)
        else:
            encoded = self._tokenizer.encode(self._context)
            self._context = self._tokenizer.decode(encoded[-self.max_context_tokens:])

    def query(
        self,
        question: str,
        system_message: Optional[str] = None,
        **kwargs
    ) -> AgentResponse:
        """Query the agent."""
        prepared = self.prepare_batch_query(question, system_message=system_message)
        messages = prepared["messages"]
        response = self._llm_client.chat(messages)
        return self.finalize_batch_query(prepared, response.content)

    def prepare_batch_query(
        self,
        question: str,
        system_message: Optional[str] = None,
        **kwargs,
    ) -> dict:
        """Prepare the immutable final-answer request for Vertex batch mode."""
        full_message = f"{self._context}\n\n{question}" if self._context else question
        return {
            "messages": format_messages(full_message, system_message),
            "retrieved_count": 0,
            "retrieved_memories": [],
            "extra": {
                "context_tokens": self.count_tokens(self._context) if self._context else 0,
                "method": "long_context",
            },
        }

    @staticmethod
    def finalize_batch_query(prepared: dict, content: str) -> AgentResponse:
        """Build the normal response shape from a completed batch row."""
        return AgentResponse(
            output=content,
            query_time=0.0,
            retrieved_count=prepared["retrieved_count"],
            retrieved_memories=prepared["retrieved_memories"],
            extra=prepared["extra"],
        )

    def reset(self) -> None:
        """Reset agent."""
        super().reset()
        self._context = ""

    def supports_memory_snapshots(self) -> bool:
        """The retained text is sufficient to restore this baseline exactly."""
        return True

    def export_memory_state(self, context_id=None) -> dict:
        """Export retained context without serializing the LLM client."""
        return {
            "method": "long_context",
            "snapshot_version": self.SNAPSHOT_VERSION,
            "context_id": self._context_id if context_id is None else context_id,
            "memory_chunks": list(self._memory_chunks),
            "context": self._context,
        }

    def import_memory_state(self, state: dict, context_id=None) -> None:
        """Restore a previously exported retained context."""
        if (
            not isinstance(state, dict)
            or state.get("method") != "long_context"
            or state.get("snapshot_version") != self.SNAPSHOT_VERSION
        ):
            raise ValueError("Invalid long-context memory snapshot")
        chunks = state.get("memory_chunks")
        context = state.get("context")
        if not isinstance(chunks, list) or not all(isinstance(chunk, str) for chunk in chunks):
            raise ValueError("Long-context snapshot chunks are invalid")
        if not isinstance(context, str):
            raise ValueError("Long-context snapshot context is invalid")
        self._memory_chunks = list(chunks)
        self._context = context
        self._is_initialized = bool(chunks or context)
        self._context_id = context_id if context_id is not None else state.get("context_id")

    @property
    def context(self) -> str:
        """Get current context."""
        return self._context

    @property
    def context_tokens(self) -> int:
        """Get current context token count."""
        return self.count_tokens(self._context) if self._context else 0
