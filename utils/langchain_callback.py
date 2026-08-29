"""LangChain callback handler for token usage tracking."""

import logging
import time
from typing import Any, Dict, List, Optional, Union
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.outputs import LLMResult

logger = logging.getLogger(__name__)


class TokenUsageCallbackHandler(BaseCallbackHandler):
    """Callback handler to track LLM token usage in LangChain."""

    def __init__(self, model_name: str = "langchain"):
        super().__init__()
        self._model_name = model_name
        self._start_time: Optional[float] = None

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> None:
        self._start_time = time.time()

    def on_chat_model_start(self, serialized: Dict[str, Any], messages: List[List], **kwargs: Any) -> None:
        self._start_time = time.time()

    @staticmethod
    def _usage_counts(usage: Any) -> tuple[int, int, int, int]:
        """Read normalized input, total output, visible, and thinking counts."""
        from utils.llm_client import extract_usage_token_counts

        return extract_usage_token_counts(usage)

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        latency = time.time() - self._start_time if self._start_time else 0.0
        self._start_time = None

        input_tokens = 0
        output_tokens = 0
        visible_output_tokens = 0
        thinking_tokens = 0

        if response.llm_output:
            for key in ("token_usage", "usage", "usage_metadata"):
                input_tokens, output_tokens, visible_output_tokens, thinking_tokens = (
                    self._usage_counts(response.llm_output.get(key))
                )
                if input_tokens or output_tokens:
                    break

        if input_tokens == 0 and output_tokens == 0 and response.generations:
            for gen_list in response.generations:
                for gen in gen_list:
                    if hasattr(gen, "generation_info") and gen.generation_info:
                        for key in ("usage", "usage_metadata", "token_usage"):
                            input_tokens, output_tokens, visible_output_tokens, thinking_tokens = (
                                self._usage_counts(gen.generation_info.get(key))
                            )
                            if input_tokens or output_tokens:
                                break
                        if input_tokens or output_tokens:
                            break

                    if hasattr(gen, "message") and hasattr(gen.message, "usage_metadata"):
                        input_tokens, output_tokens, visible_output_tokens, thinking_tokens = (
                            self._usage_counts(gen.message.usage_metadata)
                        )
                        if input_tokens or output_tokens:
                            break

                    # Some LangChain Google versions retain the raw Vertex
                    # field only in response metadata instead of promoting it
                    # to ``AIMessage.usage_metadata``.
                    if hasattr(gen, "message"):
                        response_metadata = getattr(gen.message, "response_metadata", None)
                        if isinstance(response_metadata, dict):
                            for key in ("usage_metadata", "usageMetadata", "usage", "token_usage"):
                                input_tokens, output_tokens, visible_output_tokens, thinking_tokens = (
                                    self._usage_counts(response_metadata.get(key))
                                )
                                if input_tokens or output_tokens:
                                    break
                            if input_tokens or output_tokens:
                                break
                if input_tokens or output_tokens:
                    break

        if input_tokens == 0 and output_tokens == 0:
            logger.warning(
                "[TokenUsageCallback] No token usage data found in LLM response "
                f"(model={self._model_name}). This LLM call will NOT be counted "
                "in token statistics. Check if the API provider returns usage data."
            )
            return

        from utils.llm_client import get_usage_tracker, LLMResponse

        llm_response = LLMResponse(
            content="",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            visible_output_tokens=visible_output_tokens,
            thinking_tokens=thinking_tokens,
            latency=latency,
            model=self._model_name,
        )
        get_usage_tracker().record(llm_response)
