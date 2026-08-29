"""Gemini LLM for the vendored Mem0 runtime."""

from typing import Dict, List, Optional

from methods.mem0.configs.llms.base import BaseLlmConfig
from methods.mem0.llms.base import LLMBase


class GeminiLLM(LLMBase):
    """Route Mem0 extraction requests through the configured Gemini provider."""

    def __init__(self, config: Optional[BaseLlmConfig] = None):
        super().__init__(config)
        from utils.llm_client import create_llm_client

        self.config.model = self.config.model or "gemini-2.5-flash"
        self.client = create_llm_client(
            provider=self.config.gemini_provider,
            model=self.config.model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            api_key=self.config.api_key,
            reasoning_effort=self.config.reasoning_effort,
        )

    def generate_response(
        self,
        messages: List[Dict[str, str]],
        response_format=None,
        tools: Optional[List[Dict]] = None,
        tool_choice: str = "auto",
    ):
        if tools:
            raise NotImplementedError("Mem0 graph-memory tool calls are not enabled by this adapter.")

        response = self.client.chat(
            messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            response_format=response_format,
        )
        return response.content
