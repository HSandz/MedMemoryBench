"""Gemini Enterprise LLM adapter for the vendored MemOS runtime."""

from collections.abc import Generator

from memos.configs.llm import BaseLLMConfig
from memos.llms.base import BaseLLM


class GeminiLLM(BaseLLM):
    """Use MedMemoryBench's service-account-backed Gemini client."""

    def __init__(self, config: BaseLLMConfig):
        from utils.llm_client import create_llm_client

        self.config = config
        self.client = create_llm_client(
            provider="gemini",
            model=config.model_name_or_path,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )

    def generate(self, messages, **kwargs) -> str:
        response = self.client.chat(
            list(messages),
            temperature=kwargs.get("temperature", self.config.temperature),
            max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
            top_p=kwargs.get("top_p", self.config.top_p),
            top_k=kwargs.get("top_k", self.config.top_k),
        )
        return response.content

    def generate_stream(self, messages, **kwargs) -> Generator[str, None, None]:
        yield self.generate(messages, **kwargs)
