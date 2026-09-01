"""MIRIX adapter for the repository's service-account-backed Gemini client."""

import asyncio

from google.genai import types

from mirix.llm_api.google_ai_client import GoogleAIClient


class GoogleVertexClient(GoogleAIClient):
    """Keep MIRIX's Google message/tool conversion while using Agent Platform auth."""

    def __init__(self, llm_config):
        super().__init__(llm_config=llm_config)
        from utils.llm_client import create_llm_client

        self._vertex_client = create_llm_client(
            provider="gemini",
            model=llm_config.model,
            temperature=llm_config.temperature,
            max_tokens=llm_config.max_tokens or 4096,
        )

    async def build_request_data(self, *args, **kwargs) -> dict:
        request = await super().build_request_data(*args, **kwargs)
        config = request.pop("generation_config")
        config["tools"] = request.pop("tools")
        config["tool_config"] = request.pop("tool_config")
        return {"contents": request["contents"], "config": config}

    async def request(self, request_data: dict) -> dict:
        response = await asyncio.to_thread(
            self._run_request,
            request_data,
        )
        return response.model_dump(by_alias=True)

    def _run_request(self, request_data: dict):
        """Keep tool calls on the shared 50-attempt Gemini retry policy."""
        from utils.llm_client import run_with_gemini_retry

        return run_with_gemini_retry(
            self._vertex_client.client.models.generate_content,
            model=self.llm_config.model,
            contents=request_data["contents"],
            config=types.GenerateContentConfig(**request_data["config"]),
        )
