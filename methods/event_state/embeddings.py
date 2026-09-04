"""Small, method-local dense embedding adapter."""

from __future__ import annotations

import threading
from typing import Any, List, Optional, Sequence


class DenseEmbedder:
    """Normalizes vectors from the repository's supported embedding families."""

    # Query workers use isolated Event-State stores, but local Transformer
    # weights are immutable and can be shared within one process.
    _local_clients: dict[tuple[str, str], Any] = {}
    _local_clients_lock = threading.Lock()

    def __init__(self, provider: str = "local", model: str = "sentence-transformers/all-MiniLM-L6-v2", model_path: Optional[str] = None, api_key: Optional[str] = None, base_url: Optional[str] = None, client: Optional[Any] = None) -> None:
        self.provider, self.model, self.model_path = provider.lower(), model, model_path
        self.api_key, self.base_url, self._client = api_key, base_url, client
        self._client_init_lock = threading.Lock()

    @staticmethod
    def _normalize(vector: Sequence[float]) -> List[float]:
        values = [float(value) for value in vector]
        magnitude = sum(value * value for value in values) ** 0.5
        return [value / magnitude for value in values] if magnitude else values

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        with self._client_init_lock:
            if self._client is not None:
                return self._client
            if self.provider in {"local", "huggingface"}:
                from sentence_transformers import SentenceTransformer
                cache_key = (self.provider, self.model_path or self.model)
                with self._local_clients_lock:
                    client = self._local_clients.get(cache_key)
                    if client is None:
                        client = SentenceTransformer(self.model_path or self.model)
                        self._local_clients[cache_key] = client
                self._client = client
            elif self.provider == "openai":
                from langchain_openai import OpenAIEmbeddings
                options = {"model": self.model}
                if self.api_key:
                    options["api_key"] = self.api_key
                if self.base_url:
                    options["base_url"] = self.base_url
                self._client = OpenAIEmbeddings(**options)
            else:
                raise ValueError(f"Unsupported Event-State embedding provider: {self.provider}")
        return self._client

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        if not texts:
            return []
        client = self._get_client()
        if hasattr(client, "encode"):
            return [self._normalize(vector) for vector in client.encode(list(texts), normalize_embeddings=True)]
        return [self._normalize(vector) for vector in client.embed_documents(list(texts))]

    def embed_query(self, text: str) -> List[float]:
        client = self._get_client()
        if hasattr(client, "encode"):
            return self._normalize(client.encode(text, normalize_embeddings=True))
        return self._normalize(client.embed_query(text))


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    return sum(float(a) * float(b) for a, b in zip(left, right))
