import hashlib
import json
import os
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from typing import List, Optional

import backoff
import numpy as np
import requests
import torch
from filelock import FileLock
from openai import AzureOpenAI, OpenAI
from tqdm import tqdm

from ..utils.config_utils import BaseConfig
from ..utils.logging_utils import get_logger
from .base import BaseEmbeddingModel, EmbeddingConfig

logger = get_logger(__name__)


def _get_embedding_dimension(embedding_model_name: str) -> int:
    """Get the expected embedding dimension based on the model name."""
    if "text-embedding-3-large" in embedding_model_name:
        return 3072
    elif "text-embedding-3-small" in embedding_model_name or "text-embedding-ada-002" in embedding_model_name:
        return 1536
    elif "Qwen3-Embedding-8B" in embedding_model_name or "NV-Embed-v2" in embedding_model_name:
        return 4096
    else:
        # Default dimension for unknown models
        logger.warning(f"Unknown embedding model {embedding_model_name}, defaulting to 1536 dimensions")
        return 1536


def _is_retryable_error(e):
    """Check if an exception is retryable."""
    # Retry on any requests exception (network issues, timeouts, etc.)
    if isinstance(e, requests.exceptions.RequestException):
        return True

    if isinstance(e, Exception) and "Request failed with status code 422" in str(e):
        return False

    # Retry on other HTTP errors (non-200 status code)
    if isinstance(e, Exception) and "Request failed with status code" in str(e):
        return True

    return False


def _make_http_request_with_retry(url, payload, headers=None, max_retries=5):
    """Make HTTP request with retry using backoff decorator.

    Uses exponential backoff with jitter:
    - Base delay: 1 second
    - Max delay: 60 seconds (capped to avoid excessive waiting)
    - Exponential factor: 2 (doubles each retry)
    - Random jitter to avoid thundering herd
    """

    @backoff.on_exception(
        backoff.expo,
        (requests.exceptions.RequestException, Exception),
        giveup=lambda e: not _is_retryable_error(e),
        max_tries=max_retries,
        base=1,  # Base delay of 1 second
        factor=2,  # Exponential factor (1s, 2s, 4s, 8s, 16s, ...)
        max_value=60,  # Maximum delay of 60 seconds
        jitter=backoff.random_jitter,  # Add random jitter
        on_backoff=lambda details: logger.warning(
            f"HTTP request failed (attempt {details['tries']}/{max_retries}): {details['exception']}. "
            f"Retrying in {details['wait']:.2f} seconds..."
        ),
        on_giveup=lambda details: logger.error(
            f"HTTP request failed after {details['tries']} attempts. Last error: {details['exception']}"
        ),
    )
    def _make_request():
        response = requests.post(url, json=payload, headers=headers)
        if response.status_code != 200:
            logger.error(f"Request failed with status code {response.status_code}")
        return response

    return _make_request()


class OpenAIEmbeddingModel(BaseEmbeddingModel):

    def __init__(
        self,
        global_config: Optional[BaseConfig] = None,
        embedding_model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = "https://api.openai.com/v1/embeddings",
        max_retries: int = 3,
        **kwargs,
    ) -> None:
        super().__init__(global_config=global_config)

        # Store retry configuration
        self.max_retries = max_retries

        if embedding_model_name is not None:
            self.embedding_model_name = embedding_model_name
            logger.debug(
                f"Overriding {self.__class__.__name__}'s embedding_model_name with: {self.embedding_model_name}"
            )

        self.base_url = base_url

        # Initialize OpenAI client
        if kwargs.get("use_azure", False) and "text-embedding-3-" in self.embedding_model_name:
            # Use Azure OpenAI for embeddings
            assert os.getenv("AZURE_OPENAI_API_KEY") or os.getenv(
                "AZURE_OPENAI_AD_TOKEN"
            ), "Azure OpenAI API key or AD token must be set"
            assert os.getenv("OPENAI_API_VERSION"), "OPENAI_API_VERSION must be set"
            assert os.getenv("AZURE_OPENAI_ENDPOINT"), "AZURE_OPENAI_ENDPOINT must be set"
            self.openai_client = AzureOpenAI(timeout=60, max_retries=5)
        else:
            # Use OpenAI client for any online embedding service
            assert self.base_url is not None, "Base URL must be provided for OpenAI embedding service"

            if "text-embedding-3-" in self.embedding_model_name:
                if api_key is None:
                    api_key = os.getenv("OPENAI_API_KEY")
                assert api_key is not None, "API key must be provided or OPENAI_API_KEY must be set"
            else:
                # For custom/local embedding endpoints that don't require authentication,
                # use a placeholder key. The OpenAI client requires a non-None key, but
                # the custom endpoint will ignore it.
                if api_key is None:
                    api_key = os.getenv("EMBEDDING_API_KEY", "not-needed-for-local-server")

            self.openai_client = OpenAI(api_key=api_key, timeout=60, max_retries=5, base_url=base_url)

        self._init_embedding_config()

        # Initializing the embedding model
        logger.debug(
            f"Initializing {self.__class__.__name__}'s embedding model with params: {self.embedding_config.model_init_params}"
        )

    def _init_embedding_config(self) -> None:
        """
        Extract embedding model-specific parameters to init the EmbeddingConfig.

        Returns:
            None
        """

        config_dict = {
            "embedding_model_name": self.embedding_model_name,
            "norm": self.global_config.embedding_return_as_normalized,
            # "max_seq_length": self.global_config.embedding_max_seq_len,
            "model_init_params": {
                # "model_name_or_path": self.embedding_model_name2mode_name_or_path[self.embedding_model_name],
                "pretrained_model_name_or_path": self.embedding_model_name,
                "trust_remote_code": True,
                # "torch_dtype": "auto",
                "device_map": "auto",  # added this line to use multiple GPUs
                # **kwargs
            },
            "encode_params": {
                "max_length": self.global_config.embedding_max_seq_len,  # 32768 from official example,
                "instruction": "",
                "batch_size": self.global_config.embedding_batch_size,
                "num_workers": 32,
            },
        }

        self.embedding_config = EmbeddingConfig.from_dict(config_dict=config_dict)
        logger.debug(f"Init {self.__class__.__name__}'s embedding_config: {self.embedding_config}")

    def encode(self, texts: List[str], **kwargs) -> np.array:
        # If an instruction is provided, prepend it to each text using the delimiter.
        if kwargs.get("instruction") is not None:
            instruction = kwargs.get("instruction")
            if instruction is not None and instruction != "":
                instruction += "<|endofprefix|>"
            texts = [f"{instruction}{text}" for text in texts]

        if hasattr(self, "openai_client") and self.openai_client is not None:
            # Use OpenAI client (Azure or standard)
            try:
                response = self.openai_client.embeddings.create(input=texts, model=self.embedding_model_name)
                embeddings = [item.embedding for item in response.data]
            except Exception as e:
                # Handle content filtering or other API errors by creating fallback embeddings
                if "422" in str(e):
                    logger.warning(
                        f"API rejected batch (content filtering or validation), creating fallback embeddings: {e}"
                    )
                    embeddings = [self._create_fallback_embedding(text, "API content filtering") for text in texts]
                else:
                    raise e

        else:
            # Count and handle empty strings
            empty_count = sum(1 for text in texts if text == "")
            if len(texts) > 1:
                assert empty_count < len(texts), "All texts are empty, cannot proceed with embedding"

            if empty_count > 0:
                logger.warning(f"Found {empty_count} empty strings out of {len(texts)} texts. Replacing with None.")
                texts = ["None" if text == "" else text for text in texts]

            # Use direct HTTP request for custom endpoints
            payload = {"input": texts, "model": self.embedding_model_name, "encoding_format": "float"}

            if self.base_url is None:
                raise ValueError("Base URL not configured")

            try:
                response = _make_http_request_with_retry(self.base_url, payload, None, self.max_retries)
                data = response.json()
                embeddings = [item["embedding"] for item in data.get("data", [])]
            except Exception as e:
                # Handle 422 content filtering errors by creating fallback embeddings
                if "422" in str(e):
                    logger.warning(
                        f"HTTP request rejected (content filtering or validation), creating fallback embeddings: {e}"
                    )
                    embeddings = [self._create_fallback_embedding(text, "HTTP content filtering") for text in texts]
                else:
                    raise e

        return np.array(embeddings)

    def _encode_single_text(self, text: str, **kwargs) -> np.array:
        """
        Encode a single text using the embedding API.
        Used for parallel processing with text-embedding-3- models.
        """
        # If an instruction is provided, prepend it to the text using the delimiter.
        if kwargs.get("instruction") is not None:
            instruction = kwargs.get("instruction")
            if instruction is not None and instruction != "":
                instruction += "<|endofprefix|>"
            text = f"{instruction}{text}"

        if hasattr(self, "openai_client") and self.openai_client is not None:
            # Use OpenAI client (Azure or standard)
            try:
                response = self.openai_client.embeddings.create(input=[text], model=self.embedding_model_name)
                embeddings = [item.embedding for item in response.data]
            except Exception as e:
                # Handle content filtering or other API errors by creating fallback embeddings
                if "422" in str(e):
                    logger.warning(
                        f"API rejected single text (content filtering or validation), creating fallback embedding: {e}"
                    )
                    embeddings = [self._create_fallback_embedding(text, "API content filtering")]
                else:
                    raise e

        else:
            # Use direct HTTP request for custom endpoints
            payload = {"input": [text], "model": self.embedding_model_name, "encoding_format": "float"}

            if self.base_url is None:
                raise ValueError("Base URL not configured")

            try:
                response = _make_http_request_with_retry(self.base_url, payload, None, self.max_retries)
                data = response.json()
                embeddings = [item["embedding"] for item in data.get("data", [])]
            except Exception as e:
                # Handle 422 content filtering errors by creating fallback embeddings
                if "422" in str(e):
                    logger.warning(
                        f"HTTP request rejected for single text (content filtering or validation), creating fallback embedding: {e}"
                    )
                    embeddings = [self._create_fallback_embedding(text, "HTTP content filtering")]
                else:
                    raise e

        return np.array(embeddings[0]) if embeddings else np.array([])

    def _encode_texts_parallel(self, texts: List[str], max_workers: int = 10, dim=3072, **kwargs) -> np.array:
        """
        Encode texts in parallel using ThreadPoolExecutor.
        Each request contains only one text to prevent failure of entire batch.
        """
        embeddings = [None] * len(texts)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_index = {
                executor.submit(self._encode_single_text, text, **kwargs): i for i, text in enumerate(texts)
            }

            # Collect results with progress bar
            with tqdm(total=len(texts), desc="Parallel Encoding") as pbar:
                for future in as_completed(future_to_index):
                    index = future_to_index[future]
                    try:
                        embedding = future.result()
                        embeddings[index] = embedding
                    except Exception as exc:
                        logger.error(f"Text at index {index} generated an exception: {exc}")
                        # Create a zero embedding as fallback
                        embeddings[index] = np.zeros(dim)
                    pbar.update(1)

        return np.array(embeddings)

    def batch_encode(self, texts: List[str], **kwargs) -> np.array:
        if isinstance(texts, str):
            texts = [texts]

        params = deepcopy(self.embedding_config.encode_params)
        if kwargs:
            params.update(kwargs)

        if "instruction" in kwargs:
            if kwargs["instruction"] != "":
                params["instruction"] = f"Instruct: {kwargs['instruction']}\nQuery: "
            # del params["instruction"]

        logger.debug(f"Calling {self.__class__.__name__} with:\n{params}")
        batch_size = params.pop("batch_size", 16)

        try:
            if len(texts) <= batch_size:
                # Pass texts directly to encode, not in params
                results = self.encode(texts, **params)
            else:
                pbar = tqdm(total=len(texts), desc="Batch Encoding")
                results = []
                for i in range(0, len(texts), batch_size):
                    batch_texts = texts[i : i + batch_size]
                    results.append(self.encode(batch_texts, **params))
                    pbar.update(batch_size)
                results = np.concatenate(results)
        except Exception as e:
            # If content was rejected (e.g., content filtering / validation), retry with batch_size=1
            if "422" in str(e):
                logger.warning(
                    f"Content filtering/validation failed for batch of size {batch_size}, retrying with batch_size=1: {e}"
                )
                results = []
                for text in texts:
                    try:
                        result = self.encode([text], **params)
                        results.append(result)
                    except Exception as single_e:
                        # Create fallback embedding for this single text
                        logger.warning(f"Failed to encode single text '{text}': {single_e}")
                        fallback = self._create_fallback_embedding(text, "batch retry content filtering")
                        results.append(np.array([fallback]))
                results = np.concatenate(results)
            else:
                raise e

        if isinstance(results, torch.Tensor):
            results = results.cpu()
            results = results.numpy()
        if self.embedding_config.norm:
            results = (results.T / np.linalg.norm(results, axis=1)).T

        return results

    def _create_fallback_embedding(self, text: str, reason: str = "content filtering"):
        """Create a fallback zero embedding when the service refuses to generate one."""
        dim = _get_embedding_dimension(self.embedding_model_name)
        logger.warning(f"Creating fallback zero embedding for text due to {reason}. Text length: {len(text)}")
        return np.zeros(dim, dtype=np.float32)


class CacheOpenAIEmbeddingModel(OpenAIEmbeddingModel):
    """Cached version of OpenAI embedding service."""

    def __init__(
        self,
        cache_filename: Optional[str] = None,
        global_config: Optional[BaseConfig] = None,
        embedding_model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_retries: int = 5,
        **kwargs,
    ) -> None:
        super().__init__(
            global_config=global_config,
            embedding_model_name=embedding_model_name,
            api_key=api_key,
            base_url=base_url,
            max_retries=max_retries,
            **kwargs,
        )

        # Setup cache
        if global_config and hasattr(global_config, "dataset"):
            self.cache_dir = f"outputs/{global_config.dataset}/embedding_cache"
        else:
            self.cache_dir = "outputs/default/embedding_cache"
        os.makedirs(self.cache_dir, exist_ok=True)
        if cache_filename is None:
            cache_filename = f"{self.embedding_model_name.replace('/', '_')}_embedding_cache.sqlite"
        self.cache_file_name = os.path.join(self.cache_dir, cache_filename)

        logger.debug(
            f"{self.__class__.__name__} initialized, model name: {embedding_model_name}, base URL: {self.base_url}, cache file: {self.cache_file_name}"
        )

    def encode(self, texts: List[str], **kwargs) -> np.array:
        """Cache embeddings using SQLite database."""
        # Create cache key from texts and parameters
        instruction = kwargs.get("instruction", "")
        model = self.embedding_model_name

        # Build key data for each text
        cache_keys = []
        for text in texts:
            key_data = {
                "text": text,
                "model": model,
                "instruction": instruction,
                "max_length": kwargs.get("max_length", ""),
            }
            key_str = json.dumps(key_data, sort_keys=True, default=str)
            key_hash = hashlib.sha256(key_str.encode("utf-8")).hexdigest()
            cache_keys.append(key_hash)

        lock_file = self.cache_file_name + ".lock"
        cached_embeddings = [None] * len(texts)
        missed_indices = []

        # Try to get cached embeddings
        with FileLock(lock_file):
            conn = sqlite3.connect(self.cache_file_name)
            c = conn.cursor()
            c.execute(
                """
                      CREATE TABLE IF NOT EXISTS embeddings_cache
                      (
                          key TEXT PRIMARY KEY,
                          embedding BLOB,
                          embedding_dim INTEGER
                      )
                      """
            )
            conn.commit()

            for i, key_hash in enumerate(cache_keys):
                c.execute("SELECT embedding, embedding_dim FROM embeddings_cache WHERE key = ?", (key_hash,))
                row = c.fetchone()
                if row is not None:
                    # Convert BLOB back to numpy array with proper shape
                    embedding_blob, embedding_dim = row
                    embedding = np.frombuffer(embedding_blob, dtype=np.float32)
                    if embedding_dim:
                        embedding = embedding.reshape(-1)  # Ensure 1D for single embedding
                    cached_embeddings[i] = embedding
                else:
                    missed_indices.append(i)
            conn.close()

        # Get embeddings for cache misses
        if missed_indices:
            missed_texts = [texts[i] for i in missed_indices]
            try:
                new_embeddings = super().encode(missed_texts, **kwargs)  # Call OpenAI embedding service
            except Exception as e:
                logger.error(f"Failed to get embeddings for {len(missed_texts)} texts: {e}")
                # Create fallback embeddings for all missed texts
                new_embeddings = np.array(
                    [self._create_fallback_embedding(text, "cache miss") for text in missed_texts]
                )

            # Insert new embeddings into cache and results
            with FileLock(lock_file):
                conn = sqlite3.connect(self.cache_file_name)
                c = conn.cursor()
                for idx, emb_idx in enumerate(missed_indices):
                    if idx < len(new_embeddings):
                        embedding = new_embeddings[idx]
                        cached_embeddings[emb_idx] = embedding

                        # Store in cache with embedding dimension
                        key_hash = cache_keys[emb_idx]
                        emb_bytes = embedding.astype(np.float32).tobytes()
                        embedding_dim = len(embedding)
                        c.execute(
                            "INSERT OR REPLACE INTO embeddings_cache (key, embedding, embedding_dim) VALUES (?, ?, ?)",
                            (key_hash, emb_bytes, embedding_dim),
                        )
                    else:
                        # Fallback: create zero embedding if we don't have enough embeddings
                        logger.warning(f"Creating fallback embedding for cache miss at index {idx}")
                        fallback_embedding = self._create_fallback_embedding(
                            missed_texts[idx], "insufficient embeddings"
                        )
                        cached_embeddings[emb_idx] = fallback_embedding

                        # Store fallback in cache
                        key_hash = cache_keys[emb_idx]
                        emb_bytes = fallback_embedding.astype(np.float32).tobytes()
                        embedding_dim = len(fallback_embedding)
                        c.execute(
                            "INSERT OR REPLACE INTO embeddings_cache (key, embedding, embedding_dim) VALUES (?, ?, ?)",
                            (key_hash, emb_bytes, embedding_dim),
                        )
                conn.commit()
                conn.close()

        logger.debug(f"Cache stats: {len(texts) - len(missed_indices)} hits, {len(missed_indices)} misses")
        return np.array(cached_embeddings)
