import functools
import hashlib
import json
import os
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from typing import Dict, List, Tuple

from filelock import FileLock
from openai import OpenAI
from tqdm import tqdm

from remem.utils.config_utils import BaseConfig
from remem.utils.llm_utils import TextChatMessage
from remem.utils.logging_utils import get_logger

from .base import BaseLLM, LLMConfig

logger = get_logger(__name__)


def cache_response(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs) -> tuple[str, Dict, bool]:
        # get messages from args or kwargs
        if args:
            messages = args[0]
        else:
            messages = kwargs.get("messages")
        if messages is None:
            raise ValueError("Missing required 'messages' parameter for caching.")

        # get model, seed and temperature from kwargs or self.llm_config.generate_params
        gen_params = getattr(self, "llm_config", {}).generate_params if hasattr(self, "llm_config") else {}
        model = kwargs.get("model", gen_params.get("model"))
        seed = kwargs.get("seed", gen_params.get("seed"))
        temperature = kwargs.get("temperature", gen_params.get("temperature"))
        response_format = kwargs.get("response_format", gen_params.get("response_format"))

        # build key data, convert to JSON string and hash to generate key_hash
        key_data = {
            "messages": messages,  # messages requires JSON serializable
            "model": model,
            "seed": seed,
            "temperature": temperature,
            "response_format": response_format,
        }
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        key_hash = hashlib.sha256(key_str.encode("utf-8")).hexdigest()

        # the file name of lock, ensure mutual exclusion when accessing concurrently
        lock_file = self.cache_file_name + ".lock"

        # Try to read from SQLite cache with timeout
        try:
            with FileLock(lock_file, timeout=30):  # 30 second timeout
                conn = sqlite3.connect(self.cache_file_name)
                c = conn.cursor()
                # if the table does not exist, create it
                c.execute(
                    """
                    CREATE TABLE IF NOT EXISTS cache (
                        key TEXT PRIMARY KEY,
                        message TEXT,
                        metadata TEXT
                    )
                """
                )
                conn.commit()  # commit to save the table creation
                c.execute("SELECT message, metadata FROM cache WHERE key = ?", (key_hash,))
                row = c.fetchone()
                conn.close()
                if row is not None:  # cache hit
                    message, metadata_str = row
                    metadata = json.loads(metadata_str)
                    # return cached result and mark as hit
                    return message, metadata, True
        except Exception as e:
            logger.warning(f"Cache read failed: {e}. Proceeding without cache.")

        # if cache miss, call the original function to get the result
        try:
            result = func(self, *args, **kwargs)
            response, metadata = result
        except Exception as e:
            logger.error(f"Error during LLM inference: {e}")
            return "", {"error": e}, False

        # insert new result into cache
        try:
            with FileLock(lock_file, timeout=30):  # 30 second timeout
                conn = sqlite3.connect(self.cache_file_name)
                c = conn.cursor()
                # make sure the table exists again (if it doesn't exist, it would be created)
                c.execute(
                    """
                    CREATE TABLE IF NOT EXISTS cache (
                        key TEXT PRIMARY KEY,
                        message TEXT,
                        metadata TEXT
                    )
                """
                )
                metadata_str = json.dumps(metadata)
                c.execute(
                    "INSERT OR REPLACE INTO cache (key, message, metadata) VALUES (?, ?, ?)",
                    (key_hash, response, metadata_str),
                )
                conn.commit()
                conn.close()
        except Exception as e:
            logger.warning(f"Cache write failed: {e}. Continuing without caching.")

        return response, metadata, False

    return wrapper


class CacheOpenAI(BaseLLM):
    """OpenAI LLM implementation."""

    @classmethod
    def from_experiment_config(cls, global_config: BaseConfig):
        config_dict = global_config.__dict__
        cache_dir = os.path.join(global_config.save_dir, "llm_cache")
        return cls(cache_dir=cache_dir, **config_dict)

    def __init__(
        self,
        cache_dir,
        cache_filename: str = None,
        llm_name: str = "gpt-4o-mini",
        api_key: str = None,
        llm_base_url: str = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        self.llm_name = llm_name  # Set llm_name first
        if cache_filename is None:
            cache_filename = f"{llm_name.replace('/', '_')}_cache.sqlite"
        self.cache_file_name = os.path.join(self.cache_dir, cache_filename)

        # Log cache directory and file path for debugging
        logger.debug(
            f"[{self.llm_name}] CacheOpenAI initialized, cache_dir: {self.cache_dir}, cache_filename: {cache_filename}, exists: {os.path.exists(self.cache_file_name)}"
        )

        # Check for stale lock files and safely clean them
        lock_file_path = self.cache_file_name + ".lock"
        if os.path.exists(lock_file_path):
            logger.debug(f"[{self.llm_name}] Lock file exists: {lock_file_path}")

            # Try to safely remove the lock file
            try:
                # Test if we can acquire the lock with a very short timeout
                # If we can acquire it, it means no other process is using it
                with FileLock(lock_file_path, timeout=0.1):
                    # We got the lock, which means it was stale
                    logger.debug(f"[{self.llm_name}] Successfully acquired lock, it was stale")
                # The lock is automatically released when exiting the 'with' block

                # Now safely remove the stale lock file
                if os.path.exists(lock_file_path):
                    os.remove(lock_file_path)
                    logger.debug(f"[{self.llm_name}] Removed stale lock file: {lock_file_path}")

            except Exception as e:
                # If we can't acquire the lock, another process might be using it
                logger.warning(f"[{self.llm_name}] Cannot remove lock file (might be in use by another process): {e}")
                logger.warning(f"[{self.llm_name}] Will proceed with timeout-based locking")
        self.llm_base_url = llm_base_url
        self._init_llm_config(**kwargs)

        # Initialize OpenAI client
        if kwargs.get("use_azure", False):
            # Use Azure OpenAI
            from openai import AzureOpenAI

            assert os.getenv("AZURE_OPENAI_API_KEY") or os.getenv(
                "AZURE_OPENAI_AD_TOKEN"
            ), "Azure OpenAI API key or AD token must be set"
            assert os.getenv("OPENAI_API_VERSION"), "OPENAI_API_VERSION must be set"
            assert os.getenv("AZURE_OPENAI_ENDPOINT"), "AZURE_OPENAI_ENDPOINT must be set"
            self.openai_client = AzureOpenAI(timeout=60, max_retries=5)
        else:
            # Use standard OpenAI API
            if api_key is None:
                api_key = os.getenv("OPENAI_API_KEY")
            assert api_key is not None, "OPENAI_API_KEY must be set or provided as api_key parameter"
            assert self.llm_base_url is not None, "llm_base_url must be set"
            self.openai_client = OpenAI(base_url=self.llm_base_url, api_key=api_key, timeout=60, max_retries=5)

    def _init_llm_config(self, **kwargs) -> None:
        config_dict = {
            "llm_name": self.llm_name,
            "llm_base_url": self.llm_base_url,
            "generate_params": {
                "model": self.llm_name,
                "n": kwargs.get("num_gen_choices", 1),
                "seed": kwargs.get("seed", 0),
                "temperature": kwargs.get("temperature", 0.0),
            },
        }
        self.llm_config = LLMConfig.from_dict(config_dict=config_dict)
        logger.debug(f"Init {self.__class__.__name__}'s llm_config: {self.llm_config}")

    @cache_response
    def infer(self, messages: List[TextChatMessage], **kwargs) -> Tuple[List[TextChatMessage], dict]:
        params = deepcopy(self.llm_config.generate_params)
        if kwargs:
            params.update(kwargs)
        params["messages"] = messages
        logger.debug(f"Calling OpenAI GPT API with:\n{params}")

        if params.get("model").startswith("Qwen/Qwen3-"):
            if "extra_body" not in params:
                params["extra_body"] = {}
            if "chat_template_kwargs" not in params["extra_body"]:
                params["extra_body"]["chat_template_kwargs"] = {}
            params["extra_body"]["chat_template_kwargs"]["enable_thinking"] = kwargs.get("enable_thinking", False)
        try:
            response = self.openai_client.chat.completions.create(**params)
            response_message = response.choices[0].message.content
        except Exception as e:
            logger.error(f"LLM API call failed: {e}")

        metadata = {
            "prompt": messages,
            "response": response_message,
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "finish_reason": response.choices[0].finish_reason,
        }

        if (
            kwargs.get("response_format", None) is not None
            and kwargs["response_format"].get("type", "") == "json_object"
        ):
            if response_message.startswith("```json\n") and response_message.endswith("```"):
                response_message = response_message[8:-3].strip()  # Remove the ```json\n and ``` at the end

        return response_message, metadata

    def batch_infer(
        self, messages_list: List[List[TextChatMessage]], max_workers: int = 10, **kwargs
    ) -> List[Tuple[List[TextChatMessage], dict, bool]]:
        """
        Run infer on multiple inputs in parallel while preserving cache integrity.

        Args:
            messages_list: A list of message sequences to send.
            max_workers: Number of threads to use.
            **kwargs: Passed through to infer.

        Returns:
            A list of (response, metadata, cache_hit) tuples in the same order.
        """
        results = [None] * len(messages_list)
        assert messages_list, "messages_list must be non-empty"
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {executor.submit(self.infer, msgs, **kwargs): idx for idx, msgs in enumerate(messages_list)}

            with tqdm(total=len(messages_list), desc="batch_infer") as pbar:
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        result = future.result()
                    except Exception:
                        result = None

                    results[idx] = result
                    pbar.update(1)

        return results
