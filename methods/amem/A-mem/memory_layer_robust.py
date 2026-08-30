"""
Robust A-MEM memory layer — drop-in replacement for memory_layer.py.

Key differences from the original:
  - No response_format / JSON schema dependency in LLM calls
  - Plain-text prompts with section-marker parsing (via llm_text_parsers)
  - Structured logging instead of print()
  - Retry wrapper for transient LLM failures
  - Connectivity check on controller init
  - Graceful degradation: evolution failure -> memory stored without evolution
"""

from typing import List, Dict, Optional, Literal, Any, Union
import json
import re
import uuid
import os
import time
import logging
import functools
from datetime import datetime
from abc import ABC, abstractmethod

from memory_layer import SimpleEmbeddingRetriever, simple_tokenize
from utils.llm_client import (
    LLMAPIError,
    ModalModelNotReadyError,
    extract_usage_token_counts,
    get_usage_tracker,
)
from llm_text_parsers import (
    ANALYZE_CONTENT_PROMPT,
    EVOLUTION_DECISION_PROMPT,
    STRENGTHEN_DETAILS_PROMPT,
    UPDATE_NEIGHBORS_PROMPT,
    FOCUSED_KEYWORDS_PROMPT,
    parse_analyze_content,
    parse_evolution_decision,
    parse_strengthen_details,
    parse_update_neighbors,
    validate_analysis_result,
)

logger = logging.getLogger("amem_robust")

# ---------------------------------------------------------------------------
# Retry decorator
# ---------------------------------------------------------------------------

def retry_llm_call(max_retries: int = 2, base_delay: float = 1.0):
    """Decorator: retry an LLM call with exponential backoff."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(max_retries + 1):
                tracker = get_usage_tracker()
                successful_calls_before = tracker.current_successful_calls()
                failure_duration_before = tracker.current_failure_duration()
                attempt_started_at = time.perf_counter()
                tracker.record_attempt()
                try:
                    result = func(*args, **kwargs)
                    if tracker.current_successful_calls() == successful_calls_before:
                        tracker.record_success_without_usage()
                    return result
                except Exception as e:
                    tracker.record_failure()
                    if tracker.current_failure_duration() == failure_duration_before:
                        tracker.record_failure_duration(
                            time.perf_counter() - attempt_started_at
                        )
                    last_exc = e
                    if isinstance(e, ModalModelNotReadyError):
                        break
                    if attempt < max_retries:
                        delay = base_delay * (2 ** attempt)
                        logger.warning(
                            "LLM call %s failed (attempt %d/%d): %s — retrying in %.1fs",
                            func.__name__, attempt + 1, max_retries + 1, e, delay,
                        )
                        tracker.record_retry()
                        sleep_started_at = time.perf_counter()
                        try:
                            time.sleep(delay)
                        finally:
                            tracker.record_failure_duration(
                                time.perf_counter() - sleep_started_at
                            )
            logger.error("LLM call %s failed after %d attempts: %s",
                         func.__name__, max_retries + 1, last_exc)
            try:
                last_exc._retry_exhausted = True
            except Exception:
                pass
            callback = getattr(args[0], "failure_callback", None) if args else None
            if callable(callback):
                callback(func.__name__, last_exc, max_retries + 1)
            raise last_exc
        return wrapper
    return decorator


# ---------------------------------------------------------------------------
# Robust LLM Controllers — no response_format parameter
# ---------------------------------------------------------------------------

class RobustBaseLLMController(ABC):
    """Base class for robust LLM controllers (no JSON schema dependency)."""

    SYSTEM_MESSAGE = "Follow the format specified in the prompt exactly. Do not add extra commentary."
    default_temperature = 0.7
    retry_temperature = 0.3
    connectivity_temperature = 0.0

    def _resolve_temperature(self, temperature: Optional[float]) -> float:
        return self.default_temperature if temperature is None else temperature

    @abstractmethod
    def get_completion(self, prompt: str, temperature: Optional[float] = None) -> str:
        """Get a plain-text completion from the LLM."""
        pass

    def check_connectivity(self):
        """Send a test call to verify the backend is reachable."""
        try:
            response = self.get_completion(
                "Reply with exactly one word: READY",
                temperature=self.connectivity_temperature,
            )
            if not response or not response.strip():
                raise ConnectionError("Empty response from LLM backend")
            logger.info("LLM connectivity check passed (response: %s)", response.strip()[:50])
        except Exception as e:
            raise ConnectionError(
                f"Cannot reach LLM backend: {e}. "
                "Check that the server is running and accessible."
            ) from e


class RobustOpenAIController(RobustBaseLLMController):
    def __init__(
        self,
        model: str = "gpt-4",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        max_tokens: int = 1000,
        usage_tracker: Optional[Any] = None,
        wait_for_model: bool = False,
        provider_routing: Optional[Dict[str, Any]] = None,
        service_tier: Optional[str] = None,
        reasoning_effort: Optional[Union[str, int]] = None,
        reasoning_in_extra_body: bool = False,
    ):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("OpenAI package not found. Install it with: pip install openai")
        self.model = model
        self.max_tokens = max_tokens
        self.usage_tracker = usage_tracker
        self.provider_routing = provider_routing
        self.service_tier = service_tier
        self.reasoning_effort = reasoning_effort
        self.reasoning_in_extra_body = reasoning_in_extra_body
        if api_key is None:
            api_key = os.getenv('OPENAI_API_KEY')
        if api_key is None:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
        self.client = OpenAI(api_key=api_key, base_url=api_base)
        self._readiness_gate = None
        if wait_for_model:
            from utils.llm_client import OpenAIModelReadinessGate
            self._readiness_gate = OpenAIModelReadinessGate(self.client, self.model)

    def _use_max_completion_tokens(self) -> bool:
        """Check if should use max_completion_tokens param (new models)."""
        new_patterns = ["gpt-5", "o1-", "o3-"]
        return any(p in self.model.lower() for p in new_patterns)

    def _record_usage(self, response, latency: float) -> None:
        if self.usage_tracker is None or response.usage is None:
            return
        try:
            from utils.llm_client import LLMResponse
            input_tokens, output_tokens, visible_output_tokens, thinking_tokens = (
                extract_usage_token_counts(response.usage)
            )
            llm_response = LLMResponse(
                content="",
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                visible_output_tokens=visible_output_tokens,
                thinking_tokens=thinking_tokens,
                latency=latency,
                model=self.model,
            )
            self.usage_tracker.record(llm_response)
        except Exception as e:
            logger.debug("Failed to record usage: %s", e)

    @retry_llm_call(max_retries=2)
    def get_completion(self, prompt: str, temperature: Optional[float] = None) -> str:
        if self._readiness_gate is not None:
            self._readiness_gate.wait()
        start_time = time.time()
        params = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.SYSTEM_MESSAGE},
                {"role": "user", "content": prompt}
            ],
            "temperature": self._resolve_temperature(temperature),
        }
        if self._use_max_completion_tokens():
            params["max_completion_tokens"] = self.max_tokens
        else:
            params["max_tokens"] = self.max_tokens
        extra_body = {}
        if self.provider_routing is not None:
            extra_body["provider"] = self.provider_routing
        if self.service_tier is not None:
            extra_body["service_tier"] = self.service_tier
        if self.reasoning_effort is not None and self.reasoning_in_extra_body:
            extra_body["reasoning"] = {"effort": self.reasoning_effort}
        elif self.reasoning_effort is not None:
            params["reasoning_effort"] = self.reasoning_effort
        if extra_body:
            params["extra_body"] = extra_body
        response = self.client.chat.completions.create(**params)
        latency = time.time() - start_time
        self._record_usage(response, latency)
        return response.choices[0].message.content


class RobustGeminiController(RobustBaseLLMController):
    """Route A-Mem's metadata and evolution prompts through Vertex Gemini."""

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        max_tokens: int = 1000,
        provider: str = "gemini",
        temperature: float = 0.7,
        reasoning_effort: Optional[Union[str, int]] = None,
    ):
        from utils.llm_client import create_llm_client

        self._client_factory = create_llm_client
        self._client_kwargs = {
            "provider": provider,
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "reasoning_effort": reasoning_effort,
        }
        # Query-only snapshot restores need retrieval state, not build credentials.
        self.client: Optional[Any] = None

    def _get_client(self) -> Any:
        if self.client is None:
            self.client = self._client_factory(**self._client_kwargs)
        return self.client

    def get_completion(self, prompt: str, temperature: Optional[float] = None) -> str:
        response = self._get_client().chat(
            [
                {"role": "system", "content": self.SYSTEM_MESSAGE},
                {"role": "user", "content": prompt},
            ],
            temperature=self._resolve_temperature(temperature),
        )
        return response.content


class RobustOllamaController(RobustBaseLLMController):
    """Direct Ollama library controller (no LiteLLM proxy)."""

    def __init__(self, model: str = "llama2", max_tokens: int = 1000):
        self.model = model
        self.max_tokens = max_tokens

    @retry_llm_call(max_retries=2)
    def get_completion(self, prompt: str, temperature: Optional[float] = None) -> str:
        try:
            from ollama import chat
        except ImportError:
            raise ImportError("ollama package not found. Install it with: pip install ollama")
        response = chat(
            model=self.model,
            messages=[
                {"role": "system", "content": self.SYSTEM_MESSAGE},
                {"role": "user", "content": prompt}
            ],
            options={
                "temperature": self._resolve_temperature(temperature),
                "num_predict": self.max_tokens,
            },
        )
        return response["message"]["content"]


class RobustSGLangController(RobustBaseLLMController):
    def __init__(self, model: str = "llama2",
                 sglang_host: str = "http://localhost",
                 sglang_port: int = 30000,
                 max_tokens: int = 1000):
        import requests as _requests
        self._requests = _requests
        self.model = model
        self.max_tokens = max_tokens
        self.base_url = f"{sglang_host}:{sglang_port}"

    @retry_llm_call(max_retries=2)
    def get_completion(self, prompt: str, temperature: Optional[float] = None) -> str:
        payload = {
            "text": prompt,
            "sampling_params": {
                "temperature": self._resolve_temperature(temperature),
                "max_new_tokens": self.max_tokens,
            }
        }
        response = self._requests.post(
            f"{self.base_url}/generate",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=60,
        )
        if response.status_code == 200:
            return response.json().get("text", "")
        raise RuntimeError(f"SGLang server returned status {response.status_code}: {response.text}")


class RobustVLLMController(RobustBaseLLMController):
    """Controller for vLLM's OpenAI-compatible API server."""

    def __init__(self, model: str = "llama2",
                 vllm_host: str = "http://localhost",
                 vllm_port: int = 30000,
                 max_tokens: int = 1000):
        import requests as _requests
        self._requests = _requests
        self.model = model
        self.max_tokens = max_tokens
        self.base_url = f"{vllm_host}:{vllm_port}"

    @retry_llm_call(max_retries=2)
    def get_completion(self, prompt: str, temperature: Optional[float] = None) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.SYSTEM_MESSAGE},
                {"role": "user", "content": prompt},
            ],
            "temperature": self._resolve_temperature(temperature),
            "max_tokens": self.max_tokens,
        }
        response = self._requests.post(
            f"{self.base_url}/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=120,
        )
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        raise RuntimeError(f"vLLM server returned status {response.status_code}: {response.text}")


class RobustLiteLLMController(RobustBaseLLMController):
    """LiteLLM controller for universal LLM access (Ollama, SGLang, etc.)."""

    def __init__(self, model: str, api_base: Optional[str] = None,
                 api_key: Optional[str] = None, max_tokens: int = 1000):
        from litellm import completion as _completion
        self._completion = _completion
        self.model = model
        self.api_base = api_base
        self.api_key = api_key or "EMPTY"
        self.max_tokens = max_tokens

    @retry_llm_call(max_retries=2)
    def get_completion(self, prompt: str, temperature: Optional[float] = None) -> str:
        completion_args = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.SYSTEM_MESSAGE},
                {"role": "user", "content": prompt}
            ],
            "temperature": self._resolve_temperature(temperature),
            "max_tokens": self.max_tokens,
        }
        if self.api_base:
            completion_args["api_base"] = self.api_base
        if self.api_key:
            completion_args["api_key"] = self.api_key

        response = self._completion(**completion_args)
        return response.choices[0].message.content


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

class RobustLLMController:
    """Factory that selects the right robust LLM controller."""

    def __init__(self,
                 backend: Literal[
                     "openai", "modal", "gemini", "vertex", "ai_studio",
                     "openrouter", "ollama", "sglang", "vllm"
                 ] = "sglang",
                 model: str = "gpt-4",
                 api_key: Optional[str] = None,
                 api_base: Optional[str] = None,
                 sglang_host: str = "http://localhost",
                 sglang_port: int = 30000,
                 max_tokens: int = 1000,
                 temperature: float = 0.7,
                 retry_temperature: float = 0.3,
                 connectivity_temperature: float = 0.0,
                 check_connection: bool = False,
                 usage_tracker: Optional[Any] = None,
                 provider_routing: Optional[Dict[str, Any]] = None,
                 service_tier: Optional[str] = None,
                 reasoning_effort: Optional[Union[str, int]] = None):
        if backend in {"openai", "openrouter", "modal"}:
            if backend == "modal":
                api_key = (
                    api_key
                    or os.getenv("MODAL_API_KEY")
                    or os.getenv("MODAL_PROXY_TOKEN")
                )
                api_base = api_base or os.getenv("MODAL_BASE_URL")
            elif backend == "openrouter":
                api_key = api_key or os.getenv("OPENROUTER_API_KEY")
                api_base = (
                    api_base
                    or os.getenv("OPENROUTER_BASE_URL")
                    or "https://openrouter.ai/api/v1"
                )
            controller_kwargs = {
                "api_key": api_key,
                "api_base": api_base,
                "max_tokens": max_tokens,
                "usage_tracker": usage_tracker,
                "wait_for_model": backend == "modal",
            }
            if reasoning_effort is not None:
                controller_kwargs["reasoning_effort"] = reasoning_effort
                controller_kwargs["reasoning_in_extra_body"] = backend == "openrouter"
            if provider_routing is not None:
                controller_kwargs["provider_routing"] = provider_routing
            if service_tier is not None:
                controller_kwargs["service_tier"] = service_tier
            self.llm = RobustOpenAIController(model, **controller_kwargs)
        elif backend in {"gemini", "vertex", "ai_studio"}:
            self.llm = RobustGeminiController(
                model, max_tokens=max_tokens, provider=backend, temperature=temperature,
                reasoning_effort=reasoning_effort,
            )
        elif backend == "ollama":
            self.llm = RobustOllamaController(model, max_tokens=max_tokens)
        elif backend == "sglang":
            self.llm = RobustSGLangController(
                model, sglang_host, sglang_port, max_tokens=max_tokens
            )
        elif backend == "vllm":
            self.llm = RobustVLLMController(model, sglang_host, sglang_port, max_tokens=max_tokens)
        else:
            raise ValueError("Unsupported LLM backend")

        self.llm.default_temperature = temperature
        self.llm.retry_temperature = retry_temperature
        self.llm.connectivity_temperature = connectivity_temperature

        if check_connection:
            self.llm.check_connectivity()


# ---------------------------------------------------------------------------
# RobustMemoryNote
# ---------------------------------------------------------------------------

class RobustMemoryNote:
    """Memory note that uses plain-text LLM calls for metadata extraction."""

    # Default max chars for content analysis to avoid context length exceeded
    # Conservative estimate: 200k tokens * 1.5 chars/token = 300k chars
    DEFAULT_MAX_ANALYSIS_CHARS = 300000

    def __init__(self,
                 content: str,
                 id: Optional[str] = None,
                 keywords: Optional[List[str]] = None,
                 links: Optional[Dict] = None,
                 importance_score: Optional[float] = None,
                 retrieval_count: Optional[int] = None,
                 timestamp: Optional[str] = None,
                 last_accessed: Optional[str] = None,
                 context: Optional[str] = None,
                 evolution_history: Optional[List] = None,
                 category: Optional[str] = None,
                 tags: Optional[List[str]] = None,
                 llm_controller: Optional[RobustLLMController] = None,
                 max_analysis_chars: int = None):

        self.content = content

        if llm_controller and any(p is None for p in [keywords, context, category, tags]):
            # Truncate content for analysis to avoid context length exceeded
            max_chars = max_analysis_chars or self.DEFAULT_MAX_ANALYSIS_CHARS
            content_for_analysis = content
            if len(content) > max_chars:
                content_for_analysis = content[:max_chars] + "\n... [truncated for analysis]"
                logger.info("Content truncated for analysis: %d -> %d chars", len(content), max_chars)
            analysis = self.analyze_content(content_for_analysis, llm_controller)
            logger.debug("analysis result: %s", analysis)
            keywords = keywords or analysis["keywords"]
            context = context or analysis["context"]
            tags = tags or analysis["tags"]

        self.id = id or str(uuid.uuid4())
        self.keywords = keywords or []
        self.links = links or []
        self.importance_score = importance_score or 1.0
        self.retrieval_count = retrieval_count or 0
        current_time = datetime.now().strftime("%Y%m%d%H%M")
        self.timestamp = timestamp or current_time
        self.last_accessed = last_accessed or current_time

        self.context = context or "General"
        if isinstance(self.context, list):
            self.context = " ".join(self.context)

        self.evolution_history = evolution_history or []
        self.category = category or "Uncategorized"
        self.tags = tags or []

    @staticmethod
    def analyze_content(content: str, llm_controller: RobustLLMController) -> Dict:
        """Analyze content using plain-text prompt + section-marker parsing."""
        with get_usage_tracker().scope("amem.base.note_analysis"):
            return RobustMemoryNote._analyze_content(content, llm_controller)

    @staticmethod
    def _analyze_content(content: str, llm_controller: RobustLLMController) -> Dict:
        prompt = ANALYZE_CONTENT_PROMPT.format(content=content)
        try:
            response = llm_controller.llm.get_completion(prompt)
            analysis = parse_analyze_content(response, content)

            # If keywords still empty after parsing, try focused retry
            if not analysis["keywords"]:
                logger.info("Keywords empty after initial parse — retrying with focused prompt")
                retry_prompt = FOCUSED_KEYWORDS_PROMPT.format(content=content)
                retry_response = llm_controller.llm.get_completion(
                    retry_prompt, temperature=llm_controller.llm.retry_temperature
                )
                from llm_text_parsers import _parse_list_items
                analysis["keywords"] = _parse_list_items(retry_response)

            # Final validation
            analysis = validate_analysis_result(analysis, content)
            return analysis

        except Exception as e:
            if isinstance(e, LLMAPIError):
                raise
            logger.error("Error analyzing content: %s", e)
            # Graceful degradation: heuristic keywords/context
            from llm_text_parsers import _heuristic_keywords, _heuristic_context
            return {
                "keywords": _heuristic_keywords(content),
                "context": _heuristic_context(content),
                "tags": _heuristic_keywords(content, 3),
            }


# ---------------------------------------------------------------------------
# RobustAgenticMemorySystem
# ---------------------------------------------------------------------------

class RobustAgenticMemorySystem:
    """Memory management system using plain-text LLM calls (no JSON schema)."""

    def __init__(self,
                 model_name: str = 'all-MiniLM-L6-v2',
                 llm_backend: str = "sglang",
                 llm_model: str = "gpt-4o-mini",
                 evo_threshold: int = 100,
                 api_key: Optional[str] = None,
                 api_base: Optional[str] = None,
                 sglang_host: str = "http://localhost",
                 sglang_port: int = 30000,
                 max_tokens: int = 1000,
                 temperature: float = 0.7,
                 retry_temperature: float = 0.3,
                 connectivity_temperature: float = 0.0,
                 max_context_chars: int = 100000,
                 check_connection: bool = False,
                 usage_tracker: Optional[Any] = None,
                 provider_routing: Optional[Dict[str, Any]] = None,
                 service_tier: Optional[str] = None,
                 reasoning_effort: Optional[Union[str, int]] = None):

        self.memories: Dict[str, RobustMemoryNote] = {}
        self.retriever = SimpleEmbeddingRetriever(model_name)
        self.llm_controller = RobustLLMController(
            llm_backend, llm_model, api_key, api_base,
            sglang_host, sglang_port, max_tokens,
            temperature, retry_temperature, connectivity_temperature,
            check_connection,
            usage_tracker=usage_tracker,
            provider_routing=provider_routing,
            service_tier=service_tier,
            reasoning_effort=reasoning_effort,
        )
        self.evo_cnt = 0
        self.evo_threshold = evo_threshold
        self.max_context_chars = max_context_chars

    # ---- public API (mirrors AgenticMemorySystem) ----

    def add_note(self, content: str, time: str = None, **kwargs) -> str:
        """Add a new memory note."""
        note = RobustMemoryNote(
            content=content,
            llm_controller=self.llm_controller,
            timestamp=time,
            max_analysis_chars=self.max_context_chars,
            **kwargs,
        )
        evo_label, note = self.process_memory(note)
        self.memories[note.id] = note
        with get_usage_tracker().scope("amem.embedding.index"):
            self.retriever.add_documents([
                "content:" + note.content +
                " context:" + note.context +
                " keywords: " + ", ".join(note.keywords) +
                " tags: " + ", ".join(note.tags)
            ])
        if evo_label:
            self.evo_cnt += 1
            if self.evo_cnt % self.evo_threshold == 0:
                self.consolidate_memories()
        return note.id

    def consolidate_memories(self):
        """Re-initialize the retriever with current memory state."""
        with get_usage_tracker().scope("amem.embedding.consolidation"):
            self._consolidate_memories()

    def _consolidate_memories(self):
        try:
            model_name = self.retriever.model.get_config_dict()['model_name']
        except (AttributeError, KeyError):
            model_name = 'all-MiniLM-L6-v2'

        self.retriever = SimpleEmbeddingRetriever(model_name)
        for memory in self.memories.values():
            metadata_text = f"{memory.context} {' '.join(memory.keywords)} {' '.join(memory.tags)}"
            self.retriever.add_documents([memory.content + " , " + metadata_text])

    def find_related_memories(self, query: str, k: int = 5) -> tuple:
        """Find related memories using embedding retrieval."""
        if not self.memories:
            return "", []

        indices = self.retriever.search(query, k)
        all_memories = list(self.memories.values())
        memory_str = ""
        for i in indices:
            memory_str += (
                "memory index:" + str(i) +
                "\t talk start time:" + all_memories[i].timestamp +
                "\t memory content: " + all_memories[i].content +
                "\t memory context: " + all_memories[i].context +
                "\t memory keywords: " + str(all_memories[i].keywords) +
                "\t memory tags: " + str(all_memories[i].tags) + "\n"
            )
        return memory_str, indices

    def find_related_memories_raw(self, query: str, k: int = 5) -> str:
        """Find related memories with neighborhood expansion."""
        if not self.memories:
            return ""

        indices = self.retriever.search(query, k)
        all_memories = list(self.memories.values())
        memory_str = ""
        for i in indices:
            j = 0
            memory_str += (
                "talk start time:" + all_memories[i].timestamp +
                "memory content: " + all_memories[i].content +
                "memory context: " + all_memories[i].context +
                "memory keywords: " + str(all_memories[i].keywords) +
                "memory tags: " + str(all_memories[i].tags) + "\n"
            )
            neighborhood = all_memories[i].links
            for neighbor in neighborhood:
                memory_str += (
                    "talk start time:" + all_memories[neighbor].timestamp +
                    "memory content: " + all_memories[neighbor].content +
                    "memory context: " + all_memories[neighbor].context +
                    "memory keywords: " + str(all_memories[neighbor].keywords) +
                    "memory tags: " + str(all_memories[neighbor].tags) + "\n"
                )
                if j >= k:
                    break
                j += 1
        return memory_str

    # ---- evolution (3 sequential plain-text calls) ----

    def _truncate_memory_str(self, memory_str: str) -> str:
        """Truncate memory string to avoid context length exceeded."""
        if len(memory_str) <= self.max_context_chars:
            return memory_str
        return memory_str[:self.max_context_chars] + "\n... [truncated due to length]"

    def process_memory(self, note: RobustMemoryNote) -> tuple:
        """Process a memory note for evolution using plain-text LLM calls.

        Uses up to 3 sequential calls (conditional):
          1. Evolution decision
          2. Strengthen details (skip if no strengthen)
          3. Update neighbors (skip if no update)
        """
        with get_usage_tracker().scope("amem.original_evolution"):
            return self._process_memory(note)

    def _process_memory(self, note: RobustMemoryNote) -> tuple:
        neighbor_memory, indices = self.find_related_memories(note.content, k=5)

        if len(indices) == 0:
            return False, note

        # Truncate neighbor memory to avoid context length exceeded
        neighbor_memory = self._truncate_memory_str(neighbor_memory)

        # Also truncate note content for evolution prompts if too long
        # Reserve space for neighbor_memory and prompt template
        max_note_chars = max(self.max_context_chars // 3, 50000)  # At least 50k chars
        note_content_for_prompt = note.content
        if len(note.content) > max_note_chars:
            note_content_for_prompt = note.content[:max_note_chars] + "\n... [truncated]"
            logger.info("Note content truncated for evolution: %d -> %d chars", len(note.content), max_note_chars)

        try:
            # ---- Call 1: Evolution decision ----
            decision_prompt = EVOLUTION_DECISION_PROMPT.format(
                context=note.context,
                content=note_content_for_prompt,
                keywords=note.keywords,
                nearest_neighbors_memories=neighbor_memory,
            )
            decision_response = self.llm_controller.llm.get_completion(decision_prompt)
            decision = parse_evolution_decision(decision_response)
            logger.debug("Evolution decision: %s", decision)

            if decision["decision"] == "NO_EVOLUTION":
                return False, note

            should_strengthen = decision["decision"] in ("STRENGTHEN", "STRENGTHEN_AND_UPDATE")
            should_update = decision["decision"] in ("UPDATE_NEIGHBOR", "STRENGTHEN_AND_UPDATE")

            # ---- Call 2: Strengthen details (conditional) ----
            if should_strengthen:
                strengthen_prompt = STRENGTHEN_DETAILS_PROMPT.format(
                    content=note_content_for_prompt,
                    keywords=note.keywords,
                    nearest_neighbors_memories=neighbor_memory,
                )
                strengthen_response = self.llm_controller.llm.get_completion(strengthen_prompt)
                strengthen = parse_strengthen_details(strengthen_response)
                logger.debug("Strengthen details: %s", strengthen)

                note.links.extend(strengthen["connections"])
                if strengthen["tags"]:
                    note.tags = strengthen["tags"]

            # ---- Call 3: Update neighbors (conditional) ----
            if should_update:
                update_prompt = UPDATE_NEIGHBORS_PROMPT.format(
                    content=note_content_for_prompt,
                    context=note.context,
                    nearest_neighbors_memories=neighbor_memory,
                    max_neighbor_idx=len(indices) - 1,
                    neighbor_count=len(indices),
                )
                update_response = self.llm_controller.llm.get_completion(update_prompt)
                neighbor_updates = parse_update_neighbors(update_response, len(indices))
                logger.debug("Neighbor updates: %s", neighbor_updates)

                noteslist = list(self.memories.values())
                notes_id = list(self.memories.keys())
                for i in range(min(len(indices), len(neighbor_updates))):
                    upd = neighbor_updates[i]
                    memorytmp_idx = indices[i]
                    if memorytmp_idx >= len(noteslist):
                        continue
                    notetmp = noteslist[memorytmp_idx]
                    if upd["tags"]:
                        notetmp.tags = upd["tags"]
                    if upd["context"]:
                        notetmp.context = upd["context"]
                    self.memories[notes_id[memorytmp_idx]] = notetmp

            return True, note

        except Exception as e:
            if isinstance(e, LLMAPIError):
                raise
            logger.error("Evolution failed for note %s: %s — storing without evolution", note.id, e)
            return False, note
