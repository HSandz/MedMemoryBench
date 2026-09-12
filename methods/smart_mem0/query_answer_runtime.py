"""Active answer lifecycle; no retrieval or planner authority."""

import time
from typing import Any, Dict, Optional
from methods.base import AgentResponse
from .read_usage_contract import record_read_usage

QUERY_TOKEN_STAGES = (
    "controller",
    "fast_gate",
    "planner",
    "slot_validation",
    "replan",
    "answer",
)


class QueryAnswerRuntimeMixin:
    @staticmethod
    def _total_query_tokens(tokens: Dict[str, Any]) -> int:
        return sum(int(tokens.get(stage, 0) or 0) for stage in QUERY_TOKEN_STAGES)

    def query(
        self, question: str, system_message: Optional[str] = None, **kwargs
    ) -> AgentResponse:
        prepared = self.prepare_batch_query(
            question, system_message=system_message, **kwargs
        )
        return self.generate_prepared_batch_answer(prepared)

    def generate_prepared_batch_answer(self, prepared: Dict[str, Any]) -> AgentResponse:
        """Consume a fresh or restored request without regenerating terminal answers."""
        extra = prepared.setdefault("extra", {})
        terminal = prepared.get("precomputed_answer") not in (None, "")
        extra["precomputed_answer_present"] = terminal
        extra["answer_llm_called"] = False
        extra["direct_generation_violation"] = False
        if terminal:
            result = self.finalize_batch_query(
                prepared, str(prepared["precomputed_answer"])
            )
            result.query_time = (
                prepared["extra"].get("retrieval_elapsed_ms", 0) / 1000.0
            )
            latency = prepared["extra"].setdefault("query_latency", {})
            latency["total_wall"] = round(result.query_time, 3)
            return result
        started = time.time()
        # Query evaluation must be reproducible; write-time creativity is
        # configured separately and is already frozen in the memory snapshot.
        extra["answer_llm_called"] = True
        if "second_call" in extra:
            extra["second_call"]["called"] = True
        response = self._llm_client.chat(
            prepared["messages"], temperature=0.0, max_tokens=1024
        )
        usage = self._response_usage(
            response,
            "\n".join(message.get("content", "") for message in prepared["messages"]),
        )
        answer_latency = round(float(usage.get("latency", 0.0) or 0.0), 3)
        prepared["extra"]["query_tokens"]["answer"] = usage["total_tokens"]
        prepared["extra"]["query_tokens"]["total"] = self._total_query_tokens(
            prepared["extra"]["query_tokens"]
        )
        latency = prepared["extra"].setdefault("query_latency", {})
        latency["answer"] = answer_latency
        result = self.finalize_batch_query(prepared, response.content)
        result.query_time = (
            time.time()
            - started
            + prepared["extra"].get("retrieval_elapsed_ms", 0) / 1000.0
        )
        latency["total_wall"] = round(result.query_time, 3)
        return result

    def finalize_batch_query(
        self, prepared: Dict[str, Any], content: str
    ) -> AgentResponse:
        if prepared.get("precomputed_answer") not in (None, ""):
            content = str(prepared["precomputed_answer"])
        extra = prepared.setdefault("extra", {})
        extra["precomputed_answer_present"] = prepared.get(
            "precomputed_answer"
        ) not in (None, "")
        extra.setdefault("answer_llm_called", False)
        extra["direct_generation_violation"] = bool(
            extra["precomputed_answer_present"] and extra["answer_llm_called"]
        )
        return AgentResponse(
            output=content,
            query_time=0.0,
            retrieved_count=prepared["retrieved_count"],
            retrieved_memories=prepared["retrieved_memories"],
            extra=prepared["extra"],
        )

    @staticmethod
    def record_batch_query_usage(
        response: AgentResponse,
        input_tokens: int,
        output_tokens: int,
    ) -> None:
        tokens = response.extra.setdefault("query_tokens", {})
        answer_tokens = int(input_tokens or 0) + int(output_tokens or 0)
        tokens["answer"] = answer_tokens
        tokens["total"] = QueryAnswerRuntimeMixin._total_query_tokens(tokens)
        response.extra["answer_llm_called"] = bool(
            answer_tokens
        ) or not response.extra.get("precomputed_answer_present")
        if "second_call" in response.extra:
            response.extra["second_call"]["called"] = response.extra[
                "answer_llm_called"
            ]
        response.extra["direct_generation_violation"] = bool(
            response.extra.get("precomputed_answer_present") and answer_tokens
        )
        record_read_usage(response.extra)

    def reset(self) -> None:
        super().reset()
        self._memories, self._evidence, self._relations = [], [], []
        self._atom_dispositions, self._capture_dispositions = [], []
        self._state_spine = {}
        self._subject_postings = {}
        self._object_postings = {}
        self._entity_postings = {}
        self._value_postings = {}
        self._unit_postings = {}
        self._predicate_postings = {}
        self._belief_status, self._state_heads, self._profile_pack = {}, {}, {}
        self._memory_seq = self._evidence_seq = self._session_seq = 0
        self._loaded_frozen = False
        self._bm25 = self._embedding_matrix = None
        self._embedding_cache, self._index_dirty = {}, True
        self._last_write_stats = {
            "skipped_recaps": 0,
            "committed_memories": 0,
            "promoted_state_updates": 0,
            "reused_state_identities": 0,
        }
        if self._write_context is not None:
            self._write_context.clear()
        self._write_context = None
