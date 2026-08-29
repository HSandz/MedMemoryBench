"""Paper-aligned A-Mem adapter with existing provider and batch transports."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .amem_agent import AMemAgent
from .base import AgentResponse, MemoryBuildResult
from utils.llm_client import LLMAPIError, format_messages


logger = logging.getLogger(__name__)


class AMemFixAgent(AMemAgent):
    """A-Mem adapter that restores the released evaluator's logical flow.

    The provider controllers and final-answer client are inherited from
    ``AMemAgent``. This variant changes only memory/query orchestration:
    atomic timestamped notes, query-keyword generation, and linked retrieval.
    """

    METHOD_TYPE = "agentic_memory"
    DEFAULT_RETRIEVE_NUM = 10

    def __init__(
        self,
        *args,
        retrieve_num: int = DEFAULT_RETRIEVE_NUM,
        amem_query_keywords: bool = True,
        amem_expand_links: bool = True,
        **kwargs,
    ):
        super().__init__(*args, retrieve_num=retrieve_num, **kwargs)
        self.amem_query_keywords = amem_query_keywords
        self.amem_expand_links = amem_expand_links

    @staticmethod
    def _speaker_label(item: Dict[str, Any]) -> str:
        speaker = str(item.get("speaker") or "").strip()
        if speaker:
            return speaker

        role = str(item.get("role") or "").lower()
        return {
            "user": "Patient",
            "assistant": "Doctor",
            "system": "System",
        }.get(role, role.title() or "Unknown")

    @classmethod
    def _normalize_memory_items(
        cls,
        memory_items: Sequence[Dict[str, Any]],
        fallback_timestamp: Optional[str],
    ) -> List[Dict[str, str]]:
        notes: List[Dict[str, str]] = []
        for item in memory_items:
            content = str(item.get("content") or item.get("text") or "").strip()
            if not content:
                continue

            caption = str(item.get("blip_caption") or "").strip()
            if caption:
                content = f"{content} Shared image: {caption}"

            speaker = cls._speaker_label(item)
            timestamp = str(item.get("timestamp") or fallback_timestamp or "").strip()
            notes.append({
                "content": f"Speaker {speaker} says: {content}",
                "timestamp": timestamp,
            })
        return notes

    @staticmethod
    def _strip_memorize_wrapper(text: str) -> str:
        markers = ("\n\n[", "\n\nDATE:", "\n<User>")
        starts = [text.find(marker) for marker in markers if text.find(marker) >= 0]
        if starts:
            start = min(starts)
            if text[start:start + 3] == "\n\n[":
                return text[start + 2:].strip()
            if text[start:start + 7] == "\n\nDATE:":
                return text[start + 2:].strip()
            return text[start + 1:].strip()
        return text.strip()

    @classmethod
    def _fallback_atomic_notes(
        cls,
        text: str,
        timestamp: Optional[str],
    ) -> List[Dict[str, str]]:
        """Best-effort turn parsing for callers that provide only formatted text."""
        content = cls._strip_memorize_wrapper(text)
        date_match = re.search(r"^\[([^\]]+)\]|^DATE:\s*(.+)$", content, re.MULTILINE)
        parsed_timestamp = timestamp
        if date_match:
            parsed_timestamp = (date_match.group(1) or date_match.group(2) or timestamp or "").strip()

        turn_pattern = re.compile(
            r"^(Patient|Doctor|User|Assistant|[^:\n]{1,80}):\s*(.+?)(?=\n+(?:Patient|Doctor|User|Assistant|[^:\n]{1,80}):\s|\Z)",
            re.MULTILINE | re.DOTALL,
        )
        items = [
            {"speaker": match.group(1), "content": match.group(2).strip()}
            for match in turn_pattern.finditer(content)
            if match.group(2).strip()
        ]
        if items:
            return cls._normalize_memory_items(items, parsed_timestamp)

        return [{"content": content, "timestamp": parsed_timestamp or ""}] if content else []

    def _atomic_notes(
        self,
        text: str,
        memory_items: Optional[Sequence[Dict[str, Any]]],
        timestamp: Optional[str],
    ) -> List[Dict[str, str]]:
        notes = self._normalize_memory_items(memory_items or [], timestamp)
        if notes:
            return notes
        logger.warning("amem_fix received no structured turns; using text fallback parsing")
        return self._fallback_atomic_notes(text, timestamp)

    def memorize(self, text: str, **kwargs) -> MemoryBuildResult:
        """Create one timestamped A-Mem note per interaction turn."""
        context_id = self._get_context_id()
        memory_system = self._get_memory_system(context_id)
        notes = self._atomic_notes(
            text,
            memory_items=kwargs.get("memory_items"),
            timestamp=kwargs.get("timestamp"),
        )

        note_ids: List[str] = []
        memory_entries: List[Dict[str, Any]] = []
        stored_notes: List[str] = []

        for turn_index, note_data in enumerate(notes):
            content = note_data["content"]
            timestamp = note_data["timestamp"] or None
            parts = self._split_text_into_chunks(content, self.amem_chunk_size_tokens)
            for part_index, part in enumerate(parts):
                note_id = memory_system.add_note(content=part, time=timestamp)
                note_ids.append(str(note_id))
                stored_notes.append(part)
                self._memory_chunks.append(part)
                memory_entries.append({
                    "event": "ADD",
                    "memory": part[:400],
                    "id": str(note_id),
                    "turn_index": turn_index,
                    "part_index": part_index,
                    "timestamp": timestamp,
                })

        self._is_initialized = True
        return MemoryBuildResult(
            success=True,
            method="amem_fix",
            action="add_atomic_notes",
            input_content=text,
            stored_content="\n\n".join(stored_notes),
            memory_entries=memory_entries,
            all_passages=list(memory_entries),
            chunk_count=len(self._memory_chunks),
            extra={
                "context_id": context_id,
                "retrieve_num": self.retrieve_num,
                "turns_received": len(notes),
                "notes_created": len(note_ids),
                "note_ids": note_ids,
                "inserted_count": len(note_ids),
            },
        )

    @staticmethod
    def _parse_keyword_response(response: str) -> str:
        cleaned = response.strip()
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.MULTILINE)
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict) and parsed.get("keywords"):
                return str(parsed["keywords"]).strip()
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

        marker = re.search(r"^KEYWORDS?\s*:\s*(.+)$", cleaned, re.IGNORECASE | re.MULTILINE)
        return marker.group(1).strip() if marker else cleaned

    def _generate_retrieval_query(self, question: str) -> str:
        if not self.amem_query_keywords:
            return question

        prompt = f"""Given the following question, generate several concise retrieval keywords separated by commas.

Question: {question}

Return only a JSON object with a \"keywords\" field.
Example: {{\"keywords\": \"keyword1, keyword2, keyword3\"}}"""
        try:
            # Query rewrites use the query model, not A-Mem's build controller.
            response = self._llm_client.chat(format_messages(prompt))
            retrieval_query = self._parse_keyword_response(response.content)
            return retrieval_query or question
        except Exception as exc:
            if isinstance(exc, LLMAPIError):
                raise
            logger.warning("A-Mem query keyword generation failed; using raw question: %s", exc)
            return question

    @staticmethod
    def _normalize_indices(indices: Any) -> List[int]:
        values = indices.tolist() if hasattr(indices, "tolist") else list(indices)
        normalized: List[int] = []
        for value in values:
            try:
                normalized.append(int(value))
            except (TypeError, ValueError):
                continue
        return normalized

    @staticmethod
    def _linked_index(link: Any, memory_ids: Sequence[str]) -> Optional[int]:
        if isinstance(link, str) and link in memory_ids:
            return memory_ids.index(link)
        try:
            return int(link)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _format_memory(index: int, memory: Any) -> str:
        return (
            f"memory index:{index}\t talk start time:{memory.timestamp}"
            f"\t memory content: {memory.content}"
            f"\t memory context: {memory.context}"
            f"\t memory keywords: {memory.keywords}"
            f"\t memory tags: {memory.tags}"
        )

    def _retrieve_with_links(
        self,
        memory_system: Any,
        retrieval_query: str,
    ) -> Tuple[str, List[int], List[int], List[Dict[str, Any]]]:
        memories = list(memory_system.memories.values())
        if not memories:
            return "", [], [], []

        direct_indices = self._normalize_indices(
            memory_system.retriever.search(retrieval_query, self.retrieve_num)
        )
        memory_ids = list(memory_system.memories.keys())
        selected_indices: List[int] = []
        seen = set()

        def add_index(index: Optional[int]) -> None:
            if index is None or index < 0 or index >= len(memories) or index in seen:
                return
            seen.add(index)
            selected_indices.append(index)

        for index in direct_indices:
            add_index(index)
            if not self.amem_expand_links or index < 0 or index >= len(memories):
                continue
            for link in list(getattr(memories[index], "links", []))[:self.retrieve_num]:
                add_index(self._linked_index(link, memory_ids))

        records = [
            {
                "memory": memories[index].content,
                "context": memories[index].context,
                "keywords": list(memories[index].keywords),
                "tags": list(memories[index].tags),
                "timestamp": memories[index].timestamp,
                "index": index,
                "linked_expansion": index not in direct_indices,
                "type": "amem_fix_retrieval",
            }
            for index in selected_indices
        ]
        memory_str = "\n".join(
            self._format_memory(index, memories[index]) for index in selected_indices
        )
        return memory_str, direct_indices, selected_indices, records

    def prepare_batch_query(
        self,
        question: str,
        system_message: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Run original-style retrieval locally and batch only final generation."""
        context_id = self._get_context_id()
        memory_system = self._get_memory_system(context_id)
        raw_question = str(kwargs.get("raw_question") or question).strip()
        retrieval_query = self._generate_retrieval_query(raw_question)
        memory_str, direct_indices, selected_indices, retrieved_memories = self._retrieve_with_links(
            memory_system,
            retrieval_query,
        )

        system_tokens = self._llm_client.count_tokens(system_message) if system_message else 0
        question_tokens = self._llm_client.count_tokens(question)
        max_memory_tokens = max(
            self.amem_max_context_tokens
            - system_tokens
            - question_tokens
            - self.max_tokens
            - 200,
            0,
        )
        if memory_str and max_memory_tokens > 0:
            memory_str = self._truncate_to_token_limit(memory_str, max_memory_tokens)
        full_question = (
            f"[Retrieved A-Mem Notes]\n{memory_str}\n\n{question}"
            if memory_str.strip()
            else question
        )

        return {
            "messages": format_messages(full_question, system_message),
            "retrieved_count": len(selected_indices),
            "retrieved_memories": retrieved_memories,
            "extra": {
                "method": "amem_fix",
                "context_id": context_id,
                "raw_question": raw_question,
                "retrieval_query": retrieval_query,
                "direct_indices": direct_indices,
                "expanded_indices": selected_indices,
            },
        }

    @staticmethod
    def finalize_batch_query(prepared: Dict[str, Any], content: str) -> AgentResponse:
        return AgentResponse(
            output=content,
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
        response.extra["tokens_used"] = {
            "input": input_tokens,
            "output": output_tokens,
        }

    def get_info(self) -> Dict[str, Any]:
        info = super().get_info()
        info.update({
            "method": "amem_fix",
            "amem_query_keywords": self.amem_query_keywords,
            "amem_expand_links": self.amem_expand_links,
        })
        return info
