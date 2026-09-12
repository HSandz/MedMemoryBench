"""Domain-neutral rendering of stored evidence."""

from typing import Any, Dict, Iterable, List


class QueryRenderingMixin:
    def _format_answer_memory(self, memory: Dict[str, Any]) -> str:
        """Render evidence without exposing internal ledger identifiers."""
        status = memory.get(
            "_status", self._belief_status.get(memory.get("id"), "active")
        )
        context_marker = (
            " supplementary_context=true"
            if memory.get("_supplementary_context")
            else ""
        )
        return (
            f"- kind={memory.get('kind', 'FACT')} status={status}{context_marker} "
            f"session={memory.get('session_idx', 'UNKNOWN')} "
            f"owner={memory.get('owner_id') or memory.get('subject_id') or memory.get('subject') or 'UNKNOWN'} "
            f"predicate={memory.get('state_key') or memory.get('predicate') or 'UNKNOWN'} "
            f"object={memory.get('object_anchor') or 'NONE'} "
            f"stance={memory.get('stance', 'AFFIRM')} "
            f"event_time={memory.get('event_time') or 'UNKNOWN'} "
            f"document_time={memory.get('document_time') or 'UNKNOWN'} "
            f"origin_document_time={memory.get('origin_document_time') or 'UNKNOWN'} "
            f"effective_event_time={self._date_for(memory, 'effective_event_time') or 'UNKNOWN'} "
            f"value={self._memory_value(memory) or 'NONE'}: "
            f"{memory.get('claim', '')}"
            + (
                f" [verbatim_value={memory['verbatim_value']}]"
                if memory.get("verbatim_value")
                else ""
            )
        )

    def _dereference_evidence(
        self,
        memory_ids: Iterable[str],
        limit: int = 3,
        per_memory_limit: int = 2,
    ) -> List[Dict[str, Any]]:
        ordered_ids: List[str] = []
        for memory_id in memory_ids:
            if memory_id not in ordered_ids:
                ordered_ids.append(memory_id)
        by_memory = {memory["id"]: memory for memory in self._memories}
        by_id, output, seen = {e["id"]: e for e in self._evidence}, [], set()
        per_memory_limit = max(1, int(per_memory_limit))
        for memory_id in ordered_ids:
            memory = by_memory.get(memory_id)
            if not memory:
                continue
            evidence_ids = list(dict.fromkeys(memory.get("evidence_ids") or []))
            for evidence_id in evidence_ids[:per_memory_limit]:
                if evidence_id in by_id and evidence_id not in seen:
                    output.append(self._snapshot(by_id[evidence_id]))
                    seen.add(evidence_id)
                    if len(output) >= limit:
                        return output
        return output
