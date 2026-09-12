"""Active structural primitives: state heads, explicit time axes, provenance, usage."""

from typing import Any, Dict
from .canonicalization import state_identity
from .core import CoreMemoryMixin


class ReadStructuralResolutionMixin:
    @staticmethod
    def _recency_date(memory: Dict[str, Any]) -> str:
        return CoreMemoryMixin._parse_date(
            memory.get("event_time", "")
        ) or CoreMemoryMixin._parse_date(memory.get("document_time", ""))

    def _is_state_head(self, memory: Dict[str, Any]) -> bool:
        """Return whether a state-like memory is in its resolved head set."""
        identity = state_identity(memory)
        return bool(
            identity and memory.get("id") in self._state_heads.get(identity, [])
        )

    @staticmethod
    def _date_for(memory: Dict[str, Any], axis: str = "event_time") -> str:
        if axis == "effective_event_time":
            event = CoreMemoryMixin._parse_date(memory.get("event_time", ""))
            origin = CoreMemoryMixin._parse_date(memory.get("origin_document_time", ""))
            document = CoreMemoryMixin._parse_date(memory.get("document_time", ""))
            # A day-level source date may safely refine a coarse YYYY-MM event
            # only when it falls inside that same month. This is the explicit
            # semantics of effective_event_time, not an event_time fallback.
            if event and event.count("-") == 1:
                refinement = next(
                    (
                        date
                        for date in (origin, document)
                        if len(date) == 10 and date.startswith(event)
                    ),
                    "",
                )
                return refinement or event
            return event or origin or document
        if axis == "origin_document_time":
            return CoreMemoryMixin._parse_date(memory.get("origin_document_time", ""))
        if axis == "document_time":
            return CoreMemoryMixin._parse_date(memory.get("document_time", ""))
        return CoreMemoryMixin._parse_date(memory.get("event_time", ""))

    @staticmethod
    def _date_matches(date: str, constraint: str) -> bool:
        parsed = CoreMemoryMixin._parse_date(date)
        if not parsed:
            return False
        if constraint.startswith("*-"):
            return len(parsed) >= 10 and parsed[4:] == constraint[1:]
        return parsed == constraint or parsed.startswith(constraint)

    def _valid_causal_relation(
        self,
        relation: Dict[str, Any],
        by_id: Dict[str, Dict[str, Any]],
    ) -> bool:
        if relation.get("type") != "CAUSES":
            return False
        source = by_id.get(str(relation.get("source_id") or ""))
        target = by_id.get(str(relation.get("target_id") or ""))
        provenance = set(relation.get("provenance_evidence_ids") or [])
        if (
            not source
            or not target
            or not source.get("evidence_ids")
            or not target.get("evidence_ids")
        ):
            return False
        # A causal edge may only cite evidence attached to one of its
        # endpoints, or a focal turn explicitly marked by the write extractor
        # as stating this causal link. This prevents an unrelated raw turn from
        # laundering temporal or topical association into a usable path.
        allowed = set(source["evidence_ids"]) | set(target["evidence_ids"])
        if provenance and provenance.issubset(allowed):
            return True
        evidence_ids = {str(evidence.get("id") or "") for evidence in self._evidence}
        return bool(
            relation.get("provenance_kind") == "FOCAL_CAUSAL_TURN"
            and provenance
            and provenance.issubset(evidence_ids)
        )

    def _response_usage(self, response: Any, prompt: str) -> Dict[str, Any]:
        input_tokens = int(getattr(response, "input_tokens", 0) or 0)
        output_tokens = int(getattr(response, "output_tokens", 0) or 0)
        if not input_tokens:
            input_tokens = len(self._tokenizer.encode(prompt))
        if not output_tokens:
            output_tokens = len(
                self._tokenizer.encode(str(getattr(response, "content", "")))
            )
        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "latency": float(getattr(response, "latency", 0.0) or 0.0),
        }
