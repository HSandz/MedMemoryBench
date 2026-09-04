"""Read-only temporal constraint hardening for two-stage SmartMem0.

Month names without a year are common comparison selectors (January vs March)
but the durable memory dates are ISO strings. Represent those selectors as
month wildcards without changing the write schema.
"""

import re

from .contracts import MONTH_NUMBERS, QueryFrame, VALID_TEMPORAL_AXES
from .core import CoreMemoryMixin


class ReadTemporalContractMixin:
    @staticmethod
    def _rc_month_constraint(value):
        text = str(value or "").strip()
        if re.fullmatch(r"\*-(?:0[1-9]|1[0-2])(?:-(?:0[1-9]|[12]\d|3[01]))?", text):
            return text
        match = re.fullmatch(
            r"(?:in\s+)?(" + "|".join(MONTH_NUMBERS) + r")",
            text,
            flags=re.IGNORECASE,
        )
        if match:
            return f"*-{MONTH_NUMBERS[match.group(1).lower()]:02d}"
        return ""

    @classmethod
    def _rc_bare_months(cls, text):
        occupied = []
        source = str(text or "")
        explicit = re.compile(
            r"\b(?:" + "|".join(MONTH_NUMBERS) + r")\s+(?:\d{1,2}(?:st|nd|rd|th)?(?:,?\s+\d{4})?|\d{4})\b",
            re.IGNORECASE,
        )
        occupied.extend(match.span() for match in explicit.finditer(source))
        found = []
        pattern = re.compile(r"\b(" + "|".join(MONTH_NUMBERS) + r")\b", re.IGNORECASE)
        for match in pattern.finditer(source):
            if any(start <= match.start() < end for start, end in occupied):
                continue
            value = f"*-{MONTH_NUMBERS[match.group(1).lower()]:02d}"
            if value not in found:
                found.append(value)
        return found

    def _query_frame(self, question):
        frame = super()._query_frame(question)
        extras = self._rc_bare_months(self._question_stem(question))
        dates = list(frame.dates)
        for value in extras:
            if value not in dates:
                dates.append(value)
        return QueryFrame(
            dates=tuple(dates),
            speaker_role=frame.speaker_role,
            entities=frame.entities,
            hard_entities=frame.hard_entities,
        )

    @staticmethod
    def _date_matches(date, constraint):
        wanted = str(constraint or "")
        parsed = CoreMemoryMixin._parse_date(str(date or ""))
        if not parsed:
            return False
        if re.fullmatch(r"\*-(?:0[1-9]|1[0-2])", wanted):
            return len(parsed) >= 7 and parsed[4:7] == wanted[1:]
        if re.fullmatch(r"\*-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])", wanted):
            return len(parsed) >= 10 and parsed[4:10] == wanted[1:]
        normalized = CoreMemoryMixin._parse_date(wanted)
        return bool(normalized and (parsed == normalized or parsed.startswith(normalized)))

    def _rc_gap_from_req(self, decision, req, fallback_id, frame=None):
        gap = super()._rc_gap_from_req(decision, req, fallback_id, frame=frame)
        month_anchor = self._rc_month_constraint(gap.temporal_anchor)
        if month_anchor:
            gap.temporal_anchor = month_anchor
            gap.temporal_relation = gap.temporal_relation or "EXACT"

        if gap.role == "COMPARAND" and not gap.temporal_anchor:
            months = self._rc_bare_months(gap.target_surface)
            if len(months) == 1:
                gap.temporal_anchor = months[0]
                gap.temporal_relation = "EXACT"
                gap.temporal_axis = gap.temporal_axis or "effective_event_time"

        if gap.temporal_anchor and not gap.temporal_axis:
            gap.temporal_axis = "effective_event_time"
        if gap.temporal_axis:
            gap.required_fields = [gap.temporal_axis]
        return gap

    def _temporal_filter(self, operation, outputs, seeds, frame=QueryFrame()):
        relation = str(operation.get("relation") or "").upper()
        anchor = self._rc_month_constraint(operation.get("anchor"))
        if relation != "EXACT" or not anchor:
            return super()._temporal_filter(operation, outputs, seeds, frame)

        axis = str(operation.get("axis") or "effective_event_time").lower()
        if axis not in VALID_TEMPORAL_AXES:
            return []
        candidate_refs = operation.get("candidate_refs")
        scoped = isinstance(candidate_refs, list) and bool(candidate_refs)
        candidates = self._resolve_refs(candidate_refs, outputs, seeds) if scoped else []
        scoped_ids = {memory["id"] for memory in candidates}
        if scoped and not scoped_ids:
            return []

        ids = [
            memory["id"]
            for memory in self._memories
            if (not scoped or memory["id"] in scoped_ids)
            and self._date_matches(self._date_for(memory, axis), anchor)
            and self._memory_satisfies_frame(
                memory,
                frame,
                include_dates=False,
                include_entities=False,
            )
        ]
        if not ids:
            return []
        query = str(operation.get("query") or "event")
        return self._hybrid_search(query, top_k=min(6, len(ids)), candidate_ids=ids)

    def _slot_covered(self, slot, support_ids, selected, relations):
        if str(slot.get("type") or "").upper() != "TEMPORAL":
            return super()._slot_covered(slot, support_ids, selected, relations)
        relation = str(slot.get("temporal_relation") or slot.get("time_relation") or "LOCATE").upper()
        anchor = self._rc_month_constraint(slot.get("time_anchor"))
        if relation != "EXACT" or not anchor:
            return super()._slot_covered(slot, support_ids, selected, relations)
        axis = str(slot.get("time_axis") or "effective_event_time").lower()
        if axis not in VALID_TEMPORAL_AXES:
            return False
        support = set(support_ids)
        return any(
            memory.get("id") in support
            and self._date_matches(self._date_for(memory, axis), anchor)
            and self._slot_contract_match(slot, memory, True)
            for memory in selected
        )
