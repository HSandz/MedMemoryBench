"""Experimental typed-relation extension for the robust A-MEM memory layer."""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import re
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from memory_layer_robust import RobustAgenticMemorySystem, RobustMemoryNote
from utils.llm_client import get_usage_tracker


logger = logging.getLogger("amem_typed")

RELATION_TYPES = ("SUPPORT", "REFINE", "SUPERSEDE", "CONFLICT", "RELATED")
DEFAULT_EXPANSION_TYPES = frozenset({"SUPPORT", "REFINE", "SUPERSEDE", "CONFLICT"})
RELATION_PRIORITY = {
    "SUPERSEDE": 0,
    "CONFLICT": 1,
    "REFINE": 2,
    "SUPPORT": 3,
    "RELATED": 4,
}

_MONTH_NAMES = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}
_MISSING = object()

TYPED_RELATION_PROMPT = """Classify how a new memory relates to each candidate memory.

Direction is always NEW_MEMORY --RELATION_TYPE--> EXISTING_MEMORY.

Use exactly one of these types when a meaningful relation exists:
- SUPPORT: independently confirms essentially the same information; the existing memory remains valid.
- REFINE: adds detail, precision, qualification, or specificity; the existing memory remains basically valid.
- SUPERSEDE: states a newer replacement state or value. Do not choose this merely because the new memory is later.
- CONFLICT: makes an incompatible claim for the same scope/state and neither claim can safely replace the other.
- RELATED: meaningfully connected, but none of the more specific relations applies.

Important distinctions:
- Similarity alone is not SUPPORT.
- Repeated but separate events can be RELATED.
- SUPERSEDE requires evidence of replacement or a changed current state.
- REFINE preserves the earlier claim; SUPERSEDE replaces its current validity.
- CONFLICT preserves uncertainty; do not resolve it by recency alone.

Return one line per candidate that has a relation:
RELATION|<zero-based candidate position>|<TYPE>|<confidence from 0 to 1>|<short reason>

Omit candidates with no meaningful relation. Return NO_RELATIONS if none apply.

NEW_MEMORY:
{new_memory}

CANDIDATES:
{candidates}
"""


def _optional_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _parse_timestamp(value: Any) -> Optional[datetime]:
    """Parse supplied source timestamps without substituting execution time."""
    text = _optional_text(value)
    if text is None:
        return None

    normalized = text.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
        return parsed.replace(tzinfo=None) if parsed.tzinfo else parsed
    except ValueError:
        pass

    for timestamp_format in (
        "%Y%m%d%H%M%S",
        "%Y%m%d%H%M",
        "%Y%m%d",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y/%m/%d",
        "%Y-%m-%d",
        "%Y-%m",
        "%Y",
    ):
        try:
            return datetime.strptime(text, timestamp_format)
        except ValueError:
            continue
    natural_timestamp = re.sub(r"\s+on\s+", " ", text, flags=re.IGNORECASE)
    for timestamp_format in (
        "%I:%M %p %d %B, %Y",
        "%I:%M %p %d %b, %Y",
        "%d %B, %Y",
        "%d %b, %Y",
    ):
        try:
            return datetime.strptime(natural_timestamp, timestamp_format)
        except ValueError:
            continue
    return None


def _next_month(value: datetime) -> datetime:
    if value.month == 12:
        return datetime(value.year + 1, 1, 1)
    return datetime(value.year, value.month + 1, 1)


def detect_temporal_query(question: str) -> Dict[str, Any]:
    """Classify only the lightweight temporal intent needed for state retrieval."""
    text = str(question or "").strip()
    lowered = text.lower()
    target: Dict[str, Any] = {
        "raw": None,
        "start": None,
        "end": None,
        "year": None,
        "month": None,
        "precision": None,
    }

    iso_match = re.search(r"\b(\d{4})[-/](\d{1,2})(?:[-/](\d{1,2}))?\b", lowered)
    if iso_match:
        year, month = int(iso_match.group(1)), int(iso_match.group(2))
        day = int(iso_match.group(3)) if iso_match.group(3) else None
        try:
            start = datetime(year, month, day or 1)
            end = start + timedelta(days=1) if day else _next_month(start)
            target.update({
                "raw": iso_match.group(0),
                "start": start.isoformat(),
                "end": end.isoformat(),
                "year": year,
                "month": month,
                "precision": "day" if day else "month",
            })
        except ValueError:
            pass
    else:
        month_pattern = "|".join(_MONTH_NAMES)
        month_match = re.search(
            rf"\b({month_pattern})\b(?:\s+(\d{{1,2}})(?:st|nd|rd|th)?(?:,?\s+(\d{{4}}))?|,?\s+(\d{{4}}))?",
            lowered,
        )
        if month_match:
            month = _MONTH_NAMES[month_match.group(1)]
            day = int(month_match.group(2)) if month_match.group(2) else None
            year_text = month_match.group(3) or month_match.group(4)
            year = int(year_text) if year_text else None
            target.update({
                "raw": month_match.group(0),
                "year": year,
                "month": month,
                "precision": "day" if day else "month",
            })
            if year is not None:
                try:
                    start = datetime(year, month, day or 1)
                    end = start + timedelta(days=1) if day else _next_month(start)
                    target["start"] = start.isoformat()
                    target["end"] = end.isoformat()
                except ValueError:
                    pass
        else:
            chinese_match = re.search(
                r"(?:(\d{4})\s*年\s*)?(\d{1,2})\s*月(?:\s*(\d{1,2})\s*[日号])?",
                text,
            )
            if chinese_match:
                year = int(chinese_match.group(1)) if chinese_match.group(1) else None
                month = int(chinese_match.group(2))
                day = int(chinese_match.group(3)) if chinese_match.group(3) else None
                target.update({
                    "raw": chinese_match.group(0),
                    "year": year,
                    "month": month,
                    "precision": "day" if day else "month",
                })
                if year is not None:
                    try:
                        start = datetime(year, month, day or 1)
                        end = start + timedelta(days=1) if day else _next_month(start)
                        target["start"] = start.isoformat()
                        target["end"] = end.isoformat()
                    except ValueError:
                        pass
            else:
                year_match = re.search(r"\b(19\d{2}|20\d{2}|21\d{2})\b", lowered)
                if year_match:
                    year = int(year_match.group(1))
                    target.update({
                        "raw": year_match.group(0),
                        "start": datetime(year, 1, 1).isoformat(),
                        "end": datetime(year + 1, 1, 1).isoformat(),
                        "year": year,
                        "precision": "year",
                    })

    relative_match = re.search(
        r"\b(\d+)\s+(day|week|month|year)s?\s+(ago|before)\b",
        lowered,
    )
    relative_reference = None
    if relative_match and target["raw"] is None:
        relative_reference = {
            "amount": int(relative_match.group(1)),
            "unit": relative_match.group(2),
            "direction": "past",
            "raw": relative_match.group(0),
        }
        target.update({
            "raw": relative_match.group(0),
            "precision": "relative",
        })

    change_pattern = re.compile(
        r"\b(?:what changed|how (?:has|did|does).{0,80}\bchange|changes?|changed|"
        r"transition(?:ed)?|before and after|from .{1,80} to)\b"
    )
    historical_pattern = re.compile(
        r"\b(?:before|previously|previous|formerly|earlier|prior|past|used to|at the time)\b"
    )
    current_pattern = re.compile(
        r"\b(?:now|currently|current|latest|today|presently|at present|still)\b"
    )

    if change_pattern.search(lowered) or re.search(r"变化|改变|变更|前后", text):
        intent = "change"
    elif target["raw"] is not None:
        intent = "time_specific"
    elif historical_pattern.search(lowered) or re.search(r"之前|以前|过去|曾经|原来|此前", text):
        intent = "historical"
    elif current_pattern.search(lowered) or re.search(r"现在|目前|如今|当前|最近", text):
        intent = "current"
    else:
        intent = "none"

    return {
        "intent": intent,
        "target": target,
        "relative_reference": relative_reference,
    }


def _clamp_confidence(value: Any) -> float:
    try:
        text = str(value).strip()
        is_percent = text.endswith("%")
        number = float(text.rstrip("%"))
        if is_percent:
            number /= 100.0
        return max(0.0, min(1.0, number))
    except (TypeError, ValueError):
        return 0.5


def _normalize_prediction(item: Any, candidate_count: int) -> Optional[Dict[str, Any]]:
    if not isinstance(item, dict):
        return None

    relation_type = str(
        item.get("relation_type") or item.get("relation") or item.get("type") or ""
    ).strip().upper()
    if relation_type in {"", "NONE", "NO_RELATION", "NO_RELATIONS"}:
        return None
    if relation_type not in RELATION_TYPES:
        return None

    position = item.get("candidate_position", item.get("position", item.get("index")))
    try:
        position = int(position)
    except (TypeError, ValueError):
        return None
    if position < 0 or position >= candidate_count:
        return None

    return {
        "candidate_position": position,
        "relation_type": relation_type,
        "confidence": _clamp_confidence(item.get("confidence", 0.5)),
        "reason": str(item.get("reason") or item.get("reasoning") or item.get("explanation") or "").strip(),
    }


def _json_predictions(text: str) -> Optional[List[Any]]:
    candidates = [text]
    if "[" in text and "]" in text:
        candidates.append(text[text.find("["):text.rfind("]") + 1])
    if "{" in text and "}" in text:
        candidates.append(text[text.find("{"):text.rfind("}") + 1])

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict):
            relations = parsed.get("relations") or parsed.get("typed_relations")
            if isinstance(relations, list):
                return relations
            return [parsed]
    return None


def parse_typed_relations(response: Any, candidate_count: int) -> List[Dict[str, Any]]:
    """Parse provider-agnostic JSON or line-oriented typed-relation output."""
    if candidate_count <= 0 or response is None:
        return []

    text = str(response).strip()
    text = re.sub(r"^```(?:json|text)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
    if not text or re.fullmatch(r"NO[_ ]RELATIONS?\.?", text, re.IGNORECASE):
        return []

    raw_items = _json_predictions(text)
    if raw_items is None:
        raw_items = []
        line_pattern = re.compile(
            r"^(?:RELATION\s*\|\s*)?(\d+)\s*\|\s*"
            r"(SUPPORT|REFINE|SUPERSEDE|CONFLICT|RELATED|NO_RELATION|NONE)"
            r"(?:\s*\|\s*([^|]+))?(?:\s*\|\s*(.*))?$",
            re.IGNORECASE,
        )
        compact_pattern = re.compile(
            r"^CANDIDATE\s*(\d+)\s*:\s*"
            r"(SUPPORT|REFINE|SUPERSEDE|CONFLICT|RELATED|NO_RELATION|NONE)"
            r"(?:\s*\(([\d.]+%?)\))?(?:\s*[-:]\s*(.*))?$",
            re.IGNORECASE,
        )
        for line in text.splitlines():
            cleaned = line.strip().lstrip("-* ")
            match = line_pattern.match(cleaned) or compact_pattern.match(cleaned)
            if not match:
                continue
            raw_items.append({
                "candidate_position": match.group(1),
                "relation_type": match.group(2),
                "confidence": match.group(3) or 0.5,
                "reason": match.group(4) or "",
            })

        marker_pattern = re.compile(
            r"CANDIDATE(?:_POSITION)?\s*:\s*(\d+).*?"
            r"(?:RELATION(?:_TYPE)?|TYPE)\s*:\s*"
            r"(SUPPORT|REFINE|SUPERSEDE|CONFLICT|RELATED|NO_RELATION|NONE).*?"
            r"CONFIDENCE\s*:\s*([\d.]+%?)"
            r"(?:.*?(?:REASON|REASONING|EXPLANATION)\s*:\s*(.*?))?"
            r"(?=\n\s*CANDIDATE(?:_POSITION)?\s*:|\Z)",
            re.IGNORECASE | re.DOTALL,
        )
        for match in marker_pattern.finditer(text):
            raw_items.append({
                "candidate_position": match.group(1),
                "relation_type": match.group(2),
                "confidence": match.group(3),
                "reason": (match.group(4) or "").strip(),
            })

    by_position: Dict[int, Dict[str, Any]] = {}
    for item in raw_items:
        prediction = _normalize_prediction(item, candidate_count)
        if prediction is None:
            continue
        position = prediction["candidate_position"]
        previous = by_position.get(position)
        if previous is None or prediction["confidence"] > previous["confidence"]:
            by_position[position] = prediction
    return [by_position[position] for position in sorted(by_position)]


def map_relation_predictions(
    source_id: str,
    candidate_ids: Sequence[str],
    predictions: Iterable[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Convert prompt positions to stable directed memory-ID edges."""
    edges: List[Dict[str, Any]] = []
    seen_pairs = set()
    for prediction in predictions:
        normalized = _normalize_prediction(prediction, len(candidate_ids))
        if normalized is None:
            continue
        target_id = str(candidate_ids[normalized["candidate_position"]])
        if not target_id or target_id == source_id or target_id in seen_pairs:
            continue
        seen_pairs.add(target_id)
        edges.append({
            "source_id": str(source_id),
            "target_id": target_id,
            "relation_type": normalized["relation_type"],
            "direction": "NEW_MEMORY_TO_EXISTING_MEMORY",
            "confidence": normalized["confidence"],
            "reason": normalized["reason"],
            "candidate_position": normalized["candidate_position"],
        })
    return edges


class TypedRelationMemorySystem(RobustAgenticMemorySystem):
    """A-MEM with optional typed relations, temporal state, and provenance."""

    def __init__(
        self,
        *args,
        original_evolution_enabled: bool = True,
        typed_relations_enabled: bool = True,
        temporal_state_enabled: bool = False,
        provenance_enabled: bool = False,
        relation_candidate_count: int = 5,
        relation_temperature: float = 0.2,
        temporal_min_confidence: float = 0.5,
        **kwargs,
    ):
        if temporal_state_enabled and not typed_relations_enabled:
            raise ValueError(
                "amem_temporal_state requires amem_typed_relations=true because "
                "state transitions are derived from typed relations"
            )
        super().__init__(*args, **kwargs)
        self.original_evolution_enabled = bool(original_evolution_enabled)
        self.typed_relations_enabled = bool(typed_relations_enabled)
        self.temporal_state_enabled = bool(temporal_state_enabled)
        self.provenance_enabled = bool(provenance_enabled)
        self.relation_candidate_count = max(0, int(relation_candidate_count))
        self.relation_temperature = float(relation_temperature)
        self.temporal_min_confidence = _clamp_confidence(temporal_min_confidence)
        self.typed_relations: List[Dict[str, Any]] = []
        self._typed_edge_keys = set()
        self.relation_audit: List[Dict[str, Any]] = []
        self._relation_audit_by_memory: Dict[str, Dict[str, Any]] = {}
        self.temporal_audit: List[Dict[str, Any]] = []
        self._temporal_audit_by_memory: Dict[str, List[Dict[str, Any]]] = {}
        self.evidence_store: Dict[str, Dict[str, Any]] = {}
        self.provenance_audit: List[Dict[str, Any]] = []
        self._provenance_audit_by_memory: Dict[str, Dict[str, Any]] = {}

    @staticmethod
    def _append_unique(values: List[str], value: str) -> None:
        if value and value not in values:
            values.append(value)

    @staticmethod
    def _json_evidence_value(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                str(key): TypedRelationMemorySystem._json_evidence_value(item)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple, set)):
            return [TypedRelationMemorySystem._json_evidence_value(item) for item in value]
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        return str(value)

    @staticmethod
    def _initial_temporal_state(source_timestamp: Any) -> Dict[str, Any]:
        timestamp = _optional_text(source_timestamp)
        return {
            "status": "current",
            "historical": False,
            "uncertainty": "certain",
            "valid_from": timestamp,
            "valid_until": None,
            "superseded_by": [],
            "supersedes": [],
            "conflicts_with": [],
            "refined_by": [],
            "refines": [],
            "transition_history": [],
        }

    def _ensure_temporal_state(
        self,
        memory: RobustMemoryNote,
        source_timestamp: Any = _MISSING,
    ) -> Dict[str, Any]:
        state = getattr(memory, "temporal_state", None)
        if not isinstance(state, dict):
            if source_timestamp is _MISSING:
                source_timestamp = getattr(memory, "source_timestamp", None)
            memory.temporal_state = self._initial_temporal_state(source_timestamp)
            return memory.temporal_state

        defaults = self._initial_temporal_state(
            getattr(memory, "source_timestamp", None)
            if source_timestamp is _MISSING else source_timestamp
        )
        for field_name, default_value in defaults.items():
            if field_name not in state:
                state[field_name] = copy.deepcopy(default_value)
        for list_field in (
            "superseded_by",
            "supersedes",
            "conflicts_with",
            "refined_by",
            "refines",
            "transition_history",
        ):
            if not isinstance(state.get(list_field), list):
                state[list_field] = []
        return memory.temporal_state

    def get_temporal_state(self, memory_id: str) -> Dict[str, Any]:
        memory = self.memories.get(str(memory_id))
        if memory is None or not hasattr(memory, "temporal_state"):
            return {}
        return copy.deepcopy(memory.temporal_state)

    def get_temporal_audit(self, memory_id: str) -> Dict[str, Any]:
        memory_id = str(memory_id)
        return {
            "memory_id": memory_id,
            "state": self.get_temporal_state(memory_id),
            "transitions": copy.deepcopy(
                getattr(self, "_temporal_audit_by_memory", {}).get(memory_id, [])
            ),
        }

    def _record_temporal_transition(
        self,
        memory_ids: Sequence[str],
        transition: Dict[str, Any],
    ) -> None:
        stored = copy.deepcopy(transition)
        self.temporal_audit.append(stored)
        for memory_id in dict.fromkeys(str(item) for item in memory_ids):
            self._temporal_audit_by_memory.setdefault(memory_id, []).append(
                copy.deepcopy(stored)
            )

    @staticmethod
    def _state_summary(state: Dict[str, Any]) -> Dict[str, Any]:
        return {
            key: copy.deepcopy(value)
            for key, value in state.items()
            if key != "transition_history"
        }

    def _apply_temporal_relation(self, edge: Dict[str, Any]) -> None:
        if not getattr(self, "temporal_state_enabled", False):
            return
        with get_usage_tracker().scope("amem.temporal_state.transition"):
            self._apply_temporal_relation_enabled(edge)

    def _apply_temporal_relation_enabled(self, edge: Dict[str, Any]) -> None:

        source_id = str(edge["source_id"])
        target_id = str(edge["target_id"])
        source = self.memories[source_id]
        target = self.memories[target_id]
        source_state = self._ensure_temporal_state(source)
        target_state = self._ensure_temporal_state(target)
        relation_type = edge["relation_type"]
        transition = {
            "source_id": source_id,
            "target_id": target_id,
            "relation_type": relation_type,
            "confidence": _clamp_confidence(edge.get("confidence")),
            "reason": str(edge.get("reason") or ""),
            "applied": True,
            "error": "",
        }

        if transition["confidence"] < getattr(self, "temporal_min_confidence", 0.5):
            transition.update({
                "applied": False,
                "error": "relation confidence below temporal state threshold",
                "source_state_after": self._state_summary(source_state),
                "target_state_after": self._state_summary(target_state),
            })
            self._record_temporal_transition((source_id, target_id), transition)
            return

        if relation_type == "SUPERSEDE":
            source_timestamp = source_state.get("valid_from")
            target_timestamp = target_state.get("valid_from")
            source_time = _parse_timestamp(source_timestamp)
            target_time = _parse_timestamp(target_timestamp)
            if source_time is not None and target_time is not None and source_time < target_time:
                transition.update({
                    "applied": False,
                    "error": "superseding memory has an earlier source timestamp",
                    "source_state_after": self._state_summary(source_state),
                    "target_state_after": self._state_summary(target_state),
                })
                self._record_temporal_transition((source_id, target_id), transition)
                logger.warning(
                    "Skipped invalid temporal transition %s SUPERSEDE %s: %s < %s",
                    source_id,
                    target_id,
                    source_timestamp,
                    target_timestamp,
                )
                return

            previous_superseders = list(target_state["superseded_by"])
            self._append_unique(source_state["supersedes"], target_id)
            self._append_unique(target_state["superseded_by"], source_id)
            target_state["status"] = "superseded"
            target_state["historical"] = True
            if source_time is not None and target_time is not None and source_time > target_time:
                target_state["valid_until"] = source_timestamp
            if previous_superseders and source_id not in previous_superseders:
                target_state["uncertainty"] = "conflicting"
                transition["error"] = "multiple memories supersede the same state"
                for previous_id in previous_superseders:
                    previous = self.memories.get(previous_id)
                    if previous is None:
                        continue
                    previous_state = self._ensure_temporal_state(previous)
                    previous_state["uncertainty"] = "conflicting"
                    source_state["uncertainty"] = "conflicting"
                    self._append_unique(source_state["conflicts_with"], previous_id)
                    self._append_unique(previous_state["conflicts_with"], source_id)

        elif relation_type == "REFINE":
            self._append_unique(source_state["refines"], target_id)
            self._append_unique(target_state["refined_by"], source_id)

        elif relation_type == "CONFLICT":
            self._append_unique(source_state["conflicts_with"], target_id)
            self._append_unique(target_state["conflicts_with"], source_id)
            source_state["uncertainty"] = "conflicting"
            target_state["uncertainty"] = "conflicting"

        else:
            return

        transition["source_state_after"] = self._state_summary(source_state)
        transition["target_state_after"] = self._state_summary(target_state)
        source_state["transition_history"].append(copy.deepcopy(transition))
        target_state["transition_history"].append(copy.deepcopy(transition))
        self._record_temporal_transition((source_id, target_id), transition)

    def register_evidence(self, source_evidence: Dict[str, Any]) -> str:
        """Store one immutable, content-addressed source evidence record."""
        if not isinstance(source_evidence, dict):
            raise ValueError("source evidence must be a dictionary")
        record = self._json_evidence_value(copy.deepcopy(source_evidence))
        record["raw_text"] = str(record.get("raw_text") or "")
        for optional_field in (
            "source_context_id",
            "source_session_id",
            "source_session_index",
            "source_event_id",
            "source_turn_id",
            "source_timestamp",
            "speaker",
            "role",
            "blip_caption",
            "source_text_scope",
        ):
            record.setdefault(optional_field, None)
        canonical = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        evidence_id = "ev_" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        existing = self.evidence_store.get(evidence_id)
        stored_record = {"evidence_id": evidence_id, **record}
        if existing is not None and existing != stored_record:
            raise ValueError("evidence hash collision")
        if existing is None:
            self.evidence_store[evidence_id] = copy.deepcopy(stored_record)
        return evidence_id

    def get_evidence(self, evidence_id: str) -> Dict[str, Any]:
        return copy.deepcopy(getattr(self, "evidence_store", {}).get(str(evidence_id), {}))

    def evidence_for_memory(self, memory_id: str) -> List[Dict[str, Any]]:
        memory = self.memories.get(str(memory_id))
        provenance = getattr(memory, "provenance", {}) if memory is not None else {}
        records = []
        for evidence_id in provenance.get("evidence_ids", []):
            evidence = self.get_evidence(evidence_id)
            if evidence:
                records.append(evidence)
        return records

    def get_provenance_audit(self, memory_id: str) -> Dict[str, Any]:
        memory_id = str(memory_id)
        audit = copy.deepcopy(
            getattr(self, "_provenance_audit_by_memory", {}).get(memory_id, {})
        )
        if not audit:
            return {"memory_id": memory_id, "evidence_ids": [], "error": ""}
        return audit

    def _attach_provenance(
        self,
        note: RobustMemoryNote,
        source_evidence: Optional[Dict[str, Any]],
        part_index: Optional[int],
    ) -> None:
        if not getattr(self, "provenance_enabled", False):
            return
        with get_usage_tracker().scope("amem.provenance.attach"):
            self._attach_provenance_enabled(note, source_evidence, part_index)

    def _attach_provenance_enabled(
        self,
        note: RobustMemoryNote,
        source_evidence: Optional[Dict[str, Any]],
        part_index: Optional[int],
    ) -> None:

        evidence = copy.deepcopy(source_evidence or {})
        evidence.setdefault("raw_text", note.content)
        error = ""
        evidence_ids: List[str] = []
        try:
            evidence_ids.append(self.register_evidence(evidence))
        except Exception as exc:
            error = str(exc)
            logger.warning(
                "Provenance storage failed for memory %s; keeping memory without evidence: %s",
                note.id,
                exc,
            )

        note.provenance = {
            "memory_id": str(note.id),
            "evidence_ids": list(evidence_ids),
            "part_index": part_index,
            "error": error,
        }
        audit = {
            "memory_id": str(note.id),
            "evidence_ids": list(evidence_ids),
            "source_timestamp": _optional_text(evidence.get("source_timestamp")),
            "source_session_id": evidence.get("source_session_id"),
            "source_turn_id": evidence.get("source_turn_id"),
            "part_index": part_index,
            "error": error,
        }
        self.provenance_audit.append(copy.deepcopy(audit))
        self._provenance_audit_by_memory[str(note.id)] = copy.deepcopy(audit)

    @staticmethod
    def _normalize_indices(indices: Any) -> List[int]:
        values = indices.tolist() if hasattr(indices, "tolist") else list(indices)
        normalized = []
        for value in values:
            try:
                normalized.append(int(value))
            except (TypeError, ValueError):
                continue
        return normalized

    def _candidate_ids(self, content: str) -> List[str]:
        if not self.memories or self.relation_candidate_count <= 0:
            return []
        with get_usage_tracker().scope("amem.typed_relations.candidate_search"):
            return self._candidate_ids_enabled(content)

    def _candidate_ids_enabled(self, content: str) -> List[str]:
        memory_ids = list(self.memories.keys())
        result = []
        for index in self._normalize_indices(
            self.retriever.search(content, self.relation_candidate_count)
        ):
            if 0 <= index < len(memory_ids) and memory_ids[index] not in result:
                result.append(memory_ids[index])
        return result

    def _infer_relations(
        self,
        note: RobustMemoryNote,
        candidate_ids: Sequence[str],
    ) -> Tuple[List[Dict[str, Any]], str]:
        if not candidate_ids:
            return [], ""
        with get_usage_tracker().scope("amem.typed_relations.inference"):
            return self._infer_relations_enabled(note, candidate_ids)

    def _infer_relations_enabled(
        self,
        note: RobustMemoryNote,
        candidate_ids: Sequence[str],
    ) -> Tuple[List[Dict[str, Any]], str]:
        prompt_budget = max(int(self.max_context_chars), 4000)
        new_memory_budget = max(prompt_budget // 3, 1000)
        candidate_budget = max(prompt_budget - new_memory_budget - 2000, 1000)
        per_candidate_budget = max(candidate_budget // len(candidate_ids), 500)
        candidate_lines = []
        for position, memory_id in enumerate(candidate_ids):
            memory = self.memories.get(memory_id)
            if memory is None:
                continue
            candidate_content = memory.content[:per_candidate_budget]
            candidate_lines.append(
                f"[{position}] id={memory_id}\n"
                f"timestamp={memory.timestamp}\ncontent={candidate_content}\n"
                f"context={memory.context}"
            )
        if not candidate_lines:
            return [], ""

        prompt = TYPED_RELATION_PROMPT.format(
            new_memory=note.content[:new_memory_budget],
            candidates="\n\n".join(candidate_lines),
        )
        response = self.llm_controller.llm.get_completion(
            prompt, temperature=getattr(self, "relation_temperature", 0.2)
        )
        predictions = parse_typed_relations(response, len(candidate_ids))
        return map_relation_predictions(note.id, candidate_ids, predictions), str(response)

    @staticmethod
    def _ensure_relation_lists(memory: RobustMemoryNote) -> None:
        if not hasattr(memory, "typed_relations_out"):
            memory.typed_relations_out = []
        if not hasattr(memory, "typed_relations_in"):
            memory.typed_relations_in = []

    def _store_edge(self, edge: Dict[str, Any]) -> bool:
        with get_usage_tracker().scope("amem.typed_relations.store"):
            source_id = edge.get("source_id")
            target_id = edge.get("target_id")
            relation_type = edge.get("relation_type")
            if (
                source_id == target_id
                or source_id not in self.memories
                or target_id not in self.memories
                or relation_type not in RELATION_TYPES
            ):
                return False
            edge_key = (source_id, target_id)
            if edge_key in self._typed_edge_keys:
                return False

            stored = dict(edge)
            stored["confidence"] = _clamp_confidence(stored.get("confidence"))
            self._ensure_relation_lists(self.memories[source_id])
            self._ensure_relation_lists(self.memories[target_id])
            self.memories[source_id].typed_relations_out.append(dict(stored))
            self.memories[target_id].typed_relations_in.append(dict(stored))
            self.typed_relations.append(stored)
            self._typed_edge_keys.add(edge_key)
        try:
            self._apply_temporal_relation(stored)
        except Exception as exc:
            transition = {
                "source_id": source_id,
                "target_id": target_id,
                "relation_type": relation_type,
                "confidence": stored["confidence"],
                "reason": str(stored.get("reason") or ""),
                "applied": False,
                "error": str(exc),
            }
            if getattr(self, "temporal_state_enabled", False):
                self._record_temporal_transition((source_id, target_id), transition)
            logger.warning(
                "Temporal state update failed for %s %s %s; edge remains stored: %s",
                source_id,
                relation_type,
                target_id,
                exc,
            )
        return True

    def add_typed_relation(self, edge: Dict[str, Any]) -> bool:
        """Validate and add an edge; exposed for deterministic tests/imports."""
        return self._store_edge(edge)

    def add_note(self, content: str, time: str = None, **kwargs) -> str:
        experimental_metadata_enabled = (
            getattr(self, "temporal_state_enabled", False)
            or getattr(self, "provenance_enabled", False)
        )
        if (
            self.original_evolution_enabled
            and not self.typed_relations_enabled
            and not experimental_metadata_enabled
        ):
            return super().add_note(content=content, time=time, **kwargs)

        source_evidence = kwargs.pop("source_evidence", None)
        source_timestamp = kwargs.pop("source_timestamp", time)
        provenance_part_index = kwargs.pop("provenance_part_index", None)

        note = RobustMemoryNote(
            content=content,
            llm_controller=self.llm_controller,
            timestamp=time,
            max_analysis_chars=self.max_context_chars,
            **kwargs,
        )
        if experimental_metadata_enabled:
            note.source_timestamp = _optional_text(source_timestamp)
            # RobustMemoryNote otherwise substitutes execution time for missing input.
            note.timestamp = note.source_timestamp or ""
        if getattr(self, "temporal_state_enabled", False):
            with get_usage_tracker().scope("amem.temporal_state.initialize"):
                note.temporal_state = self._initial_temporal_state(source_timestamp)
        self._attach_provenance(note, source_evidence, provenance_part_index)
        candidate_ids = self._candidate_ids(note.content) if self.typed_relations_enabled else []
        if self.original_evolution_enabled:
            evo_label, note = self.process_memory(note)
        else:
            # Typed-only mode keeps the initial note metadata and all older notes unchanged.
            evo_label = False

        inferred_edges: List[Dict[str, Any]] = []
        raw_response = ""
        inference_error = ""
        if self.typed_relations_enabled:
            try:
                inferred_edges, raw_response = self._infer_relations(note, candidate_ids)
            except Exception as exc:
                inference_error = str(exc)
                logger.warning(
                    "Typed relation inference failed for memory %s; "
                    "storing without typed relations: %s",
                    note.id,
                    exc,
                )

        self._ensure_relation_lists(note)
        self.memories[note.id] = note
        stored_edges = [edge for edge in inferred_edges if self._store_edge(edge)]
        with get_usage_tracker().scope("amem.embedding.index"):
            self.retriever.add_documents([
                "content:" + note.content
                + " context:" + note.context
                + " keywords: " + ", ".join(note.keywords)
                + " tags: " + ", ".join(note.tags)
            ])
        if evo_label:
            self.evo_cnt += 1
            if self.evo_cnt % self.evo_threshold == 0:
                self.consolidate_memories()

        candidate_records = [
            {
                "candidate_position": position,
                "memory_id": memory_id,
                "content": self.memories[memory_id].content[:400],
            }
            for position, memory_id in enumerate(candidate_ids)
            if memory_id in self.memories
        ]
        if self.typed_relations_enabled:
            audit = {
                "created_memory_id": note.id,
                "candidate_neighbors": candidate_records,
                "predicted_relations": stored_edges,
                "relation_inference_error": inference_error,
                "raw_relation_response": raw_response,
            }
            self.relation_audit.append(audit)
            self._relation_audit_by_memory[note.id] = audit
            logger.info(
                "Typed relation construction memory=%s candidates=%s relations=%s",
                note.id,
                candidate_ids,
                [
                    (
                        edge["source_id"],
                        edge["relation_type"],
                        edge["target_id"],
                        edge["confidence"],
                    )
                    for edge in stored_edges
                ],
            )
        return note.id

    def get_relation_audit(self, memory_id: str) -> Dict[str, Any]:
        return dict(self._relation_audit_by_memory.get(str(memory_id), {}))

    def _memory_edges(self, memory_id: str) -> List[Dict[str, Any]]:
        memory = self.memories.get(memory_id)
        if memory is None:
            return []
        self._ensure_relation_lists(memory)
        edges = list(memory.typed_relations_out) + list(memory.typed_relations_in)
        return sorted(
            edges,
            key=lambda edge: (
                RELATION_PRIORITY.get(edge["relation_type"], 99),
                -float(edge.get("confidence", 0.0)),
                edge["source_id"],
                edge["target_id"],
            ),
        )

    def expand_typed_relations(
        self,
        seed_ids: Sequence[str],
        expansion_budget: int = 5,
        include_related: bool = False,
        min_confidence: float = 0.5,
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Bounded bidirectional traversal from semantic seed memories."""
        selected_ids = [memory_id for memory_id in seed_ids if memory_id in self.memories]
        selected_ids = list(dict.fromkeys(selected_ids))
        seen = set(selected_ids)
        queue = list(selected_ids)
        expansions: List[Dict[str, Any]] = []
        budget = max(0, int(expansion_budget))
        threshold = _clamp_confidence(min_confidence)
        allowed_types = set(DEFAULT_EXPANSION_TYPES)
        if include_related:
            allowed_types.add("RELATED")

        while queue and len(expansions) < budget:
            current_id = queue.pop(0)
            for edge in self._memory_edges(current_id):
                if edge["relation_type"] not in allowed_types:
                    continue
                if float(edge.get("confidence", 0.0)) < threshold:
                    continue
                if edge["source_id"] == current_id:
                    neighbor_id = edge["target_id"]
                    traversal_direction = "outgoing"
                else:
                    neighbor_id = edge["source_id"]
                    traversal_direction = "incoming"
                if neighbor_id not in self.memories or neighbor_id in seen:
                    continue
                seen.add(neighbor_id)
                selected_ids.append(neighbor_id)
                queue.append(neighbor_id)
                expansions.append({
                    "added_memory_id": neighbor_id,
                    "from_memory_id": current_id,
                    "traversal_direction": traversal_direction,
                    "relation": dict(edge),
                })
                if len(expansions) >= budget:
                    break
        return selected_ids, expansions

    def retrieve_with_typed_relations(
        self,
        query: str,
        k: int = 5,
        expansion_budget: int = 5,
        include_related: bool = False,
        min_confidence: float = 0.5,
    ) -> Dict[str, Any]:
        memory_ids = list(self.memories.keys())
        seed_indices = []
        if memory_ids:
            for index in self._normalize_indices(self.retriever.search(query, k)):
                if 0 <= index < len(memory_ids) and index not in seed_indices:
                    seed_indices.append(index)
        seed_ids = [memory_ids[index] for index in seed_indices]
        final_ids, expansions = self.expand_typed_relations(
            seed_ids,
            expansion_budget=expansion_budget,
            include_related=include_related,
            min_confidence=min_confidence,
        )
        relations = self.relations_among(final_ids, min_confidence=min_confidence)
        return {
            "semantic_seed_indices": seed_indices,
            "semantic_seed_ids": seed_ids,
            "expanded_memories": expansions,
            "final_memory_ids": final_ids,
            "relations": relations,
        }

    def relations_among(
        self,
        memory_ids: Sequence[str],
        min_confidence: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """Return validated typed edges whose endpoints are both selected."""
        selected = set(memory_ids)
        threshold = _clamp_confidence(min_confidence)
        return [
            dict(edge)
            for edge in self.typed_relations
            if edge["source_id"] in selected
            and edge["target_id"] in selected
            and float(edge.get("confidence", 0.0)) >= threshold
        ]

    def _state_neighbor_ids(self, memory_id: str) -> List[str]:
        memory = self.memories.get(memory_id)
        if memory is None:
            return []
        state = self._ensure_temporal_state(memory)
        ordered = []
        for field_name in (
            "superseded_by",
            "supersedes",
            "conflicts_with",
            "refined_by",
            "refines",
        ):
            for neighbor_id in state.get(field_name, []):
                if neighbor_id in self.memories and neighbor_id not in ordered:
                    ordered.append(neighbor_id)
        return ordered

    def _referenced_memory_ids(
        self,
        question: str,
        memory_ids: Sequence[str],
    ) -> List[str]:
        question_tokens = set(re.findall(r"[a-z0-9]+", str(question).lower()))
        ignored = {
            "the", "a", "an", "is", "was", "were", "does", "did", "do",
            "user", "person", "what", "where", "when", "how", "which",
            "before", "previously", "previous", "earlier", "prior", "past",
        }
        question_tokens = {
            token for token in question_tokens
            if len(token) >= 3 and token not in ignored
        }
        scored = []
        for position, memory_id in enumerate(memory_ids):
            memory = self.memories[memory_id]
            searchable = " ".join([
                str(getattr(memory, "content", "")),
                " ".join(str(item) for item in getattr(memory, "keywords", [])),
                " ".join(str(item) for item in getattr(memory, "tags", [])),
            ]).lower()
            memory_tokens = set(re.findall(r"[a-z0-9]+", searchable))
            overlap = question_tokens & memory_tokens
            if overlap:
                scored.append((sum(len(token) for token in overlap), -position, memory_id))
        if not scored:
            return []
        best_score = max(item[0] for item in scored)
        return [
            memory_id
            for score, _, memory_id in sorted(scored, reverse=True)
            if score == best_score
        ]

    @staticmethod
    def _time_specific_match(state: Dict[str, Any], target: Dict[str, Any]) -> bool:
        valid_from = _parse_timestamp(state.get("valid_from"))
        valid_until = _parse_timestamp(state.get("valid_until"))
        target_start = _parse_timestamp(target.get("start"))
        target_end = _parse_timestamp(target.get("end"))

        if target_start is not None and target_end is not None:
            if valid_from is not None and valid_from >= target_end:
                return False
            if valid_until is not None and valid_until <= target_start:
                return False
            return valid_from is not None or valid_until is not None

        year = target.get("year")
        month = target.get("month")
        if year is None and month is not None and valid_from is not None:
            end_year = valid_until.year if valid_until is not None else valid_from.year
            for candidate_year in range(valid_from.year, end_year + 1):
                month_start = datetime(candidate_year, int(month), 1)
                month_end = _next_month(month_start)
                if valid_from < month_end and (
                    valid_until is None or valid_until > month_start
                ):
                    return True
            return False
        if valid_from is not None:
            if year is not None and valid_from.year != int(year):
                return False
            if month is not None and valid_from.month != int(month):
                return False
            return True
        return False

    def _temporal_priority(
        self,
        memory_id: str,
        intent: str,
        target: Dict[str, Any],
        original_position: int,
    ) -> Tuple[Any, ...]:
        memory = self.memories[memory_id]
        state = self._ensure_temporal_state(memory)
        timestamp = _parse_timestamp(state.get("valid_from"))
        timestamp_rank = timestamp.timestamp() if timestamp is not None else float("-inf")
        has_conflict = state.get("uncertainty") == "conflicting"

        if intent == "current":
            group = 0 if state.get("status") == "current" else 1
            return (group, has_conflict, -timestamp_rank, original_position)
        if intent == "historical":
            group = 0 if state.get("historical") or state.get("status") == "superseded" else 1
            return (group, has_conflict, -timestamp_rank, original_position)
        if intent == "time_specific":
            group = 0 if self._time_specific_match(state, target) else 1
            return (group, has_conflict, -timestamp_rank, original_position)
        if intent == "change":
            participates = bool(state.get("superseded_by") or state.get("supersedes"))
            group = 0 if participates else 1
            return (group, timestamp_rank, original_position)
        return (original_position,)

    def select_temporal_memories(
        self,
        memory_ids: Sequence[str],
        question: str,
        expansion_budget: int = 5,
        apply_ordering: bool = True,
    ) -> Dict[str, Any]:
        """Expand temporal evidence and optionally reorder it for the query intent."""
        query = detect_temporal_query(question)
        intent = query["intent"]
        selected = [memory_id for memory_id in memory_ids if memory_id in self.memories]
        selected = list(dict.fromkeys(selected))
        expansions: List[Dict[str, Any]] = []

        if not getattr(self, "temporal_state_enabled", False) or intent == "none":
            return {
                "query": query,
                "input_memory_ids": list(selected),
                "expanded_memories": expansions,
                "selected_memory_ids": list(selected),
                "states": {
                    memory_id: self.get_temporal_state(memory_id)
                    for memory_id in selected
                },
            }

        queue = list(selected)
        seen = set(selected)
        budget = max(0, int(expansion_budget))
        while queue and len(expansions) < budget:
            from_memory_id = queue.pop(0)
            for neighbor_id in self._state_neighbor_ids(from_memory_id):
                if neighbor_id in seen:
                    continue
                seen.add(neighbor_id)
                selected.append(neighbor_id)
                queue.append(neighbor_id)
                expansions.append({
                    "added_memory_id": neighbor_id,
                    "from_memory_id": from_memory_id,
                    "reason": "temporal_state_traversal",
                })
                if len(expansions) >= budget:
                    break

        positions = {memory_id: position for position, memory_id in enumerate(selected)}
        if (
            intent == "time_specific"
            and query["target"].get("month") is not None
            and query["target"].get("year") is None
            and query["target"].get("precision") != "relative"
        ):
            known_years = set()
            for memory_id in selected:
                state = self._ensure_temporal_state(self.memories[memory_id])
                for field_name in ("valid_from", "valid_until"):
                    timestamp = _parse_timestamp(state.get(field_name))
                    if timestamp is not None:
                        known_years.add(timestamp.year)
            if len(known_years) == 1:
                inferred_year = known_years.pop()
                month_start = datetime(
                    inferred_year,
                    int(query["target"]["month"]),
                    1,
                )
                query["target"].update({
                    "year": inferred_year,
                    "start": month_start.isoformat(),
                    "end": _next_month(month_start).isoformat(),
                    "year_inferred_from_timeline": True,
                })
        if (
            intent == "time_specific"
            and query.get("relative_reference")
            and selected
        ):
            dated = [
                _parse_timestamp(self._ensure_temporal_state(self.memories[memory_id]).get("valid_from"))
                for memory_id in selected
            ]
            dated = [value for value in dated if value is not None]
            if dated:
                anchor = max(dated)
                relative = query["relative_reference"]
                amount = relative["amount"]
                unit = relative["unit"]
                if unit == "day":
                    start = anchor - timedelta(days=amount)
                    end = start + timedelta(days=1)
                elif unit == "week":
                    start = anchor - timedelta(weeks=amount)
                    end = start + timedelta(weeks=1)
                elif unit == "month":
                    start = anchor - timedelta(days=30 * amount)
                    end = start + timedelta(days=31)
                else:
                    start = anchor - timedelta(days=365 * amount)
                    end = start + timedelta(days=366)
                query["target"].update({
                    "start": start.isoformat(),
                    "end": end.isoformat(),
                    "anchored_to": anchor.isoformat(),
                })
        if apply_ordering:
            selected.sort(
                key=lambda memory_id: self._temporal_priority(
                    memory_id,
                    intent,
                    query["target"],
                    positions[memory_id],
                )
            )
        historical_pivot_ids: List[str] = []
        preferred_historical_ids: List[str] = []
        if intent == "historical":
            historical_pivot_ids = self._referenced_memory_ids(question, selected)
            for pivot_id in historical_pivot_ids:
                pivot_state = self._ensure_temporal_state(self.memories[pivot_id])
                for previous_id in pivot_state.get("supersedes", []):
                    if previous_id in selected and previous_id not in preferred_historical_ids:
                        preferred_historical_ids.append(previous_id)
            if preferred_historical_ids and apply_ordering:
                preferred_set = set(preferred_historical_ids)
                selected.sort(key=lambda memory_id: memory_id not in preferred_set)
        return {
            "query": query,
            "input_memory_ids": list(memory_ids),
            "expanded_memories": expansions,
            "selected_memory_ids": selected,
            "historical_pivot_ids": historical_pivot_ids,
            "preferred_historical_ids": preferred_historical_ids,
            "states": {
                memory_id: self.get_temporal_state(memory_id)
                for memory_id in selected
            },
        }
