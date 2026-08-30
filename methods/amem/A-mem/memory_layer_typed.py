"""Experimental typed-relation extension for the robust A-MEM memory layer."""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import math
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from collections import deque
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from rank_bm25 import BM25Okapi

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
DEFAULT_GRAPH_RELATION_WEIGHTS = {
    "SUPERSEDE": 1.25,
    "CONFLICT": 0.75,
    "REFINE": 1.15,
    "SUPPORT": 1.0,
    "RELATED": 0.5,
}

_RETRIEVAL_STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "been", "before", "by",
    "did", "do", "does", "for", "from", "had", "has", "have", "he",
    "her", "his", "how", "i", "in", "is", "it", "me", "my", "of",
    "on", "or", "our", "she", "that", "the", "their", "them", "they",
    "this", "to", "user", "was", "were", "what", "when", "where",
    "which", "who", "why", "with", "you", "your",
})

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


def detect_temporal_query(
    question: str,
    semantic_intents_enabled: bool = True,
) -> Dict[str, Any]:
    """Parse explicit time and optionally classify regex temporal intent."""
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
            rf"\b({month_pattern})\b\s+(\d{{4}})\b",
            lowered,
        )
        if month_match:
            month = _MONTH_NAMES[month_match.group(1)]
            year = int(month_match.group(2))
            target.update({
                "raw": month_match.group(0),
                "year": year,
                "month": month,
                "precision": "month",
            })
            start = datetime(year, month, 1)
            target["start"] = start.isoformat()
            target["end"] = _next_month(start).isoformat()
        else:
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

    if semantic_intents_enabled and (
        change_pattern.search(lowered) or re.search(r"变化|改变|变更|前后", text)
    ):
        intent = "change"
    elif target["raw"] is not None:
        intent = "time_specific"
    elif not semantic_intents_enabled:
        intent = "none"
    else:
        if historical_pattern.search(lowered) or re.search(r"之前|以前|过去|曾经|原来|此前", text):
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

def detect_graph_query_intents(
    question: str,
    semantic_intents_enabled: bool = True,
) -> List[str]:
    """Detect deterministic relation preferences for query-conditioned diffusion."""
    if not semantic_intents_enabled:
        return []
    lowered = str(question or "").lower()
    intents = []
    patterns = {
        "causal": r"\b(?:why|cause[ds]?|because|led to|result(?:ed)? in|reason)\b",
        "conflict": r"\b(?:conflict|contradict|inconsistent|disagree|different)\b",
        "detail": r"\b(?:detail|specific|exact|precise|clarify|refine)\b",
        "change": r"\b(?:change[ds]?|changed|transition|before and after|replace[ds]?)\b",
    }
    for intent, pattern in patterns.items():
        if re.search(pattern, lowered):
            intents.append(intent)
    return intents

def detect_query_intents(
    question: str,
    semantic_intents_enabled: bool = True,
) -> Dict[str, Any]:
    """Return the shared temporal and graph query policy metadata."""
    temporal_query = detect_temporal_query(
        question,
        semantic_intents_enabled=semantic_intents_enabled,
    )
    temporal_query["relation_intents"] = detect_graph_query_intents(
        question,
        semantic_intents_enabled=semantic_intents_enabled,
    )
    return temporal_query


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

def _retrieval_tokens(value: Any) -> List[str]:
    return [
        token
        for token in re.findall(
            r"[a-z0-9]+|[\u3400-\u9fff]", str(value or "").lower()
        )
        if (len(token) > 1 or "\u3400" <= token <= "\u9fff")
        and token not in _RETRIEVAL_STOPWORDS
    ]


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
        parallel_workers: int = 1,
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
        self.parallel_workers = max(1, int(parallel_workers))
        self.typed_relations: List[Dict[str, Any]] = []
        self._typed_edge_keys = set()
        self.relation_audit: List[Dict[str, Any]] = []
        self._relation_audit_by_memory: Dict[str, Dict[str, Any]] = {}
        self.temporal_audit: List[Dict[str, Any]] = []
        self._temporal_audit_by_memory: Dict[str, List[Dict[str, Any]]] = {}
        self.evidence_store: Dict[str, Dict[str, Any]] = {}
        self.provenance_audit: List[Dict[str, Any]] = []
        self._provenance_audit_by_memory: Dict[str, Dict[str, Any]] = {}
        self._api_failure_events: List[Dict[str, Any]] = []
        self._api_failure_lock = threading.Lock()
        self.llm_controller.failure_callback = self._record_llm_failure
        underlying_llm = getattr(self.llm_controller, "llm", None)
        if underlying_llm is not None:
            underlying_llm.failure_callback = self._record_llm_failure

    def _record_llm_failure(
        self,
        operation_name: str,
        error: Exception,
        attempts: int,
    ) -> None:
        """Capture retry-exhausted build calls while preserving A-MEM fallbacks."""
        tracker = get_usage_tracker()
        operation = tracker.current_operation()
        if operation == "unscoped":
            operation = operation_name
        event = {
            "operation": operation,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "attempts": int(getattr(error, "attempts", attempts)),
        }
        if getattr(error, "failure_type", None):
            event["failure_type"] = error.failure_type
        if getattr(error, "retry_counts", None):
            event["retry_counts"] = dict(error.retry_counts)
        root_error = getattr(error, "last_exception", None)
        if root_error is not None:
            event["root_error_type"] = type(root_error).__name__
            event["root_error_message"] = str(root_error)
        else:
            event["root_error_type"] = type(error).__name__
            event["root_error_message"] = str(error)
        with self._api_failure_lock:
            self._api_failure_events.append(event)

    def drain_api_failure_events(self) -> List[Dict[str, Any]]:
        with self._api_failure_lock:
            events = list(self._api_failure_events)
            self._api_failure_events.clear()
        return events

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

    def prepare_note_metadata(
        self,
        contents: Sequence[str],
        max_workers: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Extract note metadata concurrently while preserving input order."""
        values = [str(content) for content in contents]
        workers = max(
            1,
            int(
                getattr(self, "parallel_workers", 1)
                if max_workers is None
                else max_workers
            ),
        )

        def analyze(content: str) -> Dict[str, Any]:
            get_usage_tracker().set_phase("memorize")
            max_chars = getattr(
                self,
                "max_context_chars",
                getattr(RobustMemoryNote, "DEFAULT_MAX_ANALYSIS_CHARS", 300000),
            )
            analysis_content = content
            if len(content) > max_chars:
                analysis_content = content[:max_chars] + "\n... [truncated for analysis]"
            return RobustMemoryNote.analyze_content(analysis_content, self.llm_controller)

        if workers == 1 or len(values) < 2:
            return [analyze(value) for value in values]
        with ThreadPoolExecutor(max_workers=min(workers, len(values))) as executor:
            return list(executor.map(analyze, values))

    def add_notes_parallel(
        self,
        note_specs: Sequence[Dict[str, Any]],
        max_workers: Optional[int] = None,
    ) -> List[str]:
        """Add typed-only notes with parallel analysis/relation inference.

        Notes, indexes, edges, temporal transitions, and audits are committed in
        source order. Only calls whose inputs are immutable at that point are
        dispatched concurrently, preserving the serial memory semantics.
        """
        specs = list(note_specs)
        workers = max(
            1,
            int(
                getattr(self, "parallel_workers", 1)
                if max_workers is None
                else max_workers
            ),
        )
        if (
            workers == 1
            or len(specs) < 2
            or self.original_evolution_enabled
            or not self.typed_relations_enabled
        ):
            return [
                self.add_note(
                    content=spec.get("content", ""),
                    time=spec.get("time"),
                    **dict(spec.get("kwargs") or {}),
                )
                for spec in specs
            ]

        # Preserve input order while allowing independent metadata calls.
        analyses = self.prepare_note_metadata(
            [spec.get("content", "") for spec in specs], workers
        )

        prepared: List[Dict[str, Any]] = []
        for spec, analysis in zip(specs, analyses):
            content = str(spec.get("content", ""))
            timestamp = spec.get("time")
            options = dict(spec.get("kwargs") or {})
            source_session_id = options.pop("source_session_id", None)
            source_session_index = options.pop("source_session_index", None)
            source_evidence = options.pop("source_evidence", None)
            source_timestamp = options.pop("source_timestamp", timestamp)
            provenance_part_index = options.pop("provenance_part_index", None)
            note = RobustMemoryNote(
                content=content,
                timestamp=timestamp,
                keywords=analysis.get("keywords"),
                context=analysis.get("context"),
                tags=analysis.get("tags"),
                llm_controller=None,
                max_analysis_chars=self.max_context_chars,
                **options,
            )
            if self.temporal_state_enabled or self.provenance_enabled:
                note.source_timestamp = _optional_text(source_timestamp)
                note.timestamp = note.source_timestamp or ""
            if source_session_id is not None:
                note.source_session_id = source_session_id
                note.source_session_index = source_session_index
            if self.temporal_state_enabled:
                note.temporal_state = self._initial_temporal_state(source_timestamp)
            self._attach_provenance(note, source_evidence, provenance_part_index)
            candidate_ids = self._candidate_ids(note.content)
            self._ensure_relation_lists(note)
            self.memories[note.id] = note
            with get_usage_tracker().scope("amem.embedding.index"):
                self.retriever.add_documents([
                    "content:" + note.content
                    + " context:" + note.context
                    + " keywords: " + ", ".join(note.keywords)
                    + " tags: " + ", ".join(note.tags)
                ])
            prepared.append({
                "note": note,
                "candidate_ids": candidate_ids,
            })

        def infer(item: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], str, str]:
            get_usage_tracker().set_phase("memorize")
            try:
                edges, raw_response = self._infer_relations(
                    item["note"], item["candidate_ids"]
                )
                return edges, raw_response, ""
            except Exception as exc:
                logger.warning(
                    "Typed relation inference failed for memory %s; "
                    "storing without typed relations: %s",
                    item["note"].id,
                    exc,
                )
                return [], "", str(exc)

        with ThreadPoolExecutor(max_workers=min(workers, len(prepared))) as executor:
            relation_results = list(executor.map(infer, prepared))

        note_ids: List[str] = []
        for item, (inferred_edges, raw_response, inference_error) in zip(
            prepared,
            relation_results,
        ):
            note = item["note"]
            candidate_ids = item["candidate_ids"]
            stored_edges = [edge for edge in inferred_edges if self._store_edge(edge)]
            note_ids.append(str(note.id))
            candidate_records = [
                {
                    "candidate_position": position,
                    "memory_id": memory_id,
                    "content": self.memories[memory_id].content[:400],
                }
                for position, memory_id in enumerate(candidate_ids)
                if memory_id in self.memories
            ]
            audit = {
                "created_memory_id": note.id,
                "candidate_neighbors": candidate_records,
                "predicted_relations": stored_edges,
                "relation_inference_error": inference_error,
                "raw_relation_response": raw_response,
            }
            self.relation_audit.append(audit)
            self._relation_audit_by_memory[note.id] = audit
        return note_ids

    def add_note(self, content: str, time: str = None, **kwargs) -> str:
        prepared_analysis = kwargs.pop("_prepared_analysis", None)
        source_session_id = kwargs.pop("source_session_id", None)
        source_session_index = kwargs.pop("source_session_index", None)
        experimental_metadata_enabled = (
            getattr(self, "temporal_state_enabled", False)
            or getattr(self, "provenance_enabled", False)
        )
        if (
            self.original_evolution_enabled
            and not self.typed_relations_enabled
            and not experimental_metadata_enabled
            and not isinstance(prepared_analysis, dict)
        ):
            note_id = super().add_note(content=content, time=time, **kwargs)
            if source_session_id is not None:
                note = self.memories[note_id]
                note.source_session_id = source_session_id
                note.source_session_index = source_session_index
            return note_id

        source_evidence = kwargs.pop("source_evidence", None)
        source_timestamp = kwargs.pop("source_timestamp", time)
        provenance_part_index = kwargs.pop("provenance_part_index", None)

        note_options = dict(kwargs)
        if isinstance(prepared_analysis, dict):
            note_options.update({
                "keywords": prepared_analysis.get("keywords"),
                "context": prepared_analysis.get("context"),
                "tags": prepared_analysis.get("tags"),
            })
        note = RobustMemoryNote(
            content=content,
            llm_controller=self.llm_controller,
            timestamp=time,
            max_analysis_chars=self.max_context_chars,
            **note_options,
        )
        if experimental_metadata_enabled:
            note.source_timestamp = _optional_text(source_timestamp)
            # RobustMemoryNote otherwise substitutes execution time for missing input.
            note.timestamp = note.source_timestamp or ""
        if source_session_id is not None:
            note.source_session_id = source_session_id
            note.source_session_index = source_session_index
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

    def _memory_search_text(self, memory_id: str) -> str:
        memory = self.memories[memory_id]
        return " ".join([
            str(getattr(memory, "content", "")),
            str(getattr(memory, "context", "")),
            " ".join(str(item) for item in getattr(memory, "keywords", [])),
            " ".join(str(item) for item in getattr(memory, "tags", [])),
        ])

    def _ordinary_neighbor_ids(self, memory_id: str) -> List[str]:
        memory_ids = list(self.memories.keys())
        memory = self.memories.get(memory_id)
        if memory is None:
            return []
        neighbors = []
        for link in getattr(memory, "links", []):
            if isinstance(link, str) and link in self.memories:
                neighbor_id = link
            else:
                try:
                    index = int(link)
                except (TypeError, ValueError):
                    continue
                if index < 0 or index >= len(memory_ids):
                    continue
                neighbor_id = memory_ids[index]
            if neighbor_id != memory_id and neighbor_id not in neighbors:
                neighbors.append(neighbor_id)
        return neighbors

    def _graph_proximity_ranking(
        self,
        seed_ids: Sequence[str],
        *,
        min_confidence: float,
        include_related: bool,
        use_typed_relations: bool,
        use_ordinary_links: bool,
        regex_intent_conditioning: bool = True,
    ) -> List[str]:
        allowed_types = set(DEFAULT_EXPANSION_TYPES)
        if include_related:
            allowed_types.add("RELATED")
        threshold = _clamp_confidence(min_confidence)
        seed_positions = {
            memory_id: position
            for position, memory_id in enumerate(seed_ids)
            if memory_id in self.memories
        }
        distances = {memory_id: 0 for memory_id in seed_positions}
        origins = dict(seed_positions)
        queue = deque(seed_positions)
        while queue:
            current_id = queue.popleft()
            typed_neighbors = []
            if use_typed_relations:
                for edge in self._memory_edges(current_id):
                    if edge["relation_type"] not in allowed_types:
                        continue
                    if float(edge.get("confidence", 0.0)) < threshold:
                        continue
                    typed_neighbors.append(
                        edge["target_id"]
                        if edge["source_id"] == current_id
                        else edge["source_id"]
                    )
            ordinary_neighbors = (
                self._ordinary_neighbor_ids(current_id) if use_ordinary_links else []
            )
            for neighbor_id in ordinary_neighbors + typed_neighbors:
                candidate = (distances[current_id] + 1, origins[current_id])
                previous = (
                    distances.get(neighbor_id, math.inf),
                    origins.get(neighbor_id, math.inf),
                )
                if candidate >= previous:
                    continue
                distances[neighbor_id], origins[neighbor_id] = candidate
                queue.append(neighbor_id)
        return [
            memory_id
            for memory_id in sorted(
                distances,
                key=lambda item: (
                    distances[item],
                    origins[item],
                    list(self.memories).index(item),
                ),
            )
            if distances[memory_id] > 0
        ]

    def hybrid_candidate_retrieval(
        self,
        query: str,
        raw_question: str,
        *,
        k: int,
        candidate_count: int,
        rrf_k: float,
        channel_weights: Mapping[str, float],
        min_confidence: float = 0.5,
        include_related: bool = False,
        use_typed_relations: bool = True,
        use_ordinary_links: bool = True,
        regex_intent_conditioning: bool = True,
    ) -> Dict[str, Any]:
        """Fuse deterministic candidate rankings with weighted reciprocal rank."""
        memory_ids = list(self.memories.keys())
        if not memory_ids or k <= 0:
            return {
                "selected_memory_ids": [],
                "seed_scores": {},
                "channel_rankings": {},
                "memory_scores": [],
            }

        pool_size = min(len(memory_ids), max(int(candidate_count), int(k)))
        weights = {
            str(name): max(0.0, float(value))
            for name, value in channel_weights.items()
        }
        rankings: Dict[str, List[str]] = {}

        dense_indices = self._normalize_indices(self.retriever.search(query, pool_size))
        dense_ids = [
            memory_ids[index]
            for index in dense_indices
            if 0 <= index < len(memory_ids)
        ]
        rankings["dense"] = list(dict.fromkeys(dense_ids))

        tokenized_corpus = [
            _retrieval_tokens(self._memory_search_text(memory_id))
            for memory_id in memory_ids
        ]
        query_tokens = _retrieval_tokens(query)
        if query_tokens and any(tokenized_corpus):
            bm25_scores = BM25Okapi(tokenized_corpus).get_scores(query_tokens)
            query_token_set = set(query_tokens)
            ranked = sorted(
                (
                    index
                    for index, document_tokens in enumerate(tokenized_corpus)
                    if query_token_set.intersection(document_tokens)
                ),
                key=lambda index: (-float(bm25_scores[index]), index),
            )
            rankings["bm25"] = [
                memory_ids[index]
                for index in ranked[:pool_size]
            ]
        else:
            rankings["bm25"] = []

        raw_tokens = set(_retrieval_tokens(raw_question))
        overlap_scores = []
        for position, memory_id in enumerate(memory_ids):
            memory = self.memories[memory_id]
            metadata_tokens = set(_retrieval_tokens(" ".join([
                str(getattr(memory, "context", "")),
                " ".join(str(item) for item in getattr(memory, "keywords", [])),
                " ".join(str(item) for item in getattr(memory, "tags", [])),
                str(getattr(memory, "content", "")),
            ])))
            overlap = raw_tokens & metadata_tokens
            if overlap:
                overlap_scores.append((sum(len(token) for token in overlap), position, memory_id))
        rankings["entity_attribute"] = [
            memory_id
            for _, _, memory_id in sorted(
                overlap_scores,
                key=lambda item: (-item[0], item[1]),
            )[:pool_size]
        ]

        temporal_query = detect_query_intents(
            raw_question,
            semantic_intents_enabled=regex_intent_conditioning,
        )
        target = temporal_query["target"]
        timestamp_matches = []
        state_matches = []
        for position, memory_id in enumerate(memory_ids):
            memory = self.memories[memory_id]
            state = self.get_temporal_state(memory_id)
            if not state:
                state = self._initial_temporal_state(
                    getattr(memory, "source_timestamp", None)
                    or getattr(memory, "timestamp", None)
                )
            source_timestamp = _parse_timestamp(
                getattr(memory, "source_timestamp", None)
                or getattr(memory, "timestamp", None)
            )
            target_start = _parse_timestamp(target.get("start"))
            target_end = _parse_timestamp(target.get("end"))
            if source_timestamp is not None and target.get("raw"):
                direct_match = (
                    target_start is not None
                    and target_end is not None
                    and target_start <= source_timestamp < target_end
                )
                if target_start is None or target_end is None:
                    direct_match = (
                        (target.get("year") is None or source_timestamp.year == target["year"])
                        and (
                            target.get("month") is None
                            or source_timestamp.month == target["month"]
                        )
                    )
                if direct_match:
                    timestamp_matches.append(memory_id)
            if getattr(self, "temporal_state_enabled", False):
                priority = self._temporal_priority(
                    memory_id,
                    temporal_query["intent"],
                    target,
                    position,
                )
                if temporal_query["intent"] != "none" and priority[0] == 0:
                    state_matches.append((priority, memory_id))
        rankings["timestamp"] = timestamp_matches[:pool_size]
        state_candidate_ids = set(
            memory_id
            for channel in ("dense", "bm25", "entity_attribute", "timestamp")
            for memory_id in rankings[channel]
        )
        rankings["state"] = [
            memory_id
            for _, memory_id in sorted(state_matches, key=lambda item: item[0])[:pool_size]
            if memory_id in state_candidate_ids
        ]
        graph_seed_ids = list(dict.fromkeys(
            memory_id
            for channel in (
                "dense", "bm25", "entity_attribute", "timestamp", "state"
            )
            for memory_id in rankings[channel][:max(1, k)]
        ))
        rankings["graph"] = self._graph_proximity_ranking(
            graph_seed_ids,
            min_confidence=min_confidence,
            include_related=include_related,
            use_typed_relations=use_typed_relations,
            use_ordinary_links=use_ordinary_links,
        )[:pool_size]

        fused_scores = {memory_id: 0.0 for memory_id in memory_ids}
        ranks_by_memory: Dict[str, Dict[str, int]] = {
            memory_id: {} for memory_id in memory_ids
        }
        contributions: Dict[str, Dict[str, float]] = {
            memory_id: {} for memory_id in memory_ids
        }
        denominator_offset = max(0.0, float(rrf_k))
        for channel, ranking in rankings.items():
            weight = weights.get(channel, 0.0)
            if weight <= 0.0:
                continue
            for rank, memory_id in enumerate(ranking, start=1):
                contribution = weight / (denominator_offset + rank)
                fused_scores[memory_id] += contribution
                ranks_by_memory[memory_id][channel] = rank
                contributions[memory_id][channel] = contribution

        positions = {memory_id: position for position, memory_id in enumerate(memory_ids)}
        ranked_ids = sorted(
            memory_ids,
            key=lambda memory_id: (-fused_scores[memory_id], positions[memory_id]),
        )
        selected_ids = [
            memory_id for memory_id in ranked_ids if fused_scores[memory_id] > 0.0
        ][:min(k, len(memory_ids))]
        total = sum(fused_scores[memory_id] for memory_id in selected_ids)
        seed_scores = {
            memory_id: (
                fused_scores[memory_id] / total
                if total > 0.0 else 1.0 / len(selected_ids)
            )
            for memory_id in selected_ids
        }
        return {
            "selected_memory_ids": selected_ids,
            "seed_scores": seed_scores,
            "channel_rankings": rankings,
            "memory_scores": [
                {
                    "memory_id": memory_id,
                    "fused_score": fused_scores[memory_id],
                    "normalized_seed_score": seed_scores.get(memory_id, 0.0),
                    "ranks": ranks_by_memory[memory_id],
                    "contributions": contributions[memory_id],
                }
                for memory_id in ranked_ids
                if fused_scores[memory_id] > 0.0
            ],
            "temporal_query": temporal_query,
        }

    def _temporal_transition_weight(
        self,
        memory_id: str,
        temporal_query: Dict[str, Any],
    ) -> float:
        if not getattr(self, "temporal_state_enabled", False):
            return 1.0
        intent = temporal_query.get("intent", "none")
        if intent == "none":
            return 1.0
        memory = self.memories[memory_id]
        state = self.get_temporal_state(memory_id)
        if not state:
            state = self._initial_temporal_state(
                getattr(memory, "source_timestamp", None)
                or getattr(memory, "timestamp", None)
            )
        if intent == "current":
            return 1.25 if state.get("status") == "current" else 0.5
        if intent == "historical":
            return 1.25 if state.get("historical") else 0.5
        if intent == "time_specific":
            return 1.25 if self._time_specific_match(
                state, temporal_query.get("target", {})
            ) else 0.5
        if intent == "change":
            participates = bool(state.get("superseded_by") or state.get("supersedes"))
            return 1.25 if participates else 0.75
        return 1.0

    def _graph_adjacency(
        self,
        *,
        mode: str,
        temporal_query: Dict[str, Any],
        relation_weights: Mapping[str, float],
        min_confidence: float,
        include_related: bool,
        use_typed_relations: bool,
        use_ordinary_links: bool,
        regex_intent_conditioning: bool = True,
    ) -> Dict[str, Dict[str, float]]:
        adjacency: Dict[str, Dict[str, float]] = {
            memory_id: {} for memory_id in self.memories
        }

        def add_neighbor(source_id: str, target_id: str, weight: float) -> None:
            if source_id == target_id or weight <= 0.0:
                return
            previous = adjacency[source_id].get(target_id, 0.0)
            adjacency[source_id][target_id] = previous + weight

        if use_ordinary_links:
            for memory_id in self.memories:
                for neighbor_id in self._ordinary_neighbor_ids(memory_id):
                    add_neighbor(memory_id, neighbor_id, 1.0)
                    add_neighbor(neighbor_id, memory_id, 1.0)

        threshold = _clamp_confidence(min_confidence)
        relation_intents = (
            set(temporal_query.get("relation_intents", []))
            if regex_intent_conditioning else set()
        )
        allowed_types = set(DEFAULT_EXPANSION_TYPES)
        if include_related:
            allowed_types.add("RELATED")
        for edge in self.typed_relations if use_typed_relations else []:
            relation_type = edge.get("relation_type")
            confidence = float(edge.get("confidence", 0.0))
            if relation_type not in allowed_types or confidence < threshold:
                continue
            source_id = edge["source_id"]
            target_id = edge["target_id"]
            if mode == "untyped_ppr":
                add_neighbor(source_id, target_id, 1.0)
                add_neighbor(target_id, source_id, 1.0)
                continue

            base_weight = max(0.0, float(relation_weights.get(relation_type, 0.0)))
            if base_weight <= 0.0:
                continue
            outgoing_weight = base_weight * confidence
            incoming_weight = base_weight * confidence
            intent = temporal_query.get("intent", "none")
            if "causal" in relation_intents and relation_type in {"SUPPORT", "RELATED"}:
                outgoing_weight *= 1.25
                incoming_weight *= 1.25
            if "conflict" in relation_intents and relation_type == "CONFLICT":
                outgoing_weight *= 1.5
                incoming_weight *= 1.5
            if "detail" in relation_intents and relation_type == "REFINE":
                outgoing_weight *= 1.35
                incoming_weight *= 1.35
            if "change" in relation_intents and relation_type in {
                "SUPERSEDE", "REFINE", "CONFLICT"
            }:
                outgoing_weight *= 1.25
                incoming_weight *= 1.25
            if relation_type == "SUPERSEDE":
                if intent == "current":
                    outgoing_weight *= 0.5
                    incoming_weight *= 1.5
                elif intent in {"historical", "time_specific"}:
                    outgoing_weight *= 1.5
                    incoming_weight *= 0.75
            elif intent == "change" and relation_type in {
                "SUPERSEDE", "REFINE", "CONFLICT"
            }:
                outgoing_weight *= 1.25
                incoming_weight *= 1.25
            add_neighbor(
                source_id,
                target_id,
                outgoing_weight
                * self._temporal_transition_weight(target_id, temporal_query),
            )
            add_neighbor(
                target_id,
                source_id,
                incoming_weight
                * self._temporal_transition_weight(source_id, temporal_query),
            )
        return adjacency

    def query_conditioned_graph_rank(
        self,
        seed_scores: Mapping[str, float],
        raw_question: str,
        *,
        mode: str,
        expansion_budget: int,
        alpha: float,
        iterations: int,
        tolerance: float,
        relation_weights: Mapping[str, float],
        min_confidence: float = 0.5,
        include_related: bool = False,
        use_typed_relations: bool = True,
        use_ordinary_links: bool = True,
        regex_intent_conditioning: bool = True,
    ) -> Dict[str, Any]:
        """Rank graph memories with personalized PageRank from query seeds."""
        if mode not in {"untyped_ppr", "typed_ppr"}:
            raise ValueError(f"Unsupported graph ranking mode: {mode}")
        memory_ids = list(self.memories.keys())
        positions = {memory_id: position for position, memory_id in enumerate(memory_ids)}
        seeds = {
            memory_id: max(0.0, float(score))
            for memory_id, score in seed_scores.items()
            if memory_id in self.memories
        }
        total_seed_score = sum(seeds.values())
        if total_seed_score <= 0.0:
            seeds = {
                memory_id: 1.0
                for memory_id in seed_scores
                if memory_id in self.memories
            }
            total_seed_score = sum(seeds.values())
        if total_seed_score <= 0.0:
            return {
                "selected_memory_ids": [],
                "expanded_memories": [],
                "scores": {},
                "iterations_run": 0,
                "converged": True,
                "temporal_query": detect_query_intents(
                    raw_question,
                    semantic_intents_enabled=regex_intent_conditioning,
                ),
            }
        restart = {
            memory_id: seeds.get(memory_id, 0.0) / total_seed_score
            for memory_id in memory_ids
        }
        temporal_query = detect_query_intents(
            raw_question,
            semantic_intents_enabled=regex_intent_conditioning,
        )
        adjacency = self._graph_adjacency(
            mode=mode,
            temporal_query=temporal_query,
            relation_weights=relation_weights,
            min_confidence=min_confidence,
            include_related=include_related,
            use_typed_relations=use_typed_relations,
            use_ordinary_links=use_ordinary_links,
            regex_intent_conditioning=regex_intent_conditioning,
        )
        transition = {}
        for source_id, neighbors in adjacency.items():
            total_weight = sum(neighbors.values())
            transition[source_id] = {
                target_id: weight / total_weight
                for target_id, weight in neighbors.items()
            } if total_weight > 0.0 else {}

        damping = max(0.0, min(0.999999, float(alpha)))
        max_iterations = max(1, int(iterations))
        convergence_tolerance = max(0.0, float(tolerance))
        scores = dict(restart)
        converged = False
        iterations_run = 0
        for iteration in range(max_iterations):
            next_scores = {
                memory_id: (1.0 - damping) * restart[memory_id]
                for memory_id in memory_ids
            }
            dangling_mass = 0.0
            for source_id, source_score in scores.items():
                neighbors = transition[source_id]
                if not neighbors:
                    dangling_mass += source_score
                    continue
                for target_id, probability in neighbors.items():
                    next_scores[target_id] += damping * source_score * probability
            if dangling_mass:
                for memory_id, restart_score in restart.items():
                    next_scores[memory_id] += damping * dangling_mass * restart_score
            delta = sum(
                abs(next_scores[memory_id] - scores[memory_id])
                for memory_id in memory_ids
            )
            scores = next_scores
            iterations_run = iteration + 1
            if delta <= convergence_tolerance:
                converged = True
                break

        ranked_ids = sorted(
            memory_ids,
            key=lambda memory_id: (-scores[memory_id], positions[memory_id]),
        )
        seed_ids = set(seeds)
        expansion_ids = [
            memory_id
            for memory_id in ranked_ids
            if memory_id not in seed_ids and scores[memory_id] > 0.0
        ][:max(0, int(expansion_budget))]
        selected_set = seed_ids | set(expansion_ids)
        selected_ids = [
            memory_id for memory_id in ranked_ids if memory_id in selected_set
        ]
        return {
            "selected_memory_ids": selected_ids,
            "expanded_memories": [
                {
                    "added_memory_id": memory_id,
                    "reason": mode,
                    "graph_score": scores[memory_id],
                }
                for memory_id in expansion_ids
            ],
            "scores": {
                memory_id: scores[memory_id]
                for memory_id in ranked_ids
                if scores[memory_id] > 0.0
            },
            "iterations_run": iterations_run,
            "converged": converged,
            "temporal_query": temporal_query,
        }

    def _chain_edge_weights(
        self,
        candidate_ids: Sequence[str],
        raw_question: str,
        *,
        relation_weights: Mapping[str, float],
        min_confidence: float,
        include_related: bool,
        use_typed_relations: bool,
        use_ordinary_links: bool,
        regex_intent_conditioning: bool = True,
    ) -> Tuple[Dict[Tuple[str, str], float], List[Dict[str, Any]]]:
        candidate_set = set(candidate_ids)
        positions = {memory_id: position for position, memory_id in enumerate(candidate_ids)}
        edge_weights: Dict[Tuple[str, str], float] = {}
        edge_records: Dict[Tuple[str, str], Dict[str, Any]] = {}

        def add_edge(
            source_id: str,
            target_id: str,
            weight: float,
            relation_type: str,
            confidence: float,
        ) -> None:
            if (
                source_id == target_id
                or source_id not in candidate_set
                or target_id not in candidate_set
                or weight <= 0.0
            ):
                return
            key = tuple(sorted(
                (source_id, target_id),
                key=lambda memory_id: positions[memory_id],
            ))
            bounded_weight = max(0.0, min(1.0, float(weight)))
            if bounded_weight <= edge_weights.get(key, 0.0):
                return
            edge_weights[key] = bounded_weight
            edge_records[key] = {
                "source_id": source_id,
                "target_id": target_id,
                "relation_type": relation_type,
                "confidence": confidence,
                "weight": bounded_weight,
            }

        if use_ordinary_links:
            for memory_id in candidate_ids:
                for neighbor_id in self._ordinary_neighbor_ids(memory_id):
                    add_edge(memory_id, neighbor_id, 0.5, "ORDINARY_LINK", 1.0)

        if use_typed_relations:
            allowed_types = set(DEFAULT_EXPANSION_TYPES)
            if include_related:
                allowed_types.add("RELATED")
            threshold = _clamp_confidence(min_confidence)
            temporal_query = detect_temporal_query(
                raw_question,
                semantic_intents_enabled=regex_intent_conditioning,
            )
            relation_intents = set(detect_graph_query_intents(
                raw_question,
                semantic_intents_enabled=regex_intent_conditioning,
            ))
            max_relation_weight = max(
                [max(0.0, float(value)) for value in relation_weights.values()]
                or [1.0]
            )
            for edge in self.typed_relations:
                relation_type = str(edge.get("relation_type") or "")
                confidence = float(edge.get("confidence", 0.0))
                if relation_type not in allowed_types or confidence < threshold:
                    continue
                type_weight = max(
                    0.0, float(relation_weights.get(relation_type, 0.0))
                ) / max(max_relation_weight, 1e-12)
                intent_weight = 1.0
                if "conflict" in relation_intents and relation_type == "CONFLICT":
                    intent_weight *= 1.25
                if "detail" in relation_intents and relation_type == "REFINE":
                    intent_weight *= 1.2
                if "change" in relation_intents and relation_type in {
                    "SUPERSEDE", "REFINE", "CONFLICT"
                }:
                    intent_weight *= 1.2
                if "causal" in relation_intents and relation_type == "SUPPORT":
                    intent_weight *= 1.1
                temporal_weight = 1.0
                if temporal_query["intent"] == "change" and relation_type in {
                    "SUPERSEDE", "REFINE"
                }:
                    temporal_weight = 1.2
                add_edge(
                    str(edge["source_id"]),
                    str(edge["target_id"]),
                    confidence * type_weight * intent_weight * temporal_weight,
                    relation_type,
                    confidence,
                )
        return edge_weights, [
            edge_records[key]
            for key in sorted(
                edge_records,
                key=lambda item: (positions[item[0]], positions[item[1]]),
            )
        ]

    def _chain_question_facets(
        self,
        raw_question: str,
        candidate_ids: Sequence[str],
        *,
        temporal_state_enabled: bool,
        temporal_query: Optional[Mapping[str, Any]] = None,
        regex_intent_conditioning: bool = True,
    ) -> Dict[str, Any]:
        question_tokens = list(dict.fromkeys(_retrieval_tokens(raw_question)))
        document_tokens = {
            memory_id: set(_retrieval_tokens(self._memory_search_text(memory_id)))
            for memory_id in candidate_ids
        }
        lexical_weights = {}
        for token in question_tokens:
            document_frequency = sum(
                token in tokens for tokens in document_tokens.values()
            )
            lexical_weights[token] = math.log(
                (len(candidate_ids) + 1) / (document_frequency + 1)
            ) + 1.0

        temporal_query = copy.deepcopy(temporal_query or detect_temporal_query(
            raw_question,
            semantic_intents_enabled=regex_intent_conditioning,
        ))
        relation_intents = detect_graph_query_intents(
            raw_question,
            semantic_intents_enabled=regex_intent_conditioning,
        )
        structural = []
        intent = temporal_query["intent"]
        if intent == "current" and temporal_state_enabled:
            structural.append("current_state")
        elif intent == "historical" and temporal_state_enabled:
            structural.append("historical_state")
        elif intent == "time_specific":
            structural.append("target_time")
        elif intent == "change":
            if temporal_state_enabled:
                structural.extend(("earlier_state", "later_state"))
            structural.append("transition")
        for relation_intent in relation_intents:
            structural.append(f"operation:{relation_intent}")
        lowered = str(raw_question or "").lower()
        if re.search(r"\b(?:compare|difference|versus|vs\.?|both)\b", lowered):
            structural.append("operation:comparison")
        if re.search(r"\b(?:list|what are|name all|all the)\b", lowered):
            structural.append("operation:enumeration")

        all_weights = dict(lexical_weights)
        structural_weight = max(
            sum(lexical_weights.values()) / max(len(lexical_weights), 1),
            1.0,
        )
        for facet in structural:
            all_weights[facet] = structural_weight
        total_weight = sum(all_weights.values())
        normalized_weights = {
            facet: weight / total_weight
            for facet, weight in all_weights.items()
        } if total_weight > 0.0 else {}
        return {
            "lexical": question_tokens,
            "structural": structural,
            "weights": normalized_weights,
            "temporal_query": temporal_query,
            "relation_intents": relation_intents,
            "document_tokens": document_tokens,
            "temporal_state_enabled": temporal_state_enabled,
        }

    def _chain_facet_matches(
        self,
        candidate_ids: Sequence[str],
        facets: Dict[str, Any],
        edge_weights: Mapping[Tuple[str, str], float],
        edge_records: Sequence[Dict[str, Any]],
    ) -> Dict[str, Dict[str, float]]:
        matches: Dict[str, Dict[str, float]] = {}
        temporal_query = facets["temporal_query"]
        target = temporal_query["target"]
        for memory_id in candidate_ids:
            memory = self.memories[memory_id]
            memory_matches = {
                token: 1.0
                for token in facets["lexical"]
                if token in facets["document_tokens"][memory_id]
            }
            state = (
                self.get_temporal_state(memory_id)
                if facets["temporal_state_enabled"] else {}
            )
            if "current_state" in facets["structural"]:
                memory_matches["current_state"] = float(
                    state.get("status") == "current"
                )
            if "historical_state" in facets["structural"]:
                memory_matches["historical_state"] = float(
                    bool(state.get("historical"))
                    or state.get("status") == "superseded"
                )
            if "target_time" in facets["structural"]:
                source_timestamp = _parse_timestamp(
                    getattr(memory, "source_timestamp", None)
                    or getattr(memory, "timestamp", None)
                )
                target_start = _parse_timestamp(target.get("start"))
                target_end = _parse_timestamp(target.get("end"))
                direct_match = bool(
                    source_timestamp is not None
                    and target_start is not None
                    and target_end is not None
                    and target_start <= source_timestamp < target_end
                )
                has_calendar_bound = bool(
                    target.get("year") is not None
                    or target.get("month") is not None
                )
                if (target_start is None or target_end is None) and has_calendar_bound:
                    direct_match = bool(
                        source_timestamp is not None
                        and (
                            target.get("year") is None
                            or source_timestamp.year == target["year"]
                        )
                        and (
                            target.get("month") is None
                            or source_timestamp.month == target["month"]
                        )
                    )
                memory_matches["target_time"] = float(direct_match)
            if "earlier_state" in facets["structural"]:
                memory_matches["earlier_state"] = float(
                    bool(state.get("historical"))
                    or state.get("status") == "superseded"
                )
            if "later_state" in facets["structural"]:
                memory_matches["later_state"] = float(
                    state.get("status") == "current"
                )
            if "transition" in facets["structural"]:
                memory_matches["transition"] = 0.0
            for relation_intent in facets["relation_intents"]:
                compatible_types = {
                    "conflict": {"CONFLICT"},
                    "detail": {"REFINE"},
                    "change": {"SUPERSEDE", "REFINE", "CONFLICT"},
                    "causal": {"SUPPORT"},
                }.get(relation_intent, set())
                participates = any(
                    memory_id in {edge["source_id"], edge["target_id"]}
                    and edge.get("relation_type") in compatible_types
                    for edge in edge_records
                )
                memory_matches[f"operation:{relation_intent}"] = float(participates)
            if "operation:comparison" in facets["structural"]:
                memory_matches["operation:comparison"] = float(
                    bool(state.get("conflicts_with"))
                    or bool(state.get("superseded_by"))
                    or bool(state.get("supersedes"))
                )
            if "operation:enumeration" in facets["structural"]:
                memory_matches["operation:enumeration"] = 1.0
            matches[memory_id] = memory_matches
        return matches

    def _chain_short_paths(
        self,
        candidate_ids: Sequence[str],
        relevance: Mapping[str, float],
        facet_matches: Mapping[str, Mapping[str, float]],
        edge_weights: Mapping[Tuple[str, str], float],
        max_hops: int,
    ) -> List[Dict[str, Any]]:
        positions = {memory_id: position for position, memory_id in enumerate(candidate_ids)}
        adjacency: Dict[str, List[Tuple[str, float]]] = {
            memory_id: [] for memory_id in candidate_ids
        }
        for (source_id, target_id), weight in edge_weights.items():
            adjacency[source_id].append((target_id, weight))
            adjacency[target_id].append((source_id, weight))
        for memory_id in adjacency:
            adjacency[memory_id].sort(
                key=lambda item: (-item[1], positions[item[0]])
            )
            adjacency[memory_id] = adjacency[memory_id][:8]

        anchors = sorted(
            candidate_ids,
            key=lambda memory_id: (-relevance.get(memory_id, 0.0), positions[memory_id]),
        )
        paths = []
        seen_paths = set()
        hop_limit = min(3, max(1, int(max_hops)))
        for source_position, source_id in enumerate(anchors):
            source_facets = {
                facet for facet, value in facet_matches[source_id].items() if value > 0.0
            }
            for target_id in anchors[source_position + 1:]:
                target_facets = {
                    facet for facet, value in facet_matches[target_id].items() if value > 0.0
                }
                if source_facets == target_facets and source_facets:
                    continue
                best_path = None
                best_quality = -1.0
                queue = deque([([source_id], [])])
                while queue:
                    path, weights = queue.popleft()
                    current_id = path[-1]
                    if len(path) - 1 >= hop_limit:
                        continue
                    for neighbor_id, edge_weight in adjacency[current_id]:
                        if neighbor_id in path:
                            continue
                        next_path = path + [neighbor_id]
                        next_weights = weights + [edge_weight]
                        if neighbor_id == target_id:
                            geometric_mean = math.prod(next_weights) ** (
                                1.0 / len(next_weights)
                            )
                            quality = (
                                math.sqrt(
                                    relevance.get(source_id, 0.0)
                                    * relevance.get(target_id, 0.0)
                                )
                                * geometric_mean
                                / len(next_weights)
                            )
                            if quality > best_quality:
                                best_quality = quality
                                best_path = next_path
                        else:
                            queue.append((next_path, next_weights))
                if best_path is None:
                    continue
                key = tuple(best_path)
                reverse_key = tuple(reversed(best_path))
                if key in seen_paths or reverse_key in seen_paths:
                    continue
                seen_paths.add(key)
                paths.append({"memory_ids": best_path, "quality": best_quality})
        paths.sort(key=lambda item: (
            -item["quality"],
            len(item["memory_ids"]),
            tuple(positions[memory_id] for memory_id in item["memory_ids"]),
        ))
        paths = paths[:100]
        total_quality = sum(item["quality"] for item in paths)
        for item in paths:
            item["normalized_quality"] = (
                item["quality"] / total_quality if total_quality > 0.0 else 0.0
            )
        return paths

    def select_chain_preserving_evidence(
        self,
        candidate_ids: Sequence[str],
        raw_question: str,
        *,
        candidate_rankings: Optional[Mapping[str, Sequence[str]]] = None,
        hybrid_scores: Optional[Mapping[str, float]] = None,
        graph_scores: Optional[Mapping[str, float]] = None,
        token_budget: int,
        token_cost: Callable[[Sequence[str]], int],
        candidate_count: int = 50,
        evidence_count: int = 30,
        max_hops: int = 2,
        max_groups: int = 3,
        relevance_weight: float = 1.0,
        coverage_weight: float = 1.0,
        connectivity_weight: float = 0.35,
        path_weight: float = 0.75,
        temporal_weight: float = 0.5,
        redundancy_weight: float = 0.25,
        relation_weights: Optional[Mapping[str, float]] = None,
        min_confidence: float = 0.5,
        include_related: bool = False,
        use_typed_relations: bool = True,
        use_ordinary_links: bool = True,
        temporal_state_enabled: bool = False,
        temporal_query: Optional[Mapping[str, Any]] = None,
        regex_intent_conditioning: bool = True,
    ) -> Dict[str, Any]:
        """Select complete, connected evidence bundles under an exact token budget."""
        rankings = {
            str(channel): [
                memory_id
                for memory_id in dict.fromkeys(ranking)
                if memory_id in self.memories
            ]
            for channel, ranking in (candidate_rankings or {}).items()
        }
        if not rankings:
            rankings = {"candidates": list(dict.fromkeys(candidate_ids))}
        ranking_rrf = {
            memory_id: 0.0
            for memory_id in dict.fromkeys(
                list(candidate_ids)
                + [memory_id for ranking in rankings.values() for memory_id in ranking]
            )
            if memory_id in self.memories
        }
        ranking_positions: Dict[str, Dict[str, int]] = {
            memory_id: {} for memory_id in ranking_rrf
        }
        for channel, ranking in rankings.items():
            for rank, memory_id in enumerate(ranking, start=1):
                if memory_id not in ranking_rrf:
                    continue
                ranking_rrf[memory_id] += 1.0 / (60.0 + rank)
                ranking_positions[memory_id][channel] = rank
        insertion_positions = {
            memory_id: position
            for position, memory_id in enumerate(ranking_rrf)
        }
        ordered_candidates = sorted(
            ranking_rrf,
            key=lambda memory_id: (
                -ranking_rrf[memory_id], insertion_positions[memory_id]
            ),
        )[:max(1, int(candidate_count))]
        positions = {
            memory_id: position for position, memory_id in enumerate(ordered_candidates)
        }
        budget = max(0, int(token_budget))
        target_count = min(
            len(ordered_candidates), max(1, int(evidence_count))
        )
        if not ordered_candidates or budget <= 0:
            return {
                "candidate_memory_ids": ordered_candidates,
                "selected_memory_ids": [],
                "selected_paths": [],
                "question_facets": {},
                "candidate_scores": {},
                "evidence_graph_edges": [],
                "utility": {},
                "token_budget": budget,
                "target_evidence_count": target_count,
                "selected_tokens": 0,
                "selection_steps": [],
                "rejected_candidates": [
                    {"memory_id": memory_id, "reason": "no_token_budget"}
                    for memory_id in ordered_candidates
                ],
            }

        def normalize_scores(scores: Mapping[str, float]) -> Dict[str, float]:
            positive = {
                memory_id: max(0.0, float(scores.get(memory_id, 0.0)))
                for memory_id in ordered_candidates
            }
            maximum = max(positive.values(), default=0.0)
            return {
                memory_id: value / maximum
                for memory_id, value in positive.items()
            } if maximum > 0.0 else {}

        normalized_hybrid = normalize_scores(hybrid_scores or {})
        normalized_graph = normalize_scores(graph_scores or {})
        normalized_fusion = normalize_scores(ranking_rrf)
        relevance = {
            memory_id: (
                0.6 * normalized_fusion.get(memory_id, 0.0)
                + 0.25 * normalized_hybrid.get(memory_id, 0.0)
                + 0.15 * normalized_graph.get(memory_id, 0.0)
            )
            for memory_id in ordered_candidates
        }

        edge_weights, edge_records = self._chain_edge_weights(
            ordered_candidates,
            raw_question,
            relation_weights=relation_weights or DEFAULT_GRAPH_RELATION_WEIGHTS,
            min_confidence=min_confidence,
            include_related=include_related,
            use_typed_relations=use_typed_relations,
            use_ordinary_links=use_ordinary_links,
            regex_intent_conditioning=regex_intent_conditioning,
        )
        facets = self._chain_question_facets(
            raw_question,
            ordered_candidates,
            temporal_state_enabled=temporal_state_enabled,
            temporal_query=temporal_query,
            regex_intent_conditioning=regex_intent_conditioning,
        )
        facet_matches = self._chain_facet_matches(
            ordered_candidates, facets, edge_weights, edge_records
        )
        paths = self._chain_short_paths(
            ordered_candidates,
            relevance,
            facet_matches,
            edge_weights,
            max_hops,
        )

        embeddings = getattr(getattr(self, "retriever", None), "embeddings", None)
        memory_positions = {
            memory_id: position for position, memory_id in enumerate(self.memories)
        }

        def cosine_similarity(first_id: str, second_id: str) -> float:
            if embeddings is None:
                return 0.0
            try:
                first = embeddings[memory_positions[first_id]]
                second = embeddings[memory_positions[second_id]]
                dot = sum(float(a) * float(b) for a, b in zip(first, second))
                first_norm = math.sqrt(sum(float(value) ** 2 for value in first))
                second_norm = math.sqrt(sum(float(value) ** 2 for value in second))
                if first_norm <= 0.0 or second_norm <= 0.0:
                    return 0.0
                return max(0.0, min(1.0, dot / (first_norm * second_norm)))
            except (IndexError, KeyError, TypeError, ValueError):
                return 0.0

        relation_pairs = set(edge_weights)
        redundancy_cache: Dict[Tuple[str, str], float] = {}

        def pair_redundancy(first_id: str, second_id: str) -> float:
            cache_key = tuple(sorted(
                (first_id, second_id), key=lambda memory_id: positions[memory_id]
            ))
            if cache_key in redundancy_cache:
                return redundancy_cache[cache_key]
            first_tokens = facets["document_tokens"][first_id]
            second_tokens = facets["document_tokens"][second_id]
            lexical = len(first_tokens & second_tokens) / max(
                len(first_tokens | second_tokens), 1
            )
            first_memory = self.memories[first_id]
            second_memory = self.memories[second_id]
            first_evidence = set(
                getattr(first_memory, "provenance", {}).get("evidence_ids", [])
            )
            second_evidence = set(
                getattr(second_memory, "provenance", {}).get("evidence_ids", [])
            )
            provenance = (
                len(first_evidence & second_evidence)
                / max(len(first_evidence | second_evidence), 1)
                if first_evidence or second_evidence else 0.0
            )
            relation_adjustment = 0.25 if cache_key in relation_pairs else 1.0
            result = min(
                1.0,
                0.5 * provenance
                + relation_adjustment * (
                    0.3 * cosine_similarity(first_id, second_id)
                    + 0.2 * lexical
                ),
            )
            redundancy_cache[cache_key] = result
            return result

        def connected_components(selected: Set[str]) -> int:
            if not selected:
                return 0
            remaining = set(selected)
            components = 0
            while remaining:
                components += 1
                queue = [remaining.pop()]
                while queue:
                    current_id = queue.pop()
                    neighbors = {
                        target_id if source_id == current_id else source_id
                        for source_id, target_id in edge_weights
                        if current_id in {source_id, target_id}
                    }
                    for neighbor_id in neighbors & remaining:
                        remaining.remove(neighbor_id)
                        queue.append(neighbor_id)
            return components

        def maximum_spanning_forest(selected: Set[str]) -> float:
            if len(selected) <= 1:
                return 0.0
            parent = {memory_id: memory_id for memory_id in selected}

            def find(memory_id: str) -> str:
                while parent[memory_id] != memory_id:
                    parent[memory_id] = parent[parent[memory_id]]
                    memory_id = parent[memory_id]
                return memory_id

            total = 0.0
            eligible_edges = sorted(
                (
                    (weight, source_id, target_id)
                    for (source_id, target_id), weight in edge_weights.items()
                    if source_id in selected and target_id in selected
                ),
                key=lambda item: (-item[0], positions[item[1]], positions[item[2]]),
            )
            for weight, source_id, target_id in eligible_edges:
                source_root = find(source_id)
                target_root = find(target_id)
                if source_root == target_root:
                    continue
                parent[target_root] = source_root
                total += weight
            return total / max(target_count - 1, 1)

        def temporal_utility(selected: Set[str]) -> float:
            intent = facets["temporal_query"]["intent"]
            if intent == "none" or not selected:
                return 0.0
            if not temporal_state_enabled:
                if intent != "time_specific":
                    return 0.0
                return max(
                    (
                        facet_matches[memory_id].get("target_time", 0.0)
                        for memory_id in selected
                    ),
                    default=0.0,
                )
            if intent == "change":
                best = 0.0
                for edge in edge_records:
                    if (
                        edge["source_id"] in selected
                        and edge["target_id"] in selected
                        and edge["relation_type"] in {"SUPERSEDE", "REFINE", "CONFLICT"}
                    ):
                        best = max(best, float(edge["weight"]))
                return best
            facet = {
                "current": "current_state",
                "historical": "historical_state",
                "time_specific": "target_time",
            }.get(intent)
            return max(
                (facet_matches[memory_id].get(facet, 0.0) for memory_id in selected),
                default=0.0,
            )

        def utility(selected: Set[str]) -> Dict[str, float]:
            selected_in_order = [
                memory_id for memory_id in ordered_candidates if memory_id in selected
            ]
            relevance_score = sum(
                relevance[memory_id] for memory_id in selected_in_order
            )
            coverage_score = 0.0
            for facet, weight in facets["weights"].items():
                if facet == "transition":
                    covered = any(
                        edge["source_id"] in selected
                        and edge["target_id"] in selected
                        and edge["relation_type"] in {
                            "SUPERSEDE", "REFINE", "CONFLICT"
                        }
                        for edge in edge_records
                    )
                    facet_coverage = float(covered)
                else:
                    facet_coverage = min(
                        1.0,
                        sum(
                            facet_matches[memory_id].get(facet, 0.0)
                            for memory_id in selected_in_order
                        ),
                    )
                coverage_score += weight * facet_coverage
            connectivity_score = maximum_spanning_forest(selected)
            path_score = sum(
                path["normalized_quality"]
                for path in paths
                if set(path["memory_ids"]).issubset(selected)
            )
            temporal_score = temporal_utility(selected)
            pairs = [
                (first_id, second_id)
                for first_position, first_id in enumerate(selected_in_order)
                for second_id in selected_in_order[first_position + 1:]
            ]
            redundancy_score = sum(pair_redundancy(*pair) for pair in pairs)
            if pairs:
                redundancy_score /= len(pairs)
            component_count = connected_components(selected)
            excess_groups = max(0, component_count - max(1, int(max_groups)))
            group_penalty = excess_groups / max(target_count, 1)
            total = (
                max(0.0, float(relevance_weight)) * relevance_score
                + max(0.0, float(coverage_weight)) * coverage_score
                + max(0.0, float(connectivity_weight)) * (
                    connectivity_score - group_penalty
                )
                + max(0.0, float(path_weight)) * path_score
                + max(0.0, float(temporal_weight)) * temporal_score
                - max(0.0, float(redundancy_weight)) * redundancy_score
            )
            return {
                "relevance": relevance_score,
                "coverage": coverage_score,
                "connectivity": connectivity_score,
                "path": path_score,
                "temporal": temporal_score,
                "redundancy": redundancy_score,
                "component_count": component_count,
                "group_penalty": group_penalty,
                "total": total,
            }

        actions = [
            {"type": "memory", "memory_ids": [memory_id], "quality": relevance[memory_id]}
            for memory_id in ordered_candidates
        ] + [
            {"type": "path", **path} for path in paths
        ]
        selected: Set[str] = set()
        current_utility = utility(selected)
        current_tokens = 0
        selection_steps = []
        singleton_token_costs = {
            memory_id: int(token_cost([memory_id]))
            for memory_id in ordered_candidates
        }
        cheapest_target_ids = sorted(
            ordered_candidates,
            key=lambda memory_id: (
                singleton_token_costs[memory_id], positions[memory_id]
            ),
        )[:target_count]
        cheapest_target_ids.sort(key=lambda memory_id: positions[memory_id])
        target_feasible = bool(
            len(cheapest_target_ids) == target_count
            and int(token_cost(cheapest_target_ids)) <= budget
        )

        def preserves_target_feasibility(proposed: Set[str]) -> bool:
            if not target_feasible or len(proposed) >= target_count:
                return True
            completion_count = target_count - len(proposed)
            completion = sorted(
                (
                    memory_id
                    for memory_id in ordered_candidates
                    if memory_id not in proposed
                ),
                key=lambda memory_id: (
                    singleton_token_costs[memory_id], positions[memory_id]
                ),
            )[:completion_count]
            if len(completion) < completion_count:
                return False
            completed = proposed | set(completion)
            completed_order = [
                memory_id
                for memory_id in ordered_candidates
                if memory_id in completed
            ]
            return int(token_cost(completed_order)) <= budget

        while len(selected) < target_count:
            ranked_proposals = []
            for action_position, action in enumerate(actions):
                additions = set(action["memory_ids"]) - selected
                if not additions:
                    continue
                proposed = selected | additions
                if len(proposed) > target_count:
                    continue
                proposed_order = [
                    memory_id for memory_id in ordered_candidates if memory_id in proposed
                ]
                proposed_utility = utility(proposed)
                marginal = proposed_utility["total"] - current_utility["total"]
                density = marginal / len(additions)
                tie_key = (
                    density,
                    marginal,
                    sum(relevance[memory_id] for memory_id in additions),
                    -len(additions),
                    -action_position,
                )
                ranked_proposals.append({
                    "action": action,
                    "additions": additions,
                    "proposed": proposed,
                    "proposed_order": proposed_order,
                    "utility": proposed_utility,
                    "marginal": marginal,
                    "density": density,
                    "tie_key": tie_key,
                })
            best = None
            for proposal in sorted(
                ranked_proposals, key=lambda item: item["tie_key"], reverse=True
            ):
                proposed_tokens = int(token_cost(proposal["proposed_order"]))
                if (
                    proposed_tokens <= budget
                    and preserves_target_feasibility(proposal["proposed"])
                ):
                    best = proposal
                    best["tokens"] = proposed_tokens
                    break
            if best is None:
                break
            selected = best["proposed"]
            current_tokens = best["tokens"]
            current_utility = best["utility"]
            selection_steps.append({
                "action_type": best["action"]["type"],
                "added_memory_ids": [
                    memory_id
                    for memory_id in ordered_candidates
                    if memory_id in best["additions"]
                ],
                "marginal_utility": best["marginal"],
                "utility_per_added_memory": best["density"],
                "selected_tokens": current_tokens,
                "utility_after": dict(current_utility),
            })

        selected_ids = [
            memory_id for memory_id in ordered_candidates if memory_id in selected
        ]
        selected_paths = [
            path for path in paths if set(path["memory_ids"]).issubset(selected)
        ]
        rejected = []
        for memory_id in ordered_candidates:
            if memory_id in selected:
                continue
            proposed_ids = selected_ids + [memory_id]
            proposed_ids.sort(key=lambda item: positions[item])
            if len(selected_ids) >= target_count:
                reason = "evidence_count"
            elif int(token_cost(proposed_ids)) > budget:
                reason = "token_budget"
            else:
                reason = "no_fitting_action"
            rejected.append({"memory_id": memory_id, "reason": reason})
        return {
            "candidate_memory_ids": ordered_candidates,
            "selected_memory_ids": selected_ids,
            "selected_paths": selected_paths,
            "question_facets": {
                "lexical": facets["lexical"],
                "structural": facets["structural"],
                "weights": facets["weights"],
                "temporal_query": facets["temporal_query"],
                "relation_intents": facets["relation_intents"],
            },
            "candidate_scores": {
                memory_id: {
                    "relevance": relevance[memory_id],
                    "hybrid": normalized_hybrid.get(memory_id, 0.0),
                    "graph": normalized_graph.get(memory_id, 0.0),
                    "fusion": normalized_fusion.get(memory_id, 0.0),
                    "ranks": ranking_positions.get(memory_id, {}),
                    "facet_matches": facet_matches[memory_id],
                }
                for memory_id in ordered_candidates
            },
            "evidence_graph_edges": edge_records,
            "utility": current_utility,
            "token_budget": budget,
            "target_evidence_count": target_count,
            "selected_tokens": current_tokens,
            "selection_steps": selection_steps,
            "rejected_candidates": rejected,
        }

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
        regex_intent_conditioning: bool = True,
    ) -> Dict[str, Any]:
        """Expand temporal evidence and optionally reorder it for the query intent."""
        query = detect_temporal_query(
            question,
            semantic_intents_enabled=regex_intent_conditioning,
        )
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
