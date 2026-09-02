"""Validation helpers for the optional Event-State query planner."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Dict, List, Optional, Tuple

from .temporal import TemporalQueryConstraint


MAX_REQUEST_QUERY_LENGTH = 500
_SOURCES = {"claims", "episodes", "both"}
_STATE_VIEWS = {"current", "all_versions", "as_of"}
_TIME_MODES = {
    "record_exact",
    "record_before",
    "record_after",
    "record_interval",
    "knowledge_as_of",
}


@dataclass(frozen=True)
class PlannerRequest:
    query: str
    sources: str
    state_view: str
    temporal_constraint: Optional[TemporalQueryConstraint] = None

    def key(self) -> Tuple[Any, ...]:
        temporal = self.temporal_constraint
        return (
            self.query,
            self.sources,
            self.state_view,
            temporal.kind if temporal else None,
            temporal.target_date.isoformat() if temporal and temporal.target_date else None,
            temporal.start_date.isoformat() if temporal and temporal.start_date else None,
            temporal.end_date.isoformat() if temporal and temporal.end_date else None,
        )

    def to_dict(self) -> Dict[str, Any]:
        temporal = self.temporal_constraint
        return {
            "query": self.query,
            "sources": self.sources,
            "state_view": self.state_view,
            "time": (
                None
                if temporal is None
                else {
                    "mode": {
                        "exact_record_time": "record_exact",
                        "before": "record_before",
                        "after": "record_after",
                        "interval": "record_interval",
                        "as_of": "knowledge_as_of",
                    }[temporal.kind],
                    "date": temporal.target_date.isoformat() if temporal.target_date else None,
                    "start": temporal.start_date.isoformat() if temporal.start_date else None,
                    "end": temporal.end_date.isoformat() if temporal.end_date else None,
                }
            ),
        }


@dataclass(frozen=True)
class PlannerDecision:
    action: str
    answer: Optional[str]
    requests: List[PlannerRequest]
    invalid_request_count: int = 0
    duplicate_request_count: int = 0


def _parse_date(value: Any) -> date:
    if not isinstance(value, str) or not value:
        raise ValueError("date must be an ISO date")
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError("date must be a valid YYYY-MM-DD") from exc


def _parse_request(value: Any) -> PlannerRequest:
    if not isinstance(value, dict):
        raise ValueError("request must be an object")
    query = value.get("query")
    if not isinstance(query, str) or not query.strip():
        raise ValueError("request query must be non-empty")
    query = " ".join(query.split())
    if len(query) > MAX_REQUEST_QUERY_LENGTH:
        raise ValueError("request query is too long")
    sources = value.get("sources")
    state_view = value.get("state_view")
    if sources not in _SOURCES or state_view not in _STATE_VIEWS:
        raise ValueError("invalid request source or state view")
    raw_time = value.get("time")
    if raw_time is None:
        return PlannerRequest(query, sources, state_view)
    if not isinstance(raw_time, dict) or raw_time.get("mode") not in _TIME_MODES:
        raise ValueError("invalid temporal mode")
    mode = raw_time["mode"]
    target = start = end = None
    if mode in {"record_exact", "record_before", "record_after", "knowledge_as_of"}:
        target = _parse_date(raw_time.get("date"))
    elif mode == "record_interval":
        start, end = _parse_date(raw_time.get("start")), _parse_date(raw_time.get("end"))
        if start > end:
            raise ValueError("interval start must not exceed end")
    if mode == "knowledge_as_of" and state_view != "as_of":
        raise ValueError("knowledge_as_of requires state_view=as_of")
    kind = {
        "record_exact": "exact_record_time",
        "record_before": "before",
        "record_after": "after",
        "record_interval": "interval",
        "knowledge_as_of": "as_of",
    }[mode]
    temporal = TemporalQueryConstraint(kind, target_date=target, start_date=start, end_date=end, intent="record_time")
    return PlannerRequest(query, sources, state_view, temporal)


def validate_planner_output(value: Any, max_requests: int) -> PlannerDecision:
    """Validate untrusted planner JSON without repairing semantic content."""
    if not isinstance(value, dict) or value.get("action") not in {"answer", "retrieve"}:
        raise ValueError("planner output must contain action=answer or retrieve")
    action = value["action"]
    answer = value.get("answer")
    raw_requests = value.get("requests")
    if not isinstance(raw_requests, list):
        raise ValueError("planner requests must be a list")
    if action == "answer":
        if not isinstance(answer, str) or not answer.strip() or raw_requests:
            raise ValueError("answer action requires a non-empty answer and no requests")
        return PlannerDecision("answer", answer.strip(), [])
    if answer is not None:
        raise ValueError("retrieve action requires answer=null")
    valid: List[PlannerRequest] = []
    invalid = 0
    duplicates = 0
    seen = set()
    for item in raw_requests:
        try:
            request = _parse_request(item)
        except ValueError:
            invalid += 1
            continue
        if request.key() in seen:
            duplicates += 1
            continue
        seen.add(request.key())
        if len(valid) < max_requests:
            valid.append(request)
        else:
            invalid += 1
    return PlannerDecision("retrieve", None, valid, invalid, duplicates)
