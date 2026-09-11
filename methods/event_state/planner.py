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
_QUERY_ROLES = {"target", "support", "anchor"}
_QUERY_AXES = {"none", "event", "record", "knowledge"}
_QUERY_RELATIONS = {"none", "overlap", "before", "after", "as_of", "latest", "earliest"}
_QUERY_PRECISIONS = {"exact", "bounded", "approximate", "unknown"}


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


@dataclass(frozen=True)
class QuerySearch:
    query: str
    role: str


@dataclass(frozen=True)
class QueryTemporal:
    axis: str = "none"
    relation: str = "none"
    start: Optional[date] = None
    end: Optional[date] = None
    anchor_search: Optional[int] = None
    precision: str = "unknown"

    def to_dict(self) -> Dict[str, Any]:
        return {"axis": self.axis, "relation": self.relation,
                "start": self.start.isoformat() if self.start else None,
                "end": self.end.isoformat() if self.end else None,
                "anchor_search": self.anchor_search, "precision": self.precision}


@dataclass(frozen=True)
class QueryPlan:
    searches: List[QuerySearch]
    temporal: QueryTemporal
    state_view: str
    duplicate_search_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {"searches": [{"query": item.query, "role": item.role} for item in self.searches],
                "temporal": self.temporal.to_dict(), "state_view": self.state_view}


def fallback_query_plan(question: str, state_view: str = "current") -> QueryPlan:
    """The caller always retains the raw question; this represents no compiler constraints."""
    return QueryPlan([], QueryTemporal(), state_view)


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


def query_compiler_json_schema() -> Dict[str, Any]:
    """Provider-side guard matching the locally canonicalized compiler shape."""
    return {
        "type": "object",
        "properties": {
            "searches": {"type": "array", "items": {"type": "object", "properties": {
                "query": {"type": "string"}, "role": {"type": "string", "enum": sorted(_QUERY_ROLES)},
            }, "required": ["query", "role"]}},
            "temporal": {"type": "object", "properties": {
                "axis": {"type": "string", "enum": sorted(_QUERY_AXES)}, "relation": {"type": "string", "enum": sorted(_QUERY_RELATIONS)},
                "start": {"type": ["string", "null"]}, "end": {"type": ["string", "null"]},
                "anchor_search": {"type": ["integer", "null"]},
                "precision": {"type": "string", "enum": sorted(_QUERY_PRECISIONS)},
            }, "required": ["axis", "relation", "start", "end", "anchor_search", "precision"]},
            "state_view": {"type": "string", "enum": sorted(_STATE_VIEWS)},
        },
        "required": ["searches", "temporal", "state_view"],
    }


def salvage_query_compiler_output(value: Any, max_searches: int) -> tuple[QueryPlan, List[str], bool]:
    """Validate independent plan components without inventing semantics.

    A parsed object is useful even if one component is malformed: the visible
    question remains channel zero, so dropping an unsafe expansion or date is
    preferable to discarding valid sibling fields.
    """
    if not isinstance(value, dict):
        raise ValueError("query compiler output must be an object")
    raw_searches = value.get("searches")
    warnings: List[str] = []
    salvage_used = False
    if not isinstance(raw_searches, list):
        raw_searches = []
        warnings.append("invalid_searches")
        salvage_used = True
    searches: List[QuerySearch] = []
    seen = set()
    duplicates = 0
    for raw in raw_searches[:max(0, int(max_searches))]:
        if not isinstance(raw, dict) or raw.get("role") not in _QUERY_ROLES:
            warnings.append("invalid_search")
            salvage_used = True
            continue
        query = raw.get("query")
        if not isinstance(query, str) or not (query := " ".join(query.split())):
            warnings.append("invalid_search")
            salvage_used = True
            continue
        if len(query) > MAX_REQUEST_QUERY_LENGTH:
            warnings.append("search_too_long")
            salvage_used = True
            continue
        key = query.casefold()
        if key in seen:
            duplicates += 1
            warnings.append("duplicate_search")
            salvage_used = True
            continue
        seen.add(key)
        searches.append(QuerySearch(query, raw["role"]))
    if len(raw_searches) > max_searches:
        warnings.append("search_limit_exceeded")
        salvage_used = True

    state_view = value.get("state_view")
    if state_view not in _STATE_VIEWS:
        state_view = "current"
        warnings.append("invalid_state_view")
        salvage_used = True

    raw_temporal = value.get("temporal")
    temporal = QueryTemporal()
    if not isinstance(raw_temporal, dict):
        warnings.append("invalid_temporal")
        salvage_used = True
    else:
        axis, relation = raw_temporal.get("axis"), raw_temporal.get("relation")
        precision = raw_temporal.get("precision")
        try:
            if axis not in _QUERY_AXES or relation not in _QUERY_RELATIONS or precision not in _QUERY_PRECISIONS:
                raise ValueError("enum")
            start_raw, end_raw = raw_temporal.get("start"), raw_temporal.get("end")
            if (start_raw is not None and not isinstance(start_raw, str)) or (end_raw is not None and not isinstance(end_raw, str)):
                raise ValueError("date_type")
            start = _parse_date(start_raw) if start_raw else None
            end = _parse_date(end_raw) if end_raw else None
            if start and end and start > end:
                raise ValueError("reversed_interval")
            if precision == "exact" and start and end and start != end:
                precision = "bounded"
                warnings.append("exact_interval_canonicalized")
                salvage_used = True
            anchor = raw_temporal.get("anchor_search")
            if anchor is not None:
                if not isinstance(anchor, int) or isinstance(anchor, bool) or not 0 <= anchor < len(searches):
                    raise ValueError("invalid_anchor")
                if searches[anchor].role != "anchor":
                    searches[anchor] = QuerySearch(searches[anchor].query, "anchor")
                    warnings.append("anchor_role_canonicalized")
                    salvage_used = True
            if axis == "none" and relation != "none":
                raise ValueError("none_axis_relation")
            if relation in {"before", "after"} and not (anchor is not None or start or end):
                raise ValueError("unbound_ordering")
            if relation == "overlap" and (start is None or end is None):
                raise ValueError("incomplete_overlap")
            if relation == "as_of" and (axis != "knowledge" or end is None or state_view != "as_of"):
                raise ValueError("invalid_as_of")
            if axis == "knowledge" and relation != "as_of":
                raise ValueError("invalid_knowledge")
            temporal = QueryTemporal(axis, relation, start, end, anchor, precision)
        except ValueError as exc:
            warnings.append(f"temporal_{str(exc)}")
            salvage_used = True
    if state_view == "as_of" and temporal.relation != "as_of":
        state_view = "current"
        warnings.append("as_of_without_knowledge_constraint")
        salvage_used = True
    return QueryPlan(searches, temporal, state_view, duplicates), sorted(set(warnings)), salvage_used


def validate_query_compiler_output(value: Any, max_searches: int) -> QueryPlan:
    """Compatibility wrapper returning the locally canonicalized plan."""
    return salvage_query_compiler_output(value, max_searches)[0]
