"""Conservative, deterministic temporal helpers for query-time retrieval."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime
from typing import Optional

from .schemas import Claim, Episode


_DATE_PATTERN = r"(?<!\d)\d{4}[-/]\d{2}[-/]\d{2}(?!\d)"
_MONTH_NAMES = (
    "january|february|march|april|may|june|july|august|september|"
    "october|november|december"
)
_MONTH_ABBREVIATIONS = "jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec"
_MONTH_PATTERN = rf"(?:{_MONTH_NAMES}|{_MONTH_ABBREVIATIONS})"
_MONTH_NAME_DATE_PATTERN = rf"\b(?:{_MONTH_PATTERN})\s+\d{{1,2}},\s*\d{{4}}\b"
_DAY_MONTH_DATE_PATTERN = rf"\b\d{{1,2}}\s+(?:{_MONTH_PATTERN})\s+\d{{4}}\b"
_QUERY_DATE_PATTERN = (
    rf"(?:{_DATE_PATTERN}|{_MONTH_NAME_DATE_PATTERN}|{_DAY_MONTH_DATE_PATTERN})"
)


@dataclass(frozen=True)
class TemporalQueryConstraint:
    """An explicit date bound parsed from a user question, or nothing."""

    kind: str
    target_date: Optional[date] = None
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    intent: str = "hybrid"


@dataclass(frozen=True)
class TemporalMatch:
    """A deterministic temporal candidate contribution."""

    score: float
    match_type: str


def parse_stored_date(value: object) -> Optional[date]:
    """Parse an ISO-like stored timestamp without guessing incomplete dates."""
    if not isinstance(value, str):
        return None
    match = re.search(_DATE_PATTERN, value.strip())
    if not match:
        return None
    try:
        return date.fromisoformat(match.group(0).replace("/", "-"))
    except ValueError:
        return None


def _query_dates(question: str) -> list[date]:
    dates = []
    for match in re.finditer(_QUERY_DATE_PATTERN, question, re.IGNORECASE):
        value = match.group(0)
        try:
            if re.fullmatch(_DATE_PATTERN, value):
                dates.append(date.fromisoformat(value.replace("/", "-")))
                continue
            for pattern in ("%B %d, %Y", "%b %d, %Y", "%d %B %Y", "%d %b %Y"):
                try:
                    dates.append(datetime.strptime(value, pattern).date())
                    break
                except ValueError:
                    continue
            else:
                return []
        except ValueError:
            return []
    return dates


def parse_temporal_query(question: str) -> Optional[TemporalQueryConstraint]:
    """Recognize explicit complete calendar dates and bounded cue phrases."""
    normalized = " ".join(question.casefold().split())
    dates = _query_dates(normalized)
    if not dates:
        return None

    interval = re.search(
        rf"\b(?:between|from)\s+{_QUERY_DATE_PATTERN}\s+(?:and|to)\s+{_QUERY_DATE_PATTERN}\b",
        normalized,
    )
    if interval:
        start, end = _query_dates(interval.group(0))
        if start <= end:
            return TemporalQueryConstraint("interval", start_date=start, end_date=end, intent="hybrid")
        return None
    if len(dates) != 1:
        return None
    target = dates[0]
    if re.search(rf"\b(?:as of|by)\s+{_QUERY_DATE_PATTERN}\b", normalized):
        return TemporalQueryConstraint("as_of", target_date=target, intent="valid_time")
    if re.search(rf"\bbefore\s+{_QUERY_DATE_PATTERN}\b", normalized):
        return TemporalQueryConstraint("before", target_date=target, intent="hybrid")
    if re.search(rf"\bafter\s+{_QUERY_DATE_PATTERN}\b", normalized):
        return TemporalQueryConstraint("after", target_date=target, intent="hybrid")
    if re.search(rf"\b(?:record dated|record on|discuss(?:ed)? on)\s+{_QUERY_DATE_PATTERN}\b", normalized):
        return TemporalQueryConstraint("exact_record_time", target_date=target, intent="record_time")
    # A bare date or "on DATE" has no reliable axis; use both candidate kinds.
    return TemporalQueryConstraint("exact_record_time", target_date=target, intent="hybrid")


def _date_matches(value: date, constraint: TemporalQueryConstraint) -> bool:
    if constraint.kind == "exact_record_time":
        return value == constraint.target_date
    if constraint.kind == "interval":
        return bool(constraint.start_date <= value <= constraint.end_date)
    if constraint.kind == "before":
        return bool(value < constraint.target_date)
    if constraint.kind == "after":
        return bool(value > constraint.target_date)
    if constraint.kind == "as_of":
        return bool(value <= constraint.target_date)
    return False


def episode_temporal_match(episode: Episode, constraint: TemporalQueryConstraint) -> Optional[TemporalMatch]:
    """Match immutable episode record time against an explicit query bound."""
    if constraint.intent == "valid_time":
        return None
    recorded_at = parse_stored_date(episode.recorded_at)
    if recorded_at is None or not _date_matches(recorded_at, constraint):
        return None
    return TemporalMatch(0.5 if constraint.kind == "as_of" else 1.0, "as_of_fallback" if constraint.kind == "as_of" else "exact_record_time")


def _evidence_record_match(claim: Claim, episodes: dict[str, Episode], constraint: TemporalQueryConstraint) -> bool:
    return any(
        (episode := episodes.get(ref.episode_id)) is not None
        and (recorded_at := parse_stored_date(episode.recorded_at)) is not None
        and _date_matches(recorded_at, constraint)
        for ref in claim.evidence
    )


def _valid_interval_contains(claim: Claim, target: date) -> Optional[TemporalMatch]:
    """Resolve stored valid intervals for historical visibility or known hybrid matches."""
    valid_from = parse_stored_date(claim.valid_from)
    valid_to = parse_stored_date(claim.valid_to)
    if valid_from is not None and valid_from > target:
        return None
    # Stored state intervals are half-open: [valid_from, valid_to).
    if valid_to is not None and target >= valid_to:
        return None
    if valid_from is not None or valid_to is not None:
        # A non-state claim with only a start date is a bounded observation,
        # not an indefinitely continuing fact.
        if claim.persistence != "state" and valid_to is None and valid_from != target:
            return None
        return TemporalMatch(1.0, "valid_interval")
    return TemporalMatch(0.5, "as_of_fallback")


def claim_visible_as_of(claim: Claim, target: date) -> Optional[TemporalMatch]:
    """Return knowledge-as-of eligibility; unknown record time is not assumed safe."""
    recorded_at = parse_stored_date(claim.recorded_at)
    if recorded_at is None or recorded_at > target:
        return None
    return _valid_interval_contains(claim, target)


def _interval_overlaps(claim: Claim, start: date, end: date) -> bool:
    valid_from = parse_stored_date(claim.valid_from)
    valid_to = parse_stored_date(claim.valid_to)
    if valid_from is None and valid_to is None:
        return False
    return (valid_to is None or start < valid_to) and (valid_from is None or valid_from <= end)


def claim_temporal_match(claim: Claim, episodes: dict[str, Episode], constraint: TemporalQueryConstraint) -> Optional[TemporalMatch]:
    """Match claim valid time and its own evidence, never unrelated episodes."""
    if constraint.kind == "as_of":
        return claim_visible_as_of(claim, constraint.target_date)  # type: ignore[arg-type]
    if _evidence_record_match(claim, episodes, constraint):
        return TemporalMatch(1.0, "evidence_record_time")
    if constraint.kind == "exact_record_time" and constraint.intent == "hybrid":
        valid_from = parse_stored_date(claim.valid_from)
        valid_to = parse_stored_date(claim.valid_to)
        if valid_from is not None or valid_to is not None:
            return _valid_interval_contains(claim, constraint.target_date)  # type: ignore[arg-type]
    if constraint.kind == "interval" and _interval_overlaps(claim, constraint.start_date, constraint.end_date):  # type: ignore[arg-type]
        return TemporalMatch(1.0, "valid_interval")
    recorded_at = parse_stored_date(claim.recorded_at)
    if recorded_at is not None and _date_matches(recorded_at, constraint):
        return TemporalMatch(1.0, "exact_record_time")
    return None
