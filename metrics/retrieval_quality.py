"""Retrieval-quality metrics based on gold and retrieved source sessions."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence


RETRIEVAL_QUALITY_GROUP = "retrieval_quality"


def _normalize_session_id(value: Any) -> Optional[str]:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    normalized = str(value).strip()
    return normalized or None


def _unique_session_ids(values: Iterable[Any]) -> List[str]:
    session_ids: List[str] = []
    seen = set()
    for value in values:
        session_id = _normalize_session_id(value)
        if session_id is None or session_id in seen:
            continue
        seen.add(session_id)
        session_ids.append(session_id)
    return session_ids


def gold_session_ids(
    source_key_points: Optional[Sequence[Dict[str, Any]]],
    metadata: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """Return deduplicated gold source-session IDs in dataset order."""
    values: List[Any] = []
    for key_point in source_key_points or []:
        if isinstance(key_point, dict):
            values.append(
                key_point.get("session_id", key_point.get("source_session_id"))
            )

    metadata = metadata if isinstance(metadata, dict) else {}
    for field_name in (
        "source_session_ids",
        "gold_session_ids",
        "ground_truth_session_ids",
    ):
        field_value = metadata.get(field_name)
        if isinstance(field_value, (list, tuple, set)):
            values.extend(field_value)
        elif field_value is not None:
            values.append(field_value)
    for field_name in ("ground_truth", "gold", "reference"):
        ground_truth = metadata.get(field_name)
        if not isinstance(ground_truth, dict):
            continue
        for id_field in (
            "session_id",
            "source_session_id",
            "session_ids",
            "source_session_ids",
        ):
            field_value = ground_truth.get(id_field)
            if isinstance(field_value, (list, tuple, set)):
                values.extend(field_value)
            elif field_value is not None:
                values.append(field_value)
        for key_point in ground_truth.get("source_key_points", []):
            if isinstance(key_point, dict):
                values.append(
                    key_point.get("session_id", key_point.get("source_session_id"))
                )
    return _unique_session_ids(values)


def _record_session_ids(record: Dict[str, Any]) -> List[str]:
    values: List[Any] = [
        record.get("source_session_id", record.get("session_id"))
    ]

    provenance = record.get("provenance")
    if isinstance(provenance, dict):
        values.append(
            provenance.get("source_session_id", provenance.get("session_id"))
        )

    for evidence_field in ("raw_evidence_added", "provenance_evidence"):
        evidence_items = record.get(evidence_field)
        if not isinstance(evidence_items, list):
            continue
        for item in evidence_items:
            if not isinstance(item, dict):
                continue
            evidence = item.get("evidence", item)
            if isinstance(evidence, dict):
                values.append(evidence.get("source_session_id"))

    retrieval_audit = record.get("retrieval_audit")
    if isinstance(retrieval_audit, dict):
        for item in retrieval_audit.get("provenance_evidence", []):
            if not isinstance(item, dict):
                continue
            evidence = item.get("evidence", item)
            if isinstance(evidence, dict):
                values.append(evidence.get("source_session_id"))
    return _unique_session_ids(values)


def retrieved_session_ids(
    retrieved_memories: Optional[Sequence[Dict[str, Any]]],
) -> List[str]:
    """Return unique retrieved session IDs in first-retrieved order."""
    values: List[Any] = []
    for record in retrieved_memories or []:
        if isinstance(record, dict):
            values.extend(_record_session_ids(record))
    return _unique_session_ids(values)


def compute_session_retrieval_quality(
    retrieved_memories: Optional[Sequence[Dict[str, Any]]],
    source_key_points: Optional[Sequence[Dict[str, Any]]],
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Compute set- and rank-based retrieval quality at session granularity."""
    predicted_ids = retrieved_session_ids(retrieved_memories)
    expected_ids = gold_session_ids(source_key_points, metadata)
    predicted_set = set(predicted_ids)
    expected_set = set(expected_ids)
    matched_ids = [item for item in predicted_ids if item in expected_set]
    false_positive_ids = [item for item in predicted_ids if item not in expected_set]
    missed_ids = [item for item in expected_ids if item not in predicted_set]

    retrieved_records = [
        record for record in retrieved_memories or [] if isinstance(record, dict)
    ]
    identified_records = sum(
        bool(_record_session_ids(record)) for record in retrieved_records
    )
    result: Dict[str, Any] = {
        "level": "session",
        "unit": "unique_session_id",
        "available": bool(expected_ids),
        "predicted_session_ids": predicted_ids,
        "gold_session_ids": expected_ids,
        "matched_session_ids": matched_ids,
        "false_positive_session_ids": false_positive_ids,
        "missed_session_ids": missed_ids,
        "retrieved_note_count": len(retrieved_records),
        "retrieved_notes_with_session_id": identified_records,
        "predicted_count": len(predicted_ids),
        "gold_count": len(expected_ids),
        "true_positive_count": len(matched_ids),
        "false_positive_count": len(false_positive_ids),
        "false_negative_count": len(missed_ids),
    }
    unavailable_reason = None
    if not expected_ids:
        unavailable_reason = "no gold source session IDs"
    elif retrieved_records and identified_records != len(retrieved_records):
        unavailable_reason = "one or more retrieved notes lack source session IDs"
    if unavailable_reason:
        result["available"] = False
        result.update({
            "unavailable_reason": unavailable_reason,
            "precision": None,
            "recall": None,
            "f1": None,
            "average_precision": None,
            "reciprocal_rank": None,
            "hit": None,
            "exact_match": None,
        })
        return result

    precision = len(matched_ids) / len(predicted_ids) if predicted_ids else 0.0
    recall = len(matched_ids) / len(expected_ids)
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )
    relevant_seen = 0
    precision_sum = 0.0
    first_relevant_rank = None
    for rank, session_id in enumerate(predicted_ids, start=1):
        if session_id not in expected_set:
            continue
        relevant_seen += 1
        precision_sum += relevant_seen / rank
        if first_relevant_rank is None:
            first_relevant_rank = rank

    result.update({
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "average_precision": precision_sum / len(expected_ids),
        "reciprocal_rank": 1.0 / first_relevant_rank if first_relevant_rank else 0.0,
        "hit": bool(matched_ids),
        "exact_match": predicted_set == expected_set,
    })
    return result


def aggregate_session_retrieval_quality(results: Sequence[Any]) -> Dict[str, Any]:
    """Aggregate per-query session retrieval groups with macro and micro scores."""
    by_query_type: Dict[str, List[Dict[str, Any]]] = {}
    total_with_group = 0
    for result in results:
        details = getattr(result, "details", {})
        metric_groups = (
            details.get("metric_groups") if isinstance(details, dict) else None
        )
        group = (
            metric_groups.get(RETRIEVAL_QUALITY_GROUP)
            if isinstance(metric_groups, dict) else None
        )
        if not isinstance(group, dict):
            continue
        total_with_group += 1
        by_query_type.setdefault(
            str(getattr(result, "query_type", "unknown")), []
        ).append(group)

    if total_with_group == 0:
        return {}

    def summarize(items: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        available_items = [item for item in items if item.get("available", False)]
        total_items = len(items)
        evaluated = len(available_items)
        tp = sum(
            int(item.get("true_positive_count", 0)) for item in available_items
        )
        fp = sum(
            int(item.get("false_positive_count", 0)) for item in available_items
        )
        fn = sum(
            int(item.get("false_negative_count", 0)) for item in available_items
        )
        micro_precision = tp / (tp + fp) if tp + fp else 0.0
        micro_recall = tp / (tp + fn) if tp + fn else 0.0
        micro_f1 = (
            2.0 * micro_precision * micro_recall / (micro_precision + micro_recall)
            if micro_precision + micro_recall else 0.0
        )

        def average(field_name: str) -> float:
            return (
                sum(
                    float(item.get(field_name, 0.0))
                    for item in available_items
                ) / evaluated
                if evaluated else 0.0
            )

        return {
            "total_queries": total_items,
            "evaluated_queries": evaluated,
            "unavailable_queries": total_items - evaluated,
            "coverage": evaluated / total_items if total_items else 0.0,
            "macro_precision": average("precision"),
            "macro_recall": average("recall"),
            "macro_f1": average("f1"),
            "mean_average_precision": average("average_precision"),
            "mean_reciprocal_rank": average("reciprocal_rank"),
            "hit_rate": average("hit"),
            "exact_match_rate": average("exact_match"),
            "avg_predicted_sessions": average("predicted_count"),
            "avg_gold_sessions": average("gold_count"),
            "micro": {
                "true_positive_count": tp,
                "false_positive_count": fp,
                "false_negative_count": fn,
                "precision": micro_precision,
                "recall": micro_recall,
                "f1": micro_f1,
            },
        }

    summary = summarize([
        group for type_groups in by_query_type.values() for group in type_groups
    ])
    summary.update({
        "level": "session",
        "unit": "unique_session_id",
        "by_query_type": {
            query_type: summarize(type_groups)
            for query_type, type_groups in sorted(by_query_type.items())
        },
    })
    return summary
