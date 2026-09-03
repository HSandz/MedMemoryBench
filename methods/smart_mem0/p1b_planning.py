import json
import re
from typing import Any, Dict, List, Tuple

from methods.smart_mem0.p1b_execution import EvidenceGap


GAP_PLANNER_PROMPT = """You are an Evidence-Gap Planner for a general-domain evidence-grounded memory retrieval agent.
Your task is to decompose a complex query into specific, missing pieces of evidence ("gaps") that are required to answer the question, but are NOT already present in the provided local seed evidence.

DO NOT output a monolithic slot for the whole query.
DO NOT output memory IDs.
DO NOT provide the final answer.
One gap must correspond to one independently retrievable evidence requirement.
Limit to a MAXIMUM of 4 gaps.

VALID ROLES:
FOCAL_TRIGGER: The event that initiated a causal chain.
PRIOR_TRAJECTORY: A prior state establishing a baseline or longitudinal context.
CAUSAL_BRIDGE: Stored evidence connecting a trigger to an outcome.
OUTCOME: The resulting state, symptom, or consequence.
OPTION_SUPPORT: Evidence supporting or contradicting a specific option in a multiple-choice query.
COMPARISON_SIDE: One side of a comparison across time or entities.
GENERIC_EVIDENCE: A simple missing fact if no other role applies.

REQUIRED JSON OUTPUT FORMAT:
{
    "evidence_gaps": [
        {
            "id": "g1",
            "role": "FOCAL_TRIGGER",
            "description": "Event that immediately preceded the delay",
            "required": true,
            "target_entities": ["project alpha", "blockers"],
            "target_property": "status",
            "temporal_axis": "event_time",
            "temporal_relation": "EXACT",
            "temporal_anchor": "2024-01-20",
            "option_label": "A"
        }
    ]
}

TEMPORAL SCHEMA RULES:
- `temporal_axis` MUST ONLY BE ONE OF: "event_time", "document_time", "effective_event_time", or "". Do NOT output ranges or prose here.
- `temporal_relation` MUST BE: "EXACT", "EARLIEST", "LATEST", "BEFORE", "AFTER", "BETWEEN", or "".
- `temporal_anchor` should contain an actual date/range only when it is explicitly supplied by the query.

OPTION RULES:
- If `role` is "OPTION_SUPPORT", provide the specific `option_label` (e.g., "A", "B") this gap supports or refutes.
- Do not invent option text. The executor receives the deterministic option proposition from the query parser.

If the supplied seed evidence already completely answers the semantic part of the question, output an empty list for evidence_gaps. Deterministic structural requirements (state, temporal, comparison, options, causal chains) are enforced separately and cannot be removed by an empty planner response.
"""


def _frame_dates(frame: Any) -> List[str]:
    return [str(value) for value in (getattr(frame, "dates", ()) or ()) if value]


def _frame_entities(frame: Any) -> List[str]:
    return [str(value) for value in (getattr(frame, "entities", ()) or ()) if value]


def _build_qrf(agent, question: str, frame: Any) -> Dict[str, Any]:
    q = question.lower().strip()
    dates = _frame_dates(frame)
    entities = _frame_entities(frame)

    options = {}
    if hasattr(agent, "_question_options"):
        options = agent._question_options(question) or {}

    qrf = {
        "operator": "DIRECT",
        "answer_slot": "TEXT",
        "temporal_axis": "",
        "temporal_relation": "",
        "temporal_anchor": "",
        "temporal_end": "",
        "comparison_sides": [],
        "visible_options": {},
        "required_fields": [],
        "requires_inference": False,
        "target_entities": entities,
    }

    if options:
        qrf["operator"] = "MULTI_OPTION"
        qrf["answer_slot"] = "OPTION_SET"
        qrf["visible_options"] = dict(options)

    elif (
        "compare" in q
        or "vs " in q
        or "versus" in q
        or "difference" in q
        or "compared with" in q
    ):
        qrf["operator"] = "COMPARISON"
        qrf["answer_slot"] = "VALUE"
        if len(dates) >= 2:
            left_anchor, right_anchor = dates[0], dates[1]
        elif len(dates) == 1:
            left_anchor, right_anchor = "", dates[0]
        else:
            left_anchor = right_anchor = ""
        qrf["comparison_sides"] = [
            {"label": "left_side", "temporal_anchor": left_anchor},
            {"label": "right_side", "temporal_anchor": right_anchor},
        ]

    elif (
        "should i" in q
        or "recommend" in q
        or "choice" in q
        or "can i" in q
        or "safe" in q
    ):
        qrf["operator"] = "DECISION"

    elif (
        "why" in q
        or "cause" in q
        or "lead to" in q
        or "result in" in q
        or "related to" in q
        or "lingering effect" in q
    ):
        qrf["operator"] = "CAUSAL"

    else:
        # Interrogative temporal "when" can appear after a leading record/date
        # clause. Subordinate forms such as "when I stand up, should I..." are
        # intentionally excluded because they are not temporal questions.
        interrogative_when = bool(
            re.search(r"\bwhen\s+(?:did|does|do|was|were|is|are|will)\b", q)
        )
        temporal_main_clause = (
            q.startswith("when")
            or interrogative_when
            or "what date" in q
            or "what year" in q
            or q.startswith("how long")
        )

        if temporal_main_clause:
            qrf["operator"] = "TEMPORAL"
            explicit_calendar_request = "what date" in q or "what year" in q
            relative_time_question = bool(
                re.search(r"\bwhen\s+(?:does|do)\b", q)
            ) and not explicit_calendar_request
            qrf["answer_slot"] = (
                "RELATIVE_TIME" if relative_time_question else "DATE"
            )

            if (
                "document" in q
                or "record" in q
                or "note" in q
                or "file" in q
            ):
                qrf["temporal_axis"] = "document_time"
            else:
                qrf["temporal_axis"] = "event_time"

            if "latest" in q or "most recent" in q or "last" in q:
                qrf["temporal_relation"] = "LATEST"
            elif any(
                marker in q
                for marker in (
                    "first appear",
                    "start taking",
                    "first",
                    "earliest",
                    "started",
                    "began",
                    "begun",
                    "first used",
                    "first took",
                    "initially",
                    "onset",
                )
            ):
                qrf["temporal_relation"] = "EARLIEST"
            elif "recent" in q and not dates:
                qrf["temporal_relation"] = "LATEST"

            # Explicit dates are deterministic query constraints, not planner facts.
            if dates:
                qrf["temporal_anchor"] = dates[0]
                if len(dates) > 1:
                    qrf["temporal_end"] = dates[1]
                if not qrf["temporal_relation"]:
                    qrf["temporal_relation"] = (
                        "BETWEEN" if len(dates) > 1 else "EXACT"
                    )

            if qrf["temporal_axis"]:
                qrf["required_fields"].append(qrf["temporal_axis"])

        elif "current" in q or "latest" in q or "now" in q or "present" in q:
            qrf["operator"] = "STATE"
            qrf["answer_slot"] = "VALUE"

    inference_markers = (
        "suggest",
        "imply",
        "reason",
        "explain",
        "does this mean",
        "could this mean",
        "what does this mean",
        "is this serious",
        "is it serious",
    )
    if any(marker in q for marker in inference_markers):
        qrf["requires_inference"] = True

    return qrf


def _make_gap(
    *,
    gap_id: str,
    role: str,
    frame: Any,
    qrf: Dict[str, Any],
    description: str,
    target_entities: List[str] = None,
    target_property: str = "",
    temporal_axis: str = "",
    temporal_relation: str = "",
    temporal_anchor: str = "",
    temporal_end: str = "",
    option_label: str = "",
    option_proposition: str = "",
    comparison_side_label: str = "",
) -> EvidenceGap:
    gap = EvidenceGap(
        id=gap_id,
        role=role,
        required=True,
        subject_id=getattr(frame, "speaker_role", "") or "primary_user",
        target_entities=target_entities or [],
        target_property=target_property,
        temporal_axis=temporal_axis,
        temporal_relation=temporal_relation,
        temporal_anchor=temporal_anchor,
        temporal_end=temporal_end,
        option_label=option_label,
        option_proposition=option_proposition,
        comparison_side_label=comparison_side_label,
    )
    gap.qrf_operator = qrf["operator"]
    gap.description = description
    if qrf.get("required_fields"):
        gap.required_fields = list(qrf["required_fields"])
    return gap


def _ensure_structural_gaps(
    evidence_gaps: List[EvidenceGap],
    qrf: Dict[str, Any],
    frame: Any,
    question: str,
) -> List[EvidenceGap]:
    operator = qrf["operator"]
    entities = _frame_entities(frame)

    if operator == "MULTI_OPTION":
        by_label = {
            gap.option_label: gap
            for gap in evidence_gaps
            if gap.role == "OPTION_SUPPORT" and gap.option_label
        }
        rebuilt = []
        for label, proposition in list(qrf["visible_options"].items())[:4]:
            gap = by_label.get(label)
            if gap is None:
                gap = _make_gap(
                    gap_id=f"g_opt_{label}",
                    role="OPTION_SUPPORT",
                    frame=frame,
                    qrf=qrf,
                    description="",
                    option_label=label,
                )
            gap.role = "OPTION_SUPPORT"
            gap.required = True
            gap.qrf_operator = operator
            gap.option_label = label
            gap.option_proposition = str(proposition)
            gap.description = (
                f"Evidence supporting or refuting option {label}: {proposition}"
            )
            # Query-time proposition contract. This is ephemeral and never durable.
            gap.target_property = str(proposition)
            if not gap.target_entities:
                gap.target_entities = entities
            rebuilt.append(gap)
        return rebuilt

    if operator == "COMPARISON":
        planner_sides = [
            gap for gap in evidence_gaps if gap.role == "COMPARISON_SIDE"
        ][:2]
        rebuilt = []
        side_specs = qrf.get("comparison_sides") or [
            {"label": "left_side", "temporal_anchor": ""},
            {"label": "right_side", "temporal_anchor": ""},
        ]
        for index, spec in enumerate(side_specs[:2]):
            gap = planner_sides[index] if index < len(planner_sides) else None
            if gap is None:
                gap = _make_gap(
                    gap_id=f"g_comp_{index}",
                    role="COMPARISON_SIDE",
                    frame=frame,
                    qrf=qrf,
                    description="",
                )
            gap.role = "COMPARISON_SIDE"
            gap.required = True
            gap.qrf_operator = operator
            gap.comparison_side_label = str(spec.get("label") or f"side_{index}")
            if not gap.target_entities:
                gap.target_entities = entities
            anchor = str(spec.get("temporal_anchor") or "")
            if anchor:
                gap.temporal_axis = gap.temporal_axis or "event_time"
                gap.temporal_relation = "EXACT"
                gap.temporal_anchor = anchor
            gap.description = (
                f"Evidence for comparison {gap.comparison_side_label}"
                + (f" at {anchor}" if anchor else "")
                + f": {question}"
            )
            rebuilt.append(gap)
        return rebuilt

    if evidence_gaps:
        return evidence_gaps[:4]

    if operator == "STATE":
        return [
            _make_gap(
                gap_id="g_state",
                role="GENERIC_EVIDENCE",
                frame=frame,
                qrf=qrf,
                description=f"Current/latest state required to answer: {question}",
                target_entities=entities,
            )
        ]

    if operator == "TEMPORAL":
        return [
            _make_gap(
                gap_id="g_temporal",
                role="GENERIC_EVIDENCE",
                frame=frame,
                qrf=qrf,
                description=f"Temporal evidence required to answer: {question}",
                target_entities=entities,
                temporal_axis=qrf["temporal_axis"],
                temporal_relation=qrf["temporal_relation"] or "EXACT",
                temporal_anchor=qrf["temporal_anchor"],
                temporal_end=qrf["temporal_end"],
            )
        ]

    if operator == "CAUSAL":
        return [
            _make_gap(
                gap_id="g_trigger",
                role="FOCAL_TRIGGER",
                frame=frame,
                qrf=qrf,
                description=f"Trigger/exposure relevant to: {question}",
                target_entities=entities,
            ),
            _make_gap(
                gap_id="g_bridge",
                role="CAUSAL_BRIDGE",
                frame=frame,
                qrf=qrf,
                description=f"Stored causal bridge relevant to: {question}",
                target_entities=entities,
            ),
            _make_gap(
                gap_id="g_outcome",
                role="OUTCOME",
                frame=frame,
                qrf=qrf,
                description=f"Outcome/state relevant to: {question}",
                target_entities=entities,
            ),
        ]

    if operator == "DECISION" or qrf.get("requires_inference"):
        return [
            _make_gap(
                gap_id="g_focal",
                role="FOCAL_TRIGGER",
                frame=frame,
                qrf=qrf,
                description=f"Current focal evidence for decision/inference: {question}",
                target_entities=entities,
            ),
            _make_gap(
                gap_id="g_history",
                role="PRIOR_TRAJECTORY",
                frame=frame,
                qrf=qrf,
                description=f"Relevant prior trajectory or constraint for: {question}",
                target_entities=entities,
            ),
        ]

    return evidence_gaps


def _gap_planner(
    agent,
    question: str,
    seeds: List[Dict[str, Any]],
    frame: Any,
) -> Tuple[List[EvidenceGap], Dict[str, Any]]:
    qrf = _build_qrf(agent, question, frame)

    compact_seeds = []
    for seed in seeds[:3]:
        compact_seeds.append(
            {
                "id": seed["id"],
                "kind": seed.get("kind"),
                "role": seed.get("semantic_role"),
                "object": seed.get("object_anchor"),
                "value": seed.get("value"),
                "claim": seed.get("claim"),
                "event_time": seed.get("event_time"),
                "document_time": seed.get("document_time"),
            }
        )

    context = (
        f"Question: {question}\n\n"
        f"Already-covered seed evidence:\n{json.dumps(compact_seeds, indent=2)}"
    )

    response = agent._llm_client.chat(
        [
            {"role": "system", "content": GAP_PLANNER_PROMPT},
            {"role": "user", "content": context},
        ],
        temperature=0.0,
        response_format={"type": "json_object"},
    )

    usage = agent._response_usage(response, GAP_PLANNER_PROMPT + context)

    try:
        parsed = json.loads(response.content)
    except (json.JSONDecodeError, TypeError):
        parsed = {"evidence_gaps": []}

    gaps_data = parsed.get("evidence_gaps", [])
    if not isinstance(gaps_data, list):
        gaps_data = []
    gaps_data = gaps_data[:4]

    valid_roles = {
        "FOCAL_TRIGGER",
        "PRIOR_TRAJECTORY",
        "CAUSAL_BRIDGE",
        "OUTCOME",
        "OPTION_SUPPORT",
        "COMPARISON_SIDE",
        "GENERIC_EVIDENCE",
    }
    valid_axes = {"event_time", "document_time", "effective_event_time", ""}
    valid_relations = {
        "EXACT",
        "EARLIEST",
        "LATEST",
        "BEFORE",
        "AFTER",
        "BETWEEN",
        "",
    }

    evidence_gaps = []
    used_ids = set()
    option_keys = list((qrf.get("visible_options") or {}).keys())

    for index, raw_gap in enumerate(gaps_data):
        if not isinstance(raw_gap, dict):
            continue

        gap_id = str(raw_gap.get("id", f"gap_{index}"))
        if gap_id in used_ids:
            gap_id = f"{gap_id}_{index}"
        used_ids.add(gap_id)

        role = str(raw_gap.get("role", "GENERIC_EVIDENCE")).upper()
        if role not in valid_roles:
            role = "GENERIC_EVIDENCE"

        temp_axis = str(raw_gap.get("temporal_axis", ""))
        if temp_axis not in valid_axes:
            temp_axis = ""

        temp_relation = str(raw_gap.get("temporal_relation", "")).upper()
        if temp_relation not in valid_relations:
            temp_relation = ""

        temp_anchor = str(raw_gap.get("temporal_anchor", ""))
        temp_end = ""

        if qrf["operator"] == "TEMPORAL":
            temp_axis = qrf["temporal_axis"]
            temp_relation = qrf["temporal_relation"] or temp_relation
            temp_anchor = qrf["temporal_anchor"] or temp_anchor
            temp_end = qrf["temporal_end"]

        if qrf["operator"] == "STATE":
            temp_axis = ""
            temp_relation = ""
            temp_anchor = ""
            temp_end = ""

        option_label = str(raw_gap.get("option_label", ""))
        if qrf["operator"] == "MULTI_OPTION" and option_keys:
            if option_label not in option_keys:
                option_label = option_keys[len(evidence_gaps) % len(option_keys)]

        proposition = ""
        if option_label and qrf["operator"] == "MULTI_OPTION":
            proposition = str(qrf["visible_options"].get(option_label, ""))

        gap = EvidenceGap(
            id=gap_id,
            role=role,
            required=bool(raw_gap.get("required", True)),
            subject_id=getattr(frame, "speaker_role", "") or "primary_user",
            target_entities=raw_gap.get("target_entities", []) or [],
            target_property=raw_gap.get("target_property", "") or "",
            temporal_axis=temp_axis,
            temporal_relation=temp_relation,
            temporal_anchor=temp_anchor,
            temporal_end=temp_end,
            option_label=option_label,
            option_proposition=proposition,
        )
        gap.qrf_operator = qrf["operator"]
        gap.description = str(raw_gap.get("description", ""))

        if qrf.get("required_fields"):
            gap.required_fields = list(qrf["required_fields"])

        if proposition:
            gap.description = (
                f"Evidence supporting or refuting option {option_label}: {proposition}"
            )
            gap.target_property = proposition

        evidence_gaps.append(gap)

    evidence_gaps = _ensure_structural_gaps(
        evidence_gaps,
        qrf,
        frame,
        question,
    )
    return evidence_gaps[:4], usage
