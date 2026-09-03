import json
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
- `temporal_anchor` should contain the actual date/range string (e.g., "2024-01-20" or "recent").

OPTION RULES:
- If `role` is "OPTION_SUPPORT", you MUST provide the specific `option_label` (e.g., "A", "B") this gap supports or refutes.

If the supplied seed evidence already completely answers the question, output an empty list for evidence_gaps.
"""

def _build_qrf(agent, question: str, frame: Any) -> Dict[str, Any]:
    q = question.lower()
    
    # Use agent's deterministic option parser
    options = {}
    if hasattr(agent, "_question_options"):
        options = agent._question_options(question)
        
    qrf = {
        "operator": "DIRECT",
        "answer_slot": "TEXT",
        "temporal_axis": "",
        "temporal_relation": "",
        "temporal_anchor": "",
        "comparison_sides": [],
        "visible_options": [],
        "required_fields": []
    }
    
    # 1. Visible options
    if options:
        qrf["operator"] = "MULTI_OPTION"
        qrf["answer_slot"] = "OPTION_SET"
        qrf["visible_options"] = list(options.keys())
    # 2. Explicit comparison
    elif "compare" in q or "vs " in q or "versus" in q or "difference" in q or "compared with" in q:
        qrf["operator"] = "COMPARISON"
        qrf["comparison_sides"] = ["left_side", "right_side"]
        qrf["answer_slot"] = "VALUE"
    # 3. DATE requested
    elif "when" in q or "what date" in q or "what year" in q:
        qrf["operator"] = "TEMPORAL"
        qrf["answer_slot"] = "DATE"
        if "document" in q or "record" in q or "note" in q or "file" in q:
            qrf["temporal_axis"] = "document_time"
        else:
            qrf["temporal_axis"] = "event_time"
        
        if "latest" in q or "most recent" in q or "last" in q:
            qrf["temporal_relation"] = "LATEST"
        elif "first" in q or "earliest" in q:
            qrf["temporal_relation"] = "EARLIEST"
        elif "recent" in q:
            qrf["temporal_relation"] = "LATEST"
            
        if qrf["temporal_axis"]:
            qrf["required_fields"].append(qrf["temporal_axis"])
            
    # 4. Causal
    elif "why" in q or "cause" in q or "lead to" in q or "result in" in q:
        qrf["operator"] = "CAUSAL"
    # 5. Decision
    elif "should i" in q or "recommend" in q or "choice" in q:
        qrf["operator"] = "DECISION"
    # 6. Current/latest versioned property
    elif "current" in q or "latest" in q or "now" in q or "present" in q:
        qrf["operator"] = "STATE"
        qrf["answer_slot"] = "VALUE"
        
    # Check for required inference
    if "suggest" in q or "imply" in q or "reason" in q or "explain" in q:
        qrf["requires_inference"] = True
        
    return qrf

def _gap_planner(
    agent,
    question: str,
    seeds: List[Dict[str, Any]],
    frame: Any
) -> Tuple[List[EvidenceGap], Dict[str, Any]]:
    qrf = _build_qrf(agent, question, frame)
    
    # Format the seeds as compact representation
    compact_seeds = []
    for s in seeds[:3]:
        compact = {
            "id": s["id"],
            "kind": s.get("kind"),
            "role": s.get("semantic_role"),
            "object": s.get("object_anchor"),
            "value": s.get("value"),
            "claim": s.get("claim"),
            "event_time": s.get("event_time")
        }
        compact_seeds.append(compact)
        
    context = f"Question: {question}\n\nAlready-covered seed evidence:\n{json.dumps(compact_seeds, indent=2)}"
    
    # Call LLM
    response = agent._llm_client.chat(
        [
            {"role": "system", "content": GAP_PLANNER_PROMPT},
            {"role": "user", "content": context}
        ],
        temperature=0.0,
        response_format={"type": "json_object"}
    )
    
    usage = agent._response_usage(response, GAP_PLANNER_PROMPT + context)
    
    try:
        parsed = json.loads(response.content)
    except json.JSONDecodeError:
        parsed = {"evidence_gaps": []}
        
    gaps_data = parsed.get("evidence_gaps", [])
    if not isinstance(gaps_data, list):
        gaps_data = []
        
    # Validation
    gaps_data = gaps_data[:4]
    valid_roles = {"FOCAL_TRIGGER", "PRIOR_TRAJECTORY", "CAUSAL_BRIDGE", "OUTCOME", "OPTION_SUPPORT", "COMPARISON_SIDE", "GENERIC_EVIDENCE"}
    
    evidence_gaps = []
    used_ids = set()
    for i, g in enumerate(gaps_data):
        if not isinstance(g, dict): continue
        
        gid = str(g.get("id", f"gap_{i}"))
        if gid in used_ids:
            gid = f"{gid}_{i}"
        used_ids.add(gid)
        
        role = str(g.get("role", "GENERIC_EVIDENCE")).upper()
        if role not in valid_roles:
            role = "GENERIC_EVIDENCE"
            
        required = bool(g.get("required", True))
        
        temp_axis = str(g.get("temporal_axis", ""))
        if temp_axis not in {"event_time", "document_time", "effective_event_time", ""}:
            temp_axis = ""
            
        # Planner cannot invent a temporal axis if QRF has a strict requirement,
        # nor can it drop a required QRF axis.
        if qrf["temporal_axis"]:
            temp_axis = qrf["temporal_axis"]
            
        temp_relation = str(g.get("temporal_relation", ""))
        temp_anchor = str(g.get("temporal_anchor", ""))
        
        if qrf["operator"] == "TEMPORAL":
            if qrf["temporal_relation"]:
                temp_relation = qrf["temporal_relation"]
            if qrf["temporal_anchor"]:
                temp_anchor = qrf["temporal_anchor"]
        
        # If the operator is STATE, force STATE temporal semantics
        if qrf["operator"] == "STATE":
            temp_axis = "" # State spine handles this
            temp_relation = ""
            temp_anchor = ""

        # Visible options deterministic parsing
        opt_label = str(g.get("option_label", ""))
        if qrf["operator"] == "MULTI_OPTION" and qrf["visible_options"]:
            if opt_label not in qrf["visible_options"] and len(qrf["visible_options"]) > len(evidence_gaps):
                # Force options mapping sequentially if planner failed to label them
                opt_label = qrf["visible_options"][len(evidence_gaps) % len(qrf["visible_options"])]
                
        gap = EvidenceGap(
            id=gid,
            role=role,
            required=required,
            subject_id=getattr(frame, "speaker_role", "primary_user"),
            target_entities=g.get("target_entities", []),
            target_property=g.get("target_property", ""),
            temporal_axis=temp_axis,
            temporal_relation=temp_relation,
            temporal_anchor=temp_anchor,
            option_label=opt_label
        )
        if qrf.get("required_fields"):
            gap.required_fields = qrf["required_fields"]
        # Store description via generic property for adapter
        gap.description = str(g.get("description", ""))
        
        # Keep track of QRF operator to map correctly in legacy slots
        gap.qrf_operator = qrf["operator"]
        
        evidence_gaps.append(gap)
        
    # Ensure options are exhaustively covered
    if qrf["operator"] == "MULTI_OPTION" and qrf["visible_options"]:
        covered_options = {g.option_label for g in evidence_gaps if g.option_label}
        for opt in qrf["visible_options"]:
            if opt not in covered_options and len(evidence_gaps) < 4:
                gid = f"g_opt_{opt}"
                gap = EvidenceGap(
                    id=gid,
                    role="OPTION_SUPPORT",
                    required=True,
                    subject_id=getattr(frame, "speaker_role", "primary_user"),
                    option_label=opt
                )
                gap.qrf_operator = qrf["operator"]
                gap.description = f"Evidence required to evaluate option {opt}"
                evidence_gaps.append(gap)
                
    # Ensure comparisons have at least two sides
    if qrf["operator"] == "COMPARISON":
        sides = [g for g in evidence_gaps if g.role == "COMPARISON_SIDE"]
        side_labels = ["left_side", "right_side"]
        for i in range(len(sides), 2):
            if len(evidence_gaps) >= 4:
                break
            gid = f"g_comp_{i}"
            gap = EvidenceGap(
                id=gid,
                role="COMPARISON_SIDE",
                required=True,
                subject_id=getattr(frame, "speaker_role", "primary_user")
            )
            gap.qrf_operator = qrf["operator"]
            gap.description = f"Evidence required to evaluate comparison {side_labels[i]}"
            # Comparison sides are contextual. Planner should define distinct entities.
            # But if it fails, we fall back to generic metadata
            gap.comparison_side_label = side_labels[i]
            evidence_gaps.append(gap)
            sides.append(gap)
        
    return evidence_gaps, usage
