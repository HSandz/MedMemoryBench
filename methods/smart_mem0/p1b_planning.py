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
            "temporal_axis": "event_time"
        }
    ]
}

If the supplied seed evidence already completely answers the question, output an empty list for evidence_gaps.
"""

def _gap_planner(
    agent,
    question: str,
    seeds: List[Dict[str, Any]],
    frame: Any
) -> Tuple[List[EvidenceGap], Dict[str, Any]]:
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
        
        gap = EvidenceGap(
            id=gid,
            role=role,
            required=required,
            subject_id=getattr(frame, "speaker_role", "primary_user"),
            target_entities=g.get("target_entities", []),
            target_property=g.get("target_property", ""),
            temporal_axis=g.get("temporal_axis", "")
        )
        # Store description via generic property for adapter
        gap.description = str(g.get("description", ""))
        evidence_gaps.append(gap)
        
    return evidence_gaps, usage
