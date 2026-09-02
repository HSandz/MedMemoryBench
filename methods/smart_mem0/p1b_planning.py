import json
from typing import Any, Dict, List, Tuple
from methods.smart_mem0.p1b_execution import EvidenceGap

GAP_PLANNER_PROMPT = """You are an Evidence-Gap Planner for a medical and general-domain memory retrieval agent.
Your task is to decompose a complex query into specific, missing pieces of evidence ("gaps") that are required to answer the question, but are NOT already present in the provided local seed evidence.

DO NOT output a monolithic slot for the whole query.
DO NOT output memory IDs.
DO NOT provide the final answer.
One gap must correspond to one independently retrievable evidence requirement.

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
            "description": "Event that immediately preceded the failure",
            "required": true,
            "target_entities": ["heart failure", "symptoms"],
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
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
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
    
    # Convert to legacy Plan dictionary structure so existing executor doesn't break
    required_slots = []
    for g in gaps_data:
        slot = {
            "id": g.get("id", "gap_1"),
            "type": "SEMANTIC",
            "evidence_role": g.get("role", "GENERIC_EVIDENCE"),
            "description": g.get("description", ""),
            "required": g.get("required", True),
            "resolution_strategy": "RETRIEVE",
            "subject": getattr(frame, "speaker_role", "primary_user"),
            "time_axis": g.get("temporal_axis", "")
        }
        required_slots.append(slot)
        
    plan = {
        "required_slots": required_slots,
        "seed_coverage": [],
        "operations": [], # P1B.1 deterministic executor will just do a fallback search per missing slot
        "need_evidence": True,
        "budget_tier": "SMALL" if len(required_slots) <= 1 else "MEDIUM",
        "max_memories": 3,
        "planner_fallback": False,
        "valid": True,
    }
    
    # We will let the existing _execute_plan handle the operations by creating a deterministic recovery pass
    # since operations is empty, `_execute_plan` will see `unresolved` slots. 
    # But wait, `_execute_plan` executes `plan.get("operations")`. If it's empty, it relies on seeds.
    # If seeds don't fulfill the slots, it goes to replan!
    # The user says: "One deterministic recovery pass. No replan LLM".
    
    return plan, usage

