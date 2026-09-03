from typing import Dict, Any, Optional
import time
from methods.smart_mem0.router import DeterministicRouter, RouteDecision

def _prepare_p1a_query(agent, routing_question: str, answer_question: str, subject_aliases: dict = None, telemetry_out: dict = None) -> Optional[Dict[str, Any]]:
    if telemetry_out is None: telemetry_out = {}
    router = DeterministicRouter(subject_postings=agent._subject_postings, subject_aliases=subject_aliases)
    telemetry_out.update({
        "attempted": True,
        "routing_question": routing_question,
        "accepted": False
    })
    
    decision = router.route_query(routing_question)
    telemetry_out["route"] = decision.route
    
    if decision.route == "HARD":
        telemetry_out["fallback_reason"] = "HARD_SURFACE_OR_UNRESOLVED"
        return None
    
    if decision.route == "HARD":
        telemetry_out["fallback_reason"] = "HARD_SURFACE_OR_UNRESOLVED"
        return None
        
    start_time = time.time()
    used_ids = set()
    precomputed = None
    
    # 1. Subject Firewall
    subject_ids = agent._subject_postings.get(decision.subject_id, set())
    if not subject_ids:
        telemetry_out["fallback_reason"] = "SUBJECT_NOT_AUTHORIZED"
        return None
        
    if decision.route == "STATE_LATEST":
        # Target concept -> canonical identity
        ident = _resolve_canonical_identity(agent, decision.content_terms, decision.subject_id)
        if not ident:
            telemetry_out["fallback_reason"] = "STATE_IDENTITY_UNRESOLVED"
            return None
            
        spine = agent._state_spine.get(ident)
        if not spine:
            telemetry_out["fallback_reason"] = "STATE_SPINE_MISSING"
            return None
            
        latest = spine.latest()
        if not latest:
            telemetry_out["fallback_reason"] = "STATE_LATEST_MISSING"
            return None
            
        val = str(latest.get("value") or "").strip()
        if not val:
                telemetry_out["fallback_reason"] = "VALUE_MISSING"
                return None
            
        if decision.answer_slot == "VALUE":
            precomputed = val
            
        used_ids.add(latest["id"])
        
    elif decision.route == "TEMPORAL":
        # Candidate pool
        pool = _candidate_pool(agent, decision.content_terms, subject_ids)
        if not pool:
            telemetry_out["fallback_reason"] = "NO_EXACT_CANDIDATE"
            return None
            
        # Axis constraint
        dated_pool = [m for m in pool if _get_date(m, decision.temporal_axis)]
        if not dated_pool:
            telemetry_out["fallback_reason"] = "TEMPORAL_NO_DATES"
            return None
            
        # Arbitration
        def temp_key(m):
            return (
                _get_date(m, decision.temporal_axis),
                _get_date(m, "document_time"),
                int(m.get("session_idx", 0)),
                int(m["id"].split("_")[-1]) if "_" in m["id"] and m["id"].split("_")[-1].isdigit() else 0
            )
            
        if decision.temporal_relation == "EARLIEST":
            selected = min(dated_pool, key=temp_key)
            if selected.get("assertion_mode") == "RECAP":
                telemetry_out["fallback_reason"] = "TEMPORAL_RECAP_ONSET_UNSAFE"
                return None
        elif decision.temporal_relation == "LATEST":
            selected = max(dated_pool, key=temp_key)
        elif decision.temporal_relation == "EXACT":
            if decision.temporal_anchor:
                matched = []
                for m in dated_pool:
                    date_str = _get_date(m, decision.temporal_axis)
                    # YYYY-MM-DD
                    m_year, m_month, m_day = None, None, None
                    parts = date_str.split('-')
                    if len(parts) >= 1 and parts[0].isdigit(): m_year = int(parts[0])
                    if len(parts) >= 2 and parts[1].isdigit(): m_month = int(parts[1])
                    if len(parts) >= 3 and parts[2].isdigit(): m_day = int(parts[2])
                    
                    if decision.temporal_year and m_year != decision.temporal_year:
                        continue
                    if decision.temporal_month and m_month != decision.temporal_month:
                        continue
                    if decision.temporal_day and m_day != decision.temporal_day:
                        continue
                    matched.append(m)
                
                if not matched:
                    telemetry_out["fallback_reason"] = "TEMPORAL_NO_EXACT_FOR_ANCHOR"
                    return None
                selected = min(matched, key=temp_key)
            else:
                # Unanchored MATCH fails HARD for now until focal-event relevance is implemented
                telemetry_out["fallback_reason"] = "TEMPORAL_EXACT_AMBIGUOUS"
                return None
        else:
            telemetry_out["fallback_reason"] = "ROUTE_UNKNOWN"
            return None
            
        if decision.answer_slot == "DATE":
            precomputed = _get_date(selected, decision.temporal_axis)
        elif decision.answer_slot == "VALUE":
            val = str(selected.get("value") or "").strip()
            if not val:
                telemetry_out["fallback_reason"] = "VALUE_MISSING"
                return None
            precomputed = val
        else:
            telemetry_out["fallback_reason"] = "ROUTE_UNKNOWN"
            return None
            
        used_ids.add(selected["id"])
        
    elif decision.route == "EXACT":
        pool = _candidate_pool(agent, decision.content_terms, subject_ids)
        if not pool:
            telemetry_out["fallback_reason"] = "NO_EXACT_CANDIDATE"
            return None
            
        # exact/bm25 bounded
        # if only a few, just pass them to answer LLM
        # pick highest confidence or just up to 3
        # exact pool is already ranked by tier, just tie-break by confidence and recency if needed
        # Actually it's ordered by tier, so we just take up to top 3
        selected = pool[:3]
        for m in selected:
            used_ids.add(m["id"])
        # DO NOT precompute answer, let the 1 answer LLM format it
        
    else:
            telemetry_out["fallback_reason"] = "ROUTE_UNKNOWN"
            return None
        
    telemetry_out["accepted"] = True
    elapsed = (time.time() - start_time) * 1000
    
    # Return fast path frame
    if decision.route == "STATE_LATEST":
        retrieved_mems = [agent._state_spine[ident].latest()] if 'ident' in locals() and ident in agent._state_spine else []
    elif decision.route == "TEMPORAL" and 'selected' in locals() and isinstance(selected, dict):
        retrieved_mems = [selected]
    elif decision.route == "EXACT" and 'selected' in locals() and isinstance(selected, list):
        retrieved_mems = selected
    else:
        retrieved_mems = []
    ret = {
        "evidence": "", 
        "raw_context": "",
        "used_memory_ids": list(used_ids),
        "retrieved_memories": retrieved_mems,
        "retrieved_count": len(retrieved_mems),
        "gate": {"called": False, "skip_reason": f"p1a_{decision.route.lower()}"},
        "supports": None,
        "controller_usage": {},
        "planner_usage": {},
        "question": answer_question,
        "system_message": None, 
        "extra": {
            "planner_called": False,
            "semantic_controller": {"called": False},
            "query_tokens": {
                "fast_gate": 0,
                "planner": 0,
                "slot_validation": 0,
                "replan": 0,
                "answer": 0,
                "total": 0
            },
            "retrieval_elapsed_ms": elapsed
        }
    }
    
    telemetry_out.update({
        "route": decision.route,
        "subject_id": decision.subject_id,
        "content_terms": decision.content_terms,
        "answer_slot": decision.answer_slot,
        "temporal_axis": decision.temporal_axis,
        "temporal_relation": decision.temporal_relation,
        "temporal_anchor": decision.temporal_anchor,
        "candidate_ids": [m["id"] for m in pool] if 'pool' in locals() else list(used_ids),
        "selected_memory_id": selected["id"] if 'selected' in locals() and isinstance(selected, dict) else (list(used_ids)[0] if len(used_ids) == 1 else None),
        "precomputed": precomputed is not None,
        "accepted_reason": f"{decision.route}_RESOLVED"
    })
    
    if precomputed is not None:
        ret["precomputed_answer"] = precomputed
    elif decision.route == "EXACT":
        # Format the evidence for the answer LLM
        evidence_text = agent._format_episode_context(selected)
        ret["evidence"] = evidence_text
        ret["raw_context"] = evidence_text
        ret["messages"] = [
            {"role": "system", "content": "Ground every subject-specific claim in supplied memory evidence. For inference, general knowledge may connect grounded endpoints, but must not substitute for missing subject-specific evidence. Never invent subject history, values, events, decisions, or states."},
            {"role": "user", "content": f"Context:\n{evidence_text}\n\nQuestion: {answer_question}"}
        ]
        
    return ret

def _resolve_canonical_identity(agent, content_terms: list, subject_id: str):
    if not content_terms:
        return None
        
    matches = set()
    for ident, spine in agent._state_spine.items():
        if not ident.startswith(f"{subject_id}::"):
            continue
            
        canonical_text = set()
        parts = ident.split("::")
        if len(parts) >= 2:
            canonical_text.update(parts[1].lower().split())
            
        for m in spine.versions:
            if "object_anchor" in m and m["object_anchor"]:
                canonical_text.update(str(m["object_anchor"]).lower().split())
            if "entities" in m:
                for e in m["entities"]:
                    canonical_text.update(str(e).lower().split())
                    
        if any(t in canonical_text for t in content_terms):
            matches.add(ident)
            
    if len(matches) == 1:
        return list(matches)[0]
        
    return None

def _candidate_pool(agent, content_terms: list, subject_ids: set) -> list:
    if not content_terms:
        return []
        
    matched_memories = []
    for m in agent._memories:
        if m["id"] not in subject_ids:
            continue
            
        canonical_text = set()
        if "object_anchor" in m and m["object_anchor"]:
            canonical_text.update(str(m["object_anchor"]).lower().split())
        if "state_key" in m and m["state_key"]:
            canonical_text.update(str(m["state_key"]).lower().split())
        if "entities" in m:
            for e in m["entities"]:
                canonical_text.update(str(e).lower().split())
                
        if any(t in canonical_text for t in content_terms):
            matched_memories.append(m)
            
    if not matched_memories:
        return []
        
    # Check coherence: do they all belong to the same canonical concept family?
    distinct_anchors = set()
    for m in matched_memories:
        anchor = str(m.get("object_anchor") or "").lower().strip()
        if anchor:
            distinct_anchors.add(anchor)
            
    if len(distinct_anchors) > 1:
        # Ambiguous distinct concepts mixed together -> HARD
        return []
        
    return matched_memories

def _get_date(m: dict, axis: str) -> str:
    val = str(m.get(axis) or "").strip()
    if not val or val.upper() == "UNKNOWN":
        return ""
    return val
