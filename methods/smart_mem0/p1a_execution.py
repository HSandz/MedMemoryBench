from typing import Dict, Any, Optional
import time
from methods.smart_mem0.router import DeterministicRouter, RouteDecision

def _prepare_p1a_query(agent, question: str) -> Optional[Dict[str, Any]]:
    router = DeterministicRouter(subject_postings=agent._subject_postings)
    decision = router.route_query(question)
    
    if decision.route == "HARD":
        return None
        
    start_time = time.time()
    used_ids = set()
    precomputed = None
    
    # 1. Subject Firewall
    subject_ids = agent._subject_postings.get(decision.subject_id, set())
    if not subject_ids:
        # Ambiguous or unknown subject fails HARD
        return None
        
    if decision.route == "STATE_LATEST":
        # Target concept -> canonical identity
        ident = _resolve_canonical_identity(agent, decision.target_concept, decision.subject_id)
        if not ident:
            return None
            
        spine = agent._state_spine.get(ident)
        if not spine:
            return None
            
        latest = spine.latest()
        if not latest:
            return None
            
        val = str(latest.get("value") or "").strip()
        if not val:
            return None
            
        if decision.answer_slot == "VALUE":
            precomputed = val
            
        used_ids.add(latest["id"])
        
    elif decision.route == "TEMPORAL":
        # Candidate pool
        pool = _temporal_candidate_pool(agent, decision.target_concept, subject_ids)
        if not pool:
            return None
            
        # Axis constraint
        dated_pool = [m for m in pool if _get_date(m, decision.temporal_axis)]
        if not dated_pool:
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
        elif decision.temporal_relation == "LATEST":
            selected = max(dated_pool, key=temp_key)
        elif decision.temporal_relation == "MATCH":
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
                    return None
                selected = min(matched, key=temp_key)
            else:
                # Unanchored MATCH
                selected = min(dated_pool, key=temp_key)
        else:
            return None
            
        if decision.answer_slot == "DATE":
            precomputed = _get_date(selected, decision.temporal_axis)
        elif decision.answer_slot == "VALUE":
            val = str(selected.get("value") or "").strip()
            if not val:
                return None
            precomputed = val
        else:
            return None
            
        used_ids.add(selected["id"])
        
    elif decision.route == "EXACT":
        pool = _temporal_candidate_pool(agent, decision.target_concept, subject_ids)
        if not pool:
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
        return None
        
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
        "question": question,
        "system_message": None, 
        "planner_called": False,
        "semantic_controller": {"called": False},
        "extra": {
            "query_tokens": {
                "fast_gate": 0,
                "planner": 0,
                "slot_validation": 0,
                "replan": 0,
                "answer": 0,
                "total": 0
            },
            "retrieval_elapsed_ms": elapsed,
            "p1a": {
                "route": decision.route,
                "subject_id": decision.subject_id,
                "target_concept": decision.target_concept,
                "answer_slot": decision.answer_slot,
                "temporal_axis": decision.temporal_axis,
                "temporal_relation": decision.temporal_relation,
                "temporal_anchor": decision.temporal_anchor,
                "candidate_ids": [m["id"] for m in pool] if 'pool' in locals() else list(used_ids),
                "selected_memory_id": selected["id"] if 'selected' in locals() and isinstance(selected, dict) else (list(used_ids)[0] if len(used_ids) == 1 else None),
                "precomputed": precomputed is not None
            }
        }
    }
    
    if precomputed is not None:
        ret["precomputed_answer"] = precomputed
    elif decision.route == "EXACT":
        # Format the evidence for the answer LLM
        evidence_text = agent._format_episode_context(selected)
        ret["evidence"] = evidence_text
        ret["raw_context"] = evidence_text
        ret["messages"] = [
            {"role": "system", "content": "Ground every subject-specific claim in supplied memory evidence. For inference, general knowledge may connect grounded endpoints, but must not substitute for missing subject-specific evidence. Never invent subject history, values, events, decisions, or states."},
            {"role": "user", "content": f"Context:\n{evidence_text}\n\nQuestion: {question}"}
        ]
        
    return ret

def _resolve_canonical_identity(agent, concept: str, subject_id: str) -> Optional[str]:
    # Check if exact identity exists
    concept = concept.lower()
    
    # Try exact state_keys from spine
    matches = []
    for ident in agent._state_spine.keys():
        if not ident.startswith(f"{subject_id}::"):
            continue
        parts = ident.split("::")
        if len(parts) >= 2:
            sk = parts[1].lower()
            if concept in sk or sk in concept:
                matches.append(ident)
                
    if len(matches) == 1:
        return matches[0]
        
    # Check object anchor
    matches = []
    for ident, spine in agent._state_spine.items():
        if not ident.startswith(f"{subject_id}::"):
            continue
        for m in spine.versions:
            obj = str(m.get("object_anchor") or "").lower()
            if concept in obj or obj in concept:
                matches.append(ident)
                break
                
    if len(matches) == 1:
        return matches[0]
        
    return None

def _temporal_candidate_pool(agent, concept: str, subject_ids: set) -> list:
    # Build ranked candidates
    concept = concept.lower()
    words = set(concept.split())
    
    exact_obj = []
    exact_ent = []
    exact_pred = []
    alias = []
    bm25 = []
    
    for m in agent._memories:
        if m["id"] not in subject_ids:
            continue
            
        # Subject Firewall is implicitly enforced by subject_ids
        
        # EXACT OBJECT
        obj = str(m.get("object_anchor") or "").lower()
        if obj and (obj == concept):
            exact_obj.append(m)
            continue
            
        # EXACT ENTITY
        ents = [str(e).lower() for e in m.get("entities", [])]
        if concept in ents:
            exact_ent.append(m)
            continue
            
        # EXACT PREDICATE / Canonical Key
        sk = str(m.get("state_key") or "").lower()
        if sk and sk == concept:
            exact_pred.append(m)
            continue
            
        # ALIAS / Substring Object
        if obj and (concept in obj or obj in concept):
            alias.append(m)
            continue
            
        if sk and (concept in sk or sk in concept):
            alias.append(m)
            continue
            
        # BOUNDED BM25 Fallback (simple lexical overlap threshold)
        claim = str(m.get("claim") or "").lower()
        claim_words = set(claim.split())
        overlap = len(words.intersection(claim_words))
        if overlap > 0:
            # We store overlap for sorting later
            bm25.append((overlap, m))
            
    bm25.sort(key=lambda x: x[0], reverse=True)
    bm25 = [m for _, m in bm25]
    
    # Return the highest authorized tier that has candidates
    if exact_obj: return exact_obj
    if exact_ent: return exact_ent
    if exact_pred: return exact_pred
    if alias: return alias
    
    # Only if empty, use lexical recovery
    # Margin check: for P1A, we just take the top-tier of BM25 if it exists
    return bm25
    
def _get_date(m: dict, axis: str) -> str:
    # Handle "UNKNOWN"
    val = str(m.get(axis) or "").strip()
    if not val or val.upper() == "UNKNOWN":
        return ""
    return val

