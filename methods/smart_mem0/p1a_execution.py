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
                    telemetry_out["fallback_reason"] = "TEMPORAL_NO_MATCH_FOR_ANCHOR"
                    return None
                selected = min(matched, key=temp_key)
            else:
                # Unanchored MATCH fails HARD for now until focal-event relevance is implemented
                telemetry_out["fallback_reason"] = "TEMPORAL_MATCH_AMBIGUOUS"
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
        
    scores = {}
    for ident, spine in agent._state_spine.items():
        if not ident.startswith(f"{subject_id}::"):
            continue
            
        score = 0
        parts = ident.split("::")
        if len(parts) >= 2:
            sk = parts[1].lower()
            if sk and all(t in sk for t in content_terms):
                score += 10
            elif sk and any(t in sk for t in content_terms):
                score += sum(3 for t in content_terms if t in sk)
                
        for m in spine.versions:
            obj = str(m.get("object_anchor") or "").lower()
            if obj and all(t in obj for t in content_terms):
                score += 8
                break
            elif obj and any(t in obj for t in content_terms):
                score += sum(2 for t in content_terms if t in obj)
                break
                
        if score > 0:
            scores[ident] = score
            
    if not scores:
        return None
        
    # Find unique winner with margin
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    if len(ranked) == 1:
        return ranked[0][0]
    if ranked[0][1] > ranked[1][1]: # Strict margin > 0
        return ranked[0][0]
        
    return None

def _candidate_pool(agent, content_terms: list, subject_ids: set) -> list:
    if not content_terms:
        return []
    
    scores = []
    
    for m in agent._memories:
        if m["id"] not in subject_ids:
            continue
            
        score = 0
        obj = str(m.get("object_anchor") or "").lower()
        sk = str(m.get("state_key") or "").lower()
        ents = [str(e).lower() for e in m.get("entities", [])]
        claim = str(m.get("claim") or "").lower()
        
        # EXACT OBJECT
        if obj and all(t in obj for t in content_terms): score += 20
        elif obj and any(t in obj for t in content_terms): score += sum(5 for t in content_terms if t in obj)
            
        # EXACT ENTITY
        if any(all(t in e for t in content_terms) for e in ents): score += 15
        else:
            for e in ents:
                if any(t in e for t in content_terms): score += sum(3 for t in content_terms if t in e)
            
        # EXACT PREDICATE / Canonical Key
        if sk and all(t in sk for t in content_terms): score += 12
        elif sk and any(t in sk for t in content_terms): score += sum(3 for t in content_terms if t in sk)
            
        # BOUNDED Lexical Fallback
        claim_words = set(claim.split())
        overlap = sum(1 for t in content_terms if t in claim_words)
        score += overlap * 2
        
        if score > 0:
            scores.append((score, m))
            
    scores.sort(key=lambda x: x[0], reverse=True)
    
    # If we have strong lexical candidates (threshold 8+)
    if scores and scores[0][0] >= 8:
        # Return all that share the top tier, or just top ones above threshold
        return [m for s, m in scores if s >= 8]
        
    # BOUNDED Dense Fallback
    import numpy as np
    try:
        from sklearn.metrics.pairwise import cosine_similarity
        if agent._embedder and getattr(agent, "_embedding_cache", None):
            q_emb = agent._embedder.encode([" ".join(content_terms)], show_progress_bar=False)
            dense_scores = []
            for m in agent._memories:
                if m["id"] not in subject_ids: continue
                m_emb = agent._embedding_cache.get(m["id"])
                if m_emb is not None:
                    # m_emb is usually 1D. cosine_similarity takes 2D.
                    sim = cosine_similarity(q_emb, np.array([m_emb]))[0][0]
                    dense_scores.append((sim, m))
            
            dense_scores.sort(key=lambda x: x[0], reverse=True)
            if dense_scores and dense_scores[0][0] >= 0.70: # Bounded threshold
                # return top 3 above threshold
                return [m for s, m in dense_scores if s >= 0.70][:3]
    except Exception:
        pass
    
    return [m for s, m in scores if s > 0]

def _get_date(m: dict, axis: str) -> str:
    val = str(m.get(axis) or "").strip()
    if not val or val.upper() == "UNKNOWN":
        return ""
    return val
