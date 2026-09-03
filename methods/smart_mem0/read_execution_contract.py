"""Deterministic execution/proof invariants for two-stage SmartMem0 reads."""

from collections import defaultdict

from .canonicalization import state_identity
from .contracts import QueryFrame, RETRIEVAL_BUDGETS, VALID_TEMPORAL_AXES


class ReadExecutionContractMixin:
    # ---------- option-aware retrieval ----------
    def _semantic_operation_search(
        self, query, top_k, strategy, frame=None, option_queries=None
    ):
        strategy = str(strategy or "FOCAL").upper()
        if strategy != "SHARED_OPTIONS":
            return super()._semantic_operation_search(
                query, top_k, strategy,
                frame=frame or QueryFrame(), option_queries=option_queries,
            )
        frame = frame or QueryFrame()
        eligible_ids = {
            memory["id"]
            for memory in self._memories
            if self._memory_satisfies_frame(
                memory, frame, include_entities=bool(frame.hard_entities)
            )
            and self._query_visible_memory(memory)
        }
        if not eligible_ids:
            return []
        base = self._hybrid_search(
            query, top_k=min(max(int(top_k) * 3, 12), len(eligible_ids)),
            candidate_ids=eligible_ids,
        )
        representatives, option_hits, seen = [], [], set()
        for option_query in option_queries or []:
            text = str(option_query or "").strip()
            if not text:
                continue
            hits = self._hybrid_search(
                text, top_k=min(4, len(eligible_ids)), candidate_ids=eligible_ids
            )
            option_hits.extend(hits)
            representative = next(
                (memory for memory in hits if memory["id"] not in seen), None
            )
            if representative:
                representatives.append(representative)
                seen.add(representative["id"])
        selected = []
        for memory in (*representatives, *base, *option_hits):
            if memory["id"] in {item["id"] for item in selected}:
                continue
            selected.append(self._snapshot(memory))
            if len(selected) >= int(top_k):
                break
        return selected

    # ---------- deterministic typed operations / recovery ----------
    def _compile_gap_operations(self,slots,question,budget_tier="MEDIUM",plan=None):
        if not slots: return []
        mode=str((plan or {}).get("query_mode") or slots[0].get("qrf_operator") or "DIRECT").upper(); max_ops=RETRIEVAL_BUDGETS.get(budget_tier,{}).get("max_operations",4)
        if mode=="MULTI_OPTION":
            opts=(plan or {}).get("visible_options") or self._question_options(question) or {}; return [{"op":"SEMANTIC_SEARCH","query":self._question_stem(question),"top_k":8,"strategy":"SHARED_OPTIONS","option_queries":list(opts.values()),"produces":[s["id"] for s in slots]}]
        if mode in {"DECISION","CAUSAL","MULTI_HOP"} and not any(s.get("type")=="CAUSE_PATH" for s in slots):
            trajectory=[s["id"] for s in slots if s.get("evidence_role")=="PRIOR_TRAJECTORY"]; focal=[s["id"] for s in slots if s["id"] not in trajectory]; ops=[]
            if focal: ops.append({"op":"SEMANTIC_SEARCH","query":question,"top_k":6,"strategy":"DECISION_BUNDLE","produces":focal})
            if trajectory and len(ops)<max_ops: ops.append({"op":"SEMANTIC_SEARCH","query":question,"top_k":6,"strategy":"TRAJECTORY","produces":trajectory})
            return ops[:max_ops]
        ops=[]
        for s in slots:
            if len(ops)>=max_ops: break
            sid=s["id"]; desc=str(s.get("description") or question); typ=str(s.get("type") or "DIRECT").upper()
            if typ=="CURRENT_STATE": ops.append({"op":"RESOLVE_STATE","query":desc,"produces":[sid]}); continue
            if typ=="CAUSE_PATH":
                idx=len(ops); ops.append({"op":"LOCATE_ANCHOR","query":desc,"produces":[sid]})
                if len(ops)<max_ops: ops.append({"op":"FOLLOW_CAUSES","start":[f"${idx}"],"direction":"OUT","depth":3,"goal":desc,"produces":[sid]})
                continue
            if typ=="TEMPORAL":
                rel=str(s.get("temporal_relation") or s.get("time_relation") or "LOCATE").upper(); axis=str(s.get("time_axis") or "event_time").lower(); anchor=str(s.get("time_anchor") or ""); end=str(s.get("time_end") or "")
                if rel=="EXACT" and not anchor: rel="LOCATE"
                if rel in {"BEFORE","AFTER"} and not anchor: rel="LOCATE"
                if rel=="BETWEEN" and (not anchor or not end): rel="LOCATE"
                if rel in {"EARLIEST","LATEST"}:
                    idx=len(ops); ops.append({"op":"LOCATE_ANCHOR","query":desc,"produces":[sid]})
                    if len(ops)<max_ops: ops.append({"op":"TEMPORAL_FILTER","query":desc,"relation":rel,"axis":axis,"fallback_axis":"","candidate_refs":[f"${idx}"],"produces":[sid]})
                else: ops.append({"op":"TEMPORAL_FILTER","query":desc,"relation":rel,"axis":axis,"fallback_axis":"","anchor":anchor,"end":end,"produces":[sid]})
                continue
            ops.append({"op":"SEMANTIC_SEARCH","query":desc,"top_k":8,"strategy":"FOCAL","produces":[sid]})
        return ops[:max_ops]

    def _make_deterministic_recovery_plan(self,missing_slots,question,existing_plan):
        if not missing_slots: return None
        budget=existing_plan.get("budget_tier","MEDIUM"); shell={"query_mode":existing_plan.get("query_mode","DIRECT"),"visible_options":existing_plan.get("visible_options",{})}; ops=self._compile_gap_operations(missing_slots,question,budget,plan=shell)
        return {"query_spec":existing_plan.get("query_spec",{}),"query_mode":shell["query_mode"],"required_slots":missing_slots,"seed_coverage":[],"operations":ops,"option_coverage":[],"visible_options":shell["visible_options"],"need_evidence":False,"budget_tier":budget,"max_memories":RETRIEVAL_BUDGETS.get(budget,RETRIEVAL_BUDGETS["MEDIUM"])["max_memories"],"planner_fallback":True,"fallback_reason":"deterministic_missing_evidence_recovery","valid":True}

    # ---------- deterministic proof ----------
    def _slot_contract_match(self,slot,memory,strict_targets=None):
        if not self._rc_owner_match(slot,memory): return False
        for f in slot.get("required_fields") or []:
            if f in VALID_TEMPORAL_AXES:
                if not self._date_for(memory,f): return False
            elif f and not memory.get(f): return False
        role=str(slot.get("evidence_role") or "").upper()
        if strict_targets is None: strict_targets=role in {"ANSWER","GENERIC_EVIDENCE","COMPARISON_SIDE"}
        if not strict_targets: return True
        text=" ".join(str(v or "") for v in (memory.get("claim"),self._memory_value(memory),memory.get("verbatim_value"),memory.get("scope"),memory.get("state_key"),memory.get("object_anchor")," ".join(memory.get("entities",[]))," ".join(memory.get("scope_entities",[]))))
        hay=set(self._rc_terms(text)); norm=" ".join(self._rc_terms(text)); generic={"primary_user","patient","user","person","participant","subject"}
        ents=[e for e in (slot.get("target_entities") or []) if self._rc_owner(e) not in generic and str(e).strip()]
        if ents and not any((" ".join(self._rc_terms(e)) in norm) or (self._rc_terms(e) and sum(t in hay for t in self._rc_terms(e))/len(self._rc_terms(e))>=0.6) for e in ents): return False
        pt=self._rc_terms(slot.get("target_property") or "")
        if pt and len(pt)<=6 and " ".join(pt) not in norm and sum(t in hay for t in pt)/len(pt)<0.5: return False
        return True

    def _memory_matches_slot_role(self,slot,memory):
        role=str(slot.get("evidence_role") or "").upper()
        if role=="OPTION_CONTEXT": return self._rc_owner_match(slot,memory)
        if role=="PRIOR_TRAJECTORY": return self._rc_owner_match(slot,memory) and bool(self._date_for(memory,"event_time") or self._date_for(memory,"document_time") or state_identity(memory) or "TRAJECTORY" in {str(x).upper() for x in memory.get("planning_tags",[])})
        return self._slot_contract_match(slot,memory)

    def _slot_covered(self,slot,support_ids,selected,relations):
        mem=[m for m in selected if m.get("id") in set(support_ids)]
        if not mem: return False
        typ=str(slot.get("type") or "DIRECT").upper(); role=str(slot.get("evidence_role") or "").upper()
        if typ=="DIRECT":
            ok=[m for m in mem if self._memory_value(m) and m.get("assertion_mode","DIRECT") in {"DIRECT","RECAP"} and self._memory_matches_slot_role(slot,m) and (slot.get("history") or m.get("_status",self._belief_status.get(m.get("id"),"active"))!="superseded")]
            return len({m["id"] for m in ok}) >= (2 if role=="PRIOR_TRAJECTORY" else 1)
        if typ=="CURRENT_STATE":
            heads=[m for m in mem if self._is_state_head(m) and self._memory_value(m) and self._slot_contract_match(slot,m,True)]; ids={state_identity(m) for m in heads if state_identity(m)}; return bool(heads and len(ids)==1)
        if typ=="TEMPORAL":
            axis=str(slot.get("time_axis") or "").lower(); rel=str(slot.get("temporal_relation") or slot.get("time_relation") or "LOCATE").upper(); anchor=self._parse_date(str(slot.get("time_anchor") or "")); end=self._parse_date(str(slot.get("time_end") or ""))
            if axis not in VALID_TEMPORAL_AXES: return False
            def good(m):
                date=self._date_for(m,axis)
                if not date or not self._slot_contract_match(slot,m,True): return False
                if rel=="LOCATE": return True
                if rel=="EXACT": return bool(anchor and (date==anchor or date.startswith(anchor)))
                if rel=="BEFORE": return bool(anchor and date<anchor)
                if rel=="AFTER": return bool(anchor and date>anchor)
                if rel=="BETWEEN": return bool(anchor and end and anchor<=date<=end)
                return rel in {"EARLIEST","LATEST"}
            return any(good(m) for m in mem)
        if typ=="CAUSE_PATH":
            by={m["id"]:m for m in mem}; adj=defaultdict(list)
            for r in relations:
                if self._valid_causal_relation(r,by): adj[r["source_id"]].append(r["target_id"])
            if len(by)<2 or not any(adj.values()): return False
            for root in by:
                seen={root}; q=[root]
                while q:
                    for t in adj.get(q.pop(0),[]):
                        if t not in seen: seen.add(t); q.append(t)
                if set(by).issubset(seen): return True
            return False
        if typ=="TRANSITION":
            by={m["id"]:m for m in mem}
            return any(r.get("type") in {"SUPERSEDE","REFINE"} and r.get("source_id") in by and r.get("target_id") in by for r in relations)
        if typ=="COMPARISON": return len({self._normalised_value(m) for m in mem if self._normalised_value(m)})>=2
        return False

    def _coverage_map(self,plan,slot_support,selected,relations):
        cov={s["id"]:self._slot_covered(s,slot_support.get(s["id"],[]),selected,relations) for s in plan.get("required_slots",[])}; mode=str(plan.get("query_mode") or "").upper()
        if cov and all(cov.values()) and mode in {"COMPARISON","MULTI_HOP","CAUSAL"}:
            sets=[set(slot_support.get(s["id"],[])) for s in plan.get("required_slots",[]) if s.get("required") is not False]
            if len(set().union(*sets))<2: cov={k:False for k in cov}
            if mode=="COMPARISON" and len(sets)>=2 and sets[0]==sets[1]: cov={k:False for k in cov}
        return cov

    def _operation_slot_support(self,slot,result,relations):
        if not result: return []
        role=str(slot.get("evidence_role") or "").upper(); typ=str(slot.get("type") or "DIRECT").upper(); ranked=self._hybrid_search(str(slot.get("description") or slot.get("id")),top_k=len(result),candidate_ids={m["id"] for m in result}); by={m["id"]:m for m in result}; ranked=[by[m["id"]] for m in ranked if m["id"] in by]
        if role=="OPTION_CONTEXT": return [m for m in ranked if self._rc_owner_match(slot,m)][:6]
        if typ=="DIRECT": return [m for m in ranked if self._memory_value(m) and m.get("assertion_mode","DIRECT") in {"DIRECT","RECAP"} and self._memory_matches_slot_role(slot,m)][:4]
        if typ=="TEMPORAL": return [m for m in ranked if self._date_for(m,str(slot.get("time_axis") or "event_time")) and self._slot_contract_match(slot,m,True)][:4]
        if typ=="CURRENT_STATE": return [m for m in ranked if self._is_state_head(m) and self._slot_contract_match(slot,m,True)][:2]
        if typ=="CAUSE_PATH":
            ends=set()
            for r in relations:
                if self._valid_causal_relation(r,by): ends.update((r["source_id"],r["target_id"]))
            return [m for m in ranked if m["id"] in ends]
        return ranked[:4]

    # ---------- evidence packing / compatibility ----------
    def _role_aware_support_ids(self,slots,slot_support,candidate_order,limit):
        out=[]; allowed=set(candidate_order)
        def add(mid):
            if mid in allowed and mid not in out and len(out)<limit: out.append(mid)
        for s in slots:
            ids=slot_support.get(s.get("id"),[]); role=str(s.get("evidence_role") or "").upper(); reserve=6 if role=="OPTION_CONTEXT" else (3 if role in {"PRIOR_TRAJECTORY","FOCAL_TRIGGER","OUTCOME"} else 1)
            for mid in ids[:reserve]: add(mid)
        supported={x for ids in slot_support.values() for x in ids}
        for mid in candidate_order:
            if mid in supported: add(mid)
        return out

    @staticmethod
    def _multiple_choice_answer_instruction(option_labels):
        labels=", ".join(sorted(str(x) for x in option_labels)); return f" This is multiple-choice with options {labels}. Evaluate EVERY visible option against the same predicate in the question and the shared participant-specific evidence. Use general domain knowledge only to interpret grounded participant facts. Output ONLY all matching uppercase labels separated by commas, or NONE."

    def prepare_batch_query(self,question,system_message=None,**kwargs):
        prepared=super().prepare_batch_query(question,system_message=system_message,**kwargs)
        if system_message and prepared.get("precomputed_answer"):
            prepared["precomputed_answer"]=""
        return prepared
