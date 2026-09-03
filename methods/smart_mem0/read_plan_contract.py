"""Semantic requirement -> EvidenceGap plan normalization."""

from .contracts import RETRIEVAL_BUDGETS
from .p1b_execution import EvidenceGap, EvidenceLattice
from .read_controller import VALID_ROLES


class ReadPlanContractMixin:
    # ---------- plan normalization ----------
    def _rc_gap(self,gid,role,d,description,**kw):
        gap=EvidenceGap(id=gid,role=role,required=True,subject_id=d.get("subject_id") or "primary_user",target_entities=list(kw.pop("target_entities",None) or d.get("target_entities") or []),target_property=str(kw.pop("target_property","") or ""),temporal_axis=str(kw.pop("temporal_axis","") or ""),temporal_relation=str(kw.pop("temporal_relation","") or ""),temporal_anchor=str(kw.pop("temporal_anchor","") or ""),temporal_end=str(kw.pop("temporal_end","") or ""),comparison_side_label=str(kw.pop("comparison_side_label","") or "")); gap.qrf_operator=d["operator"]; gap.description=description; gap.required_fields=[gap.temporal_axis] if gap.temporal_axis else []; return gap

    def _controller_gaps(self,d,question,frame):
        op=d["operator"]; req=d.get("requirements") or []
        if op=="MULTI_OPTION": return [self._rc_gap("g_options","OPTION_CONTEXT",d,"Shared participant-specific evidence needed to evaluate every visible option")]
        if op=="TEMPORAL":
            t=d["temporal"]; r=req[0] if req else {}; return [self._rc_gap("g_temporal",r.get("role") or "GENERIC_EVIDENCE",d,r.get("description") or f"Locate temporal evidence for: {question}",target_entities=r.get("target_entities") or d["target_entities"],target_property=r.get("target_property") or "",temporal_axis=r.get("temporal_axis") or t["axis"] or "event_time",temporal_relation=r.get("temporal_relation") or t["relation"] or "LOCATE",temporal_anchor=r.get("temporal_anchor") or t["anchor"],temporal_end=r.get("temporal_end") or t["end"])]
        if op=="COMPARISON":
            out=[]; sides=d.get("comparison_sides") or []
            for i in range(2):
                s=sides[i] if i<len(sides) else {}; r=req[i] if i<len(req) else {}; out.append(self._rc_gap(f"g_side_{i}","COMPARISON_SIDE",d,r.get("description") or f"Evidence for comparison {s.get('label') or i+1}: {question}",target_entities=s.get("target_entities") or r.get("target_entities") or d["target_entities"],target_property=s.get("target_property") or r.get("target_property") or "",temporal_axis=s.get("temporal_axis") or r.get("temporal_axis") or "",temporal_relation="EXACT" if (s.get("temporal_anchor") or r.get("temporal_anchor")) else "",temporal_anchor=s.get("temporal_anchor") or r.get("temporal_anchor") or "",comparison_side_label=str(s.get("label") or f"side_{i}")))
            return out
        if op=="CAUSAL":
            if d.get("causal_mode")=="STORED": roles=["FOCAL_TRIGGER","CAUSAL_BRIDGE","OUTCOME"]
            else: roles=["FOCAL_TRIGGER","PRIOR_TRAJECTORY","OUTCOME"]
            return [self._rc_gap(f"g_{i}",role,d,next((r.get("description") for r in req if r.get("role")==role),"") or f"{role} evidence for: {question}") for i,role in enumerate(roles)]
        if op=="DECISION":
            roles=["FOCAL_STATE","CONSTRAINT"]+["PRIOR_TRAJECTORY"] if any(r.get("role")=="PRIOR_TRAJECTORY" for r in req) else ["FOCAL_STATE","CONSTRAINT"]
            return [self._rc_gap(f"g_{i}",role,d,next((r.get("description") for r in req if r.get("role")==role),"") or f"{role} evidence for: {question}") for i,role in enumerate(roles)]
        if op=="STATE": return [self._rc_gap("g_state","GENERIC_EVIDENCE",d,(req[0].get("description") if req else "") or f"Current state required for: {question}")]
        if op=="MULTI_HOP":
            items=req[:4] or [{"role":"GENERIC_EVIDENCE","description":f"Independent evidence component 1 for: {question}"},{"role":"PRIOR_TRAJECTORY","description":f"Independent evidence component 2 for: {question}"}]
            if len(items)==1: items.append({"role":"PRIOR_TRAJECTORY","description":f"Additional independent evidence needed for: {question}"})
            return [self._rc_gap(f"g_{i}",r.get("role") if r.get("role") in VALID_ROLES else "GENERIC_EVIDENCE",d,r.get("description") or f"Evidence component {i+1}",target_entities=r.get("target_entities") or d["target_entities"],target_property=r.get("target_property") or "") for i,r in enumerate(items)]
        r=req[0] if req else {}; return [self._rc_gap("g_direct",r.get("role") or "GENERIC_EVIDENCE",d,r.get("description") or f"Answer-bearing evidence for: {question}",target_entities=r.get("target_entities") or d["target_entities"],target_property=r.get("target_property") or "")]

    def _controller_plan(self,d,question,frame):
        lattice=EvidenceLattice(); gaps=self._controller_gaps(d,question,frame)
        for gap in gaps: lattice.add_gap(gap)
        self._evidence_lattice=lattice; slots=lattice.to_legacy_slots(); tier="LARGE" if d["operator"] in {"MULTI_OPTION","CAUSAL","MULTI_HOP"} or len(slots)>=3 else ("MEDIUM" if d["operator"] in {"DECISION","COMPARISON"} or len(slots)>=2 else "SMALL")
        plan={"query_spec":{"target_entities":d["target_entities"],"target_property":"","answer_type":d["answer_slot"],"reasoning":"SYNTHESIS" if d["requires_inference"] else "NONE","requires_inference":d["requires_inference"],"temporal":dict(d["temporal"]),"causal_mode":d.get("causal_mode") or ""},"query_mode":d["operator"],"required_slots":slots,"seed_coverage":[],"operations":[],"need_evidence":True,"budget_tier":tier,"max_memories":RETRIEVAL_BUDGETS[tier]["max_memories"],"planner_fallback":False,"fallback_reason":"","valid":True,"option_coverage":[],"visible_options":dict(d["visible_options"])}
        plan["operations"]=self._compile_gap_operations(slots,question,tier,plan=plan); return plan
