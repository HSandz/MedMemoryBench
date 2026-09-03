"""Two-stage read contract for SmartMem0.

One LLM call owns natural-language semantics. Everything after its JSON contract
is deterministic: authorization, typed retrieval, recovery, and context proof.
"""

import json
import re
import unicodedata
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .canonicalization import state_identity
from .contracts import RETRIEVAL_BUDGETS, VALID_TEMPORAL_AXES
from .p1b_execution import EvidenceGap, EvidenceLattice

CONTROLLER_POLICY = """You are the single semantic controller of an evidence-grounded memory system.
Interpret QUESTION by meaning in any language, never by English keywords. Derive the independent evidence requirements from QUESTION first; SEEDS are only a tiny preview and may mark a requirement covered, never change the target to fit a seed.
subject_id is the owner/entity the memory is about. source_speakers only says who said it; never confuse them.

ANSWER is allowed only for ONE atomic stored fact/current state, fully supported by ONE seed, with no comparison, temporal population search, causal/multi-hop bridge, decision, or visible options. Otherwise PLAN. Complex operators always PLAN.
Operators: DIRECT, STATE, TEMPORAL, COMPARISON, CAUSAL, DECISION, MULTI_OPTION, MULTI_HOP.
Answer slots: ENTITY, VALUE, DATE, RELATIVE_TIME, OPTION_SET, TEXT.
Roles: ANSWER, FOCAL_STATE, PRIOR_TRAJECTORY, ACTION_RULE, CONSTRAINT, FOCAL_TRIGGER, CAUSAL_BRIDGE, OUTCOME, COMPARISON_SIDE, OPTION_CONTEXT, GENERIC_EVIDENCE.

Temporal semantics: use document_time for when something was documented/recorded/mentioned and event_time for when the underlying event happened. LOCATE means the requested date/time is unknown and must be found. EXACT means QUESTION already supplies the date/time as a filter. EARLIEST/LATEST only when QUESTION actually asks for an extremum. BEFORE/AFTER/BETWEEN require explicit bounds. A word meaning recent inside the described event does not by itself mean LATEST. Use RELATIVE_TIME when the answer is timing relative to an event rather than a calendar date.

Causality: causal_mode=STORED only when the answer requires an explicit remembered causal attribution/path. causal_mode=INFERRED when participant-specific endpoints/trajectory should be retrieved and general domain knowledge may explain the bridge. Never invent participant history.

Visible multiple-choice always PLAN. Options are retrieval probes, not four facts that must each exist in memory. Retrieve shared participant-specific evidence, then the final answerer evaluates every visible option together.
Return semantic requirements only, never retrieval operations or invented memory refs."""

CONTROLLER_SCHEMA = """Return JSON only:
{{"route":"ANSWER|PLAN|ABSTAIN","operator":"DIRECT|STATE|TEMPORAL|COMPARISON|CAUSAL|DECISION|MULTI_OPTION|MULTI_HOP","answer_slot":"ENTITY|VALUE|DATE|RELATIVE_TIME|OPTION_SET|TEXT","answer":"","support_refs":["$seed0"],"requires_inference":false,"subject_id":"","target_entities":[],"causal_mode":"|STORED|INFERRED","temporal":{{"axis":"","relation":"LOCATE|EXACT|EARLIEST|LATEST|BEFORE|AFTER|BETWEEN|","anchor":"","end":""}},"comparison_sides":[{{"label":"left","target_entities":[],"target_property":"","temporal_axis":"","temporal_anchor":""}},{{"label":"right","target_entities":[],"target_property":"","temporal_axis":"","temporal_anchor":""}}],"requirements":[{{"id":"r1","role":"ANSWER","description":"required participant-specific evidence","status":"COVERED|MISSING","refs":["$seed0"],"target_entities":[],"target_property":"","temporal_axis":"","temporal_relation":"","temporal_anchor":"","temporal_end":""}}]}}
QUESTION:\n{question}\nVISIBLE OPTIONS:\n{options}\nSYNTAX HINTS (dates/source speaker only; never evidence):\n{hints}\nSEEDS:\n{seeds}"""

VALID_OPERATORS = {"DIRECT","STATE","TEMPORAL","COMPARISON","CAUSAL","DECISION","MULTI_OPTION","MULTI_HOP"}
VALID_ROLES = {"ANSWER","FOCAL_STATE","PRIOR_TRAJECTORY","ACTION_RULE","CONSTRAINT","FOCAL_TRIGGER","CAUSAL_BRIDGE","OUTCOME","COMPARISON_SIDE","OPTION_CONTEXT","GENERIC_EVIDENCE"}
VALID_RELATIONS = {"LOCATE","EXACT","EARLIEST","LATEST","BEFORE","AFTER","BETWEEN",""}
COMPLEX = {"TEMPORAL","COMPARISON","CAUSAL","DECISION","MULTI_OPTION","MULTI_HOP"}


class ReadContractMixin:
    """Authoritative read-path controller and deterministic proof/recovery."""

    @staticmethod
    def _rc_terms(value: Any) -> List[str]:
        text = unicodedata.normalize("NFKC", str(value or "")).casefold().replace("_", " ").replace("-", " ")
        return [x for x in re.findall(r"\w+", text, flags=re.UNICODE) if len(x) > 1]

    def _rc_owner(self, value: Any) -> str:
        raw = "_".join(self._rc_terms(value))
        aliases = {"_".join(self._rc_terms(k)): "_".join(self._rc_terms(v)) for k,v in (getattr(self,"subject_aliases",{}) or {}).items()}
        return aliases.get(raw, raw)

    def _rc_owner_match(self, slot: Dict[str,Any], memory: Dict[str,Any]) -> bool:
        want = self._rc_owner(slot.get("subject_id") or slot.get("subject") or "")
        have = self._rc_owner(memory.get("subject_id") or memory.get("subject") or "")
        return not want or bool(have and have == want)

    def _planning_context_map(self, question: str, *args, **kwargs) -> List[Dict[str,Any]]:
        return []

    def _rc_seed_payload(self, seeds: Sequence[Dict[str,Any]]) -> List[Dict[str,Any]]:
        out=[]
        for i,m in enumerate(seeds[:3]):
            ident=state_identity(m) or ""
            out.append({"ref":f"$seed{i}","kind":m.get("kind"),"semantic_role":m.get("semantic_role"),"subject_id":m.get("subject_id") or m.get("subject"),"source_speakers":list(m.get("source_speakers",[])),"scope":m.get("scope"),"object":m.get("object_anchor"),"value":self._memory_value(m),"claim":str(m.get("claim") or "")[:320],"event_time":m.get("event_time"),"document_time":m.get("document_time"),"state_identity":ident,"is_state_head":bool(ident and self._is_state_head(m)),"status":m.get("_status",self._belief_status.get(m.get("id"),"active"))})
        return out

    def _rc_normalize_decision(self, parsed: Dict[str,Any], question: str, frame: Any) -> Dict[str,Any]:
        options=self._question_options(question) or {}
        route=str(parsed.get("route") or "PLAN").upper(); route=route if route in {"ANSWER","PLAN","ABSTAIN"} else "PLAN"
        op=str(parsed.get("operator") or ("MULTI_OPTION" if options else "DIRECT")).upper(); op=op if op in VALID_OPERATORS else "DIRECT"
        if options: op,route="MULTI_OPTION","PLAN"
        slot=str(parsed.get("answer_slot") or ("OPTION_SET" if options else "TEXT")).upper()
        temporal=parsed.get("temporal") if isinstance(parsed.get("temporal"),dict) else {}
        axis=str(temporal.get("axis") or "").lower(); axis=axis if axis in VALID_TEMPORAL_AXES else ""
        relation=str(temporal.get("relation") or "").upper(); relation=relation if relation in VALID_RELATIONS else ""
        anchor=str(temporal.get("anchor") or "").strip(); end=str(temporal.get("end") or "").strip()
        if op=="TEMPORAL":
            dates=[str(x) for x in (getattr(frame,"dates",()) or ()) if x]
            if not anchor and relation in {"EXACT","BEFORE","AFTER"} and len(dates)==1: anchor=dates[0]
            if relation=="BETWEEN" and not anchor and len(dates)>=2: anchor,end=dates[0],dates[1]
            axis=axis or "event_time"; relation=relation or "LOCATE"
            if relation=="EXACT" and not anchor: relation="LOCATE"
            route="PLAN"
        inference=bool(parsed.get("requires_inference",False))
        if op in COMPLEX: route="PLAN"
        if op in {"CAUSAL","DECISION","MULTI_OPTION","MULTI_HOP"}: inference=True
        causal=str(parsed.get("causal_mode") or "").upper()
        causal=causal if op=="CAUSAL" and causal in {"STORED","INFERRED"} else ("INFERRED" if op=="CAUSAL" else "")
        entities=[str(x) for x in (parsed.get("target_entities") or []) if str(x).strip()] if isinstance(parsed.get("target_entities"),list) else []
        req=[]
        for i,r in enumerate((parsed.get("requirements") or [])[:4] if isinstance(parsed.get("requirements"),list) else []):
            if not isinstance(r,dict): continue
            role=str(r.get("role") or "GENERIC_EVIDENCE").upper(); role=role if role in VALID_ROLES else "GENERIC_EVIDENCE"
            refs=[str(x) for x in (r.get("refs") or []) if str(x).startswith("$seed")][:3] if isinstance(r.get("refs"),list) else []
            req.append({"id":str(r.get("id") or f"r{i+1}"),"role":role,"description":str(r.get("description") or "").strip(),"status":str(r.get("status") or "MISSING").upper(),"refs":refs,"target_entities":[str(x) for x in (r.get("target_entities") or []) if str(x).strip()][:8] if isinstance(r.get("target_entities"),list) else [],"target_property":str(r.get("target_property") or "").strip(),"temporal_axis":str(r.get("temporal_axis") or "").lower(),"temporal_relation":str(r.get("temporal_relation") or "").upper(),"temporal_anchor":str(r.get("temporal_anchor") or "").strip(),"temporal_end":str(r.get("temporal_end") or "").strip()})
        sides=[x for x in (parsed.get("comparison_sides") or [])[:2] if isinstance(x,dict)] if isinstance(parsed.get("comparison_sides"),list) else []
        roles={r["role"] for r in req}
        if op=="DIRECT" and ("COMPARISON_SIDE" in roles or len([s for s in sides if s.get("target_entities") or s.get("target_property") or s.get("temporal_anchor")])>=2): op,route="COMPARISON","PLAN"
        elif op=="DIRECT" and ("CAUSAL_BRIDGE" in roles or (inference and {"FOCAL_TRIGGER","OUTCOME"}.issubset(roles))): op,route,causal="CAUSAL","PLAN",causal or "INFERRED"
        elif op=="DIRECT" and len(req)>1: op,route,inference="MULTI_HOP","PLAN",True
        support_refs=[str(x) for x in (parsed.get("support_refs") or []) if str(x).startswith("$seed")][:3]
        if route=="ANSWER" and not req and support_refs: req=[{"id":"r1","role":"ANSWER","description":"atomic answer evidence","status":"COVERED","refs":support_refs[:1],"target_entities":entities,"target_property":"","temporal_axis":"","temporal_relation":"","temporal_anchor":"","temporal_end":""}]
        if route=="ANSWER" and (inference or len(req)!=1 or len(support_refs)!=1): route="PLAN"; op="MULTI_HOP" if op=="DIRECT" else op
        return {"route":route,"operator":op,"answer_slot":slot,"answer":str(parsed.get("answer") or "").strip(),"support_refs":support_refs,"requires_inference":inference,"subject_id":self._rc_owner(parsed.get("subject_id") or ""),"target_entities":entities[:8],"causal_mode":causal,"temporal":{"axis":axis,"relation":relation,"anchor":anchor,"end":end},"comparison_sides":sides,"requirements":req,"visible_options":dict(options)}

    def _rc_answer_grounded(self, answer: str, memories: Sequence[Dict[str,Any]]) -> bool:
        a=unicodedata.normalize("NFKC",str(answer or "")).casefold().strip(); evidence=" ".join(str(v or "") for m in memories for v in (m.get("claim"),self._memory_value(m),m.get("verbatim_value"),m.get("object_anchor")," ".join(m.get("entities",[]))))
        e=unicodedata.normalize("NFKC",evidence).casefold()
        if not a: return False
        if a in e: return True
        nums=set(re.findall(r"\d+(?:\.\d+)?",a)); en=set(re.findall(r"\d+(?:\.\d+)?",e))
        if nums and not nums.issubset(en): return False
        at=self._rc_terms(a); et=set(self._rc_terms(e))
        if not at: return a in e
        return len(at)<=12 and sum(x in et for x in at)/len(at)>=0.8

    def _rc_requirement_match(self, req: Dict[str,Any], memory: Dict[str,Any], subject_id: str) -> bool:
        slot={"subject_id":subject_id,"target_entities":req.get("target_entities") or [],"target_property":req.get("target_property") or "","required_fields":[]}
        return self._slot_contract_match(slot,memory,strict_targets=True)

    def _authorize_controller_answer(self,d,seeds,frame):
        if d.get("operator") not in {"DIRECT","STATE"} or d.get("requires_inference") or d.get("visible_options"): return None,"NON_ATOMIC_ROUTE"
        refs=d.get("support_refs") or []
        if len(refs)!=1 or not d.get("answer"): return None,"ATOMIC_SUPPORT_REQUIRED"
        m=re.fullmatch(r"\$seed(\d+)",refs[0]);
        if not m or int(m.group(1))>=min(3,len(seeds)): return None,"INVALID_SUPPORT_REF"
        valid=self._validate_fast_support(refs[0],seeds,frame)
        if not valid: return None,"STRUCTURAL_SUPPORT_REJECTED"
        req=(d.get("requirements") or [{}])[0]
        if not self._rc_requirement_match(req,valid[0],d.get("subject_id") or self._rc_owner(valid[0].get("subject_id") or valid[0].get("subject"))): return None,"TARGET_SUPPORT_REJECTED"
        if d["operator"]=="STATE" and not self._is_state_head(valid[0]): return None,"STATE_REQUIRES_HEAD"
        if not self._rc_answer_grounded(d["answer"],valid): return None,"ANSWER_NOT_GROUNDED"
        return valid,"AUTHORIZED"

    def _semantic_controller(self,question,seeds,frame,context_map=None):
        del context_map
        options=self._question_options(question) or {}; hints={"dates":list(getattr(frame,"dates",()) or ()),"source_speaker":getattr(frame,"speaker_role",""),"hard_entities":list(getattr(frame,"hard_entities",()) or ())}
        prompt=CONTROLLER_POLICY+"\n"+CONTROLLER_SCHEMA.format(question=question,options=json.dumps(options,ensure_ascii=False),hints=json.dumps(hints,ensure_ascii=False),seeds=json.dumps(self._rc_seed_payload(seeds),ensure_ascii=False))
        try:
            response=self._llm_client.chat([{"role":"user","content":prompt}],temperature=0.0,max_tokens=640,response_format={"type":"json_object"})
            usage=self._response_usage(response,prompt); d=self._rc_normalize_decision(self._parse_json(response.content),question,frame)
            if not d["subject_id"]:
                owners={self._rc_owner(m.get("subject_id") or m.get("subject")) for m in seeds[:3] if m.get("subject_id") or m.get("subject")}; owners.discard("")
                if len(owners)==1: d["subject_id"]=next(iter(owners))
        except Exception as exc:
            d=self._rc_normalize_decision({"route":"PLAN","operator":"MULTI_OPTION" if options else "DIRECT"},question,frame); usage={}
            plan=self._controller_plan(d,question,frame)
            return None,plan,{"called":True,"route":"PLAN","decision_route":"PLAN","answer":"","support_ref":"","support_refs":[],"fallback_reason":"controller_error","error":str(exc),"usage":usage,"operator":d["operator"],"answer_slot":d["answer_slot"]}
        reason=""
        if d["route"]=="ANSWER":
            supports,reason=self._authorize_controller_answer(d,seeds,frame)
            if supports is not None:
                return supports,{}, {"called":True,"route":"DIRECT","decision_route":"ANSWER","answer":d["answer"],"support_ref":d["support_refs"][0],"support_refs":d["support_refs"],"fallback_reason":"","usage":usage,"operator":d["operator"],"answer_slot":d["answer_slot"],"requirement_count":len(d["requirements"])}
            if d["operator"]=="DIRECT": d["operator"]="MULTI_HOP"; d["requires_inference"]=True
        elif d["route"]=="ABSTAIN": reason="ABSTAIN_REQUIRES_BOUNDED_MEMORY_CHECK"
        d["route"]="PLAN"; plan=self._controller_plan(d,question,frame)
        return None,plan,{"called":True,"route":"PLAN","decision_route":"PLAN","answer":"","support_ref":"","support_refs":[],"fallback_reason":reason,"usage":usage,"operator":d["operator"],"answer_slot":d["answer_slot"],"requires_inference":d["requires_inference"],"causal_mode":d.get("causal_mode") or "","requirement_count":len(d["requirements"])}
