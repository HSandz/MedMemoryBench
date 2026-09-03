"""Two-stage semantic controller for SmartMem0 reads.

Natural-language semantics is interpreted once. The controller either answers from
compact seeds or declares covered/missing evidence; code owns retrieval execution.
"""

import json
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .contracts import RETRIEVAL_BUDGETS, VALID_TEMPORAL_AXES
from .p1b_execution import EvidenceGap, EvidenceLattice


CONTROLLER_PROMPT = """You are the single semantic controller for an evidence-grounded memory system.
Understand the QUESTION by meaning in any language; do not route by English keywords.
You see only a few SEEDS, not the full memory store.

Choose one route:
- ANSWER: seeds completely support every independent requirement. Return the final answer and only valid seed refs.
- PLAN: more subject-specific memory evidence is needed. Return covered seed requirements plus missing evidence gaps; never write retrieval operations.
- ABSTAIN: only if personal-memory retrieval cannot reasonably answer the question.

Conservative sufficiency rules: earliest/latest/range normally PLAN; decision/advice needs decision-changing state/rules, not topical evidence; comparison and multi-hop need every side/hop; participant-specific causality needs grounded endpoints and an explicit stored causal connection. Visible multiple-choice normally PLAN so every option can be evaluated.

Operators: DIRECT, STATE, TEMPORAL, COMPARISON, CAUSAL, DECISION, MULTI_OPTION, MULTI_HOP.
Answer slots: ENTITY, VALUE, DATE, RELATIVE_TIME, OPTION_SET, TEXT.
Roles: ANSWER, FOCAL_STATE, PRIOR_TRAJECTORY, ACTION_RULE, CONSTRAINT, ALTERNATIVE, FOCAL_TRIGGER, CAUSAL_BRIDGE, OUTCOME, COMPARISON_SIDE, OPTION_SUPPORT, GENERIC_EVIDENCE.
Temporal axes: event_time, document_time, effective_event_time, or empty.
Temporal relations: EXACT, EARLIEST, LATEST, BEFORE, AFTER, BETWEEN, or empty.
Use explicit query dates as anchors/bounds. Gaps are semantic requirements only and contain no memory IDs or operations. Keep controller-declared covered+gaps to at most 4; runtime may deterministically expand visible options.

Return JSON only:
{{
 "route":"ANSWER|PLAN|ABSTAIN", "operator":"DIRECT", "answer_slot":"TEXT",
 "answer":"", "support_refs":["$seed0"], "requires_inference":false,
 "target_entities":[],
 "temporal":{{"axis":"","relation":"","anchor":"","end":""}},
 "comparison_sides":[{{"label":"left_side","temporal_anchor":""}},{{"label":"right_side","temporal_anchor":""}}],
 "covered":[{{"id":"c1","role":"ANSWER","description":"already present evidence","refs":["$seed0"],"target_entities":[],"target_property":"","temporal_axis":"","temporal_relation":"","temporal_anchor":"","temporal_end":"","option_label":""}}],
 "gaps":[{{"id":"g1","role":"GENERIC_EVIDENCE","description":"missing evidence","target_entities":[],"target_property":"","temporal_axis":"","temporal_relation":"","temporal_anchor":"","temporal_end":"","option_label":""}}]
}}

QUESTION:\n{question}
VISIBLE OPTIONS:\n{options}
DETERMINISTIC SYNTAX HINTS (routing only, never evidence):\n{syntax_hints}
SEEDS:\n{seeds}
"""

VALID_OPERATORS = {"DIRECT", "STATE", "TEMPORAL", "COMPARISON", "CAUSAL", "DECISION", "MULTI_OPTION", "MULTI_HOP"}
VALID_SLOTS = {"ENTITY", "VALUE", "DATE", "RELATIVE_TIME", "OPTION_SET", "TEXT"}
VALID_ROLES = {"ANSWER", "FOCAL_STATE", "PRIOR_TRAJECTORY", "ACTION_RULE", "CONSTRAINT", "ALTERNATIVE", "FOCAL_TRIGGER", "CAUSAL_BRIDGE", "OUTCOME", "COMPARISON_SIDE", "OPTION_SUPPORT", "GENERIC_EVIDENCE"}
VALID_RELATIONS = {"EXACT", "EARLIEST", "LATEST", "BEFORE", "AFTER", "BETWEEN", ""}


class TwoStageControllerMixin:
    """One compact semantic LLM call followed by deterministic retrieval."""

    def _planning_context_map(self, question: str) -> List[Dict[str, Any]]:
        return []

    def _controller_seed_payload(self, seeds: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        from .canonicalization import state_identity
        out = []
        for index, memory in enumerate(seeds[:3]):
            identity = state_identity(memory) or ""
            out.append({
                "ref": f"$seed{index}", "kind": memory.get("kind"),
                "semantic_role": memory.get("semantic_role"),
                "subject": memory.get("subject_id") or memory.get("subject"),
                "object": memory.get("object_anchor"), "value": self._memory_value(memory),
                "claim": str(memory.get("claim") or "")[:320],
                "event_time": memory.get("event_time"), "document_time": memory.get("document_time"),
                "state_identity": identity, "is_state_head": bool(identity and self._is_state_head(memory)),
                "status": memory.get("_status", self._belief_status.get(memory.get("id"), "active")),
            })
        return out

    @staticmethod
    def _seed_index(ref: Any, count: int) -> Optional[int]:
        text = str(ref or "")
        if not text.startswith("$seed") or not text[5:].isdigit():
            return None
        index = int(text[5:])
        return index if index < min(3, count) else None

    def _normalize_controller_decision(self, parsed: Dict[str, Any], question: str, frame: Any) -> Dict[str, Any]:
        options = self._question_options(question) or {}
        route = str(parsed.get("route") or "PLAN").upper()
        route = route if route in {"ANSWER", "PLAN", "ABSTAIN"} else "PLAN"
        operator = str(parsed.get("operator") or ("MULTI_OPTION" if options else "DIRECT")).upper()
        operator = operator if operator in VALID_OPERATORS else ("MULTI_OPTION" if options else "DIRECT")
        if options:
            operator = "MULTI_OPTION"
        answer_slot = str(parsed.get("answer_slot") or ("OPTION_SET" if options else "TEXT")).upper()
        answer_slot = answer_slot if answer_slot in VALID_SLOTS else ("OPTION_SET" if options else "TEXT")
        temporal = parsed.get("temporal") if isinstance(parsed.get("temporal"), dict) else {}
        axis = str(temporal.get("axis") or "").lower()
        axis = axis if axis in VALID_TEMPORAL_AXES else ""
        relation = str(temporal.get("relation") or "").upper()
        relation = relation if relation in VALID_RELATIONS else ""
        entities = parsed.get("target_entities") if isinstance(parsed.get("target_entities"), list) else []
        return {
            "route": route, "operator": operator, "answer_slot": answer_slot,
            "answer": str(parsed.get("answer") or "").strip(),
            "support_refs": [str(v) for v in (parsed.get("support_refs") or []) if str(v).strip()][:3],
            "requires_inference": bool(parsed.get("requires_inference", False)),
            "target_entities": [str(v) for v in entities if str(v).strip()][:8],
            "temporal": {"axis": axis, "relation": relation, "anchor": str(temporal.get("anchor") or ""), "end": str(temporal.get("end") or "")},
            "comparison_sides": (parsed.get("comparison_sides") if isinstance(parsed.get("comparison_sides"), list) else [])[:2],
            "covered": parsed.get("covered") if isinstance(parsed.get("covered"), list) else [],
            "gaps": parsed.get("gaps") if isinstance(parsed.get("gaps"), list) else [],
            "visible_options": dict(options),
        }

    def _authorize_controller_answer(self, d: Dict[str, Any], seeds: List[Dict[str, Any]], frame: Any) -> Tuple[Optional[List[Dict[str, Any]]], str]:
        if not d["answer"]:
            return None, "EMPTY_ANSWER"
        if d["visible_options"]:
            return None, "MULTI_OPTION_REQUIRES_EVIDENCE_BUNDLE"
        refs = list(dict.fromkeys(d["support_refs"]))
        if not refs:
            return None, "NO_SUPPORT_REFS"
        supports = []
        for ref in refs:
            if self._seed_index(ref, len(seeds)) is None:
                return None, "INVALID_SUPPORT_REF"
            valid = self._validate_fast_support(ref, seeds, frame)
            if not valid:
                return None, "STRUCTURAL_SUPPORT_REJECTED"
            for memory in valid:
                if all(item["id"] != memory["id"] for item in supports):
                    supports.append(memory)
        op, temporal = d["operator"], d["temporal"]
        if op == "STATE" and (len(supports) != 1 or not self._is_state_head(supports[0])):
            return None, "STATE_REQUIRES_CANONICAL_HEAD"
        if op == "TEMPORAL":
            if temporal["relation"] not in {"", "EXACT"}:
                return None, "TEMPORAL_SCOPE_REQUIRES_PLAN"
            if temporal["axis"] and not any(self._date_for(m, temporal["axis"]) for m in supports):
                return None, "TEMPORAL_AXIS_UNSUPPORTED"
        if op == "COMPARISON" and len(supports) < 2:
            return None, "COMPARISON_REQUIRES_TWO_SIDES"
        if op == "MULTI_HOP" and len(supports) < 2:
            return None, "MULTI_HOP_REQUIRES_MULTIPLE_SUPPORTS"
        if op == "DECISION":
            return None, "DECISION_REQUIRES_PLAN"
        if op == "CAUSAL":
            ids = {m["id"] for m in supports}
            causal = any(str(r.get("type") or "").upper() == "CAUSES" and r.get("source_id") in ids and r.get("target_id") in ids for r in self._relations)
            if len(ids) < 2 or not causal:
                return None, "CAUSAL_REQUIRES_STORED_PATH"
        return supports, "AUTHORIZED"

    def _gap_from_spec(self, raw: Dict[str, Any], index: int, d: Dict[str, Any], frame: Any, prefix: str) -> EvidenceGap:
        role = str(raw.get("role") or "GENERIC_EVIDENCE").upper()
        role = role if role in VALID_ROLES else "GENERIC_EVIDENCE"
        temporal = d["temporal"]
        axis = str(raw.get("temporal_axis") or "").lower()
        axis = axis if axis in VALID_TEMPORAL_AXES else ""
        relation = str(raw.get("temporal_relation") or "").upper()
        relation = relation if relation in VALID_RELATIONS else ""
        if d["operator"] == "TEMPORAL":
            axis = temporal["axis"] or axis or "event_time"
            relation = temporal["relation"] or relation or "EXACT"
        if role == "CAUSAL_BRIDGE":
            axis, relation = "", ""
        entities = raw.get("target_entities") if isinstance(raw.get("target_entities"), list) else []
        if not entities:
            entities = d["target_entities"]
        gap = EvidenceGap(
            id=str(raw.get("id") or f"{prefix}{index}"), role=role, required=True,
            subject_id=getattr(frame, "speaker_role", "") or "primary_user",
            target_entities=[str(v) for v in entities if str(v).strip()][:8],
            target_property=str(raw.get("target_property") or ""),
            temporal_axis=axis, temporal_relation=relation,
            temporal_anchor=str(raw.get("temporal_anchor") or temporal["anchor"] or ""),
            temporal_end=str(raw.get("temporal_end") or temporal["end"] or ""),
            option_label=str(raw.get("option_label") or "").upper(),
        )
        gap.qrf_operator = d["operator"]
        gap.description = str(raw.get("description") or "").strip() or f"Evidence required for {role}"
        if axis:
            gap.required_fields = [axis]
        return gap

    def _new_gap(self, gap_id: str, role: str, d: Dict[str, Any], frame: Any, description: str, **kwargs) -> EvidenceGap:
        gap = EvidenceGap(id=gap_id, role=role, required=True,
            subject_id=getattr(frame, "speaker_role", "") or "primary_user",
            target_entities=list(d["target_entities"]), **kwargs)
        gap.qrf_operator = d["operator"]
        gap.description = description
        return gap

    def _normalize_requirements(self, d: Dict[str, Any], question: str, frame: Any) -> Tuple[List[Tuple[EvidenceGap, List[str]]], List[EvidenceGap]]:
        covered, missing, used = [], [], set()
        for i, raw in enumerate(d["covered"][:4]):
            if not isinstance(raw, dict):
                continue
            gap = self._gap_from_spec(raw, i, d, frame, "c")
            if gap.id in used:
                gap.id = f"c{i}"
            used.add(gap.id)
            refs = [str(r) for r in (raw.get("refs") or []) if str(r).startswith("$seed") and str(r)[5:].isdigit() and int(str(r)[5:]) < 3]
            (covered if refs else missing).append((gap, refs[:3]) if refs else gap)
        for i, raw in enumerate(d["gaps"][:4]):
            if not isinstance(raw, dict):
                continue
            gap = self._gap_from_spec(raw, i, d, frame, "g")
            if gap.id in used:
                gap.id = f"g{i}"
            used.add(gap.id)
            missing.append(gap)
        op, options = d["operator"], d["visible_options"]
        all_gaps = lambda: [*(g for g, _ in covered), *missing]
        if op == "MULTI_OPTION":
            existing = {g.option_label: g for g in missing if g.option_label}
            covered, missing = [], []
            for label, proposition in list(options.items())[:8]:
                gap = existing.get(str(label).upper()) or self._new_gap(f"g_opt_{label}", "OPTION_SUPPORT", d, frame, "")
                gap.option_label, gap.option_proposition = str(label).upper(), str(proposition)
                gap.target_property = str(proposition)
                gap.description = f"Evidence supporting or refuting option {label}: {proposition}"
                missing.append(gap)
            return [], missing
        if op == "COMPARISON":
            sides = d["comparison_sides"]
            while sum(g.role == "COMPARISON_SIDE" for g in all_gaps()) < 2:
                idx = sum(g.role == "COMPARISON_SIDE" for g in all_gaps())
                spec = sides[idx] if idx < len(sides) and isinstance(sides[idx], dict) else {}
                anchor = str(spec.get("temporal_anchor") or "")
                gap = self._new_gap(f"g_comp_{idx}", "COMPARISON_SIDE", d, frame,
                    f"Evidence for comparison {spec.get('label') or f'side_{idx}'}: {question}",
                    temporal_axis="event_time" if anchor else "", temporal_relation="EXACT" if anchor else "",
                    temporal_anchor=anchor, comparison_side_label=str(spec.get("label") or f"side_{idx}"))
                missing.append(gap)
        if op == "CAUSAL":
            roles = {g.role for g in all_gaps()}
            for role, gid, text in (("FOCAL_TRIGGER","g_trigger","Grounded trigger or exposure"),("CAUSAL_BRIDGE","g_bridge","Stored causal connection between grounded endpoints"),("OUTCOME","g_outcome","Grounded outcome or resulting state")):
                if role not in roles:
                    missing.insert(0, self._new_gap(gid, role, d, frame, f"{text}: {question}")); roles.add(role)
        if op == "DECISION":
            roles = {g.role for g in all_gaps()}
            for role, gid, text in (("FOCAL_STATE","g_focal","Current focal evidence relevant to the decision"),("CONSTRAINT","g_constraint","Applicable participant-specific rule, policy, or constraint")):
                if role not in roles:
                    missing.insert(0, self._new_gap(gid, role, d, frame, f"{text}: {question}")); roles.add(role)
        if op == "STATE" and not covered and not missing:
            missing.append(self._new_gap("g_state", "GENERIC_EVIDENCE", d, frame, f"Current/latest state required to answer: {question}"))
        if op == "TEMPORAL" and not covered and not missing:
            t = d["temporal"]
            missing.append(self._new_gap("g_temporal", "GENERIC_EVIDENCE", d, frame,
                f"Temporal evidence required to answer: {question}", temporal_axis=t["axis"] or "event_time",
                temporal_relation=t["relation"] or "EXACT", temporal_anchor=t["anchor"], temporal_end=t["end"]))
        if not covered and not missing:
            missing.append(self._new_gap("g_direct", "GENERIC_EVIDENCE", d, frame, f"Missing answer-bearing evidence: {question}"))
        role_order = {"FOCAL_TRIGGER":0,"CAUSAL_BRIDGE":1,"OUTCOME":2,"FOCAL_STATE":0,"CONSTRAINT":1,"ACTION_RULE":2,"PRIOR_TRAJECTORY":3,"COMPARISON_SIDE":0}
        missing.sort(key=lambda gap: role_order.get(gap.role, 10))
        return covered[:4], missing[:4]

    def _controller_plan(self, d: Dict[str, Any], question: str, frame: Any) -> Dict[str, Any]:
        covered, missing = self._normalize_requirements(d, question, frame)
        lattice = EvidenceLattice()
        for gap, _ in covered: lattice.add_gap(gap)
        for gap in missing: lattice.add_gap(gap)
        slots = lattice.to_legacy_slots()
        seed_coverage = [{"slot_id":gap.id,"refs":refs} for gap, refs in covered]
        missing_ids = {gap.id for gap in missing}
        missing_slots = [slot for slot in slots if slot["id"] in missing_ids]
        op = d["operator"]
        if op == "MULTI_OPTION" and missing:
            operations = [{"op":"SEMANTIC_SEARCH","query":self._question_stem(question),"top_k":8,
                "strategy":"SHARED_OPTIONS","option_queries":list(d["visible_options"].values()),
                "produces":[gap.id for gap in missing]}]
            tier = "LARGE"
        elif op == "DECISION" and missing:
            history = [g.id for g in missing if g.role == "PRIOR_TRAJECTORY"]
            focal = [g.id for g in missing if g.role != "PRIOR_TRAJECTORY"]
            operations = []
            if focal: operations.append({"op":"SEMANTIC_SEARCH","query":question,"top_k":5,"strategy":"DECISION_BUNDLE","produces":focal})
            if history: operations.append({"op":"SEMANTIC_SEARCH","query":question,"top_k":5,"strategy":"TRAJECTORY","produces":history})
            tier = "MEDIUM"
        else:
            tier = "LARGE" if op == "CAUSAL" or len(slots) >= 3 else ("MEDIUM" if len(slots) >= 2 else "SMALL")
            operations = self._compile_gap_operations(missing_slots, question, tier)
        return {
            "query_spec":{"target_entities":d["target_entities"],"target_property":"","answer_type":d["answer_slot"],
                "reasoning":"SYNTHESIS" if d["requires_inference"] else "NONE","requires_inference":d["requires_inference"],"temporal":dict(d["temporal"])},
            "query_mode":op,"required_slots":slots,"seed_coverage":seed_coverage,"operations":operations,
            "need_evidence":True,"budget_tier":tier,"max_memories":RETRIEVAL_BUDGETS[tier]["max_memories"],
            "planner_fallback":False,"fallback_reason":"","valid":True,"option_coverage":[],
        }

    def _semantic_controller(self, question: str, seeds: List[Dict[str, Any]], frame: Any, context_map: Optional[List[Dict[str, Any]]] = None) -> Tuple[Optional[List[Dict[str, Any]]], Dict[str, Any], Dict[str, Any]]:
        options = self._question_options(question) or {}
        hints = {"dates":[str(v) for v in (getattr(frame,"dates",()) or ()) if v],"speaker_role":getattr(frame,"speaker_role",""),"hard_entities":list(getattr(frame,"hard_entities",()) or ())}
        prompt = CONTROLLER_PROMPT.format(question=question, options=json.dumps(options, ensure_ascii=False), syntax_hints=json.dumps(hints, ensure_ascii=False), seeds=json.dumps(self._controller_seed_payload(seeds), ensure_ascii=False))
        try:
            response = self._llm_client.chat([{"role":"user","content":prompt}], temperature=0.0, max_tokens=512, response_format={"type":"json_object"})
            usage = self._response_usage(response, prompt)
            decision = self._normalize_controller_decision(self._parse_json(response.content), question, frame)
        except Exception as exc:
            decision = self._normalize_controller_decision({"route":"PLAN","operator":"MULTI_OPTION" if options else "DIRECT"}, question, frame)
            plan = self._controller_plan(decision, question, frame)
            return None, plan, {"called":True,"route":"PLAN","decision_route":"PLAN","answer":"","support_ref":"","support_refs":[],"fallback_reason":"controller_error","error":str(exc),"usage":{},"operator":decision["operator"]}
        fallback_reason = ""
        if decision["route"] == "ANSWER":
            supports, reason = self._authorize_controller_answer(decision, seeds, frame)
            if supports is not None:
                refs = list(dict.fromkeys(decision["support_refs"]))
                return supports, {}, {"called":True,"route":"DIRECT","decision_route":"ANSWER","answer":decision["answer"],"support_ref":refs[0] if refs else "","support_refs":refs,"fallback_reason":"","usage":usage,"operator":decision["operator"],"answer_slot":decision["answer_slot"]}
            decision["route"], fallback_reason = "PLAN", reason
        elif decision["route"] == "ABSTAIN":
            decision["route"], fallback_reason = "PLAN", "ABSTAIN_REQUIRES_RETRIEVAL_CHECK"
        plan = self._controller_plan(decision, question, frame)
        return None, plan, {"called":True,"route":"PLAN","decision_route":"PLAN","answer":"","support_ref":"","support_refs":[],"fallback_reason":fallback_reason,"usage":usage,"operator":decision["operator"],"answer_slot":decision["answer_slot"],"requires_inference":decision["requires_inference"]}
