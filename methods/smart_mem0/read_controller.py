"""Minimal two-stage semantic controller for SmartMem0 reads.

The LLM owns only natural-language intent. Durable/canonical memory identity is
resolved deterministically from fields created by the write path.
"""

import json
import re
import unicodedata
from typing import Any, Dict, List, Sequence

from .canonicalization import state_identity
from .contracts import VALID_TEMPORAL_AXES
from .p1b_execution import EvidenceLattice

CONTROLLER_POLICY = """You are the single semantic controller of an evidence-grounded memory system.
Interpret QUESTION by meaning in any language. Derive what evidence the QUESTION requires before looking at SEEDS. SEEDS may justify a direct answer, but must never change the target requested by QUESTION.

Return only a small semantic contract. Never invent state_key, object_anchor, scope, entity IDs, memory IDs, retrieval operations, or any other memory-store key. Code resolves memory identity from the durable write-time schema.

TARGET RULE: every non-empty target must be copied as one contiguous span from QUESTION. Never copy a target from SEEDS and never replace an unknown answer with a candidate seen in a seed.

ANSWER is allowed only for one atomic stored fact/current state fully supported by exactly one seed. TEMPORAL, COMPARISON, CAUSAL, DECISION, MULTI_OPTION, and MULTI_HOP always PLAN.
Operators: DIRECT, STATE, TEMPORAL, COMPARISON, CAUSAL, DECISION, MULTI_OPTION, MULTI_HOP.
Answer slots: ENTITY, VALUE, DATE, RELATIVE_TIME, OPTION_SET, TEXT.
Roles: ANSWER, FOCAL_STATE, PRIOR_TRAJECTORY, ACTION_RULE, CONSTRAINT, FOCAL_TRIGGER, CAUSAL_BRIDGE, OUTCOME, COMPARAND, OPTION_CONTEXT, GENERIC_EVIDENCE.

COMPARISON: emit exactly two COMPARAND requirements. side exists only for these requirements and must be LEFT or RIGHT. Do not represent comparison as generic MULTI_HOP.
MULTI_OPTION: visible options are retrieval probes, not required memory facts. Emit only shared participant-specific evidence requirements needed to evaluate the choices. An option may have no personal-memory match; absence is not evidence that it is false.
TEMPORAL: document_time is when something was documented/recorded/mentioned; event_time is when the underlying event happened. LOCATE means the requested date/time is unknown. EXACT means QUESTION supplies the date/time as a filter. EARLIEST/LATEST only when QUESTION actually requests an extremum. A word meaning recent inside the target description does not itself mean LATEST. Use RELATIVE_TIME when the answer is timing relative to another event.
CAUSAL: causal_mode=STORED only for an explicit remembered causal attribution/path. causal_mode=INFERRED retrieves grounded participant endpoints/trajectory and lets the final answer model explain a general-domain bridge.
subject_id is who the memory is about; source speaker is only who said it.
"""

CONTROLLER_SCHEMA = """Return JSON only:
{{"route":"ANSWER|PLAN|ABSTAIN","operator":"DIRECT|STATE|TEMPORAL|COMPARISON|CAUSAL|DECISION|MULTI_OPTION|MULTI_HOP","answer_slot":"ENTITY|VALUE|DATE|RELATIVE_TIME|OPTION_SET|TEXT","answer":"","support_refs":["$seed0"],"requires_inference":false,"subject_id":"","target":"exact contiguous span copied from QUESTION or empty","causal_mode":"|STORED|INFERRED","temporal":{{"axis":"","relation":"LOCATE|EXACT|EARLIEST|LATEST|BEFORE|AFTER|BETWEEN|","anchor":"","end":""}},"requirements":[{{"id":"r1","role":"ANSWER","target":"exact contiguous span copied from QUESTION or empty","side":"|LEFT|RIGHT","temporal_axis":"","temporal_relation":"","temporal_anchor":"","temporal_end":""}}]}}
QUESTION:\n{question}\nVISIBLE OPTIONS:\n{options}\nSYNTAX HINTS (routing constraints only; never evidence):\n{hints}\nSEEDS:\n{seeds}"""

VALID_OPERATORS = {"DIRECT", "STATE", "TEMPORAL", "COMPARISON", "CAUSAL", "DECISION", "MULTI_OPTION", "MULTI_HOP"}
VALID_SLOTS = {"ENTITY", "VALUE", "DATE", "RELATIVE_TIME", "OPTION_SET", "TEXT"}
VALID_ROLES = {"ANSWER", "FOCAL_STATE", "PRIOR_TRAJECTORY", "ACTION_RULE", "CONSTRAINT", "FOCAL_TRIGGER", "CAUSAL_BRIDGE", "OUTCOME", "COMPARAND", "OPTION_CONTEXT", "GENERIC_EVIDENCE"}
VALID_RELATIONS = {"LOCATE", "EXACT", "EARLIEST", "LATEST", "BEFORE", "AFTER", "BETWEEN", ""}
COMPLEX = {"TEMPORAL", "COMPARISON", "CAUSAL", "DECISION", "MULTI_OPTION", "MULTI_HOP"}


class ReadContractMixin:
    @staticmethod
    def _rc_text(value: Any) -> str:
        return " ".join(unicodedata.normalize("NFKC", str(value or "")).casefold().split())

    @classmethod
    def _rc_terms(cls, value: Any) -> List[str]:
        return [x for x in re.findall(r"\w+", cls._rc_text(value).replace("_", " ").replace("-", " "), flags=re.UNICODE) if len(x) > 1]

    def _rc_owner(self, value: Any) -> str:
        raw = "_".join(self._rc_terms(value))
        aliases = {"_".join(self._rc_terms(k)): "_".join(self._rc_terms(v)) for k, v in (getattr(self, "subject_aliases", {}) or {}).items()}
        return aliases.get(raw, raw)

    def _rc_owner_match(self, slot: Dict[str, Any], memory: Dict[str, Any]) -> bool:
        want = self._rc_owner(slot.get("subject_id") or slot.get("subject") or "")
        have = self._rc_owner(memory.get("subject_id") or memory.get("subject") or "")
        return not want or bool(have and have == want)

    def _rc_known_subject(self, value: Any) -> str:
        candidate = self._rc_owner(value)
        if not candidate:
            return ""
        known = {self._rc_owner(memory.get("subject_id") or memory.get("subject") or "") for memory in self._memories}
        known.discard("")
        return candidate if candidate in known else ""

    def _rc_target_from_question(self, target: Any, question: str) -> str:
        raw = str(target or "").strip()
        if not raw:
            return ""
        return raw if self._rc_text(raw) in self._rc_text(question) else ""

    def _rc_memory_concept_keys(self, memory: Dict[str, Any]) -> List[str]:
        values: List[Any] = [memory.get("scope"), memory.get("state_key"), memory.get("object_anchor")]
        values.extend(memory.get("entities") or [])
        values.extend(memory.get("scope_entities") or [])
        identity = state_identity(memory)
        if identity:
            values.extend(str(identity).split("::"))
        out: List[str] = []
        for value in values:
            key = " ".join(self._rc_terms(value))
            if key and key not in out:
                out.append(key)
        return out

    def _rc_resolve_target_keys(self, target: str, subject_id: str = "") -> List[str]:
        terms = set(self._rc_terms(target))
        if not terms:
            return []
        scored: Dict[str, float] = {}
        wanted_owner = self._rc_owner(subject_id)
        for memory in self._memories:
            if wanted_owner and self._rc_owner(memory.get("subject_id") or memory.get("subject") or "") != wanted_owner:
                continue
            for key in self._rc_memory_concept_keys(memory):
                key_terms = set(self._rc_terms(key))
                if not key_terms:
                    continue
                overlap = len(terms & key_terms)
                exact = key in self._rc_text(target) or self._rc_text(target) in key
                score = 1.0 if exact else overlap / max(1, len(key_terms))
                if exact or (overlap and score >= 0.5):
                    scored[key] = max(scored.get(key, 0.0), score)
        return [key for key, _ in sorted(scored.items(), key=lambda item: (-item[1], len(item[0]), item[0]))[:8]]

    def _planning_context_map(self, question: str, *args, **kwargs) -> List[Dict[str, Any]]:
        return []

    def _rc_seed_payload(self, seeds: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out = []
        for i, memory in enumerate(seeds[:3]):
            identity = state_identity(memory) or ""
            out.append({"ref": f"$seed{i}", "kind": memory.get("kind"), "semantic_role": memory.get("semantic_role"), "subject_id": memory.get("subject_id") or memory.get("subject"), "source_speakers": list(memory.get("source_speakers", [])), "scope": memory.get("scope"), "object": memory.get("object_anchor"), "value": self._memory_value(memory), "claim": str(memory.get("claim") or "")[:320], "event_time": memory.get("event_time"), "document_time": memory.get("document_time"), "state_identity": identity, "is_state_head": bool(identity and self._is_state_head(memory)), "status": memory.get("_status", self._belief_status.get(memory.get("id"), "active"))})
        return out

    def _rc_normalize_requirement(self, raw: Dict[str, Any], index: int, question: str, fallback_target: str) -> Dict[str, Any]:
        role = str(raw.get("role") or "GENERIC_EVIDENCE").upper()
        role = role if role in VALID_ROLES else "GENERIC_EVIDENCE"
        target = self._rc_target_from_question(raw.get("target"), question) or fallback_target
        axis = str(raw.get("temporal_axis") or "").lower()
        axis = axis if axis in VALID_TEMPORAL_AXES else ""
        relation = str(raw.get("temporal_relation") or "").upper()
        relation = relation if relation in VALID_RELATIONS else ""
        side = str(raw.get("side") or "").upper()
        side = side if role == "COMPARAND" and side in {"LEFT", "RIGHT"} else ""
        return {"id": str(raw.get("id") or f"r{index + 1}"), "role": role, "target": target, "side": side, "temporal_axis": axis, "temporal_relation": relation, "temporal_anchor": str(raw.get("temporal_anchor") or "").strip(), "temporal_end": str(raw.get("temporal_end") or "").strip()}

    def _rc_normalize_decision(self, parsed: Dict[str, Any], question: str, frame: Any) -> Dict[str, Any]:
        options = self._question_options(question) or {}
        route = str(parsed.get("route") or "PLAN").upper()
        route = route if route in {"ANSWER", "PLAN", "ABSTAIN"} else "PLAN"
        operator = str(parsed.get("operator") or ("MULTI_OPTION" if options else "DIRECT")).upper()
        operator = operator if operator in VALID_OPERATORS else "DIRECT"
        if options:
            operator, route = "MULTI_OPTION", "PLAN"
        answer_slot = str(parsed.get("answer_slot") or ("OPTION_SET" if options else "TEXT")).upper()
        answer_slot = answer_slot if answer_slot in VALID_SLOTS else ("OPTION_SET" if options else "TEXT")
        target = self._rc_target_from_question(parsed.get("target"), question)
        target_rejected = bool(str(parsed.get("target") or "").strip() and not target)
        temporal = parsed.get("temporal") if isinstance(parsed.get("temporal"), dict) else {}
        axis = str(temporal.get("axis") or "").lower()
        axis = axis if axis in VALID_TEMPORAL_AXES else ""
        relation = str(temporal.get("relation") or "").upper()
        relation = relation if relation in VALID_RELATIONS else ""
        anchor, end = str(temporal.get("anchor") or "").strip(), str(temporal.get("end") or "").strip()
        dates = [str(x) for x in (getattr(frame, "dates", ()) or ()) if x]
        if operator == "TEMPORAL":
            axis, relation = axis or "event_time", relation or "LOCATE"
            if relation in {"EXACT", "BEFORE", "AFTER"} and not anchor and len(dates) == 1:
                anchor = dates[0]
            if relation == "BETWEEN" and not anchor and len(dates) >= 2:
                anchor, end = dates[0], dates[1]
            if relation == "EXACT" and not anchor:
                relation = "LOCATE"
            route = "PLAN"
        raw_requirements = parsed.get("requirements") if isinstance(parsed.get("requirements"), list) else []
        requirements = [self._rc_normalize_requirement(raw, i, question, target) for i, raw in enumerate(raw_requirements[:4]) if isinstance(raw, dict)]
        if operator == "COMPARISON":
            route = "PLAN"
            source = requirements[:2]
            while len(source) < 2:
                source.append({"id": f"r{len(source)+1}", "role": "COMPARAND", "target": target, "side": "", "temporal_axis": "", "temporal_relation": "", "temporal_anchor": "", "temporal_end": ""})
            normalized = []
            for i, req in enumerate(source[:2]):
                req = dict(req)
                req["role"], req["side"] = "COMPARAND", "LEFT" if i == 0 else "RIGHT"
                req["target"] = req.get("target") or target
                if len(dates) >= 2 and not req.get("temporal_anchor"):
                    req["temporal_axis"], req["temporal_relation"], req["temporal_anchor"] = req.get("temporal_axis") or "event_time", "EXACT", dates[i]
                normalized.append(req)
            requirements = normalized
        else:
            for req in requirements:
                req["side"] = ""
        if operator == "MULTI_OPTION":
            route = "PLAN"
            requirements = [req for req in requirements if req["role"] != "COMPARAND"][:3]
            if not requirements:
                requirements = [{"id": "r_options", "role": "OPTION_CONTEXT", "target": target, "side": "", "temporal_axis": "", "temporal_relation": "", "temporal_anchor": "", "temporal_end": ""}]
        inference = bool(parsed.get("requires_inference", False))
        if operator in COMPLEX:
            route = "PLAN"
        if operator in {"CAUSAL", "DECISION", "MULTI_OPTION", "MULTI_HOP", "COMPARISON"}:
            inference = True
        roles = {req["role"] for req in requirements}
        if operator == "DIRECT" and "COMPARAND" in roles:
            return self._rc_normalize_decision({**parsed, "operator": "COMPARISON", "requirements": raw_requirements}, question, frame)
        if operator == "DIRECT" and ("CAUSAL_BRIDGE" in roles or (inference and {"FOCAL_TRIGGER", "OUTCOME"}.issubset(roles))):
            operator, route = "CAUSAL", "PLAN"
        elif operator == "DIRECT" and len(requirements) > 1:
            operator, route, inference = "MULTI_HOP", "PLAN", True
        causal_mode = str(parsed.get("causal_mode") or "").upper()
        causal_mode = causal_mode if operator == "CAUSAL" and causal_mode in {"STORED", "INFERRED"} else ("INFERRED" if operator == "CAUSAL" else "")
        support_refs = [str(ref) for ref in (parsed.get("support_refs") or []) if re.fullmatch(r"\$seed[0-2]", str(ref))][:1]
        subject_id = self._rc_known_subject(parsed.get("subject_id") or "")
        if route == "ANSWER" and not requirements:
            requirements = [{"id": "r1", "role": "ANSWER", "target": target, "side": "", "temporal_axis": "", "temporal_relation": "", "temporal_anchor": "", "temporal_end": ""}]
        if route == "ANSWER" and (operator not in {"DIRECT", "STATE"} or inference or len(requirements) != 1 or len(support_refs) != 1):
            route = "PLAN"
            if operator == "DIRECT":
                operator, inference = "MULTI_HOP", True
        return {"route": route, "operator": operator, "answer_slot": answer_slot, "answer": str(parsed.get("answer") or "").strip(), "support_refs": support_refs, "requires_inference": inference, "subject_id": subject_id, "target": target, "target_rejected": target_rejected, "causal_mode": causal_mode, "temporal": {"axis": axis, "relation": relation, "anchor": anchor, "end": end}, "requirements": requirements, "visible_options": dict(options)}

    def _rc_answer_grounded(self, answer: str, memories: Sequence[Dict[str, Any]]) -> bool:
        answer_norm = self._rc_text(answer)
        evidence = self._rc_text(" ".join(str(value or "") for memory in memories for value in (memory.get("claim"), self._memory_value(memory), memory.get("verbatim_value"), memory.get("object_anchor"), " ".join(memory.get("entities", [])))))
        if not answer_norm:
            return False
        if answer_norm in evidence:
            return True
        nums, evidence_nums = set(re.findall(r"\d+(?:\.\d+)?", answer_norm)), set(re.findall(r"\d+(?:\.\d+)?", evidence))
        if nums and not nums.issubset(evidence_nums):
            return False
        answer_terms, evidence_terms = self._rc_terms(answer_norm), set(self._rc_terms(evidence))
        return bool(answer_terms and len(answer_terms) <= 12 and sum(term in evidence_terms for term in answer_terms) / len(answer_terms) >= 0.8)

    def _rc_requirement_match(self, req: Dict[str, Any], memory: Dict[str, Any], subject_id: str) -> bool:
        target = req.get("target") or ""
        slot = {"subject_id": subject_id, "target_surface": target, "resolved_keys": self._rc_resolve_target_keys(target, subject_id), "required_fields": [], "evidence_role": req.get("role") or "ANSWER"}
        return self._slot_contract_match(slot, memory, strict_targets=True)

    def _authorize_controller_answer(self, decision, seeds, frame):
        if decision.get("operator") not in {"DIRECT", "STATE"} or decision.get("requires_inference") or decision.get("visible_options"):
            return None, "NON_ATOMIC_ROUTE"
        refs = decision.get("support_refs") or []
        if len(refs) != 1 or not decision.get("answer"):
            return None, "ATOMIC_SUPPORT_REQUIRED"
        match = re.fullmatch(r"\$seed(\d+)", refs[0])
        if not match or int(match.group(1)) >= min(3, len(seeds)):
            return None, "INVALID_SUPPORT_REF"
        valid = self._validate_fast_support(refs[0], seeds, frame)
        if not valid:
            return None, "STRUCTURAL_SUPPORT_REJECTED"
        req = (decision.get("requirements") or [{}])[0]
        owner = decision.get("subject_id") or self._rc_owner(valid[0].get("subject_id") or valid[0].get("subject"))
        if not self._rc_requirement_match(req, valid[0], owner):
            return None, "TARGET_SUPPORT_REJECTED"
        if decision["operator"] == "STATE" and not self._is_state_head(valid[0]):
            return None, "STATE_REQUIRES_HEAD"
        if not self._rc_answer_grounded(decision["answer"], valid):
            return None, "ANSWER_NOT_GROUNDED"
        return valid, "AUTHORIZED"

    def _semantic_controller(self, question, seeds, frame, context_map=None):
        del context_map
        self._evidence_lattice, self._last_option_probe_coverage = EvidenceLattice(), {}
        options = self._question_options(question) or {}
        hints = {"dates": list(getattr(frame, "dates", ()) or ()), "source_speaker": getattr(frame, "speaker_role", ""), "hard_entities": list(getattr(frame, "hard_entities", ()) or ())}
        prompt = CONTROLLER_POLICY + "\n" + CONTROLLER_SCHEMA.format(question=question, options=json.dumps(options, ensure_ascii=False), hints=json.dumps(hints, ensure_ascii=False), seeds=json.dumps(self._rc_seed_payload(seeds), ensure_ascii=False))
        try:
            response = self._llm_client.chat([{"role": "user", "content": prompt}], temperature=0.0, max_tokens=512, response_format={"type": "json_object"})
            usage = self._response_usage(response, prompt)
            decision = self._rc_normalize_decision(self._parse_json(response.content), question, frame)
            if not decision["subject_id"]:
                owners = {self._rc_owner(memory.get("subject_id") or memory.get("subject") or "") for memory in seeds[:3] if memory.get("subject_id") or memory.get("subject")}
                owners.discard("")
                if len(owners) == 1:
                    decision["subject_id"] = next(iter(owners))
        except Exception as exc:
            decision = self._rc_normalize_decision({"route": "PLAN", "operator": "MULTI_OPTION" if options else "DIRECT"}, question, frame)
            plan = self._controller_plan(decision, question, frame)
            return None, plan, {"called": True, "route": "PLAN", "decision_route": "PLAN", "answer": "", "support_ref": "", "support_refs": [], "fallback_reason": "controller_error", "error": str(exc), "usage": {}, "operator": decision["operator"], "answer_slot": decision["answer_slot"]}
        fallback_reason = ""
        if decision["route"] == "ANSWER":
            supports, fallback_reason = self._authorize_controller_answer(decision, seeds, frame)
            if supports is not None:
                return supports, {}, {"called": True, "route": "DIRECT", "decision_route": "ANSWER", "answer": decision["answer"], "support_ref": decision["support_refs"][0], "support_refs": decision["support_refs"], "fallback_reason": "", "usage": usage, "operator": decision["operator"], "answer_slot": decision["answer_slot"], "requirement_count": len(decision["requirements"]), "target_rejected": decision["target_rejected"]}
            decision["route"] = "PLAN"
            if decision["operator"] == "DIRECT":
                decision["operator"], decision["requires_inference"] = "MULTI_HOP", True
        elif decision["route"] == "ABSTAIN":
            decision["route"], fallback_reason = "PLAN", "ABSTAIN_REQUIRES_BOUNDED_MEMORY_CHECK"
        plan = self._controller_plan(decision, question, frame)
        return None, plan, {"called": True, "route": "PLAN", "decision_route": "PLAN", "answer": "", "support_ref": "", "support_refs": [], "fallback_reason": fallback_reason, "usage": usage, "operator": decision["operator"], "answer_slot": decision["answer_slot"], "requires_inference": decision["requires_inference"], "causal_mode": decision.get("causal_mode") or "", "requirement_count": len(decision["requirements"]), "target_rejected": decision["target_rejected"]}
