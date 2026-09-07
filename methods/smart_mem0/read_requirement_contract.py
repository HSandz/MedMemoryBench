"""Minimal evidence-lookup requirements for SmartMem0 reads.

This layer owns semantic normalization only: participant-memory requirements, temporal
selectors, optional canonical-key hints, and answer-connected reasoning edges. It does
not prove evidence, rank answer context, decide terminality, or vary behavior by
benchmark query type or natural language.
"""

import re
import unicodedata
from copy import deepcopy
from typing import Any, Dict, List

from .read_controller import VALID_ANSWER_TYPES, VALID_IR_RELATIONS


REQUIREMENT_CONTROLLER_POLICY = """
You are the single semantic controller for an evidence-grounded memory system.
Produce the MINIMAL evidence-lookup IR needed for QUESTION.

LANGUAGE INVARIANT:
- Understand QUESTION in whatever natural language it uses.
- Derive semantic operators from meaning, not from English cue words.
- Keep focus_span copied exactly from QUESTION.
- Keep target and retrieval_hint in the language of QUESTION by default.
- If SEEDS expose a clearly equivalent stored phrase in another language, retrieval_hint
  may include that phrase as a soft cross-language recall expansion. Never let a
  translation or stored phrase redefine the question.
- The deterministic runtime will NOT maintain per-language keyword lists.

The QUESTION is the final answer obligation. REQUIREMENTS answer only:
"What participant-specific evidence values must memory retrieval supply before the final
answer model can answer this QUESTION well?"

Use the smallest sufficient set: usually 1-3 requirements, never more than 4.
A requirement must be:
1. MEMORY-VALUED: participant memory can supply its concrete value.
2. ANSWER-SENSITIVE: a different retrieved value could change the answer/reasoning path.
3. ATOMIC: one independently retrievable evidence obligation.
4. NON-MECHANISTIC: general-domain rules belong on reasoning bridges, not requirements.

grounding_kind:
- QUESTION: explicitly named/constrained by QUESTION. focus_span is the shortest useful
  contiguous span copied exactly from QUESTION.
- DERIVED: an additional participant-memory lookup variable required by answer reasoning.
  focus_span is empty. DERIVED does not assert that the evidence exists or what its value is.

For every requirement:
- target = concise participant-memory evidence variable, not the final answer/conclusion.
- retrieval_hint = soft semantic retrieval expansion; it is never proof.
- time_constraint = a selector/filter only when time changes which evidence value is
  needed. Do not emit LOCATE merely because QUESTION contains a date or temporal context.

TEMPORAL SEMANTICS (interpret these meanings in any language):
- document_time: when something was documented/recorded/reported in the source.
- event_time: when the participant event/state happened.
- origin_document_time: date of the original source document.
- effective_event_time: explicitly combined event/source chronology.
- LOCATE: the requested answer itself is a time/date.
- EXACT/BEFORE/AFTER/BETWEEN: constrain which evidence is selected by a known time.
- EARLIEST/LATEST: select the first/last occurrence and remain valid when the final
  answer is ENTITY/VALUE/TEXT.
Expressions whose meaning is onset/first occurrence should map to EARLIEST; expressions
whose meaning is latest/most recent should map to LATEST. Select the time axis from the
meaning of QUESTION, not from its language.

Optional proof_spec is an exact ledger-field certificate, never text similarity.
Use {"match":{"subject_id":"...","scope":"...","state_key":"...","stance":"AFFIRM"},
"answer_field":"value"} only when those canonical fields exactly express the obligation.
Match may also include object_anchor or semantic_role. Omit proof_spec if unavailable.
No word overlap constitutes a certificate.

VISIBLE OPTIONS are answer propositions, not memory facts. Do not create one requirement
per option merely because it is visible. Retrieve the shared participant evidence needed
to discriminate among the options.

For ADVICE/ACTION/MEDICATION decisions, retrieve decision-changing participant evidence
when it is needed: safety constraints/contraindications or prior explicit guidance first,
then current relevant state/regimen, then preferences when they materially affect the
choice. These are DERIVED only when QUESTION does not explicitly name them.
Do not replace participant evidence with a general domain rule.

SEEDS may reveal an answer-sensitive DERIVED lookup variable, but may not pre-fill its
value or redefine QUESTION. A seed can suggest WHAT participant evidence must be looked
up; the retrieved memory must still supply the value.

RELATIONS:
- COMPARE: compare grounded participant evidence variables.
- CAUSES: requires an explicit stored causal relation.
- POSSIBLE_CAUSE: grounded endpoints plus authorized general-domain knowledge.
- DEPENDS_ON: FROM depends on TO; does not assert causality.
- TEMPORAL_ORDER: order grounded endpoints.
- INFER: authorize a general-domain bridge only when grounded participant facts must be
  combined with a general rule to produce an answer not explicit in memory.
- CURRENT: mark a current-state requirement.
- VERIFY_SOURCE: request exact linked source evidence.

INFER is exceptional. Do not emit it for extraction, paraphrase, entity/value/date
selection, ordinary synthesis, or choosing a supported option. General mechanisms remain
bridge goals, never invented participant requirements.
Every DERIVED requirement must participate in at least one answer-relevant relation.
Never invent an edge merely to legalize an orphan DERIVED node.

candidate is optional only when exactly ONE QUESTION requirement is enough and exactly
ONE seed already contains the COMPLETE answer. The candidate may be a scalar, sentence,
or short paragraph. Answer length never controls terminality. Do not emit candidate for
advice, world-knowledge inference, cross-memory synthesis, or visible options.
"""

REQUIREMENT_SCHEMA = """Return JSON only:
{{"answer_type":"ENTITY|VALUE|DATE|RELATIVE_TIME|OPTION_SET|TEXT","subject_span":"exact contiguous subject span from QUESTION or empty","requirements":[{{"id":"r1","grounding_kind":"QUESTION|DERIVED","focus_span":"exact QUESTION span for QUESTION nodes, otherwise empty","target":"concise participant-memory evidence variable","retrieval_hint":"soft semantic retrieval expansion","time_constraint":{{"axis":"event_time|document_time|origin_document_time|effective_event_time|","relation":"LOCATE|EXACT|EARLIEST|LATEST|BEFORE|AFTER|BETWEEN|","anchor":"","end":""}}}}],"relations":[{{"type":"COMPARE|CAUSES|POSSIBLE_CAUSE|DEPENDS_ON|TEMPORAL_ORDER|INFER|CURRENT|VERIFY_SOURCE","from":"r1","to":"r2|ANSWER|","relation":"BEFORE|AFTER|OVERLAPS|","bridge_goal":"short reasoning obligation or empty"}}],"candidate":null}}
When candidate exists use: {{"candidate":{{"answer":"complete answer already contained in one seed","support_ref":"$seed0"}}}}
Each requirement may optionally include proof_spec as defined above; omit it when no exact ledger certificate is available.
QUESTION:
{question}
VISIBLE OPTIONS:
{options}
STRUCTURAL HINTS (constraints only; never evidence):
{hints}
SEEDS:
{seeds}"""


class ReadRequirementContractMixin:
    """Compile controller semantics into stable, language-agnostic evidence obligations."""

    REQUIREMENT_TARGET_MAX_CHARS = 160
    REQUIREMENT_TARGET_MAX_TERMS = 24

    def _rq_compact_target(self, value: Any) -> str:
        target = " ".join(str(value or "").split()).strip(" -:;")
        if not target:
            return ""
        if len(target) > self.REQUIREMENT_TARGET_MAX_CHARS:
            return ""
        if len(self._rc_terms(target)) > self.REQUIREMENT_TARGET_MAX_TERMS:
            return ""
        return target

    def _rq_fallback_target(self, value: Any) -> str:
        text = " ".join(str(value or "").split()).strip(" -:;")
        if not text:
            return ""
        candidate = text[: self.REQUIREMENT_TARGET_MAX_CHARS].rstrip()
        return self._rq_compact_target(candidate)

    @classmethod
    def _rq_surface_chars(cls, value: Any) -> str:
        return "".join(ch for ch in cls._rc_text(value) if ch.isalnum())

    @classmethod
    def _rq_char_ngrams(cls, value: Any, width: int = 3) -> set:
        surface = cls._rq_surface_chars(value)
        if not surface:
            return set()
        if len(surface) <= width:
            return {surface}
        return {surface[index : index + width] for index in range(len(surface) - width + 1)}

    @classmethod
    def _rq_surface_similarity(cls, target: Any, surface: Any) -> float:
        left, right = cls._rq_surface_chars(target), cls._rq_surface_chars(surface)
        if not left or not right:
            return 0.0
        if left == right or left in right:
            return 1.0
        left_grams = cls._rq_char_ngrams(left)
        right_grams = cls._rq_char_ngrams(right)
        char_coverage = len(left_grams & right_grams) / len(left_grams) if left_grams else 0.0
        left_terms = list(dict.fromkeys(cls._rc_terms(target)))
        right_terms = set(cls._rc_terms(surface))
        token_coverage = sum(term in right_terms for term in left_terms) / len(left_terms) if left_terms else 0.0
        return max(char_coverage, token_coverage)

    @classmethod
    def _rq_focus_is_answer_obligation(cls, focus: Any, question: Any) -> bool:
        focus_surface = cls._rq_surface_chars(focus)
        question_surface = cls._rq_surface_chars(question)
        if not focus_surface or not question_surface:
            return False
        if focus_surface == question_surface:
            return True
        clauses, current = [], []
        for char in str(question or ""):
            name = unicodedata.name(char, "")
            if "QUESTION MARK" in name:
                text = "".join(current).strip()
                if text:
                    clauses.append(text)
                current = []
            elif unicodedata.category(char).startswith("P") and any(marker in name for marker in ("FULL STOP", "EXCLAMATION MARK")):
                current = []
            else:
                current.append(char)
        if any(focus_surface == cls._rq_surface_chars(clause) for clause in clauses):
            return True
        return len(focus_surface) >= 12 and len(focus_surface) / max(1, len(question_surface)) >= 0.72

    @classmethod
    def _rq_memory_concept_surfaces(cls, memory: Dict[str, Any]) -> List[str]:
        values: List[Any] = [memory.get("scope"), memory.get("state_key"), memory.get("object_anchor")]
        values.extend(memory.get("entities") or [])
        values.extend(memory.get("scope_entities") or [])
        output, seen = [], set()
        for value in values:
            text = " ".join(str(value or "").replace("_", " ").split()).strip()
            normalized = cls._rc_text(text)
            if text and normalized not in seen:
                output.append(text)
                seen.add(normalized)
        return output

    def _rc_resolve_target_keys(self, target: str, subject_id: str = "") -> List[str]:
        if not self._rq_surface_chars(target):
            return []
        wanted_owner = self._rc_owner(subject_id)
        scores: Dict[str, float] = {}
        surface_by_norm = {}
        for memory in getattr(self, "_memories", []) or []:
            owner = self._rc_owner(memory.get("subject_id") or memory.get("subject") or "")
            if wanted_owner and owner != wanted_owner:
                continue
            for key in self._rq_memory_concept_surfaces(memory):
                normalized = self._rc_text(key)
                surface_by_norm.setdefault(normalized, key)
                score = self._rq_surface_similarity(target, key)
                if score >= 0.72:
                    scores[normalized] = max(scores.get(normalized, 0.0), score)
        return [surface_by_norm[key] for key, _ in sorted(scores.items(), key=lambda item: (-item[1], -len(self._rq_surface_chars(item[0])), item[0]))[:3] if key in surface_by_norm]

    def _rq_repair_time_constraint(self, constraint, _semantic_surface="", answer_type=None):
        result = dict(constraint or {})
        answer_type = str(answer_type or "TEXT").upper()
        axis = str(result.get("axis") or "")
        relation = str(result.get("relation") or "").upper()
        anchor = str(result.get("anchor") or "")
        end = str(result.get("end") or "")
        if not axis:
            return {"axis": "", "relation": "", "anchor": "", "end": ""}
        if relation in {"EARLIEST", "LATEST"}:
            return {"axis": axis, "relation": relation, "anchor": "", "end": ""}
        if relation == "LOCATE":
            if answer_type in {"DATE", "RELATIVE_TIME"}:
                return {"axis": axis, "relation": "LOCATE", "anchor": "", "end": ""}
            if anchor:
                return {"axis": axis, "relation": "BETWEEN" if end else "EXACT", "anchor": anchor, "end": end}
            return {"axis": "", "relation": "", "anchor": "", "end": ""}
        if not relation:
            if answer_type in {"DATE", "RELATIVE_TIME"}:
                return {"axis": axis, "relation": "LOCATE", "anchor": "", "end": ""}
            if anchor:
                return {"axis": axis, "relation": "BETWEEN" if end else "EXACT", "anchor": anchor, "end": end}
            return {"axis": "", "relation": "", "anchor": "", "end": ""}
        if relation in {"EXACT", "BEFORE", "AFTER"} and not anchor:
            return {"axis": "", "relation": "", "anchor": "", "end": ""}
        if relation == "BETWEEN" and (not anchor or not end):
            return {"axis": "", "relation": "", "anchor": "", "end": ""}
        return {"axis": axis, "relation": relation, "anchor": anchor, "end": end}

    @staticmethod
    def _rq_graph_validation(requirements, relations):
        nodes = {item["id"] for item in requirements}
        adjacency = {node: set() for node in nodes}
        explicit_answer = False
        for edge in relations:
            source, target = edge.get("from"), edge.get("to")
            kind = edge.get("type")
            if source not in nodes or target not in nodes | {"ANSWER"}:
                continue
            if kind == "DEPENDS_ON":
                if target in nodes:
                    adjacency[target].add(source)
            else:
                adjacency[source].add(target)
                if kind in {"COMPARE", "TEMPORAL_ORDER"} and target in nodes:
                    adjacency[target].add(source)
            explicit_answer |= target == "ANSWER"
        implicit_outputs = set() if explicit_answer else {item["id"] for item in requirements if item.get("grounding_kind") == "QUESTION"}
        for node in implicit_outputs:
            adjacency[node].add("ANSWER")
        connected = {}
        for node in nodes:
            seen, pending = set(), [node]
            while pending:
                current = pending.pop()
                if current in seen:
                    continue
                seen.add(current)
                pending.extend(adjacency.get(current, set()) - seen)
            connected[node] = "ANSWER" in seen
        orphans = sorted(node for node, reachable in connected.items() if not reachable)
        return {"connected_to_answer": dict(sorted(connected.items())), "orphan_requirements": orphans, "implicit_answer_requirements": sorted(implicit_outputs), "valid": not orphans, "scope": "dependency_reachability_not_semantic_proof"}

    def _rc_normalize_ir(self, parsed: Dict[str, Any], question: str, frame: Any):
        parsed = parsed if isinstance(parsed, dict) else {}
        options = self._question_options(question) or {}
        answer_type = str(parsed.get("answer_type") or "TEXT").upper()
        answer_type = "OPTION_SET" if options else answer_type if answer_type in VALID_ANSWER_TYPES else "TEXT"
        subject_span = self._rc_question_span(parsed.get("subject_span"), question)
        actions = []
        raw_requirements = parsed.get("requirements") if isinstance(parsed.get("requirements"), list) else []
        requirements, seen_ids = [], set()
        for index, raw in enumerate(raw_requirements[:4]):
            if not isinstance(raw, dict):
                actions.append({"index": index, "action": "DROP", "reason": "NOT_OBJECT"})
                continue
            kind = str(raw.get("grounding_kind") or "QUESTION").upper()
            if kind not in {"QUESTION", "DERIVED"}:
                actions.append({"index": index, "action": "DROP", "reason": "INVALID_KIND"})
                continue
            focus = ""
            if kind == "QUESTION":
                focus = self._rc_question_span(raw.get("focus_span"), question)
                if not focus:
                    focus = self._rc_question_span(raw.get("target"), question)
                if not focus:
                    actions.append({"index": index, "id": str(raw.get("id") or f"r{index + 1}"), "action": "DROP", "reason": "INVALID_QUESTION_FOCUS"})
                    continue
            target = self._rq_compact_target(raw.get("target") or raw.get("evidence_target"))
            if not target and kind == "QUESTION" and not self._rq_focus_is_answer_obligation(focus, question):
                target = self._rq_compact_target(focus)
                if target:
                    actions.append({"index": index, "id": str(raw.get("id") or f"r{index + 1}"), "action": "REPAIR", "reason": "INVALID_TARGET_USE_FOCUS"})
            if not target:
                actions.append({"index": index, "id": str(raw.get("id") or f"r{index + 1}"), "action": "DROP", "reason": "INVALID_TARGET"})
                continue
            requirement_id = str(raw.get("id") or f"r{index + 1}")
            if not re.fullmatch(r"r[\w-]{0,31}", requirement_id) or requirement_id in seen_ids:
                requirement_id = f"r{index + 1}"
            while requirement_id in seen_ids:
                requirement_id += "x"
            seen_ids.add(requirement_id)
            time_constraint = self._rc_normalize_time_constraint(raw.get("time_constraint"), frame, question)
            if kind == "QUESTION":
                binder = getattr(self, "_rc_bind_focus_time_constraint", None)
                if callable(binder):
                    time_constraint = binder(focus, time_constraint)
                time_constraint = self._rq_repair_time_constraint(time_constraint, "", answer_type)
            hint = " ".join(str(raw.get("retrieval_hint") or target).split())[:320]
            requirements.append({"id": requirement_id, "grounding_kind": kind, "focus_span": focus, "target": target, "retrieval_hint": hint, "time_constraint": time_constraint})
        degraded = not requirements
        if degraded:
            stem = self._question_stem(question).strip()
            fallback_target = self._rq_fallback_target(subject_span or stem) or stem[: self.REQUIREMENT_TARGET_MAX_CHARS]
            requirements = [{"id": "r1", "grounding_kind": "DERIVED" if options else "QUESTION", "focus_span": "" if options else stem, "target": fallback_target, "retrieval_hint": stem if options else fallback_target, "time_constraint": {"axis": "", "relation": "", "anchor": "", "end": ""}, "degraded": True}]
            actions.append({"action": "FALLBACK", "reason": "OPTION_SHARED_EVIDENCE" if options else "ALL_REQUIREMENTS_INVALID"})
        valid_nodes = {item["id"] for item in requirements}
        raw_relations = parsed.get("relations") if isinstance(parsed.get("relations"), list) else []
        relations = []
        for raw in raw_relations[:8]:
            if not isinstance(raw, dict):
                continue
            relation_type = str(raw.get("type") or "").upper()
            source, target = str(raw.get("from") or ""), str(raw.get("to") or "")
            if degraded or relation_type not in VALID_IR_RELATIONS or source not in valid_nodes:
                actions.append({"action": "DROP_RELATION", "reason": "INVALID_RELATION_OR_ENDPOINT", "from": source, "to": target})
                continue
            if relation_type in {"CURRENT", "VERIFY_SOURCE"}:
                target = ""
            elif target != "ANSWER" and target not in valid_nodes:
                actions.append({"action": "DROP_RELATION", "reason": "INVALID_ENDPOINT", "from": source, "to": target})
                continue
            if relation_type in {"COMPARE", "CAUSES", "POSSIBLE_CAUSE", "DEPENDS_ON", "TEMPORAL_ORDER"} and target not in valid_nodes:
                continue
            relation = {"type": relation_type, "from": source, "to": target}
            if relation_type == "TEMPORAL_ORDER":
                order = str(raw.get("relation") or "").upper()
                if order not in {"BEFORE", "AFTER", "OVERLAPS"}:
                    continue
                relation["relation"] = order
            if relation_type in {"POSSIBLE_CAUSE", "INFER", "CAUSES"}:
                goal = " ".join(str(raw.get("bridge_goal") or "").split())[:220]
                if goal:
                    relation["bridge_goal"] = goal
            if relation not in relations:
                relations.append(relation)
        referenced = {endpoint for relation in relations for endpoint in (relation.get("from"), relation.get("to")) if endpoint and endpoint != "ANSWER"}
        drop_ids = {requirement["id"] for requirement in requirements if requirement.get("grounding_kind") == "DERIVED" and requirement["id"] not in referenced}
        if drop_ids:
            requirements = [r for r in requirements if r["id"] not in drop_ids]
            relations = [rel for rel in relations if rel.get("from") not in drop_ids and rel.get("to") not in drop_ids]
            actions.append({"action": "DROP_ORPHAN_DERIVED", "reason": "ZERO_SEMANTIC_EDGES", "requirement_ids": sorted(drop_ids)})
        graph_validation = self._rq_graph_validation(requirements, relations)
        unreachable_derived = [requirement["id"] for requirement in requirements if requirement.get("grounding_kind") == "DERIVED" and requirement["id"] in set(graph_validation.get("orphan_requirements") or [])]
        if unreachable_derived:
            actions.append({"action": "GRAPH_WARNING", "reason": "DERIVED_NOT_REACHABLE_TO_ANSWER", "requirement_ids": unreachable_derived})
        candidate = parsed.get("candidate") if not degraded and isinstance(parsed.get("candidate"), dict) else None
        if candidate is not None:
            answer = str(candidate.get("answer") or "").strip()
            support_ref = str(candidate.get("support_ref") or "")
            only_question = len(requirements) == 1 and requirements[0].get("grounding_kind") == "QUESTION"
            candidate = {"answer": answer, "support_ref": support_ref} if answer and re.fullmatch(r"\$seed[0-2]", support_ref) and only_question else None
        self._last_requirement_normalization_actions = list(actions)
        self._last_orphan_derived_ids = sorted(drop_ids)
        repaired = any(action.get("action") in {"DROP", "REPAIR", "DROP_RELATION", "DROP_ORPHAN_DERIVED"} for action in actions)
        return {"answer_type": answer_type, "subject_span": subject_span, "_resolved_subject_id": self._rc_known_subject(subject_span), "requirements": requirements[:4], "relations": relations[:8], "candidate": candidate, "visible_options": dict(options), "normalization_status": "DEGRADED" if degraded else "REPAIRED" if repaired else "VALID", "normalization_actions": deepcopy(actions), "graph_validation": graph_validation}

    def _requirement_slot(self, requirement, ir, compiled_mode):
        slot = super()._requirement_slot(requirement, ir, compiled_mode)
        kind = str(requirement.get("grounding_kind") or "QUESTION").upper()
        target = str(requirement.get("target") or "").strip()
        focus = str(requirement.get("focus_span") or "").strip()
        slot["grounding_kind"] = kind
        slot["focus_span"] = focus
        slot["target_surface"] = target
        slot["retrieval_target"] = target
        slot["degraded"] = bool(requirement.get("degraded"))
        slot["proof_anchor"] = focus if kind == "QUESTION" and focus else target
        slot["description"] = str(requirement.get("retrieval_hint") or "").strip() or target or focus
        slot["resolved_keys"] = self._rc_resolve_target_keys(target, str(slot.get("subject_id") or ""))
        return slot

    def _controller_plan(self, ir, question, frame):
        plan = super()._controller_plan(ir, question, frame)
        plan.setdefault("query_spec", {})["semantic_ir_version"] = "minimal-v3-language-neutral"
        for key in ("normalization_status", "normalization_actions", "graph_validation"):
            plan[key] = deepcopy(ir.get(key))
            plan["semantic_ir"][key] = deepcopy(ir.get(key))
        if ir.get("normalization_status") == "DEGRADED":
            plan["planner_fallback"] = True
            plan["fallback_reason"] = "ALL_REQUIREMENTS_INVALID"
            if not ir.get("visible_options"):
                plan["compiled_mode"] = plan["query_mode"] = "DEGRADED"
        return plan

    @staticmethod
    def _compact_reasoning_output_instruction(_query_type: str) -> str:
        return ""

    def prepare_batch_query(self, question, system_message=None, **kwargs):
        prepared = super().prepare_batch_query(question, system_message=system_message, **kwargs)
        extra = prepared.setdefault("extra", {})
        extra["read_contract_version"] = "minimal-ir-v4-language-neutral"
        extra["requirement_normalization_actions"] = list(getattr(self, "_last_requirement_normalization_actions", []) or [])
        controller = extra.get("semantic_controller") or {}
        for key in ("controller_raw_ir", "normalized_ir", "normalization_status", "normalization_actions"):
            extra[key] = deepcopy(controller.get(key))
        return prepared
