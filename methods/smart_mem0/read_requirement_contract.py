"""Evidence-lookup requirement policy for the minimal semantic read IR.

The QUESTION remains the answer obligation. Requirements are participant-memory lookup
variables; general-domain mechanisms remain reasoning bridges.
"""

import json
import re
from typing import Any, Dict

from .read_controller import VALID_ANSWER_TYPES, VALID_IR_RELATIONS


REQUIREMENT_CONTROLLER_POLICY = """
You are the single semantic controller for an evidence-grounded memory system.
Produce a MINIMAL evidence-lookup IR. The QUESTION itself is the final answer obligation;
never turn the requested answer, recommendation, decision, or yes/no question into a
requirement.

REQUIREMENTS answer only:
"What participant-specific evidence values must memory retrieval supply before the final
answer model can answer this QUESTION well?"

Use the smallest sufficient set: usually 1-3 requirements, never more than 4.
A requirement is valid only when ALL are true:
1. MEMORY-VALUED: participant memory can supply a concrete fact/state/event/measurement/
   decision/preference/constraint/exposure/symptom or other subject-specific value.
2. ANSWER-SENSITIVE: different plausible retrieved values could change the answer or the
   grounded reasoning path.
3. ATOMIC: it is one independently retrievable evidence obligation, not a bundle.
4. NON-MECHANISTIC: it is not merely a general-domain rule, physiology, pharmacology, or
   reasoning step. General mechanisms belong on bridges.

Each requirement has grounding_kind:
- QUESTION: the participant evidence variable is explicitly named or constrained by the
  QUESTION. focus_span MUST be the shortest useful contiguous span copied exactly from
  QUESTION.
- DERIVED: an additional participant-memory lookup variable is needed even though it is
  not directly named in QUESTION. focus_span MUST be empty. DERIVED does not assert that
  the evidence exists or what its value is.

For EVERY requirement:
- target is the concise evidence variable to retrieve, not an answer or conclusion.
  Prefer a noun/evidence phrase of about 3-10 content words; never exceed 120 characters.
- retrieval_hint is a soft semantic expansion for search. It may use QUESTION and SEEDS
  to broaden the neighborhood, but it is never proof and must not assert an unseen fact.
- time_constraint is only for a time obligation required by QUESTION.

focus_span is QUESTION provenance, not the retrieval query. target says WHAT evidence is
needed. retrieval_hint says HOW to search for it.

TEMPORAL SEMANTICS:
- document_time = when something was documented, recorded, noted, charted, or mentioned.
- event_time = when the participant event/state actually happened.
- origin_document_time = the date of the original source document.
- effective_event_time = explicitly combined event/source chronology.
- LOCATE means the requested time is unknown. EXACT/BEFORE/AFTER/BETWEEN constrain a known
  selector. EARLIEST/LATEST request an extremum.
Never map "when was it documented" to event_time merely because a memory also has an event date.

VISIBLE OPTIONS are answer propositions, not memory facts. Do not create one requirement
per option. Create only the minimal shared participant-specific evidence needed to
discriminate among the visible propositions; deterministic option probes explore each one.

For ADVICE/ACTION/MEDICATION decisions, prioritize decision-changing participant evidence:
safety constraints and contraindications, prior explicit guidance, current relevant state
or regimen, then preferences when they materially affect the choice. Do not substitute a
general preference for a more decision-changing safety/guidance variable.

SEEDS may reveal a useful DERIVED lookup variable, but may not redefine the QUESTION or
pre-fill a participant fact. A seed mentioning repeated morning glucose may justify
"morning glycemic state"; it must not make the controller assert a glucose value.

Do NOT create requirements such as "Can I take painkillers?", "what should I do?", or
"whether this is safe". These are answer obligations. Do not create general mechanisms
such as cortisol release, HPA activation, or hepatic glucose output as memory requirements.

RELATIONS are reasoning obligations between grounded requirement values:
- COMPARE compares two participant evidence variables.
- CAUSES requires an explicit stored causal relation.
- POSSIBLE_CAUSE connects grounded participant endpoints with authorized general knowledge.
- DEPENDS_ON means the FROM evidence/decision depends on the TO evidence; it does not assert causality.
- TEMPORAL_ORDER orders grounded endpoints.
- INFER authorizes a general-domain bridge; use to="ANSWER" only when a domain rule is
  actually needed to produce an answer not explicit in memory.
- CURRENT marks a current-state requirement.
- VERIFY_SOURCE asks for exact linked source evidence.

INFER IS EXCEPTIONAL. Do not emit INFER for direct extraction, paraphrase, entity/value/date
selection, ordinary synthesis, or choosing a supported option. Emit it only when grounded
participant facts must be combined with a general-domain rule to produce the answer.
POSSIBLE_CAUSE is only for a genuine causal question, not merely an action/recommendation.
For POSSIBLE_CAUSE, INFER, and CAUSES, bridge_goal is a short reasoning obligation and must
not encode unstored participant history.
If a DERIVED variable mediates a causal/inferential path, route the graph THROUGH that
variable instead of bypassing it with a direct endpoint edge. Every DERIVED node must
participate in at least one relation used by answer reasoning.

candidate is optional and only for one atomic non-temporal QUESTION requirement when exactly
one seed directly contains the answer. Code independently authorizes it.
"""

REQUIREMENT_SCHEMA = """Return JSON only:
{{"answer_type":"ENTITY|VALUE|DATE|RELATIVE_TIME|OPTION_SET|TEXT","subject_span":"exact contiguous subject span from QUESTION or empty","requirements":[{{"id":"r1","grounding_kind":"QUESTION|DERIVED","focus_span":"exact QUESTION span for QUESTION nodes, otherwise empty","target":"concise participant-memory evidence variable","retrieval_hint":"soft semantic retrieval expansion","time_constraint":{{"axis":"event_time|document_time|origin_document_time|effective_event_time|","relation":"LOCATE|EXACT|EARLIEST|LATEST|BEFORE|AFTER|BETWEEN|","anchor":"","end":""}}}}],"relations":[{{"type":"COMPARE|CAUSES|POSSIBLE_CAUSE|DEPENDS_ON|TEMPORAL_ORDER|INFER|CURRENT|VERIFY_SOURCE","from":"r1","to":"r2|ANSWER|","relation":"BEFORE|AFTER|OVERLAPS|","bridge_goal":"short reasoning obligation or empty"}}],"candidate":null}}
When candidate exists use: {{"candidate":{{"answer":"exact answer value","support_ref":"$seed0"}}}}
QUESTION:
{question}
VISIBLE OPTIONS:
{options}
STRUCTURAL HINTS (constraints only; never evidence):
{hints}
SEEDS:
{seeds}"""


class ReadRequirementContractMixin:
    """Make requirements first-class evidence lookup variables."""

    REQUIREMENT_TARGET_MAX_CHARS = 120
    REQUIREMENT_TARGET_MAX_TERMS = 16

    def _rq_compact_target(self, value: Any) -> str:
        target = " ".join(str(value or "").split()).strip(" -:;")
        if not target or "?" in target:
            return ""
        if len(target) > self.REQUIREMENT_TARGET_MAX_CHARS:
            return ""
        if len(self._rc_terms(target)) > self.REQUIREMENT_TARGET_MAX_TERMS:
            return ""
        return target

    def _rq_fallback_target(self, value: Any) -> str:
        """Bound degraded fallback by the exact same normal target contract."""
        words = " ".join(str(value or "").split()).strip(" -:;").split()
        kept = []
        for word in words:
            candidate = " ".join([*kept, word])
            if len(candidate) > self.REQUIREMENT_TARGET_MAX_CHARS:
                break
            if len(self._rc_terms(candidate)) > self.REQUIREMENT_TARGET_MAX_TERMS:
                break
            kept.append(word)
        return self._rq_compact_target(" ".join(kept))

    @staticmethod
    def _rq_focus_is_answer_obligation(focus: Any) -> bool:
        text = " ".join(str(focus or "").lower().split()).strip()
        return bool(
            re.match(
                r"^(?:can|could|should|would|may|might|do|does|did|is|are|was|were)\b",
                text,
            )
            or text.startswith("what should")
            or text.startswith("which should")
            or text.startswith("whether ")
        )

    def _rq_repair_time_constraint(self, constraint, question, answer_type):
        """Deterministically preserve question-owned temporal-axis semantics."""
        result = dict(constraint or {})
        if answer_type not in {"DATE", "RELATIVE_TIME"}:
            return result
        relation = str(result.get("relation") or "").upper()
        if relation not in {"", "LOCATE"}:
            return result
        text = self._rc_text(question)
        documented = re.search(
            r"\b(documented|documentation|recorded|noted|mentioned|charted)\b", text
        )
        original_source = re.search(
            r"\b(original|source)\b.*\b(document|note|record|session)\b", text
        )
        happened = re.search(r"\b(happened|occurred|took place)\b", text)
        if original_source:
            result["axis"] = "origin_document_time"
        elif documented:
            result["axis"] = "document_time"
        elif happened and not result.get("axis"):
            result["axis"] = "event_time"
        if result.get("axis") and not result.get("relation"):
            result["relation"] = "LOCATE"
        return result

    def _rc_normalize_ir(self, parsed: Dict[str, Any], question: str, frame: Any):
        """Normalize Requirement-v2 without collapsing a partly valid graph."""
        options = self._question_options(question) or {}
        answer_type = str(parsed.get("answer_type") or "TEXT").upper()
        answer_type = (
            "OPTION_SET" if options
            else answer_type if answer_type in VALID_ANSWER_TYPES else "TEXT"
        )
        subject_span = self._rc_question_span(parsed.get("subject_span"), question)
        actions = []
        raw_requirements = (
            parsed.get("requirements") if isinstance(parsed.get("requirements"), list) else []
        )
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
                    actions.append({
                        "index": index, "id": str(raw.get("id") or f"r{index + 1}"),
                        "action": "DROP", "reason": "INVALID_QUESTION_FOCUS",
                    })
                    continue

            raw_target = raw.get("target") or raw.get("evidence_target")
            target = self._rq_compact_target(raw_target)
            if (
                not target
                and kind == "QUESTION"
                and not self._rq_focus_is_answer_obligation(focus)
            ):
                target = self._rq_compact_target(focus)
                if target:
                    actions.append({
                        "index": index, "id": str(raw.get("id") or f"r{index + 1}"),
                        "action": "REPAIR", "reason": "INVALID_TARGET_USE_FOCUS",
                    })
            if not target:
                actions.append({
                    "index": index, "id": str(raw.get("id") or f"r{index + 1}"),
                    "action": "DROP", "reason": "INVALID_TARGET",
                })
                continue

            requirement_id = str(raw.get("id") or f"r{index + 1}")
            if not re.fullmatch(r"r[\w-]{0,31}", requirement_id) or requirement_id in seen_ids:
                requirement_id = f"r{index + 1}"
            while requirement_id in seen_ids:
                requirement_id += "x"
            seen_ids.add(requirement_id)

            time_constraint = self._rc_normalize_time_constraint(
                raw.get("time_constraint"), frame, question
            )
            if kind == "QUESTION":
                binder = getattr(self, "_rc_bind_focus_time_constraint", None)
                if callable(binder):
                    time_constraint = binder(focus, time_constraint)
            time_constraint = self._rq_repair_time_constraint(
                time_constraint, question, answer_type
            )
            hint = " ".join(str(raw.get("retrieval_hint") or target).split())[:240]
            requirements.append({
                "id": requirement_id,
                "grounding_kind": kind,
                "focus_span": focus,
                "target": target,
                "retrieval_hint": hint,
                "time_constraint": time_constraint,
            })

        if not requirements:
            if options:
                requirements = [{
                    "id": "r1", "grounding_kind": "DERIVED", "focus_span": "",
                    "target": "participant evidence relevant to visible options",
                    "retrieval_hint": "participant-specific evidence needed to evaluate the visible answer options",
                    "time_constraint": {"axis": "", "relation": "", "anchor": "", "end": ""},
                }]
                actions.append({"action": "FALLBACK", "reason": "OPTION_SHARED_EVIDENCE"})
            else:
                stem = self._question_stem(question).strip()
                fallback_target = self._rq_fallback_target(subject_span or stem) or "participant evidence"
                requirements = [{
                    "id": "r1", "grounding_kind": "QUESTION", "focus_span": stem,
                    "target": fallback_target, "retrieval_hint": fallback_target,
                    "time_constraint": {"axis": "", "relation": "", "anchor": "", "end": ""},
                }]
                actions.append({"action": "FALLBACK", "reason": "ALL_REQUIREMENTS_INVALID"})

        valid_nodes = {item["id"] for item in requirements}
        raw_relations = parsed.get("relations") if isinstance(parsed.get("relations"), list) else []
        relations = []
        for raw in raw_relations[:8]:
            if not isinstance(raw, dict):
                continue
            relation_type = str(raw.get("type") or "").upper()
            source, target = str(raw.get("from") or ""), str(raw.get("to") or "")
            if relation_type not in VALID_IR_RELATIONS or source not in valid_nodes:
                continue
            if relation_type in {"CURRENT", "VERIFY_SOURCE"}:
                target = ""
            elif target != "ANSWER" and target not in valid_nodes:
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

        temporal_dependencies = {
            endpoint for relation in relations if relation.get("type") == "TEMPORAL_ORDER"
            for endpoint in (relation.get("from"), relation.get("to"))
        }
        temporal_dependencies.update(
            relation.get("to") for relation in relations if relation.get("type") == "DEPENDS_ON"
        )
        if answer_type not in {"DATE", "RELATIVE_TIME"}:
            for requirement in requirements:
                constraint = requirement.get("time_constraint") or {}
                if constraint.get("relation") == "LOCATE" and requirement["id"] not in temporal_dependencies:
                    requirement["time_constraint"] = {"axis": "", "relation": "", "anchor": "", "end": ""}

        referenced = {
            endpoint for relation in relations
            for endpoint in (relation.get("from"), relation.get("to"))
            if endpoint and endpoint != "ANSWER"
        }
        orphan_derived = [
            requirement["id"] for requirement in requirements
            if requirement.get("grounding_kind") == "DERIVED" and requirement["id"] not in referenced
        ]
        if orphan_derived:
            actions.append({
                "action": "GRAPH_WARNING", "reason": "ORPHAN_DERIVED",
                "requirement_ids": orphan_derived,
            })

        candidate = parsed.get("candidate") if isinstance(parsed.get("candidate"), dict) else None
        if candidate is not None:
            answer = str(candidate.get("answer") or "").strip()
            support_ref = str(candidate.get("support_ref") or "")
            only_question = len(requirements) == 1 and requirements[0].get("grounding_kind") == "QUESTION"
            temporal = any(
                (requirement.get("time_constraint") or {}).get("axis")
                or (requirement.get("time_constraint") or {}).get("relation")
                for requirement in requirements
            )
            if not answer or not re.fullmatch(r"\$seed[0-2]", support_ref) or not only_question or temporal:
                candidate = None
            else:
                candidate = {"answer": answer, "support_ref": support_ref}

        if candidate and relations and all(
            relation.get("type") == "INFER"
            and relation.get("from") == requirements[0]["id"]
            and relation.get("to") == "ANSWER"
            for relation in relations
        ):
            relations = []

        self._last_requirement_normalization_actions = actions
        self._last_orphan_derived_ids = orphan_derived
        return {
            "answer_type": answer_type,
            "subject_span": subject_span,
            "_resolved_subject_id": self._rc_known_subject(subject_span),
            "requirements": requirements[:4],
            "relations": relations[:8],
            "candidate": candidate,
            "visible_options": dict(options),
        }

    def _requirement_slot(self, requirement, ir, compiled_mode):
        """Compile recall target separately from conservative proof anchor."""
        slot = super()._requirement_slot(requirement, ir, compiled_mode)
        kind = str(requirement.get("grounding_kind") or "QUESTION").upper()
        target = str(requirement.get("target") or "").strip()
        focus = str(requirement.get("focus_span") or "").strip()
        slot["grounding_kind"] = kind
        slot["focus_span"] = focus
        slot["target_surface"] = target
        slot["proof_anchor"] = focus if kind == "QUESTION" and focus else target
        slot["description"] = str(requirement.get("retrieval_hint") or "").strip() or target or "participant evidence"
        slot["resolved_keys"] = self._rc_resolve_target_keys(target, str(slot.get("subject_id") or ""))
        return slot

    def _requirement_target_proof(self, slot, memory):
        """Proof uses proof_anchor; semantic target stays recall-oriented."""
        if str(slot.get("evidence_role") or "").upper() != "REQUIREMENT":
            return True
        target = str(slot.get("proof_anchor") or slot.get("target_surface") or "").strip()
        if not target:
            return False
        text = self._rc_memory_target_text(memory)
        if self._rc_token_sequence_present(target, text):
            return True
        target_terms = list(dict.fromkeys(self._rc_content_terms(target)))
        if not target_terms:
            return False
        text_terms = set(self._rc_content_terms(text))
        overlap = sum(term in text_terms for term in target_terms)
        if len(target_terms) == 1:
            return overlap == 1
        if len(target_terms) == 2:
            return overlap == 2
        return overlap >= max(2, (3 * len(target_terms) + 4) // 5)

    def _rq_context_candidate_strength(self, slot, memory):
        target = str(slot.get("proof_anchor") or slot.get("target_surface") or "")
        target_terms = set(self._rc_content_terms(target))
        text_terms = set(self._rc_content_terms(self._rc_memory_target_text(memory)))
        overlap = len(target_terms & text_terms)
        coverage = overlap / max(1, len(target_terms))
        key_bonus = 0.0
        for key in slot.get("resolved_keys") or []:
            key_terms = set(self._rc_content_terms(key))
            if key_terms and key_terms.issubset(text_terms):
                key_bonus = max(key_bonus, 0.25)
        return coverage + key_bonus, overlap

    def _rq_repair_context_candidate_order(self, run, initial_seeds):
        """Recovery adds recall without automatically evicting a strong earlier hit."""
        slots = {
            str(slot.get("id") or ""): slot
            for slot in (run.get("plan") or {}).get("required_slots") or []
            if str(slot.get("evidence_role") or "").upper() == "REQUIREMENT"
        }
        candidate_by_id = {}
        for memory in (
            *(run.get("operation_candidates") or []), *(run.get("beliefs") or []),
            *(run.get("planning_seeds") or []), *(initial_seeds or []),
        ):
            if memory and memory.get("id"):
                candidate_by_id[memory["id"]] = memory
        round_outputs = {slot_id: {} for slot_id in slots}
        for item in run.get("trace") or []:
            round_id = int(item.get("retrieval_round") or 0)
            for slot_id in item.get("produces") or []:
                if slot_id not in round_outputs:
                    continue
                values = round_outputs[slot_id].setdefault(round_id, [])
                for memory_id in item.get("output_ids") or []:
                    if memory_id not in values:
                        values.append(memory_id)

        context = run.get("requirement_context_candidates") or {}
        reorders = []
        for slot_id, slot in slots.items():
            if (run.get("requirement_status") or {}).get(slot_id) == "FOUND":
                continue
            groups = round_outputs.get(slot_id) or {}
            rounds = sorted(round_id for round_id in groups if round_id > 0)
            if len(rounds) < 2:
                continue
            earlier_ids = [mid for rid in rounds[:-1] for mid in groups.get(rid, [])]
            strong = []
            for mid in earlier_ids:
                memory = candidate_by_id.get(mid)
                if not memory:
                    continue
                strength, overlap = self._rq_context_candidate_strength(slot, memory)
                if overlap >= 2 and strength >= 0.40:
                    strong.append((strength, mid))
            if not strong:
                continue
            strong.sort(key=lambda item: (-item[0], earlier_ids.index(item[1])))
            protected = strong[0][1]
            values = list(context.get(slot_id) or [])
            if protected not in values or values.index(protected) <= 1:
                continue
            values.remove(protected)
            values.insert(1 if values else 0, protected)
            context[slot_id] = values
            reorders.append({
                "requirement_id": slot_id,
                "protected_memory_id": protected,
                "latest_recovery_round": rounds[-1],
                "reason": "STRONG_EARLIER_CANDIDATE",
            })

        if reorders:
            run["requirement_context_candidates"] = context
            self._last_requirement_context_candidates = {k: list(v) for k, v in context.items()}
            pool = []
            for slot_id in slots:
                for memory_id in context.get(slot_id, []):
                    if memory_id not in pool:
                        pool.append(memory_id)
            run.setdefault("slot_support", {})[self.CONTEXT_POOL_KEY] = pool
        self._last_requirement_context_reorders = reorders
        run["requirement_context_reorders"] = reorders
        return run

    def _prepare_requirement_context_state(self, run, initial_seeds):
        run = super()._prepare_requirement_context_state(run, initial_seeds)
        return self._rq_repair_context_candidate_order(run, initial_seeds)

    def _controller_plan(self, ir, question, frame):
        plan = super()._controller_plan(ir, question, frame)
        plan.setdefault("query_spec", {})["semantic_ir_version"] = "minimal-v2-evidence-lookup"
        return plan

    def _semantic_controller(self, question, seeds, frame, context_map=None):
        del context_map
        self._last_option_probe_coverage = {}
        self._active_controller_seeds = list(seeds[:3])
        options = self._question_options(question) or {}
        hints = {
            "dates": list(getattr(frame, "dates", ()) or ()),
            "source_speaker": getattr(frame, "speaker_role", ""),
            "explicit_entities": list(getattr(frame, "entities", ()) or ()),
        }
        prompt = REQUIREMENT_CONTROLLER_POLICY + "\n" + REQUIREMENT_SCHEMA.format(
            question=question,
            options=json.dumps(options, ensure_ascii=False),
            hints=json.dumps(hints, ensure_ascii=False),
            seeds=json.dumps(self._rc_seed_payload(seeds), ensure_ascii=False),
        )
        raw_ir = {}
        try:
            response = self._llm_client.chat(
                [{"role": "user", "content": prompt}], temperature=0.0, max_tokens=650,
                response_format={"type": "json_object"},
            )
            usage = self._response_usage(response, prompt)
            raw_ir = self._parse_json(response.content)
            ir = self._rc_normalize_ir(raw_ir, question, frame)
            error = ""
        except Exception as exc:
            usage = {}
            ir = self._rc_normalize_ir({}, question, frame)
            error = str(exc)

        actions = list(getattr(self, "_last_requirement_normalization_actions", []) or [])
        warnings = [item for item in actions if item.get("action") == "GRAPH_WARNING"]
        supports, authorization = self._authorize_controller_answer(ir, seeds, frame)
        common = {
            "called": True,
            "fallback_reason": error or (authorization if ir.get("candidate") else ""),
            "error": error,
            "usage": usage,
            "answer_type": ir["answer_type"],
            "requirement_count": len(ir["requirements"]),
            "relation_count": len(ir["relations"]),
            "semantic_ir": self._rc_public_ir(ir),
            "controller_raw_ir": raw_ir,
            "normalization_actions": actions,
            "graph_warnings": warnings,
        }
        if supports is not None:
            candidate = ir["candidate"]
            telemetry = dict(common)
            telemetry.update({
                "route": "DIRECT", "route_source": "derived_from_authorized_candidate",
                "answer": candidate["answer"], "support_ref": candidate["support_ref"],
                "support_refs": [candidate["support_ref"]], "fallback_reason": "",
            })
            return supports, {}, telemetry

        plan = self._controller_plan(ir, question, frame)
        telemetry = dict(common)
        telemetry.update({
            "route": "PLAN", "route_source": "derived_from_candidate_authorization",
            "answer": "", "support_ref": "", "support_refs": [],
        })
        return None, plan, telemetry

    @staticmethod
    def _compact_reasoning_output_instruction(query_type: str) -> str:
        if str(query_type or "") != "multi_hop_clinical_deduction":
            return ""
        return (
            " OUTPUT EFFICIENCY FOR MULTI-HOP: satisfy any request to show memory content, "
            "reasoning, and judgment without repetition. Use at most one short `Evidence:` "
            "line and one short `Conclusion:`. The `Reasoning:` chain must be COMPLETE and "
            "mechanism-explicit; be concise only where completeness is preserved. Do not "
            "repeat the same participant facts across sections. Do not print session labels "
            "or provenance unless the QUESTION asks for a source/date. Spend answer tokens "
            "on the full causal/inferential chain, not on re-listing retrieved memories."
        )

    def prepare_batch_query(self, question, system_message=None, **kwargs):
        prepared = super().prepare_batch_query(question, system_message=system_message, **kwargs)
        extra = prepared.setdefault("extra", {})
        extra["read_contract_version"] = "minimal-ir-v3-evidence-lookup-requirements"
        extra["requirement_normalization_actions"] = list(
            getattr(self, "_last_requirement_normalization_actions", []) or []
        )
        extra["requirement_context_reorders"] = list(
            getattr(self, "_last_requirement_context_reorders", []) or []
        )
        instruction = self._compact_reasoning_output_instruction(kwargs.get("query_type"))
        if instruction:
            for message in prepared.get("messages") or []:
                if message.get("role") == "system":
                    message["content"] = str(message.get("content") or "") + instruction
                    break
        return prepared
