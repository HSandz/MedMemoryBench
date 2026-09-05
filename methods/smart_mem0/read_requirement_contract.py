"""Evidence-lookup requirement policy for the minimal semantic read IR.

The QUESTION remains the answer obligation. Requirements are only participant-memory
lookup variables whose retrieved values can change the answer or its grounded reasoning
path. General-domain mechanisms stay on relation bridges.
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

REQUIREMENTS answer only this question:
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

SEEDS may reveal a useful DERIVED lookup variable, but may not redefine the QUESTION or
pre-fill a participant fact. For example, a seed mentioning repeated morning glucose can
justify target="morning glycemic state"; it must not make the controller assert a glucose
value unless that value is returned as an authorized direct candidate.

Do NOT create requirements such as:
- "Can I take painkillers?", "what should I do?", or "whether this is safe" (answer obligations)
- "cortisol release", "HPA activation", "hepatic glucose output" when they are only
  general-domain mechanisms.
Instead, a painkiller question may require "work-related neck symptoms" plus DERIVED
"analgesic safety constraints". A causal question about late-night food and morning
symptoms may require QUESTION endpoints plus DERIVED "morning glycemic state" when that
participant-specific state is answer-sensitive.

RELATIONS are reasoning obligations between grounded requirement values:
- COMPARE compares two participant evidence variables.
- CAUSES requires an explicit stored causal relation.
- POSSIBLE_CAUSE connects grounded participant endpoints with authorized general knowledge.
- DEPENDS_ON records evidence dependency without asserting causality.
- TEMPORAL_ORDER orders grounded endpoints.
- INFER authorizes general-domain reasoning; use to="ANSWER" when the inference produces
  the requested answer rather than another participant evidence variable.
- CURRENT marks a current-state requirement.
- VERIFY_SOURCE asks for exact linked source evidence.

POSSIBLE_CAUSE is only for a genuine causal question; do not use it merely because the
QUESTION asks whether an action/recommendation is appropriate.
For POSSIBLE_CAUSE, INFER, and CAUSES, bridge_goal is a short reasoning obligation. It may
request a general mechanism at answer time but must not encode unstored participant history.

candidate is optional and only for one atomic QUESTION requirement when exactly one seed
directly contains the answer. Code independently authorizes it.
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
        """Enforce a small retrieval variable without performing semantic judgment."""
        target = " ".join(str(value or "").split()).strip(" -:;")
        if not target or "?" in target:
            return ""
        if len(target) > self.REQUIREMENT_TARGET_MAX_CHARS:
            return ""
        if len(self._rc_terms(target)) > self.REQUIREMENT_TARGET_MAX_TERMS:
            return ""
        return target

    def _rq_fallback_target(self, focus: str) -> str:
        """Create a bounded last-resort target only when controller output is unusable."""
        compact = " ".join(str(focus or "").split())
        if len(compact) <= self.REQUIREMENT_TARGET_MAX_CHARS:
            return compact
        words = compact.split()
        kept = []
        for word in words:
            candidate = " ".join([*kept, word])
            if len(candidate) > self.REQUIREMENT_TARGET_MAX_CHARS:
                break
            kept.append(word)
        return " ".join(kept)

    def _rc_normalize_ir(self, parsed: Dict[str, Any], question: str, frame: Any):
        """Normalize Requirement-v2 directly; do not pass through question-only v1 rules."""
        options = self._question_options(question) or {}
        answer_type = str(parsed.get("answer_type") or "TEXT").upper()
        answer_type = (
            "OPTION_SET"
            if options
            else answer_type if answer_type in VALID_ANSWER_TYPES else "TEXT"
        )
        subject_span = self._rc_question_span(parsed.get("subject_span"), question)

        raw_requirements = (
            parsed.get("requirements")
            if isinstance(parsed.get("requirements"), list)
            else []
        )
        requirements = []
        seen_ids = set()
        for index, raw in enumerate(raw_requirements[:4]):
            if not isinstance(raw, dict):
                continue
            kind = str(raw.get("grounding_kind") or "QUESTION").upper()
            if kind not in {"QUESTION", "DERIVED"}:
                continue

            focus = ""
            if kind == "QUESTION":
                focus = self._rc_question_span(raw.get("focus_span"), question)
                if not focus:
                    continue

            target = self._rq_compact_target(
                raw.get("target")
                or raw.get("evidence_target")  # one-release compatibility only
                or (focus if kind == "QUESTION" else "")
            )
            if not target:
                continue

            requirement_id = str(raw.get("id") or f"r{index + 1}")
            if (
                not re.fullmatch(r"r[\w-]{0,31}", requirement_id)
                or requirement_id in seen_ids
            ):
                requirement_id = f"r{index + 1}"
            while requirement_id in seen_ids:
                requirement_id += "x"
            seen_ids.add(requirement_id)

            time_constraint = self._rc_normalize_time_constraint(
                raw.get("time_constraint"), frame, question
            )
            if kind == "QUESTION":
                focus_binder = getattr(self, "_rc_bind_focus_time_constraint", None)
                if callable(focus_binder):
                    time_constraint = focus_binder(focus, time_constraint)

            hint = " ".join(str(raw.get("retrieval_hint") or target).split())[:240]
            requirements.append(
                {
                    "id": requirement_id,
                    "grounding_kind": kind,
                    "focus_span": focus,
                    "target": target,
                    "retrieval_hint": hint,
                    "time_constraint": time_constraint,
                }
            )

        if not requirements:
            stem = self._question_stem(question).strip()
            requirements = [
                {
                    "id": "r1",
                    "grounding_kind": "QUESTION",
                    "focus_span": stem,
                    "target": self._rq_fallback_target(stem),
                    "retrieval_hint": self._rq_fallback_target(stem),
                    "time_constraint": {
                        "axis": "",
                        "relation": "",
                        "anchor": "",
                        "end": "",
                    },
                }
            ]

        valid_nodes = {item["id"] for item in requirements}
        raw_relations = (
            parsed.get("relations") if isinstance(parsed.get("relations"), list) else []
        )
        relations = []
        for raw in raw_relations[:8]:
            if not isinstance(raw, dict):
                continue
            relation_type = str(raw.get("type") or "").upper()
            source = str(raw.get("from") or "")
            target = str(raw.get("to") or "")
            if relation_type not in VALID_IR_RELATIONS or source not in valid_nodes:
                continue
            if relation_type in {"CURRENT", "VERIFY_SOURCE"}:
                target = ""
            elif target != "ANSWER" and target not in valid_nodes:
                continue
            if relation_type in {
                "COMPARE",
                "CAUSES",
                "POSSIBLE_CAUSE",
                "DEPENDS_ON",
                "TEMPORAL_ORDER",
            } and target not in valid_nodes:
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

        # LOCATE is useful for an answer date or a graph dependency, but should
        # not turn every incidental mention of time into a retrieval obligation.
        temporal_dependencies = {
            endpoint
            for relation in relations
            if relation.get("type") == "TEMPORAL_ORDER"
            for endpoint in (relation.get("from"), relation.get("to"))
        }
        temporal_dependencies.update(
            relation.get("to")
            for relation in relations
            if relation.get("type") == "DEPENDS_ON"
        )
        if answer_type not in {"DATE", "RELATIVE_TIME"}:
            for requirement in requirements:
                constraint = requirement.get("time_constraint") or {}
                if (
                    constraint.get("relation") == "LOCATE"
                    and requirement["id"] not in temporal_dependencies
                ):
                    requirement["time_constraint"] = {
                        "axis": "",
                        "relation": "",
                        "anchor": "",
                        "end": "",
                    }

        candidate = parsed.get("candidate")
        if not isinstance(candidate, dict):
            candidate = None
        if candidate is not None:
            answer = str(candidate.get("answer") or "").strip()
            support_ref = str(candidate.get("support_ref") or "")
            only_question_requirement = (
                len(requirements) == 1
                and requirements[0].get("grounding_kind") == "QUESTION"
            )
            if (
                not answer
                or not re.fullmatch(r"\$seed[0-2]", support_ref)
                or not only_question_requirement
            ):
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
        """Compile proof/retrieval against target; focus_span remains provenance only."""
        slot = super()._requirement_slot(requirement, ir, compiled_mode)
        kind = str(requirement.get("grounding_kind") or "QUESTION").upper()
        target = str(requirement.get("target") or "").strip()
        slot["grounding_kind"] = kind
        slot["focus_span"] = str(requirement.get("focus_span") or "").strip()
        slot["target_surface"] = target
        slot["description"] = (
            str(requirement.get("retrieval_hint") or "").strip()
            or target
            or "participant evidence"
        )
        slot["resolved_keys"] = self._rc_resolve_target_keys(
            target, str(slot.get("subject_id") or "")
        )
        return slot

    def _controller_plan(self, ir, question, frame):
        plan = super()._controller_plan(ir, question, frame)
        plan.setdefault("query_spec", {})[
            "semantic_ir_version"
        ] = "minimal-v2-evidence-lookup"
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
        try:
            response = self._llm_client.chat(
                [{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=650,
                response_format={"type": "json_object"},
            )
            usage = self._response_usage(response, prompt)
            ir = self._rc_normalize_ir(
                self._parse_json(response.content), question, frame
            )
            error = ""
        except Exception as exc:
            usage = {}
            ir = self._rc_normalize_ir({}, question, frame)
            error = str(exc)

        supports, authorization = self._authorize_controller_answer(ir, seeds, frame)
        if supports is not None:
            candidate = ir["candidate"]
            telemetry = {
                "called": True,
                "route": "DIRECT",
                "route_source": "derived_from_authorized_candidate",
                "answer": candidate["answer"],
                "support_ref": candidate["support_ref"],
                "support_refs": [candidate["support_ref"]],
                "fallback_reason": "",
                "usage": usage,
                "answer_type": ir["answer_type"],
                "requirement_count": 1,
                "relation_count": 0,
                "semantic_ir": self._rc_public_ir(ir),
            }
            return supports, {}, telemetry

        plan = self._controller_plan(ir, question, frame)
        telemetry = {
            "called": True,
            "route": "PLAN",
            "route_source": "derived_from_candidate_authorization",
            "answer": "",
            "support_ref": "",
            "support_refs": [],
            "fallback_reason": error or (authorization if ir.get("candidate") else ""),
            "error": error,
            "usage": usage,
            "answer_type": ir["answer_type"],
            "requirement_count": len(ir["requirements"]),
            "relation_count": len(ir["relations"]),
            "semantic_ir": self._rc_public_ir(ir),
        }
        return None, plan, telemetry

    @staticmethod
    def _compact_reasoning_output_instruction(query_type: str) -> str:
        if str(query_type or "") != "multi_hop_clinical_deduction":
            return ""
        return (
            " OUTPUT EFFICIENCY FOR MULTI-HOP: satisfy any request to show memory content, "
            "reasoning, and judgment compactly. Use at most one short `Evidence:` line, "
            "one `Reasoning:` cause-to-effect chain containing the necessary mechanism, "
            "and one short `Conclusion:`. Do not repeat the same participant facts across "
            "sections. Do not print session labels or provenance unless the QUESTION asks "
            "for a source/date. Spend answer tokens on the causal/inferential chain, not "
            "on re-listing retrieved memories."
        )

    def prepare_batch_query(self, question, system_message=None, **kwargs):
        prepared = super().prepare_batch_query(
            question, system_message=system_message, **kwargs
        )
        prepared.setdefault("extra", {})[
            "read_contract_version"
        ] = "minimal-ir-v3-evidence-lookup-requirements"
        compact_instruction = self._compact_reasoning_output_instruction(
            kwargs.get("query_type")
        )
        if compact_instruction:
            for message in prepared.get("messages") or []:
                if message.get("role") == "system":
                    message["content"] = (
                        str(message.get("content") or "") + compact_instruction
                    )
                    break
        return prepared
