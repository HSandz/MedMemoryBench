"""Answer-or-plan control flow for SmartMem0 reads.

The single semantic controller either proposes a complete answer already contained in one
seed or emits the minimal evidence obligations required for retrieval. Candidate
propositions are a generic evidence-evaluation primitive; visible multiple-choice options
are only one producer. DIRECT is semantic completeness, never answer length or language.
"""

import json
import re
from copy import deepcopy
from typing import Any, Dict

from .contracts import VALID_TEMPORAL_AXES
from .read_requirement_contract import REQUIREMENT_CONTROLLER_POLICY, REQUIREMENT_SCHEMA


ANSWER_OR_PLAN_PRIORITY = """
CONTROL-FLOW PRIORITY — ANSWER OR PLAN:
Inspect exactly the Top-3 SEEDS before building a retrieval plan. Candidate-proposition
pack entries are additional bounded retrieval candidates, NOT extra seeds.

CANONICAL OUTPUT:
Always emit answer_mode as exactly one of:
- EXTRACT: the answer is explicitly stored in participant memory and requires no semantic
  bridge beyond selecting/rendering the stored content.
- COMPARE: the answer requires comparing two or more participant-specific evidence values.
- ADVISE: the answer is a recommendation/action decision.
- EXPLAIN: the answer requires explaining a reason/mechanism.
- INFER: the answer requires combining participant evidence with a general-domain bridge.
- SELECT: the answer requires deciding among explicit candidate propositions.
This vocabulary is language-independent. Infer it from meaning, not from cue words.
If uncertain between EXTRACT and any other mode, choose the non-EXTRACT mode.

CANDIDATE PROPOSITION CONTRACT:
An explicit candidate proposition is a proposition to EVALUATE using memory, not an
evidence value that memory itself must supply. Visible answer options are one producer,
but the same primitive can represent candidate actions, hypotheses, explanations, or
entities supplied structurally by another caller.

When CANDIDATE PROPOSITIONS are supplied, also emit:
"proposition_semantics": {
  "predicate": "short semantic predicate a selected proposition must satisfy"
}
and:
"proposition_evidence": {
  "<proposition id>": [
    {
      "memory_ref": "$seed0|$seed1|$seed2|one packet ref authorized for this proposition",
      "relation": "SUPPORTS|CONTRADICTS|CONTEXT_FOR|UNKNOWN",
      "confidence": 0.0
    }
  ]
}

Relation meanings are relative to the QUESTION predicate:
- SUPPORTS: the memory materially supports this proposition satisfying the predicate.
- CONTRADICTS: the memory materially argues against this proposition satisfying it.
- CONTEXT_FOR: the memory is genuinely relevant and useful but does not by itself support
  or contradict the proposition strongly enough.
- UNKNOWN: the semantic relation cannot be determined reliably.

confidence means P(the memory↔proposition RELATION LABEL is correct). It is NOT the
probability that the proposition itself is correct.
- If relation confidence is below 0.50, use UNKNOWN.
- Reserve SUPPORTS/CONTRADICTS for clear material evidence; when confidence is below 0.70,
  prefer UNKNOWN rather than overstating a strong evidential relation.
- CONTEXT_FOR may be used when relevance is clear even though support direction is not.
- No annotation, UNKNOWN, or no personal-memory support does NOT mean the proposition is
  false. A proposition can still be correct via later retrieved evidence, elimination, or
  an authorized world-knowledge bridge.
- Do NOT choose final proposition labels in LLM #1. LLM #2 owns the final answer.
- Do not invent memory refs. Packet refs are proposition-scoped. Top-3 seed refs are global.

Emit candidate only when answer_mode=EXTRACT and EXACTLY ONE cited seed already contains
the COMPLETE participant-specific answer to QUESTION. The answer may be an entity, value,
date, sentence, or short paragraph. Length never decides routing. Every proposition in
the candidate must be grounded by that one seed; do not combine seeds, add a general-domain
rule, infer a cause, or turn advice/recommendation into a direct memory answer.

ANSWER-ROLE INVARIANT:
- answer_type describes the semantic role of the requested answer, not the surface form
  of a convenient seed sentence.
- A request for a relative timing/occasion/delay/sequence answer is RELATIVE_TIME even if
  QUESTION also supplies an exact document/event date as a filter.
- A request whose answer is a calendar date/time is DATE.
- A candidate that answers a related WHY/PURPOSE/STATE but not the requested answer role
  is not complete and must not be emitted.
Interpret these meanings in whatever language QUESTION uses; do not rely on cue words.

RETRIEVAL-HINT CONTRACT:
retrieval_hint is a SEARCH-ONLY semantic projection of ONE evidence obligation. It is not
a paraphrase of the final answer and never carries proof or terminal authority.
It should contain:
1. the evidence family/entity/state/event to retrieve;
2. the answer-bearing relation needed from that evidence, when relevant;
3. optional short aliases useful for cross-language or stored-phrase recall.
It should NOT contain:
- the expected/final answer;
- a causal/mechanistic conclusion that memory itself need not store;
- generic conversational framing;
- temporal selector words whose semantics already live in time_constraint;
- purpose/reason wording when the requested field is timing, value, entity, or state.
Keep it concise, selector-neutral, and in QUESTION's language by default.
Examples by semantics, not dataset:
- target: "timing of drinking milk coffee"; time_constraint: EXACT(record date)
  retrieval_hint: "milk coffee consumption after meal / post-meal timing"
  NOT "late-night energy kick during overtime".
- target: "start of medication X"; time_constraint: EARLIEST(event_time)
  retrieval_hint: "medication X use / prescription / taking medication X"
  NOT "start date or earliest onset".
- target: "antibiotic instructed to avoid"
  retrieval_hint: "antibiotic avoidance / allergy / contraindication instruction".
The deterministic runtime may broaden or drop retrieval_hint during recovery; therefore
target remains the semantic obligation and time_constraint remains the selector.

When the answer is already explicit in one seed, emit exactly ONE minimal QUESTION
requirement and no DERIVED requirements. Do not add DEPENDS_ON/INFER merely to explain an
answer already present in the seed.

Otherwise emit no candidate and build the smallest Requirement-v2 evidence plan.
DERIVED is an answer-critical participant-memory lookup variable, not optional context.
"""

_HARD_DIRECT_RELATIONS = {
    "COMPARE",
    "CAUSES",
    "POSSIBLE_CAUSE",
    "TEMPORAL_ORDER",
    "INFER",
    "VERIFY_SOURCE",
}

_VALID_ANSWER_MODES = {"EXTRACT", "COMPARE", "ADVISE", "EXPLAIN", "INFER", "SELECT"}


def _answer_or_plan_policy() -> str:
    return REQUIREMENT_CONTROLLER_POLICY + "\n" + ANSWER_OR_PLAN_PRIORITY


class ReadAnswerOrPlanContractMixin:
    @staticmethod
    def _aop_raw_candidate(parsed: Any):
        if not isinstance(parsed, dict) or not isinstance(parsed.get("candidate"), dict):
            return None
        candidate = parsed["candidate"]
        answer = str(candidate.get("answer") or "").strip()
        support_ref = str(candidate.get("support_ref") or "")
        if not answer or not re.fullmatch(r"\$seed[0-2]", support_ref):
            return None
        return {"answer": answer, "support_ref": support_ref}

    @staticmethod
    def _aop_answer_mode(parsed: Any) -> str:
        if not isinstance(parsed, dict):
            return ""
        mode = str(parsed.get("answer_mode") or "").upper()
        return mode if mode in _VALID_ANSWER_MODES else ""

    @staticmethod
    def _aop_proposition_semantics(parsed: Any) -> Dict[str, str]:
        if not isinstance(parsed, dict):
            return {}
        raw = parsed.get("proposition_semantics")
        if not isinstance(raw, dict):
            raw = parsed.get("option_semantics")
        if not isinstance(raw, dict):
            return {}
        predicate = " ".join(
            str(raw.get("predicate") or raw.get("selection_predicate") or "").split()
        )[:180]
        return {"predicate": predicate} if predicate else {}

    def _aop_direct_surface_allowed(self, question: str, ir: Dict[str, Any]) -> bool:
        del question
        if ir.get("visible_options") or ir.get("candidate_propositions"):
            return False
        if str(ir.get("answer_mode") or "").upper() != "EXTRACT":
            return False
        if str(ir.get("answer_type") or "").upper() in {"RELATIVE_TIME", "OPTION_SET"}:
            return False
        relation_types = {
            str(relation.get("type") or "").upper()
            for relation in ir.get("relations") or []
        }
        return not bool(relation_types & _HARD_DIRECT_RELATIONS)

    def _rc_normalize_ir(self, parsed: Dict[str, Any], question: str, frame: Any):
        ir = super()._rc_normalize_ir(parsed, question, frame)
        raw_candidate = self._aop_raw_candidate(parsed)
        answer_mode = self._aop_answer_mode(parsed)
        if not answer_mode:
            if ir.get("visible_options"):
                answer_mode = "SELECT"
            elif any(
                str(r.get("type") or "").upper() == "COMPARE"
                for r in ir.get("relations") or []
            ):
                answer_mode = "COMPARE"
            elif any(
                str(r.get("type") or "").upper()
                in {"INFER", "POSSIBLE_CAUSE", "CAUSES"}
                for r in ir.get("relations") or []
            ):
                answer_mode = "INFER"
            else:
                answer_mode = "UNKNOWN"
        ir["answer_mode"] = answer_mode
        ir["proposition_semantics"] = self._aop_proposition_semantics(parsed)
        requirements = ir.get("requirements") or []
        single_question = (
            len(requirements) == 1
            and str(requirements[0].get("grounding_kind") or "QUESTION").upper()
            == "QUESTION"
        )
        ir["candidate"] = (
            raw_candidate
            if raw_candidate
            and single_question
            and self._aop_direct_surface_allowed(question, ir)
            else None
        )
        return ir

    def _rc_public_ir(self, ir):
        public = super()._rc_public_ir(ir)
        public["answer_mode"] = str(ir.get("answer_mode") or "UNKNOWN")
        if ir.get("candidate_propositions"):
            public["candidate_propositions"] = deepcopy(
                ir.get("candidate_propositions") or {}
            )
            public["proposition_semantics"] = deepcopy(
                ir.get("proposition_semantics") or {}
            )
            public["proposition_evidence"] = deepcopy(
                ir.get("proposition_evidence") or {}
            )
        return public

    def _aop_direct_projection(self, ir: Dict[str, Any], question: str):
        if not ir.get("candidate") or not self._aop_direct_surface_allowed(question, ir):
            return None
        requirements = ir.get("requirements") or []
        if (
            len(requirements) != 1
            or str(requirements[0].get("grounding_kind") or "QUESTION").upper()
            != "QUESTION"
        ):
            return None
        projected = deepcopy(ir)
        projected["normalization_actions"] = [
            *(projected.get("normalization_actions") or []),
            {
                "action": "DIRECT_CANDIDATE_VALIDATED",
                "reason": "CANONICAL_EXTRACT_COMPLETE_ONE_SEED_ANSWER",
                "kept_requirement_id": requirements[0].get("id"),
            },
        ]
        return projected

    def _authorize_controller_answer(self, ir, seeds, frame):
        supports, reason = super()._authorize_controller_answer(ir, seeds, frame)
        if supports is not None:
            return supports, reason
        candidate = ir.get("candidate")
        requirements = ir.get("requirements") or []
        if (
            not candidate
            or len(requirements) != 1
            or ir.get("visible_options")
            or ir.get("candidate_propositions")
            or str(ir.get("answer_mode") or "").upper() != "EXTRACT"
            or str(ir.get("answer_type") or "").upper() != "DATE"
        ):
            return None, reason
        if any(
            str(relation.get("type") or "").upper() != "CURRENT"
            for relation in ir.get("relations") or []
        ):
            return None, "RELATIONAL_QUERY_REQUIRES_RETRIEVAL"
        requirement = requirements[0]
        constraint = requirement.get("time_constraint") or {}
        axis = str(constraint.get("axis") or "")
        relation = str(constraint.get("relation") or "").upper()
        if axis not in VALID_TEMPORAL_AXES or relation not in {"", "LOCATE", "EXACT"}:
            return None, "TEMPORAL_SELECTOR_REQUIRES_RETRIEVAL"
        reference = str(candidate.get("support_ref") or "")
        match = re.fullmatch(r"\$seed(\d+)", reference)
        if not match or int(match.group(1)) >= min(3, len(seeds)):
            return None, "INVALID_SUPPORT_REF"
        valid = self._validate_fast_support(reference, seeds, frame)
        if not valid:
            return None, "STRUCTURAL_SUPPORT_REJECTED"
        actual = self._date_for(valid[0], axis)
        if not actual or not self._date_matches(
            actual, str(candidate.get("answer") or "")
        ):
            return None, "DATE_CANDIDATE_NOT_ON_REQUESTED_AXIS"
        anchor = str(constraint.get("anchor") or "")
        if relation == "EXACT" and anchor and not self._date_matches(actual, anchor):
            return None, "TEMPORAL_FILTER_MISMATCH"
        if ir.get("relations") and not self._is_state_head(valid[0]):
            return None, "CURRENT_CANDIDATE_IS_NOT_STATE_HEAD"
        return valid, "AUTHORIZED"

    def _rc_search_query(self, slot: Dict[str, Any], question: str) -> str:
        """Compile the semantic retrieval projection without schema-hint pollution."""
        hint = str(slot.get("retrieval_hint") or "").strip()
        target = str(slot.get("target_surface") or "").strip()
        parts = [hint, target]
        if not any(parts):
            parts.append(str(question or "").strip())
        unique = []
        for part in parts:
            if part and self._rc_text(part) not in {
                self._rc_text(x) for x in unique
            }:
                unique.append(part)
        return " | ".join(unique)

    def _rc_bundle_query(self, slots, question: str) -> str:
        parts = []
        for slot in slots:
            for value in (
                str(slot.get("retrieval_hint") or "").strip(),
                str(slot.get("target_surface") or "").strip(),
            ):
                if value and self._rc_text(value) not in {
                    self._rc_text(x) for x in parts
                }:
                    parts.append(value)
        if not parts and question:
            parts.append(str(question).strip())
        return " | ".join(parts)

    def _controller_plan(self, ir, question, frame):
        plan = super()._controller_plan(ir, question, frame)
        mode = str(ir.get("answer_mode") or "UNKNOWN").upper()
        spec = plan.setdefault("query_spec", {})
        spec["answer_mode"] = mode
        if mode in {"ADVISE", "EXPLAIN", "INFER"}:
            spec["requires_inference"] = True
            spec["world_knowledge_bridge_allowed"] = True
        elif mode in {"EXTRACT", "COMPARE", "SELECT"}:
            spec["requires_inference"] = bool(spec.get("requires_inference", False))

        propositions = deepcopy(ir.get("candidate_propositions") or {})
        if propositions:
            plan["candidate_propositions"] = propositions
            plan["proposition_semantics"] = deepcopy(
                ir.get("proposition_semantics") or {}
            )
            plan["proposition_evidence"] = deepcopy(
                ir.get("proposition_evidence") or {}
            )
            semantic_ir = plan.setdefault("semantic_ir", {})
            semantic_ir["candidate_propositions"] = deepcopy(propositions)
            semantic_ir["proposition_semantics"] = deepcopy(
                ir.get("proposition_semantics") or {}
            )
            semantic_ir["proposition_evidence"] = deepcopy(
                ir.get("proposition_evidence") or {}
            )
        return plan

    def _semantic_controller(self, question, seeds, frame, context_map=None):
        self._last_option_probe_coverage = {}
        self._last_option_probe_relations = {}
        self._last_option_support_views = {}
        self._last_option_contradict_views = {}
        self._last_option_semantics = {}
        self._last_candidate_propositions = {}
        self._last_candidate_proposition_pack = {}
        self._last_proposition_probe_coverage = {}
        self._last_proposition_relation_views = {}
        self._last_proposition_support_views = {}
        self._last_proposition_contradict_views = {}
        self._last_proposition_context_views = {}
        self._last_proposition_unknown_views = {}
        self._last_proposition_semantics = {}

        reset_terminal = getattr(self, "_terminal_reset_state", None)
        if callable(reset_terminal):
            reset_terminal()
        self._active_controller_seeds = list(seeds[:3])

        options = self._question_options(question) or {}
        propositions = {}
        normalize_propositions = getattr(self, "_normalize_candidate_propositions", None)
        if callable(normalize_propositions):
            propositions = normalize_propositions(options)
            if not propositions and isinstance(context_map, dict):
                propositions = normalize_propositions(
                    context_map.get("candidate_propositions")
                )

        pack_builder = getattr(self, "_build_candidate_proposition_pack", None)
        proposition_pack = (
            pack_builder(
                propositions,
                frame=frame,
                seeds=seeds[:3],
                question=question,
            )
            if callable(pack_builder) and propositions
            else {}
        )

        hints = {
            "dates": list(getattr(frame, "dates", ()) or ()),
            "source_speaker": getattr(frame, "speaker_role", ""),
            "explicit_entities": list(getattr(frame, "entities", ()) or ()),
        }
        proposition_instruction = ""
        if propositions:
            proposition_instruction = (
                "\nCANDIDATE PROPOSITIONS (structural, not verdicts):\n"
                + json.dumps(propositions, ensure_ascii=False)
                + "\nCANDIDATE PROPOSITION PACK (bounded retrieval candidates; not extra seeds):\n"
                + json.dumps(proposition_pack, ensure_ascii=False)
                + "\nAnnotate proposition_evidence only with $seed0..$seed2 or packet refs "
                "authorized under candidate_refs for that proposition. Do not output final "
                "selected proposition labels in this controller response."
            )

        prompt = (
            _answer_or_plan_policy()
            + "\n"
            + REQUIREMENT_SCHEMA.format(
                question=question,
                options=json.dumps(options, ensure_ascii=False),
                hints=json.dumps(hints, ensure_ascii=False),
                seeds=json.dumps(self._rc_seed_payload(seeds), ensure_ascii=False),
            )
            + '\nAlso include top-level "answer_mode".'
            + (
                '\nWhen candidate propositions are supplied, also include top-level '
                '"proposition_semantics" and "proposition_evidence" using the canonical '
                "contract above."
                if propositions
                else ""
            )
            + proposition_instruction
        )

        raw_ir = {}
        try:
            response = self._llm_client.chat(
                [{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=820,
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

        ir["candidate_propositions"] = dict(propositions)
        if propositions and not ir.get("proposition_semantics"):
            ir["proposition_semantics"] = {}
        normalizer = getattr(self, "_normalize_candidate_proposition_evidence", None)
        proposition_evidence = (
            normalizer(
                raw_ir.get("proposition_evidence"),
                proposition_pack,
                seeds[:3],
                propositions,
            )
            if callable(normalizer) and propositions
            else {}
        )
        ir["proposition_evidence"] = deepcopy(proposition_evidence)

        activator = getattr(self, "_activate_candidate_proposition_evidence", None)
        if callable(activator) and propositions:
            activator(
                propositions,
                proposition_evidence,
                visible_options=bool(options),
                predicate=str(
                    (ir.get("proposition_semantics") or {}).get("predicate") or ""
                ),
            )

        projection = self._aop_direct_projection(ir, question)
        supports, authorization, active_ir = None, "NO_COMPLETE_CANDIDATE", ir
        if projection is not None:
            supports, authorization = self._authorize_controller_answer(
                projection, seeds, frame
            )
            if supports is not None:
                active_ir = projection

        actions = list(active_ir.get("normalization_actions") or [])
        self._last_requirement_normalization_actions = list(actions)
        warnings = [
            item for item in actions if item.get("action") == "GRAPH_WARNING"
        ]
        common = {
            "called": True,
            "fallback_reason": error
            or (authorization if ir.get("candidate") and supports is None else ""),
            "error": error,
            "usage": usage,
            "answer_type": active_ir["answer_type"],
            "answer_mode": active_ir.get("answer_mode", "UNKNOWN"),
            "requirement_count": len(active_ir["requirements"]),
            "relation_count": len(active_ir["relations"]),
            "semantic_ir": self._rc_public_ir(active_ir),
            "controller_raw_ir": raw_ir,
            "normalized_ir": self._rc_public_ir(active_ir),
            "normalization_status": active_ir.get("normalization_status", "VALID"),
            "graph_validation": active_ir.get("graph_validation") or {},
            "normalization_actions": actions,
            "graph_warnings": warnings,
            "candidate_authorization": authorization,
            "candidate_proposition_pack_size": len(
                proposition_pack.get("candidates") or []
            ),
            "candidate_proposition_count": len(propositions),
        }

        if supports is not None:
            candidate = active_ir["candidate"]
            answer = candidate["answer"]
            renderer = getattr(self, "_terminal_render_seed_answer", None)
            if callable(renderer):
                answer = renderer(question, active_ir, candidate["answer"], supports[0])
            telemetry = dict(common)
            telemetry.update(
                {
                    "route": "DIRECT",
                    "route_source": "answer_or_plan_complete_grounded_seed",
                    "answer": answer,
                    "support_ref": candidate["support_ref"],
                    "support_refs": [candidate["support_ref"]],
                    "fallback_reason": "",
                    "terminal_rendered": answer != candidate["answer"],
                }
            )
            return supports, {}, telemetry

        plan = self._controller_plan(ir, question, frame)
        telemetry = dict(common)
        telemetry.update(
            {
                "route": "PLAN",
                "route_source": "answer_or_plan_requires_retrieval",
                "answer": "",
                "support_ref": "",
                "support_refs": [],
                "terminal_rendered": False,
            }
        )
        return None, plan, telemetry
