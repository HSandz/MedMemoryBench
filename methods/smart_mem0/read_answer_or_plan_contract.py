"""Answer-or-plan control flow for SmartMem0 reads.

The semantic controller gets one chance to finish an atomic grounded query from the
Top-3 seeds. Only when that candidate cannot be deterministically authorized do we
compile and execute Requirement-v2 retrieval. This keeps Requirement-v2 as the planned
branch rather than turning the controller into a mandatory semantic planner.
"""

import json
import re
from copy import deepcopy
from typing import Any, Dict

from .contracts import VALID_TEMPORAL_AXES
from .read_requirement_contract import (
    REQUIREMENT_CONTROLLER_POLICY,
    REQUIREMENT_SCHEMA,
)


ANSWER_OR_PLAN_PRIORITY = """
CONTROL-FLOW PRIORITY — ANSWER OR PLAN:
Before inventing DERIVED requirements or a reasoning graph, inspect the Top-3 SEEDS and
ask whether ONE seed already contains the exact participant-specific answer requested by
QUESTION.

If YES, emit candidate and exactly ONE minimal QUESTION requirement. Do not emit DERIVED
requirements, DEPENDS_ON, or INFER merely to explain an answer already explicit in that
seed. The candidate is a proposed final answer, not a retrieval hint. Code will still
reject it unless the cited seed structurally and atomically grounds it.

A direct candidate is allowed only for atomic extraction/localization:
- ENTITY, VALUE, or short TEXT explicitly present in one seed; or
- DATE when the requested temporal axis is present on that same seed and the candidate is
  exactly that date.
Never use the direct path for visible options, recommendations/advice, comparisons,
causal explanations, source-verification questions, or answers that require combining
multiple participant memories or applying a general-domain rule.

If NO, then and only then build the minimal Requirement-v2 evidence plan. DERIVED is not
"potentially useful context": create a DERIVED lookup only when retrieving that value is
necessary to answer the question or to complete a genuinely required reasoning bridge.
Prefer zero DERIVED nodes for ordinary extraction and usually no more than one for a
planned query. More nodes are justified only by genuinely independent answer-critical
evidence obligations.
"""

_DIRECT_BLOCK_PATTERNS = (
    r"\bwhy\b",
    r"\bexplain\b",
    r"\bcompare\b",
    r"\bversus\b",
    r"\bvs\.?\b",
    r"\bshould i\b",
    r"\bcan i\b",
    r"\bcould i\b",
    r"\bwhat should\b",
    r"\bwhat do i do\b",
    r"\brecommend(?:ed|ation|ing)?\b",
    r"\bis it safe\b",
    r"\bsafe to\b",
    r"\bwhat caused\b",
    r"\blikely cause\b",
    r"\bpossible cause\b",
)

_HARD_DIRECT_RELATIONS = {
    "COMPARE",
    "CAUSES",
    "POSSIBLE_CAUSE",
    "TEMPORAL_ORDER",
    "INFER",
    "VERIFY_SOURCE",
}


def _answer_or_plan_policy() -> str:
    """Reuse Requirement-v2 policy while removing its obsolete temporal fast-path ban."""
    old = (
        "candidate is optional and only for one atomic non-temporal QUESTION requirement when exactly\n"
        "one seed directly contains the answer. Code independently authorizes it."
    )
    new = (
        "candidate is optional only for one atomic QUESTION answer directly contained in exactly\n"
        "one seed. DATE localization may use a candidate when the requested temporal axis is\n"
        "present on that same seed. Code independently authorizes every candidate."
    )
    return ANSWER_OR_PLAN_PRIORITY + "\n" + REQUIREMENT_CONTROLLER_POLICY.replace(old, new)


class ReadAnswerOrPlanContractMixin:
    """Restore the original SmartMem0 semantic invariant: answer first, otherwise plan."""

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

    def _aop_direct_surface_allowed(self, question: str, ir: Dict[str, Any]) -> bool:
        if ir.get("visible_options"):
            return False
        if str(ir.get("answer_type") or "").upper() == "RELATIVE_TIME":
            return False
        text = self._rc_text(question)
        if any(re.search(pattern, text) for pattern in _DIRECT_BLOCK_PATTERNS):
            return False
        relation_types = {
            str(relation.get("type") or "").upper()
            for relation in ir.get("relations") or []
        }
        return not bool(relation_types & _HARD_DIRECT_RELATIONS)

    def _rc_normalize_ir(self, parsed: Dict[str, Any], question: str, frame: Any):
        """Keep a safe raw candidate available even when Requirement-v2 over-decomposes."""
        ir = super()._rc_normalize_ir(parsed, question, frame)
        raw_candidate = self._aop_raw_candidate(parsed)
        # Requirement-v2 intentionally used to drop temporal candidates and candidates
        # accompanied by DERIVED nodes. The answer-or-plan layer owns that decision now.
        ir["candidate"] = (
            raw_candidate
            if raw_candidate
            and ir.get("normalization_status") != "DEGRADED"
            and self._aop_direct_surface_allowed(question, ir)
            else None
        )
        return ir

    def _aop_direct_projection(self, ir: Dict[str, Any], question: str):
        """Project over-decomposed IR back to one atomic QUESTION obligation for fast exit."""
        if not ir.get("candidate") or not self._aop_direct_surface_allowed(question, ir):
            return None
        question_requirements = [
            requirement
            for requirement in ir.get("requirements") or []
            if str(requirement.get("grounding_kind") or "QUESTION").upper() == "QUESTION"
        ]
        if len(question_requirements) != 1:
            return None
        primary = deepcopy(question_requirements[0])
        projected = deepcopy(ir)
        projected["requirements"] = [primary]
        projected["relations"] = [
            deepcopy(relation)
            for relation in ir.get("relations") or []
            if str(relation.get("type") or "").upper() == "CURRENT"
            and relation.get("from") == primary.get("id")
        ]
        projected["graph_validation"] = self._rq_graph_validation(
            projected["requirements"], projected["relations"]
        )
        projected["normalization_actions"] = [
            *(projected.get("normalization_actions") or []),
            {
                "action": "DIRECT_CANDIDATE_PROJECTION",
                "reason": "ANSWERABLE_ATOMIC_SEED",
                "kept_requirement_id": primary.get("id"),
                "dropped_derived_count": max(
                    0, len(ir.get("requirements") or []) - 1
                ),
            },
        ]
        return projected

    def _authorize_controller_answer(self, ir, seeds, frame):
        """Authorize normal atomic candidates plus directly grounded DATE localization."""
        supports, reason = super()._authorize_controller_answer(ir, seeds, frame)
        if supports is not None:
            return supports, reason

        candidate = ir.get("candidate")
        requirements = ir.get("requirements") or []
        if (
            not candidate
            or len(requirements) != 1
            or ir.get("visible_options")
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
            return None, "TEMPORAL_QUERY_REQUIRES_RETRIEVAL"

        reference = str(candidate.get("support_ref") or "")
        match = re.fullmatch(r"\$seed(\d+)", reference)
        if not match or int(match.group(1)) >= min(3, len(seeds)):
            return None, "INVALID_SUPPORT_REF"
        valid = self._validate_fast_support(reference, seeds, frame)
        if not valid:
            return None, "STRUCTURAL_SUPPORT_REJECTED"

        actual = self._date_for(valid[0], axis)
        if not actual or not self._date_matches(actual, str(candidate.get("answer") or "")):
            return None, "DATE_CANDIDATE_NOT_ON_REQUESTED_AXIS"
        anchor = str(constraint.get("anchor") or "")
        if relation == "EXACT" and anchor and not self._date_matches(actual, anchor):
            return None, "TEMPORAL_FILTER_MISMATCH"
        if ir.get("relations") and not self._is_state_head(valid[0]):
            return None, "CURRENT_CANDIDATE_IS_NOT_STATE_HEAD"
        return valid, "AUTHORIZED"

    def _semantic_controller(self, question, seeds, frame, context_map=None):
        """One LLM call decides: authorized atomic answer, otherwise Requirement-v2 plan."""
        del context_map
        self._last_option_probe_coverage = {}
        self._active_controller_seeds = list(seeds[:3])
        options = self._question_options(question) or {}
        hints = {
            "dates": list(getattr(frame, "dates", ()) or ()),
            "source_speaker": getattr(frame, "speaker_role", ""),
            "explicit_entities": list(getattr(frame, "entities", ()) or ()),
        }
        prompt = _answer_or_plan_policy() + "\n" + REQUIREMENT_SCHEMA.format(
            question=question,
            options=json.dumps(options, ensure_ascii=False),
            hints=json.dumps(hints, ensure_ascii=False),
            seeds=json.dumps(self._rc_seed_payload(seeds), ensure_ascii=False),
        )
        raw_ir = {}
        try:
            response = self._llm_client.chat(
                [{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=650,
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

        projection = self._aop_direct_projection(ir, question)
        supports = None
        authorization = "NO_ATOMIC_CANDIDATE"
        active_ir = ir
        if projection is not None:
            supports, authorization = self._authorize_controller_answer(
                projection, seeds, frame
            )
            if supports is not None:
                active_ir = projection

        actions = list(active_ir.get("normalization_actions") or [])
        self._last_requirement_normalization_actions = list(actions)
        warnings = [item for item in actions if item.get("action") == "GRAPH_WARNING"]
        common = {
            "called": True,
            "fallback_reason": error or (
                authorization if ir.get("candidate") and supports is None else ""
            ),
            "error": error,
            "usage": usage,
            "answer_type": active_ir["answer_type"],
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
        }
        if supports is not None:
            candidate = active_ir["candidate"]
            telemetry = dict(common)
            telemetry.update(
                {
                    "route": "DIRECT",
                    "route_source": "answer_or_plan_authorized_atomic_candidate",
                    "answer": candidate["answer"],
                    "support_ref": candidate["support_ref"],
                    "support_refs": [candidate["support_ref"]],
                    "fallback_reason": "",
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
            }
        )
        return None, plan, telemetry
