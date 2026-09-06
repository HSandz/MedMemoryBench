"""Complete one-memory terminal answers for SmartMem0.

The controller may terminate on a grounded answer of any length. Termination depends on
semantic completeness, not on whether the answer is a scalar, a sentence, or a short paragraph.
This layer never adds an LLM call.
"""

import json
import re
from typing import Any, Dict, List

from .read_answer_or_plan_contract import (
    _DIRECT_BLOCK_PATTERNS,
    _HARD_DIRECT_RELATIONS,
    _answer_or_plan_policy,
)
from .read_requirement_contract import REQUIREMENT_SCHEMA


TERMINAL_COMPLETENESS_POLICY = """
TERMINAL COMPLETENESS OVERRIDE:
A candidate is allowed whenever ONE cited seed already contains the complete participant-specific
answer and no new general-domain rule or cross-memory synthesis is required. Do NOT reject or omit
a candidate merely because the faithful answer is a full sentence or a short paragraph. Answer
length is never a routing criterion. Every proposition in candidate.answer must be grounded by the
single cited seed. Prefer a complete faithful proposition over a one-word state label when the
QUESTION asks for a status, change, trend, or description. If the answer is not completely present
in one seed, emit no candidate and build the minimal evidence plan instead.
"""

_PROPOSITION_QUESTION_PATTERNS = (
    r"\bhow (?:has|have|did|is|are|was|were)\b",
    r"\bstatus\b",
    r"\bchang(?:e|ed|ing)\b",
    r"\bcompared\b",
    r"\btrend\b",
    r"\bwhat happened\b",
    r"\bdescribe\b",
)


class ReadTerminalAnswerContractMixin:
    """Own the distinction between a controller proposal and a complete terminal answer."""

    TERMINAL_ANSWER_CONTRACT_VERSION = "complete-grounded-answer-v1"

    def _aop_direct_surface_allowed(self, question: str, ir: Dict[str, Any]) -> bool:
        """Permit explicit one-memory comparison while keeping reasoning/advice gates."""
        if ir.get("visible_options"):
            return False
        if str(ir.get("answer_type") or "").upper() == "RELATIVE_TIME":
            return False
        text = self._rc_text(question)
        relaxed_patterns = tuple(
            pattern
            for pattern in _DIRECT_BLOCK_PATTERNS
            if pattern not in {r"\bcompare\b", r"\bversus\b", r"\bvs\.?\b"}
        )
        if any(re.search(pattern, text) for pattern in relaxed_patterns):
            return False
        relation_types = {
            str(relation.get("type") or "").upper()
            for relation in ir.get("relations") or []
        }
        if relation_types & (set(_HARD_DIRECT_RELATIONS) - {"COMPARE"}):
            return False
        if "COMPARE" in relation_types:
            question_requirements = [
                requirement
                for requirement in ir.get("requirements") or []
                if str(requirement.get("grounding_kind") or "QUESTION").upper() == "QUESTION"
            ]
            return len(question_requirements) == 1
        return True

    @classmethod
    def _terminal_stem(cls, token: Any) -> str:
        """Cheap morphology for proposal grounding only; never used as retrieval proof."""
        value = cls._rc_text(token)
        if len(value) > 6 and value.endswith("ing"):
            return value[:-3]
        if len(value) > 5 and value.endswith("ied"):
            return value[:-3] + "y"
        if len(value) > 5 and value.endswith("ed"):
            return value[:-2]
        if len(value) > 5 and value.endswith("es"):
            return value[:-2]
        if len(value) > 4 and value.endswith("s"):
            return value[:-1]
        return value

    def _terminal_grounding_terms(self, value: Any) -> List[str]:
        return [self._terminal_stem(term) for term in self._rc_content_terms(value)]

    def _terminal_memory_text(self, memory: Dict[str, Any]) -> str:
        return " ".join(
            str(value or "")
            for value in (
                memory.get("claim"),
                memory.get("value"),
                memory.get("verbatim_value"),
                str(memory.get("object_anchor") or "").replace("_", " "),
                " ".join(memory.get("entities") or []),
                " ".join(memory.get("scope_entities") or []),
            )
        )

    def _terminal_answer_grounded(self, answer: str, memory: Dict[str, Any]) -> bool:
        """Authorize a one-seed proposition with no maximum answer-length rule."""
        answer = str(answer or "").strip()
        if not answer:
            return False
        evidence = self._terminal_memory_text(memory)
        if self._rc_token_sequence_present(answer, evidence):
            return True
        answer_numbers = set(re.findall(r"\d+(?:\.\d+)?", self._rc_text(answer)))
        evidence_numbers = set(re.findall(r"\d+(?:\.\d+)?", self._rc_text(evidence)))
        if answer_numbers and not answer_numbers.issubset(evidence_numbers):
            return False
        polarity = {"no", "not", "never", "without", "denied", "deny", "stopped"}
        if not polarity.intersection(self._rc_terms(answer)).issubset(
            polarity.intersection(self._rc_terms(evidence))
        ):
            return False
        answer_terms = self._terminal_grounding_terms(answer)
        evidence_terms = set(self._terminal_grounding_terms(evidence))
        if not answer_terms:
            return False
        return sum(term in evidence_terms for term in answer_terms) / len(answer_terms) >= 0.82

    @staticmethod
    def _terminal_question_needs_proposition(question: str) -> bool:
        text = " ".join(str(question or "").casefold().split())
        return any(re.search(pattern, text) for pattern in _PROPOSITION_QUESTION_PATTERNS)

    def _terminal_render_seed_answer(
        self,
        question: str,
        ir: Dict[str, Any],
        candidate_answer: str,
        memory: Dict[str, Any],
    ) -> str:
        """Render stored semantics without introducing a new claim."""
        answer_type = str(ir.get("answer_type") or "TEXT").upper()
        candidate_answer = str(candidate_answer or "").strip()
        if answer_type in {"ENTITY", "VALUE", "DATE"}:
            return candidate_answer
        if answer_type != "TEXT":
            return candidate_answer
        if len(self._rc_content_terms(candidate_answer)) >= 4:
            return candidate_answer
        if self._terminal_question_needs_proposition(question):
            claim = " ".join(str(memory.get("claim") or "").split())
            if claim:
                return claim
        return candidate_answer or " ".join(str(memory.get("claim") or "").split())

    def _authorize_controller_answer(self, ir, seeds, frame):
        """Retain strict atomic checks, then allow a fully grounded TEXT proposition."""
        supports, reason = super()._authorize_controller_answer(ir, seeds, frame)
        if supports is not None:
            return supports, reason
        candidate = ir.get("candidate")
        requirements = ir.get("requirements") or []
        if (
            str(ir.get("answer_type") or "TEXT").upper() != "TEXT"
            or not candidate
            or len(requirements) != 1
            or ir.get("visible_options")
        ):
            return None, reason
        relations = ir.get("relations") or []
        if any(str(edge.get("type") or "").upper() != "CURRENT" for edge in relations):
            return None, "RELATIONAL_QUERY_REQUIRES_RETRIEVAL"
        constraint = requirements[0].get("time_constraint") or {}
        if getattr(frame, "dates", ()) and not constraint.get("axis"):
            return None, "TEMPORAL_CONSTRAINT_REQUIRES_PLAN"
        if constraint.get("axis") or constraint.get("relation"):
            return None, "TEMPORAL_QUERY_REQUIRES_RETRIEVAL"
        reference = str(candidate.get("support_ref") or "")
        match = re.fullmatch(r"\$seed(\d+)", reference)
        if not match or int(match.group(1)) >= min(3, len(seeds)):
            return None, "INVALID_SUPPORT_REF"
        valid = self._validate_fast_support(reference, seeds, frame)
        if not valid:
            return None, "STRUCTURAL_SUPPORT_REJECTED"
        memory = valid[0]
        if relations and not self._is_state_head(memory):
            return None, "CURRENT_CANDIDATE_IS_NOT_STATE_HEAD"
        if not self._terminal_answer_grounded(str(candidate.get("answer") or ""), memory):
            return None, "ANSWER_PROPOSITION_NOT_GROUNDED"
        return valid, "AUTHORIZED_COMPLETE_PROPOSITION"

    def _semantic_controller(self, question, seeds, frame, context_map=None):
        """One LLM call: complete grounded answer of any length, otherwise minimal plan."""
        del context_map
        self._last_option_probe_coverage = {}
        self._last_terminal_closure_diagnostic = {}
        self._last_requirement_answerability = {}
        self._last_answerability_state = ""
        self._last_answerability_candidate_lifecycle = []
        self._active_controller_seeds = list(seeds[:3])
        options = self._question_options(question) or {}
        hints = {
            "dates": list(getattr(frame, "dates", ()) or ()),
            "source_speaker": getattr(frame, "speaker_role", ""),
            "explicit_entities": list(getattr(frame, "entities", ()) or ()),
        }
        prompt = (
            _answer_or_plan_policy()
            + "\n"
            + TERMINAL_COMPLETENESS_POLICY
            + "\n"
            + REQUIREMENT_SCHEMA.format(
                question=question,
                options=json.dumps(options, ensure_ascii=False),
                hints=json.dumps(hints, ensure_ascii=False),
                seeds=json.dumps(self._rc_seed_payload(seeds), ensure_ascii=False),
            )
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
            supports, authorization = self._authorize_controller_answer(projection, seeds, frame)
            if supports is not None:
                active_ir = projection

        actions = list(active_ir.get("normalization_actions") or [])
        self._last_requirement_normalization_actions = list(actions)
        warnings = [item for item in actions if item.get("action") == "GRAPH_WARNING"]
        common = {
            "called": True,
            "fallback_reason": error or (authorization if ir.get("candidate") and supports is None else ""),
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
            "terminal_answer_contract_version": self.TERMINAL_ANSWER_CONTRACT_VERSION,
        }
        if supports is not None:
            candidate = active_ir["candidate"]
            answer = self._terminal_render_seed_answer(
                question, active_ir, candidate["answer"], supports[0]
            )
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
