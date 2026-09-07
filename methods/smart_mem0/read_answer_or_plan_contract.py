"""Requirement-first controller contract for SmartMem0 READ.

LLM #1 owns semantic normalization only: requested projection, evidence obligations,
selectors, and reasoning bridges. Runtime code owns route derivation, retrieval execution,
CandidateSet recall, proof, answerability, and final context.
"""

import json
import re
from copy import deepcopy
from typing import Any, Dict, List

from .contracts import VALID_TEMPORAL_AXES


MINIMAL_CONTROLLER_POLICY = """
You are the single semantic controller for an evidence-grounded memory system.

Your only job is to answer: WHAT participant-specific evidence is required to answer
QUESTION? Understand meaning in any language. Do not route by language-specific keywords.

Inspect exactly the Top-3 SEEDS. Seeds are compact planning hints, not proof. A seed may
justify one optional direct candidate only when that single seed already contains the
complete participant-specific answer to one atomic QUESTION obligation.

Return only:
1. requested_projection: the semantic shape of the requested answer.
2. requirements: the smallest sufficient set of participant-memory obligations.
3. bridges: semantic relations needed after those obligations are grounded.
4. candidate: optional one-seed complete answer.

Do NOT emit route, query class, answer_mode, requires_inference, difficulty, budget,
retrieval operations, proof status, or final-context decisions. Runtime derives them.

REQUIREMENTS
- Usually 1-3, never more than 4.
- grounding_kind=QUESTION when answer_obligation is the shortest useful contiguous span
  copied exactly from QUESTION.
- grounding_kind=DERIVED only for an additional participant-memory variable whose value can
  change the answer. Its answer_obligation is a concise variable name, not invented facts.
- evidence_family is a selector-neutral semantic family used for recall. It must not contain
  EARLIEST/LATEST or another temporal selector already represented in selector.
- selector contains temporal selection only when time changes which evidence value is
  required. Choose event_time, document_time, origin_document_time, or
  effective_event_time by meaning, not by language.
- constraints are short semantic restrictions that narrow this obligation. They are
  retrieval constraints, never proof by themselves.
- General-domain mechanisms/rules are not participant-memory requirements.

BRIDGES
- COMPARE: compare grounded obligations.
- CAUSES: require an explicit stored participant causal relation.
- POSSIBLE_CAUSE: grounded participant endpoints plus authorized general-domain knowledge.
- DEPENDS_ON: one obligation depends on another without asserting causality.
- TEMPORAL_ORDER: order grounded endpoints; relation is BEFORE, AFTER, or OVERLAPS.
- INFER: authorize a general-domain bridge only after referenced participant evidence is
  grounded.
- CURRENT: require current-state resolution for one obligation.
- VERIFY_SOURCE: require exact linked source evidence.
Use goal only to describe the remaining reasoning obligation; never turn a mechanism into
a fake memory requirement.

CANDIDATE SET
If CANDIDATE SET is supplied, it is structural runtime input: propositions to evaluate,
not memory facts and not extra seeds. Use it only to determine the shared participant
evidence obligations needed to discriminate among candidates. Do not classify
candidate-to-memory stance, do not attach relation confidence, and do not choose final
candidate labels. LLM #2 owns final semantic evaluation after retrieval.

DIRECT CANDIDATE
candidate is allowed only when exactly one QUESTION requirement is sufficient and exactly
one of $seed0, $seed1, or $seed2 already contains the complete answer. Do not emit candidate
for CandidateSet selection, advice/action decisions, comparison, source verification,
world-knowledge inference, cross-memory synthesis, temporal extremum/range selection, or
multi-step reasoning. Answer length never controls this decision.
"""

MINIMAL_CONTROLLER_SCHEMA = """
Return JSON only:
{{
  "requested_projection": "ENTITY|VALUE|DATE|RELATIVE_TIME|OPTION_SET|TEXT",
  "subject_span": "exact contiguous subject span from QUESTION or empty",
  "requirements": [
    {{
      "id": "r1",
      "grounding_kind": "QUESTION|DERIVED",
      "answer_obligation": "exact QUESTION span for QUESTION nodes; concise variable for DERIVED",
      "evidence_family": "selector-neutral participant-memory family",
      "selector": {{
        "axis": "event_time|document_time|origin_document_time|effective_event_time|",
        "relation": "LOCATE|EXACT|EARLIEST|LATEST|BEFORE|AFTER|BETWEEN|",
        "anchor": "",
        "end": ""
      }},
      "constraints": []
    }}
  ],
  "bridges": [
    {{
      "type": "COMPARE|CAUSES|POSSIBLE_CAUSE|DEPENDS_ON|TEMPORAL_ORDER|INFER|CURRENT|VERIFY_SOURCE",
      "from": "r1",
      "to": "r2|ANSWER|",
      "relation": "BEFORE|AFTER|OVERLAPS|",
      "goal": ""
    }}
  ],
  "candidate": null
}}
When candidate exists:
{{"candidate":{{"answer":"complete answer already present in one seed","support_ref":"$seed0"}}}}

QUESTION:
{question}
CANDIDATE SET:
{candidate_set}
STRUCTURAL HINTS (constraints only; never evidence):
{hints}
TOP-3 SEEDS:
{seeds}
"""

# Backwards import name used by offline contract tests. This is now the compact
# Requirement-vNext policy rather than the old answer-mode / proposition-verdict policy.
ANSWER_OR_PLAN_PRIORITY = MINIMAL_CONTROLLER_POLICY

_HARD_DIRECT_RELATIONS = frozenset(
    {
        "COMPARE",
        "CAUSES",
        "POSSIBLE_CAUSE",
        "TEMPORAL_ORDER",
        "INFER",
        "VERIFY_SOURCE",
    }
)


class ReadAnswerOrPlanContractMixin:
    """Compile Requirement-vNext into the existing deterministic retrieval runtime."""

    CONTROLLER_SCHEMA_VERSION = "requirement-vnext-1"
    CONTROLLER_MAX_OUTPUT_TOKENS = 512

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
    def _aop_constraints(value: Any) -> List[str]:
        if not isinstance(value, list):
            return []
        output = []
        for item in value[:6]:
            text = " ".join(str(item or "").split()).strip()
            if text and text not in output:
                output.append(text[:160])
        return output

    @staticmethod
    def _aop_bridge_goal(raw: Dict[str, Any]) -> str:
        return " ".join(
            str(raw.get("goal") or raw.get("bridge_goal") or "").split()
        )[:220]

    def _controller_seed_payload(self, seeds):
        """Compact planning view; never serialize the full memory object to LLM #1."""
        output = []
        for index, memory in enumerate((seeds or [])[:3]):
            output.append(
                {
                    "ref": f"$seed{index}",
                    "claim": str(memory.get("claim") or "")[:220],
                    "kind": str(memory.get("kind") or "FACT"),
                    "value": str(self._memory_value(memory) or "")[:160],
                    "subject": str(
                        memory.get("subject_id") or memory.get("subject") or ""
                    )[:80],
                    "object": str(memory.get("object_anchor") or "")[:120],
                    "evidence_family": str(
                        memory.get("evidence_family")
                        or memory.get("state_key")
                        or memory.get("scope")
                        or ""
                    )[:120],
                    "event_time": memory.get("event_time") or "",
                    "document_time": memory.get("document_time") or "",
                }
            )
        return output

    # Keep the old helper name as an alias because other read contracts may call it.
    def _rc_seed_payload(self, seeds):
        return self._controller_seed_payload(seeds)

    def _aop_vnext_to_legacy(
        self, parsed: Dict[str, Any], question: str
    ) -> Dict[str, Any]:
        """Adapt the lean controller surface to the stable internal compiler contract."""
        parsed = parsed if isinstance(parsed, dict) else {}
        projection = str(
            parsed.get("requested_projection") or parsed.get("answer_type") or "TEXT"
        ).upper()

        raw_requirements = (
            parsed.get("requirements")
            if isinstance(parsed.get("requirements"), list)
            else []
        )
        requirements = []
        vnext_metadata: Dict[str, Dict[str, Any]] = {}
        for index, raw in enumerate(raw_requirements[:4]):
            if not isinstance(raw, dict):
                continue
            requirement_id = str(raw.get("id") or f"r{index + 1}")
            kind = str(raw.get("grounding_kind") or "QUESTION").upper()
            obligation = " ".join(
                str(
                    raw.get("answer_obligation")
                    or raw.get("focus_span")
                    or raw.get("target")
                    or ""
                ).split()
            )
            family = " ".join(
                str(
                    raw.get("evidence_family")
                    or raw.get("target")
                    or raw.get("retrieval_hint")
                    or obligation
                ).split()
            )[:320]
            selector = raw.get("selector")
            if not isinstance(selector, dict):
                selector = (
                    raw.get("time_constraint")
                    if isinstance(raw.get("time_constraint"), dict)
                    else {}
                )
            constraints = self._aop_constraints(raw.get("constraints"))
            focus_span = obligation if kind == "QUESTION" else ""
            requirements.append(
                {
                    "id": requirement_id,
                    "grounding_kind": kind,
                    "focus_span": focus_span,
                    # Compatibility fields consumed by Requirement-v2 compiler.
                    # Family is recall-only; proof is rebound to answer_obligation below.
                    "target": family or obligation,
                    "retrieval_hint": family or obligation,
                    "time_constraint": dict(selector),
                }
            )
            vnext_metadata[requirement_id] = {
                "answer_obligation": obligation,
                "evidence_family": family or obligation,
                "selector": dict(selector),
                "constraints": constraints,
            }

        raw_bridges = (
            parsed.get("bridges")
            if isinstance(parsed.get("bridges"), list)
            else parsed.get("relations")
            if isinstance(parsed.get("relations"), list)
            else []
        )
        bridges = []
        for raw in raw_bridges[:8]:
            if not isinstance(raw, dict):
                continue
            item = {
                "type": str(raw.get("type") or "").upper(),
                "from": str(raw.get("from") or ""),
                "to": str(raw.get("to") or ""),
            }
            relation = str(raw.get("relation") or "").upper()
            if relation:
                item["relation"] = relation
            goal = self._aop_bridge_goal(raw)
            if goal:
                item["bridge_goal"] = goal
            bridges.append(item)

        return {
            "answer_type": projection,
            "subject_span": str(parsed.get("subject_span") or ""),
            "requirements": requirements,
            "relations": bridges,
            "candidate": self._aop_raw_candidate(parsed),
            "_vnext_metadata": vnext_metadata,
        }

    def _aop_direct_surface_allowed(
        self, question: str, ir: Dict[str, Any]
    ) -> bool:
        del question
        if ir.get("visible_options") or ir.get("candidate_propositions"):
            return False
        requirements = ir.get("requirements") or []
        if (
            len(requirements) != 1
            or str(requirements[0].get("grounding_kind") or "QUESTION").upper()
            != "QUESTION"
        ):
            return False
        if str(ir.get("answer_type") or "").upper() == "OPTION_SET":
            return False
        relation_types = {
            str(relation.get("type") or "").upper()
            for relation in ir.get("relations") or []
        }
        return not bool(relation_types & _HARD_DIRECT_RELATIONS)

    def _rc_normalize_ir(self, parsed: Dict[str, Any], question: str, frame: Any):
        lean = self._aop_vnext_to_legacy(parsed, question)
        metadata = lean.pop("_vnext_metadata", {})
        ir = super()._rc_normalize_ir(lean, question, frame)

        # Rebind the public/proof obligation to the question-owned surface while
        # keeping evidence_family selector-neutral and retrieval-only.
        for requirement in ir.get("requirements") or []:
            meta = metadata.get(requirement.get("id"), {})
            obligation = str(
                meta.get("answer_obligation")
                or requirement.get("focus_span")
                or requirement.get("target")
                or ""
            ).strip()
            family = str(
                meta.get("evidence_family")
                or requirement.get("target")
                or requirement.get("retrieval_hint")
                or obligation
            ).strip()
            requirement["answer_obligation"] = obligation
            requirement["evidence_family"] = family
            requirement["selector"] = dict(
                requirement.get("time_constraint") or meta.get("selector") or {}
            )
            requirement["constraints"] = list(meta.get("constraints") or [])

        raw_candidate = self._aop_raw_candidate(parsed)
        ir["candidate"] = (
            raw_candidate
            if raw_candidate and self._aop_direct_surface_allowed(question, ir)
            else None
        )
        return ir

    def _rc_public_ir(self, ir):
        requirements = []
        for requirement in ir.get("requirements") or []:
            requirements.append(
                {
                    "id": requirement.get("id"),
                    "grounding_kind": requirement.get("grounding_kind", "QUESTION"),
                    "answer_obligation": requirement.get("answer_obligation")
                    or requirement.get("focus_span")
                    or "",
                    "evidence_family": requirement.get("evidence_family")
                    or requirement.get("target")
                    or "",
                    "selector": dict(
                        requirement.get("selector")
                        or requirement.get("time_constraint")
                        or {}
                    ),
                    "constraints": list(requirement.get("constraints") or []),
                }
            )
        bridges = []
        for relation in ir.get("relations") or []:
            item = {
                "type": relation.get("type"),
                "from": relation.get("from"),
                "to": relation.get("to"),
            }
            if relation.get("relation"):
                item["relation"] = relation.get("relation")
            if relation.get("bridge_goal"):
                item["goal"] = relation.get("bridge_goal")
            bridges.append(item)
        public = {
            "requested_projection": ir.get("answer_type") or "TEXT",
            "subject_span": ir.get("subject_span") or "",
            "requirements": requirements,
            "bridges": bridges,
            "candidate": dict(ir["candidate"]) if ir.get("candidate") else None,
        }
        if ir.get("candidate_propositions"):
            public["candidate_set"] = deepcopy(ir.get("candidate_propositions") or {})
        return public

    def _requirement_slot(self, requirement, ir, compiled_mode):
        slot = super()._requirement_slot(requirement, ir, compiled_mode)
        kind = str(requirement.get("grounding_kind") or "QUESTION").upper()
        obligation = str(
            requirement.get("answer_obligation")
            or requirement.get("focus_span")
            or ""
        ).strip()
        family = str(
            requirement.get("evidence_family")
            or requirement.get("target")
            or requirement.get("retrieval_hint")
            or ""
        ).strip()
        constraints = list(requirement.get("constraints") or [])
        selector = dict(
            requirement.get("selector")
            or requirement.get("time_constraint")
            or {}
        )

        slot["answer_obligation"] = obligation
        slot["evidence_family"] = family
        slot["selector"] = selector
        slot["constraints"] = constraints
        slot["retrieval_target"] = family or obligation
        slot["retrieval_hint"] = family
        if kind == "QUESTION" and obligation:
            # Family/hints/resolved keys can rank recall but never become proof.
            slot["target_surface"] = obligation
            slot["proof_anchor"] = obligation
        return slot

    def _aop_direct_projection(self, ir: Dict[str, Any], question: str):
        if not ir.get("candidate") or not self._aop_direct_surface_allowed(question, ir):
            return None
        projected = deepcopy(ir)
        projected["normalization_actions"] = [
            *(projected.get("normalization_actions") or []),
            {
                "action": "DIRECT_CANDIDATE_VALIDATED",
                "reason": "ONE_SEED_COMPLETE_ATOMIC_ANSWER",
                "kept_requirement_id": projected["requirements"][0].get("id"),
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
            or str(ir.get("answer_type") or "").upper() != "DATE"
        ):
            return None, reason
        if any(
            str(relation.get("type") or "").upper() != "CURRENT"
            for relation in ir.get("relations") or []
        ):
            return None, "RELATIONAL_QUERY_REQUIRES_RETRIEVAL"

        constraint = requirements[0].get("time_constraint") or {}
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
        """Compile recall projection; temporal selectors remain separate fields."""
        parts = [
            str(slot.get("evidence_family") or slot.get("retrieval_hint") or "").strip(),
            *[
                str(item).strip()
                for item in (slot.get("constraints") or [])
                if str(item).strip()
            ],
            str(slot.get("retrieval_target") or "").strip(),
        ]
        if not any(parts):
            parts.append(str(slot.get("answer_obligation") or "").strip())
        if not any(parts):
            parts.append(str(question or "").strip())
        unique = []
        for part in parts:
            if part and self._rc_text(part) not in {
                self._rc_text(value) for value in unique
            }:
                unique.append(part)
        return " | ".join(unique)

    def _rc_bundle_query(self, slots, question: str) -> str:
        parts = []
        for slot in slots:
            for value in (
                str(slot.get("evidence_family") or "").strip(),
                *[
                    str(item).strip()
                    for item in (slot.get("constraints") or [])
                    if str(item).strip()
                ],
                str(slot.get("retrieval_target") or "").strip(),
            ):
                if value and self._rc_text(value) not in {
                    self._rc_text(item) for item in parts
                }:
                    parts.append(value)
        if not parts and question:
            parts.append(str(question).strip())
        return " | ".join(parts)

    @staticmethod
    def _aop_derived_mode(ir: Dict[str, Any]) -> str:
        """Telemetry only. Never feed this label back into routing or retrieval."""
        if ir.get("candidate"):
            return "EXTRACT"
        if ir.get("candidate_propositions") or ir.get("visible_options"):
            return "SELECT"
        relation_types = {
            str(relation.get("type") or "").upper()
            for relation in ir.get("relations") or []
        }
        if "COMPARE" in relation_types:
            return "COMPARE"
        if relation_types.intersection({"INFER", "POSSIBLE_CAUSE"}):
            return "INFER"
        if "CAUSES" in relation_types:
            return "EXPLAIN"
        if len(ir.get("requirements") or []) > 1 or "DEPENDS_ON" in relation_types:
            return "SYNTHESIZE"
        return "RETRIEVE"

    def _controller_plan(self, ir, question, frame):
        plan = super()._controller_plan(ir, question, frame)
        spec = plan.setdefault("query_spec", {})
        spec["controller_schema_version"] = self.CONTROLLER_SCHEMA_VERSION
        spec["semantic_ir_version"] = self.CONTROLLER_SCHEMA_VERSION

        # CandidateSet is structural data. It is carried to retrieval/context without
        # LLM #1 support/contradict verdicts or confidence.
        propositions = deepcopy(ir.get("candidate_propositions") or {})
        if propositions:
            candidate_set = {
                "version": "candidate-set-v1",
                "candidates": propositions,
            }
            plan["candidate_set"] = deepcopy(candidate_set)
            plan["candidate_propositions"] = deepcopy(propositions)
            plan.setdefault("semantic_ir", {})["candidate_set"] = deepcopy(
                candidate_set
            )
        return plan

    def _reset_candidate_set_state(self):
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

    def _semantic_controller(self, question, seeds, frame, context_map=None):
        self._reset_candidate_set_state()
        reset_terminal = getattr(self, "_terminal_reset_state", None)
        if callable(reset_terminal):
            reset_terminal()
        self._active_controller_seeds = list((seeds or [])[:3])

        options = self._question_options(question) or {}
        propositions = {}
        normalize_propositions = getattr(self, "_normalize_candidate_propositions", None)
        if callable(normalize_propositions):
            propositions = normalize_propositions(options)
            if not propositions and isinstance(context_map, dict):
                propositions = normalize_propositions(
                    context_map.get("candidate_set")
                    or context_map.get("candidate_propositions")
                )
        self._last_candidate_propositions = dict(propositions)
        self._last_proposition_probe_coverage = {
            proposition_id: [] for proposition_id in propositions
        }

        hints = {
            "dates": list(getattr(frame, "dates", ()) or ()),
            "source_speaker": getattr(frame, "speaker_role", ""),
            "explicit_entities": list(getattr(frame, "entities", ()) or ()),
        }
        prompt = MINIMAL_CONTROLLER_POLICY + "\n" + MINIMAL_CONTROLLER_SCHEMA.format(
            question=question,
            candidate_set=json.dumps(propositions, ensure_ascii=False),
            hints=json.dumps(hints, ensure_ascii=False),
            seeds=json.dumps(
                self._controller_seed_payload(seeds), ensure_ascii=False
            ),
        )

        raw_ir = {}
        try:
            response = self._llm_client.chat(
                [{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=self.CONTROLLER_MAX_OUTPUT_TOKENS,
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

        # Keep the existing deterministic bounded candidate recall as a structural
        # accelerator, but build it AFTER LLM #1 and never expose its memories to
        # the controller. Phase 3 can fold this into the retrieval executor.
        proposition_pack = {}
        pack_builder = getattr(self, "_build_candidate_proposition_pack", None)
        if callable(pack_builder) and propositions:
            proposition_pack = pack_builder(
                propositions,
                frame=frame,
                seeds=(seeds or [])[:3],
                question=question,
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
            "requested_projection": active_ir.get("answer_type", "TEXT"),
            "answer_type": active_ir.get("answer_type", "TEXT"),
            "answer_mode": self._aop_derived_mode(active_ir),
            "answer_mode_source": "derived_telemetry_only",
            "requirement_count": len(active_ir.get("requirements") or []),
            "bridge_count": len(active_ir.get("relations") or []),
            "relation_count": len(active_ir.get("relations") or []),
            "semantic_ir": self._rc_public_ir(active_ir),
            "controller_raw_ir": raw_ir,
            "normalized_ir": self._rc_public_ir(active_ir),
            "normalization_status": active_ir.get("normalization_status", "VALID"),
            "graph_validation": active_ir.get("graph_validation") or {},
            "normalization_actions": actions,
            "graph_warnings": warnings,
            "candidate_authorization": authorization,
            "candidate_set_count": len(propositions),
            "candidate_proposition_count": len(propositions),
            "candidate_proposition_pack_size": len(
                proposition_pack.get("retrieval_views") or []
            ),
            "controller_schema_version": self.CONTROLLER_SCHEMA_VERSION,
        }

        if supports is not None:
            candidate = active_ir["candidate"]
            answer = candidate["answer"]
            renderer = getattr(self, "_terminal_render_seed_answer", None)
            if callable(renderer):
                answer = renderer(
                    question, active_ir, candidate["answer"], supports[0]
                )
            telemetry = dict(common)
            telemetry.update(
                {
                    "route": "DIRECT",
                    "route_source": "derived_from_authorized_candidate",
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
                "route_source": "derived_from_missing_authorized_candidate",
                "answer": "",
                "support_ref": "",
                "support_refs": [],
                "terminal_rendered": False,
            }
        )
        return None, plan, telemetry


    def prepare_batch_query(self, question, system_message=None, **kwargs):
        prepared = super().prepare_batch_query(
            question, system_message=system_message, **kwargs
        )
        prepared.setdefault("extra", {})["read_contract_version"] = (
            self.CONTROLLER_SCHEMA_VERSION
        )
        return prepared
