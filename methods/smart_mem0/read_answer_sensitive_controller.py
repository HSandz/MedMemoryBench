"""Answer-sensitive semantic controller policy for SmartMem0 READ.

This layer changes only what LLM #1 is asked to normalize.  It keeps the same
Requirement-vNext schema, deterministic compiler, retrieval executor, ProofContext and
two-call budget.  The policy is dataset-, mode- and language-neutral: it asks for
participant-memory variables whose values can change the answer, rather than treating the
requested conclusion itself as a memory lookup.
"""

import json

from .read_answer_or_plan_contract import MINIMAL_CONTROLLER_SCHEMA


ANSWER_SENSITIVE_CONTROLLER_POLICY = """
You are the single semantic controller for an evidence-grounded memory system.

Your only job is to answer: WHAT participant-specific evidence must be retrieved before
QUESTION can be answered correctly? Understand meaning in any language. Never route by
language-specific keywords. Inspect exactly the Top-3 SEEDS; they are planning hints, not proof.

Return only requested_projection, the smallest sufficient participant-memory requirements,
answer-relevant bridges, and an optional one-seed complete candidate. Do NOT emit route,
query class, answer_mode, requires_inference, difficulty, budget, retrieval operations,
proof status, or final-context decisions. Runtime derives them.

EVIDENCE REQUIREMENTS
- Usually 1-3, never more than 4. Every requirement must be MEMORY-VALUED and ANSWER-SENSITIVE:
  participant memory can supply a concrete value, and changing that value could change the answer.
- grounding_kind=QUESTION only when answer_obligation is the shortest useful contiguous span
  from QUESTION that itself names participant evidence. A requested decision/conclusion such as
  whether to act, recommend, diagnose, explain, continue, stop, increase, reduce, or choose is the
  ANSWER, not a memory requirement, unless QUESTION explicitly asks what prior guidance said.
- grounding_kind=DERIVED is an unmentioned participant-memory variable needed to answer. Its
  answer_obligation is a concise variable name, never an invented value or general-domain rule.
- evidence_family is selector-neutral semantic recall. selector owns temporal selection.
  constraints narrow recall but are never proof.
- General mechanisms and domain rules are not participant-memory requirements.

DECISION-CHANGING DECOMPOSITION
When QUESTION asks for an action, recommendation, suitability, monitoring decision, or choice,
retrieve the minimal participant facts that could change that decision rather than a generic
"advice" or "assessment" family. Depending on the question these may include: the focal current
state or trajectory; prior constraints/policies/contraindications; current regimen or prior
action/outcome; and preferences only when they change the choice. Usually 2-3 are enough.
If applying a general-domain rule is required after those facts are grounded, connect the
answer-relevant requirement(s) to ANSWER with INFER.

CAUSAL / EXPLANATORY DECOMPOSITION
When QUESTION asks whether one participant exposure/event/state could explain another, represent
the participant cause-side and effect-side as separate requirements and connect them with
POSSIBLE_CAUSE unless an explicit stored causal relation itself is required, in which case use
CAUSES. Do not collapse such a question into one generic clinical_assessment/explanation node.
For longitudinal change, retrieve a prior baseline as a separate requirement when the comparison
with current state can change the answer.

CANDIDATE SET
CANDIDATE SET contains alternative propositions, not memory facts. Derive the smallest shared or
discriminative participant evidence variables needed to distinguish the alternatives. Do not
create a generic recommendation requirement and do not create one requirement per option merely
because the options are visible. Do not classify candidate-memory stance or choose labels.

BRIDGES
- COMPARE: compare grounded obligations.
- CAUSES: require an explicit stored participant causal relation.
- POSSIBLE_CAUSE: grounded participant endpoints plus authorized general-domain knowledge.
- DEPENDS_ON: one obligation depends on another without asserting causality.
- TEMPORAL_ORDER: order grounded endpoints; BEFORE/AFTER/OVERLAPS never implies causality.
- INFER: authorize a general-domain bridge only after referenced participant evidence is grounded.
- CURRENT: require current-state resolution for one obligation.
- VERIFY_SOURCE: require exact linked source evidence.
Use goal only for the remaining reasoning obligation. Every DERIVED requirement must participate
in an answer-relevant bridge. Seeds may suggest WHAT extra evidence is needed but never pre-fill
its value.

DIRECT CANDIDATE
candidate is allowed only when exactly one QUESTION requirement is sufficient and exactly one
of $seed0, $seed1, or $seed2 already contains the complete answer. Do not emit candidate for
CandidateSet selection, advice/action decisions, comparison, source verification, world-knowledge
inference, cross-memory synthesis, temporal extremum/range selection, or multi-step reasoning.
Answer length never controls this decision.
"""


class ReadAnswerSensitiveControllerMixin:
    """Use the stricter Requirement-vNext semantic policy without another LLM call."""

    CONTROLLER_SCHEMA_VERSION = "requirement-vnext-2"
    CONTROLLER_MAX_OUTPUT_TOKENS = 512

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
        prompt = (
            ANSWER_SENSITIVE_CONTROLLER_POLICY
            + "\n"
            + MINIMAL_CONTROLLER_SCHEMA.format(
                question=question,
                candidate_set=json.dumps(propositions, ensure_ascii=False),
                hints=json.dumps(hints, ensure_ascii=False),
                seeds=json.dumps(
                    self._controller_seed_payload(seeds), ensure_ascii=False
                ),
            )
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
            "controller_policy": "answer_sensitive_memory_variables",
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
