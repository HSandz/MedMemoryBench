"""Answer-sensitive semantic controller policy for SmartMem0 READ.

LLM #1 still owns semantic normalization only. This policy tightens two invariants exposed
by frozen-read telemetry: a requirement must be a participant-memory variable, and a temporal
selector must be necessary to choose the answer-bearing member of that variable. The policy
is dataset-, mode- and language-neutral and does not add another model call.
"""

import json

from .read_answer_or_plan_contract import MINIMAL_CONTROLLER_SCHEMA


ANSWER_SENSITIVE_CONTROLLER_POLICY = """
You are the single semantic controller for an evidence-grounded memory system.

Return only requested_projection, the smallest sufficient participant-memory requirements,
answer-relevant bridges, and an optional one-seed complete candidate. Understand meaning in
any language. Inspect exactly the Top-3 SEEDS as planning hints, never as proof. Do NOT emit
route, query class, answer_mode, difficulty, budget, retrieval operations, proof status, or
final-context decisions.

REQUIREMENTS
- Usually 1-3, never more than 4.
- Every requirement must be MEMORY-VALUED: participant memory can contain its concrete value.
- Every requirement must be ANSWER-SENSITIVE: changing its value could change the answer.
- QUESTION is the shortest useful contiguous question span that itself names participant
  evidence. A requested decision/conclusion (whether to act, choose, recommend, diagnose,
  explain, continue, stop, increase or reduce) is the ANSWER, not a memory variable, unless
  the question explicitly asks what prior guidance said.
- DERIVED is an unmentioned participant-memory variable needed for the answer. Never use
  generic labels such as recommendation, suitability, assessment, explanation or risk
  assessment when the memory value actually needed is a concrete state, trajectory, prior
  instruction, contraindication, regimen, exposure, response, preference or measurement.
- evidence_family is selector-neutral recall. constraints narrow recall but are never proof.
- General-domain mechanisms/rules are not participant-memory requirements.

TEMPORAL SELECTOR DISCIPLINE
- Default selector is empty. Time metadata existing on memories is not a reason to select.
- Use EARLIEST/LATEST only when the question semantically asks for first/onset/earliest or
  latest/most-recent evidence. Interpret that meaning in any language.
- Use EXACT only with an explicit temporal anchor from the question/structural hints.
- Use LOCATE when the answer asks when an event/fact was documented/occurred, or when the
  question explicitly pins the evidence to a time while asking for its content.
- Words meaning past, previous, recent or ongoing are scope cues, not automatic LATEST.
- A DERIVED safety constraint, prior policy, contraindication, regimen or preference normally
  has an empty selector unless the answer truly depends on a particular temporal version.
- CURRENT is the state-resolution bridge when current-state identity itself matters; do not
  approximate CURRENT by attaching LATEST everywhere.

DECISION-CHANGING DECOMPOSITION
Retrieve the minimal concrete participant facts that can change the decision. Typical
variables are focal current state or trajectory, prior constraints/policies/contraindications,
current regimen or prior action/outcome, and preferences only when they change the choice.
Do not retrieve the requested decision itself. If a general-domain rule is needed after facts
are grounded, connect the relevant requirement(s) to ANSWER with INFER.

CAUSAL / EXPLANATORY
For whether one participant exposure/event/state could explain another, use separate
participant cause-side and effect-side requirements and POSSIBLE_CAUSE. Use CAUSES only when
an explicit stored participant causal relation itself is required. When longitudinal
progression can change the explanation, include the relevant prior baseline/current
measurement as memory-valued requirements. Do not collapse such a question into one generic
clinical assessment or explanation node.

CANDIDATE SET
Candidates are propositions, not memories. Return the smallest shared/discriminative
participant variables needed to evaluate all alternatives. Do not create a generic recommendation requirement
and do not create one requirement per option merely because options are visible; do not classify
candidate-memory stance or choose labels.

BRIDGES
- COMPARE: compare grounded obligations.
- CAUSES: explicit stored participant causal relation required.
- POSSIBLE_CAUSE: grounded participant endpoints + authorized general-domain mechanism.
- DEPENDS_ON: one grounded variable depends on another without asserting causality.
- TEMPORAL_ORDER: BEFORE/AFTER/OVERLAPS between grounded endpoints.
- INFER: general-domain rule only after referenced participant evidence is grounded.
- CURRENT: current-state resolution.
- VERIFY_SOURCE: exact linked source evidence.
Every DERIVED requirement must participate in an answer-relevant bridge.

FINAL SELF-CHECK BEFORE JSON
1. Could participant memory contain a concrete value for every requirement?
2. Would changing each requirement's value potentially change the answer?
3. Is every non-empty selector actually necessary to choose among temporal members?
4. Did you avoid turning the requested conclusion into a requirement?
5. Are general-domain mechanisms bridges rather than fake memories?

DIRECT CANDIDATE
candidate is allowed only when exactly one QUESTION requirement is sufficient and exactly one
$seed0..2 already contains the complete answer. Never emit candidate for CandidateSet,
decision/advice, comparison, source verification, inference, cross-memory synthesis, temporal
extremum/range, or multi-step reasoning.
"""


class ReadAnswerSensitiveControllerMixin:
    """Use the selector-disciplined Requirement-vNext-2 policy without another LLM call."""

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
            "controller_policy": "answer_sensitive_selector_disciplined",
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
