"""Requirement-node policy for the minimal semantic read IR.

A requirement is a participant-specific evidence obligation that is independently
retrievable and necessary for grounding the answer. General-domain mechanisms are
bridges between requirement nodes, not synthetic memory requirements.
"""

import json
from typing import Any, Dict

from .read_controller import SEMANTIC_IR_POLICY


REQUIREMENT_POLICY = """
REQUIREMENT CONTRACT:
Create at most four requirements. A node is a requirement only when ALL are true:
1. PARTICIPANT-SPECIFIC: it is a fact, state, event, measurement, decision, preference, exposure, symptom, or other evidence about the participant rather than a general-domain rule or mechanism.
2. INDEPENDENTLY RETRIEVABLE: it is meaningful to search the participant memory for this evidence as one atomic obligation.
3. GROUNDING-NECESSARY: removing the node would remove participant-specific evidence needed to answer the QUESTION. This is the Requirement Necessity Invariant.

Use the smallest participant-specific evidence skeleton. Multi-hop reasoning does NOT imply one requirement per reasoning hop. General mechanisms such as biochemical, physiological, causal, or domain rules belong in relation bridge_goal, not in requirements.

Each requirement has grounding_kind:
- QUESTION: directly requested or explicitly constrained by QUESTION. focus_span MUST be the shortest useful contiguous span copied from QUESTION. evidence_target must be empty.
- INTERMEDIATE: participant-specific intermediate evidence needed to ground a multi-step answer even though that evidence is not directly named in QUESTION. focus_span must be empty. evidence_target is a concise retrieval concept, not an asserted answer. Do not put a general mechanism in evidence_target. Do not create an INTERMEDIATE merely because it would make a plausible explanation more detailed.

Examples:
- late-night takeout + milk coffee -> participant morning hyperglycemia -> morning blurry vision: the participant observations may be requirements; HPA activation, cortisol, hepatic glucose output are bridge mechanism, not requirements unless the QUESTION explicitly asks to retrieve a participant record of them.
- drug exposure -> general pharmacologic mechanism -> symptom: drug exposure and symptom may be two requirements; the pharmacologic mechanism is a bridge.

RELATION BRIDGES:
For POSSIBLE_CAUSE and INFER, include bridge_goal: a short instruction describing what connection the answer model must explain between grounded endpoints. bridge_goal must not pre-fill an unstored mechanism as participant history. It may ask for the mechanism at answer time using general-domain knowledge.
For CAUSES, bridge_goal may describe the explicit stored causal relation to verify, but CAUSES still requires stored relation proof.
"""

REQUIREMENT_SCHEMA = """Return JSON only:
{{"answer_type":"ENTITY|VALUE|DATE|RELATIVE_TIME|OPTION_SET|TEXT","subject_span":"exact contiguous subject span from QUESTION or empty","requirements":[{{"id":"r1","grounding_kind":"QUESTION|INTERMEDIATE","focus_span":"exact QUESTION span for QUESTION nodes, otherwise empty","evidence_target":"concise participant evidence concept for INTERMEDIATE nodes, otherwise empty","retrieval_hint":"soft semantic retrieval description","time_constraint":{{"axis":"event_time|document_time|origin_document_time|effective_event_time|","relation":"LOCATE|EXACT|EARLIEST|LATEST|BEFORE|AFTER|BETWEEN|","anchor":"","end":""}}}}],"relations":[{{"type":"COMPARE|CAUSES|POSSIBLE_CAUSE|DEPENDS_ON|TEMPORAL_ORDER|INFER|CURRENT|VERIFY_SOURCE","from":"r1","to":"r2|ANSWER|","relation":"BEFORE|AFTER|OVERLAPS|","bridge_goal":"short reasoning obligation or empty"}}],"candidate":null}}
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
    """Enforce grounded requirements and keep general reasoning on graph edges."""

    def _rc_normalize_ir(self, parsed: Dict[str, Any], question: str, frame: Any):
        ir = super()._rc_normalize_ir(parsed, question, frame)
        raw_requirements = {
            str(item.get("id") or ""): item
            for item in (parsed.get("requirements") or [])
            if isinstance(item, dict)
        }

        normalized = []
        kept_ids = set()
        for requirement in ir.get("requirements") or []:
            item = dict(requirement)
            raw = raw_requirements.get(str(item.get("id") or ""), {})
            kind = str(raw.get("grounding_kind") or "QUESTION").upper()
            if kind not in {"QUESTION", "INTERMEDIATE"}:
                kind = "QUESTION"
            if kind == "INTERMEDIATE":
                target = " ".join(str(raw.get("evidence_target") or "").split())[:180]
                # An intermediate without its own atomic retrieval concept is not
                # independently retrievable, so it fails the requirement contract.
                if not target:
                    continue
                item["focus_span"] = ""
                item["evidence_target"] = target
            else:
                item["evidence_target"] = ""
            item["grounding_kind"] = kind
            normalized.append(item)
            kept_ids.add(str(item.get("id") or ""))

        # The base normalizer always supplies at least one QUESTION requirement.
        # If every malformed intermediate was removed, keep that fallback.
        if not normalized:
            fallback = dict((ir.get("requirements") or [{}])[0])
            fallback["grounding_kind"] = "QUESTION"
            fallback["evidence_target"] = ""
            normalized = [fallback]
            kept_ids = {str(fallback.get("id") or "r1")}

        raw_relations = [
            item for item in (parsed.get("relations") or []) if isinstance(item, dict)
        ]
        relations = []
        for relation in ir.get("relations") or []:
            source = str(relation.get("from") or "")
            target = str(relation.get("to") or "")
            if source not in kept_ids or (target and target != "ANSWER" and target not in kept_ids):
                continue
            item = dict(relation)
            raw = next(
                (
                    candidate
                    for candidate in raw_relations
                    if str(candidate.get("type") or "").upper() == str(item.get("type") or "").upper()
                    and str(candidate.get("from") or "") == source
                    and str(candidate.get("to") or "") == target
                ),
                {},
            )
            if str(item.get("type") or "").upper() in {"POSSIBLE_CAUSE", "INFER", "CAUSES"}:
                goal = " ".join(str(raw.get("bridge_goal") or "").split())[:240]
                if goal:
                    item["bridge_goal"] = goal
            relations.append(item)

        ir["requirements"] = normalized[:4]
        valid_ids = {str(item.get("id") or "") for item in ir["requirements"]}
        ir["relations"] = [
            relation
            for relation in relations[:8]
            if str(relation.get("from") or "") in valid_ids
            and (
                str(relation.get("to") or "") in {"", "ANSWER"}
                or str(relation.get("to") or "") in valid_ids
            )
        ]
        return ir

    def _requirement_slot(self, requirement, ir, compiled_mode):
        slot = super()._requirement_slot(requirement, ir, compiled_mode)
        kind = str(requirement.get("grounding_kind") or "QUESTION").upper()
        slot["grounding_kind"] = kind
        if kind == "INTERMEDIATE":
            target = str(requirement.get("evidence_target") or "").strip()
            slot["target_surface"] = target
            slot["description"] = (
                str(requirement.get("retrieval_hint") or "").strip()
                or target
                or "participant intermediate evidence"
            )
            slot["resolved_keys"] = self._rc_resolve_target_keys(
                target, str(slot.get("subject_id") or "")
            )
        return slot

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
        prompt = (
            SEMANTIC_IR_POLICY
            + "\n"
            + REQUIREMENT_POLICY
            + "\n"
            + REQUIREMENT_SCHEMA.format(
                question=question,
                options=json.dumps(options, ensure_ascii=False),
                hints=json.dumps(hints, ensure_ascii=False),
                seeds=json.dumps(self._rc_seed_payload(seeds), ensure_ascii=False),
            )
        )
        try:
            response = self._llm_client.chat(
                [{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=700,
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
