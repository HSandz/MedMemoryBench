"""Regressions for terminal rendering and read answerability ownership."""

from methods.smart_mem0.agent import SmartMem0Agent
from methods.smart_mem0.contracts import QueryFrame
from methods.smart_mem0.read_answerability_contract import (
    ReadAnswerabilityContractMixin,
)
from methods.smart_mem0.read_certificate_contract import (
    ReadCertificateContractMixin,
)
from methods.smart_mem0.read_terminal_answer_contract import (
    ReadTerminalAnswerContractMixin,
)


def _agent():
    agent = object.__new__(SmartMem0Agent)
    agent._belief_status = {}
    agent._state_heads = {}
    agent._memories = []
    agent._relations = []
    agent.subject_aliases = {"patient": "primary_user"}
    agent._last_option_probe_coverage = {}
    agent._last_requirement_normalization_actions = []
    return agent


def _memory(
    memory_id="m1",
    *,
    claim,
    value,
    semantic_role="OBSERVATION",
    scope="general",
    state_key="",
    object_anchor="",
    assertion_mode="DIRECT",
    evidence_ids=None,
    origin_memory_id="",
    kind="FACT",
):
    return {
        "id": memory_id,
        "claim": claim,
        "value": value,
        "verbatim_value": "",
        "kind": kind,
        "semantic_role": semantic_role,
        "memory_tier": "HOT",
        "subject_id": "primary_user",
        "subject": "patient",
        "scope": scope,
        "state_key": state_key,
        "object_anchor": object_anchor,
        "entities": [object_anchor] if object_anchor else [],
        "scope_entities": [],
        "planning_tags": [],
        "assertion_mode": assertion_mode,
        "origin_memory_id": origin_memory_id,
        "stance": "AFFIRM",
        "event_time": "2024-02-01",
        "document_time": "2024-02-01",
        "origin_document_time": "2024-02-01",
        "evidence_ids": ["ev1"] if evidence_ids is None else list(evidence_ids),
        "_status": "active",
    }


def _direct_slot(target, *, proof_spec=None, history=False):
    return {
        "id": "r1",
        "type": "DIRECT",
        "evidence_role": "REQUIREMENT",
        "grounding_kind": "QUESTION",
        "subject_id": "primary_user",
        "subject": "primary_user",
        "target_surface": target,
        "retrieval_target": target,
        "description": target,
        "resolved_keys": [],
        "proof_spec": proof_spec or {"status": "UNSPECIFIED"},
        "history": history,
        "degraded": False,
        "required_fields": [],
    }


def _single_slot_plan(slot, answer_type):
    return {
        "required_slots": [slot],
        "semantic_ir": {
            "answer_type": answer_type,
            "requirements": [
                {
                    "id": "r1",
                    "grounding_kind": "QUESTION",
                    "target": slot["target_surface"],
                }
            ],
        },
        "visible_options": {},
        "need_evidence": False,
        "query_spec": {"world_knowledge_bridge_allowed": False},
        "semantic_relations": [],
    }


def test_runtime_mro_keeps_answerability_and_terminal_as_small_front_contracts():
    assert SmartMem0Agent.__mro__[1].__name__ == "ReadAnswerabilityContractMixin"
    assert SmartMem0Agent.__mro__[2].__name__ == "ReadTerminalAnswerContractMixin"


def test_contract_ownership_has_no_shadow_controller_or_duplicate_closure():
    assert "_semantic_controller" not in ReadTerminalAnswerContractMixin.__dict__
    assert "_rc_normalize_ir" not in ReadAnswerabilityContractMixin.__dict__
    assert "_post_retrieval_closure" not in ReadAnswerabilityContractMixin.__dict__
    assert "_prepare_requirement_context_state" not in ReadCertificateContractMixin.__dict__
    assert "_post_retrieval_closure" not in ReadCertificateContractMixin.__dict__


def test_complete_text_candidate_is_not_limited_to_scalar_answer():
    agent = _agent()
    seed = _memory(
        claim=(
            "The patient's weight loss had stopped over the past month and the "
            "patient's weight was stable after the earlier decline."
        ),
        value="stable",
        semantic_role="MEASUREMENT",
        scope="measurement",
    )
    agent._memories = [seed]
    question = "What was the patient's weight status in February?"
    ir = agent._rc_normalize_ir(
        {
            "answer_type": "TEXT",
            "subject_span": "patient",
            "requirements": [
                {
                    "id": "r1",
                    "grounding_kind": "QUESTION",
                    "focus_span": "weight status in February",
                    "target": "February weight status",
                    "retrieval_hint": "weight status",
                }
            ],
            "candidate": {
                "answer": (
                    "The patient's weight loss had stopped over the past month and the "
                    "patient's weight was stable after the earlier decline."
                ),
                "support_ref": "$seed0",
            },
        },
        question,
        QueryFrame(),
    )
    projected = agent._aop_direct_projection(ir, question)
    supports, reason = agent._authorize_controller_answer(
        projected, [seed], QueryFrame()
    )
    assert reason == "AUTHORIZED_COMPLETE_PROPOSITION"
    assert [memory["id"] for memory in supports] == ["m1"]


def test_short_status_candidate_renders_full_grounded_proposition():
    agent = _agent()
    seed = _memory(
        claim=(
            "The patient's weight loss has stopped and the patient's "
            "weight is stabilizing."
        ),
        value="stabilized",
        semantic_role="MEASUREMENT",
        scope="measurement",
    )
    answer = agent._terminal_render_seed_answer(
        "What is the patient's current weight status?",
        {"answer_type": "TEXT"},
        "stabilized",
        seed,
    )
    assert answer == seed["claim"]


def test_retrieval_presence_uses_found_empty_not_certificate_status():
    agent = _agent()
    memory = _memory(
        claim="The patient's current metformin regimen is metformin 1500 mg per day.",
        value="metformin 1500 mg/day",
        evidence_ids=[],
    )
    slot = _direct_slot("current metformin regimen")
    plan = {"required_slots": [slot], "semantic_relations": []}
    status, relations, complete = agent._retrieval_status(
        plan, {"r1": ["m1"]}, [memory], []
    )
    assert status == {"r1": "FOUND"}
    assert relations == {}
    assert complete is True
    assert agent._last_answerability_state == "SYNTHESIZE"


def test_uncertified_relevance_cannot_open_direct_b():
    agent = _agent()
    memory = _memory(
        claim=(
            "The patient has a documented cefuroxime allergy and was explicitly "
            "instructed to avoid cefuroxime."
        ),
        value="cefuroxime",
        semantic_role="SAFETY_CONSTRAINT",
        scope="allergy",
        object_anchor="cefuroxime",
    )
    agent._memories = [memory]
    agent._active_answer_question = (
        "Which antibiotic was the patient instructed to avoid?"
    )
    slot = _direct_slot("antibiotic instructed to avoid", history=True)
    closure = agent._post_retrieval_closure(
        _single_slot_plan(slot, "ENTITY"),
        [memory],
        QueryFrame(),
        [],
    )
    assert closure is None
    assert (
        agent._last_terminal_closure_diagnostic["reason"]
        == "NO_VALID_STRUCTURED_CERTIFICATE"
    )


def test_strict_structured_certificate_can_open_direct_b():
    agent = _agent()
    memory = _memory(
        claim="The patient was explicitly instructed to avoid cefuroxime.",
        value="cefuroxime",
        semantic_role="SAFETY_CONSTRAINT",
        scope="medication_safety",
        state_key="avoid_antibiotic",
        object_anchor="cefuroxime",
    )
    agent._memories = [memory]
    agent._active_answer_question = (
        "Which antibiotic was the patient instructed to avoid?"
    )
    proof_spec = {
        "status": "VALID",
        "match": {
            "subject_id": "primary_user",
            "scope": "medication_safety",
            "state_key": "avoid_antibiotic",
            "object_anchor": "cefuroxime",
            "stance": "AFFIRM",
        },
        "answer_field": "object_anchor",
    }
    slot = _direct_slot(
        "antibiotic instructed to avoid",
        proof_spec=proof_spec,
        history=True,
    )
    closure = agent._post_retrieval_closure(
        _single_slot_plan(slot, "ENTITY"),
        [memory],
        QueryFrame(),
        [],
    )
    assert closure is not None
    assert closure["answer"] == "cefuroxime"
    assert closure["support_ids"] == ["m1"]
    assert closure["reason"] == "STRUCTURED_TERMINAL_CERTIFICATE"


def test_started_date_query_compiles_to_earliest_event_time():
    agent = _agent()
    constraint = agent._rq_repair_time_constraint(
        {"axis": "", "relation": "", "anchor": "", "end": ""},
        "When did the patient start taking metformin 1500 mg/day as prescribed?",
        "DATE",
    )
    assert constraint["axis"] == "event_time"
    assert constraint["relation"] == "EARLIEST"


def test_latest_value_query_keeps_temporal_selector_even_for_value_answer():
    agent = _agent()
    constraint = agent._rq_repair_time_constraint(
        {"axis": "", "relation": "", "anchor": "", "end": ""},
        "What was the latest HbA1c result?",
        "VALUE",
    )
    assert constraint["axis"] == "event_time"
    assert constraint["relation"] == "LATEST"


def test_orphan_derived_requirement_is_dropped_at_requirement_normalization():
    agent = _agent()
    question = "What are the patient's current symptoms?"
    ir = agent._rc_normalize_ir(
        {
            "answer_type": "TEXT",
            "requirements": [
                {
                    "id": "r1",
                    "grounding_kind": "QUESTION",
                    "focus_span": "current symptoms",
                    "target": "current symptoms",
                    "retrieval_hint": "current symptoms",
                },
                {
                    "id": "r2",
                    "grounding_kind": "DERIVED",
                    "focus_span": "",
                    "target": "unconnected background fact",
                    "retrieval_hint": "background fact",
                },
            ],
            "relations": [],
            "candidate": None,
        },
        question,
        QueryFrame(),
    )
    assert [requirement["id"] for requirement in ir["requirements"]] == ["r1"]
    assert ir["graph_validation"]["orphan_requirements"] == []
    assert any(
        action.get("action") == "DROP_ORPHAN_DERIVED"
        for action in ir["normalization_actions"]
    )


def test_dataset_query_type_cannot_change_method_behavior():
    agent = _agent()
    assert (
        agent._compact_reasoning_output_instruction(
            "multi_hop_clinical_deduction"
        )
        == ""
    )
    assert (
        agent._compact_reasoning_output_instruction("entity_exact_match")
        == ""
    )
