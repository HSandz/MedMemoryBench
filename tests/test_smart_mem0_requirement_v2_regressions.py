"""Focused regressions for Requirement-v2 semantic ownership."""

from methods.smart_mem0.agent import SmartMem0Agent
from methods.smart_mem0.contracts import QueryFrame
from methods.smart_mem0.read_requirement_contract import ReadRequirementContractMixin


def _agent():
    agent = object.__new__(SmartMem0Agent)
    agent._belief_status = {}
    agent._state_heads = {}
    agent._memories = []
    agent._relations = []
    agent.subject_aliases = {"patient": "primary_user"}
    return agent


def test_documented_date_repairs_axis_to_document_time():
    agent = _agent()
    question = (
        "When was it documented that the patient's GADA antibody "
        "was strongly positive?"
    )
    ir = agent._rc_normalize_ir(
        {
            "answer_type": "DATE",
            "requirements": [
                {
                    "id": "r1",
                    "grounding_kind": "QUESTION",
                    "focus_span": "GADA antibody was strongly positive",
                    "target": "GADA strongly positive result",
                    "time_constraint": {
                        "axis": "event_time",
                        "relation": "LOCATE",
                    },
                }
            ],
        },
        question,
        QueryFrame(),
    )
    constraint = ir["requirements"][0]["time_constraint"]
    assert constraint["axis"] == "document_time"
    assert constraint["relation"] == "LOCATE"


def test_latest_value_requirement_preserves_latest_event_selector():
    agent = _agent()
    question = "What was the latest HbA1c result?"
    ir = agent._rc_normalize_ir(
        {
            "answer_type": "VALUE",
            "requirements": [
                {
                    "id": "r1",
                    "grounding_kind": "QUESTION",
                    "focus_span": "latest HbA1c result",
                    "target": "latest HbA1c result",
                    "time_constraint": {},
                }
            ],
        },
        question,
        QueryFrame(),
    )
    constraint = ir["requirements"][0]["time_constraint"]
    assert constraint["axis"] == "event_time"
    assert constraint["relation"] == "LATEST"


def test_question_provenance_is_not_used_as_a_structured_certificate():
    agent = _agent()
    question = (
        "When was it documented that the patient's GADA antibody "
        "was strongly positive?"
    )
    ir = agent._rc_normalize_ir(
        {
            "answer_type": "DATE",
            "requirements": [
                {
                    "id": "r1",
                    "grounding_kind": "QUESTION",
                    "focus_span": "GADA antibody was strongly positive",
                    "target": (
                        "GADA antibody strongly positive test date and documentation"
                    ),
                    "retrieval_hint": "GADA titer documentation",
                    "time_constraint": {
                        "axis": "document_time",
                        "relation": "LOCATE",
                    },
                }
            ],
        },
        question,
        QueryFrame(),
    )
    plan = agent._controller_plan(ir, question, QueryFrame())
    slot = plan["required_slots"][0]
    assert (
        slot["target_surface"]
        == "GADA antibody strongly positive test date and documentation"
    )
    assert slot["focus_span"] == "GADA antibody was strongly positive"
    assert "proof_anchor" not in slot
    assert slot["proof_spec"]["status"] == "UNSPECIFIED"


def test_invalid_question_target_repairs_from_non_interrogative_focus_only():
    agent = _agent()
    question = "My neck feels sore. Can I take some painkillers?"
    ir = agent._rc_normalize_ir(
        {
            "requirements": [
                {
                    "id": "r1",
                    "grounding_kind": "QUESTION",
                    "focus_span": "My neck feels sore",
                    "target": "this target is deliberately far too long " * 8,
                },
                {
                    "id": "r2",
                    "grounding_kind": "QUESTION",
                    "focus_span": "Can I take some painkillers",
                    "target": "Can I take some painkillers?",
                },
            ]
        },
        question,
        QueryFrame(),
    )
    assert [item["id"] for item in ir["requirements"]] == ["r1"]
    assert ir["requirements"][0]["target"] == "My neck feels sore"
    actions = agent._last_requirement_normalization_actions
    assert any(
        item.get("reason") == "INVALID_TARGET_USE_FOCUS"
        for item in actions
    )
    assert any(
        item.get("id") == "r2" and item.get("action") == "DROP"
        for item in actions
    )


def test_option_fallback_never_uses_giant_question_stem():
    agent = _agent()
    question = "Which current state fits best?\n\nA. Alpha\nB. Beta\nC. Gamma"
    ir = agent._rc_normalize_ir({}, question, QueryFrame())
    requirement = ir["requirements"][0]
    assert requirement["grounding_kind"] == "DERIVED"
    assert requirement["focus_span"] == ""
    assert (
        requirement["target"]
        == "participant evidence relevant to visible options"
    )


def test_requirement_layer_does_not_own_context_ranking():
    assert (
        "_prepare_requirement_context_state"
        not in ReadRequirementContractMixin.__dict__
    )
    assert (
        "_rq_repair_context_candidate_order"
        not in ReadRequirementContractMixin.__dict__
    )


def test_query_type_is_telemetry_only():
    assert (
        SmartMem0Agent._compact_reasoning_output_instruction(
            "multi_hop_clinical_deduction"
        )
        == ""
    )
    assert (
        SmartMem0Agent._compact_reasoning_output_instruction(
            "entity_exact_match"
        )
        == ""
    )
