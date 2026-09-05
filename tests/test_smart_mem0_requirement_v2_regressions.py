"""Focused regressions for Requirement-v2 semantic contract repairs."""

from methods.smart_mem0.agent import SmartMem0Agent
from methods.smart_mem0.contracts import QueryFrame


def _agent():
    agent = object.__new__(SmartMem0Agent)
    agent._belief_status = {}
    agent._state_heads = {}
    agent._memories = []
    agent._relations = []
    agent.subject_aliases = {"patient": "primary_user"}
    return agent


def _memory(memory_id, claim, value, *, event_time="", document_time=""):
    return {
        "id": memory_id,
        "claim": claim,
        "value": value,
        "verbatim_value": "",
        "kind": "FACT",
        "semantic_role": "MEASUREMENT",
        "memory_tier": "HOT",
        "subject_id": "primary_user",
        "subject": "patient",
        "scope": "measurement",
        "state_key": "",
        "object_anchor": "",
        "entities": [],
        "scope_entities": [],
        "planning_tags": [],
        "assertion_mode": "DIRECT",
        "stance": "AFFIRM",
        "event_time": event_time,
        "document_time": document_time,
        "origin_document_time": document_time,
        "_status": "active",
    }


def _slot(target, **extra):
    slot = {
        "id": "r1",
        "type": "DIRECT",
        "evidence_role": "REQUIREMENT",
        "target_surface": target,
        "description": target,
        "required_fields": [],
        "history": False,
        "resolved_keys": [],
    }
    slot.update(extra)
    return slot


def test_documented_date_repairs_axis_to_document_time():
    agent = _agent()
    question = "When was it documented that the patient's GADA antibody was strongly positive?"
    ir = agent._rc_normalize_ir(
        {
            "answer_type": "DATE",
            "requirements": [
                {
                    "id": "r1",
                    "grounding_kind": "QUESTION",
                    "focus_span": "GADA antibody was strongly positive",
                    "target": "GADA strongly positive result",
                    "time_constraint": {"axis": "event_time", "relation": "LOCATE"},
                }
            ],
        },
        question,
        QueryFrame(),
    )
    constraint = ir["requirements"][0]["time_constraint"]
    assert constraint["axis"] == "document_time"
    assert constraint["relation"] == "LOCATE"


def test_question_proof_anchor_is_separate_from_retrieval_target():
    agent = _agent()
    question = "When was it documented that the patient's GADA antibody was strongly positive?"
    ir = agent._rc_normalize_ir(
        {
            "answer_type": "DATE",
            "requirements": [
                {
                    "id": "r1",
                    "grounding_kind": "QUESTION",
                    "focus_span": "GADA antibody was strongly positive",
                    "target": "GADA antibody strongly positive test date and documentation",
                    "retrieval_hint": "GADA titer documentation",
                    "time_constraint": {"axis": "document_time", "relation": "LOCATE"},
                }
            ],
        },
        question,
        QueryFrame(),
    )
    plan = agent._controller_plan(ir, question, QueryFrame())
    slot = plan["required_slots"][0]
    assert slot["target_surface"] == "GADA antibody strongly positive test date and documentation"
    assert slot["proof_anchor"] == "GADA antibody was strongly positive"


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
    assert any(item.get("reason") == "INVALID_TARGET_USE_FOCUS" for item in actions)
    assert any(item.get("id") == "r2" and item.get("action") == "DROP" for item in actions)


def test_option_fallback_never_uses_giant_question_stem():
    agent = _agent()
    question = "Which current state fits best?\n\nA. Alpha\nB. Beta\nC. Gamma"
    ir = agent._rc_normalize_ir({}, question, QueryFrame())
    requirement = ir["requirements"][0]
    assert requirement["grounding_kind"] == "DERIVED"
    assert requirement["focus_span"] == ""
    assert requirement["target"] == "participant evidence relevant to visible options"


def test_strong_earlier_candidate_is_protected_from_recovery_eviction():
    agent = _agent()
    exact = _memory(
        "m291",
        "Patient has a strongly positive GADA result with a titer greater than 2000 U/mL.",
        "> 2000 U/mL",
        event_time="2024-03-23",
        document_time="2024-03-23",
    )
    newer = _memory("m292", "Patient has a weakly positive islet cell antibody result.", "weakly positive")
    noise1 = _memory("m18", "Patient plans a fasting glucose test.", "agreed")
    noise2 = _memory("m288", "Patient has current basal insulin guidance.", "basal insulin")
    memories = [exact, newer, noise1, noise2]
    slot = _slot(
        "GADA antibody strongly positive test date and documentation",
        proof_anchor="GADA antibody was strongly positive",
    )
    run = {
        "plan": {"required_slots": [slot]},
        "requirement_status": {"r1": "EMPTY"},
        "requirement_context_candidates": {"r1": ["m292", "m18", "m288", "m291"]},
        "slot_support": {agent.CONTEXT_POOL_KEY: ["m292", "m18", "m288", "m291"]},
        "operation_candidates": memories,
        "beliefs": memories,
        "planning_seeds": [exact, newer],
        "trace": [
            {"retrieval_round": 1, "produces": ["r1"], "output_ids": ["m291"]},
            {"retrieval_round": 2, "produces": ["r1"], "output_ids": ["m292", "m18", "m288"]},
        ],
    }
    agent._last_requirement_context_candidates = dict(run["requirement_context_candidates"])
    repaired = agent._rq_repair_context_candidate_order(run, [exact, newer])
    assert repaired["requirement_context_candidates"]["r1"][:2] == ["m292", "m291"]
    assert repaired["requirement_context_reorders"][0]["protected_memory_id"] == "m291"


def test_mcd_instruction_requires_complete_reasoning_not_short_reasoning():
    instruction = SmartMem0Agent._compact_reasoning_output_instruction(
        "multi_hop_clinical_deduction"
    )
    assert "COMPLETE" in instruction
    assert "mechanism-explicit" in instruction
    assert "short `Evidence:`" in instruction
    assert "short `Conclusion:`" in instruction
    assert "short `Reasoning:`" not in instruction
