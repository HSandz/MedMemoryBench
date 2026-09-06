"""Regressions for complete terminal answers and post-retrieval answerability."""

from methods.smart_mem0.agent import SmartMem0Agent
from methods.smart_mem0.contracts import QueryFrame


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


def _direct_slot(target, *, proof_status="UNSPECIFIED", history=False):
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
        "proof_spec": {"status": proof_status},
        "history": history,
        "degraded": False,
    }


def test_answerability_and_terminal_contracts_are_first_in_runtime_mro():
    assert SmartMem0Agent.__mro__[1].__name__ == "ReadAnswerabilityContractMixin"
    assert SmartMem0Agent.__mro__[2].__name__ == "ReadTerminalAnswerContractMixin"


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
    supports, reason = agent._authorize_controller_answer(projected, [seed], QueryFrame())
    assert reason == "AUTHORIZED_COMPLETE_PROPOSITION"
    assert [memory["id"] for memory in supports] == ["m1"]


def test_short_status_candidate_renders_full_grounded_proposition():
    agent = _agent()
    seed = _memory(
        claim="The patient's weight loss has stopped and the patient's weight is stabilizing.",
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


def test_uncertified_is_not_reported_as_empty_when_candidate_is_present():
    agent = _agent()
    memory = _memory(
        claim="The patient takes metformin 1500 mg per day.",
        value="metformin 1500 mg/day",
        evidence_ids=[],
    )
    slot = _direct_slot("current metformin regimen")
    plan = {"required_slots": [slot], "semantic_relations": []}
    status, relations, complete = agent._retrieval_status(
        plan, {"r1": ["m1"]}, [memory], []
    )
    assert status == {"r1": "UNCERTIFIED"}
    assert relations == {}
    assert complete is False


def test_round_one_cefuroxime_can_close_as_direct_b_without_exact_proof_spec():
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
    agent._active_answer_question = "Which antibiotic was the patient instructed to avoid?"
    slot = _direct_slot("antibiotic instructed to avoid", history=True)
    plan = {
        "required_slots": [slot],
        "semantic_ir": {
            "answer_type": "ENTITY",
            "requirements": [
                {
                    "id": "r1",
                    "grounding_kind": "QUESTION",
                    "target": "antibiotic instructed to avoid",
                }
            ],
        },
        "visible_options": {},
        "need_evidence": False,
        "query_spec": {"world_knowledge_bridge_allowed": False},
        "semantic_relations": [],
    }
    closure = agent._post_retrieval_closure(plan, [memory], QueryFrame(), [])
    assert closure is not None
    assert closure["answer"] == "cefuroxime"
    assert closure["support_ids"] == ["m1"]
    assert closure["reason"] == "RUNTIME_STABLE_SEMANTIC_CERTIFICATE"


def test_started_date_query_compiles_to_earliest_event_time():
    agent = _agent()
    constraint = agent._rq_repair_time_constraint(
        {"axis": "", "relation": "", "anchor": "", "end": ""},
        "When did the patient start taking metformin 1500 mg/day as prescribed?",
        "DATE",
    )
    assert constraint["axis"] == "event_time"
    assert constraint["relation"] == "EARLIEST"


def test_orphan_derived_requirement_is_dropped_not_silently_legalized():
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


def test_dataset_query_type_cannot_change_reasoning_instruction():
    agent = _agent()
    assert agent._compact_reasoning_output_instruction("multi_hop_clinical_deduction") == ""
    assert agent._compact_reasoning_output_instruction("entity_exact_match") == ""
    infer_plan = {
        "semantic_relations": [
            {"type": "INFER", "from": "r1", "to": "ANSWER"}
        ]
    }
    assert agent._semantic_reasoning_output_instruction(infer_plan)
