"""Regression tests for proof/context separation and evidence-lookup requirements."""

import json
from types import SimpleNamespace

import pytest

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


def _memory(memory_id, claim, value, *, event_time="", subject_id="primary_user"):
    return {
        "id": memory_id,
        "claim": claim,
        "value": value,
        "verbatim_value": "",
        "kind": "FACT",
        "semantic_role": "OBSERVATION",
        "memory_tier": "COLD",
        "subject_id": subject_id,
        "subject": "patient",
        "scope": "",
        "state_key": "",
        "object_anchor": "",
        "entities": [],
        "scope_entities": [],
        "planning_tags": [],
        "assertion_mode": "DIRECT",
        "stance": "AFFIRM",
        "event_time": event_time,
        "document_time": "",
        "origin_document_time": "",
        "_status": "active",
    }


def _requirement(target, *, slot_id="r1", slot_type="DIRECT", **extra):
    slot = {
        "id": slot_id,
        "type": slot_type,
        "evidence_role": "REQUIREMENT",
        "target_surface": target,
        "description": target,
        "required_fields": [],
        "history": False,
    }
    slot.update(extra)
    return slot


def test_generic_requirement_keeps_candidates_but_junk_does_not_prove_found():
    agent = _agent()
    candidates = [
        _memory("m1", "The patient follows an irregular daily routine.", "irregular"),
        _memory("m2", "The patient does not have a home glucose meter.", "false"),
        _memory("m3", "Another unrelated observation.", "recorded"),
    ]
    agent._memories = candidates
    agent._hybrid_search = lambda *_args, **_kwargs: candidates

    slot = _requirement("chronic metabolic disease")
    supports = agent._operation_slot_support(slot, candidates, [])

    assert [memory["id"] for memory in supports][:2] == ["m1", "m2"]
    assert not agent._slot_covered(
        slot,
        [memory["id"] for memory in supports],
        supports,
        [],
    )


def test_target_proof_is_stricter_than_one_generic_token_overlap():
    agent = _agent()
    weak = _memory("m1", "A disease was mentioned in history.", "recorded")
    # "state" is deliberately removed by the existing retrieval vocabulary.
    # Exercise two retained concept terms rather than a one-term normalized target.
    slot = _requirement("renal disease")
    assert not agent._requirement_target_proof(slot, weak)

    exact = _memory("m2", "Renal disease was explicitly recorded.", "recorded")
    assert agent._requirement_target_proof(slot, exact)


def test_target_compatible_direct_requirement_can_be_found():
    agent = _agent()
    diabetes = _memory(
        "m1",
        "The chronic metabolic disease in the history is diabetes.",
        "diabetes",
    )
    slot = _requirement("chronic metabolic disease")
    assert agent._slot_covered(slot, ["m1"], [diabetes], [])


def test_temporal_requirement_binds_target_before_accepting_date():
    agent = _agent()
    wrong_latest = _memory(
        "m83",
        "Observe and report body responses about an hour after eating modified takeout.",
        "observe response",
        event_time="2024-01-31",
    )
    slot = _requirement(
        "latest HbA1c result on the follow-up test",
        slot_type="TEMPORAL",
        time_axis="event_time",
        temporal_relation="LATEST",
    )
    assert not agent._slot_covered(slot, ["m83"], [wrong_latest], [])

    correct = _memory(
        "m97",
        "The latest HbA1c result on the follow-up test was 8.1%.",
        "8.1%",
        event_time="2024-02-06",
    )
    assert agent._slot_covered(slot, ["m97"], [correct], [])


def test_relation_proof_cannot_use_wrong_requirement_endpoint():
    agent = _agent()
    first = _memory(
        "m1", "Unrelated dated observation.", "one", event_time="2024-01-01"
    )
    second = _memory(
        "m2", "Another unrelated dated observation.", "two", event_time="2024-01-02"
    )
    plan = {
        "required_slots": [
            _requirement("HbA1c result", time_axis="event_time"),
            _requirement("insulin dose", slot_id="r2", time_axis="event_time"),
        ],
        "semantic_relations": [
            {
                "type": "TEMPORAL_ORDER",
                "from": "r1",
                "to": "r2",
                "relation": "BEFORE",
            }
        ],
    }
    statuses = agent._relation_status_map(
        plan,
        {"r1": ["m1"], "r2": ["m2"]},
        [first, second],
        [],
    )
    assert statuses == {"TEMPORAL_ORDER:r1:r2:BEFORE": "UNPROVEN"}


def test_single_requirement_seed_is_context_only_and_never_enters_proof_support():
    agent = _agent()
    cefuroxime = _memory(
        "m17",
        "Patient has a documented allergy to cefuroxime (cephalosporins).",
        "cefuroxime",
    )
    run = {
        "fast_supports": None,
        "plan": {"required_slots": [_requirement("antibiotic instructed to avoid")]},
        "requirement_status": {"r1": "EMPTY"},
        "relation_status": {},
        "retrieval_complete": False,
        "slot_support": {"r1": []},
        "planning_seeds": [cefuroxime.copy()],
        "operation_output_ids": set(),
        "operation_candidates": [],
        "beliefs": [],
        "trace": [],
        "relations": [],
    }

    agent._reserve_initial_requirement_context(run, [cefuroxime])

    assert run["slot_support"]["r1"] == []
    assert run["requirement_proof_support"] == {"r1": []}
    assert run["requirement_context_candidates"] == {"r1": ["m17"]}
    assert run["slot_support"][agent.CONTEXT_POOL_KEY] == ["m17"]
    assert run["requirement_status"] == {"r1": "EMPTY"}
    assert run["retrieval_complete"] is False
    assert run["reserved_seed_context"] == []


def test_recovery_adds_recall_without_displacing_all_earlier_candidates():
    agent = _agent()
    seed = _memory("m67", "Emergency danger sign safety instruction.", "urgent care")
    old = _memory("m70", "Avoid NSAIDs because of gastric bleeding history.", "NSAIDs")
    recovery_first = _memory(
        "m65", "Stop empagliflozin before a procedure.", "empagliflozin"
    )
    cefuroxime = _memory(
        "m21",
        "Patient has a documented allergic reaction to cefuroxime (cephalosporins).",
        "cefuroxime",
    )
    candidates = [old, recovery_first, cefuroxime]
    run = {
        "fast_supports": None,
        "plan": {"required_slots": [_requirement("antibiotic instructed to avoid")]},
        "requirement_status": {"r1": "EMPTY"},
        "relation_status": {},
        "retrieval_complete": False,
        "slot_support": {"r1": ["m70", "m65", "m21"]},
        "planning_seeds": [seed.copy()],
        "operation_output_ids": {"m70", "m65", "m21"},
        "operation_candidates": [memory.copy() for memory in candidates],
        "beliefs": [],
        "trace": [
            {"retrieval_round": 1, "produces": ["r1"], "output_ids": ["m70"]},
            {
                "retrieval_round": 2,
                "produces": ["r1"],
                "output_ids": ["m65", "m21"],
            },
        ],
        "relations": [],
    }

    agent._prepare_requirement_context_state(run, [seed])

    assert set(run["requirement_context_candidates"]["r1"]) == {"m65", "m21", "m70", "m67"}
    assert run["slot_support"]["r1"] == []
    packed = agent._role_aware_support_ids(
        run["plan"]["required_slots"],
        run["slot_support"],
        ["m65", "m21", "m70", "m67"],
        3,
    )
    assert set(packed) == {"m70", "m65", "m21"}


def test_found_reserves_proof_but_does_not_exclude_high_recall_alternate():
    agent = _agent()
    proof = _memory("m1", "The requested HbA1c result was 8.1%.", "8.1%")
    alternate = _memory(
        "m2", "A nearby lab note discussed glycemic control.", "poor control"
    )
    run = {
        "fast_supports": None,
        "plan": {"required_slots": [_requirement("HbA1c result")]},
        "requirement_status": {"r1": "FOUND"},
        "relation_status": {},
        "retrieval_complete": True,
        "slot_support": {"r1": ["m1", "m2"]},
        "planning_seeds": [],
        "operation_output_ids": {"m1", "m2"},
        "operation_candidates": [proof.copy(), alternate.copy()],
        "beliefs": [proof.copy(), alternate.copy()],
        "trace": [
            {
                "retrieval_round": 1,
                "produces": ["r1"],
                "output_ids": ["m1", "m2"],
            }
        ],
        "relations": [],
    }
    agent._prepare_requirement_context_state(run, [])
    assert run["slot_support"]["r1"] == ["m1"]
    assert run["requirement_context_candidates"]["r1"] == ["m1", "m2"]
    packed = agent._role_aware_support_ids(
        run["plan"]["required_slots"], run["slot_support"], ["m1", "m2"], 2
    )
    assert packed == ["m1", "m2"]


def test_multi_requirement_packing_is_round_robin_before_second_candidate():
    agent = _agent()
    agent._last_requirement_status = {"r1": "EMPTY", "r2": "EMPTY"}
    agent._last_requirement_proof_support = {"r1": [], "r2": []}
    agent._last_requirement_context_candidates = {
        "r1": ["a1", "a2"],
        "r2": ["b1", "b2"],
    }
    slots = [
        _requirement("first fact"),
        _requirement("second fact", slot_id="r2"),
    ]
    support = {agent.CONTEXT_POOL_KEY: ["a1", "a2", "b1", "b2"]}
    packed = agent._role_aware_support_ids(
        slots, support, ["a1", "a2", "b1", "b2"], 4
    )
    assert packed[:2] == ["a1", "b1"]
    assert set(packed[2:]) == {"a2", "b2"}


def test_question_focus_is_provenance_while_target_is_lookup_variable():
    agent = _agent()
    question = "What antibiotic was the patient clearly instructed to avoid?"
    ir = agent._rc_normalize_ir(
        {
            "answer_type": "ENTITY",
            "requirements": [
                {
                    "id": "r1",
                    "grounding_kind": "QUESTION",
                    "focus_span": "antibiotic",
                    "target": "prior antibiotic avoidance",
                    "retrieval_hint": "medication allergy or explicit avoid instruction",
                }
            ],
        },
        question,
        QueryFrame(),
    )
    requirement = ir["requirements"][0]
    assert requirement["focus_span"] == "antibiotic"
    assert requirement["target"] == "prior antibiotic avoidance"

    plan = agent._controller_plan(ir, question, QueryFrame())
    slot = plan["required_slots"][0]
    assert slot["focus_span"] == "antibiotic"
    assert slot["target_surface"] == "prior antibiotic avoidance"
    assert plan["query_spec"]["semantic_ir_version"] == "minimal-v2-evidence-lookup"


def test_derived_requirement_is_first_class_lookup_without_focus_span():
    agent = _agent()
    question = "Could late-night takeout explain morning blurry vision?"
    parsed = {
        "answer_type": "TEXT",
        "requirements": [
            {
                "id": "r1",
                "grounding_kind": "QUESTION",
                "focus_span": "late-night takeout",
                "target": "late-night meal pattern",
                "retrieval_hint": "late-night food exposure",
            },
            {
                "id": "r2",
                "grounding_kind": "DERIVED",
                "focus_span": "",
                "target": "morning glycemic state",
                "retrieval_hint": "fasting glucose or morning hyperglycemia",
            },
            {
                "id": "r3",
                "grounding_kind": "QUESTION",
                "focus_span": "morning blurry vision",
                "target": "morning visual symptoms",
                "retrieval_hint": "morning visual symptom",
            },
        ],
        "relations": [
            {
                "type": "POSSIBLE_CAUSE",
                "from": "r1",
                "to": "r2",
                "bridge_goal": "Explain how the grounded late-night exposure could affect the grounded morning glycemic state.",
            },
            {
                "type": "INFER",
                "from": "r2",
                "to": "r3",
                "bridge_goal": "Explain how the grounded glycemic state could account for the grounded visual symptom.",
            },
        ],
    }

    ir = agent._rc_normalize_ir(parsed, question, QueryFrame())
    assert len(ir["requirements"]) == 3
    middle = ir["requirements"][1]
    assert middle["grounding_kind"] == "DERIVED"
    assert middle["focus_span"] == ""
    assert middle["target"] == "morning glycemic state"
    assert ir["relations"][0]["bridge_goal"].startswith("Explain how")

    plan = agent._controller_plan(ir, question, QueryFrame())
    middle_slot = next(
        slot for slot in plan["required_slots"] if slot["id"] == "r2"
    )
    assert middle_slot["target_surface"] == "morning glycemic state"
    assert middle_slot["grounding_kind"] == "DERIVED"


def test_malformed_derived_without_lookup_target_is_removed_with_relations():
    agent = _agent()
    question = "Could late-night takeout explain morning blurry vision?"
    ir = agent._rc_normalize_ir(
        {
            "requirements": [
                {
                    "id": "r1",
                    "grounding_kind": "QUESTION",
                    "focus_span": "late-night takeout",
                    "target": "late-night meal pattern",
                },
                {
                    "id": "r2",
                    "grounding_kind": "DERIVED",
                    "focus_span": "",
                    "target": "",
                },
            ],
            "relations": [{"type": "INFER", "from": "r2", "to": "r1"}],
        },
        question,
        QueryFrame(),
    )
    assert [item["id"] for item in ir["requirements"]] == ["r1"]
    assert ir["relations"] == []


def test_question_like_target_with_question_mark_is_not_accepted_as_requirement():
    agent = _agent()
    question = "My neck feels sore. Can I take some painkillers?"
    ir = agent._rc_normalize_ir(
        {
            "requirements": [
                {
                    "id": "r1",
                    "grounding_kind": "QUESTION",
                    "focus_span": "My neck feels sore",
                    "target": "work-related neck symptoms",
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


def test_requirement_target_is_bounded_and_atomic_by_shape():
    agent = _agent()
    question = "Could my recent routine affect how I feel in the morning?"
    overlong = " ".join(f"evidence{i}" for i in range(20))
    ir = agent._rc_normalize_ir(
        {
            "requirements": [
                {
                    "id": "r1",
                    "grounding_kind": "QUESTION",
                    "focus_span": "recent routine",
                    "target": "recent routine pattern",
                },
                {
                    "id": "r2",
                    "grounding_kind": "DERIVED",
                    "target": overlong,
                },
            ]
        },
        question,
        QueryFrame(),
    )
    assert [item["id"] for item in ir["requirements"]] == ["r1"]


def test_multi_hop_output_instruction_preserves_structure_but_avoids_repetition():
    instruction = SmartMem0Agent._compact_reasoning_output_instruction(
        "multi_hop_clinical_deduction"
    )
    assert "Evidence:" in instruction
    assert "Reasoning:" in instruction
    assert "Conclusion:" in instruction
    assert "Do not repeat" in instruction
    assert "complete" in instruction.lower()
    assert "mechanism-explicit" in instruction
    assert SmartMem0Agent._compact_reasoning_output_instruction(
        "entity_exact_match"
    ) == ""


@pytest.mark.parametrize("bad_target", [
    "x" * 121,
    " ".join(["aa"] * 17),
])
def test_bad_question_target_repairs_only_its_node_and_preserves_graph(bad_target):
    agent = _agent()
    parsed = {
        "requirements": [
            {"id": "r1", "grounding_kind": "QUESTION", "focus_span": "late meals", "target": bad_target},
            {"id": "r2", "grounding_kind": "DERIVED", "target": "morning glucose"},
        ],
        "relations": [{"type": "DEPENDS_ON", "from": "r1", "to": "r2"}],
    }
    ir = agent._rc_normalize_ir(parsed, "Could late meals explain my symptoms?", QueryFrame())
    assert [r["target"] for r in ir["requirements"]] == ["late meals", "morning glucose"]
    assert ir["relations"] == parsed["relations"]
    assert ir["normalization_actions"] == [{
        "index": 0, "id": "r1", "action": "REPAIR", "reason": "INVALID_TARGET_USE_FOCUS",
    }]
    assert ir["graph_validation"]["valid"]


@pytest.mark.parametrize("parsed", [{}, {"requirements": [None]}, [], {"requirements": [
    {"grounding_kind": "DERIVED", "target": ""}
]}])
def test_degraded_fallback_is_bounded_uncertified_and_not_direct_decomposition(parsed):
    agent = _agent()
    question = " ".join(["aa"] * 30) + "?"
    ir = agent._rc_normalize_ir(parsed, question, QueryFrame())
    assert ir["normalization_status"] == "DEGRADED"
    target = ir["requirements"][0]["target"]
    assert agent._rq_compact_target(target) == target
    assert len(agent._rc_terms(target)) <= 16
    assert "?" not in target
    assert ir["candidate"] is None
    assert ir["relations"] == []
    plan = agent._controller_plan(ir, question, QueryFrame())
    assert plan["compiled_mode"] == "DEGRADED"
    assert not plan["query_spec"]["world_knowledge_bridge_allowed"]
    assert not agent._requirement_target_proof(plan["required_slots"][0], _memory("m1", target, "value"))


def test_invalid_derived_removes_only_incident_edges_and_reports_reason():
    agent = _agent()
    ir = agent._rc_normalize_ir({
        "requirements": [
            {"id": "r1", "focus_span": "meals", "target": "meal pattern"},
            {"id": "r2", "grounding_kind": "DERIVED", "target": ""},
            {"id": "r3", "focus_span": "symptoms", "target": "symptoms"},
        ],
        "relations": [
            {"type": "DEPENDS_ON", "from": "r3", "to": "r2"},
            {"type": "POSSIBLE_CAUSE", "from": "r1", "to": "r3"},
        ],
    }, "Do meals explain symptoms?", QueryFrame())
    assert [r["id"] for r in ir["requirements"]] == ["r1", "r3"]
    assert ir["relations"] == [{"type": "POSSIBLE_CAUSE", "from": "r1", "to": "r3"}]
    assert any(a.get("id") == "r2" and a.get("reason") == "INVALID_TARGET"
               for a in ir["normalization_actions"])
    assert any(a.get("action") == "DROP_RELATION" for a in ir["normalization_actions"])


def test_graph_dependency_direction_and_orphan_diagnostics_do_not_invent_edges():
    requirements = [
        {"id": "r1", "grounding_kind": "QUESTION"},
        {"id": "r2", "grounding_kind": "QUESTION"},
        {"id": "r3", "grounding_kind": "DERIVED"},
        {"id": "r4", "grounding_kind": "DERIVED"},
    ]
    relations = [
        {"type": "POSSIBLE_CAUSE", "from": "r1", "to": "r2"},
        {"type": "DEPENDS_ON", "from": "r2", "to": "r3"},
        {"type": "INFER", "from": "r2", "to": "ANSWER"},
    ]
    graph = _agent()._rq_graph_validation(requirements, relations)
    assert graph["connected_to_answer"] == {"r1": True, "r2": True, "r3": True, "r4": False}
    assert graph["orphan_requirements"] == ["r4"]
    assert not graph["valid"]
    assert len(relations) == 3  # Reachability does not manufacture a mediator/causal edge.


def test_structured_certificate_does_not_replace_broader_search_target():
    agent = _agent()
    question = "When was the antibody strongly positive result documented?"
    ir = agent._rc_normalize_ir({
        "answer_type": "DATE",
        "requirements": [{
            "id": "r1", "focus_span": "antibody strongly positive",
            "target": "antibody strongly positive test timestamp documentation record",
            "proof_spec": {"match": {"subject_id": "primary_user", "scope": "measurement",
                                      "state_key": "antibody_result", "stance": "AFFIRM"},
                           "answer_field": "document_time"},
            "time_constraint": {"axis": "document_time", "relation": "LOCATE"},
        }],
    }, question, QueryFrame())
    slot = agent._controller_plan(ir, question, QueryFrame())["required_slots"][0]
    assert "proof_anchor" not in slot
    assert slot["focus_span"] == "antibody strongly positive"
    assert slot["retrieval_target"] in agent._rc_search_query(slot, question)
    memory = _memory("m1", "Antibody strongly positive result", ">2000", event_time="2024-03-20")
    memory["document_time"] = "2024-03-23"
    memory.update(scope="measurement", state_key="antibody_result", evidence_ids=["ev1"])
    assert agent._slot_covered(slot, ["m1"], [memory], [])
    memory["document_time"] = ""
    assert not agent._slot_covered(slot, ["m1"], [memory], [])


def test_relative_selector_survives_normalization():
    agent = _agent()
    ir = agent._rc_normalize_ir({"requirements": [{
        "id": "r1", "focus_span": "headache last night", "target": "headache episode",
        "time_constraint": {"axis": "event_time", "relation": "EXACT", "anchor": "last night"},
    }]}, "What about the headache last night?", QueryFrame())
    assert ir["requirements"][0]["time_constraint"]["relation"] == "EXACT"
    assert ir["requirements"][0]["time_constraint"]["anchor"] == "last night"


def test_recovery_cannot_erase_high_quality_earlier_documentation_candidate():
    agent = _agent()
    old = _memory("m1", "Antibody strongly positive titer >2000 U/mL", ">2000 U/mL")
    old.update(document_time="2024-03-23", _score=0.0327)
    recovery = [
        _memory("m2", "Another antibody weakly positive", "weakly positive"),
        _memory("m3", "Blood glucose testing planned", "planned"),
        _memory("m4", "Treatment guidance", "one injection"),
    ]
    for m in recovery:
        m.update(document_time="2024-03-24", _score=0.031)
    slot = _requirement("antibody strongly positive documentation", slot_type="TEMPORAL",
                        time_axis="document_time", temporal_relation="LOCATE")
    run = {
        "fast_supports": None, "plan": {"required_slots": [slot]},
        "requirement_status": {"r1": "EMPTY"}, "slot_support": {"r1": []},
        "operation_candidates": [old, *recovery],
        "operation_output_ids": {"m1", "m2", "m3", "m4"},
        "trace": [
            {"retrieval_round": 1, "produces": ["r1"], "output_ids": ["m1"]},
            {"retrieval_round": 2, "produces": ["r1"], "output_ids": ["m2", "m3", "m4"]},
        ],
    }
    agent._hybrid_search = lambda *_a, **_k: pytest.fail("Packing must not retrieve")
    agent._prepare_requirement_context_state(run, [])
    for cap in (1, 2, 3):
        packed = agent._role_aware_support_ids([slot], run["slot_support"], ["m2", "m3", "m4", "m1"], cap)
        assert len(packed) == cap
        # The GitHub policy keeps recovery first and reserves the second seat
        # for a strong earlier candidate; it does not apply a new utility ranker.
        assert packed[0] == "m2"
        if cap >= 2:
            assert "m1" in packed
    assert run["requirement_status"] == {"r1": "EMPTY"}


def test_controller_logs_raw_normalized_and_repair_without_another_call():
    agent = _agent()
    raw = {"requirements": [{"id": "r1", "focus_span": "dose", "target": "x" * 121}]}
    calls = []

    def chat(messages, **kwargs):
        calls.append(messages)
        return SimpleNamespace(content=json.dumps(raw))

    agent._llm_client = SimpleNamespace(chat=chat)
    agent._response_usage = lambda *_: {}
    supports, plan, telemetry = agent._semantic_controller("What dose?", [], QueryFrame())
    assert supports is None
    assert len(calls) == 1
    assert telemetry["controller_raw_ir"] == raw
    assert telemetry["normalized_ir"]["requirements"][0]["target"] == "dose"
    assert telemetry["normalization_status"] == "REPAIRED"
    assert plan["graph_validation"]["valid"]
    prompt = calls[0][0]["content"]
    assert "infer is exceptional" in prompt.lower()
    assert "document_time = when something was documented" in prompt
    assert "VISIBLE OPTIONS are answer propositions, not memory facts" in prompt
