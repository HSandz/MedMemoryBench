"""Regression tests for proof/context separation and grounded requirements."""

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
    slot = _requirement("disease state")
    assert not agent._requirement_target_proof(slot, weak)

    exact = _memory("m2", "The disease state was explicitly recorded.", "early DKA")
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
    first = _memory("m1", "Unrelated dated observation.", "one", event_time="2024-01-01")
    second = _memory("m2", "Another unrelated dated observation.", "two", event_time="2024-01-02")
    plan = {
        "required_slots": [
            _requirement("HbA1c result", time_axis="event_time"),
            _requirement("insulin dose", slot_id="r2", time_axis="event_time"),
        ],
        "semantic_relations": [
            {"type": "TEMPORAL_ORDER", "from": "r1", "to": "r2", "relation": "BEFORE"}
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


def test_recovery_candidates_precede_failed_round_and_seed_context():
    agent = _agent()
    seed = _memory("m67", "Emergency danger sign safety instruction.", "urgent care")
    old = _memory("m70", "Avoid NSAIDs because of gastric bleeding history.", "NSAIDs")
    recovery_first = _memory("m65", "Stop empagliflozin before a procedure.", "empagliflozin")
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
            {"retrieval_round": 2, "produces": ["r1"], "output_ids": ["m65", "m21"]},
        ],
        "relations": [],
    }

    agent._prepare_requirement_context_state(run, [seed])

    assert run["requirement_context_candidates"]["r1"][:4] == [
        "m65", "m21", "m70", "m67"
    ]
    assert run["slot_support"]["r1"] == []
    packed = agent._role_aware_support_ids(
        run["plan"]["required_slots"],
        run["slot_support"],
        ["m65", "m21", "m70", "m67"],
        2,
    )
    assert packed == ["m65", "m21"]


def test_found_reserves_proof_but_does_not_exclude_high_recall_alternate():
    agent = _agent()
    proof = _memory("m1", "The requested HbA1c result was 8.1%.", "8.1%")
    alternate = _memory("m2", "A nearby lab note discussed glycemic control.", "poor control")
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
        "trace": [{"retrieval_round": 1, "produces": ["r1"], "output_ids": ["m1", "m2"]}],
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
    slots = [_requirement("first fact"), _requirement("second fact", slot_id="r2")]
    support = {agent.CONTEXT_POOL_KEY: ["a1", "a2", "b1", "b2"]}
    packed = agent._role_aware_support_ids(
        slots, support, ["a1", "a2", "b1", "b2"], 4
    )
    assert packed[:2] == ["a1", "b1"]
    assert set(packed[2:]) == {"a2", "b2"}


def test_intermediate_requirement_has_own_retrieval_target_and_bridge_goal():
    agent = _agent()
    question = "Could late-night takeout explain morning blurry vision?"
    parsed = {
        "answer_type": "TEXT",
        "requirements": [
            {
                "id": "r1",
                "grounding_kind": "QUESTION",
                "focus_span": "late-night takeout",
                "retrieval_hint": "late-night food exposure",
            },
            {
                "id": "r2",
                "grounding_kind": "INTERMEDIATE",
                "focus_span": "",
                "evidence_target": "morning hyperglycemia",
                "retrieval_hint": "participant morning glucose state",
            },
            {
                "id": "r3",
                "grounding_kind": "QUESTION",
                "focus_span": "morning blurry vision",
                "retrieval_hint": "morning visual symptom",
            },
        ],
        "relations": [
            {
                "type": "POSSIBLE_CAUSE",
                "from": "r1",
                "to": "r2",
                "bridge_goal": "Explain how the grounded late-night exposure could contribute to the grounded morning glucose state.",
            },
            {
                "type": "INFER",
                "from": "r2",
                "to": "r3",
                "bridge_goal": "Explain how the grounded glucose state could account for the grounded morning symptom.",
            },
        ],
    }

    ir = agent._rc_normalize_ir(parsed, question, QueryFrame())
    assert len(ir["requirements"]) == 3
    middle = ir["requirements"][1]
    assert middle["grounding_kind"] == "INTERMEDIATE"
    assert middle["focus_span"] == ""
    assert middle["evidence_target"] == "morning hyperglycemia"
    assert ir["relations"][0]["bridge_goal"].startswith("Explain how")

    plan = agent._controller_plan(ir, question, QueryFrame())
    middle_slot = next(slot for slot in plan["required_slots"] if slot["id"] == "r2")
    assert middle_slot["target_surface"] == "morning hyperglycemia"
    assert middle_slot["grounding_kind"] == "INTERMEDIATE"


def test_malformed_intermediate_without_retrieval_target_is_removed():
    agent = _agent()
    question = "Could late-night takeout explain morning blurry vision?"
    ir = agent._rc_normalize_ir(
        {
            "requirements": [
                {
                    "id": "r1",
                    "grounding_kind": "QUESTION",
                    "focus_span": "late-night takeout",
                },
                {
                    "id": "r2",
                    "grounding_kind": "INTERMEDIATE",
                    "focus_span": "",
                    "evidence_target": "",
                },
            ],
            "relations": [{"type": "INFER", "from": "r2", "to": "r1"}],
        },
        question,
        QueryFrame(),
    )
    assert [item["id"] for item in ir["requirements"]] == ["r1"]
    assert ir["relations"] == []
