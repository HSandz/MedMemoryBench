"""Regression tests for target-bound REQUIREMENT coverage.

These tests intentionally keep candidate retrieval broader than deterministic
coverage. The final answer model may see bounded unverified context, while
FOUND/EMPTY remains a structural retrieval status rather than semantic
sufficiency.
"""

from methods.smart_mem0.agent import SmartMem0Agent


def _agent():
    agent = object.__new__(SmartMem0Agent)
    agent._belief_status = {}
    agent._state_heads = {}
    agent._memories = []
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


def _requirement(target, *, slot_type="DIRECT", **extra):
    slot = {
        "id": "r1",
        "type": slot_type,
        "evidence_role": "REQUIREMENT",
        "target_surface": target,
        "required_fields": [],
        "history": False,
    }
    slot.update(extra)
    return slot


def test_generic_requirement_keeps_two_candidates_but_junk_does_not_prove_found():
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

    # 07db's bounded backup remains intact: candidate breadth is not proof.
    assert [memory["id"] for memory in supports] == ["m1", "m2"]
    assert not agent._slot_covered(
        slot,
        [memory["id"] for memory in supports],
        supports,
        [],
    )


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

    # A valid date on the wrong concept must never make the requirement FOUND.
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
        "m1",
        "Unrelated dated observation.",
        "one",
        event_time="2024-01-01",
    )
    second = _memory(
        "m2",
        "Another unrelated dated observation.",
        "two",
        event_time="2024-01-02",
    )
    plan = {
        "required_slots": [
            _requirement("HbA1c result", time_axis="event_time"),
            {
                **_requirement("insulin dose", time_axis="event_time"),
                "id": "r2",
            },
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


def test_single_requirement_top1_seed_is_context_only_not_proof():
    agent = _agent()
    cefuroxime = _memory(
        "m17",
        "Patient has a documented allergy to cefuroxime (cephalosporins).",
        "cefuroxime",
    )
    run = {
        "fast_supports": None,
        "plan": {
            "required_slots": [
                _requirement("antibiotic instructed to avoid")
            ]
        },
        "requirement_status": {"r1": "EMPTY"},
        "relation_status": {},
        "retrieval_complete": False,
        "slot_support": {"r1": ["m_wrong"]},
        "planning_seeds": [],
    }

    agent._reserve_initial_requirement_context(run, [cefuroxime])

    assert run["slot_support"]["r1"][:2] == ["m17", "m_wrong"]
    assert run["requirement_status"] == {"r1": "EMPTY"}
    assert run["retrieval_complete"] is False
    assert run["reserved_seed_context"] == [
        {"slot_id": "r1", "memory_id": "m17", "mode": "top1_unverified"}
    ]
    assert run["planning_seeds"][0]["_supplementary_context"] is True


def test_multi_requirement_query_never_uses_unverified_top1_as_cross_slot_proof():
    agent = _agent()
    seed = _memory("m1", "A top-ranked but unrelated memory.", "value")
    run = {
        "fast_supports": None,
        "plan": {
            "required_slots": [
                _requirement("first requested fact"),
                {**_requirement("second requested fact"), "id": "r2"},
            ]
        },
        "requirement_status": {"r1": "EMPTY", "r2": "EMPTY"},
        "relation_status": {},
        "retrieval_complete": False,
        "slot_support": {"r1": [], "r2": []},
        "planning_seeds": [],
    }

    agent._reserve_initial_requirement_context(run, [seed])

    assert run["slot_support"] == {"r1": [], "r2": []}
    assert run["reserved_seed_context"] == []
