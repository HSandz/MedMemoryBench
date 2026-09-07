"""Priority Query-phase regressions: family recall and evidence-lineage context."""

from methods.smart_mem0.agent import SmartMem0Agent
from methods.smart_mem0.contracts import QueryFrame


def _agent():
    agent = object.__new__(SmartMem0Agent)
    agent._belief_status = {}
    agent._state_heads = {}
    agent._memories = []
    agent._relations = []
    agent.subject_aliases = {"patient": "primary_user"}
    agent._last_requirement_context_candidates = {}
    agent._last_requirement_proof_support = {}
    agent._last_requirement_status = {}
    agent._last_candidate_propositions = {}
    agent._last_proposition_relation_views = {}
    agent._last_proposition_probe_coverage = {}
    agent._last_proposition_context_views = {}
    agent._last_proposition_unknown_views = {}
    return agent


def test_family_recall_expands_same_object_across_semantic_role_drift():
    agent = _agent()
    jan = {
        "id": "m_jan",
        "kind": "EVENT",
        "subject_id": "primary_user",
        "scope": "medication",
        "object_anchor": "metformin",
        "semantic_role": "GUIDANCE",
        "assertion_mode": "DIRECT",
        "claim": "Metformin 1500 mg/day was prescribed.",
        "value": "1500 mg/day",
        "event_time": "2024-01-06",
        "document_time": "2024-01-06",
        "_dense_score": 0.82,
        "_overlap": 3,
    }
    feb = {
        "id": "m_feb",
        "kind": "EVENT",
        "subject_id": "primary_user",
        "scope": "medication",
        "object_anchor": "metformin",
        "semantic_role": "OBSERVATION",
        "assertion_mode": "DIRECT",
        "claim": "Patient is taking metformin 1500 mg/day.",
        "value": "1500 mg/day",
        "event_time": "2024-02-18",
        "document_time": "2024-02-18",
        "_dense_score": 0.91,
        "_overlap": 4,
    }
    agent._memories = [jan, feb]
    agent._memory_satisfies_frame = lambda memory, frame, **kwargs: True
    agent._query_visible_memory = lambda memory, include_history=False: True
    agent._hybrid_search = lambda query, top_k, candidate_ids=None: [feb, jan]
    found = agent._locate_temporal_family(
        "metformin 1500 mg/day use", QueryFrame(), "event_time"
    )
    assert [item["id"] for item in found[:2]] == ["m_jan", "m_feb"]


def test_family_identity_does_not_merge_same_object_across_scopes():
    agent = _agent()
    medication = {
        "id": "m_med",
        "subject_id": "primary_user",
        "scope": "medication",
        "object_anchor": "cefuroxime",
    }
    allergy = {
        "id": "m_allergy",
        "subject_id": "primary_user",
        "scope": "allergy",
        "object_anchor": "cefuroxime",
    }
    assert agent._family_identity(medication) != agent._family_identity(allergy)


def test_context_owner_reserves_requirement_and_each_proposition_before_fill():
    agent = _agent()
    agent._last_requirement_context_candidates = {"r1": ["m_req", "m_req2"]}
    agent._last_requirement_proof_support = {"r1": ["m_req"]}
    agent._last_requirement_status = {"r1": "FOUND"}
    agent._last_candidate_propositions = {"A": "candidate A", "B": "candidate B"}
    agent._last_proposition_relation_views = {
        "A": [{"memory_id": "m_a", "relation": "SUPPORTS", "accepted": True}],
        "B": [{"memory_id": "m_b", "relation": "UNKNOWN", "accepted": False}],
    }
    agent._last_proposition_probe_coverage = {"A": ["m_a"], "B": ["m_b"]}
    slots = [{"id": "r1", "evidence_role": "REQUIREMENT"}]
    selected = agent._role_aware_support_ids(
        slots,
        {"r1": ["m_req"]},
        ["m_req", "m_req2", "m_a", "m_b", "m_extra"],
        4,
    )
    assert selected[:3] == ["m_req", "m_a", "m_b"]
    assert len(selected) == 4


def test_proposition_context_injection_keeps_only_authorized_boundary_ids():
    agent = _agent()
    agent._last_candidate_propositions = {"A": "candidate A"}
    agent._last_proposition_relation_views = {
        "A": [
            {"memory_id": "m_op", "relation": "SUPPORTS", "accepted": True},
            {"memory_id": "m_seed", "relation": "CONTEXT_FOR", "accepted": True},
            {"memory_id": "m_unsafe", "relation": "UNKNOWN", "accepted": False},
        ]
    }
    agent._last_proposition_probe_coverage = {
        "A": ["m_op", "m_seed", "m_unsafe"]
    }
    run = {
        "operation_output_ids": {"m_op"},
        "planning_seeds": [{"id": "m_seed"}],
    }
    assert agent._safe_proposition_context_ids(run, []) == ["m_op", "m_seed"]


def test_evidence_lineage_preserves_family_and_selector_views_for_requirement():
    agent = _agent()
    agent._last_candidate_propositions = {}
    agent._last_proposition_relation_views = {}
    agent._last_proposition_probe_coverage = {}
    run = {
        "plan": {"required_slots": [{"id": "r1"}]},
        "trace": [
            {
                "retrieval_round": 1,
                "operation": "LOCATE_ANCHOR",
                "produces": ["r1"],
                "output_ids": ["m_jan", "m_feb"],
            },
            {
                "retrieval_round": 2,
                "operation": "TEMPORAL_FILTER",
                "produces": ["r1"],
                "output_ids": ["m_jan"],
            },
        ],
        "requirement_status": {"r1": "FOUND"},
        "requirement_context_candidates": {"r1": ["m_jan", "m_feb"]},
        "requirement_proof_support": {"r1": ["m_jan"]},
        "slot_support": {"r1": ["m_jan"]},
    }
    lineage = agent._build_evidence_lineage(run, [])
    assert [
        view["operation"] for view in lineage["requirements"]["r1"]["views"]
    ] == ["LOCATE_ANCHOR", "TEMPORAL_FILTER"]
    assert lineage["memories"]["m_jan"]["requirement_ids"] == ["r1"]


def test_candidate_proposition_context_is_not_multiple_choice_specific():
    agent = _agent()
    agent._last_candidate_propositions = {
        "diet": "diet explains the observation",
        "medication": "medication explains the observation",
    }
    agent._last_proposition_relation_views = {
        "diet": [
            {"memory_id": "m1", "relation": "CONTEXT_FOR", "accepted": True}
        ],
        "medication": [
            {"memory_id": "m2", "relation": "SUPPORTS", "accepted": True}
        ],
    }
    agent._last_proposition_probe_coverage = {
        "diet": ["m1"],
        "medication": ["m2"],
    }
    priority = agent._proposition_priority_map()
    assert priority == {"diet": ["m1"], "medication": ["m2"]}
