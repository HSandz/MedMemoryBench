"""Candidate-proposition regressions for SmartMem0 READ."""

from methods.smart_mem0.agent import SmartMem0Agent
from methods.smart_mem0.contracts import QueryFrame
from methods.smart_mem0.read_answer_or_plan_contract import ANSWER_OR_PLAN_PRIORITY


def _agent():
    agent = object.__new__(SmartMem0Agent)
    agent._belief_status = {}
    agent._state_heads = {}
    agent._memories = []
    agent._relations = []
    agent.subject_aliases = {"patient": "primary_user"}
    agent._last_candidate_propositions = {}
    agent._last_candidate_proposition_pack = {}
    agent._last_proposition_probe_coverage = {}
    agent._last_proposition_relation_views = {}
    agent._last_proposition_support_views = {}
    agent._last_proposition_contradict_views = {}
    agent._last_proposition_context_views = {}
    agent._last_proposition_unknown_views = {}
    agent._last_proposition_semantics = {}
    agent._last_option_probe_coverage = {}
    agent._last_option_probe_relations = {}
    agent._last_option_support_views = {}
    agent._last_option_contradict_views = {}
    agent._last_option_semantics = {}
    return agent


def test_candidate_proposition_pack_is_generic_and_keeps_seed_budget_at_three():
    agent = _agent()
    agent._memories = [
        {"id": "seed-memory", "subject_id": "primary_user", "claim": "Alpha evidence already in a seed.", "value": "alpha"},
        {"id": "m1", "subject_id": "primary_user", "claim": "Alpha candidate evidence one.", "value": "alpha one"},
        {"id": "m2", "subject_id": "primary_user", "claim": "Alpha candidate evidence two.", "value": "alpha two"},
        {"id": "m3", "subject_id": "primary_user", "claim": "Beta candidate evidence one.", "value": "beta one"},
        {"id": "m4", "subject_id": "primary_user", "claim": "Beta candidate evidence two.", "value": "beta two"},
    ]
    agent._memory_satisfies_frame = lambda memory, frame, include_entities=False: True
    agent._query_visible_memory = lambda memory: True

    def hybrid(query, top_k, candidate_ids=None):
        words = query.casefold()
        ranked = [
            memory
            for memory in agent._memories
            if memory["id"] in set(candidate_ids or [])
            and (("alpha" in words and "alpha" in memory["claim"].casefold()) or ("beta" in words and "beta" in memory["claim"].casefold()))
        ]
        return ranked[:top_k]

    agent._hybrid_search = hybrid
    propositions = {"hypothesis_alpha": "alpha", "hypothesis_beta": "beta"}
    pack = agent._build_candidate_proposition_pack(
        propositions,
        frame=QueryFrame(),
        seeds=[{"id": "seed-memory"}, {"id": "s2"}, {"id": "s3"}],
    )
    assert pack["version"] == "candidate-proposition-pack-v1"
    assert pack["limits"]["seed_budget"] == 3
    assert pack["limits"]["top_per_proposition"] == 2
    assert len(pack["candidates"]) <= 4
    assert all(item["memory_id"] != "seed-memory" for item in pack["candidates"])
    assert set(pack["candidate_refs"]) == {"hypothesis_alpha", "hypothesis_beta"}


def test_low_confidence_relation_abstains_instead_of_becoming_support():
    agent = _agent()
    pack = {"candidate_refs": {"A": ["$prop0"]}, "candidates": [{"ref": "$prop0", "memory_id": "m1"}]}
    normalized = agent._normalize_candidate_proposition_evidence(
        {"A": [{"memory_ref": "$prop0", "relation": "SUPPORTS", "confidence": 0.49}]},
        pack,
        [],
        {"A": "candidate A"},
    )
    item = normalized["A"][0]
    assert item["proposed_relation"] == "SUPPORTS"
    assert item["relation"] == "UNKNOWN"
    assert item["status"] == "ABSTAIN_LOW_CONFIDENCE"
    assert not item["accepted"]


def test_tentative_strong_relation_abstains_below_strong_threshold():
    agent = _agent()
    pack = {"candidate_refs": {"A": ["$prop0"]}, "candidates": [{"ref": "$prop0", "memory_id": "m1"}]}
    normalized = agent._normalize_candidate_proposition_evidence(
        {"A": [{"memory_ref": "$prop0", "relation": "CONTRADICTS", "confidence": 0.65}]},
        pack,
        [],
        {"A": "candidate A"},
    )
    item = normalized["A"][0]
    assert item["proposed_relation"] == "CONTRADICTS"
    assert item["relation"] == "UNKNOWN"
    assert item["status"] == "ABSTAIN_TENTATIVE_STRONG_RELATION"


def test_high_confidence_support_is_accepted_but_is_not_a_final_verdict():
    agent = _agent()
    pack = {"candidate_refs": {"A": ["$prop0"]}, "candidates": [{"ref": "$prop0", "memory_id": "m1"}]}
    normalized = agent._normalize_candidate_proposition_evidence(
        {"A": [{"memory_ref": "$prop0", "relation": "SUPPORTS", "confidence": 0.88}]},
        pack,
        [],
        {"A": "candidate A"},
    )
    agent._activate_candidate_proposition_evidence(
        {"A": "candidate A"},
        normalized,
        visible_options=True,
        predicate="candidate satisfies the requested condition",
    )
    assert agent._last_proposition_support_views["A"][0]["memory_id"] == "m1"
    assert agent._last_proposition_support_views["A"][0]["confidence"] == 0.88
    assert "verdict" not in agent._last_proposition_support_views["A"][0]


def test_context_relation_can_survive_without_becoming_support():
    agent = _agent()
    pack = {"candidate_refs": {"A": ["$prop0"]}, "candidates": [{"ref": "$prop0", "memory_id": "m1"}]}
    normalized = agent._normalize_candidate_proposition_evidence(
        {"A": [{"memory_ref": "$prop0", "relation": "CONTEXT_FOR", "confidence": 0.58}]},
        pack,
        [],
        {"A": "candidate A"},
    )
    agent._activate_candidate_proposition_evidence({"A": "candidate A"}, normalized)
    assert agent._last_proposition_context_views["A"][0]["memory_id"] == "m1"
    assert agent._last_proposition_support_views["A"] == []


def test_packet_reference_cannot_be_reassigned_to_another_proposition():
    agent = _agent()
    pack = {
        "candidate_refs": {"A": ["$prop0"], "B": ["$prop1"]},
        "candidates": [{"ref": "$prop0", "memory_id": "m1"}, {"ref": "$prop1", "memory_id": "m2"}],
    }
    normalized = agent._normalize_candidate_proposition_evidence(
        {"B": [{"memory_ref": "$prop0", "relation": "SUPPORTS", "confidence": 0.95}]},
        pack,
        [],
        {"A": "candidate A", "B": "candidate B"},
    )
    assert normalized["B"] == []


def test_top_three_seed_refs_remain_global_for_proposition_annotation():
    agent = _agent()
    pack = {"candidate_refs": {"A": []}, "candidates": []}
    seeds = [{"id": "m1"}, {"id": "m2"}, {"id": "m3"}, {"id": "m4"}]
    normalized = agent._normalize_candidate_proposition_evidence(
        {
            "A": [
                {"memory_ref": "$seed2", "relation": "SUPPORTS", "confidence": 0.90},
                {"memory_ref": "$seed3", "relation": "SUPPORTS", "confidence": 0.99},
            ]
        },
        pack,
        seeds,
        {"A": "candidate A"},
    )
    assert [item["memory_id"] for item in normalized["A"]] == ["m3"]


def test_visible_options_are_only_an_adapter_to_candidate_propositions():
    agent = _agent()
    propositions = agent._candidate_propositions_from_visible_options({"A": "choice one", "B": "choice two"})
    assert propositions == {"A": "choice one", "B": "choice two"}
    generic = agent._normalize_candidate_propositions({"hypothesis": "candidate explanation"})
    assert generic == {"hypothesis": "candidate explanation"}


def test_controller_policy_keeps_llm1_annotation_distinct_from_final_selection():
    assert "Candidate-proposition pack entries are additional bounded retrieval candidates, NOT extra seeds" in ANSWER_OR_PLAN_PRIORITY
    assert "confidence means P(the memory↔proposition RELATION LABEL is correct)" in ANSWER_OR_PLAN_PRIORITY
    assert "Do NOT choose final proposition labels in LLM #1" in ANSWER_OR_PLAN_PRIORITY


def test_option_zero_memory_hit_is_not_a_false_verdict():
    agent = _agent()
    agent._last_option_probe_coverage = {"A": [], "B": ["m1"], "C": [], "D": []}
    slot = {"evidence_role": "OPTION_CONTEXT", "option_labels": ["A", "B", "C", "D"]}
    assert agent._slot_covered(slot, ["m1"], [{"id": "m1"}], [])
    assert set(agent._last_option_probe_coverage) == {"A", "B", "C", "D"}
