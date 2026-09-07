"""CandidateSet regressions for SmartMem0 READ."""

from methods.smart_mem0.agent import SmartMem0Agent
from methods.smart_mem0.contracts import QueryFrame
from methods.smart_mem0.read_answer_or_plan_contract import (
    ANSWER_OR_PLAN_PRIORITY,
    MINIMAL_CONTROLLER_POLICY,
    MINIMAL_CONTROLLER_SCHEMA,
)


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


def _memory(memory_id, claim, value):
    return {
        "id": memory_id,
        "subject_id": "primary_user",
        "claim": claim,
        "value": value,
        "kind": "FACT",
        "object_anchor": "",
        "event_time": "",
        "document_time": "",
    }


def test_candidate_set_recall_is_generic_and_keeps_seed_budget_at_three():
    agent = _agent()
    agent._memories = [
        _memory("seed-memory", "Alpha evidence already in a seed.", "alpha"),
        _memory("m1", "Alpha candidate evidence one.", "alpha one"),
        _memory("m2", "Alpha candidate evidence two.", "alpha two"),
        _memory("m3", "Beta candidate evidence one.", "beta one"),
        _memory("m4", "Beta candidate evidence two.", "beta two"),
    ]
    agent._memory_satisfies_frame = (
        lambda memory, frame, include_entities=False: True
    )
    agent._query_visible_memory = lambda memory: True

    def hybrid(query, top_k, candidate_ids=None):
        words = query.casefold()
        ranked = [
            memory
            for memory in agent._memories
            if memory["id"] in set(candidate_ids or [])
            and (
                ("alpha" in words and "alpha" in memory["claim"].casefold())
                or ("beta" in words and "beta" in memory["claim"].casefold())
            )
        ]
        return ranked[:top_k]

    agent._hybrid_search = hybrid
    propositions = {
        "hypothesis_alpha": "alpha",
        "hypothesis_beta": "beta",
    }
    pack = agent._build_candidate_proposition_pack(
        propositions,
        frame=QueryFrame(),
        seeds=[{"id": "seed-memory"}, {"id": "s2"}, {"id": "s3"}],
    )

    assert pack["version"] == "candidate-set-recall-v1"
    assert pack["limits"]["seed_budget"] == 3
    assert pack["limits"]["top_per_candidate"] == 2
    assert len(pack["retrieval_views"]) <= 4
    assert all(
        item["memory_id"] != "seed-memory"
        for item in pack["retrieval_views"]
    )
    assert set(pack["candidate_refs"]) == {
        "hypothesis_alpha",
        "hypothesis_beta",
    }


def test_visible_options_are_only_an_adapter_to_candidate_set():
    agent = _agent()
    propositions = agent._candidate_propositions_from_visible_options(
        {"A": "choice one", "B": "choice two"}
    )
    assert propositions == {"A": "choice one", "B": "choice two"}
    generic = agent._normalize_candidate_propositions(
        {"hypothesis": "candidate explanation"}
    )
    assert generic == {"hypothesis": "candidate explanation"}


def test_llm1_schema_has_no_candidate_verdict_or_confidence_surface():
    assert ANSWER_OR_PLAN_PRIORITY == MINIMAL_CONTROLLER_POLICY
    for forbidden in (
        "proposition_evidence",
        "option_semantics",
        "support_roles",
        "contradict_roles",
        '"answer_mode"',
        '"requires_inference"',
        '"route"',
        '"operations"',
    ):
        assert forbidden not in MINIMAL_CONTROLLER_SCHEMA
    assert "$prop" not in MINIMAL_CONTROLLER_SCHEMA


def test_candidate_packet_is_not_part_of_controller_seed_payload():
    agent = _agent()
    seeds = [
        {
            "id": "m1",
            "claim": "A compact claim",
            "value": "v",
            "kind": "FACT",
            "subject_id": "primary_user",
            "object_anchor": "object",
            "evidence_family": "family",
            "event_time": "2024-01-01",
            "document_time": "2024-01-02",
            "origin_document_time": "2023-12-31",
            "state_identity": "legacy-heavy-field",
            "_status": "active",
        }
    ]
    payload = agent._controller_seed_payload(seeds)
    assert len(payload) == 1
    assert payload[0]["ref"] == "$seed0"
    assert "origin_document_time" not in payload[0]
    assert "state_identity" not in payload[0]
    assert "status" not in payload[0]
    assert "proposition_ids" not in payload[0]


def test_shared_candidate_search_updates_generic_coverage_without_verdicts():
    agent = _agent()
    agent._memories = [
        _memory("m1", "Alpha evidence", "alpha"),
        _memory("m2", "Beta evidence", "beta"),
        _memory("m3", "General context", "context"),
    ]
    agent._memory_satisfies_frame = (
        lambda memory, frame, include_entities=False: True
    )
    agent._query_visible_memory = lambda memory: True
    agent._snapshot = lambda memory: dict(memory)

    def hybrid(query, top_k, candidate_ids=None):
        eligible = [
            memory
            for memory in agent._memories
            if memory["id"] in set(candidate_ids or [])
        ]
        lowered = query.casefold()
        if "alpha" in lowered:
            eligible = [
                memory
                for memory in eligible
                if "alpha" in memory["claim"].casefold()
            ]
        elif "beta" in lowered:
            eligible = [
                memory
                for memory in eligible
                if "beta" in memory["claim"].casefold()
            ]
        return eligible[:top_k]

    agent._hybrid_search = hybrid
    result = agent._semantic_operation_search(
        "shared evidence",
        4,
        "SHARED_OPTIONS",
        frame=QueryFrame(),
        option_queries=[
            {"label": "A", "query": "alpha"},
            {"label": "B", "query": "beta"},
        ],
    )

    assert result
    assert agent._last_option_probe_coverage["A"] == ["m1"]
    assert agent._last_option_probe_coverage["B"] == ["m2"]
    assert agent._last_proposition_probe_coverage == (
        agent._last_option_probe_coverage
    )
    assert agent._last_proposition_relation_views == {}


def test_option_zero_memory_hit_is_not_a_false_verdict():
    agent = _agent()
    agent._last_option_probe_coverage = {
        "A": [],
        "B": ["m1"],
        "C": [],
        "D": [],
    }
    slot = {
        "evidence_role": "OPTION_CONTEXT",
        "option_labels": ["A", "B", "C", "D"],
    }
    assert agent._slot_covered(slot, ["m1"], [{"id": "m1"}], [])
    assert set(agent._last_option_probe_coverage) == {"A", "B", "C", "D"}


def test_legacy_option_relation_lookup_never_infers_stance():
    agent = _agent()
    relation = agent._option_memory_relation(
        "candidate A", {"id": "m1"}, rank=0, semantics={}
    )
    assert relation == {
        "memory_id": "m1",
        "relation": "UNJUDGED",
        "accepted": False,
        "status": "CANDIDATESET_RETRIEVAL_ONLY",
    }
