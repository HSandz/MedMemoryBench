"""Regressions for capability restored after the semantic-ownership refactor."""

from methods.smart_mem0.agent import SmartMem0Agent
from methods.smart_mem0.contracts import QueryFrame
from methods.smart_mem0.proof_context_contract import ProofContextContractMixin


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


def test_non_time_mentioned_question_does_not_become_temporal_locate():
    agent = _agent()
    constraint = agent._rq_repair_time_constraint(
        {
            "axis": "document_time",
            "relation": "LOCATE",
            "anchor": "",
            "end": "",
        },
        "What chronic metabolic disease is mentioned in the patient's past medical history?",
        "ENTITY",
    )
    assert constraint == {"axis": "", "relation": "", "anchor": "", "end": ""}


def test_latest_selector_survives_for_value_answer():
    agent = _agent()
    constraint = agent._rq_repair_time_constraint(
        {"axis": "", "relation": "", "anchor": "", "end": ""},
        "What was the latest HbA1c result?",
        "VALUE",
    )
    assert constraint["axis"] == "event_time"
    assert constraint["relation"] == "LATEST"


def test_exact_date_filter_survives_for_scalar_answer():
    agent = _agent()
    constraint = agent._rq_repair_time_constraint(
        {
            "axis": "event_time",
            "relation": "EXACT",
            "anchor": "2024-03-20",
            "end": "",
        },
        "What was the exact blood pH value measured on 2024-03-20?",
        "VALUE",
    )
    assert constraint["axis"] == "event_time"
    assert constraint["relation"] == "EXACT"
    assert constraint["anchor"] == "2024-03-20"


def test_raw_grounded_candidate_is_not_erased_by_degraded_plan_fields():
    agent = _agent()
    ir = agent._rc_normalize_ir(
        {"candidate": {"answer": "7.28", "support_ref": "$seed0"}},
        "What was the exact blood pH value measured in the emergency department on 2024-03-20?",
        QueryFrame(dates=("2024-03-20",)),
    )
    assert ir["normalization_status"] == "DEGRADED"
    assert ir["candidate"] == {"answer": "7.28", "support_ref": "$seed0"}


def test_symbolic_atomic_surface_can_be_terminally_grounded():
    agent = _agent()
    memory = {
        "claim": "Urine ketones were reported as ++.",
        "value": "++",
        "verbatim_value": "++",
        "object_anchor": "urine_ketones",
        "entities": ["urine ketones"],
        "scope_entities": [],
    }
    assert agent._terminal_answer_grounded("++", memory)


class _RankingHarness(ProofContextContractMixin):
    def _context_rank_signals(self, slot, memory, relations):
        return memory["signals"]


def test_answer_relevance_outranks_recovery_recency_in_context():
    harness = _RankingHarness()
    ordered = ["new_irrelevant", "old_relevant"]
    memories = {
        "new_irrelevant": {
            "signals": {
                "structural": True,
                "certificate": False,
                "resolved_key_match": False,
                "target_score": 0.1,
                "answer_bearing": True,
                "retrieval_score": 0.9,
            }
        },
        "old_relevant": {
            "signals": {
                "structural": True,
                "certificate": False,
                "resolved_key_match": True,
                "target_score": 0.9,
                "answer_bearing": True,
                "retrieval_score": 0.5,
            }
        },
    }
    harness._rank_context_candidates({}, ordered, memories, [])
    assert ordered == ["old_relevant", "new_irrelevant"]


def test_query_type_remains_behaviorally_inert():
    agent = _agent()
    assert agent._compact_reasoning_output_instruction("multi_hop_clinical_deduction") == ""
    assert agent._compact_reasoning_output_instruction("entity_exact_match") == ""
