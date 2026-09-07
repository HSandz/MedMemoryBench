"""Regressions for general, language-neutral SmartMem0 read behavior."""

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


def test_non_time_locate_is_removed_without_parsing_question_language():
    agent = _agent()
    constraint = agent._rq_repair_time_constraint({"axis": "document_time", "relation": "LOCATE", "anchor": "", "end": ""}, "Bệnh chuyển hóa mạn tính nào được nhắc trong tiền sử?", "ENTITY")
    assert constraint == {"axis": "", "relation": "", "anchor": "", "end": ""}


def test_controller_selected_latest_survives_for_value_answer_in_any_language():
    agent = _agent()
    constraint = agent._rq_repair_time_constraint({"axis": "event_time", "relation": "LATEST", "anchor": "", "end": ""}, "Kết quả HbA1c gần nhất là bao nhiêu?", "VALUE")
    assert constraint["axis"] == "event_time"
    assert constraint["relation"] == "LATEST"


def test_controller_selected_earliest_survives_for_date_answer_in_vietnamese():
    agent = _agent()
    constraint = agent._rq_repair_time_constraint({"axis": "event_time", "relation": "EARLIEST", "anchor": "", "end": ""}, "Bệnh nhân bắt đầu dùng metformin 1500 mg mỗi ngày khi nào?", "DATE")
    assert constraint == {"axis": "event_time", "relation": "EARLIEST", "anchor": "", "end": ""}


def test_controller_selected_latest_survives_for_chinese_text():
    agent = _agent()
    constraint = agent._rq_repair_time_constraint({"axis": "event_time", "relation": "LATEST", "anchor": "", "end": ""}, "最近一次空腹血糖结果是什么？", "VALUE")
    assert constraint["relation"] == "LATEST"


def test_exact_date_filter_survives_for_scalar_answer():
    agent = _agent()
    constraint = agent._rq_repair_time_constraint({"axis": "event_time", "relation": "EXACT", "anchor": "2024-03-20", "end": ""}, "pH máu đo ngày 2024-03-20 là bao nhiêu?", "VALUE")
    assert constraint["axis"] == "event_time"
    assert constraint["relation"] == "EXACT"
    assert constraint["anchor"] == "2024-03-20"


def test_resolved_keys_require_target_coverage_not_generic_key_precision():
    agent = _agent()
    agent._memories = [{"subject_id": "primary_user", "scope": "medication", "state_key": "must_avoid_medication", "object_anchor": "nsaids", "entities": ["NSAIDs"], "scope_entities": []}]
    keys = agent._rc_resolve_target_keys("patient avoided antibiotic instruction contraindication history", "primary_user")
    assert "medication" not in [agent._rc_text(key) for key in keys]


def test_resolved_keys_can_capture_high_coverage_canonical_identity():
    agent = _agent()
    agent._memories = [{"subject_id": "primary_user", "scope": "symptom", "state_key": "nighttime_thirst", "object_anchor": "nighttime_thirst", "entities": ["nighttime thirst"], "scope_entities": []}]
    keys = agent._rc_resolve_target_keys("nighttime thirst", "primary_user")
    assert any(agent._rc_text(key) == "nighttime thirst" for key in keys)


def test_unicode_surface_similarity_does_not_depend_on_english_stopwords():
    agent = _agent()
    assert agent._rq_surface_similarity("夜间口渴", "患者最近出现夜间口渴并醒来喝水") == 1.0
    assert agent._rq_surface_similarity("khát nước ban đêm", "Bệnh nhân thường bị khát nước ban đêm.") == 1.0


def test_raw_grounded_candidate_is_not_erased_by_degraded_plan_fields():
    agent = _agent()
    ir = agent._rc_normalize_ir({"candidate": {"answer": "7.28", "support_ref": "$seed0"}}, "What was the exact blood pH value measured in the emergency department on 2024-03-20?", QueryFrame(dates=("2024-03-20",)))
    assert ir["normalization_status"] == "DEGRADED"
    assert ir["candidate"] == {"answer": "7.28", "support_ref": "$seed0"}


def test_symbolic_atomic_surface_can_be_terminally_grounded():
    agent = _agent()
    memory = {"claim": "Urine ketones were reported as ++.", "value": "++", "verbatim_value": "++", "object_anchor": "urine_ketones", "entities": ["urine ketones"], "scope_entities": []}
    assert agent._terminal_answer_grounded("++", memory)


def test_non_english_proposition_can_be_terminally_grounded():
    agent = _agent()
    memory = {"claim": "Cân nặng của bệnh nhân đã ổn định và không giảm thêm.", "value": "ổn định", "verbatim_value": "ổn định", "object_anchor": "cân nặng", "entities": ["cân nặng"], "scope_entities": []}
    assert agent._terminal_answer_grounded("Cân nặng của bệnh nhân đã ổn định và không giảm thêm.", memory)


class _RankingHarness(ProofContextContractMixin):
    def _context_promotion_level(self, slot, memory):
        return int(memory.get("promotion", 0))


def test_view_aware_context_keeps_top_recovery_evidence_without_global_rescore():
    harness = _RankingHarness()
    ordered = ["round2_answer", "round2_other", "round1_a", "round1_b", "round1_c"]
    memories = {memory_id: {"promotion": 0} for memory_id in ordered}
    views = [["round2_answer", "round2_other"], ["round1_a", "round1_b", "round1_c"]]
    harness._rank_context_candidates({}, ordered, memories, [], views=views)
    assert ordered[:3] == ["round2_answer", "round1_a", "round2_other"]


def test_strict_positive_signal_may_promote_but_weak_scores_do_not_resort_views():
    harness = _RankingHarness()
    ordered = ["recovery_top", "initial_top", "certified"]
    memories = {"recovery_top": {"promotion": 0}, "initial_top": {"promotion": 0}, "certified": {"promotion": 3}}
    views = [["recovery_top"], ["initial_top", "certified"]]
    harness._rank_context_candidates({}, ordered, memories, [], views=views)
    assert ordered == ["certified", "recovery_top", "initial_top"]


def test_comparand_is_a_semantic_context_slot_not_legacy_context():
    assert ProofContextContractMixin._semantic_context_slot({"evidence_role": "COMPARAND"})
    assert ProofContextContractMixin._semantic_context_slot({"evidence_role": "REQUIREMENT"})
    assert not ProofContextContractMixin._semantic_context_slot({"evidence_role": "OPTION_CONTEXT"})


def test_direct_gate_is_canonical_ir_driven_not_english_keyword_driven():
    agent = _agent()
    ir = {"answer_type": "TEXT", "visible_options": {}, "relations": []}
    assert agent._aop_direct_surface_allowed("Why should I do this?", ir)
    assert agent._aop_direct_surface_allowed("Tại sao tôi nên làm vậy?", ir)
    ir["relations"] = [{"type": "INFER", "from": "r1", "to": "ANSWER"}]
    assert not agent._aop_direct_surface_allowed("Why should I do this?", ir)


def test_relative_time_never_direct_even_if_candidate_text_is_grounded():
    agent = _agent()
    ir = {"answer_type": "RELATIVE_TIME", "visible_options": {}, "relations": []}
    assert not agent._aop_direct_surface_allowed("Uống cà phê sữa sau bữa ăn bao lâu?", ir)


def test_option_zero_memory_hit_is_not_a_false_verdict():
    agent = _agent()
    agent._last_option_probe_coverage = {"A": [], "B": ["m1"], "C": [], "D": []}
    slot = {"evidence_role": "OPTION_CONTEXT", "option_labels": ["A", "B", "C", "D"]}
    assert agent._slot_covered(slot, ["m1"], [{"id": "m1"}], [])
    # Empty A/C/D views are allowed; they mean no personal-memory support found.
    assert set(agent._last_option_probe_coverage) == {"A", "B", "C", "D"}


def test_query_type_remains_behaviorally_inert():
    agent = _agent()
    assert agent._compact_reasoning_output_instruction("multi_hop_clinical_deduction") == ""
    assert agent._compact_reasoning_output_instruction("entity_exact_match") == ""
