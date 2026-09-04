"""Offline regression tests for SmartMem0's two-stage read invariants."""

from copy import deepcopy
from types import SimpleNamespace

from methods.smart_mem0.contracts import QueryFrame
from methods.smart_mem0.read_architecture_contract import ReadArchitectureContractMixin
from methods.smart_mem0.read_controller import ReadContractMixin
from methods.smart_mem0.read_option_contract import ReadOptionContractMixin
from methods.smart_mem0.read_plan_contract import ReadPlanContractMixin
from methods.smart_mem0.read_route_contract import ReadRouteContractMixin
from methods.smart_mem0.read_usage_contract import ReadUsageContractMixin


class _ArchitectureBase:
    @staticmethod
    def _rc_terms(value):
        import re
        return [x.lower() for x in re.findall(r"\w+", str(value or "")) if len(x) > 1]

    @classmethod
    def _rc_content_terms(cls, value):
        return [x for x in cls._rc_terms(value) if x not in {"the", "current", "condition"}]

    @staticmethod
    def _rc_memory_target_text(memory):
        return " ".join(str(memory.get(k) or "") for k in ("claim", "value", "state_key", "object_anchor"))

    def _rc_normalize_decision(self, parsed, question, frame):
        return deepcopy(parsed)

    def _authorize_controller_answer(self, decision, seeds, frame):
        return ["ok"], "AUTHORIZED"

    def _rc_gap_from_req(self, decision, req, fallback_id, frame=None):
        return SimpleNamespace(
            role=req.get("role", "ANSWER"),
            temporal_axis=req.get("temporal_axis", ""),
            temporal_relation=req.get("temporal_relation", ""),
            temporal_anchor=req.get("temporal_anchor", ""),
            temporal_end=req.get("temporal_end", ""),
            required_fields=[],
        )


class _ArchitectureHarness(ReadArchitectureContractMixin, _ArchitectureBase):
    pass


class _RouteBase:
    def _controller_plan(self, decision, question, frame):
        return deepcopy(self.plan)


class _RouteHarness(ReadRouteContractMixin, _RouteBase):
    pass


class _OptionBase:
    def _slot_covered(self, slot, support_ids, selected, relations):
        return False


class _OptionHarness(ReadOptionContractMixin, _OptionBase):
    pass


class _UsageBase:
    def prepare_batch_query(self, question, system_message=None, **kwargs):
        return deepcopy(self.prepared)


class _UsageHarness(ReadUsageContractMixin, _UsageBase):
    pass


class _PlanBase:
    _relations = []

    @staticmethod
    def _rc_resolve_target_keys(target, subject_id=""):
        return []

    @staticmethod
    def _memory_satisfies_frame(memory, frame):
        return True

    @staticmethod
    def _slot_covered(slot, support_ids, selected, relations):
        return set(slot.get("controller_seed_ids") or []).issubset(support_ids)

    @staticmethod
    def _compile_gap_operations(slots, question, budget_tier="MEDIUM", plan=None):
        return [
            {"op": "SEMANTIC_SEARCH", "produces": [slot["id"]]}
            for slot in slots
        ]


class _PlanHarness(ReadPlanContractMixin, _PlanBase):
    pass


def test_target_proof_uses_token_boundaries_not_substrings():
    h = _ArchitectureHarness()
    slot = {"target_surface": "rent"}
    assert not h._rc_memory_matches_target(slot, {"claim": "The current condition improved."})
    assert h._rc_memory_matches_target(slot, {"claim": "Monthly rent is 6,800 yuan."})


def test_direct_redundant_requirements_do_not_become_multihop_recovery():
    h = _ArchitectureHarness()
    decision = h._rc_normalize_decision(
        {
            "operator": "DIRECT",
            "requires_inference": True,
            "requirements": [
                {"id": "a", "role": "ANSWER"},
                {"id": "b", "role": "PRIOR_TRAJECTORY"},
            ],
        },
        "question",
        QueryFrame(),
    )
    assert decision["operator"] == "DIRECT"
    assert decision["requires_inference"] is False
    assert [r["role"] for r in decision["requirements"]] == ["ANSWER"]


def test_hard_date_blocks_atomic_direct_shortcut():
    h = _ArchitectureHarness()
    support, reason = h._authorize_controller_answer({}, [], QueryFrame(dates=("2024-03-01",)))
    assert support is None
    assert reason == "TEMPORAL_CONSTRAINT_REQUIRES_PLAN"


def test_comparison_preserves_explicit_sides_and_maps_dates_by_side():
    h = _ArchitectureHarness()
    decision = h._rc_normalize_decision(
        {
            "operator": "COMPARISON",
            "requirements": [
                {"id": "r", "role": "COMPARAND", "side": "RIGHT", "target": "March"},
                {"id": "l", "role": "COMPARAND", "side": "LEFT", "target": "January"},
            ],
        },
        "question",
        QueryFrame(dates=("*-01", "*-03")),
    )
    left, right = decision["requirements"]
    assert (left["side"], left["target"], left["temporal_anchor"]) == ("LEFT", "January", "*-01")
    assert (right["side"], right["target"], right["temporal_anchor"]) == ("RIGHT", "March", "*-03")


def test_ambiguous_exact_date_keeps_query_frame_as_constraint_owner():
    h = _ArchitectureHarness()
    gap = h._rc_gap_from_req(
        {"operator": "STATE"},
        {"role": "ANSWER", "temporal_relation": "EXACT", "temporal_anchor": "2024-03-01"},
        "g",
        frame=QueryFrame(dates=("2024-03-01",)),
    )
    assert gap.temporal_axis == ""
    assert gap.temporal_relation == ""
    assert gap.temporal_anchor == ""
    assert gap.required_fields == []


def test_comparand_gets_side_local_effective_event_axis():
    h = _ArchitectureHarness()
    gap = h._rc_gap_from_req(
        {"operator": "COMPARISON"},
        {"role": "COMPARAND", "temporal_relation": "EXACT", "temporal_anchor": "*-01"},
        "g",
        frame=QueryFrame(dates=("*-01", "*-03")),
    )
    assert gap.temporal_axis == "effective_event_time"
    assert gap.required_fields == ["effective_event_time"]


def test_direct_release_lock_does_not_inflate_small_budget():
    h = _RouteHarness()
    h.plan = {
        "query_mode": "DIRECT",
        "required_slots": [{"id": "r1", "type": "DIRECT"}],
        "budget_tier": "SMALL",
        "max_memories": 3,
        "query_spec": {},
    }
    plan = h._controller_plan({"operator": "DIRECT", "route": "PLAN"}, "q", QueryFrame())
    assert plan["route_lock_released"] is True
    assert plan["active_strategy"] == "FOCAL"
    assert plan["budget_tier"] == "SMALL"
    assert plan["max_memories"] == 3


def test_option_exploration_without_surviving_evidence_is_not_sufficient():
    h = _OptionHarness()
    h._last_option_probe_coverage = {"A": [], "B": ["m1"], "C": [], "D": []}
    slot = {"evidence_role": "OPTION_CONTEXT", "option_labels": ["A", "B", "C", "D"]}
    assert not h._slot_covered(slot, [], [], [])
    assert h._slot_covered(slot, ["m1"], [{"id": "m1"}], [])


def test_two_stage_usage_has_hard_two_call_budget_and_flags_middle_call():
    h = _UsageHarness()
    h.prepared = {
        "precomputed_answer": "",
        "extra": {
            "semantic_controller": {"called": True},
            "planner_called": True,
            "replan_called": False,
            "slot_validation": [],
            "query_tokens": {"planner": 100, "replan": 0},
        },
    }
    prepared = h.prepare_batch_query("q")
    calls = prepared["extra"]["method_llm_calls"]
    assert calls["total"] == 2
    assert calls["middle"] == 0
    assert calls["two_stage_budget_violation"] is False
    assert prepared["extra"]["deterministic_plan_compiled"] is True
    assert prepared["extra"]["planner_llm_called"] is False

    h.prepared["extra"]["slot_validation"] = [{"called": True}]
    prepared = h.prepare_batch_query("q")
    calls = prepared["extra"]["method_llm_calls"]
    assert calls["total"] == 3
    assert calls["middle"] == 1
    assert calls["two_stage_budget_violation"] is True


def test_controller_requirement_coverage_accepts_only_valid_seed_refs():
    h = ReadContractMixin()
    requirement = h._rc_normalize_requirement(
        {
            "id": "r1",
            "role": "FOCAL_STATE",
            "target": "blood glucose",
            "coverage": "COVERED",
            "support_refs": ["$seed0", "$seed9", "m_12"],
            "world_knowledge_bridge": True,
        },
        0,
        "What happened to blood glucose?",
        "",
    )
    assert requirement["coverage"] == "COVERED"
    assert requirement["support_refs"] == ["$seed0"]
    assert requirement["world_knowledge_bridge"] is True


def test_controller_requirement_without_valid_support_is_missing():
    h = ReadContractMixin()
    requirement = h._rc_normalize_requirement(
        {
            "coverage": "COVERED",
            "support_refs": ["$0", "m_1"],
        },
        0,
        "question",
        "",
    )
    assert requirement["coverage"] == "MISSING"
    assert requirement["support_refs"] == []


def test_covered_controller_requirement_compiles_zero_operation_plan():
    h = _PlanHarness()
    decision = {
        "operator": "DIRECT",
        "answer_slot": "VALUE",
        "requires_inference": True,
        "subject_id": "",
        "target": "",
        "causal_mode": "",
        "need_raw_evidence": False,
        "temporal": {},
        "visible_options": {},
        "requirements": [
            {
                "id": "r1",
                "role": "ANSWER",
                "coverage": "COVERED",
                "support_refs": ["$seed0"],
                "world_knowledge_bridge": True,
            }
        ],
        "_seed_candidates": [{"id": "m1", "value": "supported value"}],
    }
    plan = h._controller_plan(decision, "question", QueryFrame())
    assert plan["operations"] == []
    assert plan["seed_coverage"] == [{"slot_id": "r1", "refs": ["$seed0"]}]
    assert plan["controller_coverage"]["missing_count"] == 0
    assert plan["query_spec"]["world_knowledge_bridge_allowed"] is True
