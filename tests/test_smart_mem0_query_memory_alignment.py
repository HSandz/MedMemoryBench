from copy import deepcopy

from methods.smart_mem0.read_query_memory_alignment import ReadQueryMemoryAlignmentMixin


class _SelectionBase:
    def __init__(self):
        self._last_requirement_context_candidates = {}
        self._last_requirement_proof_support = {}
        self._last_proposition_probe_coverage = {}
        self._alignment_seed_ids = []
        self._alignment_candidate_meta = {}
        self._memories = []

    @staticmethod
    def _role_aware_support_ids(slots, slot_support, candidate_order, limit):
        del slots, slot_support
        return list(candidate_order[:limit])

    @staticmethod
    def _rc_text(value):
        return " ".join(str(value or "").lower().split())

    @staticmethod
    def _memory_value(memory):
        return memory.get("value") or memory.get("claim") or ""

    @staticmethod
    def _date_for(memory, axis):
        return memory.get(axis) or ""


class _SelectionHarness(ReadQueryMemoryAlignmentMixin, _SelectionBase):
    pass


class _ExecutionBase:
    def __init__(self):
        self.seen_plan = None

    def _execute_plan(self, plan, seeds, round_offset=0, frame=None, question=""):
        del seeds, round_offset, frame, question
        self.seen_plan = deepcopy(plan)
        return {"operation_outputs": [], "selected": []}


class _ExecutionHarness(ReadQueryMemoryAlignmentMixin, _ExecutionBase):
    pass


def test_terminal_candidate_is_structural_not_free_generated_answer():
    assert (
        ReadQueryMemoryAlignmentMixin._aop_raw_candidate(
            {"candidate": {"answer": "guessed answer", "support_ref": "$seed0"}}
        )
        is None
    )
    parsed = ReadQueryMemoryAlignmentMixin._aop_raw_candidate(
        {
            "terminal_candidate": {
                "support_ref": "$seed1",
                "projection": {"field": "value"},
            }
        }
    )
    assert parsed == {"support_ref": "$seed1", "projection": {"field": "value"}}


def test_query_shape_is_generic_evidence_topology_not_benchmark_class():
    shape = ReadQueryMemoryAlignmentMixin._derive_query_shape(
        {
            "answer_type": "VALUE",
            "requirements": [
                {"id": "r1", "time_constraint": {}},
                {"id": "r2", "time_constraint": {}},
            ],
            "relations": [{"type": "COMPARE", "from": "r1", "to": "r2"}],
        }
    )
    assert shape["reasoning_form"] == "COMPARISON"
    assert shape["requirement_count"] == 2
    assert shape["routing_semantics"] == "telemetry_and_sufficiency_only"
    assert not any(key in shape for key in ("benchmark", "dataset", "query_type"))


def test_answer_context_cap_is_topology_driven_and_bounded_four_to_eight():
    harness = _SelectionHarness()
    atomic = {"answer_type": "VALUE", "requirements": [{"id": "r1"}], "relations": []}
    complex_ir = {
        "answer_type": "TEXT",
        "requirements": [{"id": f"r{i}"} for i in range(1, 5)],
        "relations": [
            {"type": "DEPENDS_ON"},
            {"type": "TEMPORAL_ORDER"},
            {"type": "COMPARE"},
        ],
    }
    assert harness._answer_context_cap(atomic) == 4
    assert 4 <= harness._answer_context_cap(complex_ir) <= 8


def test_acquisition_budget_is_separate_from_answer_context_budget():
    harness = _ExecutionHarness()
    plan = {
        "max_memories": 4,
        "answer_context_cap": 4,
        "operations": [
            {"op": "SEMANTIC_SEARCH", "top_k": 3, "produces": ["r1"]}
        ],
    }
    original = deepcopy(plan)
    result = harness._execute_plan(plan, [], frame=None)
    assert plan == original
    assert harness.seen_plan["max_memories"] == 16
    assert harness.seen_plan["operations"][0]["top_k"] == 8
    assert result["answer_context_cap"] == 4
    assert result["candidate_world_limit"] == 16
    assert result["acquisition_separated_from_answer_context"] is True


def test_raw_lexical_and_dense_heads_get_independent_admission_rights():
    rows = [
        {"id": "semantic", "_bm25_rank": 3, "_dense_rank": 1},
        {"id": "lexical", "_bm25_rank": 1, "_dense_rank": 4},
        {"id": "dense", "_bm25_rank": 2, "_dense_rank": 2},
    ]
    heads = ReadQueryMemoryAlignmentMixin._raw_channel_heads(rows)
    assert [memory["id"] for memory in heads] == ["lexical", "semantic"]


def test_scarce_requirement_evidence_is_selected_before_redundant_global_hits():
    harness = _SelectionHarness()
    harness._last_requirement_context_candidates = {
        "r1": ["common_a", "common_b", "common_c"],
        "r2": ["unique"],
    }
    harness._alignment_candidate_meta = {
        "common_a": {"id": "common_a", "claim": "same family", "value": "A"},
        "common_b": {"id": "common_b", "claim": "same family", "value": "A"},
        "common_c": {"id": "common_c", "claim": "same family", "value": "A"},
        "unique": {"id": "unique", "claim": "only support", "value": "B"},
    }
    slots = [{"id": "r1"}, {"id": "r2"}]
    selected = harness._role_aware_support_ids(
        slots,
        {},
        ["common_a", "common_b", "common_c", "unique"],
        4,
    )
    assert selected[0] == "unique"
    assert "unique" in selected
    assert harness._last_query_memory_alignment["uncovered_lanes"] == []


def test_exact_structural_redundancy_never_replaces_unique_lane_coverage():
    harness = _SelectionHarness()
    duplicate = {
        "claim": "same",
        "subject_id": "u",
        "state_key": "k",
        "object_anchor": "o",
        "value": "1",
        "stance": "AFFIRM",
        "event_time": "2026-01-01",
        "document_time": "2026-01-01",
    }
    harness._alignment_candidate_meta = {
        "a": {"id": "a", **duplicate},
        "b": {"id": "b", **duplicate},
        "needed": {"id": "needed", "claim": "different", "value": "2"},
    }
    harness._last_requirement_context_candidates = {"r1": ["a", "b"], "r2": ["needed"]}
    selected = harness._role_aware_support_ids(
        [{"id": "r1"}, {"id": "r2"}],
        {},
        ["a", "b", "needed"],
        4,
    )
    assert "needed" in selected
    assert not ({"a", "b"} <= set(selected))
