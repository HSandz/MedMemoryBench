"""Phase-3 regressions for SmartMem0's lean retrieval program."""

from copy import deepcopy

from methods.smart_mem0.contracts import QueryFrame
from methods.smart_mem0.read_retrieval_executor import ReadRetrievalExecutorMixin


class _Base:
    HARD_MEMORY_LIMIT = 8

    def __init__(self):
        self._memories = []
        self._belief_status = {}
        self._last_option_probe_coverage = {}
        self._last_proposition_probe_coverage = {}

    @staticmethod
    def _snapshot(value):
        return deepcopy(value)

    @staticmethod
    def _memory_satisfies_frame(memory, frame, **kwargs):
        del memory, frame, kwargs
        return True

    @staticmethod
    def _query_visible_memory(memory, include_history=False):
        del memory, include_history
        return True

    def _hybrid_search(self, query, top_k, candidate_ids=None):
        candidate_ids = set(candidate_ids or [])
        query = str(query or "").casefold()
        rows = []
        for memory in self._memories:
            if memory["id"] not in candidate_ids:
                continue
            copy = deepcopy(memory)
            text = str(memory.get("claim") or "").casefold()
            copy["_overlap"] = int(any(token in text for token in query.split()))
            copy["_dense_score"] = 0.8 if copy["_overlap"] else 0.1
            rows.append(copy)
        rows.sort(
            key=lambda memory: (
                memory.get("_overlap", 0),
                memory.get("_dense_score", 0.0),
                memory["id"],
            ),
            reverse=True,
        )
        return rows[:top_k]

    def _compile_gap_operations(self, slots, question, budget_tier="MEDIUM", plan=None):
        del question, budget_tier, plan
        slot_id = slots[0]["id"] if slots else "r1"
        if slots and slots[0].get("type") == "TEMPORAL":
            return [
                {"op": "LOCATE_ANCHOR", "query": "dose", "produces": [slot_id]},
                {
                    "op": "TEMPORAL_FILTER",
                    "query": "dose",
                    "relation": "LATEST",
                    "axis": "event_time",
                    "candidate_refs": ["$0"],
                    "produces": [slot_id],
                },
            ]
        return [
            {
                "op": "SEMANTIC_SEARCH",
                "query": "family",
                "top_k": 6,
                "strategy": "FOCAL",
                "produces": [slot_id],
            }
        ]

    def _controller_plan(self, ir, question, frame):
        del question, frame
        slots = [dict(item) for item in ir.get("slots") or []]
        return {
            "required_slots": slots,
            "semantic_relations": [],
            "visible_options": dict(ir.get("visible_options") or {}),
            "query_spec": {},
            "budget_tier": "LARGE",
            "max_memories": 8,
            "operations": self._compile_gap_operations(slots, "", "LARGE"),
        }

    def _execute_operation(self, operation, outputs, seeds, frame=QueryFrame()):
        del operation, outputs, seeds, frame
        raise AssertionError("canonical SEARCH_FAMILY should execute in the facade")


class _Harness(ReadRetrievalExecutorMixin, _Base):
    pass


def _memory(memory_id, claim, *, tier="HOT", date="2024-01-01"):
    return {
        "id": memory_id,
        "claim": claim,
        "value": claim,
        "memory_tier": tier,
        "subject_id": "primary_user",
        "event_time": date,
        "document_time": date,
        "assertion_mode": "DIRECT",
    }


def test_compiler_exposes_only_lean_operation_vocabulary():
    harness = _Harness()
    direct = harness._compile_gap_operations(
        [{"id": "r1", "type": "DIRECT"}], "q", "SMALL"
    )
    assert [item["op"] for item in direct] == ["SEARCH_FAMILY"]

    temporal = harness._compile_gap_operations(
        [{"id": "r1", "type": "TEMPORAL"}], "q", "MEDIUM"
    )
    assert [item["op"] for item in temporal] == ["SEARCH_FAMILY", "SELECT"]
    assert temporal[0]["family_mode"] == "temporal_extremum"
    assert temporal[0]["axis"] == "event_time"
    assert temporal[1]["candidate_refs"] == ["$0"]


def test_budget_is_derived_from_evidence_obligation_count_not_visible_options():
    harness = _Harness()
    ir = {
        "slots": [{"id": "r1", "type": "DIRECT"}],
        "visible_options": {"A": "one", "B": "two", "C": "three", "D": "four"},
        "candidate_propositions": {"A": "one", "B": "two", "C": "three", "D": "four"},
    }
    plan = harness._controller_plan(ir, "q", QueryFrame())
    assert plan["budget_tier"] == "SMALL"
    assert plan["max_memories"] == 3
    assert plan["retrieval_budget_basis"]["evidence_obligations"] == 1
    assert plan["retrieval_budget_basis"]["candidate_count_not_budget_class"] == 4


def test_semantic_hot_view_prevents_unnecessary_cold_search():
    harness = _Harness()
    harness._memories = [
        _memory("hot", "target family evidence", tier="HOT"),
        _memory("cold", "target family older evidence", tier="COLD"),
    ]
    result = harness._search_family_hot_first(
        "target family", 4, QueryFrame()
    )
    assert [item["id"] for item in result] == ["hot"]


def test_cold_is_opened_only_after_hot_semantic_miss():
    harness = _Harness()
    harness._memories = [
        _memory("hot", "unrelated material", tier="HOT"),
        _memory("cold", "target family evidence", tier="COLD"),
    ]
    result = harness._search_family_hot_first(
        "target family", 4, QueryFrame()
    )
    assert result[0]["id"] == "cold"


def test_retrieval_view_cache_reuses_identical_family_search():
    harness = _Harness()
    harness._memories = [_memory("m1", "target family evidence")]
    harness._retrieval_view_cache = {}
    harness._retrieval_view_cache_hits = 0
    harness._retrieval_view_cache_misses = 0
    operation = {
        "op": "SEARCH_FAMILY",
        "query": "target family",
        "top_k": 3,
        "family_mode": "semantic",
        "produces": ["r1"],
    }
    first = harness._execute_operation(operation, [], [], QueryFrame())[0]
    second = harness._execute_operation(operation, [], [], QueryFrame())[0]
    assert [item["id"] for item in first] == ["m1"]
    assert [item["id"] for item in second] == ["m1"]
    assert harness._retrieval_view_cache_misses == 1
    assert harness._retrieval_view_cache_hits == 1


def test_recovery_drops_soft_family_but_keeps_answer_obligation():
    harness = _Harness()
    slot = {
        "id": "r1",
        "type": "DIRECT",
        "answer_obligation": "which antibiotic",
        "target_surface": "which antibiotic",
        "proof_anchor": "which antibiotic",
        "evidence_family": "antibiotic allergy instruction",
        "retrieval_hint": "antibiotic allergy instruction",
        "retrieval_target": "antibiotic allergy instruction",
        "resolved_keys": ["antibiotic"],
    }
    initial = {
        "query_spec": {},
        "query_mode": "DIRECT",
        "required_slots": [slot],
        "semantic_relations": [],
        "visible_options": {},
        "budget_tier": "SMALL",
        "max_memories": 3,
        "operations": [
            {
                "op": "SEARCH_FAMILY",
                "query": "antibiotic allergy instruction",
                "family_mode": "semantic",
                "produces": ["r1"],
            }
        ],
    }
    recovery = harness._make_deterministic_recovery_plan([slot], "q", initial)
    assert recovery is not None
    recovered = recovery["required_slots"][0]
    assert recovered["answer_obligation"] == "which antibiotic"
    assert recovered["target_surface"] == "which antibiotic"
    assert recovered["evidence_family"] == ""
    assert recovered["retrieval_hint"] == ""
    assert recovered["resolved_keys"] == []
    assert all(op["op"] in harness.LEAN_OPERATIONS for op in recovery["operations"])
