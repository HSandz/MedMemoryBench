"""Regressions for selector discipline and deterministic evidence precision."""

from copy import deepcopy

from methods.smart_mem0.read_answer_sensitive_controller import (
    ANSWER_SENSITIVE_CONTROLLER_POLICY,
)
from methods.smart_mem0.read_evidence_precision import ReadEvidencePrecisionMixin


def test_controller_policy_defaults_temporal_selector_to_empty():
    policy = ANSWER_SENSITIVE_CONTROLLER_POLICY
    assert "Default selector is empty" in policy
    assert "past, previous, recent or ongoing are scope cues, not automatic LATEST" in policy
    assert "CURRENT is the state-resolution bridge" in policy
    assert "FINAL SELF-CHECK BEFORE JSON" in policy


class _Base:
    def __init__(self):
        self._memories = []
        self.last_search = None

    @staticmethod
    def _rc_text(value):
        return " ".join(str(value or "").casefold().split())

    @staticmethod
    def _snapshot(value):
        return deepcopy(value)

    @staticmethod
    def _memory_value(memory):
        return memory.get("value") or memory.get("claim") or ""

    @staticmethod
    def _memory_satisfies_frame(
        memory, frame, include_dates=False, include_entities=False
    ):
        del memory, frame, include_dates, include_entities
        return True

    @staticmethod
    def _query_visible_memory(memory, include_history=False):
        del memory, include_history
        return True

    @staticmethod
    def _date_for(memory, axis):
        return memory.get(axis) or memory.get("event_time") or ""

    def _semantic_operation_search(
        self, query, top_k, strategy, frame=None, option_queries=None
    ):
        del frame
        self.last_search = {
            "query": query,
            "top_k": top_k,
            "strategy": strategy,
            "option_queries": deepcopy(option_queries or []),
        }
        return []

    def _locate_temporal_family(self, query, frame, axis):
        del query, frame, axis
        return [{"id": "base", "claim": "base", "event_time": "2024-02-01"}]

    def _hybrid_search(self, query, top_k, candidate_ids=None):
        del query
        allowed = set(candidate_ids or [])
        return [
            deepcopy(memory)
            for memory in self._memories
            if memory.get("id") in allowed
        ][:top_k]

    def prepare_batch_query(self, question, system_message=None, **kwargs):
        del question, system_message, kwargs
        return {
            "messages": [{"role": "system", "content": "base"}],
            "retrieved_memories": [
                {"id": "m1", "claim": "participant-specific important evidence"}
            ],
            "extra": {
                "plan": {
                    "required_slots": [
                        {"id": "r1", "type": "DIRECT", "time_axis": ""}
                    ],
                    "semantic_ir": {
                        "requirements": [
                            {
                                "id": "r1",
                                "answer_obligation": "decision-changing variable",
                                "selector": {},
                            }
                        ]
                    },
                },
                "requirement_context_selected": {"r1": ["m1"]},
            },
        }


class _Harness(ReadEvidencePrecisionMixin, _Base):
    pass


def test_non_temporal_context_keeps_relevance_order():
    assert _Harness._context_time_axis(
        [{"id": "r1", "type": "DIRECT", "time_axis": ""}]
    ) is None
    assert _Harness._context_time_axis(
        [{"id": "r1", "type": "TEMPORAL", "time_axis": "document_time"}]
    ) == "document_time"


def test_candidate_probe_combines_shared_predicate_and_option_before_lane_dedup():
    harness = _Harness()
    harness._semantic_operation_search(
        "shared safety predicate",
        4,
        "SHARED_OPTIONS",
        option_queries=[{"label": "A", "query": "cefuroxime"}],
    )
    assert harness.last_search["option_queries"][0]["query"] == (
        "shared safety predicate | cefuroxime"
    )


def test_numeric_temporal_backup_keeps_semantic_aliases():
    harness = _Harness()
    harness._memories = [
        {
            "id": "m20",
            "claim": "Patient started metformin 1500 mg daily.",
            "value": "metformin 1500 mg",
            "event_time": "2024-01-06",
            "_dense_score": 0.80,
            "_overlap": 4,
        },
        {
            "id": "m40",
            "claim": "Patient took metformin 1500 mg regularly for one week.",
            "value": "metformin 1500 mg",
            "event_time": "2024-01-12",
            "_dense_score": 0.82,
            "_overlap": 4,
        },
    ]
    result = harness._locate_temporal_family(
        "when did metformin 1500 mg start", object(), "event_time"
    )
    assert {"m20", "m40"}.issubset({memory["id"] for memory in result})


def test_obligation_map_materializes_selected_claims_and_locate_semantics():
    harness = _Harness()
    prepared = harness.prepare_batch_query("q")
    system = prepared["messages"][0]["content"]
    assert "EVIDENCE OBLIGATION MAP" in system
    assert "participant-specific important evidence" in system
    assert "LOCATE means choose the evidence whose proposition best matches" in system
    assert prepared["extra"]["context_order_semantics"] == (
        "relevance_unless_explicit_temporal_axis"
    )
