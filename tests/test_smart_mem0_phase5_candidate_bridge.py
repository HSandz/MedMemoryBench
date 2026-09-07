"""Phase-5 regressions for generic CandidateSet and Reasoning Bridge."""

from copy import deepcopy

from methods.smart_mem0.read_candidate_set import ReadCandidateSetMixin
from methods.smart_mem0.read_reasoning_bridge import ReadReasoningBridgeMixin


class _CandidateBase:
    def _controller_plan(self, ir, question, frame):
        del question, frame
        return {
            "candidate_propositions": deepcopy(
                ir.get("candidate_propositions") or {}
            ),
            "semantic_ir": {},
        }

    @staticmethod
    def _question_stem(question):
        return str(question).split("\nA.")[0].strip()


class _CandidateHarness(ReadCandidateSetMixin, _CandidateBase):
    pass


class _BridgeBase:
    def _controller_plan(self, ir, question, frame):
        del question, frame
        return {
            "semantic_relations": deepcopy(ir.get("relations") or []),
            "semantic_ir": {},
            "query_spec": {},
        }


class _BridgeHarness(ReadReasoningBridgeMixin, _BridgeBase):
    pass


def test_candidate_set_uses_one_shared_question_owned_predicate():
    harness = _CandidateHarness()
    plan = harness._controller_plan(
        {
            "candidate_propositions": {
                "A": "first proposition",
                "B": "second proposition",
            }
        },
        "Which proposition is supported?\nA. first proposition\nB. second proposition",
        None,
    )
    candidate_set = plan["candidate_set"]
    assert candidate_set["version"] == "candidate-set-v2"
    assert candidate_set["shared_predicate"] == "Which proposition is supported?"
    assert candidate_set["candidates"] == {
        "A": "first proposition",
        "B": "second proposition",
    }
    assert candidate_set["evidence_views"] == {"A": [], "B": []}
    assert candidate_set["empty_evidence_semantics"] == "UNKNOWN_NOT_FALSE"


def test_possible_cause_bridge_authorizes_world_knowledge_without_graph_expansion():
    harness = _BridgeHarness()
    plan = harness._controller_plan(
        {
            "relations": [
                {
                    "type": "POSSIBLE_CAUSE",
                    "from": "r1",
                    "to": "r2",
                    "bridge_goal": "explain whether the exposure can produce the outcome",
                }
            ]
        },
        "q",
        None,
    )
    bridge = plan["reasoning_bridges"][0]
    assert bridge["world_knowledge_allowed"] is True
    assert bridge["requires_stored_relation"] is False
    assert bridge["max_graph_hops"] == 0
    assert plan["query_spec"]["world_knowledge_bridge_allowed"] is True


def test_stored_causal_bridge_is_bounded_to_one_hop():
    harness = _BridgeHarness()
    plan = harness._controller_plan(
        {
            "relations": [
                {
                    "type": "CAUSES",
                    "from": "r1",
                    "to": "r2",
                }
            ]
        },
        "q",
        None,
    )
    bridge = plan["reasoning_bridges"][0]
    assert bridge["world_knowledge_allowed"] is False
    assert bridge["requires_stored_relation"] is True
    assert bridge["max_graph_hops"] == 1
    assert plan["query_spec"]["stored_relation_required"] is True


def test_current_and_verify_source_are_not_reasoning_bridges():
    harness = _BridgeHarness()
    plan = harness._controller_plan(
        {
            "relations": [
                {"type": "CURRENT", "from": "r1", "to": ""},
                {"type": "VERIFY_SOURCE", "from": "r1", "to": ""},
            ]
        },
        "q",
        None,
    )
    assert plan["reasoning_bridges"] == []
