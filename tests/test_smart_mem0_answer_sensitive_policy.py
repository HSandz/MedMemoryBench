"""Regressions for answer-sensitive controller and contrastive CandidateSet policy."""

from copy import deepcopy

from methods.smart_mem0.read_answer_sensitive_controller import (
    ANSWER_SENSITIVE_CONTROLLER_POLICY,
    ReadAnswerSensitiveControllerMixin,
)
from methods.smart_mem0.read_contrastive_candidate_policy import (
    ReadContrastiveCandidatePolicyMixin,
)


def test_controller_contract_requires_memory_valued_answer_sensitive_variables():
    policy = ANSWER_SENSITIVE_CONTROLLER_POLICY
    assert "MEMORY-VALUED" in policy
    assert "ANSWER-SENSITIVE" in policy
    assert "requested decision/conclusion" in policy
    assert "DECISION-CHANGING DECOMPOSITION" in policy
    assert "POSSIBLE_CAUSE" in policy
    assert "Do not collapse" in policy
    assert "generic recommendation requirement" in policy
    assert ReadAnswerSensitiveControllerMixin.CONTROLLER_SCHEMA_VERSION == "requirement-vnext-2"
    assert ReadAnswerSensitiveControllerMixin.CONTROLLER_MAX_OUTPUT_TOKENS == 512


class _CandidateBase:
    HARD_MEMORY_LIMIT = 8

    def __init__(self):
        self._belief_status = {}
        self._memories = [
            {"id": "shared", "claim": "shared safety constraint"},
            {"id": "a", "claim": "candidate A local evidence"},
            {"id": "b", "claim": "candidate B local evidence"},
            {"id": "base", "claim": "shared predicate evidence"},
        ]
        self._last_candidate_propositions = {"A": "option a", "B": "option b"}
        self._last_candidate_local_coverage = {"A": ["a"], "B": ["b"]}
        self._last_candidate_shared_context_ids = ["shared"]

    @staticmethod
    def _normalize_candidate_propositions(values):
        return dict(values or {})

    @staticmethod
    def _question_stem(question):
        return str(question).split("\nA.")[0].strip()

    @staticmethod
    def _memory_satisfies_frame(memory, frame, include_entities=False):
        del memory, frame, include_entities
        return True

    @staticmethod
    def _query_visible_memory(memory):
        del memory
        return True

    @staticmethod
    def _snapshot(value):
        return deepcopy(value)

    @staticmethod
    def _memory_value(memory):
        return memory.get("value") or memory.get("claim")

    def _hybrid_search(self, query, top_k, candidate_ids=None):
        del candidate_ids
        query = str(query).casefold()
        by_id = {item["id"]: item for item in self._memories}
        if "option a" in query:
            order = ["shared", "a"]
        elif "option b" in query:
            order = ["shared", "b"]
        else:
            order = ["base", "shared", "a", "b"]
        return [deepcopy(by_id[item]) for item in order[:top_k]]

    def _reset_candidate_set_state(self):
        self._last_candidate_propositions = {}
        self._last_proposition_probe_coverage = {}
        self._last_option_probe_coverage = {}

    def _semantic_operation_search(self, query, top_k, strategy, frame=None, option_queries=None):
        del query, top_k, strategy, frame, option_queries
        return []

    def _role_aware_support_ids(self, slots, slot_support, candidate_order, limit):
        del slots, slot_support
        return list(candidate_order)[:limit]

    @staticmethod
    def _bridge_endpoint_labels(plan):
        labels = {"ANSWER": "final answer requested by the user"}
        for item in (plan.get("semantic_ir") or {}).get("requirements") or []:
            labels[str(item.get("id") or "")] = str(
                item.get("answer_obligation") or item.get("id") or ""
            )
        return labels

    def prepare_batch_query(self, question, system_message=None, **kwargs):
        del question, system_message, kwargs
        return {
            "messages": [
                {
                    "role": "system",
                    "content": "base\n=== CANDIDATE SET ===\nlegacy ids\n\n=== REASONING BRIDGE CONTRACT ===\nlegacy bridge",
                }
            ],
            "retrieved_memories": deepcopy(self._memories),
            "extra": {
                "candidate_set": {
                    "candidates": {"A": "option a", "B": "option b"}
                },
                "plan": {
                    "semantic_ir": {
                        "requirements": [
                            {"id": "r1", "answer_obligation": "current state"}
                        ]
                    },
                    "reasoning_bridges": [
                        {
                            "type": "INFER",
                            "from": "r1",
                            "to": "ANSWER",
                            "world_knowledge_allowed": True,
                            "requires_stored_relation": False,
                            "goal": "decide safely",
                        }
                    ],
                },
            },
        }


class _CandidateHarness(ReadContrastiveCandidatePolicyMixin, _CandidateBase):
    pass


def test_candidate_pack_does_not_copy_shared_predicate_into_each_local_query():
    harness = _CandidateHarness()
    pack = harness._build_candidate_proposition_pack(
        {"A": "option a", "B": "option b"},
        question="Which option is safest?\nA. option a\nB. option b",
    )
    assert pack["shared_predicate"] == "Which option is safest?"
    assert pack["retrieval_query_semantics"] == "shared_lane_plus_candidate_local_lane"
    assert set(pack["candidate_set"]) == {"A", "B"}


def test_shared_option_search_separates_multi_candidate_memory_from_local_lanes():
    harness = _CandidateHarness()
    harness._semantic_operation_search(
        "shared safety predicate",
        4,
        "SHARED_OPTIONS",
        option_queries=[
            {"label": "A", "query": "option a"},
            {"label": "B", "query": "option b"},
        ],
    )
    assert harness._last_candidate_local_coverage["A"] == ["a"]
    assert harness._last_candidate_local_coverage["B"] == ["b"]
    assert "shared" in harness._last_candidate_shared_context_ids


def test_candidate_local_representatives_are_reserved_before_shared_noise():
    harness = _CandidateHarness()
    chosen = harness._role_aware_support_ids(
        [], {}, ["shared", "base", "a", "b"], 4
    )
    assert chosen[:2] == ["a", "b"]
    assert "shared" in chosen


def test_claim_level_lane_prompt_preserves_unknown_and_question_polarity():
    harness = _CandidateHarness()
    harness._last_candidate_local_coverage = {"A": ["a"], "B": []}
    prepared = _CandidateBase.prepare_batch_query(harness, "q")
    block = harness._candidate_evidence_lane_block(prepared)
    assert "candidate A local evidence" in block
    assert "UNKNOWN; absence is not falsity" in block
    assert "unsafe/forbidden" in block
    assert "Preserve question polarity" in block


def test_prepare_replaces_legacy_candidate_tail_and_readds_stronger_bridge():
    harness = _CandidateHarness()
    prepared = harness.prepare_batch_query("q")
    system = prepared["messages"][0]["content"]
    assert "legacy ids" not in system
    assert "CANDIDATE EVIDENCE LANES" in system
    assert "decide safely" in system
    assert "Name necessary intermediate mechanisms" in system
    assert prepared["extra"]["legacy_candidate_addendum_replaced"] is True
