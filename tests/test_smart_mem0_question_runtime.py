"""Offline contract tests for missing-premise grounded READ."""

import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from methods.smart_mem0.agent import SmartMem0Agent
from methods.smart_mem0.query import QueryMixin
from methods.smart_mem0.question_input import QuestionInput
from methods.smart_mem0.read_query_orchestrator import ReadQueryOrchestratorMixin
from methods.smart_mem0.read_question_runtime import QuestionReadRuntimeMixin


def memory(mid="m1", claim="Alice relocated to Boston in 2024.", value="Boston", **kwargs):
    return {
        "id": mid, "claim": claim, "value": value, "kind": "FACT",
        "evidence_ids": [f"ev-{mid}"], "stance": "AFFIRM", "assertion_mode": "DIRECT",
        "event_time": "2024-01-15", "document_time": "2024-01-20",
        "_bm25_rank": 1, "_dense_rank": 1, **kwargs,
    }


class Harness(ReadQueryOrchestratorMixin, QuestionReadRuntimeMixin, QueryMixin):
    _parse_json = staticmethod(json.loads)
    _snapshot = staticmethod(deepcopy)
    _memory_text = staticmethod(lambda m: m["claim"])
    _memory_value = staticmethod(lambda m: str(m.get("value") or ""))
    _date_for = staticmethod(lambda m, axis: (m.get("event_time") if axis == "effective_event_time" else m.get(axis)) or "")
    _effective_runtime_config = staticmethod(lambda: {})
    _refresh_index = staticmethod(lambda: None)
    _response_usage = staticmethod(lambda response, prompt: {"total_tokens": 12, "latency": 0.01})
    _valid_causal_relation = staticmethod(lambda relation, by_id: relation.get("source_id") in by_id and relation.get("target_id") in by_id and relation.get("type") == "CAUSES")

    def __init__(self, memories=None, controller=None):
        self._memories = memories if memories is not None else [memory()]
        self._evidence = [{"id": eid, "text": item["claim"]} for item in self._memories for eid in item.get("evidence_ids", [])]
        self._relations, self._belief_status = [], {}
        self._tokenizer = SimpleNamespace(encode=lambda text: list(text))
        self.enable_zero_result_recovery = True
        self.enable_two_stage_controller = True
        self.calls, self.searches = [], []
        self.controller = controller or {
            "decision": "ANSWER", "answer": "Boston",
            "supports": [{"memory_id": "m1", "quote": "relocated to Boston"}],
        }
        self._llm_client = SimpleNamespace(chat=self.chat)

    def chat(self, messages, **kwargs):
        self.calls.append(messages)
        return SimpleNamespace(content=json.dumps(self.controller) if len(self.calls) == 1 else "synthesis")

    def _ad_search(self, question, top_k=16):
        self.searches.append(question)
        return deepcopy(self._memories[:top_k])


def test_active_facade_preparation_is_owned_by_new_runtime():
    owners = [c for c in SmartMem0Agent.__mro__ if "prepare_batch_query" in c.__dict__]
    assert owners[:2] == [ReadQueryOrchestratorMixin, QuestionReadRuntimeMixin]


def test_semantic_paraphrase_can_answer_in_first_call_without_code_proof():
    result = Harness().query("Where is Alice based?")
    assert result.output == "Boston"
    assert result.extra["grounding_guard"]["accepted"]
    assert result.extra["grounding_guard_semantic_proof"] is False
    assert result.extra["two_stage_audit"]["total_calls"] == 1
    assert "certificate" not in result.extra


def test_answer_support_must_be_from_base_world():
    agent = Harness(controller={"decision": "ANSWER", "answer": "Boston", "supports": [{"memory_id": "unseen", "quote": "Boston"}]})
    result = agent.query("Where is Alice based?")
    assert result.output == "synthesis"
    assert len(agent.calls) == 2
    assert result.extra["grounding_guard"]["failures"][0]["reason"] == "OUTSIDE_BASE_WORLD"


def test_answer_support_quote_must_exist_in_displayed_memory():
    agent = Harness(controller={"decision": "ANSWER", "answer": "Boston", "supports": [{"memory_id": "m1", "quote": "invented evidence"}]})
    result = agent.query("Where is Alice based?")
    assert result.output == "synthesis"
    assert result.extra["grounding_guard"]["failures"][0]["reason"] == "QUOTE_NOT_IN_MEMORY"


def test_search_is_missing_premise_decomposition_not_only_queries():
    normalized = Harness()._ad_normalize({
        "decision": "SEARCH",
        "known_supports": [{"memory_id": "m1", "role": "location history"}],
        "needs": [
            {"need": "current residence after the move", "query": "Alice current residence after 2024 move"},
            {"need": "duplicate query should collapse", "query": "Alice current residence after 2024 move"},
        ],
    })
    assert normalized["known_supports"][0]["memory_id"] == "m1"
    assert len(normalized["needs"]) == 1
    assert normalized["queries"] == ["Alice current residence after 2024 move"]


def test_known_support_outside_base_is_removed_before_execution():
    agent = Harness(controller={
        "decision": "SEARCH",
        "known_supports": [{"memory_id": "outside", "role": "invented"}],
        "needs": [{"need": "missing residence", "query": "Alice residence"}],
    })
    result = agent.query("Where is Alice based?")
    assert result.extra["controller_known_support_ids"] == []


def test_need_to_evidence_mapping_survives_into_second_reader():
    memories = [memory("m1"), memory("m2", claim="Alice later moved to Kyoto.", value="Kyoto")]
    agent = Harness(memories, controller={
        "decision": "SEARCH",
        "known_supports": [{"memory_id": "m1", "role": "earlier residence"}],
        "needs": [{"need": "later residence after Boston", "query": "Alice later residence"}],
    })
    agent._ad_search = lambda q, top_k=16: deepcopy([memories[1]]) if q == "Alice later residence" else deepcopy(memories[:1])
    result = agent.query("Where is Alice based now?")
    prompt = "\n".join(str(m.get("content") or "") for m in agent.calls[1])
    assert "KNOWN SUPPORTS" in prompt
    assert "later residence after Boston" in prompt
    assert "Alice later moved to Kyoto" in prompt
    assert result.extra["retrieval_need_groups"][0]["output_ids"] == ["m2"]
    assert result.extra["retrieval_need_structural_coverage"] == 1.0


def test_search_probes_only_add_and_candidate_world_is_bounded():
    agent = Harness([memory(str(i), claim=f"Fact {i}", value=str(i)) for i in range(30)])
    base = deepcopy(agent._memories[:10])
    agent._ad_search = lambda query, top_k=16: deepcopy(agent._memories[10:18] if "one" in query else agent._memories[20:28])
    controller = agent._ad_normalize({"decision": "SEARCH", "needs": [
        {"need": "premise one", "query": "missing one"},
        {"need": "premise two", "query": "missing two"},
        {"need": "premise three", "query": "missing three"},
        {"need": "premise four", "query": "missing four"},
    ]})
    world, trace = agent._ad_expand("root", base, controller)
    assert {m["id"] for m in base} <= {m["id"] for m in world}
    assert len(world) <= 16
    assert len(trace["retrieval_need_groups"]) <= 4
    assert all(item["operation"] == "MISSING_PREMISE_SEARCH" for item in trace["retrieval_trace"])


@pytest.mark.parametrize("raw", [None, {}, {"decision": "MAYBE"}, {"decision": "ANSWER", "answer": "x"}, {"decision": "SEARCH", "needs": "bad"}])
def test_malformed_controller_output_defaults_safely(raw):
    normalized = Harness()._ad_normalize(raw)
    assert normalized["decision"] in {"ANSWER", "SEARCH"}
    assert len(normalized["needs"]) <= 4


def test_legacy_queries_normalize_for_resume_compatibility():
    normalized = Harness()._ad_normalize({"decision": "SEARCH", "queries": ["a", "b", "a"]})
    assert normalized["queries"] == ["a", "b"]
    assert [item["need"] for item in normalized["needs"]] == ["a", "b"]


def test_vocative_is_not_an_owner_constraint():
    result = Harness([memory(owner_id="primary_user")]).query("Doctor, where is Alice based?")
    assert result.output == "Boston"
    assert result.extra["grounding_guard"]["accepted"]


def test_only_explicit_hard_owner_metadata_can_reject_support():
    result = Harness([memory(owner_id="alice")]).query(QuestionInput("Where am I based?", hard_metadata={"owner_id": "bob"}))
    assert result.output == "synthesis"
    assert result.extra["grounding_guard"]["failures"][0]["reason"] == "HARD_OWNER_MISMATCH"


def test_second_reader_contract_allows_domain_knowledge_but_not_user_fact_invention():
    agent = Harness(controller={"decision": "SEARCH", "needs": []})
    agent.query("Which class applies?")
    prompt = "\n".join(str(m.get("content") or "") for m in agent.calls[1])
    assert "MAY use general/domain knowledge" in prompt
    assert "must not invent additional user-specific" in prompt
    assert "evaluate every relevant proposition independently" in prompt
    assert "requested attribute" in prompt
    assert "historical template" in prompt
    assert "exact stored value/verbatim surface" in prompt


def test_visible_candidates_do_not_create_a_query_type():
    result = Harness().query("Which is supported?\nA) Boston\nB) Paris")
    ir = result.extra["semantic_controller"]["semantic_ir"]
    assert result.output == "Boston"
    assert "OPTION_SET" not in json.dumps(ir)
    assert "projection_hint" not in ir


def test_empty_world_recovery_is_bounded_and_no_llm_loop():
    agent = Harness([], controller={"decision": "SEARCH", "needs": []})
    result = agent.query("Unknown?")
    assert result.extra["recovery_called"]
    assert len(agent.calls) == 2
    assert len(agent.searches) == 2
    assert result.extra["two_stage_audit"]["total_calls"] == 2


def test_restored_early_answer_is_never_regenerated():
    agent = Harness()
    prepared = json.loads(json.dumps(agent.prepare_batch_query("Where is Alice based?")))
    result = agent.generate_prepared_batch_answer(prepared)
    assert result.output == "Boston"
    assert len(agent.calls) == 1
    assert not result.extra["answer_llm_called"]


def test_grounding_guard_is_read_only():
    agent = Harness()
    base = deepcopy(agent._memories)
    before = deepcopy(base)
    agent._ad_grounding_guard(base, agent._ad_normalize(agent.controller))
    assert base == before
