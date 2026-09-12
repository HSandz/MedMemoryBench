"""Offline contract tests for the grounded answer-or-search READ architecture."""

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
        "id": mid,
        "claim": claim,
        "value": value,
        "kind": "FACT",
        "evidence_ids": [f"ev-{mid}"],
        "stance": "AFFIRM",
        "assertion_mode": "DIRECT",
        "event_time": "2024-01-15",
        "document_time": "2024-01-20",
        "_bm25_rank": 1,
        "_dense_rank": 1,
        **kwargs,
    }


class Harness(ReadQueryOrchestratorMixin, QuestionReadRuntimeMixin, QueryMixin):
    _parse_json = staticmethod(json.loads)
    _snapshot = staticmethod(deepcopy)
    _memory_text = staticmethod(lambda m: m["claim"])
    _memory_value = staticmethod(lambda m: str(m.get("value") or ""))
    _date_for = staticmethod(
        lambda m, axis: (
            m.get("event_time")
            if axis == "effective_event_time"
            else m.get(axis)
        )
        or ""
    )
    _effective_runtime_config = staticmethod(lambda: {})
    _refresh_index = staticmethod(lambda: None)
    _response_usage = staticmethod(
        lambda response, prompt: {"total_tokens": 12, "latency": 0.01}
    )
    _valid_causal_relation = staticmethod(
        lambda relation, by_id: (
            relation.get("source_id") in by_id
            and relation.get("target_id") in by_id
            and relation.get("type") == "CAUSES"
        )
    )

    def __init__(self, memories=None, controller=None):
        self._memories = memories if memories is not None else [memory()]
        self._evidence = [
            {"id": eid, "text": item["claim"]}
            for item in self._memories
            for eid in item.get("evidence_ids", [])
        ]
        self._relations = []
        self._belief_status = {}
        self._tokenizer = SimpleNamespace(encode=lambda text: list(text))
        self.enable_zero_result_recovery = True
        self.enable_two_stage_controller = True
        self.calls, self.searches = [], []
        self.controller = controller or {
            "decision": "ANSWER",
            "answer": "Boston",
            "supports": [{"memory_id": "m1", "quote": "relocated to Boston"}],
        }
        self._llm_client = SimpleNamespace(chat=self.chat)

    def chat(self, messages, **kwargs):
        self.calls.append(messages)
        return SimpleNamespace(
            content=json.dumps(self.controller) if len(self.calls) == 1 else "synthesis"
        )

    def _ad_search(self, question, top_k=16):
        self.searches.append(question)
        return deepcopy(self._memories[:top_k])


def test_active_facade_preparation_is_owned_by_new_runtime():
    mro = SmartMem0Agent.__mro__
    owners = [c for c in mro if "prepare_batch_query" in c.__dict__]
    assert owners[:2] == [ReadQueryOrchestratorMixin, QuestionReadRuntimeMixin]


def test_semantic_paraphrase_can_answer_in_first_call_without_code_proof():
    agent = Harness()
    result = agent.query("Where is Alice based?")
    assert result.output == "Boston"
    assert len(agent.calls) == 1
    assert result.extra["grounding_guard"]["accepted"]
    assert result.extra["grounding_guard_semantic_proof"] is False
    assert result.extra["two_stage_audit"]["total_calls"] == 1
    assert "certificate" not in result.extra


def test_answer_support_must_be_from_base_world():
    agent = Harness(
        controller={
            "decision": "ANSWER",
            "answer": "Boston",
            "supports": [{"memory_id": "unseen", "quote": "Boston"}],
        }
    )
    result = agent.query("Where is Alice based?")
    assert result.output == "synthesis"
    assert len(agent.calls) == 2
    assert result.extra["grounding_guard"]["reason"] == "GROUNDING_INTEGRITY_FAILED"
    assert result.extra["grounding_guard"]["failures"][0]["reason"] == "OUTSIDE_BASE_WORLD"


def test_answer_support_quote_must_exist_in_displayed_memory():
    agent = Harness(
        controller={
            "decision": "ANSWER",
            "answer": "Boston",
            "supports": [{"memory_id": "m1", "quote": "invented evidence"}],
        }
    )
    result = agent.query("Where is Alice based?")
    assert result.output == "synthesis"
    assert len(agent.calls) == 2
    assert result.extra["grounding_guard"]["failures"][0]["reason"] == "QUOTE_NOT_IN_MEMORY"


def test_missing_provenance_cannot_early_stop():
    agent = Harness([memory(evidence_ids=[])])
    result = agent.query("Where is Alice based?")
    assert result.output == "synthesis"
    assert result.extra["grounding_guard"]["failures"][0]["reason"] == "MISSING_PROVENANCE"


def test_search_probes_only_add_and_candidate_world_is_bounded():
    agent = Harness([memory(str(i), claim=f"Fact {i}", value=str(i)) for i in range(30)])
    base = deepcopy(agent._memories[:10])

    def search(query, top_k=16):
        start = 10 if query == "missing one" else 20
        return deepcopy(agent._memories[start : start + 8])

    agent._ad_search = search
    controller = agent._ad_normalize(
        {
            "decision": "SEARCH",
            "queries": ["missing one", "missing two", "missing three", "missing four"],
        }
    )
    world, trace = agent._ad_expand("root", base, controller)
    assert {m["id"] for m in base} <= {m["id"] for m in world}
    assert len(world) <= 16
    assert len(trace["retrieval_trace"]) <= 4
    assert all(item["operation"] == "CONTROLLER_SEARCH" for item in trace["retrieval_trace"])


@pytest.mark.parametrize(
    "raw",
    [
        None,
        {},
        {"decision": "MAYBE"},
        {"decision": "ANSWER", "answer": "x"},
        {"decision": "SEARCH", "queries": "not-a-list"},
        {"decision": "SEARCH", "queries": ["a", "b", "c", "d", "e"]},
    ],
)
def test_malformed_controller_output_defaults_safely(raw):
    normalized = Harness()._ad_normalize(raw)
    assert normalized["decision"] in {"ANSWER", "SEARCH"}
    assert len(normalized["queries"]) <= 4
    if raw in (None, {}) or (isinstance(raw, dict) and raw.get("decision") == "MAYBE"):
        assert normalized["decision"] == "SEARCH"


def test_vocative_is_not_an_owner_constraint():
    agent = Harness([memory(owner_id="primary_user")])
    result = agent.query("Doctor, where is Alice based?")
    assert result.output == "Boston"
    assert result.extra["grounding_guard"]["accepted"]


def test_only_explicit_hard_owner_metadata_can_reject_support():
    agent = Harness([memory(owner_id="alice")])
    result = agent.query(QuestionInput("Where am I based?", hard_metadata={"owner_id": "bob"}))
    assert result.output == "synthesis"
    assert result.extra["grounding_guard"]["failures"][0]["reason"] == "HARD_OWNER_MISMATCH"


def test_hard_metadata_survives_into_fallback_reader():
    agent = Harness(
        [memory(owner_id="alice")],
        controller={"decision": "SEARCH", "queries": ["current location"]},
    )
    result = agent.query(
        QuestionInput(
            "Where am I based?",
            hard_metadata={"owner_id": "alice", "namespace": "profile-1"},
        )
    )
    assert result.output == "synthesis"
    assert len(agent.calls) == 2
    fallback_prompt = "\n".join(
        str(message.get("content") or "") for message in agent.calls[1]
    )
    assert '"owner_id": "alice"' in fallback_prompt
    assert '"namespace": "profile-1"' in fallback_prompt


def test_visible_candidates_do_not_create_a_query_type_or_force_second_call():
    agent = Harness()
    result = agent.query("Which is supported?\nA) Boston\nB) Paris")
    ir = result.extra["semantic_controller"]["semantic_ir"]
    assert result.output == "Boston"
    assert len(agent.calls) == 1
    assert ir["decision"] == "ANSWER"
    assert "projection_hint" not in ir
    assert "OPTION_SET" not in json.dumps(ir)


def test_structured_input_has_no_benchmark_candidate_contract():
    agent = Harness()
    question = QuestionInput(
        "Which is supported?\nA) Boston\nB) Paris",
        hard_metadata={"request_id": "r1"},
    )
    agent.query(question)
    assert agent.searches[0] == "Which is supported?\nA) Boston\nB) Paris"
    rendered = question.render()
    assert not hasattr(rendered, "candidates")
    assert rendered.hard_metadata["request_id"] == "r1"


def test_controller_sees_memory_ids_and_valid_stored_relations():
    agent = Harness([memory("m1"), memory("m2", claim="Boston is in Massachusetts.")])
    agent._relations = [{"source_id": "m1", "target_id": "m2", "type": "CAUSES"}]
    agent.controller = {
        "decision": "ANSWER",
        "answer": "Boston",
        "supports": [{"memory_id": "m1", "quote": "Boston"}],
    }
    agent.query("Where is Alice based?")
    prompt = agent.calls[0][0]["content"]
    assert '"id": "m1"' in prompt
    assert '"type": "CAUSES"' in prompt


def test_search_path_keeps_acquired_evidence_for_second_reader():
    memories = [memory(str(i), claim=f"Fact {i}", value=str(i)) for i in range(12)]
    agent = Harness(
        memories,
        controller={"decision": "SEARCH", "queries": ["missing premise"]},
    )
    result = agent.query("Need synthesis")
    assert len(agent.calls) == 2
    assert result.extra["second_call"]["called"]
    assert set(result.extra["final_context_ids"]) <= set(
        result.extra["candidate_world_ids"]
    )
    assert len(result.extra["final_context_ids"]) <= 16
    assert result.extra["evidence_count"] > 0


def test_empty_world_recovery_is_bounded_and_no_llm_loop():
    agent = Harness([], controller={"decision": "SEARCH", "queries": []})
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
    controller = agent._ad_normalize(agent.controller)
    agent._ad_grounding_guard(base, controller)
    assert base == before
