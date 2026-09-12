"""Offline architecture tests: no model downloads, API requests or benchmark answers."""

import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from methods.smart_mem0.agent import SmartMem0Agent
from methods.smart_mem0.core import CoreMemoryMixin
from methods.smart_mem0.read_controller import ReadContractMixin
from methods.smart_mem0.read_question_runtime import QuestionReadRuntimeMixin
from methods.smart_mem0.read_query_orchestrator import ReadQueryOrchestratorMixin
from methods.smart_mem0.query import QueryMixin
from methods.smart_mem0.retrieval import RetrievalOperationsMixin


def memory(mid="m1", value="500 mg", **kwargs):
    return {
        "id": mid,
        "claim": "Dose is " + value,
        "value": value,
        "kind": "FACT",
        "evidence_ids": ["ev1"],
        "stance": "AFFIRM",
        "assertion_mode": "DIRECT",
        "event_time": "2024-01-15",
        "document_time": "2024-01-20",
        "_bm25_rank": 1,
        "_dense_rank": 1,
        **kwargs,
    }


class Harness(ReadQueryOrchestratorMixin, QuestionReadRuntimeMixin, QueryMixin):
    _question_options = staticmethod(CoreMemoryMixin._question_options)
    _question_stem = staticmethod(CoreMemoryMixin._question_stem)
    _parse_date = staticmethod(CoreMemoryMixin._parse_date)
    _date_for = staticmethod(RetrievalOperationsMixin._date_for)
    _date_matches = staticmethod(RetrievalOperationsMixin._date_matches)
    _rc_text = staticmethod(ReadContractMixin._rc_text)
    _unwrap_question = staticmethod(lambda q: q)
    _parse_json = staticmethod(json.loads)
    _snapshot = staticmethod(deepcopy)
    _memory_text = staticmethod(lambda m: m["claim"])
    _memory_value = staticmethod(lambda m: m["value"])
    _normalised_value = staticmethod(lambda m: m["value"].casefold())
    _effective_runtime_config = staticmethod(lambda: {})
    _refresh_index = staticmethod(lambda: None)
    _is_state_head = staticmethod(lambda m: m.get("head", False))
    _rg_question_owned_identity = staticmethod(
        lambda slot, m: (m.get("identity_ok", True), {})
    )
    _response_usage = staticmethod(
        lambda response, prompt: {"total_tokens": 12, "latency": 0.01}
    )

    def __init__(self, memories=None, advisor=None):
        self._memories = memories if memories is not None else [memory()]
        self._evidence = [{"id": "ev1", "text": "Dose is 500 mg"}]
        self._belief_status, self._state_heads, self._relations = {}, {}, []
        self._tokenizer = SimpleNamespace(encode=lambda text: list(text))
        self.enable_zero_result_recovery = True
        self.enable_two_stage_controller = True
        self.calls, self.searches = [], []
        self.advisor = advisor or {"projection_hint": "VALUE"}
        self._llm_client = SimpleNamespace(chat=self.chat)

    def chat(self, messages, **kwargs):
        self.calls.append(messages)
        return SimpleNamespace(
            content=json.dumps(self.advisor) if len(self.calls) == 1 else "synthesis"
        )

    def _ad_search(self, question, top_k=16):
        self.searches.append(question)
        return deepcopy(self._memories[:top_k])


def test_new_runtime_owns_preparation_before_legacy_wrappers():
    mro = SmartMem0Agent.__mro__
    owners = [c for c in mro if "prepare_batch_query" in c.__dict__]
    assert owners[:2] == [ReadQueryOrchestratorMixin, QuestionReadRuntimeMixin]


def test_base_before_advisor_and_terminal_uses_memory_not_hypothesis():
    agent = Harness(advisor={"projection_hint": "VALUE", "answer_hypothesis": "WRONG"})
    result = agent.query("Dose?")
    assert result.output == "500 mg"
    assert len(agent.calls) == 1
    assert "500 mg" in agent.calls[0][0]["content"]
    assert result.extra["certificate"]["hypothesis_status"] == "UNSUPPORTED"
    assert result.extra["two_stage_audit"]["total_calls"] == 1


def test_competition_calls_synthesis_and_preserves_both_values():
    agent = Harness([memory(), memory("m2", "1000 mg")])
    result = agent.query("Dose?")
    assert result.output == "synthesis"
    assert len(agent.calls) == 2
    assert result.extra["certificate"]["status"] == "SUPPORTED_COMPETING"
    assert result.extra["second_call"]["called"]
    assert set(result.extra["final_context_ids"]) == {"m1", "m2"}


def test_duplicate_value_is_not_competition():
    result = Harness([memory(), memory("m2")]).query("Dose?")
    assert result.output == "500 mg"
    assert result.extra["terminal"]["closed"]


@pytest.mark.parametrize(
    "raw",
    [
        {"projection_hint": {}},
        {"selector_hint": {"axis": []}},
        {"selector_hint": {"relation": []}},
        {"semantic_hints": "wrong"},
        {"focus_spans": [None]},
    ],
)
def test_malformed_advisory_is_bounded_and_does_not_crash(raw):
    value = Harness()._ad_normalize(raw, "Dose?")
    assert len(value["semantic_hints"]) <= 2


def test_focus_spans_are_exact_and_hints_share_one_budget():
    value = Harness()._ad_normalize(
        {
            "focus_spans": ["Dose", "invented"],
            "semantic_hints": ["a", "b"],
            "missing_evidence_hints": ["c", "d"],
        },
        "Dose?",
    )
    assert value["focus_spans"] == ["Dose"]
    assert value["semantic_hints"] == ["a", "b"]


def test_hint_cannot_replace_base_or_expand_via_relation():
    agent = Harness([memory(str(i), str(i)) for i in range(30)])
    base = deepcopy(agent._memories[:10])
    agent._ad_search = lambda *args: deepcopy(agent._memories[10:])
    advisory = agent._ad_normalize(
        {"semantic_hints": ["x", "y"], "relation_hints": ["CAUSES"]}, "q"
    )
    world, trace = agent._ad_expand("q", base, advisory)
    assert {m["id"] for m in base} <= {m["id"] for m in world}
    assert len(world) == 14
    assert len(trace["hint_novel_ids"]) == 4
    assert not trace["recovery_called"]


def test_base_is_identical_with_different_advisor_outputs():
    agent = Harness(
        [
            memory(str(i), str(i), _bm25_rank=i + 1, _dense_rank=30 - i)
            for i in range(30)
        ]
    )
    first, _ = agent._ad_base_world("Dose?")
    agent.advisor = {"semantic_hints": ["unrelated"], "answer_hypothesis": "x"}
    second, _ = agent._ad_base_world("Dose?")
    assert first == second
    assert len(first) <= 10


@pytest.mark.parametrize(
    "change",
    [
        {"evidence_ids": ["missing"]},
        {"assertion_mode": "RECAP"},
        {"identity_ok": False},
        {"_status": "superseded"},
    ],
)
def test_unusable_evidence_cannot_close_terminal(change):
    agent = Harness([memory(**change)])
    prepared = agent.prepare_batch_query("Dose?")
    assert prepared["precomputed_answer"] is None
    assert not prepared["extra"]["recovery_called"]
    assert prepared["extra"]["candidate_world_ids"] == ["m1"]


def test_denial_blocks_unique_affirmation():
    result = Harness([memory(), memory("m2", stance="DENY")]).query("Dose?")
    assert not result.extra["terminal"]["closed"]


def test_temporal_axis_never_falls_back():
    agent = Harness(
        [memory(event_time="UNKNOWN")],
        {
            "projection_hint": "DATE",
            "selector_hint": {"relation": "LOCATE", "axis": "event_time"},
        },
    )
    result = agent.query("Date?")
    assert not result.extra["terminal"]["closed"]
    assert result.extra["certificate"]["selector_resolution"] == "NO_MATCHING_TIME"


def test_document_date_is_not_event_date():
    agent = Harness(
        advisor={
            "projection_hint": "DATE",
            "selector_hint": {"relation": "LOCATE", "axis": "document_time"},
        }
    )
    assert agent.query("Date?").output == "2024-01-20"


def test_unknown_date_prevents_false_latest():
    agent = Harness(
        [memory(), memory("m2", event_time="UNKNOWN")],
        {
            "projection_hint": "DATE",
            "selector_hint": {"relation": "LATEST", "axis": "event_time"},
        },
    )
    result = agent.query("Latest date?")
    assert result.extra["certificate"]["selector_resolution"] == "INCOMPLETE_CHRONOLOGY"
    assert not result.extra["terminal"]["closed"]


def test_certificate_is_read_only_and_never_searches():
    agent = Harness()
    world = deepcopy(agent._memories)
    original = deepcopy(world)
    agent._ad_search = lambda *args: pytest.fail("Certificate retrieved memory")
    agent._evidence_certificate("Dose?", world, agent._ad_normalize({}, "Dose?"))
    assert world == original


def test_empty_world_recovery_is_single_and_no_loop():
    agent = Harness([])
    result = agent.query("Dose?")
    assert result.extra["recovery_called"]
    assert len(agent.searches) == 2
    assert len(agent.calls) == 2


def test_current_without_durable_state_does_not_terminal():
    agent = Harness(
        advisor={"projection_hint": "VALUE", "selector_hint": {"relation": "CURRENT"}}
    )
    assert not agent.query("Dose?").extra["terminal"]["closed"]


def test_options_always_synthesize_even_if_advisor_says_value():
    agent = Harness()
    result = agent.query("Choose\nA) 500 mg\nB) 1000 mg")
    assert len(agent.calls) == 2
    assert (
        result.extra["semantic_controller"]["semantic_ir"]["projection_hint"]
        == "OPTION_SET"
    )


def test_support_related_do_not_enter_context_or_expand_world():
    agent = Harness()
    agent._relations = [
        {"source_id": "m1", "target_id": "unseen", "type": "RELATED"},
        {"source_id": "m1", "target_id": "m1", "type": "SUPPORT"},
    ]
    prepared = agent.prepare_batch_query("Dose?")
    assert prepared["extra"]["candidate_world_ids"] == ["m1"]
    assert "--SUPPORT-->" not in str(prepared["messages"])
    assert "--RELATED-->" not in str(prepared["messages"])


def test_explicit_subject_cannot_be_answered_by_another_subject():
    agent = Harness([memory(subject="Alice"), memory("m2", "900 mg", subject="Bob")])
    result = agent.query("Alice dose?")
    assert result.output == "500 mg"
    assert result.extra["certificate"]["rejected"]["m2"] == "EXPLICIT_SUBJECT_MISMATCH"


@pytest.mark.parametrize("value", ["Hà Nội", "京都", "القاهرة", "0"])
def test_restored_terminal_is_not_regenerated(value):
    agent = Harness([memory(value=value)])
    prepared = json.loads(json.dumps(agent.prepare_batch_query("Value?")))
    result = agent.generate_prepared_batch_answer(prepared)
    assert result.output == value
    assert len(agent.calls) == 1


def test_real_facade_with_real_hybrid_and_identity_guard(monkeypatch):
    import numpy as np
    from methods.smart_mem0 import core

    class Embedder:
        def encode(self, texts, **kwargs):
            return np.asarray([[1.0, 0.0, 0.0] for _ in texts])

    calls = []

    def chat(messages, **kwargs):
        calls.append(messages)
        return SimpleNamespace(
            content=(
                json.dumps({"projection_hint": "VALUE"})
                if len(calls) == 1
                else "synthesis"
            ),
            input_tokens=5,
            output_tokens=5,
        )

    monkeypatch.setattr(core, "SentenceTransformer", lambda *a, **k: Embedder())
    monkeypatch.setattr(
        core, "create_llm_client", lambda **k: SimpleNamespace(chat=chat)
    )
    agent = SmartMem0Agent()
    agent._memories = [
        memory(
            subject="Alice",
            scope="measurement",
            state_key="dose",
            entities=["Alice"],
            origin_document_time="2024-01-20",
        )
    ]
    agent._evidence = [{"id": "ev1", "text": "Dose is 500 mg"}]
    result = agent.query("Dose?")
    assert result.extra["read_version"] == agent.QUESTION_READ_VERSION
    assert result.extra["two_stage_audit"]["valid"]
    assert len(calls) <= 2
    assert result.extra["base_world_retention_rate"] == 1.0
    assert result.extra["candidate_world_unchanged_after_certificate"]


def test_certificate_mutation_is_a_hard_failure():
    agent = Harness()
    original = agent._evidence_certificate

    def mutate(question, world, advisory):
        result = original(question, world, advisory)
        world[0]["value"] = "changed"
        return result

    agent._evidence_certificate = mutate
    with pytest.raises(AssertionError, match="mutated CandidateWorld"):
        agent.prepare_batch_query("Dose?")


def test_all_visible_options_reserve_a_candidate():
    agent = Harness(
        [
            memory(str(i), str(i), _bm25_rank=i + 1, _dense_rank=30 - i)
            for i in range(30)
        ]
    )
    question = "Choose\n" + "\n".join(
        f"{label}) proposition {i}" for i, label in enumerate("ABCDEFGH")
    )

    def search(q, top_k=16):
        if q != question:
            index = int(q.rsplit(" ", 1)[-1])
            return [memory(f"option{index}")]
        return deepcopy(agent._memories)

    agent._ad_search = search
    base, trace = agent._ad_base_world(question)
    assert len(base) <= 12
    assert {f"option{i}" for i in range(8)} <= {m["id"] for m in base}
    assert len(trace["option_candidate_ids"]) == 8


def test_linked_evidence_does_not_open_unlinked_turns():
    agent = Harness(
        advisor={"projection_hint": "TEXT", "relation_hints": ["VERIFY_SOURCE"]}
    )
    agent._evidence.append({"id": "secret", "text": "unlinked"})
    prepared = agent.prepare_batch_query("Quote the dose")
    assert "unlinked" not in str(prepared["messages"])
    assert prepared["extra"]["evidence_count"] == 1


def test_causal_topology_requires_stored_provenance_and_both_endpoints():
    agent = Harness([memory(), memory("m2")])
    agent._valid_causal_relation = lambda r, by_id: bool(
        r.get("provenance_evidence_ids")
    )
    agent._relations = [
        {"source_id": "m1", "target_id": "m2", "type": "CAUSES"},
        {
            "source_id": "m1",
            "target_id": "missing",
            "type": "CAUSES",
            "provenance_evidence_ids": ["ev1"],
        },
    ]
    prepared = agent.prepare_batch_query("Dose?")
    assert "--CAUSES-->" not in str(prepared["messages"])
    agent._relations[0]["provenance_evidence_ids"] = ["ev1"]
    prepared = agent.prepare_batch_query("Dose?")
    assert "E1 --CAUSES--> E2" in str(prepared["messages"])
