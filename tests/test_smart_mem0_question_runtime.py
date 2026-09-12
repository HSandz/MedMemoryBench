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
from methods.smart_mem0.read_evidence_certificate import EvidenceCertificateMixin


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
    _ec_identity = staticmethod(
        lambda question, m, advisory: (m.get("identity_ok", True), {})
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
    assert result.output == "500 mg"
    assert len(calls) == 1


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


@pytest.mark.parametrize(
    "raw,status",
    [
        (None, "NOT_REQUESTED"),
        ({}, "NOT_REQUESTED"),
        ({"relation": "", "axis": ""}, "NOT_REQUESTED"),
        ("bad", "INVALID"),
        ({"relation": []}, "INVALID"),
        ({"axis": "event_time"}, "INVALID"),
        ({"extra": 1}, "INVALID"),
        ({"relation": "LATEST", "axis": "event_time"}, "VALID"),
    ],
)
def test_selector_tristate(raw, status):
    assert (
        Harness()._ad_normalize({"selector_hint": raw}, "q")["selector_status"]
        == status
    )


@pytest.mark.parametrize(
    "relation", ["VERIFY_SOURCE", "TEMPORAL_ORDER", "INFER", "COMPARE", "CAUSES"]
)
def test_relation_advice_alone_cannot_veto_atomic_terminal(relation):
    agent = Harness(advisor={"projection_hint": "VALUE", "relation_hints": [relation]})
    result = agent.query("Dose?")
    assert result.output == "500 mg"
    assert len(agent.calls) == 1


class RealIdentityHarness(Harness):
    _ec_identity = EvidenceCertificateMixin._ec_identity


@pytest.mark.parametrize("predicate", ["dose", "liều dùng", "投与量", "الجرعة"])
def test_real_certificate_binds_predicate_without_whole_question_similarity(predicate):
    agent = RealIdentityHarness(
        [memory(state_key=predicate, claim=f"{predicate}: 500 mg")],
        {"projection_hint": "VALUE", "focus_spans": [predicate]},
    )
    result = agent.query(f"Please report the recorded {predicate}?")
    assert result.output == "500 mg"
    assert len(agent.calls) == 1


def test_name_alone_is_not_question_predicate():
    agent = RealIdentityHarness(
        [memory(object_anchor="Cefuroxime", claim="Cefuroxime was prescribed")],
        {"projection_hint": "ENTITY", "focus_spans": ["Cefuroxime"]},
    )
    assert not agent.query("Was Cefuroxime contraindicated?").extra["terminal"][
        "closed"
    ]


def test_does_not_invent_avoid_instruction_from_allergy():
    agent = RealIdentityHarness(
        [memory(object_anchor="Cefuroxime", claim="Allergy to Cefuroxime")],
        {"projection_hint": "ENTITY", "focus_spans": ["instructed to avoid"]},
    )
    assert not agent.query("Which drug was I instructed to avoid?").extra["terminal"][
        "closed"
    ]
    agent._memories[0]["claim"] = "The patient was instructed to avoid Cefuroxime"
    prepared = agent.prepare_batch_query("Which drug was I instructed to avoid?")
    # Reset the fake LLM, whose second response otherwise represents synthesis.
    agent.calls.clear()
    assert agent.query("Which drug was I instructed to avoid?").output == "Cefuroxime"


def test_exact_named_rail_rejects_high_df_in_any_language():
    agent = Harness(
        [
            memory(str(i), entities=["patient", "患者", "rare" if i == 0 else "common"])
            for i in range(10)
        ]
    )
    for item in agent._memories:
        item["claim"] = " ".join(item["entities"])
    surfaces = agent._ad_exact_surfaces('patient 患者 rare "patient" 15.1')
    assert "患者" not in surfaces
    assert "rare" in surfaces
    assert "patient" in surfaces  # Explicit quotes override DF.
    assert any("15.1" in s for s in surfaces)
    assert "patient" not in agent._ad_exact_surfaces("patient")


def test_hypothesis_is_additive_and_shares_world_cap():
    agent = Harness([memory(str(i), str(i)) for i in range(20)])
    base = deepcopy(agent._memories[:12])
    agent._ad_search = lambda *args: deepcopy(agent._memories[12:])
    ir = agent._ad_normalize(
        {
            "projection_hint": "ENTITY",
            "answer_hypothesis": "h",
            "semantic_hints": ["a", "b"],
        },
        "q",
    )
    world, trace = agent._ad_expand("q", base, ir)
    assert len(world) == 16
    assert {m["id"] for m in base} <= {m["id"] for m in world}
    assert trace["retrieval_trace"][-1]["operation"] == "ATOMIC_HYPOTHESIS"
    assert trace["retrieval_trace"][-1]["output_ids"]


def test_hint_survives_final_context_reservation():
    agent = Harness([memory(str(i), str(i)) for i in range(14)])
    acquisition = {
        "base_exact_ids": ["0"],
        "base_lexical_ids": list(map(str, range(4))),
        "base_dense_ids": list(map(str, range(4, 8))),
        "option_candidate_ids": {},
        "retrieval_trace": [{"operation": "ADVISORY_HINT", "output_ids": ["12", "13"]}],
    }
    certificate = {
        "support_ids": [],
        "supported_surfaces": [],
        "status": "INSUFFICIENT",
        "terminal": {"closed": False, "eligible": False},
    }
    selected = agent._ad_context_selection(agent._memories, certificate, acquisition)
    assert len(selected) <= 8
    assert "12" in {m["id"] for m in selected}


def test_call_counters_track_observed_not_planned_in_sync_and_batch():
    agent = Harness(advisor={"projection_hint": "TEXT"})
    prepared = agent.prepare_batch_query("Explain")
    assert prepared["extra"]["method_llm_calls"]["answer"] == 0
    result = agent.finalize_batch_query(prepared, "batch answer")
    agent.record_batch_query_usage(result, 20, 5)
    assert result.extra["method_llm_calls"]["total"] == 2
    assert result.extra["two_stage_audit"]["answer_calls"] == 1
    terminal = Harness().query("Dose?")
    assert terminal.extra["method_llm_calls"]["total"] == 1
    assert "retrieval_complete" not in terminal.extra
    assert terminal.extra["candidate_world_nonempty"]


def test_aggregate_counts_match_actual_one_and_two_call_queries():
    from benchmarks.medmemorybench.smart_mem0_batch_integration import _method_llm_usage

    one = Harness().query("Dose?")
    two = Harness(advisor={"projection_hint": "TEXT"}).query("Explain")
    evaluator = SimpleNamespace(
        aggregator=SimpleNamespace(
            results=[
                SimpleNamespace(details={"agent_telemetry": r.extra})
                for r in (one, two)
            ]
        )
    )
    summary = _method_llm_usage(evaluator)
    assert summary["controller_calls"] == 2
    assert summary["answer_calls"] == 1
    assert summary["total_calls"] == 3


def test_composition_without_one_atom_covering_all_predicates_needs_synthesis():
    agent = RealIdentityHarness(
        [memory(claim="left: 100"), memory("m2", claim="right: 100")],
        {
            "projection_hint": "VALUE",
            "focus_spans": ["left", "right"],
            "relation_hints": [],
        },
    )
    result = agent.query("Compare left and right")
    assert not result.extra["terminal"]["closed"]
    assert len(agent.calls) == 2


def test_no_legacy_certificate_or_controller_in_active_mro():
    names = {c.__name__ for c in SmartMem0Agent.__mro__}
    assert "ReadRequirementIdentityRuntimeMixin" not in names
    assert "ReadStableSemanticRuntimeMixin" not in names
    assert "ReadUsageContractMixin" not in names


def test_text_hypothesis_does_not_add_search():
    agent = Harness()
    ir = agent._ad_normalize(
        {"projection_hint": "TEXT", "answer_hypothesis": "An invented explanation"}, "q"
    )
    world, trace = agent._ad_expand("q", agent._memories, ir)
    assert not trace["retrieval_trace"]
    assert not agent.searches
