"""Production-path contracts, with real lexical/hybrid logic and no external API."""

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from methods.smart_mem0 import QuestionInput, SmartMem0Agent, core
from methods.smart_mem0.canonicalization import state_identity


@pytest.fixture
def agent_factory(monkeypatch):
    class Embedder:
        def encode(self, texts, **kwargs):
            return np.asarray([[1.0, 0.0, 0.0] for _ in texts])

    monkeypatch.setattr(core, "SentenceTransformer", lambda *a, **k: Embedder())

    def create(records, advisory=None):
        calls = []

        def chat(messages, **kwargs):
            calls.append(messages)
            return SimpleNamespace(
                content=(
                    json.dumps(advisory or {"projection_hint": "VALUE"})
                    if len(calls) == 1
                    else "synthesis"
                ),
                input_tokens=5,
                output_tokens=5,
                latency=0.0,
            )

        monkeypatch.setattr(
            core, "create_llm_client", lambda **k: SimpleNamespace(chat=chat)
        )
        agent = SmartMem0Agent()
        agent._memories = records
        agent._evidence = [{"id": "ev1", "text": "source"}]
        return agent, calls

    return create


def atom(predicate, value, mid="m1", **extra):
    return dict(
        id=mid,
        kind="FACT",
        owner_id="entity-1",
        subject_id="entity-1",
        subject="entity-1",
        scope="properties",
        state_key=predicate,
        claim=f"{predicate}: {value}",
        value=value,
        entities=["entity-1"],
        evidence_ids=["ev1"],
        stance="AFFIRM",
        assertion_mode="DIRECT",
        event_time="2024-01-15",
        document_time="2024-01-20",
        **extra,
    )


@pytest.mark.parametrize(
    "predicate,value",
    [
        ("dose", "500 mg"),
        ("runtime_version", "3.12"),
        ("deployment_status", "blocked"),
        ("hotel", "Sakura Inn"),
        ("keyboard_layout", "ANSI"),
        ("subscription_plan", "Pro"),
        ("liều dùng", "500 mg"),
        ("宿泊施設", "京都ホテル"),
        ("الخطة", "أساسية"),
    ],
)
def test_real_lexical_hybrid_certificate_across_domains(
    agent_factory, predicate, value
):
    agent, calls = agent_factory(
        [
            atom(predicate, value),
            atom("unrelated", "noise", "m2"),
            atom("different", "other", "m3"),
        ]
    )
    question = predicate.replace("_", " ")
    assert agent._tokenize(question)
    agent._refresh_index()
    candidates = agent._hybrid_search(question, top_k=2)
    assert candidates[0]["id"] == "m1"
    assert candidates[0]["_bm25_score"] > 0
    result = agent.query(question)
    assert result.output == value
    assert len(calls) == 1
    assert result.extra["final_context_ids"] == ["m1"]
    assert not result.extra["boundary_violation"]


def test_generic_identity_no_aliases_no_owner_invention():
    base = dict(
        kind="STATE",
        subject_id="account-1",
        scope="service",
        state_key="plan",
        object_anchor="cloud",
    )
    identity = state_identity(base)
    assert identity == state_identity(
        dict(base, value="new", event_time="2025", scope_entities=["x"])
    )
    assert identity != state_identity(dict(base, object_anchor="other"))
    assert identity != state_identity(dict(base, scope="service:other"))
    assert not state_identity(dict(base, subject_id=""))
    assert core.CoreMemoryMixin._canonical_state_key("blood glucose") == "blood_glucose"
    assert core.CoreMemoryMixin._canonical_scope("drug") == "drug"
    assert core.CoreMemoryMixin._canonical_state_key(
        "deployment_status"
    ) != core.CoreMemoryMixin._canonical_state_key("deployment")


def test_unicode_units_and_no_english_stemming():
    tokens = core.CoreMemoryMixin._tokenize("Hà Nội 京都 العربية 3.12 5% mg/dL running")
    assert {"hà", "nội", "京都", "العربية", "3.12", "5%", "mg/dl", "running"} <= set(
        tokens
    )
    assert "runn" not in tokens


def test_rare_topical_overlap_is_not_proof(agent_factory):
    agent, _ = agent_factory([atom("latency", "20%", claim_extra="unused")])
    agent._memories[0]["claim"] = "Production latency decreased 20%"
    result = agent.query("Which database was selected for production?")
    assert not result.extra["terminal"]["closed"]


def test_focus_span_count_does_not_control_terminal(agent_factory):
    for spans in ([], ["plan"], ["subscription", "plan"]):
        agent, _ = agent_factory(
            [atom("subscription_plan", "Pro")],
            {"projection_hint": "VALUE", "focus_spans": spans},
        )
        assert agent.query("subscription plan").extra["terminal"]["closed"]


@pytest.mark.parametrize(
    "required,closed", [(None, False), (True, False), (False, True)]
)
def test_invalid_selector_retains_evidence(agent_factory, required, closed):
    agent, _ = agent_factory(
        [atom("color", "blue")],
        {"projection_hint": "VALUE", "selector_hint": {"relation": "bogus"}},
    )
    result = agent.query(QuestionInput("color", selector_required=required))
    assert result.extra["certificate"]["support_ids"] == ["m1"]
    assert result.extra["terminal"]["closed"] is closed


def test_structured_options_use_stem_and_preserve_labels_without_leak(agent_factory):
    agent, _ = agent_factory([atom("plan", "Pro")])
    probes = []
    search = agent._ad_search

    def traced(question, top_k=16):
        probes.append(str(question))
        return search(question, top_k)

    agent._ad_search = traced
    result = agent.query(QuestionInput("plan?", {"選択甲": "Pro", "β": "Free"}))
    assert "plan?\nPro" in probes and "plan?\nFree" in probes
    assert set(result.extra["option_candidate_ids"]) == {"選択甲", "β"}
    assert not agent._question_options("plan?")
    assert len(agent._question_options("plan?\n1) Pro\n2) Free")) == 2


def test_four_premise_hints_share_bounded_world(agent_factory):
    records = [atom(f"property{i}", str(i), f"m{i}") for i in range(20)]
    agent, _ = agent_factory(
        records,
        {
            "projection_hint": "TEXT",
            "evidence_hints": [f"property{i}" for i in range(10, 16)],
        },
    )
    result = agent.query("Explain the combined outcome")
    extra = result.extra
    assert len(extra["llm_semantic_hints"]) == 4
    assert set(extra["base_world_ids"]) <= set(extra["candidate_world_ids"])
    assert len(extra["candidate_world_ids"]) <= 16
    assert len(extra["final_context_ids"]) <= 8


def test_context_does_not_fill_unassigned_tail(agent_factory):
    records = [atom("other", str(i), f"m{i}") for i in range(15)]
    agent, _ = agent_factory(records, {"projection_hint": "TEXT"})
    result = agent.query("Explain the outcome")
    assert len(result.extra["candidate_world_ids"]) > len(
        result.extra["final_context_ids"]
    )
    assert len(result.extra["final_context_ids"]) <= 3


def test_explicit_owner_rejects_missing_owner(agent_factory):
    agent, _ = agent_factory([atom("plan", "Pro")])
    for key in ("owner_id", "subject_id", "subject"):
        agent._memories[0].pop(key)
    result = agent.query(QuestionInput("plan", owner_id="entity-2"))
    assert result.extra["certificate"]["rejected"]["m1"] == "EXPLICIT_SUBJECT_MISMATCH"


def test_active_read_has_no_legacy_import_or_mro():
    root = Path(core.__file__).parent
    active = [
        "agent",
        "query",
        "query_rendering",
        "query_answer_runtime",
        "read_advisory",
        "read_evidence_certificate",
        "read_question_runtime",
        "read_query_orchestrator",
        "read_structural_resolution",
        "read_usage_contract",
    ]
    allowed = set(active) | {
        "canonicalization",
        "contracts",
        "core",
        "question_input",
        "lexical",
        "write",
        "capture",
        "consolidation",
    }
    for name in active:
        for node in ast.walk(ast.parse((root / f"{name}.py").read_text())):
            if isinstance(node, ast.ImportFrom) and node.level:
                assert (node.module or "").split(".")[0] in allowed, (name, node.module)
    modules = {cls.__module__.split(".")[-1] for cls in SmartMem0Agent.__mro__}
    assert not modules & {
        "execution",
        "retrieval",
        "planning",
        "read_controller",
        "legacy",
    }
