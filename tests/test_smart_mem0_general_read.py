"""Production-path contracts for the general-domain grounded READ pipeline."""

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
            # Deterministic local embedding stub; lexical ranking remains real.
            return np.asarray([[1.0, 0.0, 0.0] for _ in texts])

    monkeypatch.setattr(core, "SentenceTransformer", lambda *a, **k: Embedder())

    def create(records, controller=None):
        calls = []
        default = {
            "decision": "ANSWER",
            "answer": str(records[0].get("value") or records[0].get("claim") or ""),
            "supports": [
                {
                    "memory_id": records[0]["id"],
                    "quote": str(records[0].get("value") or records[0].get("claim") or ""),
                }
            ],
        } if records else {"decision": "SEARCH", "queries": []}

        def chat(messages, **kwargs):
            calls.append(messages)
            return SimpleNamespace(
                content=json.dumps(controller or default) if len(calls) == 1 else "synthesis",
                input_tokens=5,
                output_tokens=5,
                latency=0.0,
            )

        monkeypatch.setattr(
            core, "create_llm_client", lambda **k: SimpleNamespace(chat=chat)
        )
        agent = SmartMem0Agent()
        agent._memories = records
        evidence_ids = {
            eid
            for record in records
            for eid in (record.get("evidence_ids") or [])
        }
        agent._evidence = [{"id": eid, "text": "source"} for eid in sorted(evidence_ids)]
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
        evidence_ids=[f"ev-{mid}"],
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
def test_real_hybrid_and_grounded_early_answer_across_domains(
    agent_factory, predicate, value
):
    records = [
        atom(predicate, value),
        atom("unrelated", "noise", "m2"),
        atom("different", "other", "m3"),
    ]
    controller = {
        "decision": "ANSWER",
        "answer": value,
        "supports": [{"memory_id": "m1", "quote": value}],
    }
    agent, calls = agent_factory(records, controller)
    question = predicate.replace("_", " ")
    assert agent._tokenize(question)
    agent._refresh_index()
    candidates = agent._hybrid_search(question, top_k=2)
    assert candidates[0]["id"] == "m1"
    assert candidates[0]["_bm25_score"] > 0

    result = agent.query(question)
    assert result.output == value
    assert len(calls) == 1
    assert result.extra["grounding_guard"]["accepted"]
    assert result.extra["final_context_ids"] == ["m1"]
    assert result.extra["base_world_retention_rate"] == 1.0
    assert result.extra["two_stage_audit"]["valid"]


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


def test_retrieval_overlap_never_becomes_semantic_proof(agent_factory):
    record = atom("latency", "20%")
    record["claim"] = "Production latency decreased 20%"
    agent, calls = agent_factory(
        [record],
        {"decision": "SEARCH", "queries": ["selected production database"]},
    )
    result = agent.query("Which database was selected for production?")
    assert result.output == "synthesis"
    assert len(calls) == 2
    assert not result.extra["terminal"]["closed"]
    assert result.extra["grounding_guard_semantic_proof"] is False


def test_hard_owner_metadata_is_authoritative_but_raw_names_are_not(agent_factory):
    records = [atom("plan", "Pro")]
    controller = {
        "decision": "ANSWER",
        "answer": "Pro",
        "supports": [{"memory_id": "m1", "quote": "Pro"}],
    }

    agent, _ = agent_factory(records, controller)
    assert agent.query("Manager, what is the plan?").output == "Pro"

    agent, calls = agent_factory(records, controller)
    result = agent.query(
        QuestionInput("What is the plan?", hard_metadata={"owner_id": "entity-2"})
    )
    assert result.output == "synthesis"
    assert len(calls) == 2
    assert result.extra["grounding_guard"]["failures"][0]["reason"] == "HARD_OWNER_MISMATCH"


def test_question_input_has_no_benchmark_specific_candidate_or_selector_fields():
    request = QuestionInput(
        "Which statement is supported?\nA) Alpha\nB) Beta",
        hard_metadata={"request_id": "r1"},
    ).render()
    assert str(request).startswith("Which statement is supported?")
    assert request.hard_metadata == {"request_id": "r1"}
    assert not hasattr(request, "candidates")
    assert not hasattr(request, "selector_required")


def test_four_search_probes_share_one_bounded_candidate_world(agent_factory):
    records = [atom(f"property{i}", str(i), f"m{i}") for i in range(20)]
    controller = {
        "decision": "SEARCH",
        "queries": [f"property{i}" for i in range(10, 14)],
    }
    agent, calls = agent_factory(records, controller)
    result = agent.query("Explain the combined outcome")
    extra = result.extra

    assert len(calls) == 2
    assert len(extra["controller_search_queries"]) == 4
    assert set(extra["base_world_ids"]) <= set(extra["candidate_world_ids"])
    assert len(extra["candidate_world_ids"]) <= 16
    assert set(extra["final_context_ids"]) <= set(extra["candidate_world_ids"])
    assert len(extra["final_context_ids"]) <= 16


def test_fallback_context_only_deduplicates_structural_duplicates(agent_factory):
    records = [
        atom("status", "ready", "m1"),
        atom("status", "ready", "m2"),
        atom("owner", "alice", "m3"),
        atom("region", "apac", "m4"),
    ]
    records[1]["evidence_ids"] = records[0]["evidence_ids"]
    controller = {"decision": "SEARCH", "queries": []}
    agent, _ = agent_factory(records, controller)
    result = agent.query("Summarize the evidence")
    assert len(result.extra["candidate_world_ids"]) >= len(result.extra["final_context_ids"])
    assert len(result.extra["final_context_ids"]) >= 3


def test_active_read_does_not_import_semantic_certificate_or_legacy_planners():
    root = Path(core.__file__).parent
    active = [
        "agent",
        "query",
        "query_rendering",
        "query_answer_runtime",
        "read_advisory",
        "read_question_runtime",
        "read_query_orchestrator",
        "read_structural_resolution",
        "read_usage_contract",
    ]
    forbidden_modules = {
        "read_evidence_certificate",
        "execution",
        "retrieval",
        "planning",
        "read_controller",
        "legacy",
    }
    for name in active:
        tree = ast.parse((root / f"{name}.py").read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level:
                imported = (node.module or "").split(".")[0]
                assert imported not in forbidden_modules, (name, node.module)

    modules = {cls.__module__.split(".")[-1] for cls in SmartMem0Agent.__mro__}
    assert not modules & forbidden_modules
