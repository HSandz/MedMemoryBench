"""Regressions for SmartMem0's answer-or-plan read invariant."""

from types import SimpleNamespace

from benchmarks.medmemorybench import smart_mem0_batch_integration as batch_integration
from methods.smart_mem0.agent import SmartMem0Agent
from methods.smart_mem0.contracts import QueryFrame


def _agent():
    agent = object.__new__(SmartMem0Agent)
    agent._belief_status = {}
    agent._state_heads = {}
    agent._memories = []
    agent._relations = []
    agent.subject_aliases = {"patient": "primary_user"}
    return agent


def _memory(memory_id, claim, value, *, event_time="", document_time=""):
    return {
        "id": memory_id,
        "claim": claim,
        "value": value,
        "verbatim_value": "",
        "kind": "FACT",
        "semantic_role": "OBSERVATION",
        "memory_tier": "HOT",
        "subject_id": "primary_user",
        "subject": "patient",
        "scope": "",
        "state_key": "",
        "object_anchor": "",
        "entities": [],
        "scope_entities": [],
        "planning_tags": [],
        "assertion_mode": "DIRECT",
        "stance": "AFFIRM",
        "event_time": event_time,
        "document_time": document_time,
        "origin_document_time": document_time,
        "_status": "active",
    }


def test_connected_derived_obligation_cannot_be_hidden_by_direct_projection():
    agent = _agent()
    question = "Which antibiotic was the patient instructed to avoid?"
    ir = agent._rc_normalize_ir(
        {
            "answer_type": "ENTITY",
            "subject_span": "patient",
            "requirements": [
                {
                    "id": "r1",
                    "grounding_kind": "QUESTION",
                    "focus_span": "antibiotic",
                    "target": "antibiotic instructed to avoid",
                    "retrieval_hint": "antibiotic warning or instruction",
                },
                {
                    "id": "r2",
                    "grounding_kind": "DERIVED",
                    "focus_span": "",
                    "target": "medication contraindications",
                    "retrieval_hint": "drug contraindications",
                },
            ],
            "relations": [
                {"type": "DEPENDS_ON", "from": "r1", "to": "r2"}
            ],
            "candidate": {"answer": "cefuroxime", "support_ref": "$seed0"},
        },
        question,
        QueryFrame(),
    )

    assert [item["id"] for item in ir["requirements"]] == ["r1", "r2"]
    assert ir["candidate"] is None
    assert agent._aop_direct_projection(ir, question) is None


def test_advice_question_never_uses_direct_candidate_shortcut():
    agent = _agent()
    question = "My neck feels sore. Can I take some painkillers?"
    ir = agent._rc_normalize_ir(
        {
            "answer_type": "TEXT",
            "requirements": [
                {
                    "id": "r1",
                    "grounding_kind": "QUESTION",
                    "focus_span": "My neck feels sore",
                    "target": "current neck soreness",
                }
            ],
            "candidate": {"answer": "yes", "support_ref": "$seed0"},
        },
        question,
        QueryFrame(),
    )
    assert ir["candidate"] is None
    assert agent._aop_direct_projection(ir, question) is None


def test_document_date_candidate_can_finish_from_one_grounded_seed():
    agent = _agent()
    seed = _memory(
        "m1",
        "The patient's GADA antibody was strongly positive (>2000 U/mL).",
        ">2000 U/mL",
        event_time="2024-03-23",
        document_time="2024-03-23",
    )
    agent._memories = [seed]
    question = "When was it documented that the patient's GADA antibody was strongly positive?"
    ir = agent._rc_normalize_ir(
        {
            "answer_type": "DATE",
            "requirements": [
                {
                    "id": "r1",
                    "grounding_kind": "QUESTION",
                    "focus_span": "GADA antibody was strongly positive",
                    "target": "GADA strongly positive result",
                    "time_constraint": {
                        "axis": "document_time",
                        "relation": "LOCATE",
                    },
                }
            ],
            "candidate": {"answer": "2024-03-23", "support_ref": "$seed0"},
        },
        question,
        QueryFrame(),
    )
    projected = agent._aop_direct_projection(ir, question)
    supports, reason = agent._authorize_controller_answer(
        projected, [seed], QueryFrame()
    )
    assert reason == "AUTHORIZED"
    assert [memory["id"] for memory in supports] == ["m1"]


def test_restored_precomputed_batch_request_is_finalized_locally(monkeypatch):
    prepared = {
        "messages": [{"role": "user", "content": "unused"}],
        "precomputed_answer": "SAID (autoimmune diabetes)",
        "extra": {},
    }
    monkeypatch.setattr(
        batch_integration,
        "restore_prepared_query",
        lambda _saved: prepared,
    )

    class BatchClient:
        def __init__(self):
            self.run_calls = []

        def get_saved_request(self, _stage, _request_id):
            return SimpleNamespace(metadata={})

        def run_stage(self, stage, requests):
            self.run_calls.append((stage, list(requests)))
            return {}

    class AgentManager:
        def __init__(self):
            self.calls = []

        def finalize_batch_query(self, prepared_query, content, **kwargs):
            self.calls.append((prepared_query, content, kwargs))
            return SimpleNamespace(output=prepared_query["precomputed_answer"])

    class Evaluator:
        def _evaluate_batch_queries(self, unit, memory_time_per_query):
            raise AssertionError("original path should not run")

        def _generate_report(self, *args, **kwargs):
            raise AssertionError("not used")

    batch_integration.install_smart_mem0_batch_integration(Evaluator)
    evaluator = Evaluator()
    evaluator.method_config = SimpleNamespace(
        method_name="smart_mem0",
        model=SimpleNamespace(
            temperature=0.0,
            max_completion_tokens=128,
            max_tokens=128,
        ),
    )
    evaluator._checkpoint_manager = None
    evaluator._is_deferred_judge_query = lambda _query_id: False
    evaluator._log = lambda *_args, **_kwargs: None
    evaluator.prompt_manager = SimpleNamespace(
        format_query=lambda **kwargs: kwargs["question"]
    )
    evaluator.agent_manager = AgentManager()
    batch_client = BatchClient()
    evaluator._get_batch_client = lambda: batch_client
    evaluator._score_agent_response = (
        lambda query, response, **kwargs: response.output
    )

    query = SimpleNamespace(
        query_id="q1",
        query_type="entity_exact_match",
        question="diagnosis?",
    )
    unit = SimpleNamespace(
        unit_id="u1",
        context_id="c1",
        queries_to_evaluate=[query],
    )
    result = evaluator._evaluate_batch_queries(unit, 0.0)

    assert result == ["SAID (autoimmune diabetes)"]
    assert batch_client.run_calls == []
    assert evaluator.agent_manager.calls[0][1] == ""
    assert evaluator.agent_manager.calls[0][2]["input_tokens"] == 0
    assert evaluator.agent_manager.calls[0][2]["output_tokens"] == 0
