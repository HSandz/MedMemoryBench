"""Phase-6 architecture-lock regressions for SmartMem0 READ."""

from methods.smart_mem0.agent import SmartMem0Agent
from methods.smart_mem0.planning import PlanningMixin
from methods.smart_mem0.read_candidate_set import ReadCandidateSetMixin
from methods.smart_mem0.read_lean_execution_adapter import ReadLeanExecutionAdapterMixin
from methods.smart_mem0.read_option_contract import ReadOptionContractMixin
from methods.smart_mem0.read_query_orchestrator import ReadQueryOrchestratorMixin
from methods.smart_mem0.read_retrieval_executor import ReadRetrievalExecutorMixin
from methods.smart_mem0.read_unified_retrieval import UnifiedRetrievalExecutorMixin


def test_legacy_planning_mixin_is_not_in_active_agent_mro():
    assert PlanningMixin not in SmartMem0Agent.__mro__


def test_candidate_set_is_the_only_direct_candidate_recall_owner():
    assert ReadCandidateSetMixin in SmartMem0Agent.__bases__
    assert ReadOptionContractMixin not in SmartMem0Agent.__bases__
    assert issubclass(ReadCandidateSetMixin, ReadOptionContractMixin)


def test_retrieval_facade_is_the_only_direct_retrieval_owner():
    assert UnifiedRetrievalExecutorMixin in SmartMem0Agent.__bases__
    assert ReadLeanExecutionAdapterMixin not in SmartMem0Agent.__bases__
    assert ReadRetrievalExecutorMixin not in SmartMem0Agent.__bases__
    assert issubclass(
        UnifiedRetrievalExecutorMixin, ReadLeanExecutionAdapterMixin
    )
    assert issubclass(
        UnifiedRetrievalExecutorMixin, ReadRetrievalExecutorMixin
    )


def test_two_stage_audit_accepts_controller_plus_one_answer():
    extra = {
        "semantic_controller": {"called": True},
        "query_tokens": {
            "controller": 80,
            "fast_gate": 0,
            "planner": 0,
            "slot_validation": 0,
            "replan": 0,
            "answer": 0,
        },
        "slot_validation": [],
    }
    audit = ReadQueryOrchestratorMixin._two_stage_audit(
        extra, terminal=False, answer_called=True
    )
    assert audit["valid"] is True
    assert audit["total_calls"] == 2


def test_two_stage_audit_accepts_terminal_controller_only():
    extra = {
        "semantic_controller": {"called": True},
        "query_tokens": {
            "controller": 80,
            "fast_gate": 0,
            "planner": 0,
            "slot_validation": 0,
            "replan": 0,
            "answer": 0,
        },
        "slot_validation": [],
    }
    audit = ReadQueryOrchestratorMixin._two_stage_audit(
        extra, terminal=True, answer_called=False
    )
    assert audit["valid"] is True
    assert audit["total_calls"] == 1


def test_middle_llm_tokens_are_a_hard_violation():
    extra = {
        "semantic_controller": {"called": True},
        "query_tokens": {
            "controller": 80,
            "fast_gate": 0,
            "planner": 12,
            "slot_validation": 0,
            "replan": 0,
            "answer": 0,
        },
        "slot_validation": [],
    }
    audit = ReadQueryOrchestratorMixin._two_stage_audit(
        extra, terminal=False, answer_called=False
    )
    assert audit["valid"] is False
    assert "MIDDLE_LLM_ACTIVITY" in audit["violations"]


def test_terminal_answer_must_not_call_answer_llm():
    extra = {
        "semantic_controller": {"called": True},
        "query_tokens": {
            "controller": 80,
            "fast_gate": 0,
            "planner": 0,
            "slot_validation": 0,
            "replan": 0,
            "answer": 0,
        },
        "slot_validation": [],
    }
    audit = ReadQueryOrchestratorMixin._two_stage_audit(
        extra, terminal=True, answer_called=True
    )
    assert audit["valid"] is False
    assert "TERMINAL_ANSWER_REGENERATED" in audit["violations"]
