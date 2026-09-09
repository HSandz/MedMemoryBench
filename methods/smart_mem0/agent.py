"""Public SmartMem0 facade assembled from the locked two-stage READ architecture."""

from methods.base import BaseAgent
from .capture import CaptureMixin
from .consolidation import ConsolidationMixin
from .core import CoreMemoryMixin
from .execution import ExecutionMixin
from .query import QueryMixin
from .read_answerability_contract import ReadAnswerabilityContractMixin
from .read_terminal_answer_contract import ReadTerminalAnswerContractMixin
from .read_answer_or_plan_contract import ReadAnswerOrPlanContractMixin
from .read_answer_sensitive_controller import ReadAnswerSensitiveControllerMixin
from .read_candidate_set import ReadCandidateSetMixin
from .read_certificate_contract import ReadCertificateContractMixin
from .read_contrastive_candidate_policy import ReadContrastiveCandidatePolicyMixin
from .read_controller import ReadContractMixin
from .read_temporal_contract import ReadTemporalContractMixin
from .read_execution_contract import ReadExecutionContractMixin
from .read_plan_contract import ReadPlanContractMixin
from .read_requirement_contract import ReadRequirementContractMixin
from .read_reasoning_bridge import ReadReasoningBridgeMixin
from .read_reasoning_completion import ReadReasoningCompletionMixin
from .read_query_orchestrator import ReadQueryOrchestratorMixin
from .read_evidence_policy import ReadEvidencePolicyMixin
from .read_evidence_precision import ReadEvidencePrecisionMixin
from .read_progressive_retrieval import ReadProgressiveRetrievalMixin
from .read_retrieval_fusion import ReadRetrievalFusionMixin
from .read_evidence_resolve import ReadEvidenceResolveMixin
from .read_semantic_closure import ReadSemanticClosureMixin
from .read_runtime_support import ReadRuntimeSupportMixin
from .read_unified_retrieval import UnifiedRetrievalExecutorMixin
from .read_proof_context import UnifiedProofContextMixin
from .read_usage_contract import ReadUsageContractMixin
from .retrieval import RetrievalOperationsMixin
from .write import WriteLifecycleMixin


class SmartMem0Agent(
    ReadQueryOrchestratorMixin,
    ReadReasoningCompletionMixin,
    ReadProgressiveRetrievalMixin,
    ReadRetrievalFusionMixin,
    ReadEvidenceResolveMixin,
    ReadAnswerSensitiveControllerMixin,
    ReadSemanticClosureMixin,
    ReadEvidencePrecisionMixin,
    ReadContrastiveCandidatePolicyMixin,
    ReadEvidencePolicyMixin,
    ReadAnswerabilityContractMixin,
    ReadTerminalAnswerContractMixin,
    ReadCertificateContractMixin,
    ReadCandidateSetMixin,
    ReadAnswerOrPlanContractMixin,
    ReadRequirementContractMixin,
    ReadReasoningBridgeMixin,
    UnifiedProofContextMixin,
    UnifiedRetrievalExecutorMixin,
    ReadContractMixin,
    ReadTemporalContractMixin,
    ReadPlanContractMixin,
    ReadUsageContractMixin,
    ReadExecutionContractMixin,
    QueryMixin,
    ExecutionMixin,
    RetrievalOperationsMixin,
    ReadRuntimeSupportMixin,
    WriteLifecycleMixin,
    ConsolidationMixin,
    CaptureMixin,
    CoreMemoryMixin,
    BaseAgent,
):
    """Compact evidence-grounded long-term memory with one locked READ pipeline."""

    MAX_TWO_STAGE_READ_LLM_CALLS = 2

    def __init__(self, *args, **kwargs):
        requested = kwargs.pop("enable_two_stage_controller", True)
        if requested is False:
            raise ValueError(
                "SmartMem0 legacy READ planner path was removed in architecture Phase 6"
            )
        super().__init__(*args, **kwargs)
        self.enable_two_stage_controller = True
        self.max_read_llm_calls = self.MAX_TWO_STAGE_READ_LLM_CALLS

        # Keep old configuration attributes explicitly disabled so downstream
        # tooling cannot accidentally reactivate a removed middle LLM stage.
        self.enable_planner = False
        self.enable_replan = False
        self.enable_zero_result_recovery = True
        self.enable_planner_repair = False
        self.enable_slot_support_validation = False
