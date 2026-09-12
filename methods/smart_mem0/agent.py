"""Public SmartMem0 facade assembled from the locked two-stage READ architecture."""

from methods.base import BaseAgent
from .capture import CaptureMixin
from .consolidation import ConsolidationMixin
from .core import CoreMemoryMixin
from .execution import ExecutionMixin
from .query import QueryMixin
from .read_query_orchestrator import ReadQueryOrchestratorMixin
from .read_question_runtime import QuestionReadRuntimeMixin
from .read_runtime_support import ReadRuntimeSupportMixin
from .retrieval import RetrievalOperationsMixin
from .write import WriteLifecycleMixin


class SmartMem0Agent(
    ReadQueryOrchestratorMixin,
    QuestionReadRuntimeMixin,
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
                "SmartMem0 stable READ requires the single semantic controller"
            )

        # These were experiment-time authorities in v5. They are consumed only for
        # backwards-compatible configs and are no longer part of the active architecture.
        for legacy_key in (
            "enable_planner",
            "enable_unified_controller",
            "enable_slot_support_validation",
            "enable_replan",
            "enable_planner_repair",
            "enable_reasoning_completion",
            "enable_proof_status_gate",
            "enable_proof_expansion",
            "enable_reasoning_source_neighbors",
            "enable_reasoning_prompt_strengthening",
        ):
            kwargs.pop(legacy_key, None)

        zero_result_recovery = kwargs.pop("enable_zero_result_recovery", True)

        super().__init__(*args, **kwargs)
        self.enable_two_stage_controller = True
        self.max_read_llm_calls = self.MAX_TWO_STAGE_READ_LLM_CALLS

        # Removed middle/control authorities stay explicitly disabled for telemetry and
        # restored-request compatibility.
        self.enable_planner = False
        self.enable_replan = False
        self.enable_planner_repair = False
        self.enable_slot_support_validation = False
        self.enable_reasoning_completion = False
        self.enable_proof_status_gate = False
        self.enable_proof_expansion = False
        self.enable_reasoning_source_neighbors = False
        self.enable_reasoning_prompt_strengthening = False

        # The only adaptive second retrieval pass is structural zero-hit recovery.
        self.enable_zero_result_recovery = bool(zero_result_recovery)
