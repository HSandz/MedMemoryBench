"""Public SmartMem0 facade assembled from cohesive pipeline components."""

from methods.base import BaseAgent
from .capture import CaptureMixin
from .consolidation import ConsolidationMixin
from .core import CoreMemoryMixin
from .execution import ExecutionMixin
from .planning import PlanningMixin
from .query import QueryMixin
from .read_controller import ReadContractMixin
from .read_temporal_contract import ReadTemporalContractMixin
from .read_architecture_contract import ReadArchitectureContractMixin
from .read_route_contract import ReadRouteContractMixin
from .read_option_contract import ReadOptionContractMixin
from .read_execution_contract import ReadExecutionContractMixin
from .read_plan_contract import ReadPlanContractMixin
from .read_usage_contract import ReadUsageContractMixin
from .retrieval import RetrievalOperationsMixin
from .write import WriteLifecycleMixin

class SmartMem0Agent(
    ReadArchitectureContractMixin, ReadContractMixin, ReadTemporalContractMixin,
    ReadRouteContractMixin, ReadPlanContractMixin, ReadUsageContractMixin,
    ReadOptionContractMixin, ReadExecutionContractMixin, QueryMixin,
    ExecutionMixin, RetrievalOperationsMixin, PlanningMixin, WriteLifecycleMixin,
    ConsolidationMixin, CaptureMixin, CoreMemoryMixin, BaseAgent,
):
    """Compact evidence-grounded long-term memory for conversational agents."""
    MAX_TWO_STAGE_READ_LLM_CALLS = 2

    def __init__(self, *args, **kwargs):
        enable_two_stage = bool(kwargs.pop("enable_two_stage_controller", True))
        super().__init__(*args, **kwargs)
        self.enable_two_stage_controller = enable_two_stage
        self.max_read_llm_calls = self.MAX_TWO_STAGE_READ_LLM_CALLS
        self.enable_legacy_semantic_controller = enable_two_stage
        if enable_two_stage:
            self.enable_unified_controller = True
            self.enable_slot_support_validation = False
            self.enable_replan = False
            self.enable_planner_repair = False
