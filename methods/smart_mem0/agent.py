"""Public SmartMem0 facade assembled from cohesive pipeline components."""

from methods.base import BaseAgent

from .capture import CaptureMixin
from .consolidation import ConsolidationMixin
from .core import CoreMemoryMixin
from .execution import ExecutionMixin
from .planning import PlanningMixin
from .query import QueryMixin
from .read_controller import ReadContractMixin
from .read_option_contract import ReadOptionContractMixin
from .read_execution_contract import ReadExecutionContractMixin
from .read_plan_contract import ReadPlanContractMixin
from .retrieval import RetrievalOperationsMixin
from .write import WriteLifecycleMixin


class SmartMem0Agent(
    # One semantic owner, then deterministic plan/proof overrides. These precede
    # the legacy read helpers so the two-stage contract wins by normal Python MRO.
    ReadContractMixin,
    ReadPlanContractMixin,
    ReadOptionContractMixin,
    ReadExecutionContractMixin,
    QueryMixin,
    ExecutionMixin,
    RetrievalOperationsMixin,
    PlanningMixin,
    WriteLifecycleMixin,
    ConsolidationMixin,
    CaptureMixin,
    CoreMemoryMixin,
    BaseAgent,
):
    """Compact evidence-grounded long-term memory for conversational agents."""

    def __init__(self, *args, **kwargs):
        enable_two_stage = bool(kwargs.pop("enable_two_stage_controller", True))
        super().__init__(*args, **kwargs)
        self.enable_two_stage_controller = enable_two_stage
        # _run_query_retrieval already exposes a stable unified-controller hook.
        # The mixins above own that hook; no repeated semantic validator is used.
        self.enable_legacy_semantic_controller = enable_two_stage
        if enable_two_stage:
            self.enable_unified_controller = True
            self.enable_slot_support_validation = False
