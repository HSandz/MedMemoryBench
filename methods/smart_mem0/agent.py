"""Public SmartMem0 facade assembled from cohesive pipeline components."""

from methods.base import BaseAgent

from .capture import CaptureMixin
from .consolidation import ConsolidationMixin
from .controller import TwoStageControllerMixin
from .core import CoreMemoryMixin
from .execution import ExecutionMixin
from .planning import PlanningMixin
from .query import QueryMixin
from .retrieval import RetrievalOperationsMixin
from .write import WriteLifecycleMixin


class SmartMem0Agent(
    QueryMixin,
    ExecutionMixin,
    RetrievalOperationsMixin,
    TwoStageControllerMixin,
    PlanningMixin,
    WriteLifecycleMixin,
    ConsolidationMixin,
    CaptureMixin,
    CoreMemoryMixin,
    BaseAgent,
):
    """Compact evidence-grounded long-term memory for conversational agents."""

    def __init__(self, *args, **kwargs):
        # Two-stage semantic control is the default read architecture. The
        # existing execution hook is reused to keep the migration small and
        # benchmark-comparable; the implementation now lives in controller.py.
        enable_two_stage = bool(kwargs.pop("enable_two_stage_controller", True))
        super().__init__(*args, **kwargs)
        self.enable_two_stage_controller = enable_two_stage
        self.enable_legacy_semantic_controller = enable_two_stage
        if enable_two_stage:
            self.enable_unified_controller = True
            # The controller owns semantic sufficiency. Typed executor proof is
            # deterministic by default, avoiding a repeated validator prompt.
            self.enable_slot_support_validation = False
