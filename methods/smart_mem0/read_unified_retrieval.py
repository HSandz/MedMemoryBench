"""Single active Retrieval Executor owner for SmartMem0 READ."""

from .read_lean_execution_adapter import ReadLeanExecutionAdapterMixin
from .read_retrieval_executor import ReadRetrievalExecutorMixin


class UnifiedRetrievalExecutorMixin(
    ReadLeanExecutionAdapterMixin,
    ReadRetrievalExecutorMixin,
):
    """Fold lean-plan compatibility and family recall behind one active owner."""

    UNIFIED_RETRIEVAL_VERSION = "unified-retrieval-executor-v1"

    def prepare_batch_query(self, question, system_message=None, **kwargs):
        prepared = super().prepare_batch_query(
            question, system_message=system_message, **kwargs
        )
        extra = prepared.setdefault("extra", {})
        extra["unified_retrieval_version"] = self.UNIFIED_RETRIEVAL_VERSION
        extra["retrieval_single_owner"] = True
        return prepared
