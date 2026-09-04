"""MedMemoryBench dataset module."""

from .dataset import MedMemoryBenchDataset, MedSession, MedQuery
from .evaluator import MedMemoryBenchEvaluator, evaluate_medmemorybench
from .smart_mem0_batch_integration import install_smart_mem0_batch_integration

# Install transport/accounting behavior after evaluator import so the registered
# evaluator and direct class users share the same resume-safe batch semantics.
install_smart_mem0_batch_integration(MedMemoryBenchEvaluator)

__all__ = [
    "MedMemoryBenchDataset",
    "MedSession",
    "MedQuery",
    "MedMemoryBenchEvaluator",
    "evaluate_medmemorybench",
]
