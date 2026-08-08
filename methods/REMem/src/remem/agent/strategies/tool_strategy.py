"""
Retrieval strategy classes for different agent modes.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from remem.agent import AgentConfig


@dataclass
class RetrievalResult:
    """Result of a retrieval operation."""

    chunk_ids: List[str]
    chunk_scores: List[float]
    logs: Dict[str, Any]
    final_answer: str


class ToolStrategy(ABC):
    """Base class for different retrieval strategies."""

    def __init__(self, agent_config: AgentConfig):
        self.agent_config = agent_config

    @abstractmethod
    def retrieve(
        self,
        query: str,
        max_results: int = 10,
        return_chunk: Optional[str] = None,
        gold_answer: Optional[str] = None,
        question_metadata: Optional[Dict[str, Any]] = None,
    ) -> RetrievalResult:
        """Execute the retrieval strategy."""
        pass
