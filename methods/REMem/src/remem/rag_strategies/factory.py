from typing import Dict, Type

from .base_strategy import RAGStrategy
from .episodic_gist_strategy import EpisodicGistStrategy
from .passage_triple_strategy import PassageTripleStrategy
from .temporal_strategy import TemporalStrategy


class RAGStrategyFactory:
    """
    Factory class for creating RAG strategies based on extraction method.
    """

    _strategies: Dict[str, Type[RAGStrategy]] = {
        "openie": PassageTripleStrategy,
        "episodic_gist": EpisodicGistStrategy,
        "temporal": TemporalStrategy,
    }

    @classmethod
    def create_strategy(cls, extract_method: str, remem_instance) -> RAGStrategy:
        """
        Create a RAG strategy based on the extraction method.

        Args:
            extract_method: The extraction method name
            remem_instance: The ReMem instance

        Returns:
            RAGStrategy: The appropriate strategy instance

        Raises:
            ValueError: If the extraction method is not supported
        """
        if extract_method not in cls._strategies:
            raise ValueError(
                f"Unsupported extraction method: {extract_method}. "
                f"Supported methods: {list(cls._strategies.keys())}"
            )

        strategy_class = cls._strategies[extract_method]
        return strategy_class(remem_instance)

    @classmethod
    def register_strategy(cls, extract_method: str, strategy_class: Type[RAGStrategy]):
        """
        Register a new strategy for an extraction method.

        Args:
            extract_method: The extraction method name
            strategy_class: The strategy class to register
        """
        cls._strategies[extract_method] = strategy_class

    @classmethod
    def get_supported_methods(cls) -> list[str]:
        """Get list of supported extraction methods."""
        return list(cls._strategies.keys())
