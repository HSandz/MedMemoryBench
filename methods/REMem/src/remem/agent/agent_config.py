"""
Configuration for Multi-Turn Agent Framework

This module contains configuration classes and utilities for the multi-turn agent framework.
"""

import json
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Dict, Optional


class SearchStrategy(Enum):
    """Available search strategies for graph exploration."""

    BREADTH_FIRST = "breadth_first"
    DEPTH_FIRST = "depth_first"
    ADAPTIVE = "adaptive"
    SIMILARITY_GUIDED = "similarity_guided"


class ReasoningMode(Enum):
    """Available reasoning modes for the agent."""

    STEP_BY_STEP = "step_by_step"
    PARALLEL = "parallel"
    HYBRID = "hybrid"


@dataclass
class AgentConfig:
    """Configuration for the multi-turn agent."""

    # Basic agent parameters
    max_steps: int = 5
    exploration_radius: int = 3
    search_strategy: SearchStrategy = SearchStrategy.BREADTH_FIRST
    reasoning_mode: ReasoningMode = ReasoningMode.STEP_BY_STEP

    # Agent variant selection
    fixed_tools: bool = False  # If True, use semantic_retrieve + output_answer only
    fixed_retrieval_tool: str = "semantic_retrieve"  # Which retrieval tool to use in fixed mode

    # Stopping criteria
    min_steps: int = 2
    max_nodes_per_step: int = 15

    # Graph exploration parameters
    max_neighbor_depth: int = 2
    prioritize_recent_nodes: bool = True
    node_type_weights: Optional[Dict[str, float]] = None

    # LLM and reasoning parameters
    reasoning_temperature: float = 0.7
    max_reasoning_tokens: int = 512
    enable_self_reflection: bool = True

    # Performance and optimization
    enable_caching: bool = True
    parallel_reasoning: bool = False
    batch_size: int = 1

    def __post_init__(self):
        """Initialize default values after creation."""
        if self.node_type_weights is None:
            self.node_type_weights = {"entity": 1.0, "verbatim": 0.8, "gists": 0.6, "facts": 1.2}

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        result = asdict(self)
        # Convert enums to strings
        result["search_strategy"] = self.search_strategy.value
        result["reasoning_mode"] = self.reasoning_mode.value
        return result

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "AgentConfig":
        """Create config from dictionary."""
        # Convert string enums back to enum objects
        if "search_strategy" in config_dict:
            config_dict["search_strategy"] = SearchStrategy(config_dict["search_strategy"])
        if "reasoning_mode" in config_dict:
            config_dict["reasoning_mode"] = ReasoningMode(config_dict["reasoning_mode"])

        return cls(**config_dict)

    def save_to_file(self, filepath: str):
        """Save configuration to JSON file."""
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_from_file(cls, filepath: str) -> "AgentConfig":
        """Load configuration from JSON file."""
        with open(filepath, "r") as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


@dataclass
class PromptConfig:
    """Configuration for agent prompts."""

    # Node selection prompts
    node_selection_template: str = """
Given the query and current reasoning state, identify the most relevant nodes to explore next.

Query: {query}
Current reasoning: {current_reasoning}
Available nodes in neighborhood: {available_nodes}
Already visited: {visited_nodes}

Select {max_nodes} most promising nodes to explore next and explain your reasoning.
Focus on nodes that can help answer the query or provide missing information.

Response format:
Selected nodes: [node1, node2, node3]
Reasoning: Brief explanation of why these nodes were chosen
"""

    # Reasoning synthesis prompts
    synthesis_template: str = """
Based on the retrieved information and previous reasoning steps, provide an intermediate answer.

Query: {query}
Retrieved passages: {passages}
Previous reasoning: {previous_reasoning}

Provide:
1. Current understanding/partial answer
2. What information is still needed (if any)
3. Whether to continue exploring or provide final answer

Response format:
Current answer: [your current understanding]
Missing info: [what's still needed]
Continue: [yes/no]
"""

    # Final answer generation prompt
    final_answer_template: str = """
Based on all the reasoning steps and retrieved information, provide a comprehensive final answer.

Query: {query}
All reasoning steps: {reasoning_chain}
All retrieved passages: {all_passages}

Provide a clear, comprehensive answer that addresses the query based on the available evidence.
If there are gaps in information, acknowledge them explicitly.

Final Answer: [comprehensive answer]
"""

    # Self-reflection prompt
    reflection_template: str = """
Review your reasoning process and identify potential improvements or alternative approaches.

Query: {query}
Reasoning chain: {reasoning_chain}

Reflection:
1. What went well in the reasoning process?
2. What could be improved?
3. Are there alternative exploration paths?
4. How well does the current answer address the query?

Response format:
Strengths: [what went well]
Improvements: [what could be better]
Alternatives: [other approaches]
Answer assessment: [analysis of answer quality]
"""


def create_default_config() -> AgentConfig:
    """Create a default agent configuration."""
    return AgentConfig()


def create_fast_config() -> AgentConfig:
    """Create a configuration optimized for speed."""
    return AgentConfig(
        max_steps=3,
        exploration_radius=2,
        search_strategy=SearchStrategy.BREADTH_FIRST,
        max_nodes_per_step=3,
        enable_caching=True,
        parallel_reasoning=True,
    )


def create_thorough_config() -> AgentConfig:
    """Create a configuration optimized for thoroughness."""
    return AgentConfig(
        max_steps=8,
        exploration_radius=4,
        search_strategy=SearchStrategy.ADAPTIVE,
        min_steps=3,
        max_nodes_per_step=6,
        max_neighbor_depth=3,
        enable_self_reflection=True,
        reasoning_temperature=0.5,
    )


def create_adaptive_config() -> AgentConfig:
    """Create a configuration that adapts based on query complexity."""
    return AgentConfig(
        max_steps=6,
        exploration_radius=3,
        search_strategy=SearchStrategy.ADAPTIVE,
        reasoning_mode=ReasoningMode.HYBRID,
        max_nodes_per_step=4,
        enable_self_reflection=True,
        prioritize_recent_nodes=True,
        node_type_weights={"entity": 1.2, "verbatim": 1.0, "gists": 0.8, "facts": 1.5},
    )


def get_config_by_name(config_name: str) -> AgentConfig:
    """Get predefined configuration by name."""
    configs = {
        "default": create_default_config,
        "fast": create_fast_config,
        "thorough": create_thorough_config,
        "adaptive": create_adaptive_config,
    }

    if config_name not in configs:
        available = list(configs.keys())
        raise ValueError(f"Unknown config name '{config_name}'. Available: {available}")

    return configs[config_name]()


# Example configurations for different use cases
EXAMPLE_CONFIGS = {
    "qa_focused": {
        "max_steps": 4,
        "search_strategy": "breadth_first",
        "node_type_weights": {"entity": 1.0, "verbatim": 1.2, "gists": 0.9, "facts": 1.1},
    },
    "exploration_heavy": {
        "max_steps": 7,
        "exploration_radius": 4,
        "search_strategy": "adaptive",
        "max_neighbor_depth": 3,
    },
    "memory_efficient": {
        "max_steps": 3,
        "exploration_radius": 2,
        "max_nodes_per_step": 3,
        "enable_caching": False,
        "batch_size": 1,
    },
}
