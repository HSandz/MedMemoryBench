from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .agent_config import AgentConfig, PromptConfig


@dataclass
class AgentState:
    """Maintains the state of the multi-turn agent during reasoning process."""

    # Core state
    query: str
    current_step: int = 0
    max_steps: int = 5

    # Node tracking (simplified)
    visited_nodes: set = field(default_factory=set)
    current_focus_nodes: List[str] = field(default_factory=list)

    # Content and reasoning
    retrieved_passages: List[Dict[str, Any]] = field(default_factory=list)
    reasoning_chain: List[str] = field(default_factory=list)

    # Final answer tracking
    final_answer_generated: bool = False
    final_answer: str = ""

    # Question metadata including date information
    question_metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for serialization."""
        return {
            "query": self.query,
            "current_step": self.current_step,
            "max_steps": self.max_steps,
            "visited_nodes": list(self.visited_nodes),
            "current_focus_nodes": self.current_focus_nodes,
            "retrieved_passages": self.retrieved_passages,
            "reasoning_chain": self.reasoning_chain,
            "final_answer_generated": self.final_answer_generated,
            "final_answer": self.final_answer,
            "question_metadata": self.question_metadata,
        }


class GraphAgent:
    """
    Interactive agent on memory graph with ReAct-style reasoning and tool-use.

    Uses strategy pattern to handle different retrieval modes:
    - Fixed tools: semantic_retrieve + output_answer only
    - Flexible tools: dynamic tool selection based on context
    """

    def __init__(
        self,
        llm_model,
        remem_instance,
        node_chunks_dict: Dict[str, str],
        agent_config: Optional[AgentConfig] = None,
        prompt_config: Optional[PromptConfig] = None,
        logger=None,
    ):
        """Initialize the agent with strategy pattern."""
        self.llm_client = llm_model
        self.remem = remem_instance
        self.node_chunks_dict = node_chunks_dict

        # Initialize agent config
        self.agent_config = self._initialize_agent_config(agent_config, remem_instance)
        self.prompt_config = prompt_config or PromptConfig()

        # Setup logging
        self.logger = self._setup_logger(logger)

        # Initialize shared utilities
        self._initialize_utils()

        # Initialize strategies
        self._initialize_strategies()

    def _initialize_agent_config(self, agent_config: Optional[AgentConfig], remem_instance) -> AgentConfig:
        """Initialize agent configuration from various sources."""
        if agent_config is None and remem_instance and hasattr(remem_instance, "global_config"):
            global_config = remem_instance.global_config
            return AgentConfig(
                fixed_tools=getattr(global_config, "agent_fixed_tools", False),
                max_steps=getattr(global_config, "agent_max_steps", 2),
                fixed_retrieval_tool=getattr(global_config, "agent_fixed_retrieval_tool", "semantic_retrieve"),
            )
        return agent_config or AgentConfig()

    def _setup_logger(self, logger):
        """Setup logging for the agent."""
        if logger:
            return logger

        from remem.utils.logging_utils import get_logger

        return get_logger(__name__, log_file="agent_graph_agent.log")

    def _initialize_utils(self):
        """Initialize shared utility classes."""
        # Agent session tracking
        self.session_logs = {"num_steps": 0, "final_results": {}, "llm_interactions": []}

        # Initialize utility classes
        from .answer_generator import FinalAnswerGenerator
        from .logging_utils import AgentLogger
        from .node_utils import NodeUtils

        self.agent_logger = AgentLogger(self.logger, self.session_logs)
        if self.remem.global_config.qa_reader == "remem":
            self.answer_generator = FinalAnswerGenerator(self.llm_client, self.logger, self._log_llm_interaction)
        elif self.remem.global_config.qa_reader == "tiser":
            from remem.agent.answer_generator_tiser import TiserFinalAnswerGenerator

            self.answer_generator = TiserFinalAnswerGenerator(self.llm_client, self.logger, self._log_llm_interaction)
        else:
            raise ValueError(f"Unsupported qa_reader: {self.agent_config.qa_reader}")
        self.node_utils = NodeUtils()

        # Initialize tool selector
        from remem.agent.tools.tool_selector import ToolSelector

        self.tool_selector = ToolSelector(self.remem, self.llm_client) if self.remem else None

        # Set up LLM interaction callback
        if self.tool_selector:
            self.tool_selector.set_llm_interaction_callback(self._log_llm_interaction)

    def _initialize_strategies(self):
        """Initialize retrieval strategies."""
        tool_strategy_params = (
            self.agent_config,
            self.tool_selector,
            self.node_utils,
            self.agent_logger,
            self.answer_generator,
            self.node_chunks_dict,
            self.remem,
            self.logger,
        )

        from remem.agent.strategies.fixed_tools_strategy import FixedToolsStrategy
        from remem.agent.strategies.flexible_tools_strategy import FlexibleToolsStrategy

        self.fixed_tools_strategy = FixedToolsStrategy(*tool_strategy_params)
        self.flexible_tools_strategy = FlexibleToolsStrategy(*tool_strategy_params)

    def retrieve_with_agent(
        self,
        query: str,
        max_steps: Optional[int] = None,
        return_chunk: Optional[str] = None,
        beam_size: int = 10,
        gold_answer: Optional[str] = None,
        question_metadata: Optional[Dict[str, Any]] = None,
    ) -> tuple:
        """
        Main entry point for agent-based retrieval.

        Returns same format as traditional PPR retrieval: (sorted_chunk_ids, sorted_chunk_scores, graph_seeds)
        """
        # Check if simple agent variant is requested
        if self.agent_config.fixed_tools:
            result = self.fixed_tools_strategy.retrieve(query, beam_size, return_chunk, gold_answer, question_metadata)
            return result.chunk_ids, result.chunk_scores, result.logs

        # Use flexible strategy
        if max_steps:
            # Temporarily update max_steps for this call
            original_max_steps = self.agent_config.max_steps
            self.agent_config.max_steps = max_steps

            result = self.flexible_tools_strategy.retrieve(
                query, beam_size, return_chunk, gold_answer, question_metadata
            )

            # Restore original max_steps
            self.agent_config.max_steps = original_max_steps
        else:
            result = self.flexible_tools_strategy.retrieve(
                query, beam_size, return_chunk, gold_answer, question_metadata
            )

        return result.chunk_ids, result.chunk_scores, result.logs

    def get_session_logs(self) -> Dict[str, Any]:
        """Get the ordered session logs."""
        return self.agent_logger._get_session_logs()

    def _log_llm_interaction(
        self,
        messages: List[Dict[str, str]],
        response: str,
        metadata: dict,
        cache_hit: bool,
        interaction_type: str = "tool_selection",
        step: Optional[int] = None,
        **kwargs,
    ):
        """Log LLM interaction with metadata."""
        self.agent_logger.log_llm_interaction(
            messages=messages,
            response=response,
            metadata=metadata,
            cache_hit=cache_hit,
            interaction_type=interaction_type,
            step=step,
            **kwargs,
        )
