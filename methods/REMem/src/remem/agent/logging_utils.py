"""
Shared logging utilities for ReMem Agent
"""

import json
import os
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional


class AgentLogger:
    """Centralized logging utilities for agent operations with simplified output."""

    def __init__(self, logger=None, session_logs: Optional[Dict[str, Any]] = None):
        self.logger = logger
        self.session_logs = session_logs or {}
        self.interaction_history = {}  # Store metadata from tool selection for the next step
        self._lock = threading.Lock()  # Thread safety lock

    def log_session_start(self, query: str, max_steps: int, gold_answer: Optional[str] = None):
        """Initialize logging for a new agent session."""
        with self._lock:
            self.session_logs.update({"query": query, "max_steps": max_steps, "num_steps": 0, "llm_interactions": []})

        if self.logger:
            self.logger.info(f"🔍 Query: {query}")
            if gold_answer:
                self.logger.info(f"📄 Gold Answer: {gold_answer}")

    def log_step_start(self, step: int, max_steps: int):
        """Log the start of a reasoning step."""
        if self.logger:
            self.logger.info(f"Step {step}/{max_steps}")

    def log_tool_selection(
        self, tool_type: str, tool_reasoning: str, context, tool_parameters: Optional[Dict[str, Any]] = None
    ):
        """Log tool selection and reasoning."""
        if tool_parameters is None:
            tool_parameters = {}

        if self.logger:
            self.logger.info(f"Tool selected: {tool_type}")
            self.logger.info(f"Tool reasoning: {tool_reasoning}")
            if tool_parameters:
                self.logger.info(f"Tool parameters: {tool_parameters}")

    def log_step(
        self,
        step: int,
        tool_result,
        new_passages: List[Dict[str, Any]],
        context,
        selection_metadata: Optional[Dict[str, Any]] = None,
    ):
        """Log step information combining tool selection and execution results."""
        # Extract tool information
        tool_type = (
            tool_result.tool_type.value if hasattr(tool_result.tool_type, "value") else str(tool_result.tool_type)
        )
        tool_parameters = getattr(tool_result, "parameters", {})

        # Build context summary
        visited_gists = [node for node in context.visited_nodes if node.startswith("gists")]
        visited_facts = [node for node in context.visited_nodes if node.startswith("facts")]
        visited_entities = [node for node in context.visited_nodes if node.startswith("entity")]

        # Get selection metadata from pending store or provided parameter
        step_selection_metadata = selection_metadata or self.interaction_history.get(step, {})

        # Store unified step information - combining selection and execution
        # Ensure step number is consecutive based on existing steps
        with self._lock:
            existing_steps_count = len(self.session_logs.get("llm_interactions", []))
            consecutive_step_number = existing_steps_count + 1

            step_data = {
                "step": consecutive_step_number,
                "function": tool_type,
                "parameters": tool_parameters,
                "reasoning": step_selection_metadata.get("reasoning", getattr(tool_result, "selection_reasoning", "")),
                "observation": tool_result.observation,
                "context_summary": {
                    "visited_gists": len(visited_gists),
                    "visited_facts": len(visited_facts),
                    "visited_entities": len(visited_entities),
                },
                "content_retrieval": {
                    "num_passages_retrieved": len(new_passages),
                    "passages": [
                        {
                            "node_name": p.get("node_name"),
                            "content": p.get("content", ""),
                            "step": consecutive_step_number,  # Use consecutive step number
                            "score": p.get("embedding_score", p.get("score", 0.0)),  # Handle both field names
                        }
                        for p in new_passages
                    ],
                },
                "prompt_tokens": step_selection_metadata.get("prompt_tokens", 0),
                "completion_tokens": step_selection_metadata.get("completion_tokens", 0),
                "finish_reason": step_selection_metadata.get("finish_reason", ""),
                "cache_hit": step_selection_metadata.get("cache_hit", False),
            }

            # Add messages and response if available from selection
            if step_selection_metadata.get("messages"):
                step_data["messages"] = step_selection_metadata["messages"]
            if step_selection_metadata.get("response"):
                step_data["response"] = step_selection_metadata["response"]

            # Add to session logs directly
            if "llm_interactions" not in self.session_logs:
                self.session_logs["llm_interactions"] = []
            self.session_logs["llm_interactions"].append(step_data)
            self.session_logs["num_steps"] = step

            # Clean up used metadata
            if step in self.interaction_history:
                del self.interaction_history[step]

        if self.logger:
            tool_emoji = self._get_tool_emoji(tool_type)
            formatted_observation = self._format_observation(tool_result.observation)
            self.logger.info(
                f"{tool_emoji} {tool_type.replace('_', ' ').title()}: {len(tool_result.nodes_found)} nodes | {formatted_observation}"
            )

    def log_stop_condition(self, reason: str):
        """Log why the agent stopped."""
        if self.logger:
            self.logger.info(f"⏹️ Stopped: {reason}")

    def log_llm_interaction(
        self,
        messages: Optional[List[Dict[str, str]]] = None,
        response: Optional[str] = None,
        metadata: Optional[dict] = None,
        cache_hit: bool = False,
        interaction_type: str = "tool_selection",
        step: Optional[int] = None,
        **kwargs,
    ):
        """Log LLM interaction with simplified structure."""

        # For tool_selection interactions, store metadata for the next unified step
        if interaction_type == "tool_selection":
            with self._lock:
                self.interaction_history[step] = {
                    "prompt_tokens": metadata.get("prompt_tokens", 0) if metadata else 0,
                    "completion_tokens": metadata.get("completion_tokens", 0) if metadata else 0,
                    "finish_reason": metadata.get("finish_reason", "") if metadata else "",
                    "cache_hit": cache_hit,
                    "messages": messages,
                    "response": response,
                }
            return

        # Create simplified interaction log for other types
        interaction_log = {"step": step + 1, "function": interaction_type}

        # For other interaction types, merge metadata directly (if any)
        if metadata:
            interaction_log["prompt_tokens"] = metadata.get("prompt_tokens", None)
            interaction_log["completion_tokens"] = metadata.get("completion_tokens", None)
            interaction_log["finish_reason"] = metadata.get("finish_reason", None)
            interaction_log["cache_hit"] = cache_hit
        if messages:
            interaction_log["messages"] = messages
        if response:
            interaction_log["response"] = response

        # Add to session logs
        with self._lock:
            if "llm_interactions" not in self.session_logs:
                self.session_logs["llm_interactions"] = []
            self.session_logs["llm_interactions"].append(interaction_log)

        # Only log important LLM interactions to reduce verbosity
        if self.logger and not cache_hit:
            self.logger.debug(f"🤖 LLM interaction: {interaction_type}...")

    def get_llm_interaction_summary(self) -> Dict[str, Any]:
        """Generate summary of LLM interactions for session logs."""
        with self._lock:
            interactions = self.session_logs.get("llm_interactions", [])

        if not interactions:
            return {"total_calls": 0}

        # Count interactions by function type (since we no longer have interaction_type)
        function_counts = {}
        total_prompt_tokens = 0
        total_completion_tokens = 0
        cache_hits = 0

        for interaction in interactions:
            function_name = interaction.get("function", "unknown")
            function_counts[function_name] = function_counts.get(function_name, 0) + 1

            # Sum up token usage
            total_prompt_tokens += interaction.get("prompt_tokens", 0)
            total_completion_tokens += interaction.get("completion_tokens", 0)

            # Count cache hits
            if interaction.get("cache_hit", False):
                cache_hits += 1

        return {
            "total_calls": len(interactions),
            "function_usage": function_counts,
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens,
            "total_tokens": total_prompt_tokens + total_completion_tokens,
            "cache_hits": cache_hits,
            "cache_hit_rate": cache_hits / len(interactions) if interactions else 0,
        }

    def save_session_logs(self):
        """Save detailed session logs to file."""
        try:
            # Create agent_session subdirectory if it doesn't exist
            from remem.utils.logging_utils import LOG_DIR

            agent_session_dir = os.path.join(LOG_DIR, "agent_session")
            os.makedirs(agent_session_dir, exist_ok=True)

            # Create filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"agent_session_{timestamp}.json"
            filepath = os.path.join(agent_session_dir, filename)

            # Reorder the session logs for better readability
            ordered_logs = self._get_session_logs()

            # Save to JSON file
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(ordered_logs, f, indent=2, ensure_ascii=False)

            if self.logger:
                self.logger.info(f"Session logs saved to: {filepath}")

        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to save session logs: {e}")

    def _get_session_logs(self) -> Dict[str, Any]:
        """Get session logs in the desired order."""
        with self._lock:
            # Desired order: query, max_steps, num_steps, llm_interactions
            ordered_logs = {}

            # Add keys in the desired order - remove final_results and predicted_answer if empty
            if "query" in self.session_logs:
                ordered_logs["query"] = self.session_logs["query"]
            if "max_steps" in self.session_logs:
                ordered_logs["max_steps"] = self.session_logs["max_steps"]
            if "num_steps" in self.session_logs:
                ordered_logs["num_steps"] = self.session_logs["num_steps"]
            if "llm_interactions" in self.session_logs:
                ordered_logs["llm_interactions"] = self.session_logs["llm_interactions"]

            # Only add predicted_answer if it's not empty
            if self.session_logs.get("predicted_answer"):
                ordered_logs["predicted_answer"] = self.session_logs["predicted_answer"]

            # Only add gold_answer if it exists and is not empty
            if self.session_logs.get("gold_answer"):
                ordered_logs["gold_answer"] = self.session_logs["gold_answer"]

            # Add any remaining keys that weren't explicitly ordered (excluding final_results)
            for key, value in self.session_logs.items():
                if key not in ordered_logs and key != "final_results":
                    ordered_logs[key] = value

            return ordered_logs

    def _format_observation(self, observation: str, max_length: int = 150) -> str:
        """Format observation text, truncating if too long"""
        if not observation:
            return ""

        # Remove extra whitespace characters
        observation = " ".join(observation.split())

        if len(observation) <= max_length:
            return observation

        return observation[:max_length] + "..."

    def _get_tool_emoji(self, tool_type: str) -> str:
        """Get emoji for tool type."""
        emoji_map = {
            "semantic_retrieve": "🔍",
            "lexical_retrieve": "📖",
            "find_entity_contexts": "🔗",
            "find_gist_contexts": "💡",
            "output_answer": "✅",
            "default": "🛠️",
        }
        return emoji_map.get(tool_type, emoji_map["default"])

    def log_metrics(
        self, sample_id: int, sample_metrics: Dict[str, float], running_metrics: Dict[str, float], num_samples: int
    ):
        """Log sample metrics with clean formatting."""
        if not sample_metrics:
            return

        sample_metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in sample_metrics.items()])
        running_metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in running_metrics.items()])

        if self.logger:
            if num_samples > 1:
                self.logger.info(
                    f"📊 Sample {sample_id}: {sample_metrics_str} | Avg: {running_metrics_str} | Total: {num_samples}"
                )
            else:
                self.logger.info(f"📊 {sample_metrics_str}")


def configure_clean_logging():
    """
    Configure clean logging output for console display.
    Automatically sets up simplified, user-friendly log format.
    """
    import logging

    # Set third-party library log levels to reduce noise
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)

    # Get remem related logger
    remem_logger = logging.getLogger("remem")

    # Set custom formatter with clean, concise format
    formatter = logging.Formatter("%(asctime)s - %(message)s", datefmt="%H:%M:%S")

    # Update formatter for existing handlers
    for handler in remem_logger.handlers:
        handler.setFormatter(formatter)

    return remem_logger


def log_sample_metrics(
    sample_id: int, sample_metrics: Dict[str, float], running_metrics: Dict[str, float], total_samples: int, logger=None
):
    """
    Standalone function to log sample metrics with clean formatting.

    Args:
        sample_id: ID of the current sample
        sample_metrics: Dictionary of metric_name -> score for current sample
        running_metrics: Dictionary of metric_name -> average_score across samples
        total_samples: Total number of samples processed
        logger: Optional logger instance (creates temporary one if None)
    """
    if logger is None:
        import logging

        logger = logging.getLogger("metrics")

    agent_logger = AgentLogger(logger=logger)
    agent_logger.log_metrics(sample_id, sample_metrics, running_metrics, total_samples)
