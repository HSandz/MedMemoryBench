"""
Tool Selection Framework for ReMem Agent

This module implements autonomous tool selection for the multi-turn agent,
allowing it to choose between different exploration strategies based on
current context and visited information.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from remem.utils.datetime_utils import parse_flexible_datetime

logger = logging.getLogger(__name__)


def dates_overlap(
    start1: Optional[datetime], end1: Optional[datetime], start2: Optional[datetime], end2: Optional[datetime]
) -> bool:
    """
    Check if two date ranges overlap.

    Two ranges overlap if they share any common time points.
    For example:
    - [1950, 1960] and [1955, 1965] overlap
    - [1950, 1955] and [1960, 1965] do not overlap
    - [1950, 1960] and [1960, 1965] overlap (touching at 1960)

    Args:
        start1, end1: First date range (if start1 is None, extends to past; if end1 is None, extends to future)
        start2, end2: Second date range (if start2 is None, extends to past; if end2 is None, extends to future)

    Returns:
        True if ranges overlap, False otherwise
    """

    def expand_year_only_date(date: datetime) -> datetime:
        """Expand year-only date (Jan 1st 00:00:00) to end of year."""
        if date.month == 1 and date.day == 1 and date.hour == 0 and date.minute == 0 and date.second == 0:
            return datetime(date.year, 12, 31, 23, 59, 59)
        return date

    # If any range is completely undefined, assume overlap
    if not any([start1, end1, start2, end2]):
        return True
    if (not start1 and not end1) or (not start2 and not end2):
        return True

    # Expand year-only dates to full year ranges
    if start1 and end1 and start1 == end1:
        end1 = expand_year_only_date(start1)
    if start2 and end2 and start2 == end2:
        end2 = expand_year_only_date(start2)

    # Handle open-ended ranges with unified logic
    # None start means extends to past (-∞), None end means extends to future (+∞)

    # If either range has open boundaries, check specific overlap conditions
    if not start1 or not end1 or not start2 or not end2:
        # Range 1: [start1 or -∞, end1 or +∞]
        # Range 2: [start2 or -∞, end2 or +∞]

        # Get effective boundaries (use extreme values for None)
        from datetime import datetime as dt

        MIN_DATE = dt(1900, 1, 1)  # Far past
        MAX_DATE = dt(2100, 12, 31)  # Far future

        eff_start1 = start1 or MIN_DATE
        eff_end1 = end1 or MAX_DATE
        eff_start2 = start2 or MIN_DATE
        eff_end2 = end2 or MAX_DATE

        # Two ranges overlap if: start1 <= end2 AND start2 <= end1
        return eff_start1 <= eff_end2 and eff_start2 <= eff_end1

    # Both ranges are closed, use standard overlap check
    # Two ranges [a,b] and [c,d] overlap if: a <= d AND c <= b
    try:
        return start1 <= end2 and start2 <= end1
    except Exception as e:
        logger.warning(f"Date overlap check failed: {e}")
        return True  # If comparison fails, assume overlap


class ToolType(Enum):
    """Available tool types for agent operations."""

    OUTPUT_ANSWER = "output_answer"
    SEMANTIC_RETRIEVE = "semantic_retrieve"
    LEXICAL_RETRIEVE = "lexical_retrieve"
    FIND_GIST_CONTEXTS = "find_gist_contexts"
    FIND_ENTITY_CONTEXTS = "find_entity_contexts"


@dataclass
class ToolContext:
    """Context information for tool selection."""

    query: str
    visited_nodes: set = field(default_factory=set)  # All visited node IDs
    reasoning_step: int = 0
    max_steps: int = 5
    available_entries: Dict[str, List[str]] = field(default_factory=dict)  # entry_type -> list of available entries
    # Simple map from node hash IDs to content previews for LLM understanding
    content_map: Dict[str, str] = field(default_factory=dict)  # node_id -> content_preview
    # Previous steps information for tool selection
    previous_steps: List[Dict[str, Any]] = field(default_factory=list)  # Previous tool selections and observations
    # Preliminary answer from output_answer tool for reference in subsequent interactions
    preliminary_answer: Optional[str] = None
    # Question metadata including date information for temporal reasoning
    question_metadata: Optional[Dict[str, Any]] = None


@dataclass
class ToolResult:
    """Result from a tool execution."""

    tool_type: ToolType
    nodes_found: List[str]
    scores: List[float]
    observation: str
    should_continue: bool
    parameters: Dict[str, Any] = field(default_factory=dict)  # Parameters used for tool execution
    selection_reasoning: str = ""  # LLM reasoning for tool selection
    node_contents: List[str] = field(default_factory=list)  # Content for each node in nodes_found


class BaseTool(ABC):
    """Base class for agent tools."""

    def __init__(self, tool_type: ToolType, remem_instance):
        self.tool_type = tool_type
        self.remem = remem_instance
        self.logger = logging.getLogger(f"{__name__}.{tool_type.value}")

    @abstractmethod
    def get_description(self) -> str:
        """Get tool description for LLM-based selection."""
        pass

    def can_execute(self, context: ToolContext) -> bool:
        """Check if this tool can be executed given the current context."""
        return True  # Default implementation, can be overridden

    @abstractmethod
    def execute(self, context: ToolContext, max_results: int = 5, **kwargs) -> ToolResult:
        """Execute the tool and return results.

        Args:
            context: Tool execution context
            max_results: Maximum number of results to return
            **kwargs: Additional tool-specific parameters (e.g., start_time, end_time, point_in_time)
        """
        pass

    def _get_node_type(self, node_key: str) -> str:
        """Determine node type from node key or graph metadata."""
        try:
            if self.remem and hasattr(self.remem, "episodic_embedding_stores") and self.remem.episodic_embedding_stores:
                for entry_type, store in self.remem.episodic_embedding_stores.items():
                    if node_key in store.hash_id_to_idx:
                        return entry_type
        except Exception as e:
            self.logger.warning(f"Error determining node type for {node_key}: {e}")
            pass
        return "unknown"

    def _get_node_content(self, node_key: str) -> str:
        """Get content for a node."""
        try:
            node_type = self._get_node_type(node_key)
            if self.remem and hasattr(self.remem, "episodic_embedding_stores"):
                store = self.remem.episodic_embedding_stores.get(node_type)
                if store:
                    row = store.get_row(node_key)
                    if row and "content" in row:
                        content = row["content"]
                        if isinstance(content, dict):
                            return content.get("text", str(content))
                        return str(content)
            return node_key
        except Exception as e:
            self.logger.warning(f"Error getting content for {node_key}: {e}")
            return node_key

    def _get_node_metadata(self, node_key: str) -> Dict[str, Any]:
        """Get metadata for a node."""
        try:
            node_type = self._get_node_type(node_key)
            if self.remem and hasattr(self.remem, "episodic_embedding_stores"):
                store = self.remem.episodic_embedding_stores.get(node_type)
                if store:
                    row = store.get_row(node_key)
                    return row.get("metadata", {}) if row else {}
            return {}
        except Exception as e:
            self.logger.warning(f"Error getting metadata for {node_key}: {e}")
            return {}

    def _is_visited(self, node_key: str, context: ToolContext) -> bool:
        """Check if a node has been visited."""
        return node_key in context.visited_nodes

    def _get_fact_qualifiers(self, fact_key: str) -> str:
        """Get qualifiers for a fact as formatted string."""
        try:
            fact_store = self.remem.episodic_embedding_stores.get("facts")
            if fact_store:
                fact_data = fact_store.get_row(fact_key)
                fact_metadata = fact_data.get("metadata", {})

                if "qualifiers" in fact_metadata:
                    qualifiers = fact_metadata["qualifiers"]
                    if qualifiers and qualifiers != "{}":
                        if isinstance(qualifiers, str):
                            qualifiers = eval(qualifiers)

                        formatted_quals = []
                        for k, v in qualifiers.items():
                            if v and v != "null":
                                formatted_quals.append(f"{k}: {v}")

                        if formatted_quals:
                            return "{" + ", ".join(formatted_quals) + "}"
            return ""
        except Exception as e:
            self.logger.warning(f"Error getting qualifiers for {fact_key}: {e}")
            return ""

    def _get_formatted_fact_content_for_retrieval(self, fact_key: str) -> str:
        """Get formatted fact content for internal retrieval (simple readable format)."""
        # Use cache if available (for tools that implement caching)
        cache = getattr(self, "_formatted_fact_cache", None)
        if cache and fact_key in cache:
            return cache[fact_key]

        try:
            fact_store = self.remem.episodic_embedding_stores.get("facts")
            if fact_store:
                fact_data = fact_store.get_row(fact_key)
                fact_content = (
                    eval(fact_data["content"]) if isinstance(fact_data["content"], str) else fact_data["content"]
                )

                if isinstance(fact_content, dict):
                    # Use the static method to ensure consistent formatting
                    formatted_content = BaseTool.format_fact_dict_for_embedding(fact_content)

                    # Cache the result if caching is available
                    if cache is not None:
                        cache[fact_key] = formatted_content
                    return formatted_content
                else:
                    formatted_content = str(fact_content)
                    if cache is not None:
                        cache[fact_key] = formatted_content
                    return formatted_content

            # Cache the fallback result too if caching is available
            if cache is not None:
                cache[fact_key] = fact_key
            return fact_key
        except Exception as e:
            self.logger.warning(f"Error formatting fact content for retrieval {fact_key}: {e}")
            fallback = self._get_node_content(fact_key)
            if cache is not None:
                cache[fact_key] = fallback
            return fallback

    @staticmethod
    def format_fact_dict_for_embedding(fact_dict: Dict) -> str:
        """Format a fact dictionary for embedding."""
        try:
            if isinstance(fact_dict, dict):
                # Build base triple tuple
                subject = fact_dict.get("subject", "")
                predicate = fact_dict.get("predicate", "")
                object_val = fact_dict.get("object", "")

                triple_tuple = f"({subject}, {predicate}, {object_val})"

                # Add temporal qualifiers if they exist
                qualifiers = fact_dict.get("qualifiers", {})
                if qualifiers and qualifiers != "{}":
                    if isinstance(qualifiers, str):
                        qualifiers = eval(qualifiers)

                    # Filter out empty/null qualifiers and format as dict
                    valid_qualifiers = {}
                    for key, value in qualifiers.items():
                        if value and value != "null" and value != "":
                            valid_qualifiers[key] = value

                    if valid_qualifiers:
                        # Format as string representation of dict
                        qualifiers_str = str(valid_qualifiers).replace(
                            "'", '"'
                        )  # Use double quotes for JSON-like format
                        return f"{triple_tuple}, {qualifiers_str}"
                    else:
                        return triple_tuple
                else:
                    return triple_tuple
            else:
                return str(fact_dict)
        except Exception as e:
            logger.warning(f"Error formatting fact dict for embedding: {e}")
            return str(fact_dict)

    def _matches_temporal_filter(
        self,
        fact_dict: Dict,
        start_time: Optional[str],
        end_time: Optional[str],
        start_operator: str = "ge",
        end_operator: str = "le",
    ) -> bool:
        """
        Check if a fact matches the temporal filter using flexible date parsing and operators.

        Args:
            fact_dict: Fact dictionary to check
            start_time: Start time filter (optional)
            end_time: End time filter (optional)
            start_operator: Operator for start time comparison (lt, le, ge, gt), default "ge"
            end_operator: Operator for end time comparison (lt, le, ge, gt), default "le"
        """
        try:
            assert isinstance(fact_dict, dict)
            # If no qualifiers, keep the triple
            if "qualifiers" not in fact_dict:
                return True

            qualifiers = fact_dict["qualifiers"]
            if not qualifiers or qualifiers == "{}":
                return True

            assert isinstance(qualifiers, dict), "Qualifiers should be a dictionary"

            # Extract temporal information from qualifiers
            fact_start_str = qualifiers.get("start_time")
            fact_end_str = qualifiers.get("end_time")
            fact_point_str = qualifiers.get("point_in_time")
            fact_reference_str = qualifiers.get("reference_time")
            fact_record_str = qualifiers.get("record_time")

            if not any(
                [fact_start_str, fact_end_str, fact_point_str]
            ):  # If no temporal qualifiers (except for reference_time), set point_in_time to reference_time
                fact_point_str = fact_reference_str
            if not any([fact_start_str, fact_end_str, fact_point_str]):
                fact_point_str = fact_record_str
            if not any([fact_start_str, fact_end_str, fact_point_str]):  # If no temporal qualifiers, keep the triple
                return True

            # Parse filter dates using flexible parser
            filter_start = parse_flexible_datetime(start_time)
            filter_end = parse_flexible_datetime(end_time)

            # Parse fact dates using flexible parser
            fact_start = parse_flexible_datetime(fact_start_str)
            fact_end = parse_flexible_datetime(fact_end_str)
            fact_point = parse_flexible_datetime(fact_point_str)

            # If we have a point in time, treat it as both start and end
            if fact_point:
                if fact_start is None:
                    fact_start = fact_point
                if fact_end is None:
                    fact_end = fact_point

            # Validate operators
            valid_operators = {"lt", "le", "ge", "gt", "eq"}
            if start_operator not in valid_operators:
                self.logger.warning(f"Invalid start_operator: {start_operator}, using default 'ge'")
                start_operator = "ge"
            if end_operator not in valid_operators:
                self.logger.warning(f"Invalid end_operator: {end_operator}, using default 'le'")
                end_operator = "le"

            # Apply operator-based filtering
            return self._apply_temporal_operators(
                fact_start, fact_end, filter_start, filter_end, start_operator, end_operator
            )

        except Exception as e:
            self.logger.warning(f"Error checking temporal filter for {fact_dict}: {e}")
            return True  # Keep if error in filtering

    def _apply_temporal_operators(
        self,
        fact_start: Optional[datetime],
        fact_end: Optional[datetime],
        filter_start: Optional[datetime],
        filter_end: Optional[datetime],
        start_operator: str,
        end_operator: str,
    ) -> bool:
        """
        Apply temporal operators to check if fact time range matches filter criteria.

        This function simply applies each constraint independently:
        - start_operator: compares fact_start with filter_start
        - end_operator: compares fact_end with filter_end

        Examples:
            start_operator='ge', end_operator='le' => fact_start >= filter_start AND fact_end <= filter_end
           (facts contained within the filter range)

        Args:
            fact_start, fact_end: Fact's time range
            filter_start, filter_end: Filter's time range
            start_operator: Operator for start time comparison (lt, le, ge, gt)
            end_operator: Operator for end time comparison (lt, le, ge, gt)

        Returns:
            True if fact matches the filter criteria, False otherwise
        """
        try:
            # Handle cases where fact has no temporal information
            if not fact_start and not fact_end:
                return True  # Keep facts without temporal info

            # Handle point-in-time facts (treat as both start and end)
            if fact_start == fact_end and fact_start is not None:
                start_matches = self._compare_dates(fact_start, filter_start, start_operator)
                end_matches = self._compare_dates(fact_start, filter_end, end_operator)
                return start_matches and end_matches

            # Handle range facts - simply check each constraint separately
            start_constraint_satisfied = True
            if filter_start is not None:
                start_constraint_satisfied = self._compare_dates(fact_start, filter_start, start_operator)

            end_constraint_satisfied = True
            if filter_end is not None:
                end_constraint_satisfied = self._compare_dates(fact_end, filter_end, end_operator)

            return start_constraint_satisfied and end_constraint_satisfied
        except Exception as e:
            self.logger.warning(f"Error applying temporal operators: {e}")
            return True  # Keep if error in filtering

    def _compare_dates(
        self, candidate_date: Optional[datetime], filter_date: Optional[datetime], filter_operator: str
    ) -> bool:
        """
        Compare two dates using the specified operator.

        Args:
            candidate_date: First date (from fact)
            filter_date: Second date (from filter)
            filter_operator: Comparison operator (lt, le, ge, gt, eq)

        Returns:
            True if comparison holds, False otherwise

        Semantics for None values:
        - If filter date (date2) is None: constraint is ignored (return True)
        - If fact date (date1) is None:
          - None as start means "extends to infinite past"
          - None as end means "extends to infinite future"
          - Comparison depends on operator and semantic meaning
        """
        # If filter date is None, ignore this constraint
        if filter_date is None:
            return True

        # If fact date is None, handle based on infinite semantics
        if candidate_date is None:
            if filter_operator in ["ge", "gt"]:
                # For start times: None (infinite past) is always < any specific date
                # So None >= date or None > date should be False
                return False
            elif filter_operator in ["le", "lt"]:
                # For end times: None (infinite future) is always > any specific date
                # So None <= date or None < date should be False
                return False
            elif filter_operator == "eq":
                # None is never equal to a specific date
                return False
            else:
                self.logger.warning(f"Unknown operator: {filter_operator}")
                return True

        try:
            if filter_operator == "lt":
                return candidate_date < filter_date
            elif filter_operator == "le":
                return candidate_date <= filter_date
            elif filter_operator == "ge":
                return candidate_date >= filter_date
            elif filter_operator == "gt":
                return candidate_date > filter_date
            elif filter_operator == "eq":
                return candidate_date == filter_date
            else:
                self.logger.warning(f"Unknown operator: {filter_operator}")
                return True
        except Exception as e:
            self.logger.warning(f"Error comparing dates: {e}")
            return True
