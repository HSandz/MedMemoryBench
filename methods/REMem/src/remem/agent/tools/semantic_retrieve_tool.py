from typing import List

from remem.agent.tools.base_tool import BaseTool, ToolContext, ToolResult, ToolType
from remem.utils.embed_utils import get_query_entry_scores


class SemanticRetrieveTool(BaseTool):
    """Tool for semantic retrieval that retrieves both gists and triples simultaneously."""

    def __init__(self, remem_instance):
        super().__init__(ToolType.SEMANTIC_RETRIEVE, remem_instance)

    def get_description(self) -> str:
        """Get tool description for LLM-based selection."""
        import json

        return json.dumps(
            {
                "type": "function",
                "name": "semantic_retrieve",
                "description": "Search for the most semantically relevant 'gists' and 'facts' (triples) in the KG using embedding similarity. It returns the top results from both types at once. Use when you need semantic understanding and conceptual matching.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query string"},
                        "start_time": {
                            "type": "string",
                            "description": "The start of a time range to filter facts by their temporal qualifiers",
                        },
                        "end_time": {
                            "type": "string",
                            "description": "The end of a time range to filter facts by their temporal qualifiers",
                        },
                        "start_operator": {
                            "type": "string",
                            "description": "Operator for start time comparison (lt, le, ge, gt, eq).",
                        },
                        "end_operator": {
                            "type": "string",
                            "description": "Operator for end time comparison (lt, le, ge, gt, eq).",
                        },
                    },
                    "required": ["query"],
                },
            }
        )

    def can_execute(self, context: ToolContext) -> bool:
        """Can execute if gists, facts, or verbatim are available."""
        return (
            "gists" in context.available_entries
            or "facts" in context.available_entries
            or "verbatim" in context.available_entries
        )

    def execute(self, context: ToolContext, beam_size: int = 5, do_filter: bool = False, **kwargs) -> ToolResult:
        """Execute semantic retrieval based on temporal context."""
        try:
            result_parts = []
            all_nodes = []
            all_scores = []
            all_contents = []
            start_time = kwargs.get("start_time")
            end_time = kwargs.get("end_time")
            start_operator = kwargs.get("start_operator", "ge")  # Default to 'ge'
            end_operator = kwargs.get("end_operator", "le")  # Default to 'le'
            do_filter = kwargs.get("do_filter", do_filter)
            temporal_context = kwargs.get("temporal_context", True)  # Default to True

            # Initialize variables to avoid unbound errors
            gist_candidates = []
            fact_candidates = []
            verbatim_candidates = []
            top_gists = []  # Will contain (key, score, content) tuples
            top_facts = []  # Will contain (key, score, content) tuples
            top_verbatim = []  # Will contain (key, score, content) tuples

            # Choose retrieval strategy based on temporal_context
            if not temporal_context:
                # Retrieve verbatim content when temporal_context is False
                if "verbatim" in context.available_entries:
                    verbatim_scores = get_query_entry_scores(context.query, "verbatim", self.remem)
                    verbatim_keys = context.available_entries["verbatim"]

                    verbatim_candidates = []
                    for i, (key, score) in enumerate(zip(verbatim_keys, verbatim_scores)):
                        # Filter out invalid scores
                        import math

                        if not self._is_visited(key, context) and not (math.isnan(score) or math.isinf(score)):
                            verbatim_candidates.append((key, score))

                    verbatim_candidates.sort(key=lambda x: x[1], reverse=True)
                    top_verbatim = verbatim_candidates[:beam_size]

                    # Add content to top_verbatim
                    top_verbatim_with_content = []
                    for verbatim_key, score in top_verbatim:
                        verbatim_content = self._get_node_content(verbatim_key)
                        top_verbatim_with_content.append((verbatim_key, score, verbatim_content))
                    top_verbatim = top_verbatim_with_content

                    for i, (verbatim_key, score, verbatim_content) in enumerate(top_verbatim, 1):
                        result_parts.append(f"{i}. {verbatim_content}")
                        all_nodes.append(verbatim_key)
                        all_scores.append(score)
                        all_contents.append(verbatim_content)
            else:
                # Default behavior: retrieve gists and facts when temporal_context is True
                # Retrieve gists
                if "gists" in context.available_entries:
                    gist_scores = get_query_entry_scores(context.query, "gists", self.remem)
                    gist_keys = context.available_entries["gists"]

                    gist_candidates = []
                    for i, (key, score) in enumerate(zip(gist_keys, gist_scores)):
                        # Filter out invalid scores
                        import math

                        if not self._is_visited(key, context) and not (math.isnan(score) or math.isinf(score)):
                            gist_candidates.append((key, score))

                    gist_candidates.sort(key=lambda x: x[1], reverse=True)
                    top_gists = gist_candidates[:beam_size]

                    # Add content to top_gists for filtering
                    top_gists_with_content = []
                    for gist_key, score in top_gists:
                        gist_content = self._get_node_content(gist_key)
                        top_gists_with_content.append((gist_key, score, gist_content))
                    top_gists = top_gists_with_content

                    for i, (gist_key, score, gist_content) in enumerate(top_gists, 1):
                        result_parts.append(f"{i}. {gist_content}")
                        all_nodes.append(gist_key)
                        all_scores.append(score)
                        all_contents.append(gist_content)

                # Retrieve triples
                if "facts" in context.available_entries:
                    fact_scores = get_query_entry_scores(context.query, "facts", self.remem)
                    fact_keys = context.available_entries["facts"]

                    fact_candidates = []
                    for i, (key, score) in enumerate(zip(fact_keys, fact_scores)):
                        fact_dict = self._get_node_metadata(key)
                        if not self._is_visited(key, context) and self._matches_temporal_filter(
                            fact_dict, start_time, end_time, start_operator, end_operator
                        ):
                            fact_candidates.append((key, score))

                    fact_candidates.sort(key=lambda x: x[1], reverse=True)
                    top_facts = fact_candidates[:beam_size]

                    # Add content to top_facts for filtering
                    top_facts_with_content = []
                    for fact_key, score in top_facts:
                        fact_content = self._get_node_content(fact_key)
                        top_facts_with_content.append((fact_key, score, fact_content))
                    top_facts = top_facts_with_content

                    if top_facts:
                        if result_parts:
                            result_parts.append("")  # Add empty line
                        result_parts.append("Triples:")
                        for i, (fact_key, score, fact_content) in enumerate(top_facts, 1):
                            qualifiers = self._get_fact_qualifiers(fact_key)
                            if qualifiers:
                                result_parts.append(f"{i}. {fact_content} {qualifiers}")
                            else:
                                result_parts.append(f"{i}. {fact_content}")
                            all_nodes.append(fact_key)
                            all_scores.append(score)
                            all_contents.append(fact_content)

            # Apply filtering if enabled and rebuild results (only for temporal context)
            if temporal_context and do_filter and self.remem.global_config.rerank_dspy_file_path is not None:
                top_gists, top_facts = self._apply_filtering(context.query, top_gists, top_facts, beam_size)

                # Rebuild result arrays with filtered results
                result_parts = []
                all_nodes = []
                all_scores = []
                all_contents = []

                # Add filtered gists
                for i, (gist_key, score, gist_content) in enumerate(top_gists, 1):
                    result_parts.append(f"{i}. {gist_content}")
                    all_nodes.append(gist_key)
                    all_scores.append(score)
                    all_contents.append(gist_content)

                # Add filtered facts
                if top_facts:
                    if result_parts:
                        result_parts.append("")  # Add empty line
                    result_parts.append("Triples:")
                    for i, (fact_key, score, fact_content) in enumerate(top_facts, 1):
                        qualifiers = self._get_fact_qualifiers(fact_key)
                        if qualifiers:
                            result_parts.append(f"{i}. {fact_content} {qualifiers}")
                        else:
                            result_parts.append(f"{i}. {fact_content}")
                        all_nodes.append(fact_key)
                        all_scores.append(score)
                        all_contents.append(fact_content)

            # Use candidate list lengths for total, and top list lengths for display
            if temporal_context:
                total_gists = len(gist_candidates)
                total_triples = len(fact_candidates)
                num_shown_gists = len(top_gists)
                num_shown_facts = len(top_facts)
                total_gists_str = ">1000" if total_gists == 1000 else str(total_gists)
                total_triples_str = ">1000" if total_triples == 1000 else str(total_triples)

                if num_shown_gists > 0 or num_shown_facts > 0:
                    found_parts = []
                    if total_gists > 0:
                        found_parts.append(f"{total_gists_str} gists")
                    if total_triples > 0:
                        found_parts.append(f"{total_triples_str} triples")
                    show_parts = []
                    if num_shown_gists > 0:
                        show_parts.append(f"{num_shown_gists} gists")
                    if num_shown_facts > 0:
                        show_parts.append(f"{num_shown_facts} triples")
                    observation = f"Retrieved {' and '.join(found_parts)} via semantic similarity, showing {' and '.join(show_parts)}"
                else:
                    observation = "No relevant content found via semantic similarity"
            else:  # verbatim retrieved
                total_verbatim = len(verbatim_candidates)
                num_shown_verbatim = len(top_verbatim)
                total_verbatim_str = ">1000" if total_verbatim == 1000 else str(total_verbatim)

                if num_shown_verbatim > 0:
                    observation = f"Retrieved {total_verbatim_str} verbatim texts via semantic similarity, showing {num_shown_verbatim} verbatim texts"
                else:
                    observation = "No relevant verbatim content found via semantic similarity"

            assert len(all_nodes) == len(all_scores), "Mismatched length between nodes and scores"
            return ToolResult(
                tool_type=self.tool_type,
                nodes_found=all_nodes,
                scores=all_scores,
                observation=observation,
                should_continue=len(all_nodes) > 0,
                node_contents=all_contents,
            )

        except Exception as e:
            self.logger.error("Error in semantic_retrieve " + str(e))
            return ToolResult(
                tool_type=self.tool_type,
                nodes_found=[],
                scores=[],
                observation=f"Error in semantic retrieve: {e}",
                should_continue=False,
                node_contents=[],
            )

    def _apply_filtering(self, query: str, top_gists: List[tuple], top_facts: List[tuple], max_results: int) -> tuple:
        """Apply appropriate filtering to gists and triples.

        Args:
            query: The search query
            top_gists: List of (key, score, content) tuples for gists
            top_facts: List of (key, score, content) tuples for facts
            max_results: Maximum number of results to return

        Returns:
            Tuple of (filtered_gists, filtered_facts) where each is a list of (key, score, content) tuples
        """
        try:
            # Import filters here to avoid circular imports
            from remem.rerank import DSPyFilter, GistFilter

            filtered_gists = []
            filtered_facts = []

            # Filter gists using GistFilter
            if top_gists:
                gist_filter = GistFilter(self.remem)
                gist_candidates = []
                gist_indices = []
                for i, (gist_key, score, gist_content) in enumerate(top_gists):
                    gist_candidates.append(gist_content)
                    gist_indices.append(i)

                if gist_candidates:
                    try:
                        filtered_indices, filtered_items, _ = gist_filter.rerank(
                            query=query,
                            candidate_items=gist_candidates,
                            candidate_indices=gist_indices,
                            len_after_rerank=max_results,
                        )

                        # Map back to original gist format with content
                        for idx in filtered_indices:
                            if idx < len(top_gists):
                                filtered_gists.append(top_gists[idx])

                        # Log filtering results for debugging
                        self.logger.info(f"Gist filtering: {len(top_gists)} -> {len(filtered_gists)} results")
                        for i, (key, score, content) in enumerate(filtered_gists):
                            self.logger.info(f"  Filtered gist {i+1}: {content[:100]}...")

                    except Exception as e:
                        self.logger.warning(f"Error filtering gists: {e}")
                        filtered_gists = top_gists[:max_results]

            # Filter facts using DSPyFilter
            if top_facts:
                # Check if filter is available
                if not hasattr(self.remem, "triple_filter") or self.remem.triple_filter is None:
                    fact_filter = DSPyFilter(self.remem)
                else:
                    fact_filter = self.remem.triple_filter

                fact_candidates = []
                fact_indices = []
                for i, (fact_key, score, fact_content) in enumerate(top_facts):
                    # Extract triple from fact content
                    try:
                        fact_triple = eval(fact_content)
                        if len(fact_triple) == 3:
                            fact_candidates.append(fact_triple)
                            fact_indices.append(i)
                    except:
                        # If evaluation fails, create a pseudo-triple
                        fact_triple = [
                            fact_content[:50],
                            "relates to",
                            fact_content[50:100] if len(fact_content) > 50 else "information",
                        ]
                        fact_candidates.append(fact_triple)
                        fact_indices.append(i)

                if fact_candidates:
                    try:
                        filtered_indices, filtered_items, _ = fact_filter.rerank(
                            query=query,
                            candidate_items=fact_candidates,
                            candidate_indices=fact_indices,
                            len_after_rerank=max_results,
                        )

                        # Map back to original fact format with content
                        for idx in filtered_indices:
                            if idx < len(top_facts):
                                filtered_facts.append(top_facts[idx])

                        # Log filtering results for debugging
                        self.logger.info(f"Fact filtering: {len(top_facts)} -> {len(filtered_facts)} results")
                        for i, (key, score, content) in enumerate(filtered_facts):
                            self.logger.info(f"  Filtered fact {i+1}: {content}")

                    except Exception as e:
                        self.logger.warning(f"Error filtering facts: {e}")
                        filtered_facts = top_facts[:max_results]

            return filtered_gists, filtered_facts

        except Exception as e:
            self.logger.warning(f"Error in filtering: {e}")
            return top_gists[:max_results], top_facts[:max_results]

    def _get_fact_qualifiers(self, fact_key: str) -> str:
        """Get qualifiers for a fact as formatted string."""
        try:
            fact_store = self.remem.episodic_embedding_stores.get("facts")
            if fact_store:
                fact_data = fact_store.get_row(fact_key)
                fact_dict = fact_data.get("metadata", {})
                if "qualifiers" in fact_dict:
                    qualifiers = fact_dict["qualifiers"]
                    if qualifiers and qualifiers != "{}":
                        # Format qualifiers nicely
                        if isinstance(qualifiers, str):
                            qualifiers = eval(qualifiers)

                        formatted_quals = []
                        for key, value in qualifiers.items():
                            if key in ["start_time", "end_time", "point_in_time"]:
                                formatted_quals.append(f'"{key}": "{value}"')

                        if formatted_quals:
                            return "{" + ", ".join(formatted_quals) + "}"
            return ""
        except Exception as e:
            self.logger.warning(f"Error getting qualifiers for {fact_key}: {e}")
            return ""
