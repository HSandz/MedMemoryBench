from typing import List, Optional, Tuple

from remem.agent.tools.base_tool import BaseTool, ToolContext, ToolResult, ToolType


class FindGistContextsTool(BaseTool):
    """Tool for finding gist contexts through synonym edges and connected triples with temporal filtering."""

    def __init__(self, remem_instance):
        super().__init__(ToolType.FIND_GIST_CONTEXTS, remem_instance)

    def get_description(self) -> str:
        """Get tool description for LLM-based selection."""
        import json

        return json.dumps(
            {
                "type": "function",
                "name": "find_gist_contexts",
                "description": "Explore related 'gists' (via synonym relationships) and connected 'facts' (triples) for a specific gist, with optional temporal filters.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "gist_id": {
                            "type": "number",
                            "description": "The index (not content) of a gist node from the last step to explore, starting from 1",
                        },
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
                    "required": ["gist_id"],
                },
            }
        )

    def can_execute(self, context: ToolContext) -> bool:
        """Can execute if we have visited gists."""
        return any(node.startswith("gists") for node in context.visited_nodes)

    def execute(self, context: ToolContext, max_results: int = 15, **kwargs) -> ToolResult:
        """Find gist contexts with optional temporal filtering."""
        try:
            gist_index = kwargs.get("gist_id")
            start_time = kwargs.get("start_time")
            end_time = kwargs.get("end_time")
            start_operator = kwargs.get("start_operator", "ge")  # Default to 'ge'
            end_operator = kwargs.get("end_operator", "le")  # Default to 'le'

            # Resolve gist_id to actual gist node
            gist_hash_id = None
            if gist_index:
                # Find gist_id-th gist by searching backwards through previous steps to find the most recent one with sufficient gists
                if context.previous_steps:
                    gist_chunks = []
                    try:
                        target_index = int(gist_index)
                        # Search backwards through previous steps to find the most recent step with enough gists
                        for step in reversed(context.previous_steps):
                            if "content_retrieval" in step:
                                passages = step["content_retrieval"]["passages"]
                                if passages:
                                    step_gist_chunks = [
                                        p for p in passages if p.get("node_name", "").startswith("gists-")
                                    ]
                                    # Check if this step has enough gists for the requested index
                                    if len(step_gist_chunks) >= target_index and target_index > 0:
                                        gist_chunks = step_gist_chunks
                                        break

                        if gist_chunks and 0 < target_index <= len(gist_chunks):
                            gist_hash_id = gist_chunks[target_index - 1]["node_name"]
                    except (ValueError, IndexError) as e:
                        self.logger.warning(f"Invalid gist indexing, {e}")

            if not gist_hash_id:
                self.logger.warning(f"No gist found for gist_id: {gist_index}")
                return ToolResult(
                    tool_type=self.tool_type,
                    nodes_found=[],
                    scores=[],
                    observation="No gist specified and no visited gists available",
                    should_continue=False,
                    node_contents=[],
                )

            # gist_content = self._get_node_content(gist_hash_id)
            # Find similar gists and connected triples (search with larger limit first)
            all_gists = self._find_similar_gists(gist_hash_id, context.query, max_results=1000)
            all_triples = self._find_connected_triples(
                gist_hash_id,
                context.query,
                start_time,
                end_time,
                max_results=1000,
                start_operator=start_operator,
                end_operator=end_operator,
            )

            # Limit results for display
            display_gists = all_gists[:max_results]
            display_triples = all_triples[:max_results]

            # Combine results
            all_nodes = [g[0] for g in display_gists] + [t[0] for t in display_triples]
            all_scores = [g[1] for g in display_gists] + [t[1] for t in display_triples]
            all_contents = [g[2] for g in display_gists] + [t[2] for t in display_triples]

            # Build observation message
            temporal_info = ""
            if start_time or end_time:
                temporal_info = f" (time filter: {start_time or 'any'} to {end_time or 'any'})"

            if all_nodes:
                parts = []
                if all_gists:
                    count_str = f">{1000}" if len(all_gists) >= 1000 else str(len(all_gists))
                    parts.append(f"{count_str} gists")
                if all_triples:
                    count_str = f">{1000}" if len(all_triples) >= 1000 else str(len(all_triples))
                    parts.append(f"{count_str} triples")

                show_parts = []
                if display_gists:
                    show_parts.append(f"{len(display_gists)} gists")
                if display_triples:
                    show_parts.append(f"{len(display_triples)} triples")

                observation = f"Found {' and '.join(parts)} for gist {gist_index}{temporal_info}, showing {' and '.join(show_parts)}"
            else:
                observation = f"No connected content found for gist {gist_index}{temporal_info}"

            return ToolResult(
                tool_type=self.tool_type,
                nodes_found=all_nodes,
                scores=all_scores,
                observation=observation,
                should_continue=len(all_nodes) > 0,
                node_contents=all_contents,
            )

        except Exception as e:
            self.logger.error(f"Error in find gist contexts: {e}")
            return ToolResult(
                tool_type=self.tool_type,
                nodes_found=[],
                scores=[],
                observation=f"Error in find gist contexts: {e}",
                should_continue=False,
                node_contents=[],
            )

    def _find_similar_gists(self, gist_hash_id: str, query: str, max_results: int) -> List[Tuple[str, float, str]]:
        """Find similar gists through synonym edges, sorted by embedding similarity."""
        try:
            # Get edges for the gist
            from remem.utils.igraph_utils import get_node_adjacent_edges

            edges = get_node_adjacent_edges(self.remem.graph, gist_hash_id, direction="out")
            similar_gists = []

            # First collect unique neighbor gists
            unique_gist_neighbors = set()
            for edge in edges:
                # Look for synonym edges
                if "synonym" in str(edge.get("predicate", "")).lower():
                    neighbor = edge["target_hash_id"] if edge["direction"] == "out" else edge["source_hash_id"]

                    if self._get_node_type(neighbor) == "gists":
                        unique_gist_neighbors.add(neighbor)

            # Score each unique neighbor
            for neighbor in unique_gist_neighbors:
                # Use embedding similarity scoring like entity tool
                from remem.utils.embed_utils import get_node_query_similarity

                similarity_score = get_node_query_similarity(neighbor, query, self.remem, logger=self.logger)
                content = self._get_node_content(neighbor)
                similar_gists.append((neighbor, similarity_score, content))

            # Sort by similarity and limit results
            similar_gists.sort(key=lambda x: x[1], reverse=True)
            return similar_gists[:max_results]

        except Exception as e:
            self.logger.warning(f"Error finding similar gists for {gist_hash_id}: {e}")
            return []

    def _find_connected_triples(
        self,
        gist_hash_id: str,
        query: str,
        start_time: Optional[str],
        end_time: Optional[str],
        max_results: int,
        start_operator: str,
        end_operator: str,
    ) -> List[Tuple[str, float, str]]:
        """Find triples connected to the gist with optional temporal filtering."""
        try:
            # Get all edges for the gist
            from remem.utils.igraph_utils import get_node_adjacent_edges

            edges = get_node_adjacent_edges(self.remem.graph, gist_hash_id, direction="all")
            connected_triples = []

            for edge in edges:
                neighbor = edge["target_hash_id"] if edge["direction"] == "out" else edge["source_hash_id"]

                if self._get_node_type(neighbor) == "facts":
                    # Check temporal filtering if specified
                    if start_time or end_time:
                        fact_dict = self._get_node_metadata(neighbor)
                        if not self._matches_temporal_filter(
                            fact_dict, start_time, end_time, start_operator, end_operator
                        ):
                            continue

                    connected_triples.append(neighbor)

            # Remove duplicates
            connected_triples = list(set(connected_triples))

            # Score by embedding similarity to query (like entity tool)
            scored_triples = []
            for triple in connected_triples:
                from remem.utils.embed_utils import get_node_query_similarity

                similarity_score = get_node_query_similarity(triple, query, self.remem, logger=self.logger)
                content = self._get_node_content(triple)
                scored_triples.append((triple, similarity_score, content))

            # Sort by similarity and limit results
            scored_triples.sort(key=lambda x: x[1], reverse=True)
            return scored_triples[:max_results]

        except Exception as e:
            self.logger.warning(f"Error finding connected triples for {gist_hash_id}: {e}")
            return []
