from typing import List, Optional, Tuple

from remem.agent.tools.base_tool import BaseTool, ToolContext, ToolResult, ToolType
from remem.utils.embed_utils import get_query_entry_scores


class FindEntityContextsTool(BaseTool):
    """Tool for finding entity contexts through connected gists and triples."""

    def __init__(self, remem_instance):
        super().__init__(ToolType.FIND_ENTITY_CONTEXTS, remem_instance)

    def get_description(self) -> str:
        """Get tool description for LLM-based selection."""
        import json

        return json.dumps(
            {
                "type": "function",
                "name": "find_entity_contexts",
                "description": "This tool finds facts (triples) that match the given criteria. You must provide at least one of 'subject', 'object', or 'predicate'.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "subject": {
                            "type": "string",
                            "description": "The subject of a fact, e.g., 'E1' in (E1, was born in, E2)",
                        },
                        "object": {
                            "type": "string",
                            "description": "The object of a fact, e.g., 'E2' in (E1, was born in, E2)",
                        },
                        "predicate": {
                            "type": "string",
                            "description": "Filter for specific relations by exact name (e.g., 'was born in')",
                        },
                        "start_time": {
                            "type": "string",
                            "description": "Filter facts based on their start time, e.g., '1952-01-01'. Use with start_operator to control comparison",
                        },
                        "end_time": {
                            "type": "string",
                            "description": "Filter facts based on their end time, e.g., '1957-12-31'. Use with end_operator to control comparison",
                        },
                        "start_operator": {
                            "type": "string",
                            "description": "Operator for start_time comparison. Options: 'ge' (>=), 'gt' (>), 'le' (<=), 'lt' (<), 'eq' (=). Default: 'ge'",
                        },
                        "end_operator": {
                            "type": "string",
                            "description": "Operator for end_time comparison. Options: 'ge' (>=), 'gt' (>), 'le' (<=), 'lt' (<), 'eq' (=). Default: 'le'",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Restrict the number of returned results. Use with 'ordering' to get the first/last items. Skip setting this to get more contexts",
                        },
                        "ordering": {
                            "type": "string",
                            "description": "Order the results by time. 'asc' for ascending by start_time (earliest first), 'desc' for descending by end_time (latest first)",
                        },
                        "offset": {
                            "type": "integer",
                            "description": "Skip the first N results. Use with 'ordering' and 'limit=1' to get the Nth item (e.g., 'the 2nd time')",
                        },
                        "aggregation": {
                            "type": "string",
                            "description": "Perform an aggregation and return a single number. Options: 'count', 'count_unique_subjects', 'count_unique_objects'.",
                        },
                    },
                    "required": [],
                },
            }
        )

    def can_execute(self, context: ToolContext) -> bool:
        """Can execute if we have visited entities or can extract entities."""
        has_entities = any(node.startswith("entity") for node in context.visited_nodes)
        has_facts = any(node.startswith("facts") for node in context.visited_nodes)
        return has_entities or has_facts

    def execute(self, context: ToolContext, max_results: int = 15, **kwargs) -> ToolResult:
        """Find facts (triples) that match the given criteria."""
        try:
            # Extract new interface parameters
            subject = kwargs.get("subject")
            object_entity = kwargs.get("object")
            predicate = kwargs.get("predicate")
            start_time = kwargs.get("start_time")
            end_time = kwargs.get("end_time")
            start_operator = kwargs.get("start_operator", "ge")
            end_operator = kwargs.get("end_operator", "le")
            ordering = kwargs.get("ordering")
            limit = kwargs.get("limit")
            offset = kwargs.get("offset", 0)
            aggregation = kwargs.get("aggregation")

            # Validate that at least one search criterion is provided
            # Allow time-only queries if temporal filters are specified
            has_entity_filter = any([subject, object_entity, predicate])
            has_temporal_filter = any([start_time, end_time])

            if not has_entity_filter and not has_temporal_filter:
                return ToolResult(
                    tool_type=self.tool_type,
                    nodes_found=[],
                    scores=[],
                    observation="You must provide at least one of 'subject', 'object', 'predicate', or temporal filters ('start_time', 'end_time')",
                    should_continue=False,
                    node_contents=[],
                )

            # Validate operator parameters
            valid_operators = ["ge", "gt", "le", "lt", "eq"]
            if start_operator not in valid_operators:
                self.logger.warning(f"Invalid start_operator: {start_operator}. Using default 'ge'")
                start_operator = "ge"
            if end_operator not in valid_operators:
                self.logger.warning(f"Invalid end_operator: {end_operator}. Using default 'le'")
                end_operator = "le"

            # Validate ordering parameter
            if ordering and ordering not in ["asc", "desc"]:
                self.logger.warning(
                    f"Invalid ordering parameter: {ordering}. Must be one of: 'asc', 'desc' (ignoring ordering)"
                )
                ordering = None

            # Find matching facts
            matching_facts = self._find_matching_facts(
                subject=subject,
                object_entity=object_entity,
                predicate=predicate,
                start_time=start_time,
                end_time=end_time,
                start_operator=start_operator,
                end_operator=end_operator,
                ordering=ordering,
                query=context.query,
            )

            # Handle aggregation
            if aggregation:
                return self._handle_aggregation(aggregation, matching_facts, subject, object_entity, predicate)

            # Apply limit and offset
            total_found = len(matching_facts)

            # Fallback to semantic retrieve if no exact matches found
            # if total_found == 0 and not aggregation:
            #     return self._fallback_to_semantic_retrieve(
            #         context, subject, object_entity, predicate,
            #         start_time, end_time, start_operator, end_operator, max_results
            #     )

            if offset > 0:
                matching_facts = matching_facts[offset:]
            if limit:
                # set num_display according to `max_results` and tool argument `limit`
                num_display = min(max_results, limit)
            else:
                num_display = max_results
            matching_facts = matching_facts[:num_display]

            # Prepare results
            nodes_found = [fact[0] for fact in matching_facts]
            scores = [fact[1] for fact in matching_facts]
            all_contents = [self._get_node_content(fact[0]) for fact in matching_facts]

            # Create observation message
            criteria_parts = []
            if subject:
                criteria_parts.append(f"subject='{subject}'")
            if object_entity:
                criteria_parts.append(f"object='{object_entity}'")
            if predicate:
                criteria_parts.append(f"predicate='{predicate}'")

            temporal_info = ""
            if start_time or end_time:
                time_parts = []
                if start_time:
                    time_parts.append(f"start_time='{start_time}' ({start_operator})")
                if end_time:
                    time_parts.append(f"end_time='{end_time}' ({end_operator})")
                temporal_info = f" with temporal filter ({', '.join(time_parts)})"

            limit_info = ""
            if limit or offset:
                limit_parts = []
                if offset:
                    limit_parts.append(f"offset={offset}")
                if limit:
                    limit_parts.append(f"limit={limit}")
                limit_info = f" with {', '.join(limit_parts)}"

            observation = f"Found {total_found} facts matching {', '.join(criteria_parts)}{temporal_info}{limit_info}, showing {len(matching_facts)} facts"

            return ToolResult(
                tool_type=self.tool_type,
                nodes_found=nodes_found,
                scores=scores,
                observation=observation,
                should_continue=len(nodes_found) > 0,
                node_contents=all_contents,
            )

        except Exception as e:
            self.logger.error(f"Error in find entity contexts: {e}")
            return ToolResult(
                tool_type=self.tool_type,
                nodes_found=[],
                scores=[],
                observation=f"Error in find entity contexts: {e}",
                should_continue=False,
                node_contents=[],
            )

    def _find_matching_facts(
        self,
        subject: Optional[str],
        object_entity: Optional[str],
        predicate: Optional[str],
        start_time: Optional[str],
        end_time: Optional[str],
        start_operator: str,
        end_operator: str,
        ordering: Optional[str],
        query: str,
    ) -> List[Tuple[str, float, dict]]:
        """Find facts matching the specified criteria."""
        try:
            # Get all facts from the episodic store
            fact_store = self.remem.episodic_embedding_stores.get("facts")
            if not fact_store:
                return []

            # First pass: collect all matching facts (without similarity scores)
            candidate_facts = []

            # Iterate through all facts in the store
            for fact_key in fact_store.hash_id_to_row.keys():
                try:
                    fact_data = fact_store.get_row(fact_key)
                    fact_content = fact_data.get("metadata", "{}")

                    # Parse fact content
                    if isinstance(fact_content, str):
                        try:
                            fact_dict = eval(fact_content)
                        except:
                            continue
                    else:
                        fact_dict = fact_content

                    if not isinstance(fact_dict, dict):
                        continue

                    # Check subject match
                    if subject and subject not in fact_dict.get("subject", ""):
                        continue

                    # Check object match
                    if object_entity and object_entity not in fact_dict.get("object", ""):
                        continue

                    # Check predicate match
                    if predicate and predicate not in fact_dict.get("predicate", ""):
                        continue

                    # Apply temporal filtering if needed
                    if start_time or end_time:
                        if not self._matches_temporal_filter(
                            fact_dict, start_time, end_time, start_operator, end_operator
                        ):
                            continue

                    # Add to candidates (without similarity score yet)
                    candidate_facts.append((fact_key, fact_dict))

                except Exception as e:
                    self.logger.warning(f"Error processing fact {fact_key}: {e}")
                    continue

            # If no candidates found, return empty list
            if not candidate_facts:
                return []

            # Second pass: batch calculate similarity scores for all candidates
            fact_keys = [fact[0] for fact in candidate_facts]
            fact_dicts = [fact[1] for fact in candidate_facts]

            try:
                # Use batch similarity calculation
                similarity_scores = self._batch_calculate_fact_similarities(fact_keys, query)
            except Exception as e:
                self.logger.warning(
                    f"Batch similarity calculation failed: {e}, falling back to individual calculations"
                )
                # Fallback to individual calculations if batch fails
                similarity_scores = []
                for fact_key in fact_keys:
                    try:
                        from remem.utils.embed_utils import get_node_query_similarity

                        score = get_node_query_similarity(fact_key, query, self.remem, logger=self.logger)
                        similarity_scores.append(score)
                    except Exception as e:
                        self.logger.warning(f"Error calculating similarity for {fact_key}: {e}")
                        similarity_scores.append(0.0)

            # Combine results
            matching_facts = []
            for i, (fact_key, fact_dict) in enumerate(candidate_facts):
                similarity_score = similarity_scores[i] if i < len(similarity_scores) else 0.0
                matching_facts.append((fact_key, similarity_score, fact_dict))

            # Apply ordering
            if ordering == "asc":
                # Sort by start_time from earliest to latest
                matching_facts.sort(key=lambda x: self._get_temporal_sort_key(x[2], "start_time"))
            elif ordering == "desc":
                # Sort by end_time from latest to earliest
                matching_facts.sort(key=lambda x: self._get_temporal_sort_key(x[2], "end_time"), reverse=True)
            else:
                # Default: sort by similarity score
                matching_facts.sort(key=lambda x: x[1], reverse=True)

            return matching_facts

        except Exception as e:
            self.logger.error(f"Error finding matching facts: {e}")
            return []

    def _batch_calculate_fact_similarities(self, fact_keys: List[str], query: str) -> List[float]:
        """
        Batch calculate similarity scores between fact keys and query.

        Args:
            fact_keys: List of fact keys to calculate similarities for
            query: Query string

        Returns:
            List of similarity scores corresponding to the input fact keys
        """
        try:
            if not fact_keys:
                return []

            # Get the fact store
            fact_store = self.remem.episodic_embedding_stores.get("facts")
            if not fact_store:
                return [0.0] * len(fact_keys)

            # Get query embedding
            key = "triple"
            linking_method = "query_to_triple"

            # Try to get cached query embedding first
            query_embedding = None
            if hasattr(self.remem, "query_to_embedding") and self.remem.query_to_embedding:
                query_embedding = self.remem.query_to_embedding.get(key, {}).get(query)

            if query_embedding is None:
                from remem.prompts.linking import get_query_instruction

                query_embedding = self.remem.embedding_model.batch_encode(
                    [query], instruction=get_query_instruction(linking_method), norm=True
                )[0]

            # Collect all fact embeddings
            fact_embeddings = []
            valid_indices = []

            for i, fact_key in enumerate(fact_keys):
                try:
                    if fact_key in fact_store.hash_id_to_idx:
                        fact_embedding = fact_store.get_embedding(fact_key)
                        if fact_embedding is not None:
                            fact_embeddings.append(fact_embedding)
                            valid_indices.append(i)
                except Exception as e:
                    self.logger.warning(f"Error getting embedding for fact {fact_key}: {e}")
                    continue

            if not fact_embeddings:
                return [0.0] * len(fact_keys)

            # Batch compute similarities using matrix operations
            import numpy as np


            # Convert to numpy arrays for batch computation
            fact_embeddings = np.array(fact_embeddings)  # Shape: (num_facts, embedding_dim)
            query_embedding = np.array(query_embedding).reshape(1, -1)  # Shape: (1, embedding_dim)

            # Compute dot product similarity for all facts at once
            similarities = np.dot(fact_embeddings, query_embedding.T).flatten()

            # Create result array with zeros for invalid keys
            result = [0.0] * len(fact_keys)
            for i, valid_idx in enumerate(valid_indices):
                result[valid_idx] = float(similarities[i])

            return result

        except Exception as e:
            self.logger.error(f"Error in batch similarity calculation: {e}")
            # Fallback to zeros if batch calculation fails
            return [0.0] * len(fact_keys)

    def _handle_aggregation(
        self,
        aggregation: str,
        matching_facts: List[Tuple[str, float, dict]],
        subject: Optional[str],
        object_entity: Optional[str],
        predicate: Optional[str],
    ) -> ToolResult:
        """Handle aggregation operations."""
        try:
            if aggregation == "count":
                count = len(matching_facts)
                observation = f"Count of matching facts: {count}"
                all_contents = [str(count)]
                return ToolResult(
                    tool_type=self.tool_type,
                    nodes_found=[],
                    scores=[],
                    observation=observation,
                    should_continue=False,
                    node_contents=all_contents,
                )

            elif aggregation == "count_unique_subjects":
                unique_subjects = set()
                for _, _, fact_dict in matching_facts:
                    subj = fact_dict.get("subject")
                    if subj:
                        unique_subjects.add(subj)
                count = len(unique_subjects)
                observation = f"Count of unique subjects: {count}"
                all_contents = [str(count)]
                return ToolResult(
                    tool_type=self.tool_type,
                    nodes_found=[],
                    scores=[],
                    observation=observation,
                    should_continue=False,
                    node_contents=all_contents,
                )

            elif aggregation == "count_unique_objects":
                unique_objects = set()
                for _, _, fact_dict in matching_facts:
                    obj = fact_dict.get("object")
                    if obj:
                        unique_objects.add(obj)
                count = len(unique_objects)
                observation = f"Count of unique objects: {count}"
                all_contents = [str(count)]
                return ToolResult(
                    tool_type=self.tool_type,
                    nodes_found=[],
                    scores=[],
                    observation=observation,
                    should_continue=False,
                    node_contents=all_contents,
                )

            else:
                return ToolResult(
                    tool_type=self.tool_type,
                    nodes_found=[],
                    scores=[],
                    observation=f"Unknown aggregation type: {aggregation}",
                    should_continue=False,
                    node_contents=[],
                )

        except Exception as e:
            self.logger.error(f"Error in aggregation: {e}")
            return ToolResult(
                tool_type=self.tool_type,
                nodes_found=[],
                scores=[],
                observation=f"Error in aggregation: {e}",
                should_continue=False,
                node_contents=[],
            )

    def _get_temporal_sort_key(self, fact_dict: dict, time_field: str):
        """Get temporal sort key for a fact dictionary."""
        try:
            # Extract temporal information from qualifiers
            qualifiers = fact_dict.get("qualifiers", {})
            if isinstance(qualifiers, str):
                try:
                    qualifiers = eval(qualifiers)
                except:
                    qualifiers = {}

            time_str = qualifiers.get(time_field)

            # If no specific time field, check for point_in_time as fallback
            if not time_str:
                point_in_time = qualifiers.get("point_in_time")
                if point_in_time:
                    time_str = point_in_time
                else:
                    # If no temporal info, use a default value that sorts to the end
                    return float("inf")

            # Parse the time string using the same parser as in base_tool
            from remem.utils.datetime_utils import parse_flexible_datetime

            parsed_time = parse_flexible_datetime(time_str)

            if parsed_time:
                # Convert to timestamp for sorting
                return parsed_time.timestamp()
            else:
                # If parsing fails, use default value
                return float("inf")

        except Exception as e:
            self.logger.warning(f"Error getting temporal sort key for {fact_dict}: {e}")
            # Return default value on error
            return float("inf")

    def _fallback_to_semantic_retrieve(
        self,
        context: ToolContext,
        subject: Optional[str],
        object_entity: Optional[str],
        predicate: Optional[str],
        start_time: Optional[str],
        end_time: Optional[str],
        start_operator: str,
        end_operator: str,
        max_results: int,
    ) -> ToolResult:
        """Fallback to semantic retrieve when no exact matches are found."""
        try:
            # Construct search query from subject, predicate, and object
            query_parts = []
            if subject:
                query_parts.append(subject)
            if predicate:
                query_parts.append(predicate)
            if object_entity:
                query_parts.append(object_entity)

            if not query_parts:
                # If no entity/predicate criteria, use the original query
                semantic_query = context.query
            else:
                semantic_query = " ".join(query_parts)

            # Use semantic retrieve logic
            result_parts = []
            all_nodes = []
            all_scores = []
            all_contents = []
            fact_candidates = []  # Initialize here to avoid UnboundLocalError

            # Retrieve triples using semantic similarity
            if "facts" in context.available_entries:
                fact_scores = get_query_entry_scores(semantic_query, "facts", self.remem)
                fact_keys = context.available_entries["facts"]

                for i, (key, score) in enumerate(zip(fact_keys, fact_scores)):
                    fact_dict = self._get_node_metadata(key)
                    if not self._is_visited(key, context) and self._matches_temporal_filter(
                        fact_dict, start_time, end_time, start_operator, end_operator
                    ):
                        # Import numpy to check for invalid scores
                        import numpy as np

                        if not (np.isnan(score) or np.isinf(score)):
                            fact_candidates.append((key, score))

                fact_candidates.sort(key=lambda x: x[1], reverse=True)
                top_facts = fact_candidates[:max_results]

                for i, (fact_key, score) in enumerate(top_facts, 1):
                    fact_content = self._get_node_content(fact_key)
                    qualifiers = self._get_fact_qualifiers(fact_key)
                    if qualifiers:
                        result_parts.append(f"{i}. {fact_content} {qualifiers}")
                    else:
                        result_parts.append(f"{i}. {fact_content}")
                    all_nodes.append(fact_key)
                    all_scores.append(score)
                    all_contents.append(fact_content)

            # Create observation message indicating fallback
            criteria_parts = []
            if subject:
                criteria_parts.append(f"subject='{subject}'")
            if object_entity:
                criteria_parts.append(f"object='{object_entity}'")
            if predicate:
                criteria_parts.append(f"predicate='{predicate}'")

            temporal_info = ""
            if start_time or end_time:
                time_parts = []
                if start_time:
                    time_parts.append(f"start_time='{start_time}' ({start_operator})")
                if end_time:
                    time_parts.append(f"end_time='{end_time}' ({end_operator})")
                temporal_info = f" with temporal filter ({', '.join(time_parts)})"

            total_found = len(fact_candidates)
            if total_found > 0:
                observation = f"No exact matches found for {', '.join(criteria_parts)}{temporal_info}. Found {total_found} semantically similar facts, showing {len(all_nodes)} facts"
            else:
                observation = f"No exact matches or semantically similar facts found for {', '.join(criteria_parts)}{temporal_info}"

            return ToolResult(
                tool_type=self.tool_type,
                nodes_found=all_nodes,
                scores=all_scores,
                observation=observation,
                should_continue=len(all_nodes) > 0,
                node_contents=all_contents,
            )

        except Exception as e:
            self.logger.error(f"Error in semantic retrieve fallback: {e}")
            return ToolResult(
                tool_type=self.tool_type,
                nodes_found=[],
                scores=[],
                observation=f"Error in semantic retrieve fallback: {e}",
                should_continue=False,
                node_contents=[],
            )
