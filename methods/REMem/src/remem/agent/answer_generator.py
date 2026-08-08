from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from remem.remem import ReMem


def aggregate_gist_facts_to_verbatim(
    previous_steps: List[Dict[str, Any]], remem_instance: ReMem, facts_weight: float, logger=None
) -> List[Dict[str, Any]]:
    """
    Aggregate gist and facts scores to verbatim nodes and return verbatim passages with scores.

    Args:
        previous_steps: Agent interaction history steps containing gist/facts retrievals
        remem_instance: ReMem instance with graph and embedding stores
        facts_weight: Weight for facts scores
        logger: Logger instance

    Returns:
        List of verbatim passages with aggregated scores
    """
    if not previous_steps or not remem_instance:
        return []

    # Dictionary to aggregate scores for each verbatim node
    verbatim_scores = defaultdict(list)  # verbatim_key -> [(score, weight), ...]

    try:
        for step_data in previous_steps:
            if "content_retrieval" not in step_data:
                continue

            content_info = step_data["content_retrieval"]
            chunks = content_info.get("passages", [])

            assert remem_instance.graph.ecount() > 0, "ReMem graph is not loaded correctly"
            for chunk in chunks:
                node_hash_id = chunk.get("node_name", "")
                content = chunk.get("content", "")
                score = chunk.get("embedding_score", 0.0)

                # Skip if no score available
                if score is None or np.isnan(score) or np.isinf(score):
                    continue

                # Determine node type and find connected verbatim
                if "gists" in node_hash_id:
                    connected_verbatim = _find_connected_verbatim_nodes(
                        node_hash_id, remem_instance, logger, via_gist=False
                    )
                    for verbatim_key in connected_verbatim:
                        verbatim_scores[verbatim_key].append((score, 1))

                elif "facts" in node_hash_id:
                    connected_verbatim = _find_connected_verbatim_nodes(
                        node_hash_id, remem_instance, logger, via_gist=True
                    )
                    for verbatim_key in connected_verbatim:
                        verbatim_scores[verbatim_key].append((score, facts_weight))

        # Get verbatim contents and compute final scores
        aggregated_verbatim = []
        verbatim_store = remem_instance.episodic_embedding_stores.get("verbatim")

        if verbatim_store:
            verbatim_chunks = verbatim_store.get_hash_id_to_row_readonly()

            for verbatim_key, score_weight_pairs in verbatim_scores.items():
                if verbatim_key in verbatim_chunks:
                    content = verbatim_chunks[verbatim_key].get("content", "")

                    # Aggregate scores with weights
                    weighted_scores = [score * weight for score, weight in score_weight_pairs]
                    final_score = np.mean(weighted_scores) if weighted_scores else 0.0

                    aggregated_verbatim.append(
                        {
                            "node_name": verbatim_key,
                            "content": content,
                            "score": final_score,
                            "source_count": len(score_weight_pairs),  # How many gist/facts contributed
                        }
                    )

        # Sort by final score
        aggregated_verbatim.sort(key=lambda x: x["score"], reverse=True)
        return aggregated_verbatim

    except Exception as e:
        if logger:
            logger.error(f"Error aggregating gist/facts to verbatim: {e}")
        return []


def _find_connected_verbatim_nodes(node_key: str, remem_instance, logger=None, via_gist: bool = False) -> List[str]:
    """Find verbatim nodes connected to a gist or fact node, including 2-hop search through gist nodes."""
    try:
        from remem.utils.igraph_utils import get_node_adjacent_edges

        if not hasattr(remem_instance, "graph") or remem_instance.graph is None:
            return []

        verbatim_nodes = []
        gist_nodes = []

        # Get both verbatim and gist nodes connected to this node in one call
        adjacent_edges = get_node_adjacent_edges(
            remem_instance.graph,
            node_key,
            direction="all",
            node_types=["verbatim", "gists"] if via_gist else ["verbatim"],
        )

        for edge in adjacent_edges:
            # Check both source and target for verbatim and gist nodes
            source_hash = edge.get("source_hash_id", "")
            target_hash = edge.get("target_hash_id", "")

            # Collect 1-hop verbatim nodes
            if source_hash.startswith("verbatim-"):
                verbatim_nodes.append(source_hash)
            elif target_hash.startswith("verbatim-"):
                verbatim_nodes.append(target_hash)

            if via_gist:
                # Collect gist nodes for 2-hop search
                if source_hash.startswith("gists-"):
                    gist_nodes.append(source_hash)
                elif target_hash.startswith("gists-"):
                    gist_nodes.append(target_hash)

        # For each gist node found, search for connected verbatim nodes
        for gist_node in gist_nodes:
            gist_verbatim_edges = get_node_adjacent_edges(
                remem_instance.graph, gist_node, direction="all", node_types=["verbatim"]
            )

            for edge in gist_verbatim_edges:
                # Check both source and target for verbatim nodes
                source_hash = edge.get("source_hash_id", "")
                target_hash = edge.get("target_hash_id", "")

                if source_hash.startswith("verbatim-"):
                    verbatim_nodes.append(source_hash)
                elif target_hash.startswith("verbatim-"):
                    verbatim_nodes.append(target_hash)

        return list(set(verbatim_nodes))  # Remove duplicates

    except Exception as e:
        if logger:
            logger.warning(f"Error finding connected verbatim nodes for {node_key}: {e}")
        return []


def get_remem_qa_developer_prompt(question_metadata) -> str:
    """
    Get the developer prompt for final answer generation.

    Returns:
        str: The developer prompt string.
    """
    if question_metadata and question_metadata.get("date") is not None:
        from remem.prompts.templates.rag_qa_longmemeval import conversation_qa_system

        return conversation_qa_system
    from remem.prompts.templates.rag_qa_unified import conversation_qa_system

    return conversation_qa_system


class FinalAnswerGenerator:
    """Shared logic for generating final answers from retrieved context."""

    def __init__(
        self,
        llm_client,
        logger=None,
        llm_interaction_callback: Optional[Callable] = None,
        use_verbatim_context: bool = False,
        facts_weight: float = 0.1,
    ):
        self.llm_client = llm_client
        self.logger = logger
        self.llm_interaction_callback = llm_interaction_callback
        self.only_use_verbatim_contexts = use_verbatim_context
        self.facts_weight = facts_weight

    def generate_answer(
        self,
        query: str,
        retrieved_passages: Optional[List[Dict[str, Any]]] = None,
        reasoning_chain: Optional[List[str]] = None,
        visited_nodes: Optional[set] = None,
        reasoning_step: int = 0,
        previous_steps: Optional[List[Dict[str, Any]]] = None,
        preliminary_answer: Optional[str] = None,
        question_metadata: Optional[Dict[str, Any]] = None,
        interaction_type: str = "final_answer_generation",
        remem_instance: Optional[ReMem] = None,
        skip_empty_step: bool = True,
    ) -> str:
        """
        Generate an answer based on retrieved passages and/or reasoning chain.

        Args:
            query: The user query.
            retrieved_passages: List of retrieved passage dictionaries.
            reasoning_chain: List of reasoning steps.
            visited_nodes: Set of visited node identifiers.
            reasoning_step: Current reasoning step (0-indexed).
            previous_steps: Previous steps in the reasoning process.
            preliminary_answer: Preliminary answer from output_answer tool.
            question_metadata: Metadata including date for temporal reasoning.
            interaction_type: Type of interaction for logging purposes.
            remem_instance: ReMem instance (needed for verbatim aggregation when enabled)
            skip_empty_step: If True, skip displaying steps without any retrieved passages/content.

        Returns:
            Generated answer as a string.
        """
        if not any([retrieved_passages, visited_nodes]):
            return "no information available"

        # Build reasoning context (now with previous_steps and optional verbatim aggregation)
        qa_context = self._build_qa_context(
            visited_nodes, previous_steps, preliminary_answer, remem_instance, skip_empty_step=skip_empty_step
        )

        # For enhanced_qa_context, prioritize Agent Interaction History
        question_line = f"Question: {query}"
        if question_metadata and "date" in question_metadata:
            question_line += f" (question date: {question_metadata['date']})"

        # Combine with context or provide fallback
        if qa_context.strip():
            qa_context = question_line + qa_context
        else:
            # Fallback: if no reasoning context, provide basic context
            qa_context = question_line + "\n"

        # Generate answer
        developer_prompt = get_remem_qa_developer_prompt(question_metadata)
        return self._call_llm_and_parse_response(
            qa_context,
            developer_prompt,
            interaction_type,
            reasoning_step,
            retrieved_passages,
            reasoning_chain,
            visited_nodes,
        )

    def _build_qa_context(
        self,
        visited_nodes: Optional[set] = None,
        previous_steps: Optional[List[Dict[str, Any]]] = None,
        preliminary_answer: Optional[str] = None,
        remem_instance: Optional[ReMem] = None,
        skip_empty_step: bool = False,
    ) -> str:
        """Build reasoning context from available sources.

        Args:
            visited_nodes: Set of visited node identifiers.
            previous_steps: List of step dictionaries produced during reasoning.
            preliminary_answer: Preliminary answer provided by the agent/tool.
            remem_instance: ReMem instance (for verbatim aggregation mode).
            skip_empty_step: If True, omit steps that have no retrieved passages/content.
        """
        qa_context = ""

        # Add preliminary answer if available, before agent interaction history
        if preliminary_answer:
            qa_context += f"\n\nPreliminary Answer (for reference): {preliminary_answer}"

        if previous_steps:
            if not self.only_use_verbatim_contexts:
                qa_context += "\n\nAgent Interaction History:\n"
            else:
                qa_context += "\n\n"

            # Check if we should use verbatim context aggregation
            if self.only_use_verbatim_contexts and remem_instance:
                # Aggregate gist/facts to verbatim
                verbatim_passages = aggregate_gist_facts_to_verbatim(
                    previous_steps, remem_instance, self.facts_weight, self.logger
                )

                if verbatim_passages:
                    qa_context += "Retrieved verbatim contexts:\n"
                    for i, passage in enumerate(verbatim_passages[: remem_instance.global_config.qa_top_k], 1):
                        content = passage["content"]
                        score = passage["score"]
                        source_count = passage.get("source_count", 0)

                        display_content = content[:2000] + "..." if len(content) > 2000 else content
                        qa_context += f"  {i}. {display_content}\n"
                    qa_context += "\n"
                else:
                    # Fallback to original format if verbatim aggregation fails
                    if self.logger:
                        self.logger.warning("Verbatim aggregation failed, falling back to retrieved contexts display")
                    qa_context += self._build_retrieved_contexts(previous_steps, skip_empty_step=skip_empty_step)
            else:
                # Use retrieved contexts display
                qa_context += self._build_retrieved_contexts(previous_steps, skip_empty_step=skip_empty_step)

            if visited_nodes and not self.only_use_verbatim_contexts:
                qa_context += f"Total Nodes Explored: {len(visited_nodes)}\n"

        return qa_context

    def _build_retrieved_contexts(self, previous_steps: List[Dict[str, Any]], skip_empty_step: bool = False) -> str:
        """Build retrieved contexts display for all content types (gists, facts, verbatim, etc.).

        Args:
            previous_steps: List of step dictionaries.
            skip_empty_step: If True, skip steps that have no non-empty retrieved passages.
        """
        context = ""

        # Pre-filter steps if skipping empty retrieval steps
        if skip_empty_step:
            filtered_steps = []
            for s in previous_steps:
                content_info = s.get("content_retrieval")
                passages = []
                if content_info:
                    passages = content_info.get("passages", []) or []
                # A step is considered empty if there are no passages with non-empty content strings
                has_content = False
                if passages:
                    for p in passages:
                        c = p.get("content", "")
                        if isinstance(c, str) and c.strip():
                            has_content = True
                            break
                if has_content:
                    filtered_steps.append(s)
            enumerated_steps = list(enumerate(filtered_steps, start=1))  # renumber sequentially
        else:
            enumerated_steps = []
            for idx, s in enumerate(previous_steps, start=1):
                enumerated_steps.append((s.get("step", idx), s))

        for display_step_num, step_data in enumerated_steps:
            # Use the display_step_num (renumbered) instead of original step when skipping; otherwise original provided
            step_num = display_step_num

            if "function" in step_data:
                func_name = step_data.get("function", "unknown")
                func_parameters = step_data.get("parameters", {})
                observation = step_data.get("observation", "")
                reasoning = step_data.get("reasoning", "")

                step_generation = {"reasoning": reasoning, "function": func_name, "parameters": func_parameters}
                context += f"Step {step_num}: {step_generation}\n"

                if observation:
                    context += f"  - Observation: {observation}\n"

                if "content_retrieval" in step_data:
                    content_info = step_data["content_retrieval"]
                    num_passages = content_info.get("num_passages_retrieved", 0)
                    passages = content_info.get("passages", [])

                    if num_passages > 0 and passages:
                        # Group contents by type using the same pattern as tool_selector
                        content_groups = self._group_contents_by_type(passages)
                        context += self._format_grouped_contents(content_groups)

                context += "\n"
            else:
                if self.logger:
                    self.logger.error("Step data is empty or not formatted correctly.")

        return context

    def _group_contents_by_type(self, passages: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """Group contents by type based on node_name patterns.

        Args:
            passages: List of passage items with node_name and content keys

        Returns:
            Dictionary mapping content type to list of contents
        """
        # Supported content types with their prefixes and display names
        content_types = {"verbatim-": "verbatim passages", "gists-": "gists", "facts-": "facts"}

        grouped_contents = {display_name: [] for display_name in content_types.values()}

        for content_item in passages:
            node_name = content_item.get("node_name", "")
            content = content_item.get("content", "")

            # Find matching content type
            for prefix, display_name in content_types.items():
                if node_name.startswith(prefix):
                    grouped_contents[display_name].append(content)
                    break

        return grouped_contents

    def _format_grouped_contents(self, content_groups: Dict[str, List[str]]) -> str:
        """Format grouped contents for display.

        Args:
            content_groups: Dictionary mapping content type to list of contents

        Returns:
            Formatted string for display
        """
        formatted_output = ""

        for content_type, contents in content_groups.items():
            if contents:
                formatted_output += f"  - Truncated top {content_type}:\n"
                max_content_length = 1500
                max_items = 10

                for i, content in enumerate(contents[:max_items], 1):
                    display_content = (
                        content[:max_content_length] + "..." if len(content) > max_content_length else content
                    )
                    formatted_output += f"    {i}. {display_content}\n"

        return formatted_output

    def _call_llm_and_parse_response(
        self,
        qa_context: str,
        developer_prompt: str,
        interaction_type: str,
        reasoning_step: int,
        retrieved_passages: Optional[List[Dict[str, Any]]] = None,
        reasoning_chain: Optional[List[str]] = None,
        visited_nodes: Optional[set] = None,
    ) -> str:
        """Call LLM and parse the response."""
        try:
            # Use the same message format as ReMem QA
            input_messages = [{"role": "system", "content": developer_prompt}, {"role": "user", "content": qa_context}]
            response, metadata, cache_hit = self.llm_client.infer(input_messages)

            # Log LLM interaction if callback is set
            if self.llm_interaction_callback:
                context_info = {}
                if retrieved_passages:
                    context_info.update(
                        {
                            "num_retrieved_passages": len(retrieved_passages),
                            "reasoning_chain_length": len(reasoning_chain) if reasoning_chain else 0,
                        }
                    )
                elif visited_nodes:
                    context_info.update(
                        {
                            "visited_nodes_count": len(visited_nodes),
                        }
                    )

                self.llm_interaction_callback(
                    messages=input_messages,
                    response=response,
                    metadata=metadata,
                    cache_hit=cache_hit,
                    interaction_type=interaction_type,
                    step=reasoning_step,
                    final_answer_context=context_info,
                )

            # Parse response similar to ReMem's answer extraction
            response_split = response.split("Answer:")
            if len(response_split) > 1:
                return response_split[1].strip()
            else:
                # If no "Answer:" found, return the whole response
                return response.strip()

        except Exception as e:
            if self.logger:
                self.logger.error(f"Error generating final answer: {e}")
            return "Error generating final answer."
