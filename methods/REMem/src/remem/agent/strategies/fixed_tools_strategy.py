from typing import Any, Dict, List, Optional, Tuple

from remem.agent import AgentConfig
from remem.agent.strategies.tool_strategy import RetrievalResult, ToolStrategy


class FixedToolsStrategy(ToolStrategy):
    """Strategy using fixed tools (semantic_retrieve + output_answer)."""

    def __init__(
        self,
        agent_config: AgentConfig,
        tool_selector,
        node_utils,
        agent_logger,
        answer_generator,
        node_chunks_dict,
        remem,
        logger,
    ):
        super().__init__(agent_config)
        self.tool_selector = tool_selector
        self.node_utils = node_utils
        self.agent_logger = agent_logger
        self.answer_generator = answer_generator
        self.node_chunks_dict = node_chunks_dict
        self.remem = remem
        self.logger = logger

    def retrieve(
        self,
        query: str,
        max_results: int = 10,
        return_chunk: Optional[str] = None,
        gold_answer: Optional[str] = None,
        question_metadata: Optional[Dict[str, Any]] = None,
    ) -> RetrievalResult:
        """Execute fixed tools retrieval."""

        max_steps = self.agent_config.max_steps
        self.agent_logger.log_session_start(query, max_steps, gold_answer)

        from remem.agent import AgentState

        state = AgentState(query=query, max_steps=max_steps, question_metadata=question_metadata)

        try:
            # Step 1: Retrieval
            context = self._create_context(state, 0, max_steps)
            retrieval_result = self._execute_retrieval_tool(context, max_results)
            self._update_state_with_retrieval(state, retrieval_result)

            # Log the retrieval step properly
            new_passages = state.retrieved_passages  # These were just added in _update_state_with_retrieval
            self.agent_logger.log_step(1, retrieval_result, new_passages, context)

            # Step 2: Answer generation (if max_steps >= 2)
            if max_steps >= 2:
                self._generate_answer_with_tool(state, max_results)
            else:
                self._generate_fallback_answer(state)

        except Exception as e:
            self.logger.error(f"Error in fixed tools strategy: {e}")
            self._handle_error(state)

        return self._format_result(state, return_chunk)

    def _create_context(self, state, reasoning_step: int, max_steps: int):
        """Create tool context."""
        from remem.agent.tools import ToolContext

        return ToolContext(
            query=state.query,
            visited_nodes=state.visited_nodes,
            reasoning_step=reasoning_step,
            max_steps=max_steps,
            available_entries=self.node_utils.get_available_entries(self.remem, self.node_chunks_dict),
            content_map=self.node_utils.get_content_map_for_nodes(state.visited_nodes, self.node_chunks_dict),
            previous_steps=[],
            question_metadata=getattr(state, "question_metadata", None),
        )

    def _execute_retrieval_tool(self, context, max_results: int):
        """Execute the configured retrieval tool."""
        from remem.agent.tools import ToolType

        retrieval_tool_name = getattr(self.agent_config, "fixed_retrieval_tool", "semantic_retrieve")

        if retrieval_tool_name == "lexical_retrieve":
            tool = self.tool_selector.tools[ToolType.LEXICAL_RETRIEVE]
        else:
            tool = self.tool_selector.tools[ToolType.SEMANTIC_RETRIEVE]

        return tool.execute(context, max_results)

    def _update_state_with_retrieval(self, state, retrieval_result):
        """Update agent state with retrieval results."""
        self.logger.info(
            f"Retrieval result: nodes_found={len(retrieval_result.nodes_found)}, observation={retrieval_result.observation}"
        )

        state.visited_nodes.update(retrieval_result.nodes_found)
        state.current_focus_nodes = retrieval_result.nodes_found

        # Collect passages
        if hasattr(retrieval_result, "node_contents") and retrieval_result.node_contents:
            self.logger.info(f"Using node_contents from retrieval_result: {len(retrieval_result.node_contents)} items")
            new_passages = self._create_passages_from_tool_result(retrieval_result, 1)
        else:
            self.logger.info("Using node_utils.collect_node_content_with_scores")
            new_passages = self.node_utils.collect_node_content_with_scores(
                retrieval_result.nodes_found, retrieval_result.scores, 1, self.node_chunks_dict, self.remem
            )

        self.logger.info(f"Retrieved {len(new_passages)} passages")
        for i, passage in enumerate(new_passages[:3]):  # Log first 3 passages
            content_preview = (
                passage.get("content", "")[:100] + "..."
                if len(passage.get("content", "")) > 100
                else passage.get("content", "")
            )
            self.logger.info(f"Passage {i+1}: node={passage.get('node_name')}, content_preview='{content_preview}'")

        state.retrieved_passages.extend(new_passages)
        state.reasoning_chain.append(f"Tool: {retrieval_result.observation}")
        state.current_step = 1

    def _create_passages_from_tool_result(self, tool_result, step: int):
        """Create passages from tool result."""
        passages = []
        for node_key, score, content in zip(tool_result.nodes_found, tool_result.scores, tool_result.node_contents):
            node_type = self.node_utils.get_node_type(node_key, self.remem)
            passages.append(
                {
                    "node_name": node_key,
                    "content": content,
                    "embedding_score": score,
                    "step": step,
                    "node_type": node_type,
                }
            )
        return passages

    def _generate_answer_with_tool(self, state, max_results: int):
        """Generate answer using output tool."""
        from remem.agent.tools import ToolType

        context = self._create_context(state, 1, state.max_steps)
        output_tool = self.tool_selector.tools[ToolType.OUTPUT_ANSWER]
        answer_result = output_tool.execute(context, max_results)

        state.final_answer_generated = True
        if "Generated final answer: " in answer_result.observation:
            state.final_answer = answer_result.observation.replace("Generated final answer: ", "")
        else:
            state.final_answer = answer_result.observation

        state.reasoning_chain.append("Tool: Final answer generated by tool")
        state.current_step = 2

    def _generate_fallback_answer(self, state):
        """Generate fallback answer for 1-step mode."""
        if state.retrieved_passages:
            # Build previous_steps with retrieval content for proper context generation
            previous_steps = [
                {
                    "step": 1,
                    "function": "semantic_retrieve",
                    "content_retrieval": {
                        "num_passages_retrieved": len(state.retrieved_passages),
                        "passages": state.retrieved_passages,
                    },
                }
            ]

            # Generate answer with proper step tracking and retrieved passages
            state.final_answer = self.answer_generator.generate_answer(
                query=state.query,
                retrieved_passages=state.retrieved_passages,  # Pass the actual retrieved passages
                visited_nodes=state.visited_nodes,
                reasoning_step=1,  # This should be step 1 (0-indexed)
                previous_steps=previous_steps,  # Now includes retrieval content
                question_metadata=state.question_metadata,
                interaction_type="final_answer_generation",
                remem_instance=self.remem,
            )
            state.final_answer_generated = True
        else:
            state.final_answer = "Unable to retrieve relevant information to answer the query."
            state.final_answer_generated = False

        state.current_step = 2  # Set to 2 since we completed answer generation

    def _handle_error(self, state):
        """Handle errors during retrieval."""
        if state.retrieved_passages:
            # Build previous_steps with retrieval content for proper context generation
            previous_steps = [
                {
                    "step": 1,
                    "function": "semantic_retrieve",
                    "content_retrieval": {
                        "num_passages_retrieved": len(state.retrieved_passages),
                        "passages": state.retrieved_passages,
                    },
                }
            ]

            state.final_answer = self.answer_generator.generate_answer(
                query=state.query,
                retrieved_passages=state.retrieved_passages,  # Pass the actual retrieved passages
                visited_nodes=state.visited_nodes,
                reasoning_step=state.current_step,
                previous_steps=previous_steps,  # Now includes retrieval content
                question_metadata=state.question_metadata,
                interaction_type="final_answer_generation",
                remem_instance=self.remem,
            )
        else:
            state.final_answer = "Unable to retrieve relevant information to answer the query."

    def _format_result(self, state, return_chunk: Optional[str]) -> RetrievalResult:
        """Format the final result."""
        chunk_ids, chunk_scores = self._format_agent_results(state, return_chunk)

        logs = {
            "agent_observation": state.reasoning_chain,
            "agent_answer": state.final_answer,
            "agent_chunks": [p["content"] for p in state.retrieved_passages if p.get("content")],
            "agent_session_logs": self.agent_logger._get_session_logs(),
            "agent_type": "fixed",
        }

        return RetrievalResult(
            chunk_ids=chunk_ids, chunk_scores=chunk_scores, logs=logs, final_answer=state.final_answer
        )

    def _format_agent_results(self, state, return_chunk: Optional[str] = None) -> Tuple[List[str], List[float]]:
        """Format agent results to match traditional PPR output format."""
        if not state.retrieved_passages:
            return [], []

        # Filter passages by return_chunk type if specified
        filtered_passages = []
        for passage in state.retrieved_passages:
            node_name = passage.get("node_name", "")
            node_type = self.node_utils.get_node_type(node_name, self.remem)

            if return_chunk == "verbatim" and node_type == "verbatim":
                filtered_passages.append(passage)
            elif return_chunk == "gists" and node_type == "gists":
                filtered_passages.append(passage)
            elif return_chunk is None:
                filtered_passages.append(passage)

        if not filtered_passages:
            return [], []

        # Extract node names and use embedding scores
        chunk_id_to_max_score = {}
        for passage in filtered_passages:
            node_name = passage.get("node_name")
            if node_name:
                embedding_score = passage.get("embedding_score", 0.0)
                chunk_id_to_max_score[node_name] = max(chunk_id_to_max_score.get(node_name, 0.0), embedding_score)

        # Sort by score descending
        sorted_items = sorted(chunk_id_to_max_score.items(), key=lambda x: x[1], reverse=True)
        return [item[0] for item in sorted_items], [item[1] for item in sorted_items]
