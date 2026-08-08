"""
TISER-based final answer generation logic for ReMem Agent
"""

from typing import Any, Callable, Dict, List, Optional

from remem.remem import ReMem

from .answer_generator import FinalAnswerGenerator


def get_tiser_developer_prompt() -> str:
    from baselines.tiser import tiser_developer

    return tiser_developer


class TiserFinalAnswerGenerator(FinalAnswerGenerator):
    """TISER-based logic for generating final answers from retrieved context."""

    def __init__(
        self,
        llm_client,
        logger=None,
        llm_interaction_callback: Optional[Callable] = None,
        qa_prompt_template: Optional[str] = None,
        use_verbatim_context: bool = False,
        facts_weight: float = 1.0,
    ):
        # Initialize parent class
        super().__init__(
            llm_client=llm_client,
            logger=logger,
            llm_interaction_callback=llm_interaction_callback,
            use_verbatim_context=use_verbatim_context,
            facts_weight=facts_weight,
        )
        self.qa_prompt_template = qa_prompt_template

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
    ) -> str:
        """
        Generate an answer based on retrieved passages and/or reasoning chain using TISER format.

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
            remem_instance: ReMem instance for context building.

        Returns:
            Generated answer as a string.
        """
        if not any([retrieved_passages, visited_nodes]):
            return "no information available"

        # Check if a specific QA prompt template is provided
        if self.qa_prompt_template:
            return self._generate_answer_with_template(
                query,
                retrieved_passages,
                reasoning_chain,
                visited_nodes,
                reasoning_step,
                previous_steps,
                preliminary_answer,
                question_metadata,
                interaction_type,
                remem_instance,
            )

        # Default TISER format - reuse parent's context building logic
        qa_context = self._build_qa_context(visited_nodes, previous_steps, preliminary_answer, remem_instance)

        # Create TISER-formatted user message
        if qa_context.strip():
            question_line = f"Question: {query}"
            if question_metadata and "date" in question_metadata:
                question_line += f" (question date: {question_metadata['date']})"

            # For TISER, format as "Question: ... \nTemporal Context: ..."
            tiser_context = qa_context.replace(question_line, "").strip()
            if tiser_context:
                user_message = question_line + f"\nTemporal Context: {tiser_context}"
            else:
                user_message = question_line + "\nTemporal Context: No temporal context information available."
        else:
            # Fallback: if no reasoning context, provide basic context
            question_line = f"Question: {query}"
            if question_metadata and "date" in question_metadata:
                question_line += f" (question date: {question_metadata['date']})"
            user_message = question_line + "\nTemporal Context: No temporal context information available."

        # Generate answer using TISER format
        developer_prompt = get_tiser_developer_prompt()
        return self._call_llm_and_parse_tiser_response(
            user_message,
            developer_prompt,
            interaction_type,
            reasoning_step,
            retrieved_passages,
            reasoning_chain,
            visited_nodes,
        )

    def _generate_answer_with_template(
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
    ) -> str:
        """Generate answer using a specific QA prompt template."""
        if not self.qa_prompt_template:
            return "No QA prompt template provided"

        try:
            # Import and load the QA prompt template
            from remem.prompts.prompt_template_manager import PromptTemplateManager

            template_manager = PromptTemplateManager()
            template = template_manager.get_template(self.qa_prompt_template)

            # Build context for the template using parent's method
            qa_context = self._build_qa_context(visited_nodes, previous_steps, preliminary_answer, remem_instance)

            # Format the template with the context
            if isinstance(template, list):
                # Handle chat-style templates
                input_messages = []
                for msg in template:
                    if msg.get("role") == "user" and "${prompt_user}" in msg.get("content", ""):
                        # Replace template variables
                        content = msg["content"]
                        content = content.replace("${prompt_user}", qa_context)
                        if "${question}" in content:
                            content = content.replace("${question}", query)
                        input_messages.append({"role": msg["role"], "content": content})
                    else:
                        input_messages.append(msg)

                # Call LLM with the formatted template
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

                # Parse response based on template type
                if "tiser" in self.qa_prompt_template.lower():
                    # TISER format - extract from <answer> tags
                    if "<answer>" in response and "</answer>" in response:
                        start_idx = response.find("<answer>") + len("<answer>")
                        end_idx = response.find("</answer>")
                        return response[start_idx:end_idx].strip()
                    else:
                        if self.logger:
                            self.logger.warning("No <answer> tags found in TISER response")
                        return response.strip()
                else:
                    # Standard format - extract from "Answer:"
                    response_split = response.split("Answer:")
                    if len(response_split) > 1:
                        return response_split[1].strip()
                    else:
                        return response.strip()
            else:
                # Handle string templates
                if self.logger:
                    self.logger.warning("String template format not fully supported for QA prompt templates")
                return "Template format not supported"

        except Exception as e:
            if self.logger:
                self.logger.error(f"Error generating answer with template {self.qa_prompt_template}: {e}")
            return "Error generating answer with template"

    def _call_llm_and_parse_tiser_response(
        self,
        user_message: str,
        developer_prompt: str,
        interaction_type: str,
        reasoning_step: int,
        retrieved_passages: Optional[List[Dict[str, Any]]] = None,
        reasoning_chain: Optional[List[str]] = None,
        visited_nodes: Optional[set] = None,
    ) -> str:
        """Call LLM and parse the TISER response."""
        try:
            # Use TISER message format with developer and user roles
            input_messages = [
                {"role": "developer", "content": developer_prompt},
                {"role": "user", "content": user_message},
            ]
            response, metadata, cache_hit = self.llm_client.infer(input_messages)

            # Log LLM interaction if callback is set (reuse parent's logging logic)
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

            # Parse TISER response - extract content from <answer> tags
            if "<answer>" in response and "</answer>" in response:
                start_idx = response.find("<answer>") + len("<answer>")
                end_idx = response.find("</answer>")
                answer = response[start_idx:end_idx].strip()
                return answer
            else:
                # Fallback: if no <answer> tags found, return the whole response
                if self.logger:
                    self.logger.warning("No <answer> tags found in TISER response, returning full response")
                return response.strip()

        except Exception as e:
            if self.logger:
                self.logger.error(f"Error generating final answer with TISER: {e}")
            return "Error generating final answer."


def create_tiser_messages(question: str, temporal_context: str):
    """
    Create the TISER messages with the parsed question and temporal context.

    This function is kept for compatibility with the original tiser.py interface.

    Args:
        question: The extracted question
        temporal_context: The extracted temporal facts

    Returns:
        The formatted TISER messages
    """
    tiser_developer = get_tiser_developer_prompt()
    tiser_user = f"Question: {question}\nTemporal Context: {temporal_context}"

    messages = [{"role": "developer", "content": tiser_developer}, {"role": "user", "content": tiser_user}]
    return messages
