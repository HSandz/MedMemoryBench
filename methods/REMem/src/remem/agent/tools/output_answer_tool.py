from remem.agent.answer_generator import FinalAnswerGenerator
from remem.agent.tools.base_tool import BaseTool, ToolContext, ToolResult, ToolType


class OutputAnswerTool(BaseTool):
    """Tool for generating final answer when agent believes it has enough context."""

    def __init__(self, remem_instance, llm_client):
        super().__init__(ToolType.OUTPUT_ANSWER, remem_instance)
        self.llm_client = llm_client
        self.llm_interaction_callback = None  # Callback for logging LLM interactions

    def set_llm_interaction_callback(self, callback):
        """Set callback function for logging LLM interactions."""
        self.llm_interaction_callback = callback

    def get_description(self) -> str:
        """Get tool description for LLM-based selection."""
        import json

        return json.dumps(
            {
                "type": "function",
                "name": "output_answer",
                "description": "When you believe you have information, analyze retrieved information and corresponding questions meticulously, answer the original query and end the search process.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "answer": {
                            "type": "string",
                            "description": "Present a concise, definitive response, devoid of additional elaborations. Can be string, integer, or any format required by the original query.",
                        }
                    },
                    "required": ["answer"],
                },
            }
        )

    def can_execute(self, context: ToolContext) -> bool:
        """Can execute if we have collected some information and are not at the first step."""
        return context.reasoning_step > 0 and len(context.visited_nodes) > 0

    def execute(self, context: ToolContext, max_results: int = 5, **kwargs) -> ToolResult:
        """Use the final answer provided by the LLM tool selection."""
        try:
            # Get the final answer from the tool parameters (provided by LLM during tool selection)
            generated_answer = kwargs.get("answer", "")

            # Set the preliminary answer from the tool parameter immediately for this step
            if generated_answer:
                context.preliminary_answer = generated_answer

            if not generated_answer:
                # Fallback: if no answer provided in parameters, generate one
                self.logger.warning("No answer provided in tool parameters, generating final answer")
                generated_answer = self._generate_final_answer(context)
            else:  # a preliminary answer was provided by output_answer tool
                selected = [
                    "locomo",
                    "longmemeval",
                    "musique",
                    "complex_tr",
                    "realtalk",
                    "fact_consolidation",
                    "2wikimultihopqa",
                ]
                if any(self.remem.global_config.dataset.startswith(prefix) for prefix in selected):
                    generated_answer = self._generate_final_answer(context)

            final_answer = None
            if self.remem.global_config.dataset.startswith("tot"):
                from examples.tot_semantic import remove_spaces_from_prediction

                if isinstance(generated_answer, str) and generated_answer.startswith("E") and ", " in generated_answer:
                    final_answer = remove_spaces_from_prediction(generated_answer)
            if final_answer is None:
                final_answer = generated_answer

            # Store the final answer in a way that can be retrieved by the agent
            observation = f"Generated final answer: {final_answer}"

            return ToolResult(
                tool_type=ToolType.OUTPUT_ANSWER,
                nodes_found=[],
                scores=[],
                observation=observation,
                should_continue=False,
                node_contents=[],
            )

        except Exception as e:
            self.logger.error(f"Error processing final answer: {e}")
            return ToolResult(
                tool_type=ToolType.OUTPUT_ANSWER,
                nodes_found=[],
                scores=[],
                observation=f"Error processing final answer: {e}",
                should_continue=False,
                node_contents=[],
            )

    def _generate_final_answer(self, context: ToolContext) -> str:
        """Generate final answer from collected context using shared answer generator."""

        if self.remem.global_config.qa_reader == "remem":
            answer_generator = FinalAnswerGenerator(self.llm_client, self.logger, self.llm_interaction_callback)
        elif self.remem.global_config.qa_reader == "tiser":
            from remem.agent.answer_generator_tiser import TiserFinalAnswerGenerator

            answer_generator = TiserFinalAnswerGenerator(self.llm_client, self.logger, self.llm_interaction_callback)
        else:
            raise ValueError(f"Unsupported qa_reader: {self.remem.global_config.qa_reader}")

        return answer_generator.generate_answer(
            query=context.query,
            visited_nodes=context.visited_nodes,
            reasoning_step=context.reasoning_step,
            previous_steps=context.previous_steps,
            preliminary_answer=context.preliminary_answer,
            question_metadata=context.question_metadata,
            interaction_type="final_answer_generation",
        )
