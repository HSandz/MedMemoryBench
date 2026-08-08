import logging
from typing import Dict, List

from remem.agent.tools.base_tool import ToolContext, ToolResult, ToolType
from remem.agent.tools.find_entity_contexts_tool import FindEntityContextsTool
from remem.agent.tools.find_gist_contexts_tool import FindGistContextsTool
from remem.agent.tools.lexical_retrieve_tool import LexicalRetrieveTool
from remem.agent.tools.output_answer_tool import OutputAnswerTool
from remem.agent.tools.semantic_retrieve_tool import SemanticRetrieveTool


class ToolSelector:
    """LLM-based tool selector for autonomous tool choice."""

    def __init__(self, remem_instance, llm_client, use_demo=True):
        """Initialize tool selector.

        Args:
            remem_instance: The ReMem instance
            llm_client: The LLM client for tool selection
            use_demo: Whether to include the 1-shot demo in tool selection prompts (default: True)
        """
        self.remem = remem_instance
        self.llm_client = llm_client
        self.logger = logging.getLogger(f"{__name__}.ToolSelector")
        self.llm_interaction_callback = None  # Callback for logging LLM interactions
        self.use_demo = use_demo  # Configuration for 1-shot demo usage

        # Initialize all tools
        self.tools = {
            ToolType.OUTPUT_ANSWER: OutputAnswerTool(remem_instance, llm_client),
            ToolType.FIND_GIST_CONTEXTS: FindGistContextsTool(remem_instance),
            ToolType.FIND_ENTITY_CONTEXTS: FindEntityContextsTool(remem_instance),
            ToolType.SEMANTIC_RETRIEVE: SemanticRetrieveTool(remem_instance),
            ToolType.LEXICAL_RETRIEVE: LexicalRetrieveTool(remem_instance),
        }

    def set_llm_interaction_callback(self, callback):
        """Set callback function for logging LLM interactions."""
        self.llm_interaction_callback = callback
        # Also set it for the final answer tool which makes LLM calls
        if ToolType.OUTPUT_ANSWER in self.tools:
            self.tools[ToolType.OUTPUT_ANSWER].set_llm_interaction_callback(callback)

    def select_and_execute_tool(self, context: ToolContext, max_results: int = 5, first_full_query=False) -> ToolResult:
        """Select the best tool based on context and execute it."""
        try:
            # Get available tools that can execute
            available_tools = []
            for tool_type, tool in self.tools.items():
                if tool.can_execute(context):
                    available_tools.append(tool.get_description())

            if not available_tools:
                self.logger.warning("No tools available for execution")
                return self._fallback_result(context)

            # Use LLM to select best tool and parameters
            selection_result = self._llm_select_tool_with_params(context, available_tools, max_results)
            selected_tool_name = selection_result["tool_name"]
            tool_params = selection_result.get("parameters", {})
            selection_reasoning = selection_result.get("reasoning", "")

            # Find matching tool type
            selected_tool_type = None
            for tool_type in self.tools:
                if tool_type.value == selected_tool_name:
                    selected_tool_type = tool_type
                    break

            if not selected_tool_type:
                self.logger.warning(f"LLM selected unknown tool: {selected_tool_name}")
                selected_tool_type = ToolType.SEMANTIC_RETRIEVE  # Fallback to new primary tool

            if first_full_query:
                # If this is the very first step (no previous steps) and the selected tool is
                # semantic_retrieve or lexical_retrieve, always override any LLM-provided
                # 'query' parameter with the original user query. This ensures the initial
                # retrieval uses the full question rather than any reformulated / partial form.
                if (not context.previous_steps or len(context.previous_steps) == 0) and selected_tool_type in (
                    ToolType.SEMANTIC_RETRIEVE,
                    ToolType.LEXICAL_RETRIEVE,
                ):
                    self.logger.debug(
                        f"First-step retrieval: overriding `query` param from `{tool_params['query']}` to full query `{context.query}`"
                    )
                    tool_params["query"] = context.query

            # Execute selected tool with parameters
            selected_tool = self.tools[selected_tool_type]
            tool_result = selected_tool.execute(context, max_results, **tool_params)

            # Add the parameters and selection reasoning to the tool result for logging
            tool_result.parameters = tool_params
            tool_result.selection_reasoning = selection_reasoning

            return tool_result

        except Exception as e:
            self.logger.error(f"Error in tool selection/execution: {e}")
            return self._fallback_result(context)

    def _llm_select_tool_with_params(
        self, context: ToolContext, available_tools: List[str], max_results: int = 10
    ) -> Dict:
        """Use LLM to select the best tool and its parameters."""
        tools_info = []
        for tool_idx, tool in enumerate(available_tools):
            tools_info.append(f"{tool_idx + 1}. {tool}")

        # Build previous steps section
        previous_steps_section = ""
        if context.previous_steps:
            previous_steps_section = "**Previous Steps:**\n"
            for step_log in context.previous_steps:
                step_num = step_log.get("step", "?")

                if "function" in step_log:
                    func_name = step_log.get("function", "unknown")
                    tool_reasoning = step_log.get("reasoning", "")
                    tool_observation = step_log.get("observation", "")
                    tool_parameters = step_log.get("parameters", {})

                    # Get content retrieval info directly from step_log
                    content_retrieval = step_log.get("content_retrieval", {})
                    num_passages = content_retrieval.get("num_passages_retrieved", 0)
                    top_k_content = content_retrieval.get("passages", [])
                else:
                    self.logger.error(f"Step {step_num} has no function calling")
                    continue

                step_generation = {"reasoning": tool_reasoning, "function": func_name, "parameters": tool_parameters}
                previous_steps_section += f"Step {step_num}: {step_generation}\n"
                if tool_observation:
                    previous_steps_section += f"  - Observation: {tool_observation}\n"
                if num_passages > 0:
                    # Show top k content for better understanding
                    if top_k_content:
                        # Group contents by type based on node_name patterns
                        content_groups = self._group_contents_by_type(top_k_content)

                        # Display grouped contents
                        previous_steps_section += self._format_grouped_contents(content_groups, max_results)

                previous_steps_section += "\n"
        else:
            previous_steps_section = "**Previous Steps:** No steps taken yet.\n"

        # Get entity and gists statistics if available
        graph_stats_parts = []
        try:
            if hasattr(self.remem, "episodic_embedding_stores") and self.remem.episodic_embedding_stores:
                # Get gists count if available
                gists_store = self.remem.episodic_embedding_stores.get("gists")
                if gists_store:
                    num_gists = len(gists_store.get_all_ids())
                    graph_stats_parts.append(f"{num_gists} gists")

                # Get entities count
                entity_store = self.remem.episodic_embedding_stores.get("entity")
                if entity_store:
                    num_entities = len(entity_store.get_all_ids())
                    graph_stats_parts.append(f"{num_entities} entities")

                # Count entity-to-entity edges if possible
                entity_edges = 0
                if hasattr(self.remem, "graph") and self.remem.graph.ecount() > 0:
                    try:
                        for edge in self.remem.graph.es:
                            source_name = self.remem.graph.vs[edge.source].attributes().get("name", "")
                            target_name = self.remem.graph.vs[edge.target].attributes().get("name", "")
                            if source_name.startswith("entity-") and target_name.startswith("entity-"):
                                entity_edges += 1
                    except:
                        pass
                if entity_edges > 0:
                    graph_stats_parts.append(f"{entity_edges} edges between entities")
        except Exception as e:
            self.logger.error(f"Error retrieving graph statistics: {e}")

        # Format graph statistics
        graph_stats_info = ""
        if graph_stats_parts:
            graph_stats_info = f"**Graph Statistics:** {', '.join(graph_stats_parts)}"

        # Build the demo section conditionally
        demo_section = ""
        if self.use_demo:
            demo_section = """

**Demo:**
For the query 'Find the entity that was the R11 of E57 immediately after E95 was the R11 of E57.', each step are like this:
Step 1: {{"reasoning": "Find the end time of the event `E95 was the R11 of E57`, which is the start time of the next event that `was the R11 of E57`", "function": "lexical_retrieve", "parameters": {{"query": "E95 was the R11 of E57"}}}}
  - Observation: Retrieved 5 gists and 3 triples via BM25 lexical matching, showing 5 gists and 3 triples
  - Truncated top contents:
    1. E95 was appointed as the R11 of E57 on January 15, 2020. This position involves managing the technical operations and strategic planning for entity E57.
    2. (E95, was the R11 of, E57) {{"start_time": "2020-01-15", "end_time": "2022-03-20"}}
    3. The R11 role for E57 was held by E95 from 2020 to 2022, during which time significant improvements were made to the organization's infrastructure.

Step 2: {{"reasoning": "Find the next event that was the R11 of E57 after the retrieved time t1", "function": "find_entity_contexts", "parameters": {{"object": "E57", "relation": "was the R11 of", "start_time": "t1"}}}}
  - Observation: Found 12 connected gists and 8 connected triples for entity E57 (filtered by time: t1 to any), showing 15 gists and 15 triples
  - Truncated top contents:
    1. (E0, was the R11 of, E57) {{"start_time": "2022-03-21", "end_time": "2023-06-15"}}
    2. E0 succeeded E95 as the R11 of E57 on March 21, 2022, immediately after E95's term ended. E0 brought new perspectives and strategies to the role.
    3. (E95, was the R11 of, E57) {{"start_time": "2020-01-15", "end_time": "2022-03-20"}}

Step 3: {{"reasoning": "The entity that is closest to t1 and matches `was the R11 of E57` is E0", "function": "output_answer", "parameters": {{"answer": "E0"}}}}"""

        developer_prompt = f"""You are an intelligent agent exploring a large knowledge graph (KG) to answer a query. Select the best tool and its parameters given the current contexts.

**Instructions:**
Understand the contexts for previous and current steps and consider whether we need to explore on the KG or output the answer.
Pay attention to the observations from previous steps - they tell you what was found and how much information was retrieved.
Plan ahead and select one best tool from the available tools and provide its parameters for current step.

Tool Use Strategy:
1. Initial Retrieval: Begin by using `semantic_retrieve` or `lexical_retrieve` to understand the data schema and gather preliminary context. The goal is to identify the structure of the KG. For complex queries, break them down into simpler sub-queries to guide the next steps.
- Prioritize `semantic_retrieve` over `lexical_retrieve`, unless the query specifically targets entity identifiers or requires an exact match.
- Note: These tools return truncated content. This initial scan is for identifying key information (e.g., entity IDs, time ranges) required for a more detailed search.
2. Focused Exploration: Next, use `find_gist_contexts` or `find_entity_contexts` to retrieve specific, detailed information.
- Populate the parameters of these tools (e.g., subject, object, limit, ordering) using the key information discovered in the initial retrieval step.
- Prioritize `find_gist_contexts` over `find_entity_contexts`. Use the latter one ONLY when a query specifically targets entities with a well-defined KG schema you've seen; otherwise, DO NOT try to match the exact phrase using `find_entity_contexts`.
3. Answer Generation: Once you have gathered information and are confident in the answer, use the `output_answer` tool to provide the final result.

Temporal Reasoning Example:
- Find facts start after 1950: start_time='1950', start_operator='gt'
- Find facts end before 1960: end_time='1960', end_operator='lt'
- Find facts exist in specific point '1955': start_time='1955', start_operator='le', end_time='1955', end_operator='ge'
- Find facts only exist within [1950, 1960]: start_time='1950', start_operator='ge', end_time='1960', end_operator='le'
- Find facts that overlap with [1950, 1960]: start_time='1960', start_operator='le', end_time='1950', end_operator='ge' (note: a special case where the start time is larger than the end time) 

**Output Format:**
Respond with a JSON object, e.g.,
{{
    "reasoning": "Brief explanation of why this tool and parameters for this step",
    "function": "exact_tool_name_from_list",
    "parameters": {{
        "name": "value",
    }}
}}{demo_section}"""

        user_prompt = f"""{graph_stats_info}

**Query:** {context.query}
{f"**Query Date:** {context.question_metadata['date']}" if context.question_metadata and 'date' in context.question_metadata else ""}

**Available Tools:**

{(chr(10) + chr(10)).join(tools_info)}

{previous_steps_section}"""

        try:
            messages = [{"role": "developer", "content": developer_prompt}, {"role": "user", "content": user_prompt}]
            response, metadata, cache_hit = self.llm_client.infer(messages, response_format={"type": "json_object"})

            # Log LLM interaction if callback is set
            import json

            if self.llm_interaction_callback:
                self.llm_interaction_callback(
                    messages=messages,
                    response=response,
                    metadata=metadata,
                    cache_hit=cache_hit,
                    interaction_type="tool_selection",
                    step=context.reasoning_step + 1,  # Use 1-based step numbering
                    tool_selection_context={
                        "available_tools": [json.loads(t).get("name", "") for t in available_tools],
                        "visited_nodes_count": len(context.visited_nodes),
                        "available_entry_types": list(context.available_entries.keys()),
                    },
                )

            # Parse JSON response
            import json

            from remem.utils.llm_utils import parse_json_from_json_mode

            response_data = parse_json_from_json_mode(response)  # Use eval to safely parse the JSON response
            function_name = response_data.get("function", None)
            parameters = response_data.get("parameters", {})
            reasoning = response_data.get("reasoning", "LLM selection")

            # Clean up parameters - remove None/empty values
            clean_args = {k: v for k, v in parameters.items() if v and v != "null" and v != ""}

            # Simplified log output, showing only key information
            if self.logger:
                simplified_reasoning = reasoning[:500] + "..." if len(reasoning) > 500 else reasoning
                self.logger.debug(f"🛠️ Tool: {function_name} | {simplified_reasoning}")

            return {"reasoning": reasoning, "tool_name": function_name, "parameters": clean_args}

        except Exception as e:
            self.logger.error(f"Error calling LLM for tool selection: {e}")
            # Fallback response
            return {
                "tool_name": "semantic_retrieve",
                "parameters": {},
                "reasoning": "LLM selection failed, using fallback",
            }

    def _fallback_result(self, context: ToolContext) -> ToolResult:
        """Provide fallback result when tool selection fails."""
        return ToolResult(
            tool_type=ToolType.SEMANTIC_RETRIEVE,
            nodes_found=[],
            scores=[],
            observation="Tool selection failed, using fallback",
            should_continue=False,
            selection_reasoning="Tool selection failed, using fallback",
            node_contents=[],
        )

    def _group_contents_by_type(self, top_k_content: List[Dict]) -> Dict[str, List[str]]:
        """Group contents by type based on node_name patterns.

        Args:
            top_k_content: List of content items with node_name and content keys

        Returns:
            Dictionary mapping content type to list of contents
        """
        # Supported content types with their prefixes and display names
        content_types = {"verbatim-": "verbatim passages", "gists-": "gists", "facts-": "facts"}

        grouped_contents = {display_name: [] for display_name in content_types.values()}

        for content_item in top_k_content:
            node_name = content_item.get("node_name", "")
            content = content_item.get("content", "")

            # Find matching content type
            for prefix, display_name in content_types.items():
                if node_name.startswith(prefix):
                    grouped_contents[display_name].append(content)
                    break

        return grouped_contents

    def _format_grouped_contents(self, content_groups: Dict[str, List[str]], max_results: int) -> str:
        """Format grouped contents for display.

        Args:
            content_groups: Dictionary mapping content type to list of contents
            max_results: Maximum number of items to show per type

        Returns:
            Formatted string for display
        """
        formatted_output = ""

        for content_type, contents in content_groups.items():
            if contents:
                formatted_output += f"  - Truncated top {content_type}:\n"
                for i, content in enumerate(contents[:max_results], 1):
                    display_content = content[:1000] + "..." if len(content) > 1000 else content
                    formatted_output += f"    {i}. {display_content}\n"

        return formatted_output
