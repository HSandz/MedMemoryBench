import ast
import difflib
import json
import re
from copy import deepcopy
from typing import List, Tuple

from pydantic import BaseModel, Field, TypeAdapter


class Fact(BaseModel):
    fact: list[list[str]] = Field(
        description="A list of facts, each fact is a list of 3 strings: [subject, predicate, object]"
    )


class Gist(BaseModel):
    gist: list[str] = Field(description="A list of gist summaries/abstracts that are relevant to the query")


class DSPyFilter:
    def __init__(self, remem):
        dspy_file_path = remem.global_config.rerank_dspy_file_path
        self.one_input_template = """[[ ## question ## ]]\n{question}\n\n[[ ## fact_before_filter ## ]]\n{fact_before_filter}\n\nRespond with the corresponding output fields, starting with the field `[[ ## fact_after_filter ## ]]` (must be formatted as a valid Python Fact), and then ending with the marker for `[[ ## completed ## ]]`."""
        self.one_output_template = """[[ ## fact_after_filter ## ]]\n{fact_after_filter}\n\n[[ ## completed ## ]]"""
        self.message_template = self.make_template(dspy_file_path)
        self.llm_infer_fn = remem.llm.infer
        self.model_name = remem.global_config.llm_name
        self.default_gen_kwargs = {}

    def make_template(self, dspy_file_path):
        dspy_saved = json.load(open(dspy_file_path, "r"))
        if "system" in dspy_saved["prog"]:
            system_prompt = dspy_saved["prog"]["system"]
        else:
            system_prompt = dspy_saved["prog"]["signature"]["instructions"]
        message_template = [
            {"role": "system", "content": system_prompt},
        ]
        demos = dspy_saved["prog"]["demos"]
        for demo in demos:
            message_template.append(
                {
                    "role": "user",
                    "content": self.one_input_template.format(
                        question=demo["question"], fact_before_filter=demo["fact_before_filter"]
                    ),
                }
            )
            message_template.append(
                {
                    "role": "assistant",
                    "content": self.one_output_template.format(fact_after_filter=demo["fact_after_filter"]),
                }
            )
        return message_template

    def parse_filter(self, response):
        sections = [(None, [])]
        field_header_pattern = re.compile("\\[\\[ ## (\\w+) ## \\]\\]")
        for line in response.splitlines():
            match = field_header_pattern.match(line.strip())
            if match:
                sections.append((match.group(1), []))
            else:
                sections[-1][1].append(line)

        sections = [(k, "\n".join(v).strip()) for k, v in sections]
        parsed = []
        for k, value in sections:
            if k == "fact_after_filter":
                try:
                    # fields[k] = parse_value(v, signature.output_fields[k].annotation) if _parse_values else v
                    try:
                        parsed_value = json.loads(value)
                    except json.JSONDecodeError:
                        try:
                            parsed_value = ast.literal_eval(value)
                        except (ValueError, SyntaxError):
                            parsed_value = value
                    parsed = TypeAdapter(Fact).validate_python(parsed_value).fact
                except Exception as e:
                    print(f"Error parsing field {k}: {e}.\n\n\t\tOn attempting to parse the value\n```\n{value}\n```")

        return parsed

    def llm_call(self, question, fact_before_filter):
        # make prompt
        messages = deepcopy(self.message_template)
        messages.append(
            {
                "role": "user",
                "content": self.one_input_template.format(question=question, fact_before_filter=fact_before_filter),
            }
        )
        # call openai

        self.default_gen_kwargs["max_completion_tokens"] = 512

        try:
            response = self.llm_infer_fn(messages=messages, model=self.model_name, **self.default_gen_kwargs)

            if len(response) > 1:
                return response[0]
            return response
        except Exception as e:
            import logging

            logging.warn("Error in LLM call for triple filtering", str(e))
            return None

    def __call__(self, *args, **kwargs):
        return self.rerank(*args, **kwargs)

    def rerank(
        self, query: str, candidate_items: List[Tuple], candidate_indices: List[int], len_after_rerank: int = None
    ) -> Tuple[List[int], List[Tuple], dict]:
        fact_before_filter = {"fact": [list(candidate_item) for candidate_item in candidate_items]}
        try:
            # prediction = self.program(question=query, fact_before_filter=json.dumps(fact_before_filter))
            response = self.llm_call(query, json.dumps(fact_before_filter))
            generated_facts = self.parse_filter(response)
        except Exception as e:
            print("exception", e)
            generated_facts = []
        result_indices = []
        for generated_fact in generated_facts:
            closest_matched_fact = difflib.get_close_matches(
                str(generated_fact), [str(i) for i in candidate_items], n=1, cutoff=0.0
            )[0]
            try:
                result_indices.append(candidate_items.index(eval(closest_matched_fact)))
            except Exception as e:
                print("result_indices exception", e)

        sorted_candidate_indices = [candidate_indices[i] for i in result_indices]
        sorted_candidate_items = [candidate_items[i] for i in result_indices]
        return (
            sorted_candidate_indices[:len_after_rerank],
            sorted_candidate_items[:len_after_rerank],
            {"confidence": None},
        )


class GistFilter:
    """Filter for gist content using LLM-based relevance filtering."""

    def __init__(self, remem):
        self.llm_infer_fn = remem.llm.infer
        self.model_name = remem.global_config.llm_name
        self.default_gen_kwargs = {}
        self.one_input_template = """[[ ## question ## ]]\n{question}\n\n[[ ## gist_before_filter ## ]]\n{gist_before_filter}\n\nRespond with the corresponding output fields, starting with the field `[[ ## gist_after_filter ## ]]` (must be formatted as a valid Python Gist), and then ending with the marker for `[[ ## completed ## ]]`."""
        self.one_output_template = """[[ ## gist_after_filter ## ]]\n{gist_after_filter}\n\n[[ ## completed ## ]]"""
        self.message_template = self._make_gist_template()

    def _make_gist_template(self):
        """Create the template for gist filtering."""
        system_prompt = """You are a critical component of a high-stakes question-answering system used by top researchers and decision-makers worldwide. Your task is to filter gist summaries/abstracts based on their relevance to a given query, ensuring that the most crucial information is presented to these stakeholders. The query requires careful analysis and possibly multi-hop reasoning to connect different pieces of information. You must select up to 4 relevant gists from the provided candidate list that have a strong connection to the query, aiding in reasoning and providing an accurate answer. The output should be in JSON format, e.g., {"gist": ["summary1", "summary2"]}, and if no gists are relevant, return an empty list, {"gist": []}. The accuracy of your response is paramount, as it will directly impact the decisions made by these high-level stakeholders. You must only use gists from the candidate list and not generate new gists. The future of critical decision-making relies on your ability to accurately filter and present relevant information."""

        system_structure = (
            """Your input fields are:
1. `question` (str): Query for retrieval
2. `gist_before_filter` (str): Candidate gists to be filtered

Your output fields are:
1. `gist_after_filter` (Gist): Filtered gists in JSON format

All interactions will be structured in the following way, with the appropriate values filled in.

[[ ## question ## ]]
{question}

[[ ## gist_before_filter ## ]]
{gist_before_filter}

[[ ## gist_after_filter ## ]]
{gist_after_filter}        # note: the value you produce must be parsable according to the following JSON schema: {"type": "object", "properties": {"gist": {"type": "array", "description": "A list of gist summaries/abstracts that are relevant to the query", "items": {"type": "string"}, "title": "Gist"}}, "required": ["gist"], "title": "Gist"}

[[ ## completed ## ]]

In adhering to this structure, your objective is: 
        """
            + system_prompt
        )

        message_template = [
            {"role": "system", "content": system_structure},
            # Add some example demonstrations
            {
                "role": "user",
                "content": self.one_input_template.format(
                    question="What are the main features of the new product?",
                    gist_before_filter='{"gist": ["The new product includes advanced AI capabilities and machine learning algorithms.", "Weather forecast shows rain tomorrow.", "The product has a user-friendly interface with drag-and-drop functionality.", "Stock market prices fluctuated today.", "Customer reviews highlight the product\'s reliability and performance."]}',
                ),
            },
            {
                "role": "assistant",
                "content": self.one_output_template.format(
                    gist_after_filter='{"gist": ["The new product includes advanced AI capabilities and machine learning algorithms.", "The product has a user-friendly interface with drag-and-drop functionality.", "Customer reviews highlight the product\'s reliability and performance."]}'
                ),
            },
        ]
        return message_template

    def parse_gist_filter(self, response):
        """Parse gist filter response."""
        sections = [(None, [])]
        field_header_pattern = re.compile("\\[\\[ ## (\\w+) ## \\]\\]")
        for line in response.splitlines():
            match = field_header_pattern.match(line.strip())
            if match:
                sections.append((match.group(1), []))
            else:
                sections[-1][1].append(line)

        sections = [(k, "\n".join(v).strip()) for k, v in sections]
        parsed = []
        for k, value in sections:
            if k == "gist_after_filter":
                try:
                    try:
                        parsed_value = json.loads(value)
                    except json.JSONDecodeError:
                        try:
                            parsed_value = ast.literal_eval(value)
                        except (ValueError, SyntaxError):
                            parsed_value = value
                    parsed = TypeAdapter(Gist).validate_python(parsed_value).gist
                except Exception as e:
                    print(f"Error parsing field {k}: {e}.\n\n\t\tOn attempting to parse the value\n```\n{value}\n```")

        return parsed

    def llm_call(self, question, gist_before_filter):
        """Make LLM call for gist filtering."""
        messages = deepcopy(self.message_template)
        messages.append(
            {
                "role": "user",
                "content": self.one_input_template.format(question=question, gist_before_filter=gist_before_filter),
            }
        )

        self.default_gen_kwargs["max_completion_tokens"] = 512

        try:
            response = self.llm_infer_fn(messages=messages, model=self.model_name, **self.default_gen_kwargs)

            if len(response) > 1:
                return response[0]
            return response
        except Exception as e:
            import logging

            logging.warn("Error in LLM call for gist filtering", str(e))
            return None

    def rerank(
        self, query: str, candidate_items: List[str], candidate_indices: List[int], len_after_rerank: int = None
    ) -> Tuple[List[int], List[str], dict]:
        """Rerank gist candidates based on relevance to query."""
        gist_before_filter = {"gist": candidate_items}
        try:
            response = self.llm_call(query, json.dumps(gist_before_filter))
            generated_gists = self.parse_gist_filter(response)
        except Exception as e:
            print("exception", e)
            generated_gists = []

        result_indices = []
        for generated_gist in generated_gists:
            # Find closest match in candidate items
            closest_matches = difflib.get_close_matches(generated_gist, candidate_items, n=1, cutoff=0.0)
            if closest_matches:
                try:
                    result_indices.append(candidate_items.index(closest_matches[0]))
                except Exception as e:
                    print("result_indices exception", e)

        sorted_candidate_indices = [candidate_indices[i] for i in result_indices]
        sorted_candidate_items = [candidate_items[i] for i in result_indices]
        return (
            sorted_candidate_indices[:len_after_rerank],
            sorted_candidate_items[:len_after_rerank],
            {"confidence": None},
        )

    def __call__(self, *args, **kwargs):
        return self.rerank(*args, **kwargs)
