from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from remem.evaluation.base import BaseMetric
from remem.utils.config_utils import BaseConfig
from remem.utils.logging_utils import get_logger

logger = get_logger(__name__)
# Model zoo for LLM-based metric
model_zoo = {
    "llama-3.1-70b-instruct": ("meta-llama/Meta-Llama-3.1-70B-Instruct", "local"),
    "llama-3.3-70b-instruct": ("meta-llama/Meta-Llama-3.3-70B-Instruct", "local"),
    "gpt-4o-mini": (
        "gpt-4o-mini-2024-07-18",
        "azure_openai",
    ),  # set it to `openai` or `azure_openai` based on the source of the model
    "gpt-4o": ("gpt-4o-2024-08-06", "azure_openai"),
}


class QALongMemEvalLLMJudge(BaseMetric):
    metric_name: str = "qa_llm_eval"

    def __init__(self, llm_client, global_config: Optional[BaseConfig] = None, verbose: bool = False):
        """
        Initialize the LLM-based QA evaluation metric.

        Args:
            global_config (Optional[ExperimentConfig]): Experiment configuration.
            verbose (bool): If True, print detailed logs.
        """
        super().__init__(global_config)
        self.verbose = verbose
        # Get the metric model short name from configuration or use default 'gpt-4o'
        self.eval_model_short_name = getattr(global_config, "qa_metric_model", "gpt-4o") if global_config else "gpt-4o"
        if self.eval_model_short_name not in model_zoo:
            raise ValueError(f"Requested metric model is not supported: {self.eval_model_short_name}")
        self.eval_model_name, self.eval_model_source = model_zoo[self.eval_model_short_name]

        assert llm_client is not None
        self.eval_client = llm_client  # OpenAI-server client

    @staticmethod
    def _get_answer_check_prompt(
        task: str, question: str, gold_answer: str, response: str, abstention: bool = False
    ) -> str:
        """
        Generate the prompt for answer checking based on the task.

        Args:
            task (str): The question type/task.
            question (str): The question text.
            gold_answer (str): The correct answer text (or rubric).
            response (str): The model's response.
            abstention (bool): Whether the question is unanswerable.

        Returns:
            The formatted prompt string.
        """
        if not abstention:
            if task in ["single-session-user", "single-session-assistant", "multi-session"]:
                template = (
                    "I will give you a question, a correct answer, and a response from a model. "
                    "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
                    "If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. "
                    "If the response only contains a subset of the information required by the answer, answer no. \n\n"
                    "Question: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
                )
                prompt = template.format(question, gold_answer, response)
            elif task == "temporal-reasoning":
                template = (
                    "I will give you a question, a correct answer, and a response from a model. "
                    "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
                    "If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. "
                    "If the response only contains a subset of the information required by the answer, answer no. "
                    "In addition, do not penalize off-by-one errors for the number of days. If the question asks for the number of days/weeks/months, etc., "
                    "and the model makes off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's response is still correct. \n\n"
                    "Question: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
                )
                prompt = template.format(question, gold_answer, response)
            elif task == "knowledge-update":
                template = (
                    "I will give you a question, a correct answer, and a response from a model. "
                    "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
                    "If the response contains some previous information along with an updated answer, the response should be considered as correct as long as the updated answer is the required answer.\n\n"
                    "Question: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
                )
                prompt = template.format(question, gold_answer, response)
            elif task == "single-session-preference":
                template = (
                    "I will give you a question, a rubric for desired personalized response, and a response from a model. "
                    "Please answer yes if the response satisfies the desired response. Otherwise, answer no. "
                    "The model does not need to reflect all the points in the rubric. The response is correct as long as it recalls and utilizes the user's personal information correctly.\n\n"
                    "Question: {}\n\nRubric: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
                )
                prompt = template.format(question, gold_answer, response)
            else:
                raise NotImplementedError(f"Task {task} is not implemented for LLM evaluation.")
        else:
            template = (
                "I will give you an unanswerable question, an explanation, and a response from a model. "
                "Please answer yes if the model correctly identifies the question as unanswerable. The model could say that the information is incomplete, "
                "or some other information is given but the asked information is not.\n\n"
                "Question: {}\n\nExplanation: {}\n\nModel Response: {}\n\nDoes the model correctly identify the question as unanswerable? Answer yes or no only."
            )
            prompt = template.format(question, gold_answer, response)
        return prompt

    def calculate_metric_scores(
        self, gold_answers: List[List[str]], predicted_answers: List[str], aggregation_fn: Callable = np.max, **kwargs
    ) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
        """
        Calculate the evaluation scores for the given gold and predicted answers.

        Args:
            gold_answers (List[List[str]]): List of gold answers for each sample, each sample can have multiple gold answers.
            predicted_answers (List[str]): List of predicted answers for each sample, each sample has only one predicted answer.
            aggregation_fn (Callable): Function to aggregate scores. Default is np.max.
            **kwargs: Additional keyword arguments for evaluation.
        """

        if len(gold_answers) != len(predicted_answers):
            raise ValueError("The number of gold answers and predicted answers should be the same.")

        example_eval_results = []
        correct_count = 0
        total_count = 0

        for sample_idx in range(len(gold_answers)):
            question = kwargs.get("questions", [])[sample_idx]
            predicted_answer = predicted_answers[sample_idx]
            qtype = kwargs.get("question_types", [])[sample_idx]
            # Determine if this is an abstention case based on question_id
            abstention = "_abs" in qtype

            # Generate the prompt for evaluation
            prompt = self._get_answer_check_prompt(
                task=qtype,
                question=question,
                gold_answer=gold_answers[sample_idx],
                response=predicted_answer,
                abstention=abstention,
            )

            # Prepare API call parameters
            kwargs = {
                "model": self.eval_model_name,
                "messages": [{"role": "user", "content": prompt}],
                "n": 1,
                "temperature": 0,
                "max_completion_tokens": 10,
            }
            # Call the LLM API with backoff
            response, metadata, cache_hit = self.eval_client.infer(**kwargs)
            label = 0.0
            if response and "yes" in response.lower():
                label = 1.0

            example_eval_results.append({"LLMEvalAccuracy": label})
            correct_count += label
            total_count += 1

            if self.verbose:
                import json

                print(
                    json.dumps(
                        {"question": question, "answer": predicted_answer, "autoeval_label": bool(label)}, indent=4
                    ),
                    flush=True,
                )

        avg_accuracy = correct_count / total_count if total_count > 0 else 0.0
        pooled_eval_results = {"LLMEvalAccuracy": avg_accuracy}

        return pooled_eval_results, example_eval_results
