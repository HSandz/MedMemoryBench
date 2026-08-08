from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from remem.evaluation.base import BaseMetric
from remem.llm import CacheOpenAI
from remem.utils.config_utils import BaseConfig
from remem.utils.logging_utils import get_logger

logger = get_logger(__name__)

ACCURACY_PROMPT = """
Your task is to label an answer to a question as ’CORRECT’ or ’WRONG’. You will be given the following data:
    (1) a question (posed by one user to another user), 
    (2) a ’gold’ (ground truth) answer, 
    (3) a generated answer
which you will score as CORRECT/WRONG.

The point of the question is to ask about something one user should know about the other user based on their prior conversations.
The gold answer will usually be a concise and short answer that includes the referenced topic, for example:
Question: Do you remember what I got the last time I went to Hawaii?
Gold answer: A shell necklace
The generated answer might be much longer, but you should be generous with your grading - as long as it touches on the same topic as the gold answer, it should be counted as CORRECT. 

For time related questions, the gold answer will be a specific date, month, year, etc. The generated answer might be much longer or use relative time references (like "last Tuesday" or "next month"), but you should be generous with your grading - as long as it refers to the same date or time period as the gold answer, it should be counted as CORRECT. Even if the format differs (e.g., "May 7th" vs "7 May"), consider it CORRECT if it's the same date.

Now it’s time for the real question:
Question: {question}
Gold answer: {gold_answer}
Generated answer: {generated_answer}

First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG. 
Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation script.

Just return the label CORRECT or WRONG in a json format with the key as "label".
"""


class QAMem0LLMJudge(BaseMetric):
    metric_name: str = "qa_mem0_llm_judge"

    def __init__(self, client, global_config: Optional[BaseConfig] = None):
        # Initialize LLM client
        self.client = client
        self.config = global_config

    @staticmethod
    def _chat_completions(client: CacheOpenAI, **kwargs):
        try:
            return client.infer(**kwargs)
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return None

    def calculate_metric_scores(
        self, gold_answers: List[List[str]], predicted_answers: List[str], aggregation_fn: Callable = np.max, **kwargs
    ) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
        questions = kwargs.get("questions", [])
        assert (
            len(questions) == len(predicted_answers) == len(gold_answers)
        ), f"Length of questions ({len(questions)}), predicted answers({len(predicted_answers)}), and gold answers ({len(gold_answers)}) must match."

        all_eval_messages = []
        request_metadata = []  # Track which example and gold answer each request corresponds to

        for example_idx, (gold_list, pred, question) in enumerate(zip(gold_answers, predicted_answers, questions)):
            # Create evaluation messages for each gold answer
            for gold_idx, gold in enumerate(gold_list):
                prompt = ACCURACY_PROMPT.format(question=question, gold_answer=gold, generated_answer=pred)
                messages = [{"role": "user", "content": prompt}]
                all_eval_messages.append(messages)
                request_metadata.append((example_idx, gold_idx))

        responses = self.client.batch_infer(all_eval_messages)

        # Initialize scores for all examples
        example_scores = [[] for _ in range(len(gold_answers))]

        # Process responses and match them back to the correct examples
        for request_idx, (example_idx, gold_idx) in enumerate(request_metadata):
            label = 0
            if request_idx < len(responses) and responses[request_idx]:
                label = 1 if '{"label": "CORRECT"}' in responses[request_idx][0] else 0
            else:
                logger.warning(f"No response received for example {example_idx}, gold {gold_idx}")

            example_scores[example_idx].append(float(label))

        # Aggregate scores for each example
        example_eval_results = []
        for example_idx, scores in enumerate(example_scores):
            if scores:
                aggregated_score = aggregation_fn(scores)
            else:
                logger.warning(f"No valid scores for example {example_idx}, using 0.0")
                aggregated_score = 0.0
            example_eval_results.append({"LLMEvalAccuracy": aggregated_score})

        # average scores
        pooled_eval_results = {
            "LLMEvalAccuracy": float(np.mean([result["LLMEvalAccuracy"] for result in example_eval_results]))
        }
        return pooled_eval_results, example_eval_results
