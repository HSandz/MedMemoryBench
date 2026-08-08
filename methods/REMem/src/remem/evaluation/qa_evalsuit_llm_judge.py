from typing import Callable, List, Optional

import numpy as np

from remem.evaluation.base import BaseMetric
from remem.llm import CacheOpenAI
from remem.utils.config_utils import BaseConfig
from remem.utils.logging_utils import get_logger

logger = get_logger(__name__)

# System + Human messages prompting the model to return a single character judgement: '1' (correct) or '0' (incorrect)
SYSTEM_PROMPT = (
    "You are an impartial judge evaluating whether a model’s "
    "predicted answer exactly or sufficiently matches the "
    "ground-truth answer.\n\n"
    "Return only a single character: '1' when the prediction "
    "should be considered correct, and '0' when it is "
    "incorrect. Do not add any additional commentary."
)


class QAEvalSuitLLMJudge(BaseMetric):
    metric_name: str = "qa_evalsuit_llm_judge"

    def __init__(self, client: CacheOpenAI, global_config: Optional[BaseConfig] = None):
        self.client = client
        self.config = global_config

    @staticmethod
    def _chat_completions(client: CacheOpenAI, **kwargs):
        try:
            return client.infer(**kwargs)
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return None

    def calculate_metric_scores(  # type: ignore[override]
        self,
        gold_answers: List[List[str]],
        predicted_answers: List[str],
        aggregation_fn: Callable = np.max,
        **kwargs,
    ):
        questions = kwargs.get("questions", [])
        assert len(questions) == len(predicted_answers) == len(gold_answers), (
            f"Length of questions ({len(questions)}), predicted answers({len(predicted_answers)}), "
            f"and gold answers ({len(gold_answers)}) must match."
        )

        all_eval_messages = []
        request_metadata = []

        # Build chat messages per gold answer
        for example_idx, (gold_list, pred, _question) in enumerate(zip(gold_answers, predicted_answers, questions)):
            for gold_idx, gold in enumerate(gold_list):
                user_prompt = (
                    f"<ground_truth>\n{gold}\n</ground_truth>\n\n"
                    f"<prediction>\n{pred}\n</prediction>\n\n"
                    f"Judgement:"
                )
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ]
                all_eval_messages.append(messages)
                request_metadata.append((example_idx, gold_idx))

        responses = self.client.batch_infer(all_eval_messages)

        # Initialize scores for each example
        example_scores = [[] for _ in range(len(gold_answers))]

        # Parse responses back to examples
        for request_idx, (example_idx, gold_idx) in enumerate(request_metadata):
            label = 0.0
            try:
                if request_idx < len(responses) and responses[request_idx]:
                    raw_text = responses[request_idx][0]
                    # Robustly parse a single-character judgement; default to '0' if ambiguous
                    if isinstance(raw_text, str):
                        cleaned = raw_text.strip().strip("\n\r ")
                        # Some models may echo or add punctuation; take the first digit we see
                        first_digit = next((ch for ch in cleaned if ch in {"0", "1"}), None)
                        label = 1.0 if first_digit == "1" else 0.0
                    else:
                        logger.warning(
                            f"Unexpected response format for example {example_idx}, gold {gold_idx}: {type(raw_text)}"
                        )
                else:
                    logger.warning(f"No response received for example {example_idx}, gold {gold_idx}")
            except Exception as e:
                logger.warning(f"Failed to parse judgement for example {example_idx}, gold {gold_idx}: {e}")
                label = 0.0

            example_scores[example_idx].append(float(label))

        # Aggregate scores per example
        example_eval_results = []
        for example_idx, scores in enumerate(example_scores):
            if scores:
                aggregated_score = aggregation_fn(scores)
            else:
                logger.warning(f"No valid scores for example {example_idx}, using 0.0")
                aggregated_score = 0.0
            example_eval_results.append({"LLMEvalAccuracy": float(aggregated_score)})

        # Pooled average
        pooled_eval_results = {
            "LLMEvalAccuracy": float(np.mean([result["LLMEvalAccuracy"] for result in example_eval_results]))
        }
        return pooled_eval_results, example_eval_results
