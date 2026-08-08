from typing import Callable, Counter, Dict, List, Optional, Tuple

import numpy as np

from remem.evaluation.base import BaseMetric
from remem.utils.config_utils import BaseConfig


class QAF1ScoreLocomo(BaseMetric):
    metric_name: str = "qa_f1_score_locomo"

    def __init__(self, global_config: Optional[BaseConfig] = None):
        super().__init__(global_config)
        from nltk import PorterStemmer

        self.ps = PorterStemmer()

    def calculate_metric_scores(
        self, gold_answers: List[List[str]], predicted_answers: List[str], aggregation_fn: Callable = np.max, **kwargs
    ) -> Tuple[Dict[str, float], List[Dict[str, float]]]:

        def normalize_answer(s):
            def remove_articles(text):
                import regex

                return regex.sub(r"\b(a|an|the|and)\b", " ", text)

            def white_space_fix(text):
                return " ".join(text.split())

            def remove_punc(text):
                import string

                exclude = set(string.punctuation)
                return "".join(ch for ch in text if ch not in exclude)

            def lower(text):
                return text.lower()

            return white_space_fix(remove_articles(remove_punc(lower(s))))

        def compute_f1(prediction: str, ground_truth: str) -> float:
            prediction_tokens = [self.ps.stem(w) for w in normalize_answer(prediction).split()]
            ground_truth_tokens = [self.ps.stem(w) for w in normalize_answer(ground_truth).split()]
            common = Counter[str](prediction_tokens) & Counter[str](ground_truth_tokens)
            num_same = sum(common.values())

            if num_same == 0:
                return 0.0

            precision = num_same / len(prediction_tokens)
            recall = num_same / len(ground_truth_tokens)
            return 2 * precision * recall / (precision + recall)

        example_eval_results = []
        total_f1 = 0.0

        assert (
            len(gold_answers) == len(predicted_answers) == len(kwargs.get("question_types", []))
        ), "gold_answers, predicted_answers, and question_types must have the same length"

        for gold_list, predicted, category in zip(gold_answers, predicted_answers, kwargs.get("question_types")):
            # Handle potential multiple comma-separated answers
            predicted_parts = [p.strip() for p in predicted.split(",")]
            f1_scores = []

            if category in [2, 3, 4]:  # single-hop, temporal, open-domain eval without splitting for sub-answers
                f1_scores.append(compute_f1(predicted_parts[0], gold_list[0]))
            elif category in [
                1
            ]:  # multi-hop eval by splitting entire phrase into sub-answers and computing partial F1 for each
                assert len(gold_list)
                for gold in gold_list:
                    gold_parts = [g.strip() for g in gold.split(",")]
                    scores = [
                        max(compute_f1(pred_part, gold_part) for pred_part in predicted_parts)
                        for gold_part in gold_parts
                    ]
                    f1_scores.append(np.mean(scores))
            elif category in [5]:  # adversarial eval --> check for selection of correct option
                if "no information available" in predicted.lower() or "not mentioned" in predicted.lower():
                    f1_scores.append(1)
                else:
                    f1_scores.append(0)
            else:
                raise ValueError(f"Unknown question type {category}")

            aggregated_f1 = aggregation_fn(f1_scores)
            aggregated_f1 = float(round(aggregated_f1, 4))
            example_eval_results.append({"F1": aggregated_f1})
            total_f1 += aggregated_f1

        avg_f1 = total_f1 / len(gold_answers) if gold_answers else 0.0
        pooled_eval_results = {"F1": avg_f1}

        return pooled_eval_results, example_eval_results
