import threading
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from remem.utils.config_utils import BaseConfig
from remem.utils.logging_utils import get_logger

from .base import BaseMetric

logger = get_logger(__name__)


class QABleuScore(BaseMetric):
    """
    BLEU (Bilingual Evaluation Understudy) score evaluation for QA tasks.

    Uses Hugging Face's evaluate library to compute BLEU scores between predicted
    and gold answers. BLEU-1 is used by default but can be configured.

    Reference: https://huggingface.co/spaces/evaluate-metric/bleu
    """

    metric_name: str = "qa_bleu_score"

    def __init__(self, global_config: Optional[BaseConfig] = None, max_order: int = 1):
        """
        Initialize BLEU score evaluator.

        Args:
            global_config: Global configuration object
            max_order: Maximum n-gram order to use (default: 1 for BLEU-1)
        """
        super().__init__(global_config)
        self.max_order = max_order
        # Guard evaluate metric usage across threads
        self._bleu_lock = threading.Lock()

        # Try to import and initialize the evaluate library
        try:
            import evaluate

            self.bleu_metric = evaluate.load("bleu")
        except ImportError:
            logger.error(
                "The 'evaluate' library is not installed or not loaded correctly. Please install it with: pip install evaluate"
            )
            raise ImportError(
                "The 'evaluate' library is required for BLEU score calculation. Install with: pip install evaluate"
            )

        logger.debug(f"Initialized BLEU score evaluator with max_order={self.max_order}")

    def calculate_metric_scores(
        self,
        gold_answers: List[List[str]],
        predicted_answers: List[str],  # type: ignore[override]
        aggregation_fn: Callable = np.max,
        **kwargs,
    ) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
        """
        Calculates the BLEU score between predicted and gold answers.

        Args:
            gold_answers (List[List[str]]): List of lists containing ground truth answers.
            predicted_answers (List[str]): List of predicted answers.
            aggregation_fn (Callable): Function to aggregate scores across multiple gold answers (default: np.max).

        Returns:
            Tuple[Dict[str, float], List[Dict[str, float]]]:
                - A dictionary with the averaged BLEU score.
                - A list of dictionaries with BLEU scores for each example.
        """
        assert len(gold_answers) == len(
            predicted_answers
        ), "Length of gold answers and predicted answers should be the same."

        example_eval_results = []
        total_bleu = 0.0

        for gold_list, predicted in zip(gold_answers, predicted_answers):
            # Calculate BLEU score for each gold answer against the predicted answer
            bleu_scores = []

            for gold in gold_list:
                try:
                    # Prepare inputs for evaluate library
                    # predictions should be a list of strings
                    # references should be a list of list of strings (multiple references per prediction)
                    # Protect compute with a lock to avoid concurrency issues in underlying implementation
                    with self._bleu_lock:
                        result = self.bleu_metric.compute(
                            predictions=[predicted], references=[[gold]], max_order=self.max_order
                        )
                    score = 0.0
                    if isinstance(result, dict):
                        score = float(result.get("bleu", 0.0))
                    bleu_scores.append(score)
                except Exception as e:
                    logger.warning(f"Error computing BLEU score: {e}")
                    bleu_scores.append(0.0)

            # Aggregate scores across multiple gold answers
            if bleu_scores:
                aggregated_bleu = aggregation_fn(bleu_scores)
            else:
                aggregated_bleu = 0.0

            example_eval_results.append({f"BLEU-{self.max_order}": float(aggregated_bleu)})
            total_bleu += aggregated_bleu

        # Calculate average BLEU score
        avg_bleu = total_bleu / len(gold_answers) if gold_answers else 0.0
        pooled_eval_results = {f"BLEU-{self.max_order}": float(avg_bleu)}

        return pooled_eval_results, example_eval_results

    def calculate_corpus_bleu(self, gold_answers: List[List[str]], predicted_answers: List[str]) -> Dict[str, float]:
        """
        Calculate corpus-level BLEU score (alternative evaluation method).

        This computes a single BLEU score over the entire corpus rather than
        averaging individual sentence-level scores.

        Args:
            gold_answers (List[List[str]]): List of lists containing ground truth answers.
            predicted_answers (List[str]): List of predicted answers.

        Returns:
            Dict[str, float]: Dictionary containing the corpus BLEU score.
        """
        assert len(gold_answers) == len(
            predicted_answers
        ), "Length of gold answers and predicted answers should be the same."

        try:
            # Flatten gold answers for corpus-level evaluation
            # Take the first gold answer for each example (can be modified if needed)
            references = [[gold_list[0]] for gold_list in gold_answers if gold_list]

            # Protect compute with a lock to avoid concurrency issues in underlying implementation
            with self._bleu_lock:
                result = self.bleu_metric.compute(
                    predictions=predicted_answers, references=references, max_order=self.max_order
                )

            bleu = 0.0
            if isinstance(result, dict):
                bleu = float(result.get("bleu", 0.0))
            return {f"Corpus_BLEU-{self.max_order}": bleu}

        except Exception as e:
            logger.error(f"Error computing corpus BLEU score: {e}")
            return {f"Corpus_BLEU-{self.max_order}": 0.0}


# Convenience classes for different BLEU variants
class QABleu1Score(QABleuScore):
    """BLEU-1 score evaluator (unigram precision)"""

    metric_name: str = "qa_bleu1_score"

    def __init__(self, global_config: Optional[BaseConfig] = None):
        super().__init__(global_config, max_order=1)


class QABleu4Score(QABleuScore):
    """BLEU-4 score evaluator (standard BLEU with up to 4-grams)"""

    metric_name: str = "qa_bleu4_score"

    def __init__(self, global_config: Optional[BaseConfig] = None):
        super().__init__(global_config, max_order=4)
