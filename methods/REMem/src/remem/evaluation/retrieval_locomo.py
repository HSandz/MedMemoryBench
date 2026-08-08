import json
import unicodedata
from typing import Dict, List, Optional, Tuple

import regex

from remem.evaluation.base import BaseMetric
from remem.utils.config_utils import BaseConfig


class SimpleTokenizer(object):
    ALPHA_NUM = r"[\p{L}\p{N}\p{M}]+"
    NON_WS = r"[^\p{Z}\p{C}]"

    def __init__(self):
        """
        Args:
            annotators: None or empty set (only tokenizes).
        """
        self._regexp = regex.compile(
            "(%s)|(%s)" % (self.ALPHA_NUM, self.NON_WS), flags=regex.IGNORECASE + regex.UNICODE + regex.MULTILINE
        )

    def tokenize(self, text, uncased=False):
        matches = [m for m in self._regexp.finditer(text)]
        if uncased:
            tokens = [m.group().lower() for m in matches]
        else:
            tokens = [m.group() for m in matches]
        return tokens


class RetrievalRecallLocomo(BaseMetric):
    metric_name: str = "retrieval_recall_locomo"

    def __init__(self, global_config: Optional[BaseConfig] = None):
        super().__init__(global_config)
        self.tokenizer = SimpleTokenizer()

    def calculate_metric_scores(
        self, gold_docs: List[List[str]], retrieved_chunks: List[List[str]], k_list: List[int] = [1, 5, 10, 20]
    ) -> Tuple[Dict[str, float], List[Dict[str, float]]]:

        def _normalize(text):
            return unicodedata.normalize("NFD", text)

        def has_answer(gold_answers, text) -> bool:
            """Check if a document contains *any* answer string (token-level, case-insensitive)."""
            text_tokens = self.tokenizer.tokenize(_normalize(text), uncased=True)

            for answer in gold_answers:
                ans_tokens = self.tokenizer.tokenize(_normalize(answer), uncased=True)
                if len(ans_tokens) == 0:
                    continue
                for i in range(len(text_tokens) - len(ans_tokens) + 1):
                    if ans_tokens == text_tokens[i : i + len(ans_tokens)]:
                        return True
            return False

        example_eval_results = []
        pooled_eval_results = {f"Recall@{k}": 0.0 for k in k_list}

        for gold_list, retrieved_list in zip(gold_docs, retrieved_chunks):
            example_result = {f"Recall@{k}": 0.0 for k in k_list}

            for k in k_list:
                top_k_chunks = retrieved_list[:k]
                gold_answers = [json.loads(gold)["messages"] for gold in gold_list]
                gold_answer_str = [item[0]["content"] for item in gold_answers]
                hits = [has_answer(gold_answer_str, chunk) for chunk in top_k_chunks]
                recall_at_k = 1.0 if any(hits) else 0.0
                example_result[f"Recall@{k}"] = recall_at_k
                pooled_eval_results[f"Recall@{k}"] += recall_at_k

            example_eval_results.append(example_result)

        num_examples = len(gold_docs)
        for k in k_list:
            pooled_eval_results[f"Recall@{k}"] = round(pooled_eval_results[f"Recall@{k}"] / num_examples, 4)

        return pooled_eval_results, example_eval_results
