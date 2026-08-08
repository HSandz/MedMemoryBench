import json
from typing import Dict, List, Optional, Tuple

import numpy as np

from remem.utils.config_utils import BaseConfig
from remem.utils.logging_utils import get_logger

from .base import BaseMetric

logger = get_logger(__name__)


def is_chunk_from_original(chunk: str, original: str) -> bool:
    # Check if the chunk starts with `Date: `, if so, skip the first line
    if chunk.startswith("Date: "):
        chunk = "\n".join(chunk.split("\n")[1:])
    if chunk.startswith("user: ") or chunk.startswith("assistant: "):
        return chunk.split(": ")[1] in original or json.dumps(chunk).split(": ")[1] in original
    return chunk in original or json.dumps(chunk) in original


class RetrievalRecall(BaseMetric):
    metric_name: str = "retrieval_recall"

    def __init__(self, global_config: Optional[BaseConfig] = None):
        super().__init__(global_config)

    def calculate_metric_scores(
        self, gold_docs: List[List[str]], retrieved_chunks: List[List[str]], k_list: List[int] = [1, 5, 10, 20]
    ) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
        """
        Calculates Recall@k for each example and pools results for all queries.

        Args:
            gold_docs (List[List[str]]): List of lists containing the ground truth (relevant documents) for each query.
            retrieved_chunks (List[List[str]]): List of lists containing the retrieved documents for each query.
            k_list (List[int]): List of k values to calculate Recall@k for.

        Returns:
            Tuple[Dict[str, float], List[Dict[str, float]]]:
                - A pooled dictionary with the averaged Recall@k across all examples.
                - A list of dictionaries with Recall@k for each example.
        """
        k_list = sorted(set(k_list))

        example_eval_results = []
        pooled_eval_results = {f"Recall@{k}": 0.0 for k in k_list}
        for example_gold_docs, example_retrieved_docs in zip(gold_docs, retrieved_chunks):
            if len(example_retrieved_docs) < k_list[-1]:
                logger.warning(
                    f"Length of retrieved docs ({len(example_retrieved_docs)}) is smaller than largest topk for recall score ({k_list[-1]})"
                )

            example_eval_result = {f"Recall@{k}": 0.0 for k in k_list}

            # Compute Recall@k for each k
            for k in k_list:
                # Get top-k retrieved documents
                top_k_docs = example_retrieved_docs[:k]
                # Calculate intersection with gold documents
                relevant_retrieved = set(top_k_docs) & set(example_gold_docs)
                # Compute recall
                if example_gold_docs:  # Avoid division by zero
                    example_eval_result[f"Recall@{k}"] = len(relevant_retrieved) / len(set(example_gold_docs))
                else:
                    example_eval_result[f"Recall@{k}"] = 0.0

            # Append example results
            example_eval_results.append(example_eval_result)

            # Accumulate pooled results
            for k in k_list:
                pooled_eval_results[f"Recall@{k}"] += example_eval_result[f"Recall@{k}"]

        # Average pooled results over all examples
        num_examples = len(gold_docs)
        for k in k_list:
            pooled_eval_results[f"Recall@{k}"] /= num_examples

        # round off to 4 decimal places for pooled results
        pooled_eval_results = {k: round(v, 4) for k, v in pooled_eval_results.items()}
        return pooled_eval_results, example_eval_results


class RetrievalRecallAll(BaseMetric):
    metric_name: str = "retrieval_recall_all"

    def __init__(self, global_config: Optional[BaseConfig] = None):
        super().__init__(global_config)

    def calculate_metric_scores(
        self, gold_docs: List[List[str]], retrieved_chunks: List[List[str]], k_list: List[int] = [1, 5, 10, 20]
    ) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
        """
        Calculates Recall@k for each example and pools results for all queries.

        Args:
            gold_docs (List[List[str]]): List of lists containing the ground truth (relevant documents) for each query.
            retrieved_chunks (List[List[str]]): List of lists containing the retrieved documents for each query.
            k_list (List[int]): List of k values to calculate Recall@k for.

        Returns:
            Tuple[Dict[str, float], List[Dict[str, float]]]:
                - A pooled dictionary with the averaged Recall@k across all examples.
                - A list of dictionaries with Recall@k for each example.
        """
        k_list = sorted(set(k_list))

        example_eval_results = []
        pooled_eval_results = {f"Recall_all@{k}": 0.0 for k in k_list}
        for example_gold_docs, example_retrieved_chunks in zip(gold_docs, retrieved_chunks):
            if len(example_retrieved_chunks) < k_list[-1]:
                logger.warning(
                    f"Length of retrieved docs ({len(example_retrieved_chunks)}) is smaller than largest topk for recall score ({k_list[-1]})"
                )

            example_eval_result = {f"Recall_all@{k}": 0.0 for k in k_list}

            # Compute Recall@k for each k
            for k in k_list:
                # Get top-k retrieved documents
                top_k_chunks = example_retrieved_chunks[:k]

                # Check how many top-k chunks are originated from the original text
                relevant_retrieved = set()
                for gold_doc in example_gold_docs:
                    for chunk in top_k_chunks:
                        if is_chunk_from_original(chunk, gold_doc):
                            relevant_retrieved.add(gold_doc)
                            break

                # Compute recall
                if example_gold_docs:  # Avoid division by zero
                    example_eval_result[f"Recall_all@{k}"] = (
                        1.0 if len(relevant_retrieved) == len(example_gold_docs) else 0.0
                    )
                else:
                    example_eval_result[f"Recall_all@{k}"] = 0.0

            # Append example results
            example_eval_results.append(example_eval_result)

            # Accumulate pooled results
            for k in k_list:
                pooled_eval_results[f"Recall_all@{k}"] += example_eval_result[f"Recall_all@{k}"]

        # Average pooled results over all examples
        num_examples = len(gold_docs)
        for k in k_list:
            pooled_eval_results[f"Recall_all@{k}"] /= num_examples

        # round off to 4 decimal places for pooled results
        pooled_eval_results = {k: round(v, 4) for k, v in pooled_eval_results.items()}

        # assert that values in pool_eval_results and example_eval_results are in non-decreasing order
        for i in range(1, len(k_list)):
            prev_key = f"Recall_all@{k_list[i - 1]}"
            curr_key = f"Recall_all@{k_list[i]}"
            assert (
                pooled_eval_results[prev_key] <= pooled_eval_results[curr_key]
            ), f"Pooled result for {prev_key} is greater than {curr_key}"
        for result in example_eval_results:
            for i in range(1, len(k_list)):
                prev_key = f"Recall_all@{k_list[i - 1]}"
                curr_key = f"Recall_all@{k_list[i]}"
                assert result[prev_key] <= result[curr_key], f"Example result for {prev_key} is greater than {curr_key}"

        return pooled_eval_results, example_eval_results


class RetrievalNDCGAny(BaseMetric):
    metric_name: str = "retrieval_ndcg_any"

    def __init__(self, global_config: Optional[BaseConfig] = None):
        """
        Initialize the RetrievalNDCGAny metric.

        Args:
            global_config (Optional[BaseConfig]): Global configuration.
            k (int): Default cutoff rank for evaluation (not used directly if k_list is provided in calculate_metric_scores).
        """
        super().__init__(global_config)

    def calculate_metric_scores(
        self,
        gold_docs: List[List[str]],
        retrieved_chunks: List[List[str]],
        k_list: List[int] = [1, 5, 10, 20],
    ) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
        """
        Calculates NDCG@k for each example and pools results for all queries.

        Args:
            gold_docs (List[List[str]]): List of lists containing the ground truth (relevant document identifiers) for each query.
            retrieved_chunks (List[List[str]]): List of lists containing the retrieved document identifiers for each query, assumed to be in ranked order.
            k_list (List[int]): List of cutoff ranks (k values) to calculate NDCG@k for.

        Returns:
            Tuple[Dict[str, float], List[Dict[str, float]]]:
                - A pooled dictionary with the averaged NDCG@k across all examples.
                - A list of dictionaries with NDCG@k for each example.
        """

        # Ensure unique and sorted k values
        k_list = sorted(set(k_list))
        pooled_eval_results = {f"NDCG_any@{k}": 0.0 for k in k_list}
        example_eval_results = []

        # Define helper functions for DCG and NDCG calculation
        def dcg(relevances, k):
            """Discounted Cumulative Gain at k."""
            relevances = np.asarray(relevances, dtype=float)[:k]
            if relevances.size:
                return relevances[0] + np.sum(relevances[1:] / np.log2(np.arange(2, relevances.size + 1)))
            return 0.0

        def ndcg(rankings, correct_docs, candidate_chunks, k=10):
            """Normalized Discounted Cumulative Gain at k."""
            relevances = []
            for chunk in candidate_chunks:
                # 1 if it is_chunk_from_original any correct_docs, 0 otherwise
                relevances.append(1 if any(is_chunk_from_original(chunk, doc) for doc in correct_docs) else 0)
            sorted_relevances = [relevances[idx] for idx in rankings[:k]]
            ideal_relevance = sorted(relevances, reverse=True)
            ideal_dcg = dcg(ideal_relevance, k)
            actual_dcg = dcg(sorted_relevances, k)
            if ideal_dcg == 0:
                return 0.0
            return actual_dcg / ideal_dcg

        def evaluate_retrieval(rankings, correct_docs, candidate_chunks, k=10):
            """
            Evaluate retrieval performance by computing recall_any, recall_all, and ndcg_score.
            Here, we only use the ndcg_score.
            """
            # Determine the set of documents retrieved in the top k
            recalled_docs = set(candidate_chunks[idx] for idx in rankings[:k])
            top_k_chunks = [candidate_chunks[idx] for idx in rankings[:k]]
            recall_any = (
                1.0
                if any(is_chunk_from_original(chunk, gold_doc) for gold_doc in correct_docs for chunk in top_k_chunks)
                else 0.0
            )
            recall_all = (
                1.0
                if all(
                    any(is_chunk_from_original(chunk, gold_doc) for chunk in top_k_chunks) for gold_doc in correct_docs
                )
                else 0.0
            )
            ndcg_score = ndcg(rankings, correct_docs, candidate_chunks, k)
            return recall_any, recall_all, ndcg_score

        # Process each example query
        for example_gold_docs, example_retrieved_chunks in zip(gold_docs, retrieved_chunks):
            if len(example_retrieved_chunks) < k_list[-1]:
                logger.warning(
                    f"Length of retrieved docs ({len(example_retrieved_chunks)}) is smaller than largest topk for ndcg score ({k_list[-1]})"
                )

            # Assume that the ranking is the order of the retrieved chunks: 0, 1, 2, ...
            rankings = list(range(len(example_retrieved_chunks)))
            example_result = {f"NDCG_any@{k}": 0.0 for k in k_list}

            # Compute NDCG@k for each cutoff k
            for k in k_list:
                # Use evaluate_retrieval to compute ndcg_score (ignoring recall values)
                _, _, ndcg_score = evaluate_retrieval(rankings, example_gold_docs, example_retrieved_chunks, k)
                example_result[f"NDCG_any@{k}"] = ndcg_score

            example_eval_results.append(example_result)

            # Accumulate pooled results
            for k in k_list:
                pooled_eval_results[f"NDCG_any@{k}"] += example_result[f"NDCG_any@{k}"]

        # Average pooled results over all examples
        num_examples = len(gold_docs)
        for k in k_list:
            pooled_eval_results[f"NDCG_any@{k}"] /= num_examples

        # Round off pooled results to 4 decimal places
        pooled_eval_results = {key: round(val, 4) for key, val in pooled_eval_results.items()}

        return pooled_eval_results, example_eval_results
