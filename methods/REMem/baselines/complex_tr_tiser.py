#!/usr/bin/env python3
"""
Script to evaluate LLMs on the Complex Temporal Reasoning dataset using the TISER prompt template with dense retrieval.
This script uses dense retrieval to find relevant temporal facts from the corpus and uses
a Chain of Thought (CoT) approach with reflection and timeline analysis.
Features multi-threading for faster processing while maintaining result order and real-time accuracy tracking.
"""

import argparse
import json
import logging
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Tuple

# Add the src directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
src_dir = os.path.join(parent_dir, "src")
sys.path.insert(0, src_dir)
sys.path.insert(0, parent_dir)  # Add parent directory for examples module

import threading
from collections import defaultdict

from tqdm import tqdm

from examples.complex_tr import extract_fact_context_from_sample, load_corpus_facts
from remem.llm.openai_gpt import CacheOpenAI
from remem.remem import ReMem
from remem.utils.config_utils import BaseConfig
from remem.utils.misc_utils import QuerySolution, safe_dump_json, safe_serialize_query_solutions


class ThreadSafeMetrics:
    """Thread-safe metrics tracking for parallel processing"""

    def __init__(self):
        self.lock = threading.Lock()
        self.cumulative_metrics = defaultdict(float)
        self.num_samples = 0
        self.query_solutions = []

    def update(self, metrics_dict, query_solution):
        with self.lock:
            for key, value in metrics_dict.items():
                self.cumulative_metrics[key] += value
            self.num_samples += 1
            self.query_solutions.append(query_solution)

    def get_current_averages(self):
        with self.lock:
            if self.num_samples == 0:
                return {}, 0
            return {
                key: round(value / self.num_samples, 4) for key, value in self.cumulative_metrics.items()
            }, self.num_samples

    def get_final_results(self):
        with self.lock:
            final_metrics = {key: round(value / self.num_samples, 4) for key, value in self.cumulative_metrics.items()}
            return final_metrics, self.query_solutions.copy(), self.num_samples


def load_complex_tr_dataset(dataset_path: str) -> List[Dict[str, Any]]:
    """Load the Complex TR dataset."""
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    with open(dataset_path, "r") as f:
        samples = json.load(f)

    logging.info(f"Loaded {len(samples)} samples from Complex TR dataset")
    return samples


def evaluate_sample_tiser(
    llm_client: CacheOpenAI,
    sample: Dict[str, Any],
    sample_idx: int,
    corpus_facts: List[str],
    rag: ReMem,
    retrieval_top_k: int = 10,
    selected_metrics: Tuple[str, ...] = ("qa_em", "qa_f1", "qa_mem0_llm_judge"),
) -> Tuple[int, Dict[str, Any], QuerySolution]:
    """
    Evaluate a single sample using the LLM with TISER prompt template and dense retrieval.

    Args:
        llm_client: The LLM model to use for evaluation
        sample: The sample containing question, answers, and fact_context
        sample_idx: The index of the sample in the dataset
        corpus_facts: List of all corpus facts for retrieval
        rag: ReMem instance for dense retrieval
        retrieval_top_k: Number of top facts to retrieve
        selected_metrics: Metrics to evaluate

    Returns:
        Tuple of (sample_idx, metrics_dict, query_solution)
    """
    try:
        question = sample["question"]
        gold_answer = sample["answers"]
        gold_answer = (
            ["; ".join(gold_answer)] if isinstance(gold_answer, list) and len(gold_answer) > 1 else gold_answer
        )
        gold_answers = [gold_answer]  # Wrap in list as expected by evaluators
        sample_id = sample.get("id", f"sample_{sample_idx}")

        # Extract gold documents (facts mentioned in fact_context)
        gold_facts = extract_fact_context_from_sample(sample)
        gold_docs = [gold_facts]  # Wrap in list for this question

        question_metadata = {"id": sample_id, "sample_idx": sample_idx}

        # Perform dense passage retrieval to get top relevant facts
        sorted_chunk_ids, sorted_chunk_scores = rag.dense_passage_retrieval(query=question, candidate_docs=corpus_facts)

        # Get top k retrieved facts
        top_chunk_ids = sorted_chunk_ids[:retrieval_top_k]
        try:
            retrieved_facts = [rag.chunk_contents[chunk_id] for chunk_id in top_chunk_ids]
        except IndexError as e:
            logging.error(f"IndexError while retrieving facts for sample {sample_idx}: {e}")
            logging.error(f"#chunk_contents: {len(rag.chunk_contents)}, top_chunk_ids: {top_chunk_ids}")
            logging.error(f"#chunk_embedding_store.embeddings: {len(rag.chunk_embedding_store.embeddings)}")
            exit(1)

        # Use retrieved facts as temporal context instead of original fact_context
        temporal_context = "\n".join(retrieved_facts)

        # Create the TISER prompt
        from baselines.tiser import create_tiser_messages

        tiser_messages = create_tiser_messages(question, temporal_context)

        # Get response from LLM
        response, metadata, cache_hit = llm_client.infer(messages=tiser_messages)

        # Extract answer from the response
        predicted_answer = ""
        answer_match = re.search(r"<answer>\s*(.+?)\s*</answer>", response, re.IGNORECASE | re.DOTALL)
        if answer_match:
            predicted_answer = answer_match.group(1).strip()
        else:
            # Fallback: use the entire response
            predicted_answer = response.strip()

        # Create query solution with TISER response as the answer
        # For evaluation, we need to create a QuerySolution that includes the retrieved facts and predicted answer
        retrieved_chunk_contents = retrieved_facts
        retrieved_chunk_metadata = [
            {"chunk_id": int(chunk_id), "content": corpus_facts[int(chunk_id)], "score": sorted_chunk_scores[i]}
            for i, chunk_id in enumerate(top_chunk_ids)
        ]

        query_solution = QuerySolution(
            question,
            retrieved_chunk_contents,
            None,  # No named_entity_dict for TISER
            sorted_chunk_scores[:retrieval_top_k],
            retrieved_chunk_metadata,
            None,  # No query_paths
            gold_answers,
            gold_docs,
            None,  # No query_graph
            {},
            None,  # No processed_query
        )

        query_solution.answer = predicted_answer
        query_solution.raw_response = response

        qa_evaluators, retrieval_evaluators = rag.get_evaluators(gold_answers, gold_docs, selected_metrics)

        query_solutions = [query_solution]

        # Evaluate results
        retrieval_results = rag.evaluate_retrieval(gold_docs, query_solutions, retrieval_evaluators)
        qa_results = rag.evaluate_qa(gold_answers, qa_evaluators, query_solutions, [question_metadata])

        # Combine all metrics
        all_metrics = {**retrieval_results, **qa_results}

        return sample_idx, all_metrics, query_solution

    except Exception as e:
        logging.exception(f"Error evaluating sample {sample_idx}: {str(e)}")
        return sample_idx, {}, None


def evaluate_sample_with_index(args_tuple):
    """
    Wrapper for evaluate_sample_tiser that includes the index for ordered results.

    Args:
        args_tuple: Tuple of (index, llm_model, sample, corpus_facts, rag, retrieval_top_k, selected_metrics)

    Returns:
        Tuple of (index, metrics_dict, query_solution)
    """
    index, llm_model, sample, corpus_facts, rag, retrieval_top_k, selected_metrics = args_tuple
    sample_idx, metrics_dict, query_solution = evaluate_sample_tiser(
        llm_model, sample, index, corpus_facts, rag, retrieval_top_k, selected_metrics
    )
    return sample_idx, metrics_dict, query_solution


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate LLMs on Complex TR Dataset using TISER prompt template with dense retrieval"
    )
    parser.add_argument("--llm_base_url", type=str, default="https://api.openai.com/v1", help="LLM base URL")
    parser.add_argument("--llm_name", type=str, default="gpt-4.1-mini", help="LLM name")
    parser.add_argument("--embedding_name", type=str, default="nvidia/NV-Embed-v2", help="embedding model name")
    parser.add_argument("--use_azure", action="store_true", help="Use Azure for OpenAI")
    parser.add_argument("--debug", action="store_true", help="Debug mode - process only 1 sample")
    parser.add_argument(
        "--cache_dir", type=str, default="outputs/complex_tr/llm_cache_tiser", help="Cache directory for LLM responses"
    )
    parser.add_argument("--num_threads", type=int, default=4, help="Number of threads for parallel processing")
    parser.add_argument("--num_samples", type=int, default=None, help="Limit number of samples to process")
    parser.add_argument("--indices", type=str, default=None, help="JSON file with sample indices to process")
    parser.add_argument(
        "--retrieval_top_k", type=int, default=10, help="Number of top facts to retrieve for temporal context"
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    logging.info(f"Arguments: {args}")
    logging.info(
        "Using TISER prompt template with Chain of Thought (CoT) approach and dense retrieval for Complex TR dataset"
    )

    # Create cache directory
    os.makedirs(args.cache_dir, exist_ok=True)

    # Load the Complex TR dataset and corpus
    dataset_path = "reproduce/dataset/complex-tr/complex_tr_1000.json"
    corpus_path = "reproduce/dataset/complex-tr/complex_tr_1000_corpus.json"

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")
    if not os.path.exists(corpus_path):
        raise FileNotFoundError(f"Corpus not found at {corpus_path}")

    samples = load_complex_tr_dataset(dataset_path)
    corpus_facts = load_corpus_facts(corpus_path)

    logging.info(f"Loaded {len(corpus_facts)} corpus facts")

    # Filter samples if indices are provided
    if args.indices:
        indices = json.load(open(args.indices))
        samples = [samples[i] for i in indices if i < len(samples)]
        logging.info(f"Filtered to {len(samples)} samples using provided indices")

    # Limit number of samples if specified
    if args.num_samples:
        samples = samples[: args.num_samples]
        logging.info(f"Limited to {args.num_samples} samples")

    # For debug mode, use only 1 sample
    if args.debug:
        samples = samples[:1]
        logging.info("Debug mode: processing only 1 sample")

    # Setup model configuration for dense retrieval
    llm_label = args.llm_name.replace("/", "_") if args.llm_name is not None else "None"
    embedding_label = args.embedding_name.replace("/", "_") if args.embedding_name is not None else "None"

    config = BaseConfig(
        llm_base_url=args.llm_base_url,
        llm_name=args.llm_name,
        dataset="complex_tr",
        embedding_model_name=args.embedding_name,
        force_index_from_scratch=False,
        force_openie_from_scratch=False,
        retrieval_top_k=200,
        linking_top_k=5,
        qa_top_k=args.retrieval_top_k,
        do_eval_retrieval=False,
        do_eval_qa=True,
        graph_type="dpr_only",  # Use dense passage retrieval only
        embedding_batch_size=32,
        max_new_tokens=None,
        corpus_len=None,
        llm_infer_mode="online",
        preprocess_chunk_func="none",
        use_azure=args.use_azure,
        qa_passage_prefix="",
        qa_prompt_template="rag_qa_unified",
    )

    # Setup dummy LLM for ReMem (we won't use it, just need it to initialize)
    llm_model = None

    # Setup embedding model
    from remem.embedding_model import _get_embedding_client

    embedding_model = _get_embedding_client(
        global_config=config, embedding_model_name=args.embedding_name, openai_style_server=True
    )

    # Create ReMem instance for dense retrieval
    working_dir = os.path.join(config.save_dir, f"complex_tr_tiser_{llm_label}_{embedding_label}")
    rag = ReMem(global_config=config, working_dir=working_dir, llm=llm_model)
    rag.set_embedding_model(embedding_model)

    print(f"Encoding {len(corpus_facts)} corpus documents before parallel processes...")
    rag.add_chunk_and_embeddings(corpus_facts)
    print("Encoding complete. Ready for parallel processing.")

    # Metrics for evaluation (same as complex_tr_dense.py)
    selected_metrics = ("qa_em", "qa_f1", "qa_mem0_llm_judge", "qa_bleu1")

    # Initialize LLM model for TISER evaluation
    logging.info(f"Initializing LLM model: {args.llm_name}")

    try:
        llm_client = CacheOpenAI(
            cache_dir=args.cache_dir,
            llm_name=args.llm_name,
            llm_base_url=args.llm_base_url,
            use_azure=args.use_azure,
            temperature=0.0,
        )
        logging.info("LLM model initialized successfully")
    except Exception as e:
        logging.error(f"Failed to initialize LLM model: {e}")
        exit(1)

    logging.info(f"Starting evaluation of {len(samples)} samples...")

    if args.debug:
        print("Running Complex TR TISER evaluation in debug mode...")

        # Process single sample for debug
        sample = samples[0]
        sample_idx, metrics_dict, query_solution = evaluate_sample_tiser(
            llm_client, sample, 0, corpus_facts, rag, args.retrieval_top_k, selected_metrics
        )

        if query_solution is not None and metrics_dict:
            print(f"Sample {sample_idx}: {metrics_dict}")
            print(f"Predicted: {getattr(query_solution, 'predicted_answer', 'N/A')}")
            print(f"Gold: {query_solution.gold_answers}")

        # Save single result
        query_solutions = [query_solution] if query_solution else []
        final_metrics = metrics_dict
        num_processed = 1

    else:
        print(f"Running Complex TR TISER evaluation with parallel processing ({args.num_threads} workers)...")

        # Initialize thread-safe components
        safe_metrics = ThreadSafeMetrics()

        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=args.num_threads) as executor:
            # Submit all tasks
            future_to_idx = {
                executor.submit(
                    evaluate_sample_with_index,
                    (sample_idx, llm_client, sample, corpus_facts, rag, args.retrieval_top_k, selected_metrics),
                ): sample_idx
                for sample_idx, sample in enumerate(samples)
            }

            # Process completed tasks with progress bar
            with tqdm(total=len(samples), desc="Processing samples") as pbar:
                for future in as_completed(future_to_idx):
                    sample_idx, metrics_dict, query_solution = future.result()

                    if query_solution is not None and metrics_dict:
                        safe_metrics.update(metrics_dict, query_solution)

                    pbar.update(1)

                    # Print progress periodically
                    current_averages, num_processed = safe_metrics.get_current_averages()
                    if num_processed % 10 == 0 and current_averages:
                        print(f"Processed {num_processed} samples. Current averages: {current_averages}")

        # Get final results
        final_metrics, query_solutions, num_processed = safe_metrics.get_final_results()

    # Print final results
    print("\n=== Final Results ===")
    print("Dataset: complex_tr")
    print(f"Total samples: {num_processed}")
    print(f"LLM: {args.llm_name}")
    print(f"Embedding Model: {args.embedding_name}")
    print("Method: TISER with Dense Retrieval")
    print(f"Retrieval Top K: {args.retrieval_top_k}")
    print()

    # Separate retrieval and QA metrics for better display
    retrieval_metrics = {k: v for k, v in final_metrics.items() if "retrieval" in k}
    qa_metrics = {k: v for k, v in final_metrics.items() if k not in retrieval_metrics}

    if retrieval_metrics:
        print("Retrieval Metrics:")
        for key, value in retrieval_metrics.items():
            print(f"  {key}: {value}")

    if qa_metrics:
        print("\nQA Metrics:")
        for key, value in qa_metrics.items():
            print(f"  {key}: {value}")

    # Save final results
    samples_dict = safe_serialize_query_solutions(query_solutions)
    tiser_results = {
        "dataset": "complex_tr",
        "llm_name": args.llm_name,
        "embedding_name": args.embedding_name,
        "samples": samples_dict,
        "overall_metrics": final_metrics,
        "total_samples": num_processed,
        "total_corpus_facts": len(corpus_facts),
        "config": {
            "retrieval_top_k": args.retrieval_top_k,
            "num_threads": args.num_threads,
            "prompt_template": "TISER (Chain of Thought with reflection and timeline) for Complex TR with dense retrieval",
        },
    }

    results_file = f"{rag.working_dir}/tiser_results_{llm_label}_{embedding_label}.json"
    success = safe_dump_json(tiser_results, results_file)
    if not success:
        print(f"Warning: Had to use fallback serialization for {results_file}")

    logging.info("Complex TR TISER evaluation completed!")
    logging.info(f"Results saved to: {results_file}")


if __name__ == "__main__":
    main()
