#!/usr/bin/env python3
"""
Script to run ReMem on the Complex Temporal Reasoning (complex_tr) dataset.
The Complex TR dataset contains temporal reasoning questions based on temporal facts.

The corpus facts are loaded once from complex_tr_1000_corpus.json, and questions are processed from complex_tr_1000.json.
"""

import argparse
import json
import logging
import os
import traceback
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from tqdm import tqdm

from remem.remem import ReMem
from remem.utils.config_utils import BaseConfig
from remem.utils.misc_utils import string_to_bool
from remem.utils.results_utils import get_inference_type


def load_corpus_facts(corpus_path):
    """
    Load temporal facts from the corpus file.

    Args:
        corpus_path (str): Path to the corpus JSON file

    Returns:
        list: List of temporal fact strings
    """
    with open(corpus_path, "r") as f:
        corpus_facts = json.load(f)

    return corpus_facts


def extract_fact_context_from_sample(sample):
    """
    Extract the fact context from the sample for gold document identification.

    Args:
        sample (dict): The complex_tr sample

    Returns:
        list: List of relevant fact strings from the sample's fact_context
    """
    fact_context = sample.get("fact_context", "")
    if not fact_context:
        return []

    # Split by newlines and clean up
    facts = []
    for line in fact_context.split("\n"):
        line = line.strip()
        if line:
            facts.append(line)

    return facts


def update_metrics_and_progress(
    result, cumulative_metrics, num_processed, total_samples, pbar, mode, args, metrics_lock=None
):
    """
    Update cumulative metrics and progress display (thread-safe).

    Args:
        result: Processing result from process_single_sample
        cumulative_metrics: Dictionary to accumulate metrics
        num_processed: Current number of processed samples
        total_samples: Total number of samples to process
        pbar: Progress bar instance
        mode: "Parallel" or "Sequential"
        args: Command line arguments
        metrics_lock: Threading lock for parallel mode (optional)

    Returns:
        int: Updated num_processed count
    """

    def _update_metrics():
        nonlocal num_processed

        # Extract metrics from sample_results
        if "sample_results" in result:
            sample_results = result["sample_results"]

            # Handle different formats of sample_results
            if isinstance(sample_results, list) and len(sample_results) > 0:
                # For successful samples, sample_results is a list from [q.to_dict() for q in query_solutions]
                sample_metrics = sample_results[0].get("sample_metrics", {})
            elif isinstance(sample_results, dict):
                # For failed samples, sample_results is the failed_result dict
                sample_metrics = sample_results.get("sample_metrics", {})
            else:
                sample_metrics = {}

            # Add metrics to cumulative totals
            for key, value in sample_metrics.items():
                cumulative_metrics[key] += value

        num_processed += 1

        # Calculate current averages
        current_metrics = {}
        for key, value in cumulative_metrics.items():
            current_metrics[key] = value / num_processed

        # Update progress bar description with key metrics (show first few metrics)
        metrics_display = []
        for key, value in list(current_metrics.items())[:3]:  # Show first 3 metrics
            metrics_display.append(f"{key}:{value:.3f}")
        metrics_str = " ".join(metrics_display)
        pbar.set_description(f"{mode} [{num_processed}/{total_samples}] {metrics_str}")

        return num_processed

    # Use lock for parallel mode, direct execution for sequential
    if metrics_lock:
        with metrics_lock:
            return _update_metrics()
    else:
        return _update_metrics()


def process_single_sample(sample_data):
    """
    Process a single sample for parallel execution.

    Args:
        sample_data (tuple): Tuple containing (sample_idx, sample, rag, config, args, selected_metrics, force_rag_from_scratch)

    Returns:
        dict: Processing results for this sample
    """
    sample_idx, sample, rag, args, selected_metrics, force_rag_from_scratch, qa_reader_label = sample_data

    # Use verbose logging only when enabled and not in parallel mode
    verbose_logging = args.verbose and not args.parallel

    # Define file paths early for error handling
    inference_type = (
        f"agent_max_step_{args.agent_max_steps}"
        if not args.agent_fixed_tools
        else f"agent_fixed_max_step_{args.agent_max_steps}"
    )
    os.makedirs(rag.working_dir + "/samples", exist_ok=True)
    rag_results_path = os.path.join(
        rag.working_dir + "/samples", f"rag_results_{inference_type}{qa_reader_label}_sample_{sample_idx}.json"
    )

    try:
        if verbose_logging:
            logging.info(f"Processing sample {sample_idx + 1}")

        # Extract question and answer
        question = sample["question"]
        gold_answer = sample["answers"]
        gold_answer = (
            ["; ".join(gold_answer)] if isinstance(gold_answer, list) and len(gold_answer) > 1 else gold_answer
        )

        # Extract gold documents (facts mentioned in fact_context)
        gold_facts = extract_fact_context_from_sample(sample)

        if verbose_logging:
            logging.info(f"Question: {question}")
            logging.info(f"Gold answer: {gold_answer}")
            logging.info(f"Number of gold facts: {len(gold_facts)}")

        # Print sample details for sequential mode (like dense baseline)
        if not args.parallel:
            print(f"\n--- Processing Sample {sample_idx} ---")
            print(f"Question: {question}")
            print(f"Gold answer: {gold_answer}")
            print(f"Number of gold facts: {len(gold_facts)}")
            print()

        # Prepare data for ReMem
        all_questions = [question]
        all_gold_answers = [gold_answer]
        all_gold_docs = [gold_facts]  # Gold documents for this question
        all_question_metadata = [{"id": sample.get("id", f"sample_{sample_idx}"), "sample_idx": sample_idx}]

        # Check if results already exist and skip if not forcing rerun
        if os.path.exists(rag_results_path) and not force_rag_from_scratch:
            try:
                with open(rag_results_path, "r") as f:
                    existing_results = json.load(f)
                    return {
                        "sample_idx": sample_idx,
                        "success": True,
                        "skipped": True,
                        "sample_results": existing_results,
                        "message": f"Loaded existing results for sample {sample_idx}",
                    }
            except Exception as e:
                if verbose_logging:
                    logging.warning(f"Could not load existing results for sample {sample_idx}: {e}")

        # Run RAG for QA using the pre-indexed graph
        (query_solutions, all_response_message, all_metadata, retrieval_results, qa_results) = rag.rag_for_qa(
            queries=all_questions,
            gold_docs=all_gold_docs,
            gold_answers=all_gold_answers,
            metrics=selected_metrics,
            question_metadata=all_question_metadata,
        )

        # Save RAG results for this sample
        with open(rag_results_path, "w") as f:
            rag_results = [q.to_dict() for q in query_solutions] if query_solutions else []
            json.dump(rag_results, f, indent=2)

        return {
            "sample_idx": sample_idx,
            "success": True,
            "skipped": False,
            "sample_results": rag_results,
            "message": f"Successfully processed sample {sample_idx}",
        }

    except Exception as e:
        if verbose_logging:
            logging.error(f"Error processing sample {sample_idx}: {e}")
            logging.error(f"Traceback: {traceback.format_exc()}")

        # Create a failed result file with zero metrics for consistency
        failed_result = {
            "sample_idx": sample_idx,
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc(),
            "sample_metrics": {},
        }

        # Save failed result file (only if rag_results_path was defined)
        try:
            with open(rag_results_path, "w") as f:
                json.dump(failed_result, f, indent=2)
        except Exception as save_error:
            if verbose_logging:
                logging.error(f"Could not save failed result for sample {sample_idx}: {save_error}")

        # Return zero metrics for failed samples so they are still counted
        return {
            "sample_idx": sample_idx,
            "success": False,
            "skipped": False,
            "error": str(e),
            "traceback": traceback.format_exc(),
            "sample_results": failed_result,
        }


def main():
    parser = argparse.ArgumentParser(description="ReMem for Complex Temporal Reasoning Dataset")
    parser.add_argument("--llm_base_url", type=str, default="https://api.openai.com/v1", help="LLM base URL")
    parser.add_argument("--llm_name", type=str, default="gpt-4.1-mini-2025-04-14", help="LLM name")
    parser.add_argument("--embedding_name", type=str, default="nvidia/NV-Embed-v2", help="Embedding model name")
    parser.add_argument("-fi", "--force_index_from_scratch", action="store_true", help="Force index from scratch")
    parser.add_argument("-fo", "--force_openie_from_scratch", action="store_true", help="Force OpenIE from scratch")
    parser.add_argument("-fr", "--force_rag", action="store_true", help="Force rerun RAG even if results exist")
    parser.add_argument("--llm_infer_mode", choices=["online", "offline"], default="online", help="LLM inference mode")
    parser.add_argument("--use_azure", action="store_true", help="Use Azure for OpenAI")
    parser.add_argument("--indices", type=str, help="JSON file with sample indices to process")
    parser.add_argument("--extract_format", type=str, help="Extraction format")
    parser.add_argument("--extract_method", type=str, default="episodic_gist", help="Extraction method")
    parser.add_argument("--linking_top_k", type=int, default=5, help="Top-k for linking")
    parser.add_argument("--qa_top_k", type=int, default=10, help="Top-k for QA")
    parser.add_argument("--qa_reader", type=str, default="remem")
    parser.add_argument("--max_model_len", type=int, default=4096, help="Maximum model length")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of samples to process")
    parser.add_argument("--start", type=int, default=0, help="Start index")
    parser.add_argument("--end", type=int, default=None, help="End index")
    parser.add_argument("--parallel", action="store_true", help="Enable parallel processing")
    parser.add_argument("--num_workers", type=int, default=5, help="Number of parallel workers")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output (more detailed logging)")
    # Agent configuration parameters
    parser.add_argument(
        "--agent_fixed_tools", action="store_true", help="Use simple agent with only lexical_retrieve + output_answer"
    )
    parser.add_argument(
        "--agent_max_steps",
        type=int,
        default=5,
        help="Maximum reasoning steps for agent (1=retrieve only, 2=retrieve+answer, >2=full reasoning)",
    )
    parser.add_argument(
        "--agent_fixed_retrieval_tool",
        type=str,
        default="semantic_retrieve",
        help="Which retrieval tool to use in fixed mode: 'semantic_retrieve' or 'lexical_retrieve'",
    )

    args = parser.parse_args()

    qa_reader_label = "" if args.qa_reader == "remem" else f"_{args.qa_reader}"
    print(f"Arguments: {args}")

    # Setup logging based on verbosity and parallel mode
    if args.verbose and not args.parallel:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    else:
        # Reduce logging noise for parallel processing or non-verbose mode
        logging.basicConfig(level=logging.WARNING, format="%(levelname)s - %(message)s")

    # Load the Complex TR dataset
    dataset_path = "reproduce/dataset/complex-tr/complex_tr_1000.json"
    corpus_path = "reproduce/dataset/complex-tr/complex_tr_1000_corpus.json"

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")
    if not os.path.exists(corpus_path):
        raise FileNotFoundError(f"Corpus not found at {corpus_path}")

    with open(dataset_path, "r") as f:
        samples = json.load(f)

    # Load corpus facts once (they are reused for all samples)
    corpus_facts = load_corpus_facts(corpus_path)

    logging.info(f"Loaded {len(samples)} samples from Complex TR dataset")
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

    # Setup model configuration
    force_index_from_scratch = args.force_index_from_scratch
    force_openie_from_scratch = args.force_openie_from_scratch
    force_rag_from_scratch = string_to_bool(args.force_rag)

    llm_base_url = args.llm_base_url
    llm_name = args.llm_name
    embedding_label = args.embedding_name.replace("/", "_")

    if args.use_azure:
        llm_base_url = None  # prioritize Azure ENDPOINT

    # Handle LLaMA models
    if "llama" in llm_name.lower():
        args.extract_format = "json_schema"

    config = BaseConfig(
        llm_base_url=llm_base_url,
        llm_name=llm_name,
        dataset="complex_tr",
        embedding_model_name=args.embedding_name,
        force_index_from_scratch=force_index_from_scratch,
        force_openie_from_scratch=force_openie_from_scratch,
        rerank_dspy_file_path="src/remem/prompts/dspy_prompts/filter_llama3.3-70B-Instruct.json",
        retrieval_top_k=200,
        linking_top_k=args.linking_top_k,
        qa_top_k=args.qa_top_k,
        do_eval_retrieval=True,
        do_eval_qa=True,
        graph_type="facts_and_sim_passage_node_unidirectional",
        embedding_batch_size=16,
        max_new_tokens=None,
        corpus_len=None,
        llm_infer_mode=args.llm_infer_mode,
        preprocess_chunk_func="none",  # Use simple chunking for temporal facts
        use_azure=args.use_azure,
        extract_format=args.extract_format,
        extract_method=args.extract_method,
        qa_passage_prefix="Based on the temporal facts: ",
        qa_prompt_template="rag_qa_unified",  # Custom template for temporal reasoning
        qa_reader=args.qa_reader,
        max_model_len=args.max_model_len,
        # Agent configuration (only applies to episodic/temporal methods)
        agent_fixed_tools=args.agent_fixed_tools,
        agent_max_steps=args.agent_max_steps,
        agent_fixed_retrieval_tool=args.agent_fixed_retrieval_tool,
    )

    # Setup LLM model
    if args.llm_infer_mode == "offline":
        from remem.llm.vllm_offline import VLLMOffline

        llm_client = VLLMOffline(
            config,
            model_name=args.llm_name,
            cache_dir="outputs/complex_tr/llm_cache",
        )
        extract_llm_client = llm_client
        qa_llm_client = llm_client
        llm_label = args.llm_name.replace("/", "_")
        extract_llm_label = qa_llm_label = llm_label
    else:  # Online mode
        from remem.llm import CacheOpenAI

        extract_llm_label = args.llm_name.replace("/", "_")
        extract_llm_client = CacheOpenAI(
            "outputs/complex_tr/extract_llm_cache", llm_name=extract_llm_label, llm_base_url=llm_base_url
        )
        config.extract_llm_label = extract_llm_label

        qa_llm_label = args.llm_name.replace("/", "_")
        qa_llm_client = CacheOpenAI(
            "outputs/complex_tr/qa_llm_cache", llm_name=args.llm_name, llm_base_url=llm_base_url
        )
        config.qa_llm_label = qa_llm_label
        llm_label = qa_llm_label  # Define llm_label for online mode

    # Setup embedding model
    from remem.embedding_model import _get_embedding_client

    embedding_model = _get_embedding_client(global_config=config, embedding_model_name=args.embedding_name)

    # Create ReMem instance once and index the corpus facts
    print("Creating ReMem instance and indexing corpus facts...")
    working_dir = os.path.join(config.save_dir, f"complex_tr_{llm_label}_{embedding_label}")
    rag = ReMem(global_config=config, working_dir=working_dir, extract_llm=extract_llm_client, qa_llm=qa_llm_client)
    rag.set_embedding_model(embedding_model)

    # Index the corpus facts once (they are reused for all samples)
    print(f"Indexing {len(corpus_facts)} corpus facts...")
    rag.index(corpus_facts)

    # Metrics for evaluation
    selected_metrics = ("qa_em", "qa_f1", "qa_mem0_llm_judge", "qa_bleu1")

    # Track cumulative metrics (thread-safe with lock)
    cumulative_metrics = defaultdict(float)
    metrics_lock = Lock()
    num_processed = 0

    # Prepare samples for processing (apply start/end filters)
    samples_to_process = []
    for sample_idx, sample in enumerate(samples):
        if args.start is not None and sample_idx < args.start:
            continue
        if args.end is not None and sample_idx >= args.end:
            break
        samples_to_process.append((sample_idx, sample))

    if args.parallel:
        # Parallel processing
        print(f"Processing {len(samples_to_process)} samples in parallel with {args.num_workers} workers...")

        # Prepare data for parallel processing
        sample_data_list = [
            (sample_idx, sample, rag, args, selected_metrics, force_rag_from_scratch, qa_reader_label)
            for sample_idx, sample in samples_to_process
        ]

        # Process samples in parallel
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            # Submit all tasks
            future_to_sample = {
                executor.submit(process_single_sample, sample_data): sample_data[0] for sample_data in sample_data_list
            }

            # Process completed tasks with progress bar
            completed_results = []
            failed_samples = []
            skipped_samples = []

            pbar = tqdm(
                as_completed(future_to_sample),
                total=len(future_to_sample),
                desc="Processing Complex TR samples (parallel)",
                unit="sample",
                leave=True,
            )

            for future in pbar:
                sample_idx = future_to_sample[future]
                try:
                    result = future.result()
                    completed_results.append(result)

                    if result["success"]:
                        if result["skipped"]:
                            skipped_samples.append(sample_idx)
                            # Still update metrics for skipped samples (loaded from existing results)
                            num_processed = update_metrics_and_progress(
                                result,
                                cumulative_metrics,
                                num_processed,
                                len(samples_to_process),
                                pbar,
                                "Parallel",
                                args,
                                metrics_lock,
                            )
                        else:
                            # Update metrics using shared function (thread-safe)
                            num_processed = update_metrics_and_progress(
                                result,
                                cumulative_metrics,
                                num_processed,
                                len(samples_to_process),
                                pbar,
                                "Parallel",
                                args,
                                metrics_lock,
                            )
                    else:
                        failed_samples.append(sample_idx)
                        # Still update metrics for failed samples (with zero metrics)
                        num_processed = update_metrics_and_progress(
                            result,
                            cumulative_metrics,
                            num_processed,
                            len(samples_to_process),
                            pbar,
                            "Parallel",
                            args,
                            metrics_lock,
                        )
                        if args.verbose:
                            print(f"Failed to process sample {sample_idx}: {result.get('error', 'Unknown error')}")

                except Exception as e:
                    failed_samples.append(sample_idx)
                    if args.verbose:
                        print(f"Exception processing sample {sample_idx}: {e}")
                    # Create a failed result with zero metrics for consistency
                    failed_result = {
                        "sample_idx": sample_idx,
                        "success": False,
                        "skipped": False,
                        "error": str(e),
                        "sample_metrics": {},
                    }
                    failed_result_wrapper = {
                        "sample_idx": sample_idx,
                        "success": False,
                        "skipped": False,
                        "error": str(e),
                        "sample_results": failed_result,
                    }
                    num_processed = update_metrics_and_progress(
                        failed_result_wrapper,
                        cumulative_metrics,
                        num_processed,
                        len(samples_to_process),
                        pbar,
                        "Parallel",
                        args,
                        metrics_lock,
                    )

        print(
            f"Completed parallel processing: {len(completed_results)} total, {num_processed} processed for metrics, {len(skipped_samples)} skipped, {len(failed_samples)} failed"
        )

    else:
        # Sequential processing (using the same process_single_sample function)
        print(f"Processing {len(samples_to_process)} samples sequentially...")

        skipped_samples = []
        failed_samples = []
        pbar = tqdm(samples_to_process, desc="Processing Complex TR samples (sequential)", unit="sample", leave=True)

        for sample_idx, sample in pbar:
            sample_data = (sample_idx, sample, rag, args, selected_metrics, force_rag_from_scratch, qa_reader_label)
            result = process_single_sample(sample_data)

            if result["success"]:
                if result["skipped"]:
                    skipped_samples.append(sample_idx)
                    # Still update metrics for skipped samples (loaded from existing results)
                    num_processed = update_metrics_and_progress(
                        result, cumulative_metrics, num_processed, len(samples_to_process), pbar, "Sequential", args
                    )
                    if args.verbose:
                        print(f"Skipped sample {sample_idx} (results already exist)")
                else:
                    # Update metrics using shared function (no lock needed for sequential)
                    num_processed = update_metrics_and_progress(
                        result, cumulative_metrics, num_processed, len(samples_to_process), pbar, "Sequential", args
                    )
            else:
                failed_samples.append(sample_idx)
                # Still update metrics for failed samples (with zero metrics)
                num_processed = update_metrics_and_progress(
                    result, cumulative_metrics, num_processed, len(samples_to_process), pbar, "Sequential", args
                )
                if args.verbose:
                    print(f"Failed to process sample {sample_idx}: {result.get('error', 'Unknown error')}")

        print(
            f"Completed sequential processing: {num_processed} processed for metrics, {len(skipped_samples)} skipped, {len(failed_samples)} failed"
        )

    # Print final results
    print("\n=== Final Results ===")
    print("Dataset: complex_tr")
    print(f"Total samples: {num_processed}")
    print(f"LLM: {args.llm_name}")
    print(f"Embedding Model: {args.embedding_name}")
    print("Method: ReMem (Agent-based)")
    print(f"Agent Max Steps: {args.agent_max_steps}")
    print(f"Agent Fixed Tools: {args.agent_fixed_tools}")
    if args.parallel:
        print(f"Processing Mode: Parallel ({args.num_workers} workers)")
    else:
        print("Processing Mode: Sequential")
    print()

    # Separate retrieval and QA metrics for better display
    retrieval_metrics = {k: v for k, v in cumulative_metrics.items() if "retrieval" in k}
    qa_metrics = {k: v for k, v in cumulative_metrics.items() if k not in retrieval_metrics}

    if retrieval_metrics:
        print("Average Retrieval Metrics:")
        for key, value in retrieval_metrics.items():
            avg_value = value / num_processed
            print(f"  {key}: {avg_value:.4f}")

    if qa_metrics:
        print("\nAverage QA Metrics:")
        for key, value in qa_metrics.items():
            avg_value = value / num_processed
            print(f"  {key}: {avg_value:.4f}")

    # Save overall results
    inference_type = get_inference_type(args.agent_fixed_tools, args.agent_max_steps, args.extract_method)

    # Collect all individual sample results
    all_samples = []
    failed_samples_info = []
    samples_dir = os.path.join(rag.working_dir, "samples")

    if os.path.exists(samples_dir):
        # Process each sample that was attempted
        for sample_idx, sample in samples_to_process:
            rag_results_path = os.path.join(samples_dir, f"rag_results_{inference_type}_sample_{sample_idx}.json")
            if os.path.exists(rag_results_path):
                try:
                    with open(rag_results_path, "r") as f:
                        sample_data = json.load(f)
                        all_samples.append(sample_data)
                except Exception as e:
                    print(f"Warning: Could not load sample {sample_idx} results: {e}")
                    # Create a placeholder for failed sample
                    failed_samples_info.append(
                        {
                            "sample_idx": sample_idx,
                            "error": f"Could not load results: {e}",
                            "question": sample.get("question", ""),
                            "answer": sample.get("answers", ""),
                        }
                    )
            else:
                # Sample was processed but no results file exists (failed sample)
                failed_samples_info.append(
                    {
                        "sample_idx": sample_idx,
                        "error": "No results file found",
                        "question": sample.get("question", ""),
                        "answer": sample.get("answers", ""),
                    }
                )

    # Calculate total samples that should have been processed
    total_samples_attempted = len(samples_to_process)

    # Ensure we have the expected number of samples (successful + failed)
    if len(all_samples) + len(failed_samples_info) != total_samples_attempted:
        print(
            f"Warning: Expected {total_samples_attempted} samples, but have {len(all_samples)} successful + {len(failed_samples_info)} failed = {len(all_samples) + len(failed_samples_info)} total"
        )

    overall_results = {
        "num_samples": num_processed,
        "num_successful_samples": len(all_samples),
        "num_failed_samples": len(failed_samples_info),
        "total_samples_attempted": total_samples_attempted,
        "total_corpus_facts": len(corpus_facts),
        "average_metrics": (
            {key: cumulative_metrics[key] / num_processed for key in cumulative_metrics} if num_processed > 0 else {}
        ),
        "configuration": {
            "llm_name": llm_name,
            "embedding_name": args.embedding_name,
            "dataset": "complex_tr",
            "num_samples_processed": num_processed,
            "inference_type": inference_type,
            "agent_fixed_tools": args.agent_fixed_tools,
            "agent_max_steps": args.agent_max_steps,
            "agent_fixed_retrieval_tool": args.agent_fixed_retrieval_tool,
        },
        "samples": all_samples,
        "failed_samples": failed_samples_info,
    }

    output_dir = f"outputs/complex_tr/complex_tr_{llm_label}_{embedding_label}"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Collected {len(all_samples)} individual sample results")
    if len(failed_samples_info) > 0:
        print(f"Found {len(failed_samples_info)} failed samples:")
        for failed_info in failed_samples_info:
            print(f"  Sample {failed_info['sample_idx']}: {failed_info['error']}")

    overall_results_path = f"{output_dir}/overall_results_{inference_type}{qa_reader_label}.json"
    with open(overall_results_path, "w") as f:
        json.dump(overall_results, f, indent=2)

    print(f"\nOverall results saved to {overall_results_path}")
    print(f"Total samples attempted: {total_samples_attempted}")
    print(f"Successful samples: {len(all_samples)}")
    print(f"Failed samples: {len(failed_samples_info)}")
    print(f"Samples processed for metrics: {num_processed}")

    # Verify that all samples are accounted for
    if num_processed == total_samples_attempted:
        print("✓ All samples are accounted for in the metrics")
    else:
        print(
            f"⚠ Warning: Metrics calculated for {num_processed} samples, but {total_samples_attempted} samples were attempted"
        )


if __name__ == "__main__":
    main()
