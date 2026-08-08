#!/usr/bin/env python3
"""
Script to run ReMem on the ToT (Test of Time) semantic dataset.
The ToT semantic dataset contains temporal reasoning questions based on temporal facts.

Usage:
    python tot_semantic.py --llm_name gpt-4o-mini --embedding_name nvidia/NV-Embed-v2
"""

import argparse
import json
import logging
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from tqdm import tqdm

from remem.remem import ReMem
from remem.utils.config_utils import BaseConfig
from remem.utils.misc_utils import string_to_bool


def remove_spaces_from_prediction(prediction):
    """
    Remove all spaces from the prediction for ToT semantic dataset.

    Args:
        prediction (str): The predicted answer from LLM

    Returns:
        str: The prediction with all spaces removed
    """
    if prediction is None:
        return ""
    return prediction.replace(" ", "")


def extract_facts(prompt):
    """
    Extract temporal facts from the prompt and convert them to document-like format.

    Args:
        prompt (str): The full prompt containing temporal facts and question

    Returns:
        list: List of temporal facts formatted as documents
    """
    lines = prompt.split("\n")
    facts = []

    # Find the temporal facts section (before "Answer the following question")
    for line in lines:
        line = line.strip()
        if line and not line.startswith("Here is a set") and not line.startswith("Answer the following"):
            # This looks like a temporal fact
            if " was the " in line and " from " in line and " to " in line:
                facts.append(line)
            elif line.startswith("Question:"):
                break  # We've reached the question section

    return facts


def format_facts_as_documents(facts):
    """
    Convert temporal facts into document format suitable for ReMem indexing.

    Args:
        facts (list): List of temporal fact strings

    Returns:
        list: List of string documents for ReMem indexing
    """
    # Return facts as simple strings for the current preprocessing pipeline
    # The preprocessing expects a list of strings, not dictionaries
    return facts


def extract_question_from_prompt(prompt):
    """
    Extract the actual question from the prompt.

    Args:
        prompt (str): The full prompt

    Returns:
        str: The extracted question
    """
    lines = prompt.split("\n")
    question_started = False
    question_lines = []

    for line in lines:
        line = line.strip()
        if line.startswith("Question:"):
            question_started = True
            question_text = line.replace("Question:", "").strip()
            if question_text:
                question_lines.append(question_text)
        elif question_started and line and not line.startswith("Answer") and not line.startswith("Output"):
            question_lines.append(line)
        elif question_started and (line.startswith("Answer") or line.startswith("Output")):
            break

    return " ".join(question_lines).strip()


def get_gold_answer_from_sample(sample):
    """
    Extract the gold answer from the sample.

    Args:
        sample (dict): The ToT semantic sample

    Returns:
        str: The gold answer
    """
    return str(sample.get("label", ""))


def process_single_sample(sample_data):
    """
    Process a single sample for parallel execution.

    Args:
        sample_data (tuple): Tuple containing (sample_idx, sample, config, clients, args, selected_metrics)

    Returns:
        dict: Processing results for this sample
    """
    sample_idx, sample, config, clients, args, selected_metrics = sample_data
    extract_llm_client, qa_llm_client, embedding_model, qa_llm_label, embedding_label = clients

    # Use verbose logging only when enabled and not in parallel mode
    verbose_logging = args.verbose and not args.parallel

    try:
        if verbose_logging:
            logging.info(f"Processing sample {sample_idx + 1}")

        # Extract temporal facts from the prompt
        facts = extract_facts(sample["prompt"])

        if verbose_logging:
            logging.info(f"Extracted {len(facts)} temporal facts")

        # Format facts as documents for ReMem
        candidate_docs = format_facts_as_documents(facts)

        # Extract question and answer
        question = sample.get("question", extract_question_from_prompt(sample["prompt"]))
        gold_answer = get_gold_answer_from_sample(sample)

        # Prepare data for ReMem
        all_questions = [question]
        all_gold_answers = [gold_answer]
        all_question_metadata = [
            {
                "type": sample.get("question_type", "unknown"),
                "graph_algorithm": sample.get("graph_gen_algorithm", "unknown"),
                "sorting_type": sample.get("sorting_type", "unknown"),
                "sample_idx": sample_idx,
            }
        ]

        # Setup dataset-specific configuration (create a copy for thread safety)
        thread_config = BaseConfig(**config.__dict__)
        dataset_name = f"tot_semantic_{sample_idx}"
        thread_config.dataset = dataset_name
        thread_config.__post_init__()

        working_dir = os.path.join(thread_config.save_dir, f"{dataset_name}_{qa_llm_label}_{embedding_label}")
        inference_type = (
            f"agent_max_step_{args.agent_max_steps}"
            if not args.agent_fixed_tools
            else f"agent_fixed_max_step_{args.agent_max_steps}"
        )
        rag_results_path = os.path.join(working_dir, f"rag_results_{inference_type}.json")

        # Check if rag_results.json already exists and skip if not forcing rerun
        if os.path.exists(rag_results_path) and not args.force_rag:
            # Don't print during parallel processing to avoid tqdm interference
            try:
                with open(rag_results_path, "r") as f:
                    existing_results = json.load(f)
                    if "sample" in existing_results and "qa_results" in existing_results["sample"]:
                        # Return existing metrics
                        existing_qa_results = existing_results["sample"]["qa_results"]
                        return {
                            "sample_idx": sample_idx,
                            "success": True,
                            "skipped": True,
                            "retrieval_results": existing_results["sample"].get("retrieval_results", {}),
                            "qa_results": existing_qa_results,
                            "sample_results": existing_results["sample"],
                            "message": f"RAG results already exist for sample {sample_idx}, skipped",
                        }
            except Exception as e:
                return {
                    "sample_idx": sample_idx,
                    "success": False,
                    "skipped": False,
                    "error": f"Error loading existing results: {e}",
                    "will_rerun": True,
                }

        # Create ReMem instance for this thread
        rag = ReMem(
            global_config=thread_config, working_dir=working_dir, extract_llm=extract_llm_client, qa_llm=qa_llm_client
        )
        rag.set_embedding_model(embedding_model)

        # Index the temporal facts
        rag.index(candidate_docs)

        # For retrieval evaluation, we need to identify which facts are relevant
        # This is challenging for ToT as we don't have explicit relevance labels
        # We'll use all facts as potentially relevant for now
        gold_docs = [candidate_docs]  # All facts could be relevant

        # Run RAG for QA
        (query_solutions, all_response_message, all_metadata, retrieval_results, qa_results) = rag.rag_for_qa(
            queries=all_questions,
            gold_docs=gold_docs,
            gold_answers=[all_gold_answers],
            metrics=selected_metrics,
            question_metadata=all_question_metadata,
        )

        # Save individual sample results
        sample_results = {
            "sample_idx": sample_idx,
            "question": question,
            "gold_answer": gold_answer,
            "predicted_answer": (
                query_solutions[0].answer if query_solutions and hasattr(query_solutions[0], "answer") else ""
            ),
            "retrieval_results": retrieval_results,
            "qa_results": qa_results,
            "metadata": all_question_metadata[0],
        }

        # Save RAG results (thread-safe as each sample has its own directory)
        with open(f"{rag.working_dir}/rag_results_{inference_type}.json", "w") as f:
            rag_results = {
                "sample": sample_results,
                "query_solutions": [q.to_dict() for q in query_solutions] if query_solutions else [],
            }
            json.dump(rag_results, f, indent=2)

        return {
            "sample_idx": sample_idx,
            "success": True,
            "skipped": False,
            "retrieval_results": retrieval_results,
            "qa_results": qa_results,
            "sample_results": sample_results,
            "message": f"Successfully processed sample {sample_idx}",
        }

    except Exception as e:
        import traceback

        if verbose_logging:
            logging.error(f"Error processing sample {sample_idx}: {e}")
            logging.error(f"Traceback: {traceback.format_exc()}")
        return {
            "sample_idx": sample_idx,
            "success": False,
            "skipped": False,
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


def main():
    parser = argparse.ArgumentParser(description="ReMem for ToT Semantic Dataset")
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
    parser.add_argument("--extract_method", type=str, default="temporal", help="Extraction method")
    parser.add_argument("--linking_top_k", type=int, default=5, help="Top-k for linking")
    parser.add_argument("--qa_top_k", type=int, default=10, help="Top-k for QA")
    parser.add_argument("--max_model_len", type=int, default=4096, help="Maximum model length")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of samples to process")
    parser.add_argument("--start", type=int, default=0, help="Start index")
    parser.add_argument("--end", type=int, default=None, help="End index")
    parser.add_argument("--parallel", action="store_true", help="Enable parallel processing")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of parallel workers")
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
        default="lexical_retrieve",
        help="Which retrieval tool to use in fixed mode: 'semantic_retrieve' or 'lexical_retrieve'",
    )

    args = parser.parse_args()

    print(f"Arguments: {args}")

    # Setup logging based on verbosity and parallel mode
    if args.verbose and not args.parallel:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    else:
        # Reduce logging noise for parallel processing or non-verbose mode
        logging.basicConfig(level=logging.WARNING, format="%(levelname)s - %(message)s")

    # Load the ToT semantic dataset
    dataset_path = "reproduce/dataset/tot/tot_semantic.json"
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    with open(dataset_path, "r") as f:
        samples = json.load(f)

    logging.info(f"Loaded {len(samples)} samples from ToT semantic dataset")

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
        dataset="tot_semantic",
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
        preprocess_chunk_func="none",  # Use word-based chunking for plain text temporal facts
        use_azure=args.use_azure,
        extract_format=args.extract_format,
        extract_method=args.extract_method,
        qa_passage_prefix="Based on the temporal facts: ",
        qa_prompt_template="rag_qa_unified",  # Custom template for temporal reasoning
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
            cache_dir="outputs/tot_semantic/llm_cache",
        )
        extract_llm_client = llm_client
        qa_llm_client = llm_client
        llm_label = args.llm_name.replace("/", "_")
        extract_llm_label = qa_llm_label = llm_label
    else:  # Online mode
        from remem.llm import CacheOpenAI

        extract_llm_label = "gpt-4.1-mini-2025-04-14"
        extract_llm_client = CacheOpenAI("outputs/tot_semantic/extract_llm_cache", llm_name=extract_llm_label)
        config.extract_llm_label = extract_llm_label

        qa_llm_label = args.llm_name.replace("/", "_")
        qa_llm_client = CacheOpenAI("outputs/tot_semantic/qa_llm_cache", llm_name=args.llm_name)
        config.qa_llm_label = qa_llm_label
        llm_label = qa_llm_label  # Define llm_label for online mode

    # Setup embedding model
    from remem.embedding_model import _get_embedding_client

    embedding_model = _get_embedding_client(global_config=config, embedding_model_name=args.embedding_name)

    # Metrics for evaluation
    selected_metrics = ["qa_em"]

    # Track cumulative metrics (thread-safe with lock)
    cumulative_metrics = defaultdict(float)
    metrics_lock = Lock()
    num_processed = 0

    # Add category-based evaluation tracking
    question_type_metrics = defaultdict(lambda: defaultdict(float))
    question_type_counts = defaultdict(int)
    graph_algorithm_metrics = defaultdict(lambda: defaultdict(float))
    graph_algorithm_counts = defaultdict(int)
    sorting_type_metrics = defaultdict(lambda: defaultdict(float))
    sorting_type_counts = defaultdict(int)

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
        clients = (extract_llm_client, qa_llm_client, embedding_model, qa_llm_label, embedding_label)
        sample_data_list = [
            (sample_idx, sample, config, clients, args, selected_metrics) for sample_idx, sample in samples_to_process
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

            with tqdm(
                total=len(samples_to_process),
                desc="Processing ToT samples (parallel)",
                unit="sample",
                leave=True,
                position=0,
            ) as pbar:
                for future in as_completed(future_to_sample):
                    sample_idx = future_to_sample[future]
                    try:
                        result = future.result()

                        with metrics_lock:
                            if result["success"]:
                                num_processed += 1
                                completed_results.append(result)

                                if result["skipped"]:
                                    skipped_samples.append(sample_idx)

                                if not result["skipped"]:
                                    # Accumulate metrics
                                    for key in result["retrieval_results"]:
                                        cumulative_metrics[key] += result["retrieval_results"][key]
                                    for key in result["qa_results"]:
                                        cumulative_metrics[key] += result["qa_results"][key]
                                else:
                                    # Add existing metrics for skipped samples
                                    for key in result["retrieval_results"]:
                                        cumulative_metrics[key] += result["retrieval_results"][key]
                                    for key in result["qa_results"]:
                                        cumulative_metrics[key] += result["qa_results"][key]

                                # Accumulate category-based metrics
                                if "sample_results" in result and "metadata" in result["sample_results"]:
                                    metadata = result["sample_results"]["metadata"]
                                    qa_results = result["qa_results"]

                                    # Question type metrics
                                    question_type = metadata.get("type", "unknown")
                                    question_type_counts[question_type] += 1
                                    for key in qa_results:
                                        question_type_metrics[question_type][key] += qa_results[key]

                                    # Graph algorithm metrics
                                    graph_algorithm = metadata.get("graph_algorithm", "unknown")
                                    graph_algorithm_counts[graph_algorithm] += 1
                                    for key in qa_results:
                                        graph_algorithm_metrics[graph_algorithm][key] += qa_results[key]

                                    # Sorting type metrics
                                    sorting_type = metadata.get("sorting_type", "unknown")
                                    sorting_type_counts[sorting_type] += 1
                                    for key in qa_results:
                                        sorting_type_metrics[sorting_type][key] += qa_results[key]

                                # Update tqdm description with current metrics every 10 samples
                                if num_processed % 5 == 0:
                                    qa_metrics_str = ""
                                    for key in result["qa_results"]:
                                        if key in cumulative_metrics:
                                            avg_score = cumulative_metrics[key] / num_processed
                                            qa_metrics_str += f"{key}: {avg_score:.3f} "

                                    pbar.set_description(f"Processing (avg: {qa_metrics_str.strip()})")

                                    # Use tqdm.write for thread-safe output
                                    tqdm.write(
                                        f"✓ Processed {num_processed} samples | Skipped: {len(skipped_samples)} | Failed: {len(failed_samples)}"
                                    )
                            else:
                                failed_samples.append((sample_idx, result.get("error", "Unknown error")))
                                # Use tqdm.write for thread-safe error output
                                tqdm.write(
                                    f"✗ Sample {sample_idx} failed: {result.get('error', 'Unknown error')[:200]}..."
                                )

                        pbar.update(1)

                    except Exception as exc:
                        failed_samples.append((sample_idx, str(exc)))
                        tqdm.write(f"✗ Sample {sample_idx} exception: {str(exc)[:200]}...")
                        pbar.update(1)

            # Print summary after completion
            print("\n=== Parallel Processing Summary ===")
            print(f"Total processed: {num_processed}")
            print(f"Successful: {len(completed_results) - len(skipped_samples)}")
            print(f"Skipped (existing): {len(skipped_samples)}")
            print(f"Failed: {len(failed_samples)}")

            if failed_samples:
                print(
                    f"\nFailed samples: {[idx for idx, _ in failed_samples[:5]]}"
                    + (f" and {len(failed_samples)-5} more..." if len(failed_samples) > 5 else "")
                )

    else:
        # Sequential processing (using the same process_single_sample function)
        print(f"Processing {len(samples_to_process)} samples sequentially...")

        # Prepare data for sequential processing (same format as parallel)
        clients = (extract_llm_client, qa_llm_client, embedding_model, qa_llm_label, embedding_label)

        for sample_idx, sample in tqdm(
            samples_to_process, desc="Processing ToT samples (sequential)", unit="sample", leave=True
        ):
            # Use the same process_single_sample function
            sample_data = (sample_idx, sample, config, clients, args, selected_metrics)
            result = process_single_sample(sample_data)

            if result["success"]:
                num_processed += 1
                if not result["skipped"]:
                    # Accumulate metrics
                    for key in result["retrieval_results"]:
                        cumulative_metrics[key] += result["retrieval_results"][key]
                    for key in result["qa_results"]:
                        cumulative_metrics[key] += result["qa_results"][key]
                else:
                    # Add existing metrics for skipped samples
                    for key in result["retrieval_results"]:
                        cumulative_metrics[key] += result["retrieval_results"][key]
                    for key in result["qa_results"]:
                        cumulative_metrics[key] += result["qa_results"][key]

                # Accumulate category-based metrics
                if "sample_results" in result and "metadata" in result["sample_results"]:
                    metadata = result["sample_results"]["metadata"]
                    qa_results = result["qa_results"]

                    # Question type metrics
                    question_type = metadata.get("type", "unknown")
                    question_type_counts[question_type] += 1
                    for key in qa_results:
                        question_type_metrics[question_type][key] += qa_results[key]

                    # Graph algorithm metrics
                    graph_algorithm = metadata.get("graph_algorithm", "unknown")
                    graph_algorithm_counts[graph_algorithm] += 1
                    for key in qa_results:
                        graph_algorithm_metrics[graph_algorithm][key] += qa_results[key]

                    # Sorting type metrics
                    sorting_type = metadata.get("sorting_type", "unknown")
                    sorting_type_counts[sorting_type] += 1
                    for key in qa_results:
                        sorting_type_metrics[sorting_type][key] += qa_results[key]

                # Print current average metrics (less frequent in sequential mode)
                if num_processed % 5 == 0:  # Every 5 samples instead of every sample
                    tqdm.write(f"\nSample {sample_idx + 1} results:")
                    retrieval_metrics_str = ""
                    qa_metrics_str = ""

                    for key in result["retrieval_results"]:
                        avg_score = cumulative_metrics[key] / num_processed
                        retrieval_metrics_str += f"{key}: {avg_score:.4f} "

                    for key in result["qa_results"]:
                        avg_score = cumulative_metrics[key] / num_processed
                        qa_metrics_str += f"{key}: {avg_score:.4f} "

                    if retrieval_metrics_str:
                        tqdm.write(f"Retrieval metrics: {retrieval_metrics_str.strip()}")
                    if qa_metrics_str:
                        tqdm.write(f"QA metrics: {qa_metrics_str.strip()}")
            else:
                tqdm.write(f"✗ Failed to process sample {sample_idx}: {result.get('error', 'Unknown error')[:200]}...")

    # Print final results
    print(f"\n=== Final Results ({num_processed} samples) ===")
    print("Overall Average Metrics:")
    for key in cumulative_metrics:
        avg_score = cumulative_metrics[key] / num_processed
        print(f"  {key}: {avg_score:.4f}")

    # Print category breakdowns
    def print_category_breakdown(category_name, category_counts, category_metrics):
        if category_counts:
            print(f"\n{category_name} breakdown:")
            for category, count in category_counts.items():
                print(f"  {category} ({count} samples):")
                for metric_name, total_value in category_metrics[category].items():
                    avg_value = total_value / count
                    print(f"    {metric_name}: {avg_value:.4f}")

    print_category_breakdown("Question Type", question_type_counts, question_type_metrics)
    print_category_breakdown("Graph Algorithm", graph_algorithm_counts, graph_algorithm_metrics)
    print_category_breakdown("Sorting Type", sorting_type_counts, sorting_type_metrics)

    # Prepare category breakdowns for saving
    def prepare_category_breakdown(category_counts, category_metrics):
        breakdown = {}
        for category, count in category_counts.items():
            breakdown[category] = {"count": count, "metrics": {}}
            for metric_name, total_value in category_metrics[category].items():
                avg_value = total_value / count
                breakdown[category]["metrics"][metric_name] = round(avg_value, 4)
        return breakdown

    # Save overall results
    inference_type = (
        f"agent_max_step_{args.agent_max_steps}"
        if not args.agent_fixed_tools
        else f"agent_fixed_max_step_{args.agent_max_steps}"
    )
    overall_results = {
        "num_samples": num_processed,
        "average_metrics": {key: cumulative_metrics[key] / num_processed for key in cumulative_metrics},
        "configuration": {
            "llm_name": llm_name,
            "embedding_name": args.embedding_name,
            "dataset": "tot_semantic",
            "num_samples_processed": num_processed,
            "inference_type": inference_type,
            "agent_fixed_tools": args.agent_fixed_tools,
            "agent_max_steps": args.agent_max_steps,
            "agent_fixed_retrieval_tool": args.agent_fixed_retrieval_tool,
        },
    }

    # Add category breakdowns to results
    if question_type_counts:
        overall_results["question_type_breakdown"] = prepare_category_breakdown(
            question_type_counts, question_type_metrics
        )

    if graph_algorithm_counts:
        overall_results["graph_algorithm_breakdown"] = prepare_category_breakdown(
            graph_algorithm_counts, graph_algorithm_metrics
        )

    if sorting_type_counts:
        overall_results["sorting_type_breakdown"] = prepare_category_breakdown(
            sorting_type_counts, sorting_type_metrics
        )

    output_dir = f"outputs/tot_semantic_{llm_label}_{embedding_label}"
    os.makedirs(output_dir, exist_ok=True)

    overall_results_path = f"{output_dir}/overall_results_{inference_type}.json"
    with open(overall_results_path, "w") as f:
        json.dump(overall_results, f, indent=2)

    print(f"\nOverall results saved to {overall_results_path}")


if __name__ == "__main__":
    main()
