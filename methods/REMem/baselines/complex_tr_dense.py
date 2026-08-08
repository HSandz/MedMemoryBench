"""
Complex TR Dense Retrieval Baseline

This script implements a dense retrieval baseline for the Complex Temporal Reasoning (Complex TR) dataset.
It uses embedding-based retrieval to find relevant temporal facts from the corpus and then generates answers using an LLM.
"""

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from examples.complex_tr import extract_fact_context_from_sample, load_corpus_facts
from remem.remem import ReMem
from remem.utils.config_utils import BaseConfig
from remem.utils.evaluation_progress import create_evaluation_tracker
from remem.utils.misc_utils import QuerySolution, safe_dump_json, safe_serialize_query_solutions


def process_single_sample(sample_idx, sample, corpus_facts, rag, selected_metrics, progress_tracker):
    """Process a single sample - designed to be called in parallel"""
    try:
        # Extract question and answer
        question = sample["question"]
        gold_answer = sample["answers"]
        gold_answer = (
            ["; ".join(gold_answer)] if isinstance(gold_answer, list) and len(gold_answer) > 1 else gold_answer
        )
        gold_answers = [gold_answer]  # Wrap in list as expected by evaluators

        # Extract gold documents (facts mentioned in fact_context)
        gold_facts = extract_fact_context_from_sample(sample)
        gold_docs = [gold_facts]  # Wrap in list for this question

        question_metadata = {"id": sample.get("id", f"sample_{sample_idx}"), "sample_idx": sample_idx}

        # Perform dense passage retrieval
        sorted_chunk_ids, sorted_chunk_scores = rag.dense_passage_retrieval(query=question, candidate_docs=corpus_facts)

        # Get retrieved chunk contents and metadata
        retrieved_chunk_contents = [rag.chunk_contents[chunk_id] for chunk_id in sorted_chunk_ids]
        retrieved_chunk_metadata = [rag.chunk_metadata[chunk_id] for chunk_id in sorted_chunk_ids]

        # Create query solution
        query_solution = QuerySolution(
            question,
            retrieved_chunk_contents,
            None,  # No named_entity_dict for dense retrieval
            sorted_chunk_scores,
            retrieved_chunk_metadata,
            None,  # No query_paths
            gold_answers,
            gold_docs,
            None,  # No query_graph
            {},
            None,  # No processed_query
        )

        # Get evaluators and perform QA
        qa_evaluators, retrieval_evaluators = rag.get_evaluators(gold_answers, gold_docs, selected_metrics)
        queries_solution, all_response_message, all_metadata = rag.qa([query_solution])
        query_solution_result = queries_solution[0] if queries_solution else query_solution

        # Evaluate results
        retrieval_results = rag.evaluate_retrieval(gold_docs, queries_solution, retrieval_evaluators)
        qa_results = rag.evaluate_qa(gold_answers, qa_evaluators, queries_solution, [question_metadata])

        # Combine all metrics
        all_metrics = {**retrieval_results, **qa_results}

        # Update progress tracker
        progress_tracker.update_sample(sample_idx, all_metrics, query_solution_result)

        return sample_idx, all_metrics, query_solution_result

    except Exception as e:
        # Update progress tracker for failed sample
        progress_tracker.update_sample(sample_idx, {}, failed=True, error_msg=str(e))
        return sample_idx, {}, None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="complex_tr", help="Dataset name")
    parser.add_argument("--llm_base_url", type=str, help="LLM base URL", default="https://api.openai.com/v1")
    parser.add_argument("--llm_name", type=str, default="gpt-4.1-mini", help="LLM name")
    parser.add_argument("--embedding_name", type=str, default="nvidia/NV-Embed-v2", help="embedding model name")
    parser.add_argument(
        "--llm_infer_mode",
        choices=["online", "offline"],
        default="online",
        help="LLM inference mode, offline denotes using VLLM offline batch mode",
    )
    parser.add_argument("--qa_top_k", type=int, default=10, help="Number of top facts to retrieve for QA")
    parser.add_argument("--use_azure", action="store_true", help="Use Azure for OpenAI")
    parser.add_argument("--num_samples", type=int, default=None, help="Limit number of samples to process")
    parser.add_argument("--indices", type=str, default=None, help="JSON file with sample indices to process")
    parser.add_argument("--parallel", action="store_true", help="Enable parallel processing")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of worker threads for parallel processing")
    args = parser.parse_args()

    # Load the Complex TR dataset and corpus
    dataset_path = "reproduce/dataset/complex-tr/complex_tr_1000.json"
    corpus_path = "reproduce/dataset/complex-tr/complex_tr_1000_corpus.json"

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")
    if not os.path.exists(corpus_path):
        raise FileNotFoundError(f"Corpus not found at {corpus_path}")

    with open(dataset_path, "r") as f:
        samples = json.load(f)

    # Load corpus facts (these will be our candidate documents for retrieval)
    corpus_facts = load_corpus_facts(corpus_path)

    print(f"Loaded {len(samples)} samples from Complex TR dataset")
    print(f"Loaded {len(corpus_facts)} corpus facts")

    # Filter samples if indices are provided
    if args.indices:
        indices = json.load(open(args.indices))
        samples = [samples[i] for i in indices if i < len(samples)]
        print(f"Filtered to {len(samples)} samples using provided indices")

    # Limit number of samples if specified
    if args.num_samples:
        samples = samples[: args.num_samples]
        print(f"Limited to {args.num_samples} samples")

    # Setup model configuration
    llm_name = args.llm_name
    llm_label = args.llm_name.replace("/", "_") if llm_name is not None else "None"
    embedding_label = args.embedding_name.replace("/", "_") if args.embedding_name is not None else "None"

    config = BaseConfig(
        llm_base_url=args.llm_base_url,
        llm_name=llm_name,
        dataset="complex_tr",
        embedding_model_name=args.embedding_name,
        force_index_from_scratch=False,
        force_openie_from_scratch=False,
        retrieval_top_k=200,
        linking_top_k=5,
        qa_top_k=args.qa_top_k,
        do_eval_retrieval=True,
        do_eval_qa=True,
        graph_type="dpr_only",  # Use dense passage retrieval only
        embedding_batch_size=32,
        max_new_tokens=None,
        corpus_len=None,
        llm_infer_mode=args.llm_infer_mode,
        preprocess_chunk_func="none",
        use_azure=args.use_azure,
        qa_passage_prefix="",
        qa_prompt_template="rag_qa_unified",  # Custom template for temporal reasoning
    )

    # Setup LLM for offline mode if needed
    if args.llm_infer_mode == "offline":
        from remem.llm.vllm_offline import VLLMOffline

        llm_model = VLLMOffline(
            config,
            model_name=args.llm_name,
            cache_dir="outputs/complex_tr/llm_cache",
        )
    else:
        llm_model = None

    # Setup embedding model
    from remem.embedding_model import _get_embedding_client

    embedding_model = _get_embedding_client(
        global_config=config, embedding_model_name=args.embedding_name, openai_style_server=True
    )

    # Create ReMem instance and index corpus facts
    working_dir = os.path.join(config.save_dir, f"complex_tr_dense_{llm_label}_{embedding_label}")
    rag = ReMem(global_config=config, working_dir=working_dir, llm=llm_model)
    rag.set_embedding_model(embedding_model)

    print(f"Encoding {len(corpus_facts)} corpus documents before parallel processes...")
    rag.add_chunk_and_embeddings(corpus_facts)
    print("Encoding complete. Ready for parallel processing.")

    # Metrics for evaluation
    selected_metrics = ("qa_em", "qa_f1", "qa_mem0_llm_judge", "qa_bleu1")

    if args.parallel:
        # Create unified progress tracker for parallel processing
        progress_tracker = create_evaluation_tracker(
            total_samples=len(samples),
            description="Running Complex TR samples",
            dataset_name=args.dataset,
            processing_mode="parallel",
            num_workers=args.num_workers,
            display_interval=5,
            print_interval=10,
            show_individual_results=False,  # Reduce verbose output in parallel mode
        )

        # Start progress tracking
        pbar = progress_tracker.start_progress()

        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            # Submit all tasks
            future_to_idx = {
                executor.submit(
                    process_single_sample, sample_idx, sample, corpus_facts, rag, selected_metrics, progress_tracker
                ): sample_idx
                for sample_idx, sample in enumerate(samples)
            }

            # Process completed tasks (progress is handled within process_single_sample)
            for future in as_completed(future_to_idx):
                sample_idx, metrics_dict, query_solution = future.result()
                # Progress is already updated in process_single_sample

        # Finish progress tracking and get final results
        final_results = progress_tracker.finish_progress()
        final_metrics = final_results["overall_metrics"]
        query_solutions = final_results["results"]
        num_processed = final_results["num_samples"]

    else:
        # Create unified progress tracker for sequential processing
        progress_tracker = create_evaluation_tracker(
            total_samples=len(samples),
            description="Processing Complex TR samples",
            dataset_name=args.dataset,
            processing_mode="sequential",
            num_workers=1,
            display_interval=1,  # Update every sample in sequential mode
            print_interval=5,  # Print detailed metrics every 5 samples
            show_individual_results=True,  # Show detailed results in sequential mode
            show_current_averages=True,
        )

        # Start progress tracking
        pbar = progress_tracker.start_progress()

        query_solutions = []

        # Process each sample
        for sample_idx, sample in enumerate(samples):
            # Extract question and answer
            question = sample["question"]
            gold_answers = [sample["answers"]]  # Wrap in list as expected by evaluators

            # Extract gold documents (facts mentioned in fact_context)
            gold_facts = extract_fact_context_from_sample(sample)
            gold_docs = [gold_facts]  # Wrap in list for this question

            question_metadata = {"id": sample.get("id", f"sample_{sample_idx}"), "sample_idx": sample_idx}

            # Perform dense passage retrieval
            sorted_chunk_ids, sorted_chunk_scores = rag.dense_passage_retrieval(
                query=question, candidate_docs=corpus_facts
            )

            # Get retrieved chunk contents and metadata
            retrieved_chunk_contents = [rag.chunk_contents[chunk_id] for chunk_id in sorted_chunk_ids]
            retrieved_chunk_metadata = [rag.chunk_metadata[chunk_id] for chunk_id in sorted_chunk_ids]

            # Create query solution
            query_solution = QuerySolution(
                question,
                retrieved_chunk_contents,
                None,  # No named_entity_dict for dense retrieval
                sorted_chunk_scores,
                retrieved_chunk_metadata,
                None,  # No query_paths
                gold_answers,
                gold_docs,
                None,  # No query_graph
                {},
                None,  # No processed_query
            )

            # Get evaluators and perform QA
            qa_evaluators, retrieval_evaluators = rag.get_evaluators(gold_answers, gold_docs, selected_metrics)
            queries_solution, all_response_message, all_metadata = rag.qa([query_solution])

            # Evaluate results
            retrieval_results = rag.evaluate_retrieval(gold_docs, queries_solution, retrieval_evaluators)
            qa_results = rag.evaluate_qa(gold_answers, qa_evaluators, queries_solution, [question_metadata])

            query_solutions.append(query_solution)

            # Combine all metrics and update progress tracker
            all_metrics = {**retrieval_results, **qa_results}
            progress_tracker.update_sample(sample_idx, all_metrics, query_solution)

        # Finish progress tracking and get final results
        final_results = progress_tracker.finish_progress()
        final_metrics = final_results["overall_metrics"]
        num_processed = final_results["num_samples"]
    # Save final results (final results display is handled by progress tracker)
    samples_dict = safe_serialize_query_solutions(query_solutions)
    rag_results = {
        "dataset": args.dataset,
        "llm_name": args.llm_name,
        "embedding_name": args.embedding_name,
        "samples": samples_dict,
        "overall_metrics": final_metrics,
        "total_samples": num_processed,
        "total_corpus_facts": len(corpus_facts),
        "config": {
            "qa_top_k": args.qa_top_k,
            "parallel_processing": args.parallel,
            "num_workers": args.num_workers if args.parallel else 1,
        },
    }

    results_file = f"{rag.working_dir}/rag_results.json"
    success = safe_dump_json(rag_results, results_file)
    if not success:
        print(f"Warning: Had to use fallback serialization for {results_file}")

    print(f"\nResults saved to: {results_file}")
