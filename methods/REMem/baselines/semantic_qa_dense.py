#!/usr/bin/env python3
import argparse
import json
import logging
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from remem.remem import ReMem
from remem.utils.config_utils import BaseConfig
from remem.utils.evaluation_progress import EvaluationConfig, ThreadSafeLogger, ThreadSafeMetrics
from remem.utils.misc_utils import QuerySolution, safe_dump_json, safe_serialize_query_solutions


def process_single_sample(
    sample_idx,
    question,
    gold_answer,
    gold_paragraphs,
    metadata,
    rag,
    candidate_docs,
    selected_metrics,
    progress_tracker,
):
    """Process a single sample - designed to be called in parallel"""
    try:
        # Get evaluators for this question
        qa_evaluators, retrieval_evaluators = rag.get_evaluators([gold_answer], gold_paragraphs, selected_metrics)

        # Perform dense passage retrieval
        sorted_chunk_ids, sorted_chunk_scores = rag.dense_passage_retrieval(
            query=question, candidate_docs=candidate_docs
        )

        # Get retrieved content and metadata
        retrieved_chunk_contents = [rag.chunk_contents[chunk_id] for chunk_id in sorted_chunk_ids]
        retrieved_chunk_metadata = [rag.chunk_metadata[chunk_id] for chunk_id in sorted_chunk_ids]

        # Create query solution
        query_solution = QuerySolution(
            question,
            retrieved_chunk_contents,
            metadata,
            sorted_chunk_scores,
            retrieved_chunk_metadata,
            None,
            [gold_answer],
            gold_paragraphs,
            None,
            {},
            None,
        )

        # Generate answer using QA module
        queries_solution, all_response_message, all_metadata = rag.qa([query_solution])
        query_solution_result = queries_solution[0] if queries_solution else query_solution

        # Evaluate retrieval and QA
        retrieval_results = rag.evaluate_retrieval([gold_paragraphs], queries_solution, retrieval_evaluators)
        qa_results = rag.evaluate_qa([[gold_answer]], qa_evaluators, queries_solution, [metadata])

        # Combine all metrics
        all_metrics = {**retrieval_results, **qa_results}

        # Update progress tracker
        if progress_tracker:
            progress_tracker.update_sample(sample_idx, all_metrics, query_solution_result)

        return sample_idx, all_metrics, query_solution_result

    except Exception as e:
        # Update progress tracker for failed sample
        if progress_tracker:
            progress_tracker.update_sample(sample_idx, {}, failed=True, error_msg=str(e))
        return sample_idx, {}, None


def main():
    parser = argparse.ArgumentParser(description="Dense retrieval baseline for semantic QA datasets")
    parser.add_argument("--dataset", type=str, default="musique")
    parser.add_argument("--llm_base_url", type=str, default="https://api.openai.com/v1", help="LLM base URL")
    parser.add_argument("--llm_name", type=str, default="gpt-4.1-mini-2025-04-14", help="LLM name")
    parser.add_argument("--embedding_name", type=str, default="nvidia/NV-Embed-v2", help="Embedding model name")
    parser.add_argument("--llm_infer_mode", choices=["online", "offline"], default="online", help="LLM inference mode")
    parser.add_argument("--use_azure", action="store_true", help="Use Azure for OpenAI")
    parser.add_argument("--indices", type=str, help="JSON file with sample indices to process")
    parser.add_argument("--qa_top_k", type=int, default=10, help="Top-k for QA")
    parser.add_argument("--max_model_len", type=int, default=4096, help="Maximum model length")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of samples to process")
    parser.add_argument("--start", type=int, default=0, help="Start index")
    parser.add_argument("--end", type=int, default=None, help="End index")
    parser.add_argument("--parallel", action="store_true", help="Enable parallel processing")
    parser.add_argument("--num_workers", type=int, default=5, help="Number of parallel workers")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    args = parser.parse_args()

    print(f"Arguments: {args}")

    # Setup logging
    if args.verbose and not args.parallel:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    else:
        logging.basicConfig(level=logging.WARNING, format="%(levelname)s - %(message)s")

    # Load the specified dataset
    dataset_path = f"reproduce/dataset/{args.dataset}/{args.dataset}.json"
    corpus_path = f"reproduce/dataset/{args.dataset}/{args.dataset}_corpus.json"

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")
    if not os.path.exists(corpus_path):
        raise FileNotFoundError(f"Corpus not found at {corpus_path}")

    with open(dataset_path, "r") as f:
        samples = json.load(f)

    corpus = json.load(open(corpus_path))
    corpus_texts = [f"{item['title']}\n{item['text']}" for item in corpus]

    logging.info(f"Loaded {len(samples)} queries from {args.dataset} dataset")
    logging.info(f"Loaded {len(corpus_texts)} passages from {args.dataset} corpus")

    # Filter samples if indices are provided
    if args.indices:
        indices = json.load(open(args.indices))
        samples = [samples[i] for i in indices if i < len(samples)]
        logging.info(f"Filtered to {len(samples)} samples using provided indices")

    # Limit number of samples if specified
    if args.num_samples:
        samples = samples[: args.num_samples]
        logging.info(f"Limited to {args.num_samples} samples")

    # Apply start/end filters
    if args.start is not None or args.end is not None:
        start_idx = args.start if args.start is not None else 0
        end_idx = args.end if args.end is not None else len(samples)
        samples = samples[start_idx:end_idx]
        logging.info(f"Filtered to samples {start_idx}:{end_idx} ({len(samples)} samples)")

    # Setup model configuration
    llm_base_url = args.llm_base_url
    llm_name = args.llm_name
    embedding_label = args.embedding_name.replace("/", "_")

    if args.use_azure:
        llm_base_url = None  # prioritize Azure ENDPOINT

    # Handle LLaMA models
    extract_format = None
    if "llama" in llm_name.lower():
        extract_format = "json_schema"

    config = BaseConfig(
        llm_base_url=llm_base_url,
        llm_name=llm_name,
        dataset=args.dataset,
        embedding_model_name=args.embedding_name,
        force_index_from_scratch=False,  # Set to False to reuse previously stored index and embeddings
        force_openie_from_scratch=False,
        rerank_dspy_file_path="src/remem/prompts/dspy_prompts/filter_llama3.3-70B-Instruct.json",
        retrieval_top_k=200,
        linking_top_k=5,
        qa_top_k=args.qa_top_k,
        do_eval_retrieval=True,
        do_eval_qa=True,
        graph_type="dpr_only",  # Dense passage retrieval only
        embedding_batch_size=16,
        max_new_tokens=None,
        corpus_len=None,
        llm_infer_mode=args.llm_infer_mode,
        preprocess_chunk_func="none",  # Use simple chunking for corpus facts
        use_azure=args.use_azure,
        extract_format=extract_format,
        qa_passage_prefix="",
        qa_prompt_template="rag_qa_unified",
        max_model_len=args.max_model_len,
    )

    # Setup LLM model
    if args.llm_infer_mode == "offline":
        from remem.llm.vllm_offline import VLLMOffline

        llm_client = VLLMOffline(
            config,
            model_name=args.llm_name,
            cache_dir=f"outputs/{args.dataset}/llm_cache",
        )
        llm_label = args.llm_name.replace("/", "_")
    else:  # Online mode
        llm_client = None
        llm_label = args.llm_name.replace("/", "_")

    # Setup embedding model
    from remem.embedding_model import _get_embedding_client

    embedding_model = _get_embedding_client(global_config=config, embedding_model_name=args.embedding_name)

    # Create ReMem instance
    print("Creating ReMem instance for dense retrieval...")
    working_dir = os.path.join(config.save_dir, f"{args.dataset}_{llm_label}_{embedding_label}_dense")
    rag = ReMem(global_config=config, working_dir=working_dir, llm=llm_client)
    rag.set_embedding_model(embedding_model)

    # PRE-ENCODE CORPUS DOCUMENTS ONCE to avoid duplicate encoding in parallel processes
    print(f"Pre-encoding {len(corpus_texts)} corpus documents...")
    rag.add_chunk_and_embeddings(corpus_texts)
    print("Pre-encoding complete. Ready for processing.")

    # Process samples and extract required data
    questions = []
    gold_answers = []
    gold_docs = []
    question_metadata = []

    for sample_idx, sample in enumerate(samples):
        question = sample["question"]
        gold_answer = sample["answer"]
        from examples.semantic_qa import extract_supporting_paragraphs

        gold_paragraphs = extract_supporting_paragraphs(sample)
        metadata = {
            "id": sample.get("id", f"sample_{sample_idx}"),
            "sample_idx": sample_idx,
            "question_decomposition": sample.get("question_decomposition", []),
        }

        questions.append(question)
        gold_answers.append(gold_answer)
        gold_docs.append(gold_paragraphs)
        question_metadata.append(metadata)

    print(f"Processed {len(questions)} questions")

    # Metrics for evaluation
    selected_metrics = ("qa_em", "qa_f1", "qa_bleu1", "qa_mem0_llm_judge", "retrieval_recall")

    if args.parallel:
        print(f"Running dense retrieval baseline with parallel processing ({args.num_workers} workers)...")

        # Initialize thread-safe components
        eval_config = EvaluationConfig(
            processing_mode="parallel",
            num_workers=args.num_workers,
            metric_precision=4,
            enable_thread_safe_logging=True,
        )
        safe_metrics = ThreadSafeMetrics(eval_config)
        safe_logger = ThreadSafeLogger(eval_config)

        start_time = time.time()

        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            # Submit all tasks
            future_to_idx = {
                executor.submit(
                    process_single_sample,
                    sample_idx,
                    question,
                    gold_answer,
                    gold_doc,
                    metadata,
                    rag,
                    corpus_texts,
                    selected_metrics,
                    None,  # Pass None instead of safe_logger
                ): sample_idx
                for sample_idx, (question, gold_answer, gold_doc, metadata) in enumerate(
                    zip(questions, gold_answers, gold_docs, question_metadata)
                )
            }

            # Process completed tasks with progress bar
            with tqdm(total=len(questions), desc="Processing samples") as pbar:
                for future in as_completed(future_to_idx):
                    sample_idx, metrics_dict, query_solution = future.result()

                    if query_solution is not None and metrics_dict:
                        safe_metrics.update(sample_idx, metrics_dict, result=query_solution)
                    else:
                        safe_metrics.update(sample_idx, {}, failed=True, error_msg="Processing failed")

                    pbar.update(1)

                elapsed_time = time.time() - start_time
        print(f"Parallel processing completed in {elapsed_time:.2f} seconds")

        # Get final results
        print("Getting final results from safe_metrics...", flush=True)
        results_dict = safe_metrics.get_final_results()
        print("Got results_dict, extracting metrics...", flush=True)
        final_metrics = results_dict["overall_metrics"]
        print("Extracted overall_metrics, extracting query solutions...", flush=True)
        query_solutions = results_dict["results"]
        print(f"Extracted {len(query_solutions)} query solutions, extracting sample count...", flush=True)
        num_sample = results_dict["num_samples"]
        print(f"Results extraction completed. Total samples: {num_sample}", flush=True)

    else:
        print("Running dense retrieval baseline (sequential processing)...")

        cumulative_metrics = defaultdict(float)
        num_sample = 0
        query_solutions = []

        for sample_idx, (question, gold_answer, gold_paragraphs, metadata) in enumerate(
            tqdm(zip(questions, gold_answers, gold_docs, question_metadata), total=len(questions))
        ):
            print(f"\n--- Processing Sample {sample_idx} ---")
            print(f"Question: {question}")
            print(f"Gold answer: {gold_answer}")
            print(f"Number of gold paragraphs: {len(gold_paragraphs)}")
            print()

            # Use shared core function (no thread pool needed for sequential processing)
            sample_idx_result, metrics_dict, query_solution = process_single_sample(
                sample_idx,
                question,
                gold_answer,
                gold_paragraphs,
                metadata,
                rag,
                corpus_texts,
                selected_metrics,
                None,  # No progress tracker needed
            )

            if query_solution is not None and metrics_dict:
                query_solutions.append(query_solution)
                num_sample += 1

                # Update cumulative metrics
                for key, value in metrics_dict.items():
                    cumulative_metrics[key] += value

                # Print current average metrics
                print("Current metrics:")
                for key, value in metrics_dict.items():
                    print(f"  {key}: {round(cumulative_metrics[key] / num_sample, 4)}")
                print()
            else:
                print(f"Error processing sample {sample_idx}")

        # Calculate final average metrics
        final_metrics = {key: round(cumulative_metrics[key] / num_sample, 4) for key in cumulative_metrics}

    print("\n=== Final Results ===")
    print(f"Dataset: {args.dataset}")
    print(f"Total questions: {num_sample}")
    print(f"LLM: {args.llm_name}")
    print(f"Embedding Model: {args.embedding_name}")
    print("Method: Dense Retrieval Baseline")
    if args.parallel:
        print(f"Processing Mode: Parallel ({args.num_workers} workers)")
    else:
        print("Processing Mode: Sequential")
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

    # Save results
    output_path = f"{rag.working_dir}/rag_results_dense_{llm_label}_{embedding_label}.json"
    print(f"\nSaving results to {output_path}")

    print(f"Serializing {len(query_solutions)} query solutions...", flush=True)

    sys.stdout.flush()
    samples_dict = safe_serialize_query_solutions(query_solutions)
    print("Query solutions serialized successfully.", flush=True)

    print("Creating results dictionary...", flush=True)
    rag_results = {
        "dataset": args.dataset,
        "llm_name": args.llm_name,
        "embedding_name": args.embedding_name,
        "method": "dense_retrieval_baseline",
        "total_questions": num_sample,
        "total_corpus_facts": len(corpus_texts),
        "samples": samples_dict,
        "overall_metrics": final_metrics,
        "config": {
            "qa_top_k": args.qa_top_k,
            "graph_type": config.graph_type,
            "preprocess_chunk_func": config.preprocess_chunk_func,
            "parallel_processing": args.parallel,
            "num_workers": args.num_workers if args.parallel else 1,
        },
    }
    print("Results dictionary created.", flush=True)

    print("Writing JSON file...", flush=True)
    success = safe_dump_json(rag_results, output_path)
    print("JSON file writing completed.", flush=True)
    if not success:
        print(f"Warning: Had to use fallback serialization for {output_path}")

    processing_mode = "parallel" if args.parallel else "sequential"
    print(f"Dense retrieval baseline evaluation completed using {processing_mode} processing!")


if __name__ == "__main__":
    main()
