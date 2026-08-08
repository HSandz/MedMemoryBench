"""
MenatQA RAG Evaluation Script

This script evaluates ReMem on the MenatQA dataset.

Dataset structure:
- menatqa_processed.json: Contains QA pairs with context passages
- menatqa_processed_corpus.json: Contains all available documents

Usage:
    python menatqa.py [options]
"""

import argparse
import json
import os

from remem.remem import ReMem
from remem.utils.config_utils import BaseConfig
from remem.utils.misc_utils import safe_dump_json, safe_serialize_query_solutions
from remem.utils.results_utils import check_and_load_existing_results, get_inference_type, print_evaluation_results


def load_menatqa_corpus(corpus_file_path):
    """Load all documents from the MenatQA corpus file"""
    with open(corpus_file_path, "r", encoding="utf-8") as f:
        corpus = json.load(f)

    return [
        {
            "content": f"{doc['title']}\n{doc['text']}".strip(),
            "title": doc["title"],
            "text": doc["text"],
        }
        for doc in corpus
    ]


def process_menatqa_samples(samples):
    """Process MenatQA samples and extract QA data"""
    questions, gold_answers, gold_docs, question_metadata = [], [], [], []

    for sample in samples:
        questions.append(sample["question"])
        gold_answers.append([sample["answer"]])

        # Prepare gold documents from context passages
        sample_gold_docs = []
        sample_gold_docs.append(sample["annotated_para"])
        gold_docs.append(sample_gold_docs)

        sample_contexts = []
        for passage in sample["context_passages"]:
            sample_contexts.append(f"{passage['title']}\n{passage['text']}".strip())

        # Add metadata
        question_metadata.append(
            {
                "type": sample.get("type", "unknown"),
                "time_scope": sample.get("time_scope", []),
                "original_id": sample.get("original_id", None),
                "sample_id": sample.get("id", "unknown"),
            }
        )

    return questions, gold_answers, gold_docs, question_metadata


def main():
    parser = argparse.ArgumentParser(description="ReMem evaluation on MenatQA dataset")
    parser.add_argument("--llm_base_url", type=str, default="https://api.openai.com/v1", help="LLM base URL")
    parser.add_argument("--llm_name", type=str, default="gpt-4o-mini", help="LLM name")
    parser.add_argument("--dataset", type=str, default="menatqa")
    parser.add_argument("--embedding_name", type=str, default="nvidia/NV-Embed-v2", help="embedding model name")
    parser.add_argument("-fi", "--force_index_from_scratch", action="store_true")
    parser.add_argument("-fo", "--force_openie_from_scratch", action="store_true")
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("-fr", "--force_rag", action="store_true", help="Force rerun RAG even if results exist")
    parser.add_argument(
        "--llm_infer_mode",
        choices=["online", "offline"],
        default="online",
        help="OpenIE mode, offline denotes using VLLM offline batch mode for indexing, while online denotes",
    )
    parser.add_argument("--use_azure", action="store_true", help="Use Azure for OPENAI")
    parser.add_argument("--indices", type=str, help="JSON file containing indices of samples to process")
    parser.add_argument("--extract_format", type=str)
    parser.add_argument("--extract_method", type=str, default="episodic_gist")
    parser.add_argument("--qa_top_k", type=int, default=10)
    parser.add_argument("--max_samples", type=int, help="Maximum number of samples to process")
    # Agent configuration parameters
    parser.add_argument(
        "--agent_fixed_tools", action="store_true", help="Use simple agent with only semantic_retrieve + output_answer"
    )
    parser.add_argument(
        "--agent_max_steps",
        type=int,
        default=5,
        help="Maximum reasoning steps for agent (1=retrieve only, 2=retrieve+answer, >2=full reasoning)",
    )
    args = parser.parse_args()
    print(f"Arguments: {args}")

    # Load the dataset
    dataset_file = "reproduce/dataset/menatqa/menatqa_temporal.json"
    corpus_file = "reproduce/dataset/menatqa/menatqa_temporal_corpus.json"

    print(f"Loading dataset from {dataset_file}")
    with open(dataset_file, "r", encoding="utf-8") as f:
        samples = json.load(f)

    print(f"Loading corpus from {corpus_file}")
    corpus_documents = load_menatqa_corpus(corpus_file)
    corpus_strings = [doc["content"] for doc in corpus_documents]

    print(f"Loaded {len(samples)} QA samples")
    print(f"Loaded {len(corpus_documents)} corpus documents")

    # Filter and limit samples
    if args.indices is not None:
        with open(args.indices, "r") as f:
            indices = json.load(f)
        samples = [samples[i] for i in indices]
        print(f"Filtered to {len(samples)} samples using provided indices")

    if args.max_samples is not None:
        samples = samples[: args.max_samples]
        print(f"Limited to {len(samples)} samples")

    # Create labels for directory naming
    llm_label = args.llm_name.replace("/", "_") if args.llm_name else "None"
    embedding_label = args.embedding_name.replace("/", "_") if args.embedding_name else "None"

    # Configuration adapted for MenatQA dataset
    config = BaseConfig(
        llm_base_url=args.llm_base_url,
        llm_name=args.llm_name,
        dataset=args.dataset,
        embedding_model_name=args.embedding_name,
        force_index_from_scratch=args.force_index_from_scratch,
        force_openie_from_scratch=args.force_openie_from_scratch,
        rerank_dspy_file_path="src/remem/prompts/dspy_prompts/filter_llama3.3-70B-Instruct.json",
        retrieval_top_k=200,
        linking_top_k=5,
        qa_top_k=args.qa_top_k,
        do_eval_retrieval=True,
        do_eval_qa=True,
        graph_type="facts_and_sim_passage_node_unidirectional",
        embedding_batch_size=32,
        max_new_tokens=None,
        corpus_len=None,
        llm_infer_mode=args.llm_infer_mode,
        preprocess_chunk_func="none",  # Use passage-based chunking for menatqa
        use_azure=args.use_azure,
        extract_format=args.extract_format,
        extract_method=args.extract_method,
        qa_passage_prefix="",
        qa_prompt_template="rag_qa_unified",
        agent_fixed_tools=args.agent_fixed_tools,
        agent_max_steps=args.agent_max_steps,
    )

    # Initialize LLM client
    llm_client = None
    if args.llm_infer_mode == "offline":
        from remem.llm.vllm_offline import VLLMOffline

        llm_client = VLLMOffline(
            config,
            model_name=args.llm_name,
            cache_dir=f"outputs/{args.dataset}/llm_cache",
        )

    # Initialize embedding model and ReMem
    from remem.embedding_model import _get_embedding_client

    embedding_model = _get_embedding_client(
        global_config=config, embedding_model_name=args.embedding_name, openai_style_server=True
    )

    config.__post_init__()
    working_dir = os.path.join(config.save_dir, f"{args.dataset}_{llm_label}_{embedding_label}")
    rag = ReMem(global_config=config, working_dir=working_dir, llm=llm_client)
    rag.set_embedding_model(embedding_model)

    # Index the corpus documents and prepare QA data
    print("Indexing corpus documents...")
    rag.index(corpus_strings)

    print("Preparing QA data...")
    questions, gold_answers, gold_docs, question_metadata = process_menatqa_samples(samples)
    print(f"Prepared {len(questions)} questions")

    # Check for existing results
    inference_type = get_inference_type(args.agent_fixed_tools, args.agent_max_steps)

    # Try to load existing results
    existing_metrics = check_and_load_existing_results(
        working_dir=rag.working_dir,
        inference_type=inference_type,
        num_questions=len(questions),
        agent_fixed_tools=args.agent_fixed_tools,
        agent_max_steps=args.agent_max_steps,
        dataset_name=args.dataset,
        llm_name=args.llm_name,
        embedding_name=args.embedding_name,
        force_rag=args.force_rag,
    )

    if existing_metrics is not None:
        return

    # Run RAG evaluation
    print("Running RAG evaluation...")
    print(f"Processing {len(questions)} questions...")

    selected_metrics = ("retrieval_recall", "qa_f1", "qa_em", "qa_mem0_llm_judge")
    (query_solutions, all_response_message, all_metadata, retrieval_metrics, qa_metrics) = rag.rag_for_qa(
        questions,
        gold_docs,
        gold_answers,
        selected_metrics,
        question_metadata,
        parallel=args.parallel,
        evaluate_per_sample=True,
        save_per_sample=True,
        inference_type=inference_type,
        force_rag=args.force_rag,
    )

    # Calculate and print final results
    # Handle both dictionary and list return types for metrics (defensive programming)
    total_metrics = {}
    if isinstance(retrieval_metrics, dict):
        total_metrics.update(retrieval_metrics)
    if isinstance(qa_metrics, dict):
        total_metrics.update(qa_metrics)

    print_evaluation_results(
        dataset_name=args.dataset,
        num_questions=len(questions),
        llm_name=args.llm_name,
        embedding_name=args.embedding_name,
        metrics=total_metrics,
        title="Final Results",
    )

    # Save results
    results_file = f"{rag.working_dir}/rag_results_{inference_type}.json"
    print(f"\nSaving results to {results_file}")
    samples_dict = safe_serialize_query_solutions(query_solutions)
    rag_results = {
        "dataset": args.dataset,
        "llm_name": args.llm_name,
        "embedding_name": args.embedding_name,
        "total_questions": len(questions),
        "samples": samples_dict,
        "overall_metrics": total_metrics,
        "config": {
            "extract_method": args.extract_method,
            "agent_fixed_tools": args.agent_fixed_tools,
            "agent_max_steps": args.agent_max_steps,
        },
    }

    success = safe_dump_json(rag_results, results_file)
    if not success:
        print(f"Warning: Had to use fallback serialization for {results_file}")

    print("Evaluation completed!")


if __name__ == "__main__":
    main()
