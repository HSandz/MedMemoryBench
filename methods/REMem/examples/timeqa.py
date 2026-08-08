import argparse
import json
import os
from collections import defaultdict

from tqdm import tqdm

from remem.remem import ReMem
from remem.utils.config_utils import BaseConfig


def load_timeqa_dataset(file_path):
    """Load TimeQA dataset from JSON file, handling both JSON array and JSONL formats."""
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read().strip()

        # Try to load as a single JSON array first
        try:
            data = json.loads(content)
            if isinstance(data, list):
                return data
            else:
                return [data]  # Single object
        except json.JSONDecodeError:
            # Try to load as JSONL (one JSON object per line)
            data = []
            for line in content.split("\n"):
                line = line.strip()
                if line:
                    try:
                        item = json.loads(line)
                        data.append(item)
                    except json.JSONDecodeError as e:
                        print(f"Warning: Could not parse line: {line[:100]}... Error: {e}")
                        continue
            return data


def process_timeqa_paragraphs(timeqa_sample):
    """
    Process TimeQA sample to extract paragraphs in a format suitable for ReMem.

    Args:
        timeqa_sample: A single TimeQA sample with 'paragraphs' field

    Returns:
        List of processed paragraphs for indexing
    """
    processed_paragraphs = []

    for paragraph in timeqa_sample.get("paragraphs", []):
        title = paragraph.get("title", "")
        text = paragraph.get("text", "")

        # Create a structured document format similar to LoCoMo
        doc = {
            "content": f"{title}: {text}".strip(),
            "title": title,
            "text": text,
            "metadata": {"source": "timeqa", "title": title},
        }
        processed_paragraphs.append(doc)

    return processed_paragraphs


def get_timeqa_corpus(dataset_samples):
    """
    Extract and deduplicate all paragraphs from TimeQA dataset samples.

    Args:
        dataset_samples: List of TimeQA samples

    Returns:
        List of unique documents for indexing
    """
    all_paragraphs = []
    seen_docs = set()

    for sample in dataset_samples:
        for paragraph in sample.get("paragraphs", []):
            title = paragraph.get("title", "")
            text = paragraph.get("text", "")

            # Create unique identifier based on title and text
            doc_id = f"{title}|||{text}"

            if doc_id not in seen_docs:
                seen_docs.add(doc_id)
                doc = {
                    "content": f"{title}\n{text}".strip(),
                    "title": title,
                    "text": text,
                }
                all_paragraphs.append(doc)

    return all_paragraphs


def extract_gold_docs_for_timeqa(timeqa_sample):
    """
    Extract gold documents for a TimeQA sample.
    For TimeQA, we use all paragraphs as potential gold documents.
    """
    gold_docs = []
    for paragraph in timeqa_sample.get("paragraphs", []):
        title = paragraph.get("title", "")
        text = paragraph.get("text", "")
        content = f"{title}: {text}".strip()
        gold_docs.append(content)

    return gold_docs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ReMem on TimeQA dataset")
    parser.add_argument("--llm_base_url", type=str, default="https://api.openai.com/v1", help="LLM base URL")
    parser.add_argument("--llm_name", type=str, default="gpt-4o-mini", help="LLM name")
    parser.add_argument(
        "--dataset_file", type=str, default="reproduce/dataset/timeqa/dev.easy.json", help="Path to TimeQA dataset file"
    )
    parser.add_argument("--embedding_name", type=str, default="nvidia/NV-Embed-v2", help="embedding model name")
    parser.add_argument("-fi", "--force_index_from_scratch", action="store_true")
    parser.add_argument("-fo", "--force_openie_from_scratch", action="store_true")
    parser.add_argument(
        "--llm_infer_mode",
        choices=["online", "offline"],
        default="online",
        help="LLM inference mode, offline uses VLLM offline batch mode",
    )
    parser.add_argument("--use_azure", action="store_true", help="Use Azure for OpenAI")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit number of samples to process")
    parser.add_argument("--extract_format", type=str)
    parser.add_argument("--extract_method", type=str, default="episodic_gist")
    parser.add_argument("--qa_top_k", type=int, default=5)
    args = parser.parse_args()

    force_index_from_scratch = args.force_index_from_scratch
    force_openie_from_scratch = args.force_openie_from_scratch

    # Load TimeQA dataset
    print(f"Loading TimeQA dataset from {args.dataset_file}")
    dataset_samples = load_timeqa_dataset(args.dataset_file)

    if args.max_samples:
        dataset_samples = dataset_samples[: args.max_samples]

    print(f"Loaded {len(dataset_samples)} TimeQA samples")

    llm_base_url = args.llm_base_url
    llm_name = args.llm_name
    llm_label = args.llm_name.replace("/", "_") if llm_name is not None else "None"
    embedding_label = args.embedding_name.replace("/", "_") if args.embedding_name is not None else "None"

    # Create corpus from all samples for indexing
    print("Creating corpus from all TimeQA paragraphs...")
    corpus_docs = get_timeqa_corpus(dataset_samples)
    corpus_strings = [doc["content"] for doc in corpus_docs]
    print(f"Created corpus with {len(corpus_docs)} unique documents")

    config = BaseConfig(
        llm_base_url=llm_base_url,
        llm_name=llm_name,
        dataset="timeqa",
        embedding_model_name=args.embedding_name,
        force_index_from_scratch=force_index_from_scratch,
        force_openie_from_scratch=force_openie_from_scratch,
        rerank_dspy_file_path="src/remem/prompts/dspy_prompts/filter_llama3.3-70B-Instruct.json",
        retrieval_top_k=200,
        linking_top_k=5,
        qa_top_k=args.qa_top_k,
        do_eval_retrieval=True,
        do_eval_qa=True,
        graph_type="facts_and_sim_passage_node_unidirectional",
        embedding_batch_size=1,
        max_new_tokens=None,
        corpus_len=None,
        llm_infer_mode=args.llm_infer_mode,
        preprocess_chunk_func="none",
        use_azure=args.use_azure,
        extract_format=args.extract_format,
        extract_method=args.extract_method,
        qa_passage_prefix="",
        qa_prompt_template="rag_qa_unified",  # Use temporal-specific template instead of generic
    )

    if args.llm_infer_mode == "offline":
        from remem.llm.vllm_offline import VLLMOffline

        llm_client = VLLMOffline(
            config,
            model_name=args.llm_name,
            cache_dir="outputs/timeqa/llm_cache",
        )
    else:
        llm_client = None

    from remem.embedding_model import _get_embedding_client

    embedding_model = _get_embedding_client(
        global_config=config, embedding_model_name=args.embedding_name, openai_style_server=True
    )

    # Setup ReMem
    dataset_name = "timeqa"
    config.__post_init__()

    working_dir = os.path.join(config.save_dir, f"{dataset_name}_{llm_label}_{embedding_label}")
    rag = ReMem(global_config=config, working_dir=working_dir, llm=llm_client)
    rag.set_embedding_model(embedding_model)

    # Index the corpus once
    print("Indexing TimeQA corpus...")
    rag.index(corpus_strings)

    # Process each sample
    cumulative_metrics = defaultdict(float)
    selected_metrics = ("retrieval_recall", "qa_f1", "qa_em")

    total_metrics = defaultdict(float)
    num_sample = 0
    all_results = []

    for sample_idx, sample in tqdm(enumerate(dataset_samples), total=len(dataset_samples)):
        question = sample["question"]
        targets = sample["targets"]  # List of expected answers
        sample_id = sample.get("idx", f"sample_{sample_idx}")

        print(f"\nProcessing sample {sample_idx}: {sample_id}")
        print(f"Question: {question}")
        print(f"Targets: {targets}")

        # Extract gold documents for this sample
        gold_docs = [extract_gold_docs_for_timeqa(sample)]
        gold_answers = [targets]  # List of lists format expected by evaluator
        questions = [question]
        question_metadata = [{"sample_id": sample_id, "idx": sample.get("idx", "")}]

        # Get evaluators
        qa_evaluators, retrieval_evaluators = rag.get_evaluators(gold_answers, gold_docs, selected_metrics)

        # Run RAG for this question
        (query_solutions, all_response_message, all_metadata, sample_retrieval_metrics, sample_qa_metrics) = (
            rag.rag_for_qa(questions, gold_docs, gold_answers, selected_metrics, question_metadata=question_metadata)
        )

        num_sample += 1

        # Accumulate metrics
        for key in sample_retrieval_metrics:
            total_metrics[key] += sample_retrieval_metrics[key]
        for key in sample_qa_metrics:
            total_metrics[key] += sample_qa_metrics[key]

        # Store results
        sample_result = {
            "sample_id": sample_id,
            "question": question,
            "targets": targets,
            "predicted_answer": query_solutions[0].final_answer if query_solutions else "",
            "retrieval_metrics": sample_retrieval_metrics,
            "qa_metrics": sample_qa_metrics,
        }
        all_results.append(sample_result)

        # Print current running averages
        print(f"Current metrics after {num_sample} samples:")
        for key in sample_retrieval_metrics:
            avg_metric = total_metrics[key] / num_sample
            print(f"  {key}: {round(avg_metric, 4)}")
        for key in sample_qa_metrics:
            avg_metric = total_metrics[key] / num_sample
            print(f"  {key}: {round(avg_metric, 4)}")

    # Final results
    print(f"\n=== Final Results on {num_sample} TimeQA samples ===")
    final_metrics = {}
    for key in total_metrics:
        final_metrics[key] = round(total_metrics[key] / num_sample, 4)
        print(f"{key}: {final_metrics[key]}")

    # Save results
    results_file = f"{rag.working_dir}/timeqa_results.json"
    with open(results_file, "w") as f:
        output = {
            "dataset_file": args.dataset_file,
            "num_samples": num_sample,
            "final_metrics": final_metrics,
            "individual_results": all_results,
            "config": {
                "llm_name": args.llm_name,
                "embedding_name": args.embedding_name,
                "qa_top_k": args.qa_top_k,
                "extract_method": args.extract_method,
            },
        }
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {results_file}")
