#!/usr/bin/env python3
"""
Script to evaluate LLMs on the ToT semantic dataset using OpenAI models with Azure support.
This script uses the provided prompts from the dataset to directly call LLM for evaluation.
Features multi-threading for faster processing while maintaining result order and real-time accuracy tracking.
"""


import argparse
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from typing import Any, Dict, List

from tot_semantic_utils import calculate_question_type_stats, print_evaluation_summary
from tqdm import tqdm

from remem.embedding_model import _get_embedding_client
from remem.llm.openai_gpt import CacheOpenAI
from remem.utils.config_utils import BaseConfig
from remem.utils.retrieval_api import BM25SparseRetriever


def load_dataset(dataset_path: str) -> List[Dict[str, Any]]:
    """Load the ToT semantic dataset."""
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    with open(dataset_path, "r") as f:
        samples = json.load(f)

    logging.info(f"Loaded {len(samples)} samples from ToT semantic dataset")
    return samples


class EmbeddingRetriever:
    """Custom retriever using ReMem embedding model."""

    def __init__(self, corpus: List[str], embedding_model):
        self.corpus = corpus
        self.embedding_model = embedding_model
        self.corpus_embeddings = None
        self._preprocess()

    def _preprocess(self):
        """Precompute embeddings for the corpus."""
        try:
            # Use the embedding model to encode all corpus texts
            embeddings = self.embedding_model.batch_encode(self.corpus, batch_size=1)
            # Convert to numpy for easier manipulation
            if hasattr(embeddings, "cpu"):
                self.corpus_embeddings = embeddings.cpu().numpy()
            elif hasattr(embeddings, "numpy"):
                self.corpus_embeddings = embeddings.numpy()
            else:
                self.corpus_embeddings = embeddings
        except Exception as e:
            logging.error(f"Error preprocessing corpus embeddings: {e}")
            # Fallback to None, will be handled in get_top_k_sentences
            self.corpus_embeddings = None

    def get_top_k_sentences(self, query: str, k: int = 10) -> List[str]:
        """Get top-k most similar sentences to the query."""
        if self.corpus_embeddings is None:
            logging.warning("Corpus embeddings not available, returning first k sentences")
            return self.corpus[:k]

        try:
            # Encode the query
            query_embedding = self.embedding_model.batch_encode([query])
            if hasattr(query_embedding, "cpu"):
                query_embedding = query_embedding.cpu().numpy()
            elif hasattr(query_embedding, "numpy"):
                query_embedding = query_embedding.numpy()

            # Calculate similarities (cosine similarity)
            import numpy as np

            query_vec = query_embedding[0]  # Get first (and only) query embedding

            # Normalize vectors for cosine similarity
            query_norm = np.linalg.norm(query_vec)
            corpus_norms = np.linalg.norm(self.corpus_embeddings, axis=1)

            # Avoid division by zero
            if query_norm == 0:
                similarities = np.zeros(len(self.corpus))
            else:
                similarities = np.dot(self.corpus_embeddings, query_vec) / (corpus_norms * query_norm)
                # Handle any NaN values
                similarities = np.nan_to_num(similarities, nan=0.0)

            # Get top-k indices
            top_k_indices = np.argsort(similarities)[::-1][:k]

            # Return corresponding sentences
            return [self.corpus[i] for i in top_k_indices]

        except Exception as e:
            logging.error(f"Error in embedding retrieval: {e}")
            # Fallback to first k sentences
            return self.corpus[:k]


def apply_fact_retrieval(
    prompt: str, question: str, top_k: int = 15, retriever_type: str = "bm25", embedding_model=None
) -> str:
    """
    Apply retrieval to select top-k most relevant facts from the prompt.

    Args:
        prompt: The original prompt containing facts and question
        question: The question to retrieve facts for
        top_k: Number of top facts to retrieve
        retriever_type: Type of retriever to use ("bm25", "embedding", etc.)
        embedding_model: The embedding model to use for embedding-based retrieval

    Returns:
        Reconstructed prompt with only the top-k most relevant facts
    """
    # Split the prompt into lines
    lines = prompt.strip().split("\n")

    # Find where the facts end and the question begins
    # Look for the line that starts with "Question:"
    fact_lines = []
    instruction_start_idx = -1

    for i, line in enumerate(lines):
        if len(line.strip()) == 0:
            continue
        if line.strip().startswith("Here is a set of temporal facts"):
            # Skip the first line which is just a header
            continue
        if line.strip().startswith("Answer the following question"):
            instruction_start_idx = i
            break
        elif line.strip():  # Non-empty line
            fact_lines.append(line.strip())

    if instruction_start_idx == -1:
        # If no "Question:" found, assume all lines are facts except the last few
        logging.warning("No instruction line found, using heuristic to separate facts")
        fact_lines = [line.strip() for line in lines if line.strip()]

    # If we have fewer facts than top_k, return the original prompt
    if len(fact_lines) <= top_k:
        return prompt

    try:
        # Initialize retriever based on type
        if retriever_type.lower() == "bm25":
            retriever = BM25SparseRetriever(fact_lines)
            # Retrieve top-k most relevant facts using indices to ensure proper string format
            top_indices = retriever.get_top_k_indices(question, k=top_k)
            top_facts = [fact_lines[i] for i in top_indices]
        elif retriever_type.lower() == "embedding":
            assert embedding_model is not None
            retriever = EmbeddingRetriever(fact_lines, embedding_model)
            top_facts = retriever.get_top_k_sentences(question, k=top_k)
        else:
            # For future extension: other retriever types can be added here
            logging.warning(f"Retriever type '{retriever_type}' not implemented, falling back to BM25")
            exit(1)

        # Reconstruct the prompt with only the top facts
        if instruction_start_idx != -1:
            # Use the original question and instructions part
            question_and_instructions = "\n".join(lines[instruction_start_idx:])
            reconstructed_prompt = (
                "Here is a set of temporal facts:\n" + "\n".join(top_facts) + "\n\n" + question_and_instructions
            )
        else:
            # Fallback: just add the question at the end
            reconstructed_prompt = (
                "Here is a set of temporal facts:\n" + "\n".join(top_facts) + "\n\nQuestion: " + question
            )

        return reconstructed_prompt

    except Exception as e:
        logging.error(f"Error in fact retrieval: {e}")
        return prompt  # Return original prompt if retrieval fails


def evaluate_sample(llm_client: CacheOpenAI, sample: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluate a single sample using the LLM.

    Args:
        llm_client: The LLM model to use for evaluation
        sample: The sample containing prompt, question, and label

    Returns:
        Dictionary containing evaluation results
    """
    prompt = sample["prompt"]
    question = sample["question"]
    label = sample["label"]

    # Create message for the LLM
    messages = [{"role": "user", "content": prompt}]

    try:
        # Get response from LLM
        response, metadata, cache_hit = llm_client.infer(messages=messages, response_format={"type": "json_object"})

        # Try to parse JSON response
        try:
            response_json = json.loads(response)
            predicted_answer = response_json.get("answer", "")
            explanation = response_json.get("explanation", "")
        except json.JSONDecodeError:
            # If JSON parsing fails, use raw response as answer
            predicted_answer = response.strip()
            explanation = "Failed to parse JSON response"

        # Check if prediction matches label
        if predicted_answer.startswith("E") and ", " in predicted_answer:
            from examples.tot_semantic import remove_spaces_from_prediction

            predicted_answer = remove_spaces_from_prediction(predicted_answer)
        exact_match = 1.0 if str(predicted_answer).strip() == str(label).strip() else 0.0

        result = {
            "question": question,
            "prompt": prompt,
            "label": label,
            "predicted_answer": predicted_answer,
            "explanation": explanation,
            "sample_metrics": {"ExactMatch": exact_match},
            "raw_response": response,
            "metadata": metadata,
        }

        return result

    except Exception as e:
        logging.error(f"Error evaluating sample: {e}")
        return {
            "question": question,
            "prompt": prompt,
            "label": label,
            "predicted_answer": "",
            "explanation": f"Error: {str(e)}",
            "sample_metrics": {"ExactMatch": 0.0},
            "raw_response": "",
            "metadata": {},
        }


def evaluate_sample_with_index(args_tuple):
    """
    Wrapper for evaluate_sample that includes the index for ordered results.

    Args:
        args_tuple: Tuple of (index, llm_model, sample, use_retrieval, top_k, retriever_type, embedding_model)

    Returns:
        Tuple of (index, result)
    """
    index, llm_model, sample, use_retrieval, top_k, retriever_type, embedding_model = args_tuple
    result = evaluate_sample_with_retrieval(llm_model, sample, use_retrieval, top_k, retriever_type, embedding_model)
    return index, result


def evaluate_sample_with_retrieval(
    llm_client: CacheOpenAI,
    sample: Dict[str, Any],
    use_retrieval: bool = False,
    top_k: int = 15,
    retriever_type: str = "bm25",
    embedding_model=None,
) -> Dict[str, Any]:
    """
    Evaluate a single sample using the LLM, optionally with fact retrieval.

    Args:
        llm_client: The LLM model to use for evaluation
        sample: The sample containing prompt, question, and label
        use_retrieval: Whether to use retrieval to select top-k facts
        top_k: Number of top facts to retrieve when using retrieval
        retriever_type: Type of retriever to use ("bm25", "embedding", etc.)
        embedding_model: The embedding model to use for embedding-based retrieval

    Returns:
        Dictionary containing evaluation results
    """
    original_prompt = sample["prompt"]
    question = sample["question"]
    label = sample["label"]

    # Apply retrieval if requested
    if use_retrieval:
        prompt = apply_fact_retrieval(original_prompt, question, top_k, retriever_type, embedding_model)
    else:
        prompt = original_prompt

    # Create message for the LLM
    messages = [{"role": "user", "content": prompt}]

    try:
        # Get response from LLM
        response, metadata, cache_hit = llm_client.infer(messages=messages, response_format={"type": "json_object"})

        # Try to parse JSON response
        try:
            response_json = json.loads(response)
            predicted_answer = response_json.get("answer", "")
            explanation = response_json.get("explanation", "")
        except json.JSONDecodeError:
            # If JSON parsing fails, use raw response as answer
            predicted_answer = response.strip()
            explanation = "Failed to parse JSON response"

        # Check if prediction matches label
        exact_match = 1.0 if str(predicted_answer).strip() == str(label).strip() else 0.0

        result = {
            "question": question,
            "prompt": prompt,
            "label": label,
            "predicted_answer": predicted_answer,
            "explanation": explanation,
            "sample_metrics": {"ExactMatch": exact_match},
            "metadata": metadata,
        }

        return result

    except Exception as e:
        logging.error(f"Error evaluating sample: {e}")
        return {
            "question": question,
            "prompt": prompt,
            "label": label,
            "predicted_answer": "",
            "explanation": f"Error: {str(e)}",
            "sample_metrics": {"ExactMatch": 0.0},
            "metadata": {},
        }


def main():
    parser = argparse.ArgumentParser(description="Evaluate LLMs on ToT Semantic Dataset")
    parser.add_argument("--llm_base_url", type=str, default="https://api.openai.com/v1", help="LLM base URL")
    parser.add_argument("--llm_name", type=str, default="gpt-4.1-mini", help="LLM name")
    parser.add_argument("--use_azure", action="store_true", help="Use Azure for OpenAI")
    parser.add_argument("--debug", action="store_true", help="Debug mode - process only 1 sample")
    parser.add_argument(
        "--cache_dir", type=str, default="outputs/tot_semantic/llm_cache", help="Cache directory for LLM responses"
    )
    parser.add_argument("--num_threads", type=int, default=4, help="Number of threads for parallel processing")
    parser.add_argument(
        "--retriever",
        type=str,
        help='`bm25` or embedding model name (e.g., "nvidia/NV-Embed-v2"). If not set, no retrieval is used.',
    )
    parser.add_argument(
        "--retrieval_top_k", type=int, default=10, help="Number of top facts to retrieve when using retrieval"
    )
    parser.add_argument("-fr", "--force_rag", action="store_true", help="Force rerun evaluation even if results exist")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    logging.info(f"Arguments: {args}")

    # Determine retrieval settings from --retriever argument
    use_retrieval = args.retriever is not None
    if use_retrieval:
        if args.retriever.lower() == "bm25":
            retriever_type = "bm25"
            embedding_name = None
            logging.info(f"Retrieval mode enabled: using BM25 to select top {args.retrieval_top_k} most relevant facts")
        else:
            retriever_type = "embedding"
            embedding_name = args.retriever
            logging.info(
                f"Retrieval mode enabled: using embedding-based retrieval to select top {args.retrieval_top_k} most relevant facts"
            )
            logging.info(f"Embedding model: {embedding_name}")
    else:
        retriever_type = None
        embedding_name = None
        logging.info("No retrieval - using full prompts")

    # Create cache directory
    os.makedirs(args.cache_dir, exist_ok=True)

    # Load the ToT semantic dataset
    dataset_path = "reproduce/dataset/tot/tot_semantic.json"
    samples = load_dataset(dataset_path)

    # For debug mode, use only 1 sample
    if args.debug:
        samples = samples[:1]
        logging.info("Debug mode: processing only 1 sample")

    # Initialize LLM model
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

    # Initialize embedding model if needed
    embedding_model = None
    if use_retrieval and retriever_type == "embedding":
        try:
            logging.info(f"Initializing embedding model: {embedding_name}")
            config = BaseConfig()
            embedding_model = _get_embedding_client(global_config=config, embedding_model_name=embedding_name)
            logging.info("Embedding model initialized successfully")
        except Exception as e:
            logging.error(f"Failed to initialize embedding model: {e}")
            logging.info("Falling back to BM25 retrieval")
            retriever_type = "bm25"

    llm_label = args.llm_name.replace("/", "_")
    retrieval_suffix = f"_{retriever_type}_k{args.retrieval_top_k}" if use_retrieval else ""
    output_path = f"outputs/tot_semantic/tot_semantic_{len(samples)}_{llm_label}{retrieval_suffix}.json"

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Check if results already exist and skip if not forcing rerun
    if os.path.exists(output_path) and not args.force_rag:
        logging.info(f"Results already exist at {output_path}")
        logging.info("Use --force_rag (-fr) to overwrite existing results")

        # Load and display existing results
        try:
            with open(output_path, "r") as f:
                existing_data = json.load(f)

            existing_accuracy = existing_data.get("accuracy", 0)
            existing_correct = existing_data.get("correct_predictions", 0)
            existing_total = existing_data.get("total_samples", 0)

            logging.info(f"Existing results: Accuracy: {existing_accuracy:.4f} ({existing_correct}/{existing_total})")
            return
        except Exception as e:
            logging.warning(f"Could not load existing results: {e}")
            logging.info("Proceeding with evaluation...")

    # Prepare arguments for multi-threading
    args_list = [
        (i, llm_client, sample, use_retrieval, args.retrieval_top_k, retriever_type, embedding_model)
        for i, sample in enumerate(samples)
    ]

    # Initialize results storage and tracking
    results: List[Dict[str, Any]] = [None] * len(samples)  # type: ignore
    correct_count = 0
    processed_count = 0
    lock = Lock()

    logging.info(f"Starting evaluation of {len(samples)} samples with {args.num_threads} threads...")

    # Create progress bar
    with tqdm(total=len(samples), desc="Evaluating samples", unit="sample") as pbar:
        if args.debug:
            # Debug mode: process only 1 sample in main thread, no threading
            idx, result = evaluate_sample_with_index(args_list[0])
            results[idx] = result
            processed_count = 1
            correct_count = 1 if result["sample_metrics"]["ExactMatch"] == 1.0 else 0
            current_accuracy = correct_count / processed_count if processed_count > 0 else 0
            pbar.set_postfix({"Accuracy": f"{current_accuracy:.4f}", "Correct": f"{correct_count}/{processed_count}"})
            pbar.update(1)
            sample_status = "✓" if result["sample_metrics"]["ExactMatch"] == 1.0 else "✗"
            logging.info(
                f"Sample {idx + 1}: {sample_status} "
                f"(Predicted: {result['predicted_answer']}, Label: {result['label']})"
            )
        else:
            # Use ThreadPoolExecutor for parallel processing
            with ThreadPoolExecutor(max_workers=args.num_threads) as executor:
                # Submit all tasks
                future_to_index = {
                    executor.submit(evaluate_sample_with_index, args_tuple): args_tuple[0] for args_tuple in args_list
                }

                # Process completed tasks
                for future in as_completed(future_to_index):
                    index = future_to_index[future]
                    try:
                        idx, result = future.result()

                        # Thread-safe update of results
                        with lock:
                            results[idx] = result
                            processed_count += 1

                            if result["sample_metrics"]["ExactMatch"] == 1.0:
                                correct_count += 1

                            # Calculate current accuracy
                            current_accuracy = correct_count / processed_count if processed_count > 0 else 0

                            # Update progress bar with current accuracy
                            pbar.set_postfix(
                                {"Accuracy": f"{current_accuracy:.4f}", "Correct": f"{correct_count}/{processed_count}"}
                            )
                            pbar.update(1)

                            # Log individual result
                            sample_status = "✓" if result["sample_metrics"]["ExactMatch"] == 1.0 else "✗"
                            logging.info(
                                f"Sample {idx + 1}: {sample_status} "
                                f"(Predicted: {result['predicted_answer']}, Label: {result['label']})"
                            )

                    except Exception as e:
                        logging.error(f"Error processing sample {index + 1}: {e}")
                        # Create a failed result to maintain order
                        with lock:
                            if results[index] is None:
                                results[index] = {
                                    "question": samples[index].get("question", ""),
                                    "prompt": samples[index].get("prompt", ""),
                                    "original_prompt": samples[index].get("prompt", ""),
                                    "label": samples[index].get("label", ""),
                                    "predicted_answer": "",
                                    "explanation": f"Thread error: {str(e)}",
                                    "sample_metrics": {"ExactMatch": 0.0},
                                    "raw_response": "",
                                    "metadata": {},
                                    "used_retrieval": use_retrieval,
                                    "retriever_type": retriever_type if use_retrieval else None,
                                    "top_k": args.retrieval_top_k if use_retrieval else None,
                                }
                            processed_count += 1
                            pbar.update(1)

    # Verify all results are filled (should not happen with proper threading)
    if None in results:
        logging.error("Some samples were not processed!")
        exit(1)

    # Calculate final accuracy
    accuracy = correct_count / len(samples) if len(samples) > 0 else 0

    # Aggregate statistics by question type
    question_type_stats = calculate_question_type_stats(samples, results)

    # Print evaluation summary
    print_evaluation_summary(accuracy, correct_count, len(samples), question_type_stats, "LLM Evaluation")

    # Save results
    output_data = {
        "args": vars(args),
        "total_samples": len(samples),
        "correct_predictions": correct_count,
        "accuracy": accuracy,
        "question_type_stats": question_type_stats,
        "results": results,
    }

    with open(output_path, "w") as f:
        from remem.utils.misc_utils import safe_json_serializable

        output_data = safe_json_serializable(output_data)
        json.dump(output_data, f, indent=2)

    logging.info("Evaluation completed!")
    logging.info(f"Final Accuracy: {accuracy:.4f} ({correct_count}/{len(samples)})")
    logging.info(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
