#!/usr/bin/env python3
"""
Script to evaluate LLMs on the ToT semantic dataset using the TISER prompt template.
This script parses the query and temporal contexts from the original prompts and uses
a new Chain of Thought (CoT) approach with reflection and timeline analysis.
Features multi-threading for faster processing while maintaining result order and real-time accuracy tracking.
"""

import argparse
import json
import logging
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from typing import Any, Dict, List, Tuple

# Add the src directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
src_dir = os.path.join(parent_dir, "src")
sys.path.insert(0, src_dir)

from tot_semantic_utils import calculate_question_type_stats, print_evaluation_summary
from tqdm import tqdm

from remem.llm.openai_gpt import CacheOpenAI


def parse_prompt(prompt: str) -> Tuple[str, str]:
    """
    Parse the original prompt to extract the question and temporal context.

    Args:
        prompt: The original prompt containing temporal facts and question

    Returns:
        Tuple of (question, temporal_context)
    """
    # Split the prompt into lines
    lines = prompt.strip().split("\n")

    # Find where the facts end and question starts
    question = ""
    temporal_facts = []

    # Extract all lines that contain temporal facts
    in_facts_section = False
    found_question_section = False

    for i, line in enumerate(lines):
        line = line.strip()

        if line.startswith("Here is a set of temporal facts:"):
            in_facts_section = True
            continue
        elif line.startswith("Answer the following question"):
            in_facts_section = False
            found_question_section = True
            continue
        elif found_question_section and line == "Question:":
            # The question is on the next line
            if i + 1 < len(lines):
                question = lines[i + 1].strip()
            break
        elif in_facts_section and line and not line.startswith("Answer the following"):
            # This is a temporal fact line (any non-empty line in facts section)
            temporal_facts.append(line)

    # Join temporal facts as the temporal context
    temporal_context = "\n".join(temporal_facts)

    return question, temporal_context


def load_dataset(dataset_path: str) -> List[Dict[str, Any]]:
    """Load the ToT semantic dataset."""
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    with open(dataset_path, "r") as f:
        samples = json.load(f)

    logging.info(f"Loaded {len(samples)} samples from ToT semantic dataset")
    return samples


def evaluate_sample_tiser(llm_client: CacheOpenAI, sample: Dict[str, Any], sample_idx: int) -> Dict[str, Any]:
    """
    Evaluate a single sample using the LLM with TISER prompt template.

    Args:
        llm_client: The LLM model to use for evaluation
        sample: The sample containing prompt, question, and label
        sample_idx: The index of the sample in the dataset

    Returns:
        Dictionary containing evaluation results
    """
    original_prompt = sample["prompt"]
    original_question = sample["question"]
    label = sample["label"]

    # Parse the original prompt to extract question and temporal context
    parsed_question, temporal_context = parse_prompt(original_prompt)

    # Use the original question if parsing didn't extract one
    if not parsed_question:
        parsed_question = original_question

    # Create the TISER prompt
    from baselines.tiser import create_tiser_messages

    tiser_messages = create_tiser_messages(parsed_question, temporal_context)

    try:
        # Get response from LLM
        response, metadata, cache_hit = llm_client.infer(messages=tiser_messages)

        # Extract answer from the response
        predicted_answer = ""
        explanation = ""

        # Try to extract answer from <answer> tags
        answer_match = re.search(r"<answer>\s*(.+?)\s*</answer>", response, re.IGNORECASE | re.DOTALL)
        if answer_match:
            predicted_answer = answer_match.group(1).strip()
        else:
            # Fallback: use the entire response
            predicted_answer = response.strip()

        # Try to extract reasoning from <reasoning> tags for explanation
        reasoning_match = re.search(r"<reasoning>\s*(.+?)\s*</reasoning>", response, re.IGNORECASE | re.DOTALL)
        if reasoning_match:
            explanation = reasoning_match.group(1).strip()
        else:
            explanation = "No structured reasoning found"

        # Check if prediction matches label
        if predicted_answer.startswith("E") and ", " in predicted_answer:
            from examples.tot_semantic import remove_spaces_from_prediction

            predicted_answer = remove_spaces_from_prediction(predicted_answer)
        exact_match = 1.0 if str(predicted_answer).strip() == str(label).strip() else 0.0

        result = {
            "sample_idx": sample_idx,
            "question": original_question,
            # "parsed_question": parsed_question,
            # "original_prompt": original_prompt,
            # "temporal_context": temporal_context,
            "label": label,
            "predicted_answer": predicted_answer,
            "explanation": explanation,
            "sample_metrics": {"ExactMatch": exact_match},
            "raw_response": response,
            "metadata": metadata,
        }

        return result

    except Exception as e:
        logging.error(f"Error evaluating sample {sample_idx}: {e}")
        return {
            "sample_idx": sample_idx,
            "question": original_question,
            # "parsed_question": parsed_question,
            # "original_prompt": original_prompt,
            # "temporal_context": temporal_context if 'temporal_context' in locals() else "",
            "label": label,
            "predicted_answer": "",
            "explanation": f"Error: {str(e)}",
            "sample_metrics": {"ExactMatch": 0.0},
            "raw_response": "",
            "metadata": {},
        }


def evaluate_sample_with_index(args_tuple):
    """
    Wrapper for evaluate_sample_tiser that includes the index for ordered results.

    Args:
        args_tuple: Tuple of (index, llm_model, sample)

    Returns:
        Tuple of (index, result)
    """
    index, llm_model, sample = args_tuple
    result = evaluate_sample_tiser(llm_model, sample, index)
    return index, result


def main():
    parser = argparse.ArgumentParser(description="Evaluate LLMs on ToT Semantic Dataset using TISER prompt template")
    parser.add_argument("--llm_base_url", type=str, default="https://api.openai.com/v1", help="LLM base URL")
    parser.add_argument("--llm_name", type=str, default="gpt-4.1-mini", help="LLM name")
    parser.add_argument("--use_azure", action="store_true", help="Use Azure for OpenAI")
    parser.add_argument("--debug", action="store_true", help="Debug mode - process only 1 sample")
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="outputs/tot_semantic/llm_cache_tiser",
        help="Cache directory for LLM responses",
    )
    parser.add_argument("--num_threads", type=int, default=5, help="Number of threads for parallel processing")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    logging.info(f"Arguments: {args}")
    logging.info("Using TISER prompt template with Chain of Thought (CoT) approach")

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

    # Initialize results storage and tracking
    results: List[Dict[str, Any]] = [None] * len(samples)  # type: ignore
    correct_count = 0
    processed_count = 0

    logging.info(f"Starting evaluation of {len(samples)} samples...")

    # Create progress bar
    with tqdm(total=len(samples), desc="Evaluating samples", unit="sample") as pbar:
        if args.debug:
            # Debug mode: process only 1 sample in main thread, no threading
            logging.info("Debug mode: processing sample without threading")
            sample = samples[0]
            result = evaluate_sample_tiser(llm_client, sample, 0)
            results[0] = result
            processed_count = 1
            correct_count = 1 if result["sample_metrics"]["ExactMatch"] == 1.0 else 0
            current_accuracy = correct_count / processed_count if processed_count > 0 else 0
            pbar.set_postfix({"Accuracy": f"{current_accuracy:.4f}", "Correct": f"{correct_count}/{processed_count}"})
            pbar.update(1)
            sample_status = "✓" if result["sample_metrics"]["ExactMatch"] == 1.0 else "✗"
            logging.info(
                f"Sample 1 (idx=0): {sample_status} "
                f"(Predicted: {result['predicted_answer']}, Label: {result['label']})"
            )
        else:
            # Prepare arguments for multi-threading
            args_list = [(i, llm_client, sample) for i, sample in enumerate(samples)]
            lock = Lock()
            logging.info(f"Using {args.num_threads} threads for parallel processing")
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
                                f"Sample {idx + 1} (idx={idx}): {sample_status} "
                                f"(Predicted: {result['predicted_answer']}, Label: {result['label']})"
                            )

                    except Exception as e:
                        logging.error(f"Error processing sample {index + 1}: {e}")
                        # Create a failed result to maintain order
                        with lock:
                            if results[index] is None:
                                results[index] = {
                                    "sample_idx": index,
                                    "question": samples[index].get("question", ""),
                                    "label": samples[index].get("label", ""),
                                    "predicted_answer": "",
                                    "explanation": f"Thread error: {str(e)}",
                                    "sample_metrics": {"ExactMatch": 0.0},
                                    "raw_response": "",
                                    "metadata": {},
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
    print_evaluation_summary(accuracy, correct_count, len(samples), question_type_stats, "TISER Evaluation")

    # Save results
    output_data = {
        "args": vars(args),
        "total_samples": len(samples),
        "correct_predictions": correct_count,
        "accuracy": accuracy,
        "prompt_template": "TISER (Chain of Thought with reflection and timeline)",
        "question_type_stats": question_type_stats,
        "results": results,
    }

    llm_label = args.llm_name.replace("/", "_")
    output_path = f"outputs/tot_semantic/tot_semantic_tiser_{len(samples)}_{llm_label}.json"

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    logging.info("Evaluation completed!")
    logging.info(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
