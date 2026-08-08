#!/usr/bin/env python3
"""
Overall evaluation script for ToT (Test of Time) semantic dataset.
This script aggregates individual evaluation results across all tot_semantic samples.
"""

import argparse
import json
import os
from collections import defaultdict

from examples.tot_semantic import remove_spaces_from_prediction

# Try to import tqdm, fall back to basic iteration if not available
try:
    from tqdm import tqdm
except ImportError:

    def tqdm(iterable, desc=None):
        print(f"{desc}...")
        return iterable


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", default="outputs")
    parser.add_argument("--dataset", default="tot_semantic")
    parser.add_argument("--llm_name", default="gpt-4.1-mini-2025-04-14", help="LLM name")
    parser.add_argument("--embedding_model", default="nvidia/NV-Embed-v2", help="Embedding model name")
    parser.add_argument("--graph_type", default="facts_and_sim_passage_node_unidirectional")
    parser.add_argument("--inference_type", default="agent_max_step_5")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to process")
    args = parser.parse_args()

    # Build paths
    llm_label = args.llm_name.replace("/", "_")
    embedding_label = args.embedding_model.replace("/", "_")

    inference_suffix = ""
    if args.inference_type:
        inference_suffix = args.inference_type.replace("/", "_")

    metrics = defaultdict(float)
    num_samples = 0
    samples = []

    # Question type and metadata aggregation
    question_type_metrics = defaultdict(lambda: defaultdict(float))
    question_type_counts = defaultdict(int)

    # Graph algorithm metrics
    graph_algorithm_metrics = defaultdict(lambda: defaultdict(float))
    graph_algorithm_counts = defaultdict(int)

    # Initialize agent session statistics
    agent_session_stats = {
        "total_samples_with_logs": 0,
        "total_steps": 0,
        "avg_steps_per_query": 0,
        "step_distribution": defaultdict(int),
        "tool_usage": defaultdict(int),
        "max_steps_distribution": defaultdict(int),
    }

    tot_semantic_dir = os.path.join(args.log, args.dataset)

    # Look for individual sample directories
    sample_dirs = []
    for dir_name in os.listdir(tot_semantic_dir):
        if dir_name.startswith(f"{args.dataset}_") and os.path.isdir(os.path.join(tot_semantic_dir, dir_name)):
            try:
                # Extract sample index from directory name
                sample_idx = int(dir_name.replace(f"{args.dataset}_", ""))
                sample_dirs.append((sample_idx, dir_name))
            except ValueError:
                continue

    # Sort by sample index
    sample_dirs.sort(key=lambda x: x[0])

    # Limit samples if max_samples is specified
    if args.max_samples is not None:
        sample_dirs = sample_dirs[: args.max_samples]

    print(f"Found {len(sample_dirs)} sample directories to process")

    for sample_idx, dir_name in tqdm(sample_dirs, desc="Processing samples"):
        if args.graph_type == "dpr_only":
            if "dpr" not in dir_name:
                continue
        else:
            if "dpr" in dir_name:
                continue

        rag_result_path = os.path.join(
            tot_semantic_dir,
            dir_name,
            f"{dir_name}_{llm_label}_{embedding_label}",
            f"rag_results_{args.inference_type}.json",
        )

        if os.path.exists(rag_result_path):
            try:
                with open(rag_result_path, "r") as f:
                    rag_results = json.load(f)

                # Handle different JSON structures
                gold_answers = []
                gold_answer = ""
                agent_session_logs = None

                if "query_solutions" in rag_results and len(rag_results["query_solutions"]) > 0:
                    # New structure with query_solutions
                    sample_data = rag_results["query_solutions"][0]
                    question = sample_data["question"]
                    predicted_answer = sample_data["predicted_answer"]
                    if predicted_answer.startswith("E"):
                        predicted_answer = remove_spaces_from_prediction(predicted_answer)
                    gold_answers = sample_data["gold_answers"]
                    question_metadata = sample_data.get("question_metadata", {})
                    doc_metadata = sample_data.get("doc_metadata", [])
                    gold_docs = sample_data.get("gold_docs", [])
                    agent_session_logs = sample_data.get("agent_session_logs")

                    # Calculate exact match
                    exact_match = 1.0 if predicted_answer in gold_answers else 0.0

                    sample_metrics = {"ExactMatch": exact_match}

                elif "sample" in rag_results:
                    # Alternative structure with sample
                    sample_data = rag_results["sample"]
                    question = sample_data["question"]
                    predicted_answer = sample_data["predicted_answer"]
                    gold_answer = sample_data["gold_answer"]
                    question_metadata = sample_data.get("metadata", {})
                    doc_metadata = []
                    gold_docs = []

                    # Calculate exact match
                    exact_match = 1.0 if predicted_answer == gold_answer else 0.0

                    sample_metrics = {"ExactMatch": exact_match}
                else:
                    print(f"Unexpected structure in {rag_result_path}")
                    continue

                # Aggregate overall metrics
                for metric_name, metric_value in sample_metrics.items():
                    metrics[metric_name] += metric_value

                num_samples += 1

                # Store sample data
                sample_info = {
                    "sample_idx": sample_idx,
                    "question": question,
                    "predicted_answer": predicted_answer,
                    "gold_answers": gold_answers if "query_solutions" in rag_results else [gold_answer],
                    "sample_metrics": sample_metrics,
                    "question_metadata": question_metadata,
                    "doc_metadata": doc_metadata,
                    "gold_docs": gold_docs,
                }

                # Include agent session logs if available
                if agent_session_logs:
                    sample_info["agent_session_logs"] = agent_session_logs

                samples.append(sample_info)

                # Aggregate by question type
                question_type = question_metadata.get("type", "unknown")
                question_type_counts[question_type] += 1
                for metric_name, metric_value in sample_metrics.items():
                    question_type_metrics[question_type][metric_name] += metric_value

                # Aggregate by graph algorithm
                graph_algorithm = question_metadata.get("graph_algorithm", "unknown")
                graph_algorithm_counts[graph_algorithm] += 1
                for metric_name, metric_value in sample_metrics.items():
                    graph_algorithm_metrics[graph_algorithm][metric_name] += metric_value

            except Exception as e:
                print(f"Error processing {rag_result_path}: {e}")
                continue
        else:
            print(f"RAG results not found: {rag_result_path}")

    # Process agent session logs to generate statistics
    for sample in samples:
        if "agent_session_logs" in sample:
            logs = sample["agent_session_logs"]
            agent_session_stats["total_samples_with_logs"] += 1

            # Track number of steps
            num_steps = logs.get("num_steps", 0)
            agent_session_stats["total_steps"] += num_steps
            agent_session_stats["step_distribution"][num_steps] += 1

            # Track max steps setting
            max_steps = logs.get("max_steps", 0)
            agent_session_stats["max_steps_distribution"][max_steps] += 1

            # Track tool usage from llm_interactions
            if "llm_interactions" in logs:
                for interaction in logs["llm_interactions"]:
                    if "function" in interaction:
                        tool_name = interaction["function"]
                        agent_session_stats["tool_usage"][tool_name] += 1

    # Calculate average steps per query
    if agent_session_stats["total_samples_with_logs"] > 0:
        agent_session_stats["avg_steps_per_query"] = round(
            agent_session_stats["total_steps"] / agent_session_stats["total_samples_with_logs"], 2
        )

    if num_samples == 0:
        print("No samples found to process")
        return

    # Calculate overall average metrics
    overall_metrics = {k: round(v / num_samples, 4) for k, v in metrics.items()}

    # Prepare output
    output_json = {
        "num_samples": num_samples,
        "overall_metrics": overall_metrics,
        "samples": samples,
    }

    # Add question type breakdown
    if question_type_counts:
        question_type_breakdown = {}
        for q_type, count in question_type_counts.items():
            question_type_breakdown[q_type] = {"count": count, "metrics": {}}
            for metric_name, total_value in question_type_metrics[q_type].items():
                avg_value = total_value / count
                question_type_breakdown[q_type]["metrics"][metric_name] = round(avg_value, 4)
        output_json["question_type_breakdown"] = question_type_breakdown

    # Add graph algorithm breakdown
    if graph_algorithm_counts:
        graph_algorithm_breakdown = {}
        for algo, count in graph_algorithm_counts.items():
            graph_algorithm_breakdown[algo] = {"count": count, "metrics": {}}
            for metric_name, total_value in graph_algorithm_metrics[algo].items():
                avg_value = total_value / count
                graph_algorithm_breakdown[algo]["metrics"][metric_name] = round(avg_value, 4)
        output_json["graph_algorithm_breakdown"] = graph_algorithm_breakdown

    # Add agent session statistics if they exist
    if agent_session_stats["total_samples_with_logs"] > 0:
        # Convert defaultdict to regular dict for JSON serialization
        agent_session_stats_serializable = {
            "total_samples_with_logs": agent_session_stats["total_samples_with_logs"],
            "total_steps": agent_session_stats["total_steps"],
            "avg_steps_per_query": agent_session_stats["avg_steps_per_query"],
            "step_distribution": dict(agent_session_stats["step_distribution"]),
            "tool_usage": dict(agent_session_stats["tool_usage"]),
            "max_steps_distribution": dict(agent_session_stats["max_steps_distribution"]),
        }
        output_json["agent_session_stats"] = agent_session_stats_serializable

    print("Overall metrics:", overall_metrics)

    # Print question type metrics if available
    if question_type_counts:
        print("\nQuestion type breakdown:")
        for q_type, data in output_json["question_type_breakdown"].items():
            print(f"  {q_type} ({data['count']} samples):")
            for metric_name, metric_value in data["metrics"].items():
                print(f"    {metric_name}: {metric_value}")

    # Print graph algorithm metrics if available
    if graph_algorithm_counts:
        print("\nGraph algorithm breakdown:")
        for algo, data in output_json["graph_algorithm_breakdown"].items():
            print(f"  {algo} ({data['count']} samples):")
            for metric_name, metric_value in data["metrics"].items():
                print(f"    {metric_name}: {metric_value}")

    # Print agent session statistics if available
    if agent_session_stats["total_samples_with_logs"] > 0:
        print("\nAgent session statistics:")
        print(f"  Total samples with logs: {agent_session_stats['total_samples_with_logs']}")
        print(f"  Total steps: {agent_session_stats['total_steps']}")
        print(f"  Average steps per query: {agent_session_stats['avg_steps_per_query']}")
        print(f"  Step distribution: {dict(agent_session_stats['step_distribution'])}")
        print(f"  Tool usage: {dict(agent_session_stats['tool_usage'])}")
        print(f"  Max steps distribution: {dict(agent_session_stats['max_steps_distribution'])}")
    else:
        print("\nNo agent session logs found in the results.")

    # Save overall results
    os.makedirs(f"outputs/{args.dataset}/", exist_ok=True)
    output_path = (
        f"outputs/{args.dataset}/rag_results_{num_samples}_{llm_label}_{args.graph_type}_{inference_suffix}.json"
    )

    with open(output_path, "w") as f:
        json.dump(output_json, f, indent=4)
        print(f"\nOverall results saved to {output_path}")

    # Save error analysis (samples with incorrect answers)
    incorrect_samples = [sample for sample in samples if sample["sample_metrics"]["ExactMatch"] < 1.0]
    if incorrect_samples:
        error_output_path = (
            f"outputs/{args.dataset}/error_analysis_{num_samples}_{llm_label}_{args.graph_type}_{inference_suffix}.json"
        )
        with open(error_output_path, "w") as f:
            json.dump(incorrect_samples, f, indent=4)
            print(f"Error analysis saved to {error_output_path} ({len(incorrect_samples)} incorrect samples)")

    # Print summary statistics
    print("\nSummary:")
    print(f"  Total samples processed: {num_samples}")
    print(f"  Correct answers: {sum(sample['sample_metrics']['ExactMatch'] for sample in samples)}")
    print(f"  Incorrect answers: {len(incorrect_samples)}")
    print(f"  Accuracy: {overall_metrics['ExactMatch']}")


if __name__ == "__main__":
    main()
