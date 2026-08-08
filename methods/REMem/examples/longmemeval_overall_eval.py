

import argparse
import json
import os
from collections import defaultdict

from examples.longmemeval import preprocess_longmemeval
from remem.evaluation.retrieval_eval import is_chunk_from_original

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", default="outputs/longmemeval")
    parser.add_argument("--dataset", default="longmemeval_s.json", help="Dataset name")
    parser.add_argument("--llm_name", type=str, default="gpt-4.1-mini-2025-04-14", help="LLM name")
    parser.add_argument("--embedding_name", default="nvidia/NV-Embed-v2")
    parser.add_argument("--graph_type", default="facts_and_sim_passage_node_unidirectional")
    parser.add_argument("--inference_type", default="agent_max_step_3")
    parser.add_argument("--indices")
    args = parser.parse_args()

    # for each dir starts with `longmemeval_`, read the rag_results.json and aggregate the metrics
    llm_label = args.llm_name.replace("/", "_")
    embedding_label = args.embedding_name.replace("/", "_")

    inference_suffix = ""
    if args.graph_type != "dpr_only" and args.inference_type:
        inference_suffix = "_" + args.inference_type.replace("/", "_")

    metrics = defaultdict(float)
    num_samples = 0
    samples = []

    # Initialize agent session statistics
    agent_session_stats = {
        "total_samples_with_logs": 0,
        "total_steps": 0,
        "avg_steps_per_query": 0,
        "step_distribution": defaultdict(int),
        "tool_usage": defaultdict(int),
        "max_steps_distribution": defaultdict(int),
    }
    dataset = json.load(open(f"reproduce/dataset/longmemeval/{args.dataset}"))
    preprocess_longmemeval(dataset)

    indices = None
    if args.indices and args.indices.lower() != "none":
        indices = json.load(open(args.indices))

    longmemeval_dir = args.log
    for dir_name in os.listdir(longmemeval_dir):
        if not dir_name.startswith("longmemeval_"):
            continue
        if args.graph_type == "dpr_only":
            if "dpr" not in dir_name:
                continue
        else:
            if "dpr" in dir_name:
                continue

        rag_result_path = f"{longmemeval_dir}/{dir_name}/{dir_name.replace('_dpr_only', '')}_{llm_label}_{embedding_label}/rag_results{inference_suffix}.json"
        try:
            sample_idx = int(dir_name.replace("_dpr_only", "").split("_")[-1])
        except ValueError:
            print(f"Failed to parse directory name {dir_name}")
            continue
        if indices is not None:
            if sample_idx not in indices:
                continue

        if os.path.exists(rag_result_path):
            with open(rag_result_path, "r") as f:
                rag_results = json.load(f)
                for k, v in rag_results["overall_metrics"].items():
                    metrics[k] += v
            num_samples += 1
            question_type = dataset[sample_idx]["question_type"]
            retrieved_chunk_metadata = rag_results["samples"][0]["doc_metadata"]
            gold_docs = [json.loads(doc) for doc in rag_results["samples"][0]["gold_docs"]]

            retrieved_chunks = []
            for chunk_rank, chunk in enumerate(retrieved_chunk_metadata):
                has_answer = False
                # check if each retrieved chunk is in the gold_docs
                for gold_doc in gold_docs:
                    if is_chunk_from_original(chunk["content"], gold_doc["content"]):
                        has_answer = True
                        gold_doc["rank"] = min(gold_doc.get("rank", float("inf")), chunk_rank)
                        break
                chunk["has_answer"] = has_answer
                retrieved_chunks.append(chunk)
            sample_data = {
                "sample_idx": sample_idx,
                "question": rag_results["samples"][0]["question"],
                "question_metadata": rag_results["samples"][0]["question_metadata"],
                "qa_rationale": rag_results["samples"][0]["qa_rationale"],
                "predicted_answer": rag_results["samples"][0]["predicted_answer"],
                "gold_answers": rag_results["samples"][0]["gold_answers"],
                "sample_metrics": rag_results["samples"][0]["sample_metrics"],
                "graph_seeds": (
                    rag_results["samples"][0]["graph_seeds"] if "graph_seeds" in rag_results["samples"][0] else None
                ),
                "retrieved_chunks": retrieved_chunks,
                "gold_docs": gold_docs,
            }

            # Add agent session logs if they exist
            if "agent_session_logs" in rag_results["samples"][0]:
                sample_data["agent_session_logs"] = rag_results["samples"][0]["agent_session_logs"]

            samples.append(sample_data)

            # Process agent session logs for this sample
            if "agent_session_logs" in rag_results["samples"][0]:
                logs = rag_results["samples"][0]["agent_session_logs"]
                agent_session_stats["total_samples_with_logs"] += 1

                # Track number of steps
                num_steps = logs.get("num_steps", 0)
                agent_session_stats["total_steps"] += num_steps
                agent_session_stats["step_distribution"][num_steps] += 1

                # Track max steps setting
                max_steps = logs.get("max_steps", 0)
                agent_session_stats["max_steps_distribution"][max_steps] += 1

                # Track tool usage
                if "step_by_step" in logs:
                    for step in logs["step_by_step"]:
                        if "tool_selection" in step and "function" in step["tool_selection"]:
                            tool_name = step["tool_selection"]["function"]
                            agent_session_stats["tool_usage"][tool_name] += 1
        else:
            print(f"RAG results not found: {rag_result_path}")

    # Calculate average steps per query
    if agent_session_stats["total_samples_with_logs"] > 0:
        agent_session_stats["avg_steps_per_query"] = round(
            agent_session_stats["total_steps"] / agent_session_stats["total_samples_with_logs"], 2
        )

    # print the metrics
    print(f"Num samples: {num_samples}")
    if num_samples == 0:
        exit(0)
    for k, v in metrics.items():
        print(f"{k}: {round(v / num_samples, 4)}")

    os.makedirs("outputs/longmemeval/", exist_ok=True)
    output_path = f"outputs/longmemeval/rag_results_{num_samples}_{llm_label}_{args.graph_type}{inference_suffix}.json"

    output_json = {
        "num_samples": num_samples,
        "overall_metrics": {k: round(v / num_samples, 4) for k, v in metrics.items()},
        "samples": samples,
    }

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

    with open(output_path, "w") as f:
        json.dump(output_json, f, indent=4)
        print(f"Results saved to {output_path}")

    retrieval_error_output_path = (
        f"outputs/longmemeval/retrieval_error_{num_samples}_{llm_label}_{args.graph_type}{inference_suffix}.json"
    )
    try:
        with open(retrieval_error_output_path, "w") as f:
            retrieval_error_samples = [sample for sample in samples if sample["sample_metrics"]["Recall_all@15"] < 1.0]
            json.dump(retrieval_error_samples, f, indent=4)
            print(f"Retrieval error results saved to {retrieval_error_output_path}")
    except Exception as e:
        print(f"Skipping saving retrieval error results, {e}")

    qa_error_output_path = (
        f"outputs/longmemeval/qa_error_{num_samples}_{llm_label}_{args.graph_type}{inference_suffix}.json"
    )
    with open(qa_error_output_path, "w") as f:
        qa_error_samples = [sample for sample in samples if sample["sample_metrics"]["LLMEvalAccuracy"] < 1.0]
        json.dump(qa_error_samples, f, indent=4)
        print(f"QA error results saved to {qa_error_output_path}")

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

    # Metrics for each question type
    question_type_metrics = defaultdict(lambda: defaultdict(float))
    question_type_counts = defaultdict(int)

    for sample in samples:
        qtype = sample["question_metadata"]["type"]
        question_type_counts[qtype] += 1
        for metric, value in sample["sample_metrics"].items():
            question_type_metrics[qtype][metric] += value

    print("\nAverage metrics per question type:")
    if question_type_counts:
        # Get all metric names for consistent table columns
        all_metric_names = set()
        for metrics_dict in question_type_metrics.values():
            all_metric_names.update(metrics_dict.keys())

        # Print markdown table header
        print("|Question type|# of samples|", end="")
        for metric in sorted(all_metric_names):
            print(f"{metric}|", end="")
        print("\n|---|---|", end="")
        for metric in sorted(all_metric_names):
            print("---|", end="")
        print()

        # Print overall metrics row
        print(f"|overall|{num_samples}|", end="")
        overall_avg_metrics = {k: round(v / num_samples, 4) for k, v in metrics.items()}
        for metric in sorted(all_metric_names):
            if metric in overall_avg_metrics:
                print(f"{overall_avg_metrics[metric]}|", end="")
            else:
                print("N/A|", end="")
        print()

        # Print each question type row
        for qtype, metrics_dict in question_type_metrics.items():
            print(f"|{qtype}|{question_type_counts[qtype]}|", end="")
            for metric in sorted(all_metric_names):
                if metric in metrics_dict:
                    avg = metrics_dict[metric] / question_type_counts[qtype]
                    print(f"{round(avg, 4)}|", end="")
                else:
                    print("N/A|", end="")
            print()
    else:
        print("No question type data available.")

    # Add question type summary to output
    question_type_summary = {}
    for qtype in question_type_counts:
        count = question_type_counts[qtype]
        question_type_summary[qtype] = {
            "count": count,
            "metrics": {metric: round(total / count, 4) for metric, total in question_type_metrics[qtype].items()},
        }
    output_json["question_type_summary"] = question_type_summary

    # Metrics for each temporal category (if available)
    temporal_category_metrics = defaultdict(lambda: defaultdict(float))
    temporal_category_counts = defaultdict(int)

    # Check if temporal category data is available
    temporal_data_available = False
    for sample in samples:
        # Check for temporal_category in dataset
        sample_idx = sample["sample_idx"]
        if sample_idx < len(dataset) and "temporal_category" in dataset[sample_idx]:
            temporal_data_available = True
            temporal_category = dataset[sample_idx]["temporal_category"]
            temporal_category_counts[temporal_category] += 1
            for metric, value in sample["sample_metrics"].items():
                temporal_category_metrics[temporal_category][metric] += value
        # Also check if temporal_category is in question_metadata (fallback)
        elif "temporal_category" in sample["question_metadata"]:
            temporal_data_available = True
            temporal_category = sample["question_metadata"]["temporal_category"]
            temporal_category_counts[temporal_category] += 1
            for metric, value in sample["sample_metrics"].items():
                temporal_category_metrics[temporal_category][metric] += value

    if temporal_data_available:
        print("\nAverage metrics per temporal category:")
        # Get all metric names for consistent table columns
        all_metric_names = set()
        for metrics_dict in temporal_category_metrics.values():
            all_metric_names.update(metrics_dict.keys())

        # Print markdown table header
        print("|Temporal category|# of samples|", end="")
        for metric in sorted(all_metric_names):
            print(f"{metric}|", end="")
        print("\n|---|---|", end="")
        for metric in sorted(all_metric_names):
            print("---|", end="")
        print()

        # Print overall metrics row
        print(f"|overall|{num_samples}|", end="")
        overall_avg_metrics = {k: round(v / num_samples, 4) for k, v in metrics.items()}
        for metric in sorted(all_metric_names):
            if metric in overall_avg_metrics:
                print(f"{overall_avg_metrics[metric]}|", end="")
            else:
                print("N/A|", end="")
        print()

        # Print each temporal category row
        for temporal_category, metrics_dict in temporal_category_metrics.items():
            print(f"|{temporal_category}|{temporal_category_counts[temporal_category]}|", end="")
            for metric in sorted(all_metric_names):
                if metric in metrics_dict:
                    avg = metrics_dict[metric] / temporal_category_counts[temporal_category]
                    print(f"{round(avg, 4)}|", end="")
                else:
                    print("N/A|", end="")
            print()

        # Add temporal category summary to output
        temporal_category_summary = {}
        for temporal_category in temporal_category_counts:
            count = temporal_category_counts[temporal_category]
            temporal_category_summary[temporal_category] = {
                "count": count,
                "metrics": {
                    metric: round(total / count, 4)
                    for metric, total in temporal_category_metrics[temporal_category].items()
                },
            }
        output_json["temporal_category_summary"] = temporal_category_summary
    else:
        print("\nNo temporal category data found in the dataset.")
        print("To add temporal category analysis, you may need to run temporal classification first.")
        print(
            "Consider using reproduce/sample_longmemeval_temporal.py to classify questions by temporal reasoning type."
        )
