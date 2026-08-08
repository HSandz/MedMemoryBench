# python
import argparse
import json
import os
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", default="outputs/locomo")
    parser.add_argument("--dataset", default="locomo_episodic")
    parser.add_argument("--llm_name", default="gpt-4.1-mini-2025-04-14")
    parser.add_argument("--embedding_model", default="nvidia/NV-Embed-v2")
    parser.add_argument("--graph_type", default="facts_and_sim_passage_node_unidirectional")
    parser.add_argument("--inference_type", default="agent_max_step_3")
    args = parser.parse_args()

    # build paths
    llm_label = args.llm_name.replace("/", "_")
    embedding_label = args.embedding_model.replace("/", "_")

    inference_suffix = ""
    if args.graph_type != "dpr_only" and args.inference_type:
        inference_suffix = "_" + args.inference_type.replace("/", "_")

    metrics = defaultdict(float)
    num_samples = 0

    # Initialize temporal category aggregation structures
    temporal_category_metrics = defaultdict(lambda: defaultdict(float))
    temporal_category_counts = defaultdict(int)

    # Initialize agent session statistics
    agent_session_stats = {
        "total_samples_with_logs": 0,
        "total_steps": 0,
        "avg_steps_per_query": 0,
        "step_distribution": defaultdict(int),
        "tool_usage": defaultdict(int),
        "max_steps_distribution": defaultdict(int),
    }

    samples = []
    locomo_dir = args.log
    for dir_name in os.listdir(locomo_dir):  # for each LOCOMO session
        if not dir_name.startswith(f"{args.dataset}_"):
            continue
        if args.graph_type == "dpr_only":
            if "dpr" not in dir_name:
                continue
        else:
            if "dpr" in dir_name:
                continue

        rag_result_path = f"{locomo_dir}/{dir_name}/{dir_name.replace('_dpr_only', '')}_{llm_label}_{embedding_label}/rag_results{inference_suffix}.json"
        if os.path.exists(rag_result_path):
            with open(rag_result_path, "r") as f:
                rag_results = json.load(f)
            samples.extend(rag_results["samples"])
            for k, v in rag_results["overall_metrics"].items():
                metrics[k] += v * len(rag_results["samples"])
            num_samples += len(rag_results["samples"])

            # Aggregate temporal category metrics if they exist
            if "temporal_categories" in rag_results:
                for category, category_data in rag_results["temporal_categories"].items():
                    count = category_data["count"]
                    temporal_category_counts[category] += count
                    for metric_name, metric_value in category_data["metrics"].items():
                        # Weight by count to get total across all files
                        temporal_category_metrics[category][metric_name] += metric_value * count
            else:  # temporal_category exists in `question_metadata` for each QA pair, get metrics from each QA pair
                for sample in rag_results["samples"]:
                    if "question_metadata" in sample:
                        temporal_category = sample["question_metadata"].get("temporal_category")
                        if temporal_category:
                            for metric_name, metric_value in sample["sample_metrics"].items():
                                temporal_category_metrics[temporal_category][metric_name] += metric_value
                            temporal_category_counts[temporal_category] += 1

    # Process agent session logs
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

            # Track tool usage
            if "step_by_step" in logs:
                for step in logs["step_by_step"]:
                    if "tool_selection" in step and "function" in step["tool_selection"]:
                        tool_name = step["tool_selection"]["function"]
                        agent_session_stats["tool_usage"][tool_name] += 1

    # Calculate average steps per query
    if agent_session_stats["total_samples_with_logs"] > 0:
        agent_session_stats["avg_steps_per_query"] = round(
            agent_session_stats["total_steps"] / agent_session_stats["total_samples_with_logs"], 2
        )

    for sample in samples:
        gold_docs = []
        if sample.get("gold_docs"):
            for g in sample["gold_docs"]:
                g = json.loads(g)["messages"][0]
                gold_docs.append(g)
            sample["gold_docs"] = gold_docs

        if "doc_metadata" in sample and sample["doc_metadata"]:
            for g in sample["gold_docs"]:
                for rank, m in enumerate(sample["doc_metadata"]):
                    if m["content"] in g["content"]:
                        g["rank"] = rank
                        break

            for m in sample["doc_metadata"]:
                has_answer = False
                for g in sample["gold_docs"]:
                    if m["content"] in g["content"]:
                        has_answer = True
                        break
                m["has_answer"] = has_answer

    output_path = f"outputs/locomo/rag_results_{num_samples}_{llm_label}_{args.graph_type}{inference_suffix}.json"
    overall_metrics = {k: round(v / num_samples, 4) for k, v in metrics.items()}

    # Calculate aggregated temporal category metrics
    aggregated_temporal_categories = {}
    if temporal_category_counts:
        for category, total_count in temporal_category_counts.items():
            if total_count > 0:
                aggregated_temporal_categories[category] = {"count": total_count, "metrics": {}}
                for metric_name, total_weighted_value in temporal_category_metrics[category].items():
                    # Calculate average across all samples in this category
                    avg_value = total_weighted_value / total_count
                    aggregated_temporal_categories[category]["metrics"][metric_name] = round(avg_value, 4)

    output_json = {
        "num_samples": num_samples,
        "overall_metrics": overall_metrics,
        "samples": samples,
    }

    # Add temporal categories if they exist
    if aggregated_temporal_categories:
        output_json["temporal_categories"] = aggregated_temporal_categories

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

    # Print temporal category metrics if available
    if aggregated_temporal_categories:
        print("\nTemporal category metrics:")
        all_metric_names = set()
        for category_data in aggregated_temporal_categories.values():
            all_metric_names.update(category_data["metrics"].keys())
        print("|Temporal category|# of samples|", end="")
        for metric in all_metric_names:
            print(f"{metric}|", end="")
        print("\n|---|---|", end="")
        for metric in all_metric_names:
            print("---|", end="")
        print()

        print(f"|overall|{num_samples}|", end="")
        for metric in all_metric_names:
            if metric in overall_metrics:
                print(f"{overall_metrics[metric]}|", end="")
        print()

        # Calculate aggregated metrics for all non-"none" temporal categories
        temporal_total_count = 0
        temporal_weighted_metrics = defaultdict(float)
        for category, category_data in aggregated_temporal_categories.items():
            if category.lower() != "none":
                count = category_data["count"]
                temporal_total_count += count
                for metric_name, metric_value in category_data["metrics"].items():
                    temporal_weighted_metrics[metric_name] += metric_value * count

        # Print temporal (non-none) row
        if temporal_total_count > 0:
            print(f"|temporal|{temporal_total_count}|", end="")
            for metric in all_metric_names:
                if metric in temporal_weighted_metrics:
                    avg = temporal_weighted_metrics[metric] / temporal_total_count
                    print(f"{round(avg, 4)}|", end="")
                else:
                    print("|", end="")
            print()

        for category, category_data in aggregated_temporal_categories.items():
            print(f"|{category}|{category_data['count']}|", end="")

            for metric in all_metric_names:
                if metric in category_data["metrics"]:
                    avg = category_data["metrics"][metric]
                    print(f"{avg}|", end="")
            print()

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
        exit(0)

    with open(output_path, "w") as f:
        json.dump(output_json, f, indent=4)
        print("Saved overall results to", output_path)

    retrieval_error_output_path = (
        f"outputs/locomo/retrieval_error_{num_samples}_{llm_label}_{args.graph_type}{inference_suffix}.json"
    )
    try:
        with open(retrieval_error_output_path, "w") as f:
            retrieval_error_samples = [sample for sample in samples if sample["sample_metrics"]["Recall@15"] < 1.0]
            json.dump(retrieval_error_samples, f, indent=4)
            print(f"Retrieval error results saved to {retrieval_error_output_path}")
    except Exception as e:
        print(f"Skipping saving retrieval error results, {e}")

    qa_error_output_path = f"outputs/locomo/qa_error_{num_samples}_{llm_label}_{args.graph_type}{inference_suffix}.json"
    with open(qa_error_output_path, "w") as f:
        qa_error_samples = [sample for sample in samples if sample["sample_metrics"]["LLMEvalAccuracy"] < 1]
        json.dump(qa_error_samples, f, indent=4)
        print(f"QA error results saved to {qa_error_output_path}")

    # Metrics for each question type
    question_type_metrics = defaultdict(lambda: defaultdict(float))
    question_type_counts = defaultdict(int)

    for sample in samples:
        if "question_metadata" not in sample:
            continue
        qtype = sample["question_metadata"]["type"]
        question_type_counts[qtype] += 1
        for metric, value in sample["sample_metrics"].items():
            question_type_metrics[qtype][metric] += value

    qtype_to_label = {2: "single-hop", 3: "temporal", 4: "open-domain", 5: "adversarial", 1: "multi-hop"}
    all_metric_names = set()
    for metrics_dict in question_type_metrics.values():
        all_metric_names.update(metrics_dict.keys())
    if len(question_type_counts) > 0:
        print("\nAverage metrics per question type:")
        print("|Question type|# of samples|", end="")
        for metric in all_metric_names:
            print(f"{metric}|", end="")
        print("\n|---|---|", end="")
        for metric in all_metric_names:
            print("---|", end="")
        print()

        for qtype, metrics_dict in question_type_metrics.items():
            print(f"|{qtype_to_label[qtype]}|{question_type_counts[qtype]}|", end="")
            for metric in all_metric_names:
                if metric in metrics_dict:
                    avg = metrics_dict[metric] / question_type_counts[qtype]
                    print(f"{round(avg, 4)}|", end="")
            print()
    else:
        print("\nNo question type metrics available, skipping question type evaluation.")


if __name__ == "__main__":
    main()
