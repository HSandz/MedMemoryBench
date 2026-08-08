import argparse
import json
import os
from collections import defaultdict

import numpy as np

from examples.locomo import get_candidate_messages, get_gold_docs_for_qa_pair
from remem.remem import ReMem
from remem.utils.config_utils import BaseConfig
from remem.utils.misc_utils import QuerySolution, safe_dump_json, safe_serialize_query_solutions

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm_base_url", type=str, default="https://api.openai.com/v1", help="LLM base URL")
    parser.add_argument("--llm_name", type=str, default="gpt-4.1-mini-2025-04-14", help="LLM name")
    parser.add_argument("--dataset", type=str, default="locomo_episodic")
    parser.add_argument("--embedding_name", type=str, default="nvidia/NV-Embed-v2", help="embedding model name")
    parser.add_argument(
        "--llm_infer_mode",
        choices=["online", "offline"],
        default="online",
        help="OpenIE mode, offline denotes using VLLM offline batch mode for indexing, while online denotes",
    )
    parser.add_argument("--qa_top_k", type=int, default=10)
    parser.add_argument("--use_azure", action="store_true", help="Use Azure for OPENAI")
    parser.add_argument(
        "--gold_retrieval", action="store_true", default=False, help="Use gold documents directly instead of retrieval"
    )
    args = parser.parse_args()

    dataset_label = "locomo" if args.dataset in ["locomo", "locomo10"] else args.dataset
    samples = json.load(open(f"reproduce/dataset/locomo/{args.dataset}.json"))
    print("# sessions:", len(samples))
    print("# questions:", sum(len(sample["qa"]) for sample in samples))

    llm_name = args.llm_name
    llm_label = args.llm_name.replace("/", "_") if llm_name is not None else "None"
    embedding_label = args.embedding_name.replace("/", "_") if args.embedding_name is not None else "None"

    config = BaseConfig(
        llm_base_url=args.llm_base_url,
        llm_name=llm_name,
        dataset=dataset_label,
        embedding_model_name=args.embedding_name,
        force_index_from_scratch=False,  # ignore previously stored index, set it to False if you want to use the previously stored index and embeddings
        force_openie_from_scratch=False,
        rerank_dspy_file_path="src/remem/prompts/dspy_prompts/filter_llama3.3-70B-Instruct.json",
        retrieval_top_k=200,
        linking_top_k=5,
        qa_top_k=args.qa_top_k,
        do_eval_retrieval=True,
        do_eval_qa=True,
        graph_type="dpr_only",
        embedding_batch_size=32,
        max_new_tokens=None,
        corpus_len=None,
        llm_infer_mode=args.llm_infer_mode,
        preprocess_chunk_func="by_message",
        use_azure=args.use_azure,
        qa_passage_prefix="",
        qa_prompt_template="rag_qa_unified",
    )
    if args.llm_infer_mode == "offline":  # the offline vllm instance is used for all RAG instances later
        from remem.llm.vllm_offline import VLLMOffline

        llm_model = VLLMOffline(
            config,
            model_name=args.llm_name,
            cache_dir="outputs/locomo/llm_cache",
        )
    else:
        llm_model = None

    from remem.embedding_model import _get_embedding_client

    embedding_model = _get_embedding_client(
        global_config=config, embedding_model_name=args.embedding_name, openai_style_server=True
    )

    selected_metrics = ("qa_em", "retrieval_recall_locomo", "qa_f1_score_locomo", "qa_mem0_llm_judge", "qa_bleu1")

    total_metrics = defaultdict(float)
    category_metrics = defaultdict(lambda: defaultdict(float))  # category -> metric -> value
    category_counts = defaultdict(int)  # category -> count
    num_sample = 0
    for sample_idx, sample in enumerate(samples):
        conversation = sample["conversation"]
        qa_pairs = sample["qa"]
        observation = sample["observation"]
        candidate_docs = get_candidate_messages(conversation)

        print(f"sample {sample_idx} # chunk contents:", len(candidate_docs))

        dataset_name = f"{dataset_label}_{sample_idx}"
        config.dataset = dataset_name
        config.__post_init__()

        working_dir = os.path.join(config.save_dir, f"{dataset_name}_{llm_label}_{embedding_label}")
        rag = ReMem(global_config=config, working_dir=working_dir, llm=llm_model)
        rag.set_embedding_model(embedding_model)

        session_metrics = defaultdict(float)
        session_category_metrics = defaultdict(
            lambda: defaultdict(float)
        )  # category -> metric -> value for this session
        session_category_counts = defaultdict(int)  # category -> count for this session
        query_solutions = []
        for qa_idx, qa_pair in enumerate(qa_pairs):
            print(f"Sample idx {sample_idx}, QA idx {qa_idx}")
            num_sample += 1
            question = qa_pair["question"]
            temporal_category = qa_pair.get("temporal_category", None)  # Get temporal category from QA pair
            if "answer" not in qa_pair:
                gold_answers = [["no information available"]]
            else:
                gold_answers = [[str(qa_pair["answer"])]]
            gold_docs = []
            question_metadata = {"type": qa_pair["category"]}

            new_evidence_list = []
            for evidence in qa_pair["evidence"]:
                if "; " in evidence:
                    new_evidence_list.extend(evidence.split("; "))
                else:
                    new_evidence_list.append(evidence)
            qa_pair["evidence"] = new_evidence_list

            gold_docs = get_gold_docs_for_qa_pair(qa_pair, conversation)

            qa_evaluators, retrieval_evaluators = rag.get_evaluators(gold_answers, gold_docs, selected_metrics)

            if args.gold_retrieval:
                retrieved_chunk_contents = []
                retrieved_chunk_metadata = []
                for gold_doc in gold_docs:
                    gold_doc = json.loads(gold_doc)
                    content = gold_doc["messages"][0]["content"].split(":")[1].strip()
                    role = gold_doc["messages"][0]["content"].split(":")[0].strip()
                    date = gold_doc["messages"][0]["date"]
                    retrieved_chunk_contents.append(f"{role}: {content}")
                    retrieved_chunk_metadata.append({"role": role, "content": content, "date": date})
                    sorted_chunk_scores = np.array(
                        [1.0] * len(retrieved_chunk_contents)
                    )  # Assigning a score of 1.0 for each gold document
            else:
                # Original retrieval logic
                sorted_chunk_ids, sorted_chunk_scores = rag.dense_passage_retrieval(
                    query=question, candidate_docs=candidate_docs
                )
                retrieved_chunk_contents = [rag.chunk_contents[chunk_id] for chunk_id in sorted_chunk_ids]
                retrieved_chunk_metadata = [rag.chunk_metadata[chunk_id] for chunk_id in sorted_chunk_ids]

            query_solution = QuerySolution(
                question,
                retrieved_chunk_contents,
                None,
                sorted_chunk_scores,
                retrieved_chunk_metadata,
                None,
                gold_answers,
                gold_docs,
                None,
                {},
                None,
            )
            queries_solution, all_response_message, all_metadata = rag.qa([query_solution])

            retrieval_results = rag.evaluate_retrieval([gold_docs], queries_solution, retrieval_evaluators)
            qa_results = rag.evaluate_qa(gold_answers, qa_evaluators, queries_solution, [question_metadata])
            query_solutions.append(query_solution)

            for key in retrieval_results:
                total_metrics[key] += retrieval_results[key]
                session_metrics[key] += retrieval_results[key]
                if temporal_category:
                    category_metrics[temporal_category][key] += retrieval_results[key]
                    session_category_metrics[temporal_category][key] += retrieval_results[key]
            for key in qa_results:
                total_metrics[key] += qa_results[key]
                session_metrics[key] += qa_results[key]
                if temporal_category:
                    category_metrics[temporal_category][key] += qa_results[key]
                    session_category_metrics[temporal_category][key] += qa_results[key]

            if temporal_category:
                category_counts[temporal_category] += 1
                session_category_counts[temporal_category] += 1

            # print current metrics
            for key in retrieval_results:
                print(key, round(total_metrics[key] / num_sample, 4), end="\t")
            print()
            for key in qa_results:
                print(key, round(total_metrics[key] / num_sample, 4), end="\t")
            print()

        # save RAG results to file
        session_metrics = {key: round(session_metrics[key] / len(qa_pairs), 4) for key in session_metrics}
        samples_dict = safe_serialize_query_solutions(query_solutions)
        rag_results = {"samples": samples_dict, "overall_metrics": session_metrics}

        # Add temporal category metrics if they exist for this session
        if session_category_counts:
            session_category_summary = {}
            for category in session_category_counts:
                count = session_category_counts[category]
                session_category_summary[category] = {"count": count, "metrics": {}}
                for metric in session_category_metrics[category]:
                    avg_score = session_category_metrics[category][metric] / count
                    session_category_summary[category]["metrics"][metric] = round(avg_score, 4)

            # Add aggregated "temporal" category (all categories except "none")
            temporal_categories = [cat for cat in session_category_counts.keys() if cat != "none"]
            if temporal_categories:
                total_temporal_count = sum(session_category_counts[cat] for cat in temporal_categories)
                session_category_summary["temporal"] = {"count": total_temporal_count, "metrics": {}}

                # Calculate weighted average for each metric
                all_metrics = set()
                for cat in temporal_categories:
                    all_metrics.update(session_category_metrics[cat].keys())

                for metric in all_metrics:
                    weighted_sum = sum(session_category_metrics[cat].get(metric, 0) for cat in temporal_categories)
                    if total_temporal_count > 0:
                        avg_score = weighted_sum / total_temporal_count
                        session_category_summary["temporal"]["metrics"][metric] = round(avg_score, 4)

            rag_results["temporal_categories"] = session_category_summary

        results_file = f"{rag.working_dir}/rag_results.json"
        success = safe_dump_json(rag_results, results_file)
        if not success:
            print(f"Warning: Had to use fallback serialization for {results_file}")

    # Save overall results
    overall_metrics = {key: round(total_metrics[key] / num_sample, 4) for key in total_metrics}
    overall_results = {"total_qa_pairs": num_sample, "overall_metrics": overall_metrics}

    # Add category breakdown
    if category_counts:
        category_summary = {}
        for category in category_counts:
            count = category_counts[category]
            category_summary[category] = {"count": count, "metrics": {}}
            for metric in category_metrics[category]:
                avg_score = category_metrics[category][metric] / count
                category_summary[category]["metrics"][metric] = round(avg_score, 4)

        # Add aggregated "temporal" category (all categories except "none")
        temporal_categories = [cat for cat in category_counts.keys() if cat != "none"]
        if temporal_categories:
            total_temporal_count = sum(category_counts[cat] for cat in temporal_categories)
            category_summary["temporal"] = {"count": total_temporal_count, "metrics": {}}

            # Calculate weighted average for each metric
            all_metrics = set()
            for cat in temporal_categories:
                all_metrics.update(category_metrics[cat].keys())

            for metric in all_metrics:
                weighted_sum = sum(category_metrics[cat].get(metric, 0) for cat in temporal_categories)
                if total_temporal_count > 0:
                    avg_score = weighted_sum / total_temporal_count
                    category_summary["temporal"]["metrics"][metric] = round(avg_score, 4)

        overall_results["temporal_categories"] = category_summary

    overall_results_file = f"outputs/locomo_dense/overall_results_{llm_label}_{embedding_label}.json"
    safe_dump_json(overall_results, overall_results_file)

    print("Total QA pairs:", num_sample)
    print("\n=== Final Results ===")
    for key, value in overall_metrics.items():
        print(f"{key}: {value}")

    if category_counts:
        print("\n=== Category Breakdown ===")
        category_summary = overall_results["temporal_categories"]

        # Print individual categories first
        for category in sorted(category_counts.keys()):
            count = category_counts[category]
            print(f"{category}: {count} samples")
            if category in category_summary:
                for metric, score in category_summary[category]["metrics"].items():
                    print(f"  {metric}: {score}")

        # Print temporal aggregate if it exists
        if "temporal" in category_summary:
            temporal_count = category_summary["temporal"]["count"]
            print(f"temporal (aggregate): {temporal_count} samples")
            for metric, score in category_summary["temporal"]["metrics"].items():
                print(f"  {metric}: {score}")

    print(f"Overall results saved to: {overall_results_file}")
