import argparse
import json
import os
import sys
from collections import defaultdict
from glob import glob

from examples.realtalk import (
    get_gold_docs_for_qa_pair,
    get_sessions,
    map_question_category,
)
from remem.remem import ReMem
from remem.utils.config_utils import BaseConfig
from remem.utils.misc_utils import QuerySolution, safe_dump_json, safe_serialize_query_solutions
from remem.utils.results_utils import get_working_dir


def create_long_context_query_solution(question, all_sessions, gold_answers, gold_docs):
    """Build a QuerySolution that uses the entire conversation history as context.

    Messages are added in reverse session order so the most recent content appears first.
    """
    retrieved_chunk_contents = []
    retrieved_chunk_metadata = []

    for session in all_sessions[::-1]:  # reverse: most recent sessions first
        for message in session:
            content = f"[{message['date']}] {message['role']}: {message['content']}"
            retrieved_chunk_contents.append(content)
            retrieved_chunk_metadata.append(
                {
                    "role": message.get("role"),
                    "content": message.get("content"),
                    "date": message.get("date"),
                    "dialog_id": message.get("dialog_id", ""),
                    "session_idx": message.get("session_idx", None),
                    "message_idx": message.get("message_idx", None),
                }
            )

    return QuerySolution(
        question=question,
        docs=retrieved_chunk_contents,
        doc_metadata=retrieved_chunk_metadata,
        gold_answers=gold_answers,
        gold_docs=gold_docs,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm_base_url", type=str, default="https://api.openai.com/v1", help="LLM base URL")
    parser.add_argument("--llm_name", type=str, default="gpt-4.1-mini-2025-04-14", help="LLM name")
    parser.add_argument(
        "--dataset_dir", type=str, default="reproduce/dataset/realtalk", help="Path to REALTALK data directory"
    )
    parser.add_argument("--embedding_name", type=str, default="nvidia/NV-Embed-v2", help="embedding model name")
    parser.add_argument(
        "--llm_infer_mode",
        choices=["online", "offline"],
        default="online",
        help="Offline uses VLLM batch mode; online calls the configured LLM endpoint",
    )
    parser.add_argument("--use_azure", action="store_true", help="Use Azure for OPENAI")
    args = parser.parse_args()

    dataset_label = "realtalk"

    # Load REALTALK files
    files = sorted(glob(os.path.join(args.dataset_dir, "Chat_*_*.json")))
    if not files:
        print(f"No REALTALK Chat_*.json found under {args.dataset_dir}")
        sys.exit(1)

    print("# conversations:", len(files))

    llm_name = args.llm_name
    llm_label = args.llm_name.replace("/", "_") if llm_name is not None else "None"
    embedding_label = args.embedding_name.replace("/", "_") if args.embedding_name is not None else "None"

    config = BaseConfig(
        llm_base_url=args.llm_base_url,
        llm_name=llm_name,
        dataset=dataset_label,
        embedding_model_name=args.embedding_name,
        force_index_from_scratch=False,
        force_openie_from_scratch=False,
        rerank_dspy_file_path="src/remem/prompts/dspy_prompts/filter_llama3.3-70B-Instruct.json",
        retrieval_top_k=200,
        linking_top_k=5,
        qa_top_k=10000,
        do_eval_retrieval=True,
        do_eval_qa=True,
        graph_type="dpr_only",
        embedding_batch_size=32,
        max_new_tokens=None,
        corpus_len=None,
        llm_infer_mode=args.llm_infer_mode,
        preprocess_chunk_func="none",
        use_azure=args.use_azure,
        qa_passage_prefix="",
        qa_prompt_template="rag_qa_unified",
    )

    if args.llm_infer_mode == "offline":  # vLLM for direct QA calls
        from remem.llm.vllm_offline import VLLMOffline

        llm_client = VLLMOffline(
            config,
            model_name=args.llm_name,
            cache_dir="outputs/realtalk/llm_cache",
        )
    else:
        llm_client = None

    from remem.embedding_model import _get_embedding_client

    embedding_model = _get_embedding_client(
        global_config=config, embedding_model_name=args.embedding_name, openai_style_server=True
    )

    # REALTALK uses standard F1 metric key (not locomo-specific)
    selected_metrics = ("qa_em", "qa_f1", "qa_mem0_llm_judge", "qa_bleu1")

    total_metrics = defaultdict(float)
    category_metrics = defaultdict(lambda: defaultdict(float))  # category -> metric -> value
    category_counts = defaultdict(int)  # category -> count
    num_sample = 0

    for sample_idx, file_path in enumerate(files):
        try:
            sample = json.load(open(file_path))
        except Exception as e:
            print(f"Failed to load {file_path}: {e}")
            continue

        # REALTALK root contains sessions and qa
        conversation = sample
        qa_pairs = sample.get("qa", [])

        all_sessions = get_sessions(conversation)

        print(f"sample {sample_idx} ({os.path.basename(file_path)}) # sessions:", len(all_sessions))
        print(f"sample {sample_idx} # total messages:", sum(len(session) for session in all_sessions))

        dataset_name = f"{dataset_label}_{sample_idx}"
        config.dataset = dataset_name
        config.__post_init__()

        working_dir = get_working_dir(config.save_dir, dataset_name, llm_label, embedding_label)
        rag = ReMem(global_config=config, working_dir=working_dir, llm=llm_client)
        rag.set_embedding_model(embedding_model)

        session_metrics = defaultdict(float)
        session_category_metrics = defaultdict(lambda: defaultdict(float))
        session_category_counts = defaultdict(int)
        query_solutions = []

        for qa_idx, qa_pair in enumerate(qa_pairs):
            print(f"Sample idx {sample_idx}, QA idx {qa_idx}")
            num_sample += 1
            question = qa_pair.get("question", "")
            # Map dataset category to label for aggregation
            q_category_label = map_question_category(qa_pair.get("category"))

            if "answer" not in qa_pair:
                gold_answers = [["no information available"]]
            else:
                gold_answers = [[str(qa_pair.get("answer"))]]
            gold_docs = []
            question_metadata = {"type": qa_pair.get("category")}

            # Normalize evidence list
            new_evidence_list = []
            for evidence in qa_pair.get("evidence", []):
                ev = str(evidence)
                if "; " in ev:
                    new_evidence_list.extend(ev.split("; "))
                elif ";" in ev:
                    new_evidence_list.extend(ev.split(";"))
                else:
                    new_evidence_list.append(ev)
            qa_pair["evidence"] = [e.strip() for e in new_evidence_list if e.strip()]

            gold_docs = get_gold_docs_for_qa_pair(qa_pair, conversation)

            qa_evaluators, retrieval_evaluators = rag.get_evaluators(gold_answers, gold_docs, selected_metrics)

            query_solution = create_long_context_query_solution(question, all_sessions, gold_answers, gold_docs)

            # Construct QA prompt with all messages (no top-k cutoff)
            qa_context = f"Question: {question}\n\nContexts:\n"
            for passage in query_solution.docs:
                qa_context += f"{rag.global_config.qa_passage_prefix}{passage}\n"
            qa_context += "\nThought: "

            qa_message = rag.prompt_template_manager.render(name="rag_qa_unified", prompt_user=qa_context)

            # Direct QA inference
            qa_result = rag.qa_llm.infer(qa_message)
            response_content, metadata, cache_hit = qa_result

            # Parse the answer after 'Answer:' if present
            response_split = response_content.split("Answer:")
            try:
                predicted_rationale = response_split[0].strip()
                predicted_ans = response_split[1].strip()
            except Exception:
                predicted_rationale = ""
                predicted_ans = response_content

            query_solution.answer = predicted_ans
            query_solution.qa_rationale = response_content
            query_solutions.append(query_solution)

            qa_results = rag.evaluate_qa(gold_answers, qa_evaluators, [query_solution], [question_metadata])

            for key in qa_results:
                total_metrics[key] += qa_results[key]
                session_metrics[key] += qa_results[key]
                if q_category_label:
                    category_metrics[q_category_label][key] += qa_results[key]
                    session_category_metrics[q_category_label][key] += qa_results[key]

            if q_category_label:
                category_counts[q_category_label] += 1
                session_category_counts[q_category_label] += 1

            # print running averages
            for key in qa_results:
                print(key, round(total_metrics[key] / max(1, num_sample), 4), end="\t")
            print()

        # save per-conversation results
        session_metrics = {key: round(session_metrics[key] / max(1, len(qa_pairs)), 4) for key in session_metrics}
        samples_dict = safe_serialize_query_solutions(query_solutions)
        rag_results = {"samples": samples_dict, "overall_metrics": session_metrics}

        # Per-category metrics (question ability types)
        if session_category_counts:
            session_category_summary = {}
            for category in session_category_counts:
                count = session_category_counts[category]
                session_category_summary[category] = {"count": count, "metrics": {}}
                for metric in session_category_metrics[category]:
                    avg_score = session_category_metrics[category][metric] / max(1, count)
                    session_category_summary[category]["metrics"][metric] = round(avg_score, 4)
            rag_results["categories"] = session_category_summary

        results_file = f"{rag.working_dir}/rag_results.json"
        success = safe_dump_json(rag_results, results_file)
        if not success:
            print(f"Warning: Had to use fallback serialization for {results_file}")

    # Save overall results
    overall_metrics = {key: round(total_metrics[key] / max(1, num_sample), 4) for key in total_metrics}
    overall_results = {"total_qa_pairs": num_sample, "overall_metrics": overall_metrics}

    # Add category breakdown
    if category_counts:
        category_summary = {}
        for category in category_counts:
            count = category_counts[category]
            category_summary[category] = {"count": count, "metrics": {}}
            for metric in category_metrics[category]:
                avg_score = category_metrics[category][metric] / max(1, count)
                category_summary[category]["metrics"][metric] = round(avg_score, 4)
        overall_results["categories"] = category_summary

    overall_results_file = f"outputs/realtalk_long_context/overall_results_{llm_label}_{embedding_label}.json"
    os.makedirs(os.path.dirname(overall_results_file), exist_ok=True)
    safe_dump_json(overall_results, overall_results_file)

    print("Total QA pairs:", num_sample)
    print("\n=== Final Results ===")
    for key, value in overall_metrics.items():
        print(f"{key}: {value}")

    if category_counts:
        print("\n=== Question Category Breakdown ===")
        category_summary = overall_results.get("categories", {})
        for category in sorted(category_counts.keys()):
            count = category_counts[category]
            print(f"{category}: {count} samples")
            if category in category_summary:
                for metric, score in category_summary[category]["metrics"].items():
                    print(f"  {metric}: {score}")

    print(f"Overall results saved to: {overall_results_file}")
