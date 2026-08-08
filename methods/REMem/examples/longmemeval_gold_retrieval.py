import argparse
import json
import logging
import os
from collections import defaultdict

import numpy as np

from examples.longmemeval import preprocess_longmemeval
from remem.remem import ReMem
from remem.utils.config_utils import BaseConfig
from remem.utils.misc_utils import QuerySolution


def main():
    parser = argparse.ArgumentParser(description="ReMem retrieval and QA")
    parser.add_argument("--llm_base_url", type=str, default="https://api.openai.com/v1", help="LLM base URL")
    parser.add_argument("--llm_name", type=str, default="gpt-4o-mini", help="LLM name")
    parser.add_argument("--embedding_name", type=str, default="nvidia/NV-Embed-v2", help="embedding model name")
    parser.add_argument("-fi", "--force_index_from_scratch", action="store_true")
    parser.add_argument("-fo", "--force_openie_from_scratch", action="store_true")
    parser.add_argument(
        "--llm_infer_mode",
        choices=["online", "offline"],
        default="online",
        help="OpenIE mode, offline denotes using VLLM offline batch mode for indexing, while online denotes",
    )
    parser.add_argument("--use_azure", action="store_true", help="Use Azure for OPENAI")
    parser.add_argument("--indices", type=str, default="outputs/longmemeval/longmemeval_s_dev_indices_100.json")
    parser.add_argument("--extract_format", type=str)
    parser.add_argument("--extract_method", type=str, default="openie")
    parser.add_argument("--linking_top_k", type=int, default=5)
    parser.add_argument("--qa_top_k", type=int, default=5)
    parser.add_argument("--max_model_len", type=int, default=4096)
    args = parser.parse_args()

    print(args)

    llm_base_url = args.llm_base_url
    llm_name = args.llm_name
    llm_label = args.llm_name.replace("/", "_")
    embedding_label = args.embedding_name.replace("/", "_")
    force_index_from_scratch = args.force_index_from_scratch
    force_openie_from_scratch = args.force_openie_from_scratch

    samples = json.load(open("reproduce/dataset/longmemeval/longmemeval_s"))
    logging.basicConfig(level=logging.INFO)
    logging.info(f"# of samples: {len(samples)}")

    preprocess_longmemeval(samples)
    if "llama" in llm_name:
        args.extract_format = "json_schema"

    # Run ReMem for each sample
    metrics = defaultdict(float)
    if args.use_azure:
        llm_base_url = None  # prioritize Azure ENDPOINT
    config = BaseConfig(
        llm_base_url=llm_base_url,
        llm_name=llm_name,
        dataset="longmemeval",
        embedding_model_name=args.embedding_name,
        force_index_from_scratch=force_index_from_scratch,  # ignore previously stored index, set it to False if you want to use the previously stored index and embeddings
        force_openie_from_scratch=force_openie_from_scratch,
        rerank_dspy_file_path="src/remem/prompts/dspy_prompts/filter_llama3.3-70B-Instruct.json",
        retrieval_top_k=200,
        linking_top_k=args.linking_top_k,
        qa_top_k=args.qa_top_k,
        do_eval_retrieval=True,
        do_eval_qa=True,
        graph_type="facts_and_sim_passage_node_unidirectional",
        embedding_batch_size=16,
        max_new_tokens=None,
        corpus_len=None,
        llm_infer_mode=args.llm_infer_mode,
        preprocess_chunk_func="by_message",
        use_azure=args.use_azure,
        extract_format=args.extract_format,
        extract_method=args.extract_method,
        qa_passage_prefix="",
        qa_prompt_template="rag_qa_longmemeval",
        max_model_len=args.max_model_len,
    )
    if args.llm_infer_mode == "offline":  # the offline vllm instance is used for all RAG instances later
        from remem.llm.vllm_offline import VLLMOffline

        llm_model = VLLMOffline(
            config,
            model_name=args.llm_name,
            cache_dir="outputs/longmemeval/llm_cache",
        )
    else:
        llm_model = None

    from remem.embedding_model import _get_embedding_client

    embedding_model = _get_embedding_client(
        global_config=config, embedding_model_name=args.embedding_name, openai_style_server=True
    )

    indices = None
    if args.indices:
        indices = json.load(open(args.indices))
    selected_metrics = ("qa_em", "qa_longmemeval", "retrieval_recall_all", "retrieval_ndcg_any")
    num_sample = 0
    llm_metadata = []
    for sample_idx, sample in enumerate(samples):
        if indices is not None and sample_idx not in indices:
            continue
        num_sample += 1
        print(f"Sample idx {sample_idx}, num_sample {num_sample}")
        dataset_name = f"longmemeval_s_{sample_idx}"
        question = sample["question"]
        question_metadata = {
            "date": sample["question_date"],
            "type": sample["question_type"],
            "id": sample["question_id"],
        }

        gold_sessions = [sample["haystack_docs"][answer_idx] for answer_idx in sample["answer_session_idxs"]]
        gold_rounds = []
        for session in gold_sessions:
            session_json = json.loads(session)
            for message in session_json["messages"]:
                if message.get("has_answer") is True:
                    message["date"] = session_json["date"]
                    gold_rounds.append(message)

        gold_answers = [sample["answer"]]
        candidate_docs = sample["haystack_docs"]
        config.dataset = dataset_name
        config.__post_init__()

        working_dir = os.path.join(config.save_dir, f"{dataset_name}_{llm_label}_{embedding_label}")
        rag = ReMem(global_config=config, working_dir=working_dir, llm=llm_model)
        rag.set_embedding_model(embedding_model)

        qa_evaluators, retrieval_evaluators = rag.get_evaluators(gold_answers, gold_rounds, selected_metrics)

        gold_chunk_contents = [f"{item['role']}: {item['content']}" for item in gold_rounds]
        gold_chunk_scores = np.array([1.0] * len(gold_chunk_contents))
        query_solution = QuerySolution(
            question,
            gold_chunk_contents,
            question_metadata,
            gold_chunk_scores,
            gold_rounds,
            None,
            gold_answers,
            gold_rounds,
            None,
            {},
        )
        query_solutions, all_response_message, all_metadata = rag.qa([query_solution])

        retrieval_results = rag.evaluate_retrieval([gold_rounds], query_solutions, retrieval_evaluators)
        qa_results = rag.evaluate_qa([gold_answers], qa_evaluators, query_solutions, [question_metadata])

        for key in retrieval_results:
            metrics[key] += retrieval_results[key]
        for key in qa_results:
            metrics[key] += qa_results[key]

        llm_metadata.append({"question_solution": query_solution.to_dict(), "llm_call": all_metadata})

        # print current metrics
        for key in retrieval_results:
            print(key, round(metrics[key] / num_sample, 4), end="\t")
        print()
        for key in qa_results:
            print(key, round(metrics[key] / num_sample, 4), end="\t")
        print()

    llm_metadata_output_path = (
        f"outputs/longmemeval/llm_metadata_longmemeval_{num_sample}_{llm_label}_{embedding_label}.json"
    )
    with open(llm_metadata_output_path, "w") as f:
        json.dump(llm_metadata, f, indent=4)
        print("Saved LLM metadata to", llm_metadata_output_path)


if __name__ == "__main__":
    main()
