

import argparse
import json
import os
from collections import defaultdict

import numpy as np
from tqdm import tqdm

from examples.longmemeval import preprocess_longmemeval
from remem.remem import ReMem
from remem.utils.config_utils import BaseConfig
from remem.utils.misc_utils import QuerySolution

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="longmemeval_s.json", help="dataset label")
    parser.add_argument("--llm_base_url", type=str, default="https://api.openai.com/v1", help="LLM base URL")
    parser.add_argument("--llm_name", type=str, default="gpt-4.1-mini-2025-04-14", help="LLM name")
    parser.add_argument("--llm_infer_mode", type=str, default="online")
    parser.add_argument("--embedding_name", type=str, default="nvidia/NV-Embed-v2", help="embedding model name")
    parser.add_argument("--use_azure", action="store_true", help="Use Azure for OPENAI")
    parser.add_argument(
        "--gold_retrieval", action="store_true", default=False, help="Use gold documents directly instead of retrieval"
    )
    parser.add_argument("--indices", type=str)
    parser.add_argument("--qa_top_k", type=int, default=10)
    parser.add_argument("--max_model_len", type=int, default=4096)
    args = parser.parse_args()

    samples = json.load(open(f"reproduce/dataset/longmemeval/{args.dataset}"))

    preprocess_longmemeval(samples)
    indices = None
    if args.indices and args.indices.lower() != "none":
        indices = json.load(open(args.indices))

    llm_base_url = args.llm_base_url
    llm_name = args.llm_name
    llm_label = args.llm_name.replace("/", "_") if llm_name is not None else "None"
    embedding_label = args.embedding_name.replace("/", "_") if args.embedding_name is not None else "None"
    config = BaseConfig(
        llm_base_url=llm_base_url,
        llm_name=llm_name,
        dataset="longmemeval",
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
        preprocess_chunk_func="by_message",
        use_azure=args.use_azure,
        llm_infer_mode=args.llm_infer_mode,
        qa_passage_prefix="",
        qa_prompt_template="rag_qa_longmemeval",
        max_model_len=args.max_model_len,
    )
    if args.llm_infer_mode == "offline":  # the offline vllm instance is used for all RAG instances later
        from remem.llm.vllm_offline import VLLMOffline

        llm = VLLMOffline(
            config,
            model_name=args.llm_name,
            cache_dir="outputs/longmemeval/llm_cache",
        )
    else:
        llm = None

    from remem.embedding_model import _get_embedding_client

    embedding_model = _get_embedding_client(
        global_config=config, embedding_model_name=args.embedding_name, openai_style_server=True
    )

    cumulative_metrics = defaultdict(float)
    selected_metrics = ("qa_em", "qa_f1", "qa_longmemeval", "qa_bleu1", "retrieval_recall_all", "retrieval_ndcg_any")
    num_sample = 0

    for sample_idx, sample in tqdm(enumerate(samples), total=len(samples)):
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

        question_types = sample["question_type"]
        question_id = sample["question_id"]
        gold_sessions = [sample["haystack_docs"][answer_idx] for answer_idx in sample["answer_session_idxs"]]
        gold_rounds = []
        gold_rounds_dicts = []  # Store as dictionaries for gold retrieval mode
        for session in gold_sessions:
            session_json = json.loads(session)
            for message in session_json["messages"]:
                if message.get("has_answer") is True:
                    # Store as JSON string for compatibility with existing evaluation
                    gold_rounds.append(json.dumps(message))
                    # Store as dictionary with date for gold retrieval mode
                    message_dict = message.copy()
                    message_dict["date"] = session_json["date"]
                    gold_rounds_dicts.append(message_dict)

        gold_answers = [str(sample["answer"])]
        candidate_docs = sample["haystack_docs"]
        config.dataset = dataset_name
        config.__post_init__()

        working_dir = os.path.join(config.save_dir, f"{dataset_name}_{llm_label}_{embedding_label}")
        rag = ReMem(global_config=config, working_dir=working_dir, llm=llm)
        rag.set_embedding_model(embedding_model)

        qa_evaluators, retrieval_evaluators = rag.get_evaluators(gold_answers, gold_rounds, selected_metrics)

        if args.gold_retrieval:
            # Use gold documents directly instead of retrieval
            retrieved_chunk_contents = [f"{item['role']}: {item['content']}" for item in gold_rounds_dicts]
            retrieved_chunk_metadata = gold_rounds_dicts
            sorted_chunk_scores = np.array(
                [1.0] * len(retrieved_chunk_contents)
            )  # Assign score of 1.0 for each gold document
        else:
            # Original retrieval logic
            sorted_chunk_ids, sorted_chunk_scores = rag.dense_passage_retrieval(
                query=question, candidate_docs=candidate_docs
            )
            assert len(rag.chunk_embedding_store.embeddings) == len(rag.chunk_contents)
            retrieved_chunk_contents = [rag.chunk_contents[chunk_id] for chunk_id in sorted_chunk_ids]
            retrieved_chunk_metadata = [rag.chunk_metadata[chunk_id] for chunk_id in sorted_chunk_ids]
        query_solution = QuerySolution(
            question,
            retrieved_chunk_contents,
            question_metadata,
            sorted_chunk_scores,
            retrieved_chunk_metadata,
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
            cumulative_metrics[key] += retrieval_results[key]
        for key in qa_results:
            cumulative_metrics[key] += qa_results[key]

        # print current metrics
        for key in retrieval_results:
            print(key, round(cumulative_metrics[key] / num_sample, 4), end="\t")
        print()
        for key in qa_results:
            print(key, round(cumulative_metrics[key] / num_sample, 4), end="\t")
        print()

        # save RAG results to file
        with open(f"{rag.working_dir}/rag_results.json", "w") as f:
            metrics = retrieval_results
            metrics.update(qa_results)
            rag_results = {"samples": [q.to_dict() for q in query_solutions], "overall_metrics": metrics}
            json.dump(rag_results, f, indent=4)
