import argparse
import json
import os
from collections import defaultdict

from remem.remem import ReMem, logger
from remem.utils.config_utils import BaseConfig
from remem.utils.misc_utils import safe_dump_json, safe_serialize_query_solutions


def preprocess_longmemeval(samples):
    # Prepare data
    unique_sessions = set()
    for sample_idx, sample in enumerate(samples):
        sessions = []
        sample["answer_session_idxs"] = []
        for i in range(len(sample["haystack_sessions"])):
            # if len(sample['haystack_sessions'][i]) == 0:
            #     logger.info(f'Empty session for sample {sample_idx} session {i}')
            session = sample["haystack_sessions"][i]
            if session["id"] in sample["answer_session_ids"]:
                sample["answer_session_idxs"].append(i)
            sessions.append(session)
        sample["haystack_sessions"] = sessions
        del sample["haystack_dates"]
        unique_sessions.add(" ".join(sample["haystack_session_ids"]))
        del sample["haystack_session_ids"]

        docs = []
        for session_idx, session in enumerate(sessions):
            doc = {"date": session["date"], "messages": session["messages"]}
            docs.append(json.dumps(doc))
        sample["haystack_docs"] = docs
    logger.info(f"# of unique sessions: {len(unique_sessions)}")


def get_docs_from_haystack_sessions(candidate_docs, sample_idx):
    docs_to_index = (
        []
    )  # convert candidate_docs to the format to be indexed, where each item is a list of chunks, and each chunk is a dict with 'content', 'role' and 'date'.
    for doc in candidate_docs:
        doc_json = json.loads(doc)
        messages = doc_json["messages"]
        if len(messages) == 0:
            # logger.info(f"Empty session for sample {sample_idx}, doc {doc_json['date']}")
            continue
        chunks = []
        for message in messages:
            if "content" in message:
                chunks.append(
                    {"content": message["content"], "role": message.get("role", "user"), "date": doc_json["date"]}
                )
        docs_to_index.append(chunks)

    assert len(docs_to_index), "No documents to index for sample {sample_idx}."
    return docs_to_index


def main():
    parser = argparse.ArgumentParser(description="ReMem retrieval and QA")
    parser.add_argument("--dataset", type=str, default="longmemeval_episodic.json", help="dataset label")
    parser.add_argument("--llm_base_url", type=str, default="https://api.openai.com/v1", help="LLM base URL")
    parser.add_argument("--llm_name", type=str, default="gpt-4.1-mini-2025-04-14", help="LLM name")
    parser.add_argument("--embedding_name", type=str, default="nvidia/NV-Embed-v2", help="embedding model name")
    parser.add_argument("-fi", "--force_index_from_scratch", action="store_true")
    parser.add_argument("-fo", "--force_openie_from_scratch", action="store_true")
    parser.add_argument("-fr", "--force_rag", action="store_true", help="Force rerun RAG even if results exist")
    parser.add_argument(
        "--llm_infer_mode",
        choices=["online", "offline"],
        default="online",
        help="OpenIE mode, offline denotes using VLLM offline batch mode for indexing, while online denotes",
    )
    parser.add_argument("--use_azure", action="store_true", help="Use Azure for OPENAI")
    parser.add_argument("--indices", type=str)
    parser.add_argument("--extract_format", type=str)
    parser.add_argument("--extract_method", type=str, default="episodic_gist")
    parser.add_argument("--linking_top_k", type=int, default=5)
    parser.add_argument("--qa_top_k", type=int, default=10)
    parser.add_argument("--max_model_len", type=int, default=4096)
    # Agent configuration parameters
    parser.add_argument(
        "--agent_fixed_tools", action="store_true", help="Use simple agent with only semantic_retrieve + output_answer"
    )
    parser.add_argument(
        "--agent_max_steps",
        type=int,
        default=3,
        help="Maximum reasoning steps for agent (1=retrieve only, 2=retrieve+answer, >2=full reasoning)",
    )
    args = parser.parse_args()

    logger.info(f"Arguments: {args}")

    llm_base_url = args.llm_base_url
    llm_name = args.llm_name
    llm_label = args.llm_name.replace("/", "_")
    embedding_label = args.embedding_name.replace("/", "_")
    force_index_from_scratch = args.force_index_from_scratch
    force_openie_from_scratch = args.force_openie_from_scratch
    force_rag_from_scratch = args.force_rag

    samples = json.load(open(f"reproduce/dataset/longmemeval/{args.dataset}"))
    logger.info(f"# of samples: {len(samples)}")

    preprocess_longmemeval(samples)
    if "llama" in llm_name:
        args.extract_format = "json_schema"

    # Run ReMem for each sample
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
        preprocess_chunk_func="by_session",
        use_azure=args.use_azure,
        extract_format=args.extract_format,
        extract_method=args.extract_method,
        qa_passage_prefix="",
        qa_prompt_template="rag_qa_longmemeval",
        max_model_len=args.max_model_len,
        agent_fixed_tools=args.agent_fixed_tools,
        agent_max_steps=args.agent_max_steps,
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
    if args.indices and args.indices.lower() != "none":
        indices = json.load(open(args.indices))

    selected_metrics = ("qa_em", "qa_longmemeval", "qa_f1", "qa_bleu1", "retrieval_recall_all", "retrieval_ndcg_any")
    num_sample = 0
    metrics = defaultdict(float)
    for sample_idx, sample in enumerate(samples):
        if indices is not None and sample_idx not in indices:
            continue
        num_sample += 1
        logger.info(f"Processing sample idx {sample_idx}, num_sample {num_sample}")
        dataset_name = f"longmemeval_s_{sample_idx}"
        all_questions = [sample["question"]]
        all_question_metadata = [
            {"date": sample["question_date"], "type": sample["question_type"], "id": sample["question_id"]}
        ]
        gold_sessions = [sample["haystack_docs"][answer_idx] for answer_idx in sample["answer_session_idxs"]]
        gold_rounds = []
        for session in gold_sessions:
            session_json = json.loads(session)
            gold_rounds.extend(
                json.dumps(message) for message in session_json["messages"] if message.get("has_answer") is True
            )

        gold_answers = [sample["answer"]]
        candidate_docs = sample["haystack_docs"]
        config.dataset = dataset_name
        config.__post_init__()

        working_dir = os.path.join(config.save_dir, f"{dataset_name}_{llm_label}_{embedding_label}")
        inference_type = (
            f"agent_max_step_{args.agent_max_steps}"
            if not args.agent_fixed_tools
            else f"agent_fixed_max_step_{args.agent_max_steps}"
        )
        rag_results_path = os.path.join(working_dir, f"rag_results_{inference_type}.json")

        # Check if rag_results.json already exists and skip if not forcing rerun
        if os.path.exists(rag_results_path) and not force_rag_from_scratch:
            logger.info(f"RAG results already exist for sample {sample_idx}, path {rag_results_path}, skipping...")
            try:
                with open(rag_results_path, "r") as f:
                    existing_results = json.load(f)
                    if "overall_metrics" in existing_results:
                        # Add existing metrics to total
                        existing_metrics = existing_results["overall_metrics"]
                        for key in existing_metrics:
                            metrics[key] += existing_metrics[key]
                        logger.info(f"Loaded existing metrics for sample {sample_idx}")

                        # Log current average metrics
                        avg_metrics_str = "\t".join(
                            [f"{key}: {round(metrics[key] / num_sample, 4)}" for key in metrics.keys()]
                        )
                        logger.info(f"Average metrics after {num_sample} samples: {avg_metrics_str}")
            except Exception as e:
                logger.error(f"Error loading existing results for sample {sample_idx}: {e}")
                logger.info("Will rerun this sample...")
            else:
                continue

        rag = ReMem(global_config=config, working_dir=working_dir, llm=llm_model)
        rag.set_embedding_model(embedding_model)

        docs_to_index = get_docs_from_haystack_sessions(candidate_docs, sample_idx)

        rag.index(docs_to_index)

        query_solutions, all_response_message, all_metadata, retrieval_results, qa_results = rag.rag_for_qa(
            queries=all_questions,
            gold_docs=[gold_rounds],
            gold_answers=[gold_answers],
            metrics=selected_metrics,
            question_metadata=all_question_metadata,
        )

        for key in retrieval_results:
            metrics[key] += retrieval_results[key]
        for key in qa_results:
            metrics[key] += qa_results[key]

        # Log current sample metrics
        current_sample_metrics = {**retrieval_results, **qa_results}
        current_sample_str = "\t".join([f"{key}: {round(value, 4)}" for key, value in current_sample_metrics.items()])
        logger.info(f"Sample {sample_idx} metrics: {current_sample_str}")

        # save RAG results to file
        sample_metrics = {**retrieval_results, **qa_results}
        samples_dict = safe_serialize_query_solutions(query_solutions)
        rag_results = {"samples": samples_dict, "overall_metrics": sample_metrics}

        success = safe_dump_json(rag_results, rag_results_path)
        if not success:
            logger.warning(f"Warning: Had to use fallback serialization for {rag_results_path}")

        # Log current average metrics
        avg_metrics_str = "\t".join([f"{key}: {round(metrics[key] / num_sample, 4)}" for key in metrics.keys()])
        logger.info(f"Average metrics after {num_sample} samples: {avg_metrics_str}")


if __name__ == "__main__":
    main()
