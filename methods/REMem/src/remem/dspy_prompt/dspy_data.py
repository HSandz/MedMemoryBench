import argparse
import json
import os

from examples.longmemeval import preprocess_longmemeval
from remem.remem import ReMem
from remem.utils.config_utils import BaseConfig


def longmemeval_data_collect(args):
    samples = json.load(open("reproduce/dataset/longmemeval/longmemeval_s"))
    preprocess_longmemeval(samples)

    llm_label = args.llm_name.replace("/", "_")
    embedding_label = args.embedding_name.replace("/", "_")

    config = BaseConfig(
        llm_base_url=args.llm_base_url,
        llm_name=args.llm_name,
        dataset="longmemeval",
        embedding_model_name=args.embedding_name,
        force_index_from_scratch=False,
        force_openie_from_scratch=False,
        rerank_dspy_file_path=None,
        retrieval_top_k=200,
        linking_top_k=5,
        max_qa_steps=3,
        qa_top_k=5,
        do_eval_retrieval=True,
        do_eval_qa=True,
        graph_type="facts_and_sim_passage_node_unidirectional",
        embedding_batch_size=32,
        max_new_tokens=None,
        corpus_len=None,
        llm_infer_mode=args.llm_infer_mode,
        preprocess_chunk_func="by_message",
        use_azure=args.use_azure,
        extract_format=args.extract_format,
        extract_method=args.extract_method,
        qa_passage_prefix="",
        qa_prompt_template="rag_qa_episodic",
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

    embedding_model = _get_embedding_client(embedding_model_name=args.embedding_name)(
        global_config=config, embedding_model_name=args.embedding_name
    )

    indices = None
    if args.indices:
        indices = json.load(open(args.indices))

    num_sample = 0
    data = []
    for sample_idx, sample in enumerate(samples):
        if indices is not None and sample_idx not in indices:
            continue
        num_sample += 1
        dataset_name = f"longmemeval_s_{sample_idx}"
        question = sample["question"]
        gold_sessions = [sample["haystack_docs"][answer_idx] for answer_idx in sample["answer_session_idxs"]]
        gold_rounds = []
        for session in gold_sessions:
            session_json = json.loads(session)
            gold_rounds.extend(
                json.dumps(message) for message in session_json["messages"] if message.get("has_answer") is True
            )
        candidate_docs = sample["haystack_docs"]

        config.dataset = dataset_name
        config.__post_init__()

        working_dir = os.path.join(config.save_dir, f"{dataset_name}_{llm_label}_{embedding_label}")
        rag = ReMem(global_config=config, working_dir=working_dir, llm=llm_model)
        rag.set_embedding_model(embedding_model)

        rag.index(candidate_docs)
        rag.prepare_retrieval_objects()
        chunk_dict = rag.chunk_embedding_store.get_hash_id_to_row_readonly()  # key -> {'hash_id': str, 'content': str}

        gold_chunk_keys = []
        for r in gold_rounds:
            r = json.loads(r)
            for chunk in chunk_dict.items():
                if r["content"] in chunk[1]["content"]:
                    gold_chunk_keys.append(chunk[0])
                    break

        openie_by_chunk = rag.get_openie_by_chunk_keys(gold_chunk_keys)

        # retrieve triples for the query
        _, _, triple_before_filter = rag.retrieve_each_query(question)
        # triple_in_gold = set()
        # for openie in openie_by_chunk:
        #     for t in openie["extracted_triples"]:
        #         triple_in_gold.add(tuple(t))

        # triple_after_filter = triple_in_gold.intersection(triple_before_filter)
        data.append(
            {
                "question": question,
                "fact_before_filter": triple_before_filter,
                "fact_after_filter": triple_before_filter,
            }
        )
        if len(data) >= 20:
            break

    split_data = {}
    split_data["train"] = data[:10]
    split_data["dev"] = data[10:]
    for s in ["train", "dev"]:
        output_path = f"src/remem/prompts/dspy_prompts/longmemeval_triple_filter_{s}.json"
        with open(output_path, "w") as f:
            json.dump(split_data[s], f, indent=4)
        print(f"{len(split_data[s])} samples saved to", output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm_base_url", type=str, default="http://localhost:8000/v1", help="LLM base URL")
    parser.add_argument("--llm_name", type=str, default="meta-llama/Llama-3.3-70B-Instruct", help="LLM name")
    parser.add_argument("--embedding_name", type=str, default="nvidia/NV-Embed-v2", help="embedding model name")
    parser.add_argument(
        "--llm_infer_mode",
        choices=["online", "offline"],
        default="online",
        help="OpenIE mode, offline denotes using VLLM offline batch mode for indexing, while online denotes",
    )
    parser.add_argument("--extract_format", type=str)
    parser.add_argument("--extract_method", type=str, default="openie")
    parser.add_argument("--indices", type=str, default="outputs/longmemeval/longmemeval_s_dev_indices_100.json")
    args = parser.parse_args()
    args.use_azure = True

    longmemeval_data_collect(args)
