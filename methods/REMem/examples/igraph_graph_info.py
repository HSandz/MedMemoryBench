
import argparse
import json
import os

from remem.remem import ReMem
from remem.utils.config_utils import BaseConfig

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="musique", help="Dataset name")
    parser.add_argument("--llm_base_url", type=str, default="https://api.openai.com/v1", help="LLM base URL")
    parser.add_argument("--llm_name", type=str, default="gpt-4o-mini")
    parser.add_argument("--embedding_name", type=str, default="nvidia/NV-Embed-v2")
    args = parser.parse_args()

    llm_label = args.llm_name.replace("/", "_")
    embedding_label = args.embedding_name.replace("/", "_")

    corpus_path = f"reproduce/dataset/{args.dataset}_corpus.json"
    corpus = json.load(open(corpus_path))
    num_passage_node = len(corpus)

    config = BaseConfig(
        llm_base_url=args.llm_base_url,
        llm_name=args.llm_name,
        dataset=args.dataset,
        embedding_model_name=args.embedding_name,
        # embedding_model_name='GritLM/GritLM-7B',
        force_index_from_scratch=False,  # ignore previously stored index, set it to False if you want to use the previously stored index and embeddings
        # linking_method="dense_passage_retrieval",
        rerank_dspy_file_path="remem_old/prompts/dspy_prompts/filter_llama3.3-70B-Instruct.json",
        retrieval_top_k=200,
        linking_top_k=5,
        qa_top_k=5,
        do_eval_retrieval=True,
        do_eval_qa=True,
        graph_type="facts_and_sim_passage_node_unidirectional",
        embedding_batch_size=200,
        corpus_len=len(corpus),
    )

    working_dir = os.path.join(config.save_dir, f"{args.dataset}_{llm_label}_{embedding_label}")
    rag = ReMem(global_config=config, working_dir=working_dir)

    print(rag.get_graph_info())
