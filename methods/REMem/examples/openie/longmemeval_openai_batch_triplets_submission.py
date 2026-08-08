
import argparse
import json
import os

import pandas as pd

from examples.longmemeval import preprocess_longmemeval
from examples.openie.longmemeval_openai_batch_ner_submission import make_json_request, submit_batches
from remem.information_extraction.openie_openai import _extract_value_from_json_response, fix_broken_generated_json
from remem.remem import ReMem
from remem.utils.config_utils import BaseConfig
from remem.utils.llm_utils import triples_json_schema


def _fake_get_embedding_model_class(*args, **kwargs):
    print("RETURN a None for embedding model class")

    def _fake_embedding_wrapper(*args, **kwargs):
        print("RETURN a None for embedding model")
        return None

    return _fake_embedding_wrapper


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm_name", type=str, default="gpt-4o-mini", help="LLM name")
    parser.add_argument("--embedding_name", type=str, default="nvidia/NV-Embed-v2", help="embedding model name")
    parser.add_argument("--use_azure", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    assert "global-batch" not in args.llm_name, "Please provide the base model name"
    llm_label = args.llm_name.replace("/", "_")
    embedding_label = args.embedding_name.replace("/", "_")

    config = BaseConfig(
        llm_base_url=None,
        llm_name=args.llm_name,
        dataset="longmemeval",
        embedding_model_name="nvidia/NV-Embed-v2",
        force_index_from_scratch=True,  # ignore previously stored index, set it to False if you want to use the previously stored index and embeddings
        force_openie_from_scratch=True,
        rerank_dspy_file_path="src/remem/prompts/dspy_prompts/filter_llama3.3-70B-Instruct.json",
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
        llm_infer_mode="online",
        preprocess_chunk_func="by_message",
        use_azure=True,
    )

    samples = json.load(open("reproduce/dataset/longmemeval/longmemeval_s"))
    preprocess_longmemeval(samples)

    json_triplet_requests = {}
    error_ = {}

    done = False
    for sample_idx in range(0, len(samples)):
        dataset_name = f"longmemeval_s_{sample_idx}"
        config.dataset = dataset_name
        config.__post_init__()

        working_dir = os.path.join(config.save_dir, f"{dataset_name}_{llm_label}_{embedding_label}")
        rag = ReMem(global_config=config, working_dir=working_dir)
        print(working_dir)
        sample = samples[sample_idx]

        docs = sample["haystack_docs"]
        preprocessed_chunks = rag.text_preprocessor.batch_preprocess_doc(input=docs)

        chunk_contents = [chunk["content"] for chunks in preprocessed_chunks for chunk in chunks]  # a list of strings

        json_triplet_requests[sample_idx] = {}

        previous_chunk_path = f"outputs/{dataset_name}/{dataset_name}_{llm_label.replace('-global-batch', '')}_{embedding_label}/chunk_embeddings/vdb_chunk.parquet"
        saved_chunks = pd.read_parquet(previous_chunk_path)
        # saved_contents = [chunk['content'] for chunk in saved_chunks.to_dict(orient='records')]
        # saved_contents = set(saved_contents)
        content_to_hash_id = {chunk["content"]: chunk["hash_id"] for chunk in saved_chunks.to_dict(orient="records")}

        triplet_input_request = {}
        error_list = []

        for text in chunk_contents:
            assert text in content_to_hash_id
            chunk_id = content_to_hash_id[text]
            ner_input_message = rag.prompt_template_manager.render(name="ner", passage=text)

            try:
                raw_response, metadata, cache_hit = rag.openie.llm.infer(
                    messages=ner_input_message, response_format={"type": "json_object"}
                )
                metadata["cache_hit"] = cache_hit
                if metadata["finish_reason"] == "length":
                    real_response = fix_broken_generated_json(raw_response)
                else:
                    real_response = raw_response
                extracted_entities = _extract_value_from_json_response(real_response)
                unique_entities = list(dict.fromkeys(extracted_entities))  # deduplicate and keep the order
                # only keep strings in the unique entities
                unique_entities = [entity for entity in unique_entities if isinstance(entity, str)]
                named_entities_str = json.dumps({"named_entities": unique_entities})
                triplet_input_message = rag.prompt_template_manager.render(
                    name="triple_extraction", passage=text, named_entity_json=named_entities_str
                )
                custom_id = f"TRIPLET_{sample_idx}_{chunk_id}"
                triplet_input_request[custom_id] = make_json_request(
                    custom_id,
                    triplet_input_message,
                    f"{llm_label}-global-batch",
                    json_mode=False,
                    json_schema=triples_json_schema,
                    use_azure=args.use_azure,
                )

            except Exception as e:
                error_list.append((e, text))

            if args.debug and len(triplet_input_request) >= 20:
                done = True
                break

        json_triplet_requests[sample_idx] = triplet_input_request
        error_[sample_idx] = error_list
        print(f"Sample {sample_idx} done, Success: {len(triplet_input_request)}, Error: {len(error_list)}")

        if done:
            break

    os.makedirs("outputs/longmemeval", exist_ok=True)

    # split json_triplet_requests into multiple files, where each one contains at most 100,000 requests
    batch_size = 100000 if args.use_azure else 50000
    all_requests = []
    for sample_requests in json_triplet_requests.values():
        all_requests.extend(sample_requests.values())

    submit_batches("triplet", all_requests, batch_size, rag, args)
