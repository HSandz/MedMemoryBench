
import argparse
import datetime
import json
import os
import time
from typing import Dict, List

from tqdm import tqdm

from examples.longmemeval import preprocess_longmemeval
from remem.remem import ReMem
from remem.utils.config_utils import BaseConfig
from remem.utils.llm_utils import ner_json_schema
from remem.utils.misc_utils import compute_mdhash_id


def make_json_request(
    custom_id: str, messages: List[Dict], model_name: str, json_mode=False, json_schema=None, use_azure=False
):
    response_format = None
    if json_mode is True and json_schema is None:
        response_format = {"type": "json_object"}
    elif json_schema is not None:
        response_format = json_schema
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/chat/completions" if use_azure else "/v1/chat/completions",
        "body": {
            "model": model_name,
            "max_completion_tokens": None,
            "n": 1,
            "seed": None,
            "temperature": 0,
            "messages": messages,
            "response_format": response_format,
        },
    }


def submit_batches(task_name, all_requests, batch_size, rag, args):
    num_files = (len(all_requests) + batch_size - 1) // batch_size
    file_ids = []
    os.makedirs("outputs/longmemeval", exist_ok=True)
    for i in tqdm(range(num_files), desc="Creating batch files"):
        batch_requests = all_requests[i * batch_size : (i + 1) * batch_size]
        batch_file_name = f"outputs/longmemeval/openai_batch_{task_name}_submission_{i}_len_{len(batch_requests)}_{args.llm_name}.jsonl"
        import orjson

        with open(batch_file_name, "wb") as f:
            for request in batch_requests:
                f.write(orjson.dumps(request) + b"\n")
            print(f"Batch input file saved at: {batch_file_name}")

        from openai import AzureOpenAI, OpenAI

        assert isinstance(rag.llm.openai_client, OpenAI) or isinstance(rag.llm.openai_client, AzureOpenAI)

        # submit input files
        file_obj = rag.llm.openai_client.files.create(file=open(batch_file_name, "rb"), purpose="batch")
        print(file_obj)
        file_ids.append(file_obj.id)

    # For each file id, create a batch job
    batch_ids = []
    batch_responses = []
    for file_id in tqdm(file_ids, desc="Submitting batch jobs"):
        batch_response = rag.llm.openai_client.batches.create(
            input_file_id=file_id,
            endpoint="/chat/completions",
            completion_window="24h",
        )

        # Save batch ID for later use
        batch_id = batch_response.id
        batch_ids.append(batch_id)
        batch_responses.append(batch_response)
        print("Batch job created", batch_response)

    # Submission is done, now we can check the status of the batch jobs
    pending_batches = list(batch_ids)
    while pending_batches:
        for bid in pending_batches.copy():
            batch_response = rag.llm.openai_client.batches.retrieve(bid)
            status = batch_response.status
            print(f"{datetime.datetime.now()} Batch Id: {bid},  Status: {status}")
            if status in ("completed", "failed", "canceled"):
                pending_batches.remove(bid)
                if status == "failed":
                    for error in batch_response.errors.data:
                        print(f"Error code {error.code} Message {error.message}")
        time.sleep(60)

    print("All batch jobs are done (completed, failed or canceled)")
    print(batch_responses)


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

    json_ner_requests = {}

    done = False
    for sample_idx in tqdm(range(0, len(samples)), desc="Processing samples"):
        dataset_name = f"longmemeval_s_{sample_idx}"
        config.dataset = dataset_name
        config.__post_init__()

        working_dir = os.path.join(config.save_dir, f"{dataset_name}_{llm_label}_{embedding_label}")
        rag = ReMem(global_config=config, working_dir=working_dir)
        sample = samples[sample_idx]

        docs = sample["haystack_docs"]
        preprocessed_chunks = rag.text_preprocessor.batch_preprocess_doc(input=docs)

        chunk_contents = [chunk["content"] for chunks in preprocessed_chunks for chunk in chunks]  # a list of strings

        namespace = "chunk"
        nodes_dict = {}
        json_ner_requests[sample_idx] = {}
        for text in chunk_contents:
            assert isinstance(text, str)
            chunk_id = compute_mdhash_id(text, prefix=namespace + "-")
            ner_input_message = rag.prompt_template_manager.render(name="ner", passage=text)
            custom_id = f"NER_{sample_idx}_{chunk_id}"
            if custom_id in json_ner_requests[sample_idx]:
                assert ner_input_message == json_ner_requests[sample_idx][custom_id]["body"]["messages"]
                continue
            json_ner_requests[sample_idx][custom_id] = make_json_request(
                custom_id,
                ner_input_message,
                args.llm_name + "-global-batch",
                json_mode=False,
                json_schema=ner_json_schema,
                use_azure=args.use_azure,
            )

            if args.debug and len(json_ner_requests[sample_idx]) >= 20:
                done = True
                break
        if done:
            break

    os.makedirs("outputs/longmemeval", exist_ok=True)

    # split json_ner_requests into multiple files, where each one contains at most 100,000 requests
    batch_size = 100000 if args.use_azure else 50000
    all_requests = []
    for sample_requests in json_ner_requests.values():
        all_requests.extend(sample_requests.values())
    print("# of requests:", len(all_requests))

    submit_batches("ner", all_requests, batch_size, rag, args)
