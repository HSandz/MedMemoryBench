import argparse
import hashlib
import json
import os
import sqlite3
from collections import defaultdict

from filelock import FileLock  # make sure the filelock package is installed
from tqdm import tqdm


def prewrite_cache(cache_file_name, cache_entries):
    """
    Pre-populate the SQLite cache with a list of cache entries.

    Each entry in cache_entries should be a tuple:
      (message_input, model, seed, temperature, message_output, metadata)
    """
    lock_file = cache_file_name + ".lock"

    # Ensure mutual exclusion with FileLock
    with FileLock(lock_file):
        conn = sqlite3.connect(cache_file_name)
        c = conn.cursor()

        # Create the cache table if it doesn't exist
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS cache (
                key TEXT PRIMARY KEY,
                message TEXT,
                metadata TEXT
            )
        """
        )

        # Loop through each cache entry to insert it
        for entry in cache_entries:
            message_input, model, seed, temperature, message_output, metadata, response_format = entry

            # Build the key_data dictionary, similar to your wrapper code
            key_data = {
                "messages": message_input,
                "model": model,
                "seed": seed,
                "temperature": temperature,
                "response_format": response_format,
            }
            # Convert to JSON string and compute hash
            key_str = json.dumps(key_data, sort_keys=True, default=str)
            key_hash = hashlib.sha256(key_str.encode("utf-8")).hexdigest()

            # Serialize the metadata dictionary to JSON
            metadata_str = json.dumps(metadata)

            # Insert or replace the row in the cache table
            c.execute(
                "INSERT OR REPLACE INTO cache (key, message, metadata) VALUES (?, ?, ?)",
                (key_hash, message_output, metadata_str),
            )

        conn.commit()
        conn.close()


def process_request_response_files(request_response_files, llm_name="gpt-4o-mini"):
    dataset_id_to_cache_entries = defaultdict(list)

    for request_filename, response_filename in request_response_files:
        with open(request_filename, "rb") as f:
            requests = f.readlines()
            requests = [json.loads(request) for request in requests]
            id_to_request = {request["custom_id"]: request for request in requests}
        with open(response_filename, "rb") as f:
            responses = f.readlines()
            responses = [json.loads(response) for response in responses]
            id_to_response = {response["custom_id"]: response for response in responses}
        print(
            f"READ {len(requests)} requests and {len(responses)} responses from {request_filename} and {response_filename}"
        )

        for id, response in id_to_response.items():
            dataset_id = id.split("_")[1]
            request = id_to_request[id]
            r = request["body"]
            (
                message_input,
                model,
                seed,
                temperature,
            ) = (
                r["messages"],
                r["model"],
                r["seed"],
                r["temperature"],
            )
            assert r["model"] == f"{llm_name}-global-batch"

            response_body = response["response"]["body"]
            if len(response_body["choices"]) == 0:
                continue
            message_output = response_body["choices"][0]["message"].get("content", None)
            if message_output is None:
                continue
            metadata = {
                "prompt_tokens": response_body["usage"]["prompt_tokens"],
                "completion_tokens": response_body["usage"]["completion_tokens"],
                "finish_reason": response_body["choices"][0]["finish_reason"],
            }
            dataset_id_to_cache_entries[dataset_id].append(
                (message_input, llm_name, seed, temperature, message_output, metadata)
            )

    for dataset_id, cache_entries in dataset_id_to_cache_entries.items():
        cache_file_name = f"outputs/longmemeval/longmemeval_s_{dataset_id}/llm_cache/{llm_name}_cache.sqlite"
        os.makedirs(os.path.dirname(cache_file_name), exist_ok=True)
        prewrite_cache(cache_file_name, cache_entries)
        print(f"Saved {len(cache_entries)} entries to {cache_file_name}", end="; ")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--request_files", nargs="+", help="List of input file paths (OpenAI Batch submission)")
    parser.add_argument("--file_ids", nargs="+", help="List of output file IDs")
    parser.add_argument("--use_azure", action="store_true", help="Use Azure for OpenAI")
    args = parser.parse_args()

    assert len(args.request_files) >= len(
        args.file_ids
    )  # In case there's error, the number of request should be >= the number of response

    if not args.use_azure:
        from openai import OpenAI

        client = OpenAI()
    else:
        from openai import AzureOpenAI

        client = AzureOpenAI()

    llm_name = "gpt-4o-mini"
    dataset_id_to_cache_entries = defaultdict(list)

    id_to_request = {}
    for request_file_path in args.request_files:
        requests = []
        with open(request_file_path, "rb") as f:
            lines = f.readlines()  # read jsonl
            requests = [json.loads(line) for line in lines]

        llm_name = requests[0]["body"]["model"].replace("-global-batch", "")
        for request in requests:
            id_to_request[request["custom_id"]] = request

    id_to_response = {}
    for output_file_id in tqdm(args.file_ids, desc="Retrieving batch output files"):  # for each batch
        file_response = client.files.content(output_file_id)
        raw_responses = file_response.text.strip().split("\n")
        print(f"Read {len(raw_responses)} responses from {output_file_id}")

        for raw_response in raw_responses:
            json_response = json.loads(raw_response)
            custom_id = json_response["custom_id"]
            content = json_response["response"]["body"]["choices"][0]["message"].get("content", None)
            if content is None:
                content = "refused"
            print(custom_id, content)
            if content != "refused":
                id_to_response[custom_id] = json_response
    # end each batch

    for id, response in id_to_response.items():
        sample_idx = id.split("_")[1]
        request = id_to_request[id]
        request_body = request["body"]
        (
            message_input,
            model,
            seed,
            temperature,
        ) = (
            request_body["messages"],
            request_body["model"],
            request_body["seed"],
            request_body["temperature"],
        )
        assert request_body["model"] == f"{llm_name}-global-batch"
        response_format = request_body["response_format"]

        response_body = response["response"]["body"]
        if len(response_body["choices"]) == 0:
            continue
        message_output = response_body["choices"][0]["message"].get("content", None)
        if message_output is None:
            continue
        metadata = {
            "prompt_tokens": response_body["usage"]["prompt_tokens"],
            "completion_tokens": response_body["usage"]["completion_tokens"],
            "finish_reason": response_body["choices"][0]["finish_reason"],
        }
        dataset_id_to_cache_entries[sample_idx].append(
            (message_input, llm_name, seed, temperature, message_output, metadata, response_format)
        )
    # end each response

    print("# of request samples:", len(id_to_request))
    print("# of response samples:", len(id_to_response))

    for sample_idx, cache_entries in dataset_id_to_cache_entries.items():
        cache_file_name = f"outputs/longmemeval/longmemeval_s_{sample_idx}/llm_cache/{llm_name}_cache.sqlite"
        os.makedirs(os.path.dirname(cache_file_name), exist_ok=True)
        prewrite_cache(cache_file_name, cache_entries)
        print(f"Saved {len(cache_entries)} entries to {cache_file_name}", end="; ")
