
import argparse
import threading
from typing import List, Union

import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from openai.types.create_embedding_response import CreateEmbeddingResponse, Usage
from openai.types.embedding import Embedding
from pydantic import BaseModel
from transformers import AutoModel

app = FastAPI()


# Define request and response models
class EmbeddingRequest(BaseModel):
    input: Union[str, List[str]]
    model: str
    encoding_format: str


class HFNVEmbedV2Model:
    def __init__(self, model_name: str = "nvidia/NV-Embed-v2"):
        print(f"Loading model {model_name}...")
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True, device_map="auto")
        self.max_length = 4096
        print(f"Initialized Hugging Face NVEmbedV2 with model {model_name}, max_length: {self.max_length}")

    def encode(self, prompts: List[str], instruction: str = "", norm=True):
        outputs = self.model.encode(prompts=prompts, instruction=instruction, max_length=self.max_length)
        if norm:
            return F.normalize(outputs, p=2, dim=1).cpu().detach().numpy().tolist()
        return outputs.cpu().detach().numpy().tolist()


class SentenceTransformerNVEmbedV2Model:
    def __init__(
        self, model_name: str = "nvidia/NV-Embed-v2", max_seq_length: int = 4096, multi_gpu=True, device="cuda"
    ):
        # Initialize the model with specified configurations
        print(
            f"Initialized NV-Embed-v2 model, multi-GPU: {multi_gpu}, max_seq_length: {max_seq_length}, device: {device}"
        )
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, trust_remote_code=True, device=device)
        self.model.max_seq_length = max_seq_length
        self.model.tokenizer.padding_side = "right"
        self.multi_gpu = multi_gpu

        self.pool = self.model.start_multi_process_pool()

    def __del__(self):
        # Clean up resources when the object is deleted
        if hasattr(self, "pool"):
            self.model.stop_multi_process_pool(self.pool)

    def encode(self, texts: List[str], instruction: str, batch_size: int = 64) -> torch.Tensor:
        # Encode the list of texts with instruction as prefix
        if instruction is not None and instruction != "":
            prompt = f"Instruct: {instruction}\nQuery: "
        else:
            prompt = None
        print(
            f"NV-Embed-v2 encoding, batch size per GPU: {batch_size}, #GPU: {torch.cuda.device_count()}, len to encode: {len(texts)}"
        )
        try:
            emb = self.model.encode_multi_process(
                self._add_eos(texts),
                self.pool,
                prompt=prompt,
                batch_size=batch_size,
                normalize_embeddings=True,
                show_progress_bar=True,
            )
        except Exception as e:
            print("Error in encode_list_multi_gpu:", e)
            print("Type of texts:", type(texts))
            print("Len of texts:", len(texts))
            print("Batch size:", batch_size)
        return emb

    def _add_eos(self, input_examples: List[str]) -> List[str]:
        # Adds EOS token to each example
        return [example + self.model.tokenizer.eos_token for example in input_examples]


@app.post("/v1/embeddings", response_model=CreateEmbeddingResponse)
async def get_embeddings(request: EmbeddingRequest):
    if request.encoding_format != "base64":
        raise HTTPException(status_code=400, detail="Unsupported encoding format")
    if request.model != "nvidia/NV-Embed-v2":
        return {"error": f"Unsupported model: {request.model}"}

    prompts, instruction = [], ""
    if isinstance(request.input, str):
        request.input = [request.input]
    for item in request.input:
        if "<|endofprefix|>" in item:
            prefix, text = item.split("<|endofprefix|>")
            prompts.append(text)
            instruction = prefix
        else:
            prompts.append(item)
            instruction = ""
    # print(prompts, instruction)
    # print(instruction)
    embeddings = nv_embed_model.encode(prompts, instruction=instruction)
    embedding = [Embedding(embedding=e, index=0, object="embedding") for e in embeddings]
    usage = Usage(prompt_tokens=0, total_tokens=0)
    print("Received # of requests:", len(prompts))
    return CreateEmbeddingResponse(data=embedding, model=request.model, object="list", usage=usage)


# Start the FastAPI server in a separate thread
def start_server(port):
    import uvicorn

    uvicorn.run(app, port=port)


def main():
    # Initialize Nvidia embedding model
    global nv_embed_model
    # nv_embed_model = SentenceTransformerNVEmbedV2Model()
    nv_embed_model = HFNVEmbedV2Model()

    """Main function to parse arguments and start the server."""
    parser = argparse.ArgumentParser(description="Deploy the FastAPI embedding service.")
    parser.add_argument("--port", type=int, default=8001, help="Port to deploy the service on")
    args = parser.parse_args()

    # Determine the hostname (or IP address) to use
    # hostname = os.environ.get("SLURM_JOB_NODELIST", socket.gethostname())

    # Print the service URL
    # print(f"Starting service at: http://{hostname}:{args.port}")

    # Start the FastAPI server
    server_thread = threading.Thread(target=start_server, args=(args.port,), daemon=True)
    server_thread.start()
    server_thread.join()


if __name__ == "__main__":
    main()
