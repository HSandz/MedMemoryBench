import argparse
import platform
import threading
from typing import List, Union

import torch
from fastapi import FastAPI, HTTPException
from openai.types.create_embedding_response import CreateEmbeddingResponse, Usage
from openai.types.embedding import Embedding
from pydantic import BaseModel

app = FastAPI()


def get_best_device():
    """Automatically detect the best device based on platform and availability."""
    system = platform.system().lower()

    if system == "darwin":  # macOS
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


# Define request and response models
class EmbeddingRequest(BaseModel):
    input: Union[str, List[str]]
    model: str
    encoding_format: str


class SentenceTransformerQwen3Model:
    def __init__(
        self, model_name: str = "Qwen/Qwen3-Embedding-0.6B", max_seq_length: int = 4096, multi_gpu=True, device=None
    ):
        # Auto-detect device if not specified
        if device is None:
            device = get_best_device()

        # Adjust multi_gpu setting based on device and availability
        if device in ["mps", "cpu"]:
            multi_gpu = False  # MPS and CPU don't support multi-GPU
        elif device == "cuda" and torch.cuda.device_count() <= 1:
            multi_gpu = False  # Only one GPU available

        # Initialize the model with specified configurations
        print(
            f"Initialized Qwen3 embedding model, multi-GPU: {multi_gpu}, max_seq_length: {max_seq_length}, device: {device}"
        )
        from sentence_transformers import SentenceTransformer

        # Configure model kwargs for better performance
        model_kwargs = {
            "device_map": "auto" if multi_gpu else None,
        }

        # Optionally enable flash_attention_2 for better performance
        try:
            model_kwargs["attn_implementation"] = "flash_attention_2"
            tokenizer_kwargs = {"padding_side": "left"}
            self.model = SentenceTransformer(
                model_name, model_kwargs=model_kwargs, tokenizer_kwargs=tokenizer_kwargs, trust_remote_code=True
            )
        except Exception as e:
            print(f"Flash attention not available, falling back to default: {e}")
            self.model = SentenceTransformer(model_name, trust_remote_code=True, device=device)

        self.model.max_seq_length = max_seq_length
        self.multi_gpu = multi_gpu

        if multi_gpu and device == "cuda" and torch.cuda.device_count() > 1:
            self.pool = self.model.start_multi_process_pool()
        else:
            self.pool = None

    def __del__(self):
        # Clean up resources when the object is deleted
        if hasattr(self, "pool") and self.pool is not None:
            self.model.stop_multi_process_pool(self.pool)

    def encode(self, texts: List[str], instruction: str = "", batch_size: int = 64, use_query_prompt: bool = False):
        """
        Encode the list of texts with optional instruction handling

        Args:
            texts: List of texts to encode
            instruction: Instruction prefix (for compatibility with NV-Embed-v2 API)
            batch_size: Batch size for encoding
            use_query_prompt: Whether to use the "query" prompt for encoding
        """
        print(
            f"Qwen3 encoding, batch size per GPU: {batch_size}, #GPU: {torch.cuda.device_count()}, len to encode: {len(texts)}"
        )

        try:
            if self.pool is not None:
                # Multi-GPU encoding
                if use_query_prompt:
                    emb = self.model.encode_multi_process(
                        texts,
                        self.pool,
                        prompt_name="query",
                        batch_size=batch_size,
                        normalize_embeddings=True,
                        show_progress_bar=True,
                    )
                else:
                    emb = self.model.encode_multi_process(
                        texts, self.pool, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=True
                    )
            else:
                # Single GPU encoding
                if use_query_prompt:
                    emb = self.model.encode(
                        texts,
                        prompt_name="query",
                        batch_size=batch_size,
                        normalize_embeddings=True,
                        show_progress_bar=True,
                    )
                else:
                    emb = self.model.encode(
                        texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=True
                    )
        except Exception as e:
            print("Error in encode:", e)
            print("Type of texts:", type(texts))
            print("Len of texts:", len(texts))
            print("Batch size:", batch_size)
            raise e

        return emb.tolist() if hasattr(emb, "tolist") else emb


@app.post("/v1/embeddings", response_model=CreateEmbeddingResponse)
async def get_embeddings(request: EmbeddingRequest):
    if request.encoding_format != "base64":
        raise HTTPException(status_code=400, detail="Unsupported encoding format")

    # Support multiple Qwen3 embedding models
    supported_models = [
        "Qwen/Qwen3-Embedding-8B",
        "Qwen/Qwen3-Embedding-0.6B",
    ]

    if request.model not in supported_models:
        raise HTTPException(
            status_code=400, detail=f"Unsupported model: {request.model}. Supported models: {supported_models}"
        )

    prompts, instruction = [], ""
    use_query_prompt = False

    if isinstance(request.input, str):
        request.input = [request.input]

    for item in request.input:
        if "<|endofprefix|>" in item:
            prefix, text = item.split("<|endofprefix|>")
            prompts.append(text)
            instruction = prefix
            # For queries, we might want to use the query prompt
            if "query" in prefix.lower() or "question" in prefix.lower():
                use_query_prompt = True
        else:
            prompts.append(item)
            instruction = ""

    embeddings = qwen3_model.encode(prompts, instruction=instruction, use_query_prompt=use_query_prompt)
    embedding = [Embedding(embedding=e, index=i, object="embedding") for i, e in enumerate(embeddings)]
    usage = Usage(prompt_tokens=0, total_tokens=0)
    print("Received # of requests:", len(prompts))
    return CreateEmbeddingResponse(data=embedding, model=request.model, object="list", usage=usage)


# Start the FastAPI server in a separate thread
def start_server(port):
    import uvicorn

    uvicorn.run(app, port=port)


def main():
    """Main function to parse arguments and start the server."""
    # Get default device for the platform
    default_device = get_best_device()

    parser = argparse.ArgumentParser(description="Deploy the FastAPI Qwen3 embedding service.")
    parser.add_argument("--port", type=int, default=8001, help="Port to deploy the service on")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-Embedding-0.6B", help="Model name to use")
    parser.add_argument("--max-seq-length", type=int, default=4096, help="Maximum sequence length")
    parser.add_argument("--multi-gpu", action="store_true", default=True, help="Use multi-GPU processing")
    parser.add_argument(
        "--device", type=str, default=default_device, help=f"Device to use for model (auto-detected: {default_device})"
    )
    args = parser.parse_args()

    # Initialize model with specified arguments
    global qwen3_model
    qwen3_model = SentenceTransformerQwen3Model(
        model_name=args.model, max_seq_length=args.max_seq_length, multi_gpu=args.multi_gpu, device=args.device
    )

    # Print the service URL
    print(f"Starting Qwen3 embedding service on port {args.port}")

    # Start the FastAPI server
    server_thread = threading.Thread(target=start_server, args=(args.port,), daemon=True)
    server_thread.start()
    server_thread.join()


if __name__ == "__main__":
    main()
