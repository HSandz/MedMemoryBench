import argparse
from typing import List, Union

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from openai.types.create_embedding_response import CreateEmbeddingResponse, Usage
from openai.types.embedding import Embedding
from pydantic import BaseModel
from torch import Tensor
from transformers import AutoModel, AutoTokenizer

# Add src to path for imports
from remem.utils.logging_utils import get_logger

# Initialize logger
logger = get_logger(__name__)

app = FastAPI()


# Define request and response models
class EmbeddingRequest(BaseModel):
    input: Union[str, List[str]]
    model: str
    encoding_format: str


def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    """Pool embeddings using the last token strategy"""
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def get_detailed_instruct(task_description: str, query: str) -> str:
    """Format instruction and query according to Qwen3 requirements"""
    return f"Instruct: {task_description}\nQuery: {query}"


class HFQwen3EmbeddingModel:
    """HuggingFace Transformers implementation of Qwen3 Embedding"""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Embedding-8B",
        max_length: int = 8192,
        use_flash_attention: bool = True,
        device_map: str = "auto",
    ):
        self.model_name = model_name
        logger.info(f"Loading HF model {model_name}...")

        # Initialize tokenizer with left padding (recommended for Qwen3)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")

        # Model kwargs for optimization
        model_kwargs = {}
        if use_flash_attention and torch.cuda.is_available():
            model_kwargs["attn_implementation"] = "flash_attention_2"
            model_kwargs["torch_dtype"] = torch.float16
            logger.info("Using flash_attention_2 and float16")

        if device_map:
            model_kwargs["device_map"] = device_map

        # Load model
        self.model = AutoModel.from_pretrained(model_name, **model_kwargs)
        self.max_length = max_length

        # Move to CUDA if available and device_map not used
        if not device_map and torch.cuda.is_available():
            self.model = self.model.cuda()

        logger.info(f"HF Qwen3 model loaded successfully with max_length={max_length}")

    def encode(self, texts: List[str], instruction: str = "", batch_size: int = 32) -> List[List[float]]:
        """Encode texts using HuggingFace transformers approach"""
        # Handle empty inputs
        if not texts or all(not t.strip() for t in texts):
            logger.warning("Empty or whitespace-only inputs detected, returning zero embeddings")
            return [[0.0] * 4096 for _ in texts]  # Qwen3-Embedding-8B has 4096 dimensions

        # Prepare inputs with instruction if provided
        if instruction:
            processed_texts = [get_detailed_instruct(instruction, text) for text in texts]
        else:
            processed_texts = texts

        all_embeddings = []

        # Process in batches
        for i in range(0, len(processed_texts), batch_size):
            batch_texts = processed_texts[i : i + batch_size]

            # Tokenize
            batch_dict = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )

            # Move to device
            batch_dict = batch_dict.to(self.model.device)

            with torch.no_grad():
                outputs = self.model(**batch_dict)

                # Pool embeddings using last token
                embeddings = last_token_pool(outputs.last_hidden_state, batch_dict["attention_mask"])

                # Check for NaN or infinite values before normalization
                if torch.isnan(embeddings).any() or torch.isinf(embeddings).any():
                    logger.warning("NaN or infinite values detected in model output, replacing with zeros")
                    embeddings = torch.where(
                        torch.isnan(embeddings) | torch.isinf(embeddings), torch.zeros_like(embeddings), embeddings
                    )

                # Normalize embeddings with safe normalization
                try:
                    embeddings = F.normalize(embeddings, p=2, dim=1)
                except Exception as e:
                    logger.warning(f"F.normalize failed: {e}, using fallback normalization")
                    eps = 1e-6
                    norms = torch.norm(embeddings, p=2, dim=1, keepdim=True)
                    norms = torch.clamp(norms, min=eps)
                    embeddings = embeddings / norms

                # Final check for NaN values after normalization
                if torch.isnan(embeddings).any():
                    logger.warning("NaN values detected after normalization, replacing with zeros")
                    embeddings = torch.where(torch.isnan(embeddings), torch.zeros_like(embeddings), embeddings)

                # Convert to numpy and check for issues
                embeddings_numpy = embeddings.cpu().detach().numpy()

                # Final validation
                if np.isnan(embeddings_numpy).any() or np.isinf(embeddings_numpy).any():
                    logger.warning("Non-finite values in final embeddings, cleaning up")
                    embeddings_numpy = np.where(
                        np.isnan(embeddings_numpy) | np.isinf(embeddings_numpy), 0.0, embeddings_numpy
                    )

                # Check if we're getting all-zero vectors (which might indicate a problem)
                zero_count = 0
                for emb in embeddings_numpy:
                    if np.allclose(emb, 0.0, atol=1e-8):
                        zero_count += 1

                if zero_count > 0:
                    logger.warning(
                        f"Generated {zero_count} zero or near-zero embeddings out of {len(embeddings_numpy)} in this batch"
                    )
                    if zero_count == len(embeddings_numpy):
                        logger.error("All embeddings in batch are zero! This likely indicates a model or input problem")

                # Convert to list
                batch_embeddings = embeddings_numpy.tolist()
                all_embeddings.extend(batch_embeddings)

        return all_embeddings


class SentenceTransformerQwen3Model:
    """SentenceTransformers implementation of Qwen3 Embedding"""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Embedding-8B",
        max_seq_length: int = 8192,
        use_flash_attention: bool = True,
        device: str = "cuda",
        multi_gpu: bool = True,
    ):
        logger.info(f"Loading SentenceTransformer model {model_name}...")

        from sentence_transformers import SentenceTransformer

        # Model kwargs for optimization
        model_kwargs = {}
        tokenizer_kwargs = {"padding_side": "left"}

        if use_flash_attention and torch.cuda.is_available():
            model_kwargs.update({"attn_implementation": "flash_attention_2", "device_map": "auto"})
            logger.info("Using flash_attention_2 with device_map=auto")

        # Initialize model
        self.model = SentenceTransformer(
            model_name,
            trust_remote_code=True,
            device=device,
            model_kwargs=model_kwargs,
            tokenizer_kwargs=tokenizer_kwargs,
        )

        self.model.max_seq_length = max_seq_length
        self.multi_gpu = multi_gpu and torch.cuda.device_count() > 1

        # Initialize multi-process pool for multi-GPU
        if self.multi_gpu:
            try:
                self.pool = self.model.start_multi_process_pool()
                logger.info(f"Multi-GPU enabled with {torch.cuda.device_count()} GPUs")
            except Exception as e:
                logger.warning(f"Multi-GPU setup failed: {e}, falling back to single GPU")
                self.multi_gpu = False
                self.pool = None
        else:
            self.pool = None

        logger.info("SentenceTransformer Qwen3 model loaded successfully")

    def __del__(self):
        """Clean up resources when the object is deleted"""
        if hasattr(self, "pool") and self.pool is not None:
            try:
                self.model.stop_multi_process_pool(self.pool)
            except:
                pass

    def encode(self, texts: List[str], instruction: str = "", batch_size: int = 64) -> List[List[float]]:
        """Encode texts using SentenceTransformers approach"""
        # Handle empty inputs
        if not texts or all(not t.strip() for t in texts):
            logger.warning("Empty or whitespace-only inputs detected, returning zero embeddings")
            return [[0.0] * 4096 for _ in texts]  # Qwen3-Embedding-8B has 4096 dimensions

        # Prepare prompt if instruction is provided
        if instruction:
            prompt = f"Instruct: {instruction}\nQuery: "
        else:
            # Use default query prompt if available
            prompt = None

        logger.info(f"Encoding {len(texts)} texts with batch_size={batch_size}")

        try:
            if self.multi_gpu and self.pool is not None:
                # Multi-GPU encoding
                embeddings = self.model.encode_multi_process(
                    texts,
                    self.pool,
                    prompt=prompt,
                    batch_size=batch_size,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
            else:
                # Single GPU or CPU encoding
                encode_kwargs = {"batch_size": batch_size, "normalize_embeddings": True, "show_progress_bar": False}

                if prompt:
                    encode_kwargs["prompt"] = prompt
                elif hasattr(self.model, "prompts") and "query" in self.model.prompts:
                    encode_kwargs["prompt_name"] = "query"

                embeddings = self.model.encode(texts, **encode_kwargs)

            # Validate embeddings
            if hasattr(embeddings, "tolist"):
                embeddings_array = embeddings
            else:
                embeddings_array = np.array(embeddings)

            # Check for NaN or infinite values
            if np.isnan(embeddings_array).any() or np.isinf(embeddings_array).any():
                logger.warning("NaN or infinite values detected in embeddings, cleaning up")
                embeddings_array = np.where(
                    np.isnan(embeddings_array) | np.isinf(embeddings_array), 0.0, embeddings_array
                )

            # Check for zero vectors
            zero_count = 0
            for i, emb in enumerate(embeddings_array):
                if np.allclose(emb, 0.0, atol=1e-8):
                    zero_count += 1

            if zero_count > 0:
                logger.warning(f"Generated {zero_count} zero or near-zero embeddings out of {len(embeddings_array)}")
                if zero_count == len(embeddings_array):
                    logger.error("All embeddings are zero! This likely indicates a model or input problem")

            # Convert to list format
            return embeddings_array.tolist()

        except Exception as e:
            logger.error(f"Error in encode: {e}")
            raise


@app.post("/v1/embeddings", response_model=CreateEmbeddingResponse)
async def get_embeddings(request: EmbeddingRequest):
    """Handle embedding requests compatible with OpenAI API format"""
    if request.encoding_format != "base64":
        raise HTTPException(status_code=400, detail="Unsupported encoding format")
    if request.model != "Qwen/Qwen3-Embedding-8B":
        raise HTTPException(status_code=400, detail=f"Unsupported model: {request.model}")

    # Normalize input to list
    if isinstance(request.input, str):
        input_texts = [request.input]
    else:
        input_texts = request.input

    if not input_texts:
        raise HTTPException(status_code=400, detail="No input provided")

    prompts, instruction = [], ""

    # Parse input for instruction prefix
    for item in input_texts:
        if not isinstance(item, str):
            raise HTTPException(status_code=400, detail="All inputs must be strings")

        if "<|endofprefix|>" in item:
            prefix, text = item.split("<|endofprefix|>", 1)
            prompts.append(text)
            instruction = prefix
        else:
            prompts.append(item)

    try:
        # Get embeddings from the global model
        embeddings = qwen3_model.encode(prompts, instruction=instruction)

        # Create response format
        embedding_objects = [Embedding(embedding=emb, index=i, object="embedding") for i, emb in enumerate(embeddings)]

        usage = Usage(prompt_tokens=0, total_tokens=0)

        logger.info(f"Successfully processed {len(prompts)} texts")
        return CreateEmbeddingResponse(data=embedding_objects, model=request.model, object="list", usage=usage)

    except Exception as e:
        logger.error(f"Error generating embeddings: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating embeddings: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model": "Qwen/Qwen3-Embedding-8B"}


@app.get("/debug")
async def debug_info():
    """Debug endpoint to check model and system status"""
    try:
        # Test embedding with a simple input
        test_texts = ["This is a test sentence for debugging"]
        test_embeddings = qwen3_model.encode(test_texts, instruction="")

        # Analyze the test embedding
        test_emb = test_embeddings[0]
        is_all_zero = all(abs(x) < 1e-8 for x in test_emb)
        has_nan = any(not isinstance(x, (int, float)) or not np.isfinite(x) for x in test_emb)
        emb_norm = np.linalg.norm(test_emb)

        debug_info = {
            "model_type": type(qwen3_model).__name__,
            "model_loaded": True,
            "test_embedding": {
                "dimension": len(test_emb),
                "is_all_zero": is_all_zero,
                "has_nan_or_inf": has_nan,
                "norm": float(emb_norm),
                "sample_values": test_emb[:5],  # First 5 values for inspection
            },
            "system_info": {
                "torch_version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            },
        }

        # Add model-specific debug info
        if hasattr(qwen3_model, "model"):
            debug_info["model_device"] = str(qwen3_model.model.device)
            debug_info["model_dtype"] = (
                str(qwen3_model.model.dtype) if hasattr(qwen3_model.model, "dtype") else "unknown"
            )

        if hasattr(qwen3_model, "multi_gpu"):
            debug_info["multi_gpu"] = getattr(qwen3_model, "multi_gpu", False)

        return {"status": "success", "debug_info": debug_info}

    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "model_type": type(qwen3_model).__name__ if "qwen3_model" in globals() else "not_loaded",
        }


# Start the FastAPI server
def start_server(port: int, host: str = "0.0.0.0"):
    import uvicorn

    uvicorn.run(app, host=host, port=port, log_level="info")


def main():
    """Main function to parse arguments and start the server"""
    global qwen3_model

    parser = argparse.ArgumentParser(description="Deploy the FastAPI Qwen3 embedding service.")
    parser.add_argument("--port", type=int, default=8001, help="Port to deploy the service on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to deploy the service on")
    parser.add_argument("--max_length", type=int, default=8192, help="Maximum sequence length")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for processing")
    parser.add_argument(
        "--use_hf", action="store_true", help="Use HuggingFace transformers instead of SentenceTransformers"
    )
    parser.add_argument("--use_flash_attn", action="store_true", help="Disable flash attention 2")
    parser.add_argument("--disable_multi_gpu", action="store_true", help="Disable multi-GPU usage")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda, cpu, mps)")

    args = parser.parse_args()

    # Initialize the appropriate model
    if args.use_hf:
        logger.info("Using HuggingFace Transformers implementation")
        qwen3_model = HFQwen3EmbeddingModel(max_length=args.max_length, use_flash_attention=args.use_flash_attn)
    else:
        logger.info("Using SentenceTransformers implementation")
        qwen3_model = SentenceTransformerQwen3Model(
            max_seq_length=args.max_length,
            use_flash_attention=args.use_flash_attn,
            device=args.device,
            multi_gpu=not args.disable_multi_gpu,
        )

    logger.info(f"Starting Qwen3 embedding server on {args.host}:{args.port}, model name: {qwen3_model.model_name}")

    # Start the server
    start_server(args.port, args.host)


if __name__ == "__main__":
    main()
