from .base import BaseEmbeddingModel, EmbeddingConfig


def _get_embedding_client(global_config, embedding_model_name: str = "nvidia/NV-Embed-v2", openai_style_server=True):
    if "text-embedding" in embedding_model_name or openai_style_server:
        from .openai_embedding_client import CacheOpenAIEmbeddingModel

        if openai_style_server and "text-embedding" not in embedding_model_name:  # Local server, using OpenAI style API
            base_url = "http://localhost:8001/v1/"
        else:  # Using OpenAI official embedding API
            base_url = "https://api.openai.com/v1/"
        embedding_client = CacheOpenAIEmbeddingModel(None, global_config, embedding_model_name, base_url=base_url)
    elif "GritLM" in embedding_model_name:
        from .GritLM import GritLMEmbeddingModel

        embedding_client = GritLMEmbeddingModel(global_config, embedding_model_name)
    elif "NV-Embed-v2" in embedding_model_name:
        from .NVEmbedV2 import NVEmbedV2EmbeddingModel

        embedding_client = NVEmbedV2EmbeddingModel(global_config, embedding_model_name)
    else:
        assert False, f"Unknown embedding model name: {embedding_model_name}"
    return embedding_client
