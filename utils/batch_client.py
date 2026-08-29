"""Provider selection for the repository's offline batch transports."""

from pathlib import Path
from typing import Any, Optional

from utils.llm_client import OpenRouterClient
from utils.openrouter_batch import OpenRouterBatchClient
from utils.vertex_batch import VertexBatchClient


def create_batch_client(
    client: Any,
    *,
    gcs_uri: Optional[str],
    manifest_path: Path,
    wait: bool,
    config_hash: str = "",
    poll_interval: int = 30,
    progress_callback=None,
    vertex_batch_class=VertexBatchClient,
):
    """Create the batch transport matching an existing managed LLM client."""
    if isinstance(client, OpenRouterClient):
        return OpenRouterBatchClient.from_openrouter_client(
            client,
            manifest_path=manifest_path,
            wait=wait,
            config_hash=config_hash,
            poll_interval=poll_interval,
            progress_callback=progress_callback,
        )
    return vertex_batch_class.from_gemini_client(
        client,
        gcs_uri=gcs_uri,
        manifest_path=manifest_path,
        wait=wait,
        config_hash=config_hash,
        poll_interval=poll_interval,
        progress_callback=progress_callback,
    )


__all__ = ["create_batch_client"]
