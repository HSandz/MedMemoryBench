import logging
from typing import List, Optional, Union

import numpy as np
import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)


def compute_embedding_similarity(
    query_embedding: Union[np.ndarray, torch.Tensor],
    doc_embedding: Union[np.ndarray, torch.Tensor],
    similarity_method: str = "dot_product",
) -> float:
    """
    Compute similarity between query and document embeddings.

    Args:
        query_embedding: Query embedding vector
        doc_embedding: Document embedding vector
        similarity_method: Similarity computation method ("dot_product", "cosine")

    Returns:
        Similarity score as float

    Raises:
        ValueError: If embeddings have incompatible dimensions or unsupported similarity method
    """
    # Convert to numpy arrays if needed
    if isinstance(query_embedding, torch.Tensor):
        query_embedding = query_embedding.cpu().numpy()
    if isinstance(doc_embedding, torch.Tensor):
        doc_embedding = doc_embedding.cpu().numpy()

    # Ensure embeddings are 1D arrays
    query_embedding = np.asarray(query_embedding).flatten()
    doc_embedding = np.asarray(doc_embedding).flatten()

    # Check dimension compatibility
    if query_embedding.shape[0] != doc_embedding.shape[0]:
        raise ValueError(f"Embedding dimensions don't match: {query_embedding.shape[0]} vs {doc_embedding.shape[0]}")

    if similarity_method == "dot_product":
        return float(np.dot(query_embedding, doc_embedding))
    elif similarity_method == "cosine":
        # Cosine similarity = dot product of normalized vectors
        query_norm = np.linalg.norm(query_embedding)
        doc_norm = np.linalg.norm(doc_embedding)
        if query_norm == 0 or doc_norm == 0:
            return 0.0
        return float(np.dot(query_embedding, doc_embedding) / (query_norm * doc_norm))
    else:
        raise ValueError(f"Unsupported similarity method: {similarity_method}")


def get_node_query_similarity(
    node_key: str, query: str, remem_instance, query_embedding: Optional[np.ndarray] = None, logger=None
) -> float:
    """
    Get embedding similarity between a specific node and query.

    This is a higher-level function that handles ReMem-specific logic
    for retrieving embeddings and computing similarity.

    Args:
        node_key: Key identifying the node in the embedding store
        query: Query string
        remem_instance: ReMem instance with embedding stores
        query_embedding: Pre-computed query embedding (optional)
        logger: Logger instance for error reporting

    Returns:
        Similarity score between node and query

    Raises:
        ValueError: If node type is unsupported or node not found
        RuntimeError: If ReMem instance is not properly initialized
    """
    try:
        # Determine node type from the key prefix
        node_type = node_key.split("-")[0]

        # Handle different node types - map to the correct store names
        if node_type.startswith("facts"):
            store_name = "facts"
            key = "triple"
            linking_method = "query_to_triple"
        elif node_type in ["gists", "entity", "verbatim"]:
            store_name = node_type
            key = "passage"
            linking_method = "query_to_passage"
        else:
            # Fallback for unknown node types
            raise ValueError(f"Unsupported node type: {node_type}")

        if not (remem_instance and hasattr(remem_instance, "episodic_embedding_stores")):
            raise RuntimeError("ReMem instance does not have episodic embedding stores initialized")

        store = remem_instance.episodic_embedding_stores.get(store_name)
        if not store or node_key not in store.hash_id_to_idx:
            raise ValueError(f"Node {node_key} not found in store {store_name}")

        # Get or compute query embedding
        if query_embedding is None:
            if hasattr(remem_instance, "query_to_embedding") and remem_instance.query_to_embedding:
                query_embedding = remem_instance.query_to_embedding.get(key, {}).get(query)

            if query_embedding is None:
                from remem.prompts.linking import get_query_instruction

                query_embedding = remem_instance.embedding_model.batch_encode(
                    [query], instruction=get_query_instruction(linking_method), norm=True
                )[0]
                # batch_encode returns array, take first element

        # Get node embedding
        node_embedding = store.get_embedding(node_key)

        # Compute similarity using the utility function
        similarity_score = compute_embedding_similarity(query_embedding, node_embedding, "dot_product")

        return similarity_score

    except Exception as e:
        if logger:
            logger.warning(f"Error getting similarity for {node_key}: {e}")
        raise e


def retrieve_knn(
    query_ids: List[str], key_ids: List[str], query_vecs, key_vecs, k=2047, query_batch_size=1000, key_batch_size=10000
):
    """
    Retrieve the top-k nearest neighbors for each query id from the key ids.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if len(key_vecs) == 0:
        return {}

    query_vecs = torch.tensor(np.array(query_vecs), dtype=torch.float32)
    query_vecs = torch.nn.functional.normalize(query_vecs, dim=1)

    key_vecs = torch.tensor(np.array(key_vecs), dtype=torch.float32)
    key_vecs = torch.nn.functional.normalize(key_vecs, dim=1)

    results = {}

    def get_batches(vecs, batch_size):
        for i in range(0, len(vecs), batch_size):
            yield vecs[i : i + batch_size], i

    for query_batch, query_batch_start_idx in tqdm(
        get_batches(vecs=query_vecs, batch_size=query_batch_size),
        total=(len(query_vecs) + query_batch_size - 1) // query_batch_size,  # Calculate total batches
        desc="KNN for Queries",
    ):
        query_batch = query_batch.clone().detach()
        query_batch = query_batch.to(device)

        batch_topk_sim_scores = []
        batch_topk_indices = []

        offset_keys = 0

        for key_batch, key_batch_start_idx in get_batches(vecs=key_vecs, batch_size=key_batch_size):
            key_batch = key_batch.to(device)
            actual_key_batch_size = key_batch.size(0)

            similarity = torch.mm(query_batch, key_batch.T)

            topk_sim_scores, topk_indices = torch.topk(
                similarity, min(k, actual_key_batch_size), dim=1, largest=True, sorted=True
            )

            topk_indices += offset_keys

            batch_topk_sim_scores.append(topk_sim_scores)
            batch_topk_indices.append(topk_indices)

            del similarity
            key_batch = key_batch.cpu()
            torch.cuda.empty_cache()

            offset_keys += actual_key_batch_size
        # end for each kb batch

        batch_topk_sim_scores = torch.cat(batch_topk_sim_scores, dim=1)
        batch_topk_indices = torch.cat(batch_topk_indices, dim=1)

        final_topk_sim_scores, final_topk_indices = torch.topk(
            batch_topk_sim_scores, min(k, batch_topk_sim_scores.size(1)), dim=1, largest=True, sorted=True
        )
        final_topk_indices = final_topk_indices.cpu()
        final_topk_sim_scores = final_topk_sim_scores.cpu()

        for i in range(final_topk_indices.size(0)):
            query_relative_idx = query_batch_start_idx + i
            query_idx = query_ids[query_relative_idx]

            final_topk_indices_i = final_topk_indices[i]
            final_topk_sim_scores_i = final_topk_sim_scores[i]

            query_to_topk_key_relative_ids = batch_topk_indices[i][final_topk_indices_i]
            query_to_topk_key_ids = [key_ids[idx] for idx in query_to_topk_key_relative_ids.cpu().numpy()]
            results[query_idx] = (query_to_topk_key_ids, final_topk_sim_scores_i.numpy().tolist())

        query_batch = query_batch.cpu()
        torch.cuda.empty_cache()
    # end for each query batch

    return results


def get_query_entry_scores(
    query: str, entry: str, remem_instance, return_list: bool = True
) -> Union[np.ndarray, List[float]]:
    """
    Calculate similarity scores between query and entries.

    This is a general utility function that can be used across different RAG strategies
    to compute similarity scores between a query and entries in the knowledge base.

    Args:
        query: The search query string
        entry: The entry type ("facts", "verbatim", "gists", etc.)
        remem_instance: The ReMem instance containing embeddings and models
        return_list: If True, return List[float]; if False, return np.ndarray

    Returns:
        Union[np.ndarray, List[float]]: Normalized similarity scores between query and entries

    Raises:
        AssertionError: If the entry is empty and not indexed
    """
    if entry == "facts":
        key = "triple"
        linking_method = "query_to_triple"
    else:
        key = "passage"
        linking_method = "query_to_passage"

    query_embedding = remem_instance.query_to_embedding[key].get(query, None)
    if query_embedding is None:
        from remem.prompts.linking import get_query_instruction

        query_embedding = remem_instance.embedding_model.batch_encode(
            query, instruction=get_query_instruction(linking_method), norm=True
        )

    entry_embedding = remem_instance.entry_embeddings[entry]
    assert len(entry_embedding) > 0, f"Entry {entry} is empty. Please index the data first."

    # Initialize scores to zeros in case of error
    query_entry_scores = np.zeros(len(entry_embedding))

    try:
        # Check for NaN values in embeddings
        if np.any(np.isnan(entry_embedding)):
            num_nan = np.sum(np.isnan(entry_embedding))
            logger.error(f"{num_nan} NaN values found in embeddings for entry {entry}, shape: {entry_embedding.shape}")
            entry_embedding = np.nan_to_num(entry_embedding, nan=0.0)
        if np.any(np.isnan(query_embedding)):
            num_nan = np.sum(np.isnan(query_embedding))
            logger.error(f"{num_nan} NaN values found in query embedding, shape: {query_embedding.shape}")
            query_embedding = np.nan_to_num(query_embedding, nan=0.0)  # Replace NaN with 0

            return query_entry_scores  # Return zeros

        query_entry_scores = np.dot(entry_embedding, query_embedding.T).flatten()
        query_entry_scores = np.squeeze(query_entry_scores) if query_entry_scores.ndim == 2 else query_entry_scores

        # Check for NaN values in the result
        if np.any(np.isnan(query_entry_scores)):
            num_nan = np.sum(np.isnan(query_entry_scores))
            logger.error(f"{num_nan} NaN values found in query entry scores, shape: {query_entry_scores.shape}")
            query_entry_scores = np.nan_to_num(query_entry_scores, nan=0.0)  # Replace NaN with 0

    except Exception as e:
        logger.error(
            f"Error in query entry score calculation: {e}, entry dim: {entry_embedding.shape}, query dim: {query_embedding.shape}"
        )
        # Return zeros on error
        return query_entry_scores

    from remem.utils.misc_utils import min_max_normalize

    try:
        query_entry_scores = min_max_normalize(query_entry_scores)
    except Exception as e:
        error_msg = f"Error in normalization: {e}"
        if logger:
            logger.error(error_msg)
        else:
            print(error_msg)
        # If normalization fails, return unnormalized scores (better than zeros)

    # Convert to list if requested (default behavior)
    if return_list:
        return query_entry_scores.tolist()
    return query_entry_scores
