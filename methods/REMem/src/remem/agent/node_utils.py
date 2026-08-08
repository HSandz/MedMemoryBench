"""
Shared utility functions for node operations in ReMem Agent
"""

from typing import Any, Dict, List, Set


class NodeUtils:
    """Utility functions for node operations and content handling."""

    @staticmethod
    def get_node_type(node_key: str, remem_instance=None) -> str:
        """Determine node type from node key or graph metadata."""
        try:
            if (
                remem_instance
                and hasattr(remem_instance, "episodic_embedding_stores")
                and remem_instance.episodic_embedding_stores
            ):
                for entry_type, store in remem_instance.episodic_embedding_stores.items():
                    if node_key in store.get_all_ids():
                        return entry_type
        except:
            pass

        # Fallback based on node key prefix
        if node_key.startswith("gists"):
            return "gists"
        elif node_key.startswith("facts"):
            return "facts"
        elif node_key.startswith("entity"):
            return "entity"
        else:
            return "verbatim"

    @staticmethod
    def filter_nodes_by_type(nodes: Set[str], node_type: str, remem_instance=None) -> List[str]:
        """Filter nodes by type from a set of nodes."""
        filtered = []
        for node in nodes:
            if NodeUtils.get_node_type(node, remem_instance) == node_type:
                filtered.append(node)
        return filtered

    @staticmethod
    def get_content_preview(node_name: str, node_chunks_dict: Dict[str, str], max_length: int = 200) -> str:
        """Get a preview of node content."""
        if node_name in node_chunks_dict:
            content = node_chunks_dict[node_name]
            if len(content) > max_length:
                return content[:max_length] + "..."
            return content
        return "No content available"

    @staticmethod
    def get_content_map_for_nodes(
        node_ids: Set[str], node_chunks_dict: Dict[str, str], max_preview_length: int = 1024
    ) -> Dict[str, str]:
        """Get content map for a set of node IDs."""
        content_map = {}
        for node_id in node_ids:
            content_map[node_id] = NodeUtils.get_content_preview(node_id, node_chunks_dict, max_preview_length)
        return content_map

    @staticmethod
    def collect_node_content_with_scores(
        node_keys: List[str], scores: List[float], step: int, node_chunks_dict: Dict[str, str], remem_instance=None
    ) -> List[Dict[str, Any]]:
        """Collect content from given nodes with embedding scores."""
        passages = []
        for i, node_key in enumerate(node_keys):
            if node_key in node_chunks_dict:
                score = scores[i] if i < len(scores) else 0.0
                node_type = NodeUtils.get_node_type(node_key, remem_instance)

                passages.append(
                    {
                        "node_name": node_key,
                        "content": node_chunks_dict[node_key],
                        "embedding_score": score,  # Add this key for compatibility
                        "step": step,
                        "node_type": node_type,
                    }
                )
        return passages

    @staticmethod
    def get_chunk_appearance_steps(chunk_id: str, retrieved_passages: List[Dict[str, Any]]) -> List[int]:
        """Get all steps where a specific chunk appeared in retrieval results."""
        steps = []
        for passage in retrieved_passages:
            if passage.get("node_name") == chunk_id:
                step = passage.get("step")
                if step is not None and step not in steps:
                    steps.append(step)
        return sorted(steps)

    @staticmethod
    def get_available_entries(remem_instance, node_chunks_dict: Dict[str, str]) -> Dict[str, List[str]]:
        """Get available entries for each type."""
        if remem_instance and hasattr(remem_instance, "entry_keys") and remem_instance.entry_keys:
            return {entry_type: list(keys) for entry_type, keys in remem_instance.entry_keys.items()}
        else:
            return {"verbatim": list(node_chunks_dict.keys())}

    @staticmethod
    def keyword_search(query: str, node_chunks_dict: Dict[str, str], max_results: int = 3) -> List[str]:
        """Simple keyword-based search fallback."""
        query_lower = query.lower()
        candidates = []

        for node_name, content in node_chunks_dict.items():
            if content and any(word in content.lower() for word in query_lower.split()):
                candidates.append(node_name)

        return candidates[:max_results]
