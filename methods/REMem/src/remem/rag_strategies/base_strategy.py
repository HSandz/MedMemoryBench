from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union

from remem.utils.misc_utils import QuerySolution


class RAGStrategy(ABC):
    """
    Abstract base class for RAG strategies.
    Each strategy handles indexing and QA for a specific extraction method.
    """

    def __init__(self, remem_instance):
        self.remem = remem_instance

    @abstractmethod
    def index(self, docs: List[str]) -> None:
        """
        Index documents using the specific strategy.

        Args:
            docs: List of documents to index
        """
        pass

    @abstractmethod
    def rag_for_qa(
        self,
        queries: Union[List[str], List[QuerySolution]],
        num_to_retrieve: int = 10,
        gold_answers: Optional[List[List[str]]] = None,
        gold_docs: Optional[List[List[str]]] = None,
        metrics: Tuple[str, ...] = ("qa_em", "qa_f1", "retrieval_recall"),
        question_metadata: Optional[List[Dict]] = None,
        **kwargs,
    ) -> Tuple[List[QuerySolution], List[str], List[Dict], Dict, Dict]:
        """
        Perform RAG-based QA using the specific strategy.

        Args:
            queries: Input queries
            num_to_retrieve: Number of documents to retrieve
            gold_answers: Gold standard answers for evaluation
            gold_docs: Gold standard documents for evaluation
            metrics: Evaluation metrics to compute
            question_metadata: Additional metadata for questions
            **kwargs: Strategy-specific parameters

        Returns:
            Tuple containing query solutions, responses, metadata, retrieval metrics, QA metrics
        """
        pass

    def retrieve_each_query(self, query: str, return_chunk: Optional[str] = None):
        """
        Retrieve documents for a single query. Can be overridden by strategies.

        Args:
            query: Input query
            return_chunk: Type of chunk to return (strategy-specific)

        Returns:
            Retrieved documents and scores
        """
        # Default implementation - can be overridden
        return self.remem.retrieve_each_query(query, return_chunk)

    def get_graph_info(self) -> Dict:
        """
        Obtains detailed information about the graph such as the number of nodes,
        triples, and their classifications.

        This method calculates various statistics about the graph based on the
        stores and node-to-node relationships, including counts of phrase and
        passage nodes, total nodes, extracted triples, triples involving passage
        nodes, synonymy triples, and total triples.

        Returns:
            Dict
                A dictionary containing the following keys and their respective values:
                - num_phrase_nodes: The number of unique phrase nodes.
                - num_passage_nodes: The number of unique passage nodes.
                - num_total_nodes: The total number of nodes (sum of phrase and passage nodes).
                - num_extracted_triples: The number of unique extracted triples.
                - num_triples_with_passage_node: The number of triples involving at least one
                  passage node.
                - num_synonymy_triples: The number of synonymy triples (distinct from extracted
                  triples and those with passage nodes).
                - num_total_triples: The total number of triples.
        """
        graph_info = {}

        # get # of phrase nodes
        phrase_nodes_keys = self.remem.phrase_embedding_store.get_all_ids()
        graph_info["num_phrase_nodes"] = len(set(phrase_nodes_keys))

        # get # of passage nodes
        passage_nodes_keys = self.remem.chunk_embedding_store.get_all_ids()
        graph_info["num_passage_nodes"] = len(set(passage_nodes_keys))

        # get # of summarization nodes
        if (
            self.remem.paraphrase_embedding_store is not None
            and len(self.remem.paraphrase_embedding_store.embeddings) > 0
        ):
            summarization_embedding_keys = self.remem.paraphrase_embedding_store.get_all_ids()
            graph_info["num_paraphrase_nodes"] = len(set(summarization_embedding_keys))

        # get # of total nodes
        graph_info["num_total_nodes"] = graph_info["num_phrase_nodes"] + graph_info["num_passage_nodes"]

        # get # of extracted triples
        graph_info["num_extracted_edges"] = len(self.remem.triple_embedding_store.get_all_ids())

        passage_nodes_set = set(passage_nodes_keys)
        num_context_edges = sum(
            1
            for node_pair in self.remem.node_to_node_count
            if node_pair[0] in passage_nodes_set or node_pair[1] in passage_nodes_set
        )
        graph_info["num_context_edges"] = num_context_edges

        num_paraphrase_edges = 0
        if (
            self.remem.paraphrase_embedding_store is not None
            and len(self.remem.paraphrase_embedding_store.embeddings) > 0
        ):
            num_paraphrase_edges = sum(
                1
                for node_pair in self.remem.node_to_node_count
                if node_pair[0] in summarization_embedding_keys or node_pair[1] in summarization_embedding_keys
            )
            graph_info["num_paraphrase_edges"] = num_paraphrase_edges

        graph_info["num_synonymy_edges"] = (
            len(self.remem.node_to_node_count)
            - graph_info["num_extracted_edges"]
            - num_context_edges
            - num_paraphrase_edges
        )

        # get # of total triples
        graph_info["num_total_edges"] = len(self.remem.node_to_node_count)

        return graph_info
