from typing import Dict, List, Optional, Tuple, Union

from remem.utils.misc_utils import QuerySolution

from .base_strategy import RAGStrategy


class PassageTripleStrategy(RAGStrategy):
    """
    Default RAG strategy for standard OpenIE-based extraction.
    """

    def index(self, docs: List[str]) -> None:
        """Index documents using standard OpenIE approach."""
        # Call the original index logic from ReMem.index_original()
        self.remem.index_original(docs)

    def rag_for_qa(
        self,
        queries: Union[List[str], List[QuerySolution]],
        num_to_retrieve: int = 5,
        gold_answers: Optional[List[List[str]]] = None,
        gold_docs: Optional[List[List[str]]] = None,
        metrics: Tuple[str, ...] = ("qa_em", "qa_f1", "retrieval_recall"),
        question_metadata: Optional[List[Dict]] = None,
        to_save: bool = True,
        **kwargs,
    ) -> Tuple[List[QuerySolution], List[str], List[Dict], Dict, Dict]:
        """Perform QA using standard RAG approach."""
        if not isinstance(queries[0], QuerySolution):
            query_solutions = self.remem.retrieve(queries=queries)
        else:
            query_solutions = queries

        if question_metadata is not None:
            for idx, q in enumerate(query_solutions):
                q.question_metadata = question_metadata[idx]

        # Evaluating Retrieval
        qa_evaluators, retrieval_evaluators = self.remem.get_evaluators(gold_answers, gold_docs, metrics)
        overall_retrieval_metrics = self.remem.evaluate_retrieval(gold_docs, query_solutions, retrieval_evaluators)

        # Performing QA
        query_solutions, all_response_message, all_metadata = self.remem.qa(query_solutions)

        # Evaluating QA
        overall_qa_metrics = self.remem.evaluate_qa(gold_answers, qa_evaluators, query_solutions, question_metadata)

        # Save retrieval and QA results
        if to_save:
            self.remem.save_rag_results(
                gold_answers, gold_docs, query_solutions, overall_qa_metrics, overall_retrieval_metrics
            )

        return query_solutions, all_response_message, all_metadata, overall_retrieval_metrics, overall_qa_metrics

    def retrieve_each_query(self, query: str, return_chunk: Optional[str] = None):
        """Use the default retrieve_each_query implementation."""
        return self.remem.retrieve_each_query_original(query, return_chunk)
