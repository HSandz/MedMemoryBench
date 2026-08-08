import json
import logging
import os
import re
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import igraph as ig
import numpy as np
from tqdm import tqdm

from remem.embedding_model import BaseEmbeddingModel, _get_embedding_client
from remem.embedding_store import EmbeddingStore
from remem.evaluation.qa_bleu import QABleu1Score, QABleu4Score
from remem.evaluation.qa_eval import QAExactMatch, QAF1Score
from remem.evaluation.retrieval_eval import RetrievalNDCGAny, RetrievalRecall, RetrievalRecallAll
from remem.evaluation.retrieval_locomo import RetrievalRecallLocomo
from remem.graph.preprocessing.base import BasePreprocessor
from remem.information_extraction import OpenIE
from remem.information_extraction.openie_vllm_offline import VLLMOfflineOpenIE
from remem.llm import BaseLLM, _get_llm_class
from remem.prompts import PromptTemplateManager
from remem.prompts.linking import get_query_instruction
from remem.rerank import DSPyFilter
from remem.utils.config_utils import BaseConfig
from remem.utils.datetime_utils import parse_flexible_datetime
from remem.utils.embed_utils import retrieve_knn
from remem.utils.misc_utils import (
    NerRawOutput,
    ParaphraseRawOutput,
    QuerySolution,
    TripleRawOutput,
    compute_mdhash_id,
    describe_array,
    extract_phrase_nodes,
    flatten_facts,
    min_max_normalize,
    reformat_openie_results,
    skip_phrase_node,
    text_processing,
)
from remem.utils.results_utils import get_inference_type

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


class ReMem:

    def __init__(
        self,
        global_config: Optional[BaseConfig] = None,
        working_dir: Optional[str] = None,
        llm: Optional[BaseLLM] = None,
        extract_llm: Optional[BaseLLM] = None,
        qa_llm: Optional[BaseLLM] = None,
    ) -> None:
        """
        Initializes an instance of the class and its related components.

        Attributes:
            global_config (BaseConfig): The global configuration settings for the instance. An instance
                of BaseConfig is used if no value is provided.
            working_dir (str): The directory where work-specific files will be stored. A default
                directory is constructed if no value is provided.
            llm (BaseLLM): The language model used for processing based on the global
                configuration settings.
            openie (Union[OpenIE, VLLMOfflineOpenIE]): The Open Information Extraction module
                configured in either online or offline mode based on the global settings.
            graph: The graph instance initialized by the `initialize_graph` method.
            embedding_model (BaseEmbeddingModel): The embedding model associated with the current
                configuration.
            chunk_embedding_store (EmbeddingStore): The embedding store handling chunk embeddings.
            entity_embedding_store (EmbeddingStore): The embedding store handling entity embeddings.
            fact_embedding_store (EmbeddingStore): The embedding store handling fact embeddings.
            prompt_template_manager (PromptTemplateManager): The manager for handling prompt templates
                and roles mappings.
            openie_results_path (str): The file path for storing Open Information Extraction results
                based on the dataset and LLM name in the global configuration.
            rerank_filter (Optional[DSPyFilter]): The filter responsible for reranking information
                when a rerank file path is specified in the global configuration.
            ready_to_retrieve (bool): A flag indicating whether the system is ready for retrieval
                operations.

        Parameters:
            global_config: The global configuration object. Defaults to None, leading to initialization
                of a new BaseConfig object.
            working_dir: The directory for storing working files. Defaults to None, constructing a default
                directory based on the class name and timestamp.
        """
        self.global_config = global_config if global_config else BaseConfig()

        _print_config = ",\n  ".join([f"{k} = {v}" for k, v in asdict(self.global_config).items()])
        logger.debug(f"ReMem init with config:\n  {_print_config}\n")

        if working_dir is None:
            logger.debug(f"working_dir not assigned for {self.__class__.__name__}. Using the default.")
            self.working_dir = os.path.join(
                self.global_config.save_dir, f"{self.__class__.__name__}_{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}"
            )
        else:
            self.working_dir = working_dir

        if not os.path.exists(self.working_dir):  # if the working_dir doesn't exist, create this dir
            logger.info(f"Creating working directory: {self.working_dir}")
            os.makedirs(self.working_dir, exist_ok=True)

        if llm is None:  # initialize a new LLM model
            if self.global_config.llm_infer_mode == "online":
                self.llm: BaseLLM = _get_llm_class(self.global_config)
            elif self.global_config.llm_infer_mode == "offline":
                from remem.llm.vllm_offline import VLLMOffline

                self.llm = VLLMOffline(
                    self.global_config,
                    model_name=self.global_config.llm_name,
                    cache_dir=f"outputs/{self.global_config.dataset}/llm_cache",
                )
        else:  # load the give LLM
            self.llm = llm

        self.extract_llm = extract_llm or self.llm
        self.qa_llm = qa_llm or self.llm
        self.eval_llm = None

        if self.global_config.llm_infer_mode == "online":
            if self.global_config.extract_method == "openie":
                self.openie = OpenIE(llm_model=self.extract_llm)
            elif self.global_config.extract_method == "episodic":
                from remem.information_extraction.episodic_extraction_openai import EpisodicExtraction

                self.openie = EpisodicExtraction(self.extract_llm, self.global_config)
            elif self.global_config.extract_method == "episodic_gist":
                from remem.information_extraction.episodic_gist_extraction_openai import EpisodicGistExtraction

                self.openie = EpisodicGistExtraction(self.extract_llm, self.global_config)
            elif self.global_config.extract_method == "temporal":
                from remem.information_extraction.temporal_extraction_openai import TemporalExtraction

                self.openie = TemporalExtraction(self.extract_llm, self.global_config)
            else:
                raise NotImplementedError(f"Extraction method {self.global_config.extract_method} is not implemented")
        elif self.global_config.llm_infer_mode == "offline":
            assert self.llm is not None, "LLM (for offline) is None now, please set it first"
            if self.global_config.extract_method == "openie":
                self.openie = VLLMOfflineOpenIE(self.global_config, self.extract_llm)
            elif self.global_config.extract_method == "episodic":
                from remem.information_extraction.episodic_extraction_vllm_offline import VLLMOfflineEpisodicExtraction

                self.openie = VLLMOfflineEpisodicExtraction(
                    global_config=self.global_config, llm_model=self.extract_llm
                )
            elif self.global_config.extract_method == "episodic_gist":
                from remem.information_extraction.episodic_gist_extraction_vllm_offline import (
                    VLLMOfflineEpisodicGistExtraction,
                )

                self.openie = VLLMOfflineEpisodicGistExtraction(
                    global_config=self.global_config, llm_client=self.extract_llm
                )
            elif self.global_config.extract_method == "temporal":
                from remem.information_extraction.temporal_extraction_vllm_offline import VLLMOfflineTemporalExtraction

                self.openie = VLLMOfflineTemporalExtraction(
                    global_config=self.global_config, llm_model=self.extract_llm
                )
            else:
                raise NotImplementedError(f"Extraction method {self.global_config.extract_method} is not implemented")

        self.graph = self.initialize_graph()

        self._embedding_model = None
        self._chunk_embedding_store = None
        self._phrase_embedding_store = None
        self._triple_embedding_store = None
        self._summarization_embedding_store = None
        self._episodic_embedding_stores = None
        self.chunk_contents = None

        self.prompt_template_manager = PromptTemplateManager(
            role_mapping={"system": "system", "user": "user", "assistant": "assistant"}
        )

        self.openie_results = None
        self.openie_results_path = self.set_openie_results_path()

        if self.global_config.rerank_dspy_file_path is not None:
            self.triple_filter = DSPyFilter(self)
        else:
            self.triple_filter = None

        self.ready_to_retrieve = False

        from remem.graph.preprocessing import _get_text_preprocessor_cls

        self.text_preprocessor: BasePreprocessor = _get_text_preprocessor_cls(
            text_preprocessor_class_name=self.global_config.text_preprocessor_class_name
        )(
            global_config=self.global_config,
            working_dir=os.path.join(self.working_dir, "text_preprocessing"),
        )

        # Initialize RAG strategy based on extraction method
        from remem.rag_strategies import RAGStrategyFactory

        self.rag_strategy = RAGStrategyFactory.create_strategy(self.global_config.extract_method, self)

    def set_openie_results_path(self):
        base_dir = self.global_config.save_dir
        return f'{base_dir}/openie_results_ner_{self.global_config.extract_llm_label.replace("/", "_")}.json'

    def set_openie(self, llm_openie_instance: VLLMOfflineOpenIE):
        self.openie = llm_openie_instance
        self.llm = self.openie.llm_model

    def set_embedding_model(self, embedding_model: BaseEmbeddingModel):
        self._embedding_model = embedding_model

    @property
    def embedding_model(self) -> BaseEmbeddingModel:
        if self._embedding_model is None:
            self._embedding_model = _get_embedding_client(embedding_model_name=self.global_config.embedding_model_name)(
                global_config=self.global_config, embedding_model_name=self.global_config.embedding_model_name
            )
        return self._embedding_model

    @property
    def chunk_embedding_store(self):
        if self._chunk_embedding_store is None:
            self._chunk_embedding_store = EmbeddingStore(
                self.embedding_model,
                os.path.join(self.working_dir, "chunk_embeddings"),
                self.global_config.embedding_batch_size,
                "chunk",
                from_scratch=self.global_config.force_index_from_scratch,
            )
        return self._chunk_embedding_store

    @property
    def phrase_embedding_store(self):
        if self._phrase_embedding_store is None:
            self._phrase_embedding_store = EmbeddingStore(
                self.embedding_model,
                os.path.join(self.working_dir, "entity_embeddings"),
                self.global_config.embedding_batch_size,
                "entity",
                from_scratch=self.global_config.force_index_from_scratch,
            )
        return self._phrase_embedding_store

    @property
    def triple_embedding_store(self):
        if self._triple_embedding_store is None:
            self._triple_embedding_store = EmbeddingStore(
                self.embedding_model,
                os.path.join(self.working_dir, "fact_embeddings"),
                self.global_config.embedding_batch_size,
                "fact",
                from_scratch=self.global_config.force_index_from_scratch,
            )
        return self._triple_embedding_store

    @property
    def paraphrase_embedding_store(self):
        if self._summarization_embedding_store is None:
            self._summarization_embedding_store = EmbeddingStore(
                self.embedding_model,
                os.path.join(self.working_dir, "summarization_embeddings"),
                self.global_config.embedding_batch_size,
                "summarization",
                from_scratch=self.global_config.force_index_from_scratch,
            )
        return self._summarization_embedding_store

    @property
    def episodic_embedding_stores(self):
        if self._episodic_embedding_stores is None:
            if self.global_config.extract_method in ["episodic_gist"]:
                # keys = ["verbatim", "gists", "facts", "entity", "time", "space", "emotion", "causality", "intent"]
                keys = ["verbatim", "gists", "facts", "entity"]
            elif self.global_config.extract_method in ["temporal"]:
                keys = ["verbatim", "facts", "entity"]
            self._episodic_embedding_stores = {}
            for key in keys:
                self._episodic_embedding_stores[key] = EmbeddingStore(
                    self.embedding_model,
                    os.path.join(self.working_dir, f"{key}_embeddings"),
                    self.global_config.embedding_batch_size,
                    key,
                    from_scratch=self.global_config.force_index_from_scratch,
                )
        return self._episodic_embedding_stores

    def initialize_graph(self) -> ig.Graph:
        """
        Initializes a graph using a GraphML file if available or creates a new graph.

        The function attempts to load a pre-existing graph stored in a GraphML file. If the file
        is not present or the graph needs to be created from scratch, it initializes a new directed
        or undirected graph based on the global configuration. If the graph is loaded successfully
        from the file, pertinent information about the graph (number of nodes and edges) is logged.

        Returns:
            ig.Graph: A pre-loaded or newly initialized graph.

        Raises:
            None
        """
        self._graph_pickle_path = os.path.join(self.working_dir, "graph.pkl")

        loaded_graph = None

        if not self.global_config.force_index_from_scratch:  # can read from the graph pickle file
            if os.path.exists(self._graph_pickle_path):
                loaded_graph = ig.Graph.Read_Pickle(self._graph_pickle_path)

        if loaded_graph is None:  # not read, create a new graph
            logger.info("Creating a new graph")
            return ig.Graph(directed=self.global_config.is_directed_graph)
        else:
            logger.info(
                f"Loaded graph from {self._graph_pickle_path} with {loaded_graph.vcount()} nodes, {loaded_graph.ecount()} edges"
            )
            return loaded_graph

    def index(self, docs: List[str]) -> None:
        """
        Indexes the given documents using the configured RAG strategy.

        Parameters:
            docs : List[str]
                A list of documents to be indexed.
        """
        self.rag_strategy.index(docs)

    def index_original(self, docs: List[str]):
        """
        Original indexing logic for standard OpenIE-based extraction.
        This method contains the original ReMem indexing implementation.
        """
        logger.info("Indexing documents, performing OpenIE...")

        self.add_chunk_and_embeddings(docs)
        chunk_dict = self.chunk_embedding_store.get_hash_id_to_row_readonly()  # key -> {'hash_id': str, 'content': str}

        all_openie_info, chunk_keys_to_process = self.load_existing_openie(chunk_dict.keys())
        new_openie_rows = {k: chunk_dict[k] for k in chunk_keys_to_process}

        if len(chunk_keys_to_process) > 0:
            ie_results = self.openie.batch_openie(new_openie_rows)
            if len(ie_results) == 2:  # NER and triples
                new_ner_results_dict, new_triple_results_dict = ie_results[0], ie_results[1]
                self.merge_openie_results(
                    all_openie_info, chunk_keys_to_process, new_ner_results_dict, new_triple_results_dict
                )
            elif len(ie_results) == 3:  # NER, triples, and paraphrases
                new_ner_results_dict, new_triple_results_dict, new_paraphrase_results_dict = (
                    ie_results[0],
                    ie_results[1],
                    ie_results[2],
                )
                self.merge_openie_results(
                    all_openie_info,
                    chunk_keys_to_process,
                    new_ner_results_dict,
                    new_triple_results_dict,
                    new_paraphrase_results_dict,
                )

        if self.global_config.save_openie:
            self.save_openie_results(all_openie_info)

        # Import the required functions
        reformatted_ie_results = reformat_openie_results(all_openie_info)
        ner_results_dict, triple_results_dict = reformatted_ie_results[0], reformatted_ie_results[1]
        new_paraphrase_results_dict = None
        if len(reformatted_ie_results) > 2:
            paraphrase_results_dict = reformatted_ie_results[2]
            new_paraphrase_results_dict = {
                chunk_id: paraphrase_results_dict[chunk_id]
                for chunk_id in chunk_keys_to_process
                if chunk_id in paraphrase_results_dict
            }

        assert len(chunk_dict) == len(triple_results_dict)
        print(
            f"# of chunks: {len(chunk_dict)}, # of ner results: {len(ner_results_dict)}, # of triple results: {len(triple_results_dict)}"
        )

        # prepare data_store
        chunk_ids = list(chunk_dict.keys())

        chunk_triples = [[text_processing(t) for t in triple_results_dict[chunk_id].triples] for chunk_id in chunk_ids]
        phrase_nodes, chunk_triple_entities = extract_phrase_nodes(chunk_triples)
        flattened_triples = flatten_facts(chunk_triples)

        logger.info(f"Encoding Entities: {len(phrase_nodes)}")
        self.phrase_embedding_store.insert_strings(phrase_nodes)

        triple_to_encode = [str(fact) for fact in flattened_triples]
        logger.info(f"Encoding Facts: {len(triple_to_encode)}")
        self.triple_embedding_store.insert_strings(triple_to_encode)

        if new_paraphrase_results_dict:
            paraphrase_list_for_chunk = [chunk.paraphrases for chunk in list(new_paraphrase_results_dict.values())]
            paraphrase_to_encode = [s for paraphrase_list in paraphrase_list_for_chunk for s in paraphrase_list]
            self.paraphrase_embedding_store.insert_strings(paraphrase_to_encode)

        logger.info("Constructing Graph")

        self.node_to_node_count = {}
        self.phrase_node_to_num_chunk = {}

        self.add_fact_edges(chunk_ids, chunk_triples)
        num_new_chunks = self.add_passage_edges(chunk_ids, chunk_triple_entities)

        if new_paraphrase_results_dict:
            self.add_paraphrase_edges(new_paraphrase_results_dict, new_triple_results_dict)

        if num_new_chunks > 0:
            logger.info(f"Found {num_new_chunks} new chunks to save into graph.")
            self.add_synonymy_edges_between_phrases()

            self.augment_graph()
            self.save_igraph()

    def add_chunk_and_embeddings(self, docs):
        """
        Chunk documents and store their embeddings.
        Args:
            docs:

        Returns:

        """

        if self.chunk_contents is None:
            self.preprocessed_chunks: List[List[Dict[str, Any]]] = self.text_preprocessor.batch_preprocess_doc(
                input=docs
            )
            chunk_metadata = [chunk for chunks in self.preprocessed_chunks for chunk in chunks]
            if len(self.chunk_embedding_store.embeddings) == 0:
                nodes_dict = self.chunk_embedding_store.insert_chunk_dicts(chunk_metadata, "openie")
                self.chunk_contents = [chunk["content"] for chunk in nodes_dict.values()]
                self.chunk_metadata = [chunk["metadata"] for chunk in nodes_dict.values()]
            else:
                self.chunk_contents = list(self.chunk_embedding_store.hash_id_to_text.values())
                chunk_metadata = list(self.chunk_embedding_store.hash_id_to_row.values())
                self.chunk_metadata = [chunk["metadata"] for chunk in chunk_metadata]

    def retrieve(
        self, queries: List[str], num_to_retrieve: Optional[int] = None
    ) -> Union[List[QuerySolution], Tuple[List[QuerySolution], Dict]]:
        """
        Performs retrieval using the ReMem 2 framework, which consists of several steps:
        - Fact Retrieval
        - Recognition Memory for improved fact selection
        - Dense passage scoring
        - Personalized PageRank based re-ranking

        Parameters:
            queries: List[str]
                A list of query strings for which documents are to be retrieved.
            num_to_retrieve: int, optional
                The maximum number of documents to retrieve for each query. If not specified, defaults to
                the `retrieval_top_k` value defined in the global configuration.
            gold_docs: List[List[str]], optional
                A list of lists containing gold-standard documents corresponding to each query. Required
                if retrieval performance evaluation is enabled (`do_eval_retrieval` in global configuration).

        Returns:
            List[QuerySolution] or (List[QuerySolution], Dict)
                If retrieval performance evaluation is not enabled, returns a list of QuerySolution objects, each containing
                the retrieved documents and their scores for the corresponding query. If evaluation is enabled, also returns
                a dictionary containing the evaluation metrics computed over the retrieved results.

        Notes
        -----
        - Long queries with no relevant facts after reranking will default to results from dense passage retrieval.
        """

        if num_to_retrieve is None:
            num_to_retrieve = self.global_config.retrieval_top_k

        if not self.ready_to_retrieve:
            self.prepare_retrieval_objects()

        self.get_query_embeddings(queries)

        retrieval_results = []
        for q_idx, query in tqdm(enumerate(queries), desc="Retrieving", total=len(queries)):
            sorted_chunk_ids, sorted_chunk_scores, top_k_triples = self.retrieve_each_query(query)

            # Use batch retrieval to avoid redundant queries
            limited_chunk_ids = sorted_chunk_ids[:num_to_retrieve]
            hash_ids_to_fetch = [self.passage_node_keys[idx] for idx in limited_chunk_ids]

            # Single batch call to get all required rows
            chunk_rows = self.chunk_embedding_store.get_rows(hash_ids_to_fetch)

            # Extract content and metadata from the batch result (single pass)
            top_k_chunks_content = [chunk_rows[hash_id]["content"] for hash_id in hash_ids_to_fetch]
            top_k_chunks_metadata = [chunk_rows[hash_id].get("metadata", None) for hash_id in hash_ids_to_fetch]

            retrieval_results.append(
                QuerySolution(
                    question=query,
                    docs=top_k_chunks_content,
                    doc_scores=sorted_chunk_scores[:num_to_retrieve],
                    graph_seeds=top_k_triples,
                    doc_metadata=top_k_chunks_metadata,
                )
            )
        return retrieval_results

    def retrieve_each_query(self, query, return_chunk=None):
        return self.rag_strategy.retrieve_each_query(query, return_chunk)

    def retrieve_each_query_original(self, query, return_chunk=None):
        """Original retrieve_each_query implementation for default strategy."""
        query_triple_scores = self.query_to_triple_scores(query)
        top_k_triple_indices, top_k_triples, rerank_log = self.rank_triples(query, query_triple_scores)
        if len(top_k_triples) == 0:
            logger.info("No triple found after reranking, return DPR results")
            sorted_chunk_ids, sorted_chunk_scores = self.dense_passage_retrieval(query)
        else:  # use query-to-triple to search on the graph
            sorted_chunk_ids, sorted_chunk_scores = self.graph_search_with_fact_entities(
                query=query,
                link_top_k=self.global_config.linking_top_k,
                query_triple_scores=query_triple_scores,
                top_k_triples=top_k_triples,
                top_k_triple_indices=top_k_triple_indices,
                passage_node_weight=self.global_config.passage_node_weight,
            )
        return sorted_chunk_ids, sorted_chunk_scores, top_k_triples

    def rag_for_qa(
        self,
        queries: Union[List[str], List[QuerySolution]],
        gold_docs: Optional[List[List[str]]] = None,
        gold_answers: Optional[List[List[str]]] = None,
        metrics: Tuple[str, ...] = ("qa_em", "qa_f1", "retrieval_recall"),
        question_metadata: Optional[List[Dict]] = None,
        to_save: bool = True,
        **kwargs,
    ) -> Tuple[List[QuerySolution], List[str], List[Dict], Dict, Dict]:
        """
        Performs retrieval-augmented generation enhanced QA using the ReMem 2 framework.

        This method can handle both string-based queries and pre-processed QuerySolution objects. Depending
        on its inputs, it returns answers only or additionally evaluate retrieval and answer quality using
        recall @ k, exact match and F1 score metrics.

        Parameters:
            queries (List[Union[str, QuerySolution]]): A list of queries, which can be either strings or
                QuerySolution instances. If they are strings, retrieval will be performed.
            gold_docs (Optional[List[List[str]]]): A list of lists containing gold-standard documents for
                each query. This is used if document-level evaluation is to be performed. Default is None.
            gold_answers (Optional[List[List[str]]]): A list of lists containing gold-standard answers for
                each query. Required if evaluation of question answering (QA) answers is enabled. Default
                is None.

        Returns:
            Union[
                Tuple[List[QuerySolution], List[str], List[Dict]],
                Tuple[List[QuerySolution], List[str], List[Dict], Dict, Dict]
            ]: A tuple that always includes:
                - List of QuerySolution objects containing answers and metadata for each query.
                - List of response messages for the provided queries.
                - List of metadata dictionaries for each query.
                If evaluation is enabled, the tuple also includes:
                - A dictionary with overall results from the retrieval phase (if applicable).
                - A dictionary with overall QA evaluation metrics (exact match and F1 scores).

        """
        return self.rag_strategy.rag_for_qa(
            queries=queries,
            num_to_retrieve=self.global_config.retrieval_top_k,
            gold_answers=gold_answers,
            gold_docs=gold_docs,
            metrics=metrics,
            question_metadata=question_metadata,
            to_save=to_save,
            **kwargs,
        )

    def get_evaluators(self, gold_answers, gold_docs, metrics):
        retrieval_evaluators = []
        if self.global_config.do_eval_retrieval:
            if gold_docs is None:
                logger.warning("No gold docs provided. Retrieval evaluation will be skipped.")
            else:
                for metric in metrics:
                    if metric == "retrieval_recall":
                        retrieval_recall_evaluator = RetrievalRecall(global_config=self.global_config)
                        retrieval_evaluators.append(retrieval_recall_evaluator)
                    elif metric == "retrieval_recall_all":
                        retrieval_recall_all_evaluator = RetrievalRecallAll(global_config=self.global_config)
                        retrieval_evaluators.append(retrieval_recall_all_evaluator)
                    elif metric == "retrieval_ndcg_any":
                        retrieval_ndcg_any_evaluator = RetrievalNDCGAny(global_config=self.global_config)
                        retrieval_evaluators.append(retrieval_ndcg_any_evaluator)
                    elif metric == "retrieval_recall_locomo":
                        retrieval_recall_locomo_evaluator = RetrievalRecallLocomo(global_config=self.global_config)
                        retrieval_evaluators.append(retrieval_recall_locomo_evaluator)

        qa_evaluators = []

        def load_eval_llm():
            eval_llm_name = self.global_config.llm_name
            if self.eval_llm is None:
                self.eval_llm = CacheOpenAI(
                    cache_dir=self.global_config.save_dir + "/llm_cache",
                    cache_filename=f"{eval_llm_name}_eval.sqlite",
                    llm_name=eval_llm_name,
                    use_azure=self.global_config.use_azure,
                    llm_base_url=self.global_config.llm_base_url,
                )

        if self.global_config.do_eval_qa:
            if gold_answers is None:
                logger.warning("No gold answers provided. QA evaluation will be skipped.")
            else:
                for metric in metrics:
                    if metric == "qa_em":
                        qa_em_evaluator = QAExactMatch(global_config=self.global_config)
                        qa_evaluators.append(qa_em_evaluator)
                    elif metric == "qa_f1":
                        qa_f1_evaluator = QAF1Score(global_config=self.global_config)
                        qa_evaluators.append(qa_f1_evaluator)
                    elif metric == "qa_longmemeval":
                        from remem.evaluation.qa_longmemeval import QALongMemEvalLLMJudge
                        from remem.llm import CacheOpenAI

                        load_eval_llm()  # "gpt-4o" used in the paper
                        qa_llm_evaluator = QALongMemEvalLLMJudge(self.eval_llm, global_config=self.global_config)
                        qa_evaluators.append(qa_llm_evaluator)
                    elif metric == "qa_f1_score_locomo":
                        from remem.evaluation.qa_locomo import QAF1ScoreLocomo

                        qa_f1_locomo_evaluator = QAF1ScoreLocomo(global_config=self.global_config)
                        qa_evaluators.append(qa_f1_locomo_evaluator)
                    elif metric == "qa_mem0_llm_judge":
                        from remem.evaluation.qa_mem0_llm_judge import QAMem0LLMJudge
                        from remem.llm import CacheOpenAI

                        load_eval_llm()  # "gpt-4o-mini-2024-07-18" used in the paper
                        qa_mem0_llm_judge_evaluator = QAMem0LLMJudge(self.eval_llm, global_config=self.global_config)
                        qa_evaluators.append(qa_mem0_llm_judge_evaluator)
                    elif metric == "qa_evalsuit_llm_judge":
                        from remem.evaluation.qa_evalsuit_llm_judge import QAEvalSuitLLMJudge
                        from remem.llm import CacheOpenAI

                        load_eval_llm()
                        qa_evalsuit_llm_judge_evaluator = QAEvalSuitLLMJudge(
                            self.eval_llm, global_config=self.global_config
                        )
                        qa_evaluators.append(qa_evalsuit_llm_judge_evaluator)
                    elif metric == "qa_bleu1":
                        qa_bleu1_evaluator = QABleu1Score(global_config=self.global_config)
                        qa_evaluators.append(qa_bleu1_evaluator)
                    elif metric == "qa_bleu4":
                        qa_bleu4_evaluator = QABleu4Score(global_config=self.global_config)
                        qa_evaluators.append(qa_bleu4_evaluator)

        # Check for unsupported metrics and warn
        supported_metrics = {
            "retrieval_recall",
            "retrieval_recall_all",
            "retrieval_ndcg_any",
            "retrieval_recall_locomo",
            "qa_em",
            "qa_f1",
            "qa_longmemeval",
            "qa_f1_score_locomo",
            "qa_mem0_llm_judge",
            "qa_evalsuit_llm_judge",
            "qa_bleu1",
            "qa_bleu4",
        }
        unsupported_metrics = [m for m in metrics if m not in supported_metrics]
        if unsupported_metrics:
            logger.warning(f"Unsupported metrics found: {unsupported_metrics}")
            logger.warning(f"Supported metrics: {sorted(supported_metrics)}")

        return qa_evaluators, retrieval_evaluators

    def save_rag_results(self, gold_answers, gold_docs, query_solutions, overall_qa_results, overall_retrieval_results):
        if self.global_config.do_eval_qa and gold_answers is not None:
            for idx, q in enumerate(query_solutions):
                q.gold_answers = list(gold_answers[idx])
                if gold_docs is not None:
                    q.gold_docs = gold_docs[idx]

            # Generate filename based on agent configuration using helper method
            inference_type = get_inference_type(
                self.global_config.agent_fixed_tools,
                self.global_config.agent_max_steps,
                self.global_config.extract_method,
            )
            filename = f"rag_results_{inference_type}.json"
            rag_results_path = os.path.join(self.working_dir, filename)

            # Use safe serialization
            from .utils.misc_utils import safe_dump_json, safe_serialize_query_solutions

            samples_dict = safe_serialize_query_solutions(query_solutions)
            rag_results = {"samples": samples_dict, "overall_metrics": {}}
            if self.global_config.do_eval_retrieval:
                rag_results["overall_metrics"].update(overall_retrieval_results)
            if self.global_config.do_eval_qa:
                rag_results["overall_metrics"].update(overall_qa_results)

            success = safe_dump_json(rag_results, rag_results_path)
            if not success:
                logger.warning(f"Had to use fallback serialization for {rag_results_path}")

    def evaluate_qa(self, gold_answers, qa_evaluators, query_solutions, question_metadata: List = None):
        question_types = None
        question_ids = None
        if question_metadata is not None:
            if "type" in question_metadata[0]:
                question_types = [question_metadata[i]["type"] for i in range(len(question_metadata))]
                assert len(question_types) == len(
                    query_solutions
                ), f"# of question_types: {question_types}, # of query_solutions: {query_solutions}"
            if "id" in question_metadata[0]:
                question_ids = [question_metadata[i]["id"] for i in range(len(question_metadata))]
                assert len(question_ids) == len(
                    query_solutions
                ), f"# of question_ids: {question_ids}, # of query_solutions: {query_solutions}"

        overall_qa_results = {}
        if self.global_config.do_eval_qa and gold_answers is not None:
            for qa_evaluator in qa_evaluators:
                overall_metric_result, example_metric_results = qa_evaluator.calculate_metric_scores(
                    aggregation_fn=np.max,
                    gold_answers=gold_answers,
                    predicted_answers=[qs.answer for qs in query_solutions],
                    questions=[qs.question for qs in query_solutions],
                    question_types=question_types,
                    question_ids=question_ids,
                    rationales=[qs.qa_rationale for qs in query_solutions],
                )
                overall_qa_results.update(overall_metric_result)
                for sample_idx, query_solution in enumerate(query_solutions):
                    if query_solution.metrics is None:
                        query_solution.metrics = {}
                    query_solution.metrics.update(example_metric_results[sample_idx])
            overall_qa_results = {k: round(float(v), 4) for k, v in overall_qa_results.items()}
            logger.info(f"Evaluation results for QA: {overall_qa_results}")
        return overall_qa_results

    def evaluate_retrieval(self, gold_docs, query_solutions, retrieval_evaluators):
        overall_retrieval_results = {}
        if self.global_config.do_eval_retrieval and gold_docs is not None and retrieval_evaluators:
            for retrieval_evaluator in retrieval_evaluators:
                k_list = [1, 3, 5, 10, 15, 20, 30, 50]
                overall_metric_result, example_metric_results = retrieval_evaluator.calculate_metric_scores(
                    gold_docs=gold_docs,
                    retrieved_chunks=[retrieval_result.docs for retrieval_result in query_solutions],
                    k_list=k_list,
                )
                overall_retrieval_results.update(overall_metric_result)
                for sample_idx, query_solution in enumerate(query_solutions):
                    if query_solution.metrics is None:
                        query_solution.metrics = {}
                    query_solution.metrics.update(example_metric_results[sample_idx])

            logger.info(f"Evaluation results for retrieval: {overall_retrieval_results}")
        return overall_retrieval_results

    def qa(self, queries: List[QuerySolution]) -> Tuple[List[QuerySolution], List[str], List[Dict]]:
        """
        Executes question-answering (QA) inference using a provided set of query solutions and a language model.

        Parameters:
            queries: List[QuerySolution]
                A list of QuerySolution objects that contain the user queries, retrieved documents, and other related information.

        Returns:
            Tuple[List[QuerySolution], List[str], List[Dict]]
                A tuple containing:
                - A list of updated QuerySolution objects with the predicted answers embedded in them.
                - A list of raw response messages from the language model.
                - A list of metadata dictionaries associated with the results.
        """

        def sort_retrieved_passages(retrieved_passages):
            def sort_key(x):
                parsed_date = parse_flexible_datetime(x["date"])
                parsed_date = parsed_date if parsed_date else datetime.min
                return (parsed_date, 0 if x["role"] == "user" else 1)

            return sorted(retrieved_passages, key=sort_key)

        def make_qa_context(query_solution: Dict, sort_by_date=False):
            if query_solution.doc_metadata is not None:
                retrieved_passages = query_solution.doc_metadata[: self.global_config.qa_top_k]
                if sort_by_date:
                    if len(retrieved_passages):
                        if "date" in retrieved_passages[0]:
                            # sort retrieved_passages by date and role ("user" before "assistant")
                            retrieved_passages = sort_retrieved_passages(retrieved_passages)
                    else:
                        logger.warning("No retrieved passages found in doc_metadata.")
                        retrieved_passages = []
            else:
                retrieved_passages = query_solution.docs[: self.global_config.qa_top_k]

            question_date = None
            if query_solution.question_metadata is not None:
                question_date = query_solution.question_metadata.get("date", "")

            if question_date:
                qa_context = (
                    f"Question: {query_solution.question} (question date: {question_date})\n\nRetrieved contexts:\n"
                )
            else:
                qa_context = f"Question: {query_solution.question}\n\nRetrieved contexts:\n"

            for passage in retrieved_passages:
                if len(passage) == 0:
                    logger.error("Empty passage found:", retrieved_passages)
                    exit(1)
                if isinstance(passage, str):
                    qa_context += f"{self.global_config.qa_passage_prefix}{passage}\n"
                elif isinstance(passage, dict):
                    if "date" in passage and "role" in passage:
                        qa_context += f'{self.global_config.qa_passage_prefix}[{passage.get("date", "")}] {passage.get("role", "")}: {passage["content"]}\n'
                    else:
                        qa_context += f"{self.global_config.qa_passage_prefix} {passage['content']}"
                else:
                    raise TypeError("Unsupported passage type. Must be str or dict.")

            qa_context += "\nThought: "
            return qa_context

        # Running inference for QA
        all_qa_messages = []

        for query_solution in tqdm(queries, desc="Collecting QA prompts"):
            # obtain the retrieved docs
            prompt_user = make_qa_context(query_solution)
            if (
                self.global_config.qa_prompt_template is not None
                and self.prompt_template_manager.is_template_name_valid(name=self.global_config.qa_prompt_template)
            ):
                prompt_template_name = self.global_config.qa_prompt_template
            elif self.prompt_template_manager.is_template_name_valid(name=f"rag_qa_{self.global_config.dataset}"):
                # find the corresponding prompt for this dataset
                prompt_template_name = f"rag_qa_{self.global_config.dataset}"
            else:
                # the dataset does not have a customized prompt template yet
                prompt_template_name = "rag_qa_musique"
            all_qa_messages.append(
                self.prompt_template_manager.render(name=prompt_template_name, prompt_user=prompt_user)
            )

        # all_qa_results = [self.qa_llm.infer(qa_messages) for qa_messages in tqdm(all_qa_messages, desc="QA Reading")]
        all_qa_results = self.qa_llm.batch_infer(all_qa_messages)

        all_response_message, all_metadata, all_cache_hit = zip(*all_qa_results)
        all_response_message, all_metadata = list(all_response_message), list(all_metadata)

        # Process responses and extract predicted answers.
        query_solutions = []
        for query_solution_idx, query_solution in tqdm(enumerate(queries), desc="Extracting Answers from LLM Response"):
            response_content = all_response_message[query_solution_idx]
            response_split = response_content.split("Answer:")
            try:
                predicted_rationale = response_split[0].strip()
                predicted_ans = response_split[1].strip()
            except Exception as e:
                logger.warning(f"Error in parsing the answer from the raw LLM QA inference response: {str(e)}!")
                predicted_rationale = ""
                predicted_ans = response_content

            query_solution.answer = predicted_ans
            query_solution.qa_rationale = response_content
            query_solutions.append(query_solution)

        return query_solutions, all_response_message, all_metadata

    def add_fact_edges(self, chunk_ids: List[str], chunk_triples: List[Tuple]):
        """
        Adds fact edges from given triples to the graph.

        The method processes chunks of triples, computes unique identifiers
        for entities and relations, and updates various internal statistics
        to build and maintain the graph structure. Entities are uniquely
        identified and linked based on their relationships.

        Parameters:
            chunk_ids: List[str]
                A list of unique identifiers for the chunks being processed.
            chunk_triples: List[Tuple]
                A list of tuples representing triples to process. Each triple
                consists of a subject, predicate, and object.

        Raises:
            Does not explicitly raise exceptions within the provided function logic.
        """

        if "name" in self.graph.vs:
            current_graph_nodes = set(self.graph.vs["name"])
        else:
            current_graph_nodes = set()

        logger.info("Adding OpenIE triples to graph.")

        for chunk_key, triples in tqdm(zip(chunk_ids, chunk_triples)):
            phrases_in_chunk = set()

            if chunk_key not in current_graph_nodes:
                for triple in triples:
                    triple = tuple(triple)
                    # fact_key = compute_mdhash_id(content=str(triple), prefix=("fact-"))

                    subject_key = compute_mdhash_id(content=triple[0], prefix=("entity-"))
                    object_key = compute_mdhash_id(content=triple[2], prefix=("entity-"))

                    self.node_to_node_count[(subject_key, object_key)] = (
                        self.node_to_node_count.get((subject_key, object_key), 0.0) + 1
                    )
                    self.node_to_node_count[(object_key, subject_key)] = (
                        self.node_to_node_count.get((object_key, subject_key), 0.0) + 1
                    )

                    phrases_in_chunk.add(subject_key)
                    phrases_in_chunk.add(object_key)

                for node in phrases_in_chunk:
                    self.phrase_node_to_num_chunk[node] = self.phrase_node_to_num_chunk.get(node, 0) + 1

    def add_passage_edges(self, chunk_ids: List[str], chunk_triple_entities: List[List[str]]):
        """
        Adds edges connecting passage nodes to phrase nodes in the graph.

        This method is responsible for iterating through a list of chunk identifiers
        and their corresponding triple entities. It calculates and adds new edges
        between the passage nodes (defined by the chunk identifiers) and the phrase
        nodes (defined by the computed unique hash IDs of triple entities). The method
        also updates the node-to-node statistics map and keeps count of newly added
        passage nodes.

        Parameters:
            chunk_ids : List[str]
                A list of identifiers representing passage nodes in the graph.
            chunk_triple_entities : List[List[str]]
                A list of lists where each sublist contains entities (strings) associated
                with the corresponding chunk in the chunk_ids list.

        Returns:
            int
                The number of new passage nodes added to the graph.
        """

        if "name" in self.graph.vs.attribute_names():
            current_graph_nodes = set(self.graph.vs["name"])
        else:
            current_graph_nodes = set()

        num_new_chunks = 0

        logger.info("Connecting passage nodes to phrase nodes.")

        for idx, chunk_key in tqdm(enumerate(chunk_ids)):

            if chunk_key not in current_graph_nodes:
                for chunk_ent in chunk_triple_entities[idx]:
                    node_key = compute_mdhash_id(chunk_ent, prefix="entity-")

                    self.node_to_node_count[(chunk_key, node_key)] = 1.0

                num_new_chunks += 1

        return num_new_chunks

    def add_paraphrase_edges(self, new_paraphrase_results_dict, new_triple_results_dict):
        assert len(new_paraphrase_results_dict) < len(
            new_triple_results_dict
        ), "The number of paraphrase results should be less than the number of extracted chunks"
        for idx, chunk_key in tqdm(enumerate(new_paraphrase_results_dict.keys())):
            paraphrases = new_paraphrase_results_dict[chunk_key].paraphrases
            for p in paraphrases:
                paraphrase_key = compute_mdhash_id(p, prefix="paraphrase-")
                self.node_to_node_count[(chunk_key, paraphrase_key)] = 1.0
            # for chunk_ent in chunk_triple_entities[idx]:
            #     node_key = compute_mdhash_id(chunk_ent, prefix="entity-")
            #     self.node_to_node_stats[(paraphrase_key, node_key)] = 1.0

    def add_synonymy_edges_between_phrases(self):
        """
        Adds synonymy edges between similar nodes in the graph to enhance connectivity by identifying and linking synonym entities.

        This method performs key operations to compute and add synonymy edges. It first retrieves embeddings for all nodes, then conducts
        a nearest neighbor (KNN) search to find similar nodes. These similar nodes are identified based on a score threshold, and edges
        are added to represent the synonym relationship.

        Attributes:
            entity_id_to_row: dict (populated within the function). Maps each entity ID to its corresponding row data, where rows
                              contain `content` of entities used for comparison.
            phrase_embedding_store: Manages retrieval of texts and embeddings for all rows related to entities.
            global_config: Configuration object that defines parameters such as `synonymy_edge_topk`, `synonymy_edge_sim_threshold`,
                           `synonymy_edge_query_batch_size`, and `synonymy_edge_key_batch_size`.
            node_to_node_stats: dict. Stores scores for edges between nodes representing their relationship.

        """
        logger.info("Expanding graph with synonymy edges")

        self.phrase_id_to_row = (
            self.phrase_embedding_store.get_text_for_all_rows()
        )  # Need deepcopy: assigned to instance variable
        phrase_node_keys = list(self.phrase_id_to_row.keys())

        logger.info(f"Performing KNN retrieval for each phrase nodes ({len(phrase_node_keys)}).")
        phrase_embeddings = self.phrase_embedding_store.get_embeddings(phrase_node_keys)
        self.add_synonymy_edges(phrase_embeddings, phrase_node_keys, self.phrase_id_to_row)

    def add_synonymy_edges(self, node_embeddings, node_keys, id_to_row):
        # Here we build synonymy edges only between newly inserted phrase nodes and all phrase nodes in the storage to reduce cost for incremental graph updates
        query_node_key2knn_node_keys = retrieve_knn(
            query_ids=node_keys,
            key_ids=node_keys,
            query_vecs=node_embeddings,
            key_vecs=node_embeddings,
            k=self.global_config.synonymy_edge_topk,
            query_batch_size=self.global_config.synonymy_edge_query_batch_size,
            key_batch_size=self.global_config.synonymy_edge_key_batch_size,
        )
        num_synonym_triple = 0
        synonym_candidates = []  # [(node key, [(synonym node key, corresponding score), ...]), ...]

        for node_key in tqdm(query_node_key2knn_node_keys.keys(), total=len(query_node_key2knn_node_keys)):
            synonyms = []

            entity = id_to_row[node_key]["content"]

            if len(re.sub("[^A-Za-z0-9]", "", entity)) > 2:
                nns = query_node_key2knn_node_keys[node_key]

                num_nns = 0
                for nn, score in zip(nns[0], nns[1]):
                    if score < self.global_config.synonymy_edge_sim_threshold or num_nns > 100:
                        break

                    nn_phrase = id_to_row[nn]["content"]

                    if nn != node_key and nn_phrase != "":
                        sim_edge = (node_key, nn)
                        synonyms.append((nn, score))
                        num_synonym_triple += 1

                        self.node_to_node_count[sim_edge] = score  # Need to seriously discuss on this
                        num_nns += 1

            synonym_candidates.append((node_key, synonyms))

    def load_openie_results(self):
        if self.openie_results is None:
            if not self.global_config.force_openie_from_scratch and os.path.isfile(
                self.openie_results_path
            ):  # load existing openie results
                self.openie_results = json.load(open(self.openie_results_path))

    def get_openie_by_chunk_keys(self, chunk_keys: List[str]):
        if self.openie_results is not None:
            return [
                item for chunk_key in chunk_keys for item in self.openie_results["docs"] if item["idx"] in chunk_key
            ]
        return None

    def load_existing_openie(self, chunk_keys: List[str]) -> Tuple[List[dict], Set[str]]:
        """
        Loads existing OpenIE results from the specified file if it exists and combines
        them with new content while standardizing indices. If the file does not exist or
        is configured to be re-initialized from scratch with the flag `force_openie_from_scratch`,
        it prepares new entries for processing.

        Args:
            chunk_keys (List[str]): A list of chunk keys that represent identifiers
                                     for the content to be processed.

        Returns:
            Tuple[List[dict], Set[str]]: A tuple where the first element is the existing OpenIE
                                         information (if any) loaded from the file, and the
                                         second element is a set of chunk keys that still need to
                                         be saved or processed.
        """

        # combine openie_results with contents already in file, if file exists
        chunk_keys_to_save = set()

        self.load_openie_results()

        if not self.global_config.force_openie_from_scratch and os.path.isfile(
            self.openie_results_path
        ):  # load existing openie results
            all_openie_info = self.openie_results.get("docs", [])

            # Standardizing indices for OpenIE Files.

            renamed_openie_info = []
            for openie_info in all_openie_info:
                openie_info["idx"] = compute_mdhash_id(
                    openie_info.get("passage", openie_info.get("verbatim", None)), "chunk-"
                )
                renamed_openie_info.append(openie_info)

            all_openie_info = renamed_openie_info

            existing_openie_keys = set([info["idx"] for info in all_openie_info])

            for chunk_key in chunk_keys:
                if chunk_key not in existing_openie_keys:
                    chunk_keys_to_save.add(chunk_key)
        else:
            all_openie_info = []
            chunk_keys_to_save = chunk_keys

        return all_openie_info, chunk_keys_to_save

    def merge_openie_results(
        self,
        all_openie_info: List[dict],
        chunk_keys_to_save: List[str],
        ner_results_dict: Dict[str, NerRawOutput],
        triple_results_dict: Dict[str, TripleRawOutput],
        paraphrase_results_dict: Dict[str, ParaphraseRawOutput] = None,
    ) -> List[dict]:
        """
        Merges OpenIE extraction results with corresponding passage and metadata.

        This function integrates the OpenIE extraction results, including named-entity
        recognition (NER) entities and triples, with their respective text passages
        using the provided chunk keys. The resulting merged data is appended to
        the `all_openie_info` list containing dictionaries with combined and organized
        data for further processing or storage.

        Parameters:
            all_openie_info (List[dict]): A list to hold dictionaries of merged OpenIE
                results and metadata for all chunks.
            chunk_keys_to_save (List[str]): A list of chunk identifiers (keys) to process
                and merge OpenIE results for.
            ner_results_dict (Dict[str, NerRawOutput]): A dictionary mapping chunk keys
                to their corresponding NER extraction results.
            triple_results_dict (Dict[str, TripleRawOutput]): A dictionary mapping chunk
                keys to their corresponding OpenIE triple extraction results.

        Returns:
            List[dict]: The `all_openie_info` list containing dictionaries with merged
            OpenIE results, metadata, and the passage content for each chunk.

        """

        for chunk_key in chunk_keys_to_save:
            passage = self.chunk_embedding_store.get_row(chunk_key)["content"]
            chunk_openie_info = {
                "idx": chunk_key,
                "passage": passage,
                "extracted_entities": ner_results_dict[chunk_key].unique_entities,
                "extracted_triples": triple_results_dict[chunk_key].triples,
            }
            if paraphrase_results_dict is not None:
                chunk_openie_info["paraphrases"] = paraphrase_results_dict[chunk_key].paraphrases
            all_openie_info.append(chunk_openie_info)

        return all_openie_info

    def save_openie_results(self, all_openie_info: List[dict]):
        """
        Computes statistics on extracted entities from OpenIE results and saves the aggregated data in a
        JSON file. The function calculates the average character and word lengths of the extracted entities
        and writes them along with the provided OpenIE information to a file.

        Parameters:
            all_openie_info : List[dict]
                List of dictionaries, where each dictionary represents information from OpenIE, including
                extracted entities.
        """
        sum_phrase_chars = None
        sum_phrase_words = None
        if len(all_openie_info) and (
            isinstance(all_openie_info[0], NerRawOutput) or isinstance(all_openie_info[0], TripleRawOutput)
        ):
            sum_phrase_chars = sum([len(str(e)) for chunk in all_openie_info for e in chunk["extracted_entities"]])
            sum_phrase_words = sum(
                [len(str(e).split()) for chunk in all_openie_info for e in chunk["extracted_entities"]]
            )
            num_phrases = sum([len(chunk["extracted_entities"]) for chunk in all_openie_info])

        if len(all_openie_info) > 0:
            openie_dict = {"docs": all_openie_info}
            if sum_phrase_chars:
                openie_dict["avg_ent_chars"] = round(sum_phrase_chars / num_phrases, 4)
            if sum_phrase_words:
                openie_dict["avg_ent_words"] = round(sum_phrase_words / num_phrases, 4)

            openie_dict["docs"] = [dict(item) for item in openie_dict["docs"]]
            with open(self.openie_results_path, "w") as f:
                # convert each EpisodeRawOutput element in openie_dict["docs"] to a dict
                json.dump(openie_dict, f)
            logger.info(f"OpenIE results saved to {self.openie_results_path}")

    def augment_graph(self):
        """
        Provides utility functions to augment a graph by adding new nodes and edges.
        It ensures that the graph structure is extended to include additional components,
        and logs the completion status along with printing the updated graph information.
        """

        self.add_new_nodes()
        self.add_new_edges()

        logger.info("Graph construction completed!")
        print(self.get_graph_info())

    def add_new_nodes(self):
        """
        Adds new nodes to the graph from entity and passage embedding stores based on their attributes.

        This method identifies and adds new nodes to the graph by comparing existing nodes
        in the graph and nodes retrieved from the entity embedding store and the passage
        embedding store. The method checks attributes and ensures no duplicates are added.
        New nodes are prepared and added in bulk to optimize graph updates.
        """

        existing_nodes = {v["name"]: v for v in self.graph.vs if "name" in v.attributes()}

        phrase_nodes = self.phrase_embedding_store.get_text_for_all_rows()  # Need deepcopy: data will be modified below
        passage_nodes = self.chunk_embedding_store.get_text_for_all_rows()  # Need deepcopy: data will be modified below
        summarization_nodes = None
        if self.paraphrase_embedding_store is not None and len(self.paraphrase_embedding_store.embeddings) > 0:
            summarization_nodes = (
                self.paraphrase_embedding_store.get_text_for_all_rows()
            )  # Need deepcopy: data will be modified below

        all_nodes = phrase_nodes
        all_nodes.update(passage_nodes)
        if summarization_nodes is not None:
            all_nodes.update(summarization_nodes)

        new_nodes = {}
        for node_id, node in all_nodes.items():
            node["name"] = node_id
            if node_id not in existing_nodes:
                for k, v in node.items():
                    if k not in new_nodes:
                        new_nodes[k] = []
                    new_nodes[k].append(v)

        if len(new_nodes) > 0:
            self.graph.add_vertices(n=len(next(iter(new_nodes.values()))), attributes=new_nodes)

    def add_new_edges(self):
        """
        Processes edges from `node_to_node_stats` to add them into a graph object while
        managing adjacency lists, validating edges, and logging invalid edge cases.
        """

        graph_adj_list = defaultdict(dict)
        graph_inverse_adj_list = defaultdict(dict)
        edge_source_node_keys = []
        edge_target_node_keys = []
        edge_metadata = []
        for edge, weight in self.node_to_node_count.items():
            if edge[0] == edge[1]:
                continue
            graph_adj_list[edge[0]][edge[1]] = weight
            graph_inverse_adj_list[edge[1]][edge[0]] = weight

            edge_source_node_keys.append(edge[0])
            edge_target_node_keys.append(edge[1])
            edge_metadata.append({"weight": weight})

        valid_edges, valid_weights = [], {"weight": []}
        current_node_ids = set(self.graph.vs["name"])
        for source_node_id, target_node_id, edge_d in zip(edge_source_node_keys, edge_target_node_keys, edge_metadata):
            assert isinstance(source_node_id, str) and isinstance(
                target_node_id, str
            ), "Both source and target node IDs must be strings."
            if source_node_id in current_node_ids and target_node_id in current_node_ids:
                valid_edges.append((source_node_id, target_node_id))
                weight = edge_d.get("weight", 1.0)
                valid_weights["weight"].append(weight)
            else:
                if source_node_id not in current_node_ids:
                    logger.warning(
                        f"Edge {source_node_id} -> {target_node_id}: source node {source_node_id} is not valid."
                    )
                elif target_node_id not in current_node_ids:
                    logger.warning(
                        f"Edge {source_node_id} -> {target_node_id}: target node {target_node_id} is not valid."
                    )
        self.graph.add_edges(valid_edges, attributes=valid_weights)

    def save_igraph(self):
        logger.info(f"Writing graph with {len(self.graph.vs())} nodes, {len(self.graph.es())} edges")
        self.graph.write_pickle(self._graph_pickle_path)
        logger.info("Saving graph completed!")

    def get_graph_info(self) -> Dict:
        return self.rag_strategy.get_graph_info()

    def prepare_retrieval_objects(self):
        """
        Prepares various in-memory objects and attributes necessary for fast retrieval processes, such as embedding data and graph relationships, ensuring consistency
        and alignment with the underlying graph structure.
        """

        logger.info("Preparing for fast retrieval...")

        logger.info("Loading keys...")
        self.query_to_embedding: Dict = {"triple": {}, "passage": {}}

        self.phrase_node_keys: List = list(self.phrase_embedding_store.get_all_ids())  # a list of phrase node keys
        self.passage_node_keys: List = list(self.chunk_embedding_store.get_all_ids())  # a list of passage node keys
        self.summarization_node_keys: List = list(
            self.paraphrase_embedding_store.get_all_ids()
        )  # a list of summarization node keys
        self.triple_keys: List = list(self.triple_embedding_store.get_all_ids())

        if "facts_and_sim" in self.global_config.graph_type:
            if (
                len(self.phrase_node_keys) + len(self.passage_node_keys) + len(self.summarization_node_keys)
                != self.graph.vcount()
            ):
                missing_phrase = [key for key in self.phrase_node_keys if key not in set(self.graph.vs["name"])]
                missing_passage = [key for key in self.passage_node_keys if key not in set(self.graph.vs["name"])]
                logger.error("The number of graph nodes is not matched")
                logger.error(f"Missing phrase nodes: {missing_phrase}")
                logger.error(f"Missing passage nodes: {missing_passage}")
                exit(1)

            igraph_name_to_idx = {
                node["name"]: idx for idx, node in enumerate(self.graph.vs)
            }  # from node key to the index in the backbone graph
            self.node_name_to_vertex_idx = igraph_name_to_idx
            self.phrase_node_idxs = [
                igraph_name_to_idx[node_key] for node_key in self.phrase_node_keys
            ]  # a list of backbone graph node index
            self.passage_node_idxs = [
                igraph_name_to_idx[node_key] for node_key in self.passage_node_keys
            ]  # a list of backbone passage node index

        logger.info("Loading embeddings...")
        self.phrase_embeddings = np.array(self.phrase_embedding_store.get_embeddings(self.phrase_node_keys))
        self.chunk_embeddings = np.array(self.chunk_embedding_store.get_embeddings(self.passage_node_keys))
        self.triple_embeddings = np.array(self.triple_embedding_store.get_embeddings(self.triple_keys))

        self.ready_to_retrieve = True

    def get_query_embeddings(self, queries: Union[List[str], List[QuerySolution]]):
        """
        Retrieves embeddings for given queries and updates the internal query-to-embedding mapping. The method determines whether each query
        is already present in the `self.query_to_embedding` dictionary under the keys 'triple' and 'passage'. If a query is not present in
        either, it is encoded into embeddings using the embedding model and stored.

        Args:
            queries List[str] | List[QuerySolution]: A list of query strings or QuerySolution objects. Each query is checked for
            its presence in the query-to-embedding mappings.
        """

        all_query_strings = []
        for query in queries:
            if isinstance(query, QuerySolution) and (
                query.question not in self.query_to_embedding["triple"]
                or query.question not in self.query_to_embedding["passage"]
            ):
                all_query_strings.append(query.question)
            elif query not in self.query_to_embedding["triple"] or query not in self.query_to_embedding["passage"]:
                all_query_strings.append(query)

        if len(all_query_strings) > 0:
            # get all query embeddings
            logger.info(f"Encoding {len(all_query_strings)} queries for query_to_triple.")
            query_embeddings_for_triple = self.embedding_model.batch_encode(
                all_query_strings, instruction=get_query_instruction("query_to_triple"), norm=True
            )
            for query, embedding in zip(all_query_strings, query_embeddings_for_triple):
                self.query_to_embedding["triple"][query] = embedding

            logger.info(f"Encoding {len(all_query_strings)} queries for query_to_passage.")
            query_embeddings_for_passage = self.embedding_model.batch_encode(
                all_query_strings, instruction=get_query_instruction("query_to_passage"), norm=True
            )
            for query, embedding in zip(all_query_strings, query_embeddings_for_passage):
                self.query_to_embedding["passage"][query] = embedding

    def query_to_triple_scores(self, query: str) -> np.ndarray:
        """
        Retrieves and computes normalized similarity scores between the given query and pre-stored fact embeddings.

        Parameters:
        query : str
            The input query text for which similarity scores with fact embeddings
            need to be computed.

        Returns:
        numpy.ndarray
            A normalized array of similarity scores between the query and fact
            embeddings. The shape of the array is determined by the number of
            facts.

        Raises:
        KeyError
            If no embedding is found for the provided query in the stored query
            embeddings dictionary.
        """
        query_embedding = self.query_to_embedding["triple"].get(query, None)
        if query_embedding is None:
            query_embedding = self.embedding_model.batch_encode(
                query, instruction=get_query_instruction("query_to_triple"), norm=True
            )

        query_triple_scores = np.dot(self.triple_embeddings, query_embedding.T)  # shape: (#facts, )
        query_triple_scores = np.squeeze(query_triple_scores) if query_triple_scores.ndim == 2 else query_triple_scores
        query_triple_scores = min_max_normalize(query_triple_scores)

        return query_triple_scores

    def dense_passage_retrieval(
        self, query: str, candidate_docs: List[str] = None, norm=True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Conduct dense passage retrieval to find relevant documents for a query.

        This function processes a given query using a pre-trained embedding model
        to generate query embeddings. The similarity scores between the query
        embedding and passage embeddings are computed using dot product, followed
        by score normalization. Finally, the function ranks the documents based
        on their similarity scores and returns the ranked document identifiers
        and their scores.

        Parameters
        ----------
        query : str
            The input query for which relevant passages should be retrieved.

        Returns
        -------
        tuple : Tuple[np.ndarray, np.ndarray]
            A tuple containing two elements:
            - A list of sorted document identifiers based on their relevance scores.
            - A numpy array of the normalized similarity scores for the corresponding
              documents.
        """
        if not self.ready_to_retrieve:
            self.prepare_retrieval_objects()

        query_embedding = self.query_to_embedding["passage"].get(query, None)
        if query_embedding is None:
            query_embedding = self.embedding_model.batch_encode(
                query, instruction=get_query_instruction("query_to_passage"), norm=True
            )
        if self.chunk_contents is None:
            assert candidate_docs is not None, "No chunk embeddings found. Please set the `docs` to retrieve."
            self.add_chunk_and_embeddings(candidate_docs)
        query_doc_scores = np.dot(self.chunk_embedding_store.embeddings, query_embedding.T)
        query_doc_scores = np.squeeze(query_doc_scores) if query_doc_scores.ndim == 2 else query_doc_scores
        if norm:
            query_doc_scores = min_max_normalize(query_doc_scores)

        sorted_doc_ids = np.argsort(query_doc_scores)[::-1]
        sorted_doc_scores = query_doc_scores[sorted_doc_ids.tolist()]
        return sorted_doc_ids, sorted_doc_scores

    def query_to_summarization_node_scores(self, query: str, norm=True) -> np.ndarray:
        query_embedding = self.query_to_embedding["passage"].get(query, None)
        if query_embedding is None:
            query_embedding = self.embedding_model.batch_encode(
                query, instruction=get_query_instruction("query_to_passage"), norm=True
            )
        query_sum_scores = np.dot(self.paraphrase_embedding_store.embeddings, query_embedding.T)
        query_sum_scores = np.squeeze(query_sum_scores) if query_sum_scores.ndim == 2 else query_sum_scores
        if norm:
            query_sum_scores = min_max_normalize(query_sum_scores)

        sorted_sum_ids = np.argsort(query_sum_scores)[::-1]
        sorted_sum_scores = query_sum_scores[sorted_sum_ids.tolist()]
        return sorted_sum_ids, sorted_sum_scores

    def get_top_k_weights(
        self, link_top_k: int, all_phrase_weights: np.ndarray, linking_score_map: Dict[str, float], entry="entity"
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        This function filters the all_phrase_weights to retain only the weights for the
        top-ranked phrases in terms of the linking_score_map. It also filters linking scores
        to retain only the top `link_top_k` ranked nodes. Non-selected phrases in phrase
        weights are reset to a weight of 0.0.

        Args:
            link_top_k (int): Number of top-ranked nodes to retain in the linking score map.
            all_phrase_weights (np.ndarray): An array representing the phrase weights, indexed
                by phrase ID.
            linking_score_map (Dict[str, float]): A mapping of phrase content to its linking
                score, sorted in descending order of scores.

        Returns:
            Tuple[np.ndarray, Dict[str, float]]: A tuple containing the filtered array
            of all_phrase_weights with unselected weights set to 0.0, and the filtered
            linking_score_map containing only the top `link_top_k` phrases.
        """
        # choose top ranked nodes in linking_score_map
        linking_score_map = dict(sorted(linking_score_map.items(), key=lambda x: x[1], reverse=True)[:link_top_k])

        # only keep the top_k phrases in all_phrase_weights
        top_k_phrases = set(linking_score_map.keys())
        top_k_phrases_keys = set(
            [compute_mdhash_id(content=top_k_phrase, prefix=f"{entry}-") for top_k_phrase in top_k_phrases]
        )

        for phrase_key in self.node_name_to_vertex_idx:
            if phrase_key not in top_k_phrases_keys:
                phrase_id = self.node_name_to_vertex_idx.get(phrase_key, None)
                if phrase_id is not None:
                    all_phrase_weights[phrase_id] = 0.0

        assert np.count_nonzero(all_phrase_weights) == len(linking_score_map.keys())
        # return linking_score_map as defaultdict(float) to avoid KeyError
        return all_phrase_weights, defaultdict(float, linking_score_map)

    def graph_search_with_fact_entities(
        self,
        query: str,
        link_top_k: int,
        query_triple_scores: np.ndarray,
        top_k_triples: List[Tuple],
        top_k_triple_indices: List[int],
        passage_node_weight: float = 0.05,
        summarization_node_weight: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes document scores based on fact-based similarity and relevance using personalized
        PageRank (PPR) and dense retrieval models. This function combines the signal from the relevant
        facts identified with passage similarity and graph-based search for enhanced result ranking.

        Parameters:
            query (str): The input query string for which similarity and relevance computations
                need to be performed.
            link_top_k (int): The number of top phrases to include from the linking score map for
                downstream processing.
            query_triple_scores (np.ndarray): An array of scores representing fact-query similarity
                for each of the provided facts.
            top_k_triples (List[Tuple]): A list of top-ranked facts, where each fact is represented
                as a tuple of its subject, predicate, and object.
            top_k_triple_indices (List[str]): Corresponding indices or identifiers for the top-ranked
                facts in the query_fact_scores array.
            passage_node_weight (float): Default weight to scale passage scores in the graph.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing two arrays:
                - The first array corresponds to document IDs sorted based on their scores.
                - The second array consists of the PPR scores associated with the sorted document IDs.
        """
        # Assigning phrase weights based on selected facts from previous steps.
        linking_score_map = {}  # from phrase to the average scores of the facts that contain the phrase
        phrase_scores = (
            {}
        )  # store all fact scores for each phrase regardless of whether they exist in the knowledge graph or not
        phrase_weights = np.zeros(len(self.graph.vs["name"]))
        passage_weights = np.zeros(len(self.graph.vs["name"]))
        summarization_weights = np.zeros(len(self.graph.vs["name"]))

        for rank, f in enumerate(top_k_triples):
            subject_phrase = f[0].lower()
            predicate_phrase = f[1].lower()
            object_phrase = f[2].lower()
            fact_score = (
                query_triple_scores[top_k_triple_indices[rank]] if query_triple_scores.ndim > 0 else query_triple_scores
            )

            for phrase in [subject_phrase, object_phrase]:
                if skip_phrase_node(phrase):
                    continue
                phrase_key = compute_mdhash_id(content=phrase, prefix="entity-")
                phrase_id = self.node_name_to_vertex_idx.get(phrase_key, None)

                if phrase_id is not None:
                    phrase_weights[phrase_id] = fact_score

                    if self.phrase_node_to_num_chunk[phrase_key] != 0:
                        # calculate average fact score for each phrase
                        phrase_weights[phrase_id] /= self.phrase_node_to_num_chunk[phrase_key]

                if phrase not in phrase_scores:
                    phrase_scores[phrase] = []
                phrase_scores[phrase].append(fact_score)

        # calculate average fact score for each phrase
        for phrase, scores in phrase_scores.items():
            linking_score_map[phrase] = float(np.mean(scores))

        if link_top_k:
            phrase_weights, linking_score_map = self.get_top_k_weights(
                link_top_k, phrase_weights, linking_score_map
            )  # at this stage, the length of linking_scope_map is determined by link_top_k

        # Get passage scores according to chosen dense retrieval model
        dpr_sorted_doc_ids, dpr_sorted_doc_scores = self.dense_passage_retrieval(query)

        # Get summarization scores according to chosen dense retrieval model
        if self.paraphrase_embedding_store is not None and len(self.paraphrase_embedding_store.embeddings) > 0:
            dpr_sorted_summarization_ids, dpr_sorted_summarization_scores = self.query_to_summarization_node_scores(
                query
            )

        for i, dpr_sorted_doc_id in enumerate(dpr_sorted_doc_ids.tolist()):
            passage_node_key = self.passage_node_keys[dpr_sorted_doc_id]
            passage_dpr_score = dpr_sorted_doc_scores[i]
            passage_node_id = self.node_name_to_vertex_idx[passage_node_key]
            passage_weights[passage_node_id] = passage_dpr_score * passage_node_weight
            passage_node_text = self.chunk_embedding_store.get_row(passage_node_key)["content"]
            linking_score_map[passage_node_text] = passage_dpr_score * passage_node_weight

        passage_dpr_stat = describe_array(passage_weights)

        if (
            self.paraphrase_embedding_store is not None
            and len(self.paraphrase_embedding_store.embeddings) > 0
            and summarization_node_weight > 0
        ):
            for i, dpr_sorted_summarization_id in enumerate(dpr_sorted_summarization_ids.tolist()):
                summarization_node_key = self.summarization_node_keys[dpr_sorted_summarization_id]
                summarization_dpr_score = dpr_sorted_summarization_scores[i]
                summarization_node_id = self.node_name_to_vertex_idx[summarization_node_key]
                summarization_weights[summarization_node_id] = summarization_dpr_score * summarization_node_weight
                summarization_node_text = self.paraphrase_embedding_store.get_row(summarization_node_key)["content"]
                linking_score_map[summarization_node_text] = summarization_dpr_score * summarization_node_weight
            summarization_dpr_stat = describe_array(summarization_weights)

        # Combining phrase and passage scores into one array for PPR
        node_weights = phrase_weights + passage_weights + summarization_weights

        # Recording top 30 facts in linking_score_map
        if len(linking_score_map) > 30:
            linking_score_map = dict(sorted(linking_score_map.items(), key=lambda x: x[1], reverse=True)[:30])

        assert sum(node_weights) > 0, f"No phrases found in the graph for the given facts: {top_k_triples}"

        # Running PPR algorithm based on the passage and phrase weights previously assigned
        ppr_sorted_doc_ids, ppr_sorted_doc_scores = self.run_ppr(node_weights, damping=self.global_config.damping)

        assert len(ppr_sorted_doc_ids) == len(
            self.passage_node_idxs
        ), f"Doc prob length {len(ppr_sorted_doc_ids)} != corpus length {len(self.passage_node_idxs)}"

        return ppr_sorted_doc_ids, ppr_sorted_doc_scores

    def rank_triples(
        self, query: str, query_fact_scores: np.ndarray, filter=True
    ) -> Tuple[List[int], List[Tuple], dict]:
        """

        Args:

        Returns:
            top_k_fact_indicies:
            top_k_facts:
            rerank_log (dict): {'facts_before_rerank': candidate_facts, 'facts_after_rerank': top_k_facts}
                - candidate_facts (list): list of link_top_k facts (each fact is a relation triple in tuple data type).
                - top_k_facts:


        """
        # load args
        link_top_k: int = self.global_config.linking_top_k

        candidate_fact_indices = np.argsort(query_fact_scores)[-link_top_k:][
            ::-1
        ].tolist()  # list of ranked link_top_k fact relative indices
        real_candidate_fact_ids = [
            self.triple_keys[idx] for idx in candidate_fact_indices
        ]  # list of ranked link_top_k fact keys
        fact_row_dict = self.triple_embedding_store.get_rows(real_candidate_fact_ids)
        candidate_facts = [
            eval(fact_row_dict[id]["content"]) for id in real_candidate_fact_ids
        ]  # list of link_top_k facts (each fact is a relation triple in tuple data type)

        if filter and self.triple_filter is not None:
            top_k_fact_indices, top_k_facts, reranker_dict = self.triple_filter(
                query, candidate_facts, candidate_fact_indices, len_after_rerank=link_top_k
            )
        else:
            top_k_fact_indices = candidate_fact_indices
            top_k_facts = candidate_facts

        rerank_log = {"facts_before_rerank": candidate_facts, "facts_after_rerank": top_k_facts}

        return top_k_fact_indices, top_k_facts, rerank_log

    def run_ppr(self, reset_prob: np.ndarray, damping: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Runs Personalized PageRank (PPR) on a graph and computes relevance scores for
        nodes corresponding to document passages. The method utilizes a damping
        factor for teleportation during rank computation and can take a reset
        probability array to influence the starting state of the computation.

        Parameters:
            reset_prob (np.ndarray): A 1-dimensional array specifying the reset
                probability distribution for each node. The array must have a size
                equal to the number of nodes in the graph. NaNs or negative values
                within the array are replaced with zeros.
            damping (float): A scalar specifying the damping factor for the
                computation. Defaults to 0.5 if not provided or set to `None`.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing two numpy arrays. The
                first array represents the sorted node IDs of document passages based
                on their relevance scores in descending order. The second array
                contains the corresponding relevance scores of each document passage
                in the same order.
        """

        if damping is None:
            damping = 0.5  # for potential compatibility
        reset_prob = np.where(np.isnan(reset_prob) | (reset_prob < 0), 0, reset_prob)
        pagerank_scores = self.graph.personalized_pagerank(
            vertices=range(len(self.node_name_to_vertex_idx)),
            damping=damping,
            directed=False,
            weights="weight",
            reset=reset_prob,
            implementation="prpack",
        )

        doc_scores = np.array([pagerank_scores[idx] for idx in self.passage_node_idxs])
        sorted_doc_ids = np.argsort(doc_scores)[::-1]
        sorted_doc_scores = doc_scores[sorted_doc_ids.tolist()]

        return sorted_doc_ids, sorted_doc_scores
