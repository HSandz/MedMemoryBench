
import logging
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from remem.utils.misc_utils import QuerySolution, TemporalRawOutput, compute_mdhash_id

from .base_strategy import RAGStrategy

logger = logging.getLogger(__name__)


class TemporalStrategy(RAGStrategy):
    """
    Strategy for temporal-based extraction and retrieval with integrated agent framework.
    """

    def __init__(self, remem_instance):
        super().__init__(remem_instance)

    def index(self, docs: List) -> None:
        """Index documents using temporal extraction with 4-element tuples (subject, predicate, object, time)."""
        logger.info("Indexing Documents with Temporal Extraction")
        logger.info("Performing Temporal Extraction")

        # Determine if we need to do any indexing work
        force_rebuild = (
            self.remem.global_config.force_index_from_scratch or self.remem.global_config.force_openie_from_scratch
        )

        # If not forcing rebuild, check if all necessary components exist
        if not force_rebuild:
            existing_graph_exists = hasattr(self.remem, "_graph_pickle_path") and os.path.exists(
                self.remem._graph_pickle_path
            )
            existing_embeddings_exist = len(self.remem.chunk_embedding_store.embeddings) > 0
            existing_openie_exists = os.path.exists(self.remem.openie_results_path)

            if existing_graph_exists and existing_embeddings_exist and existing_openie_exists:
                logger.info("All components exist, skipping indexing")
                return

        # Continue with indexing process
        self.remem.add_chunk_and_embeddings(docs)
        chunk_dict = self.remem.chunk_embedding_store.get_hash_id_to_row_readonly()

        # Load existing OpenIE results or start fresh based on configuration
        if self.remem.global_config.force_openie_from_scratch:
            logger.info("Force OpenIE from scratch is enabled, skipping existing OpenIE results")
            all_openie_info = []
            chunk_keys_to_process = list(chunk_dict.keys())
        else:
            all_openie_info, chunk_keys_to_process = self.remem.load_existing_openie(chunk_dict.keys())

        # Determine if we need to rebuild the graph (simplified logic)
        need_graph_rebuild = force_rebuild or len(chunk_keys_to_process) > 0

        new_openie_rows = {k: chunk_dict[k] for k in chunk_keys_to_process}

        if len(chunk_keys_to_process) > 0:
            ie_results = self.remem.openie.batch_openie(new_openie_rows)
            self._merge_temporal_extraction_results(all_openie_info, chunk_keys_to_process, ie_results)

        if self.remem.global_config.save_openie:
            self.remem.save_openie_results(all_openie_info)

        temporal_results_dict = {openie_info["chunk_id"]: openie_info for openie_info in all_openie_info}
        print(f"# of chunks: {len(chunk_dict)}, # of temporal results: {len(temporal_results_dict)}")

        # prepare data_store
        element_to_encode = defaultdict(list)

        entries = ["verbatim", "facts"]
        for chunk_idx, chunk in enumerate(temporal_results_dict.values()):
            for entry in entries:
                if entry == "verbatim":
                    verbatim_content = self._get_verbatim_from_chunk(chunk)
                    if verbatim_content:
                        element_to_encode[entry].append(verbatim_content)

                elif entry == "facts":
                    facts = self._get_facts_from_chunk(chunk)
                    if facts:
                        element_to_encode["facts"].extend(facts)
                        # Extract entities from temporal facts
                        entities = self._extract_entities_from_facts(facts)
                        element_to_encode["entity"].extend(entities)

        for entry in entries + ["entity"]:
            if entry in element_to_encode and len(element_to_encode[entry]) > 0:
                logger.info(f"Encoding {entry}: {len(element_to_encode[entry])}")
                if entry == "facts":
                    # Convert temporal facts to format compatible with episodic pattern
                    fact_dicts = []
                    for temporal_triple in element_to_encode["facts"]:
                        if isinstance(temporal_triple, dict):
                            from remem.agent.tools.base_tool import BaseTool

                            formatted_content = BaseTool.format_fact_dict_for_embedding(temporal_triple)
                            fact_dict = {
                                "content": formatted_content,
                                "subject": temporal_triple.get("subject", ""),
                                "predicate": temporal_triple.get("predicate", ""),
                                "object": temporal_triple.get("object", ""),
                                "qualifiers": temporal_triple.get("qualifiers", {}),
                            }
                        else:
                            raise ValueError(f"Invalid temporal triple: {temporal_triple}")
                        fact_dicts.append(fact_dict)
                    self.remem.episodic_embedding_stores["facts"].insert_chunk_dicts(
                        fact_dicts, self.remem.global_config.extract_method
                    )
                else:
                    embed = True
                    if entry == "entity":
                        embed = False
                    # For other entries, just insert the strings directly
                    self.remem.episodic_embedding_stores[entry].insert_strings(element_to_encode[entry], embed=embed)

        logger.info("Constructing Temporal Graph")

        # Collect all connections for direct igraph construction
        temporal_edges = []

        for chunk in temporal_results_dict.values():
            verbatim = self._get_verbatim_from_chunk(chunk)
            facts = self._get_facts_from_chunk(chunk)

            # Create verbatim node key
            verbatim_key = compute_mdhash_id(verbatim, prefix="verbatim-")

            for fact in facts:
                fact_key = compute_mdhash_id(str(fact), prefix="facts-")
                temporal_edges.append(
                    {
                        "source": verbatim_key,
                        "target": fact_key,
                        "predicate": "contains_fact",
                        "qualifiers": {},
                        "weight": 1.0,
                    }
                )

        # Store temporal edges for later graph construction
        self.remem.temporal_edges = temporal_edges

        # Only rebuild graph if we have new data or are forced to rebuild
        if need_graph_rebuild:
            logger.info("Expanding temporal graph with synonymy edges")

            add_synonymy_edges = getattr(self.remem.global_config, "add_synonymy_edges", False)
            if add_synonymy_edges:
                self.remem.add_synonymy_edges_between_phrases()

            self._augment_temporal_graph()
            self.remem.save_igraph()
            logger.info("Temporal graph has been rebuilt and saved")
        else:
            logger.info("No new data to process, skipping temporal graph rebuild")

    def rag_for_qa(
        self,
        queries: Union[List[str], List[QuerySolution]],
        num_to_retrieve: int = 5,
        gold_answers: Optional[List[List[str]]] = None,
        gold_docs: Optional[List[List[str]]] = None,
        metrics: Tuple[str, ...] = ("qa_em", "qa_f1", "retrieval_recall"),
        question_metadata: Optional[List[Dict]] = None,
        **kwargs,
    ) -> Tuple[List[QuerySolution], List[str], List[Dict], Dict, Dict]:
        """Run QA evaluation using agent-based retrieval for temporal data."""
        query_solutions = []
        all_response_message = []
        all_metadata = None
        overall_retrieval_metrics = {}

        # Collect agent answers for potential evaluation
        agent_answers = []

        for q_idx, query in enumerate(queries):
            current_gold_answer = self._extract_gold_answer(gold_answers, q_idx)
            sorted_chunk_ids, sorted_chunk_scores, agent_result = self._rag_each_query(
                query, return_chunk="verbatim", gold_answer=current_gold_answer
            )

            top_k_chunks_content, top_k_chunks_metadata = self._get_chunks_content_and_metadata(
                sorted_chunk_ids, "verbatim", num_to_retrieve
            )

            # Create QuerySolution with agent answer
            query_solution = QuerySolution(
                question=query,
                docs=top_k_chunks_content,
                question_metadata=(
                    question_metadata[q_idx] if question_metadata and q_idx < len(question_metadata) else {}
                ),
                doc_scores=sorted_chunk_scores,
                doc_metadata=top_k_chunks_metadata,
                graph_seeds=[],
                gold_answers=gold_answers[q_idx] if gold_answers and q_idx < len(gold_answers) else [],
            )
            # Set the agent answer and session logs
            if len(agent_result) > 2:
                predicted_answer = agent_result[2].get("agent_answer", "")
                query_solution.answer = predicted_answer
                agent_answers.append(predicted_answer)
                query_solution.agent_session_logs = agent_result[2].get("agent_session_logs")
            else:
                agent_answers.append("")

            query_solutions.append(query_solution)

        # Compute overall QA and retrieval metrics if gold answers/docs are provided
        qa_evaluators, retrieval_evaluators = self.remem.get_evaluators(gold_answers, gold_docs, metrics)
        overall_qa_metrics = {}

        # Compute retrieval metrics if gold docs are provided
        if gold_docs and retrieval_evaluators:
            overall_retrieval_metrics = self.remem.evaluate_retrieval(gold_docs, query_solutions, retrieval_evaluators)

        if gold_answers and agent_answers:
            overall_qa_metrics = self.remem.evaluate_qa(gold_answers, qa_evaluators, query_solutions, question_metadata)

        return query_solutions, all_response_message, all_metadata, overall_retrieval_metrics, overall_qa_metrics

    def retrieve_each_query(self, query: str, return_chunk: Optional[str] = None):
        """
        Retrieve chunks using the multi-turn agent framework for temporal data.

        Args:
            query: Query string
            return_chunk: Whether to return chunk content
            gold_answer: Gold/reference answer for logging purposes
        """
        return self._rag_each_query(query, return_chunk)

    def _rag_each_query(self, query: str, return_chunk=None, gold_answer=None):
        """
        Retrieve chunks using the multi-turn agent framework for temporal data.
        """

        if not self.remem.ready_to_retrieve:
            self._prepare_retrieval_objects()

        # Prepare node chunks dictionary for temporal data
        node_chunks_dict = {}

        # Add chunks from all available stores
        store_names = ["verbatim", "temporal_entities", "time_entities", "facts"]
        for store_name in store_names:
            store = self.remem.episodic_embedding_stores.get(store_name)
            if store:
                store_chunks = store.get_hash_id_to_row_readonly()
                node_chunks_dict.update(store_chunks)

        # Create a new GraphAgent instance for each query to ensure thread safety
        from ..agent.graph_agent import GraphAgent

        graph_agent = GraphAgent(
            llm_model=self.remem.qa_llm, node_chunks_dict=node_chunks_dict, remem_instance=self.remem, logger=logger
        )

        # Run agent-based retrieval (no need to pass max_steps, it's read from config)
        agent_result = graph_agent.retrieve_with_agent(
            query=query, gold_answer=gold_answer, beam_size=self.remem.global_config.qa_top_k
        )

        # Convert agent results to format compatible with traditional retrieval
        assert isinstance(agent_result, tuple) and len(agent_result) >= 2, "Agent result should be a tuple"
        agent_chunk_ids, agent_chunk_scores = agent_result[:2]

        # Convert chunk hash IDs to indices based on return_chunk type
        if return_chunk == "verbatim":
            entry_keys = self.remem.entry_keys["verbatim"]
        elif return_chunk == "facts":
            entry_keys = self.remem.entry_keys["facts"]
        else:
            # Default to verbatim if return_chunk is None or unspecified
            entry_keys = self.remem.entry_keys["verbatim"]

        # Create hash_id to index mapping
        hash_to_index = {hash_id: idx for idx, hash_id in enumerate(entry_keys)}

        # Convert hash IDs to indices
        sorted_chunk_ids = []
        sorted_chunk_scores = []
        for i, chunk_id in enumerate(agent_chunk_ids):
            if chunk_id in hash_to_index:
                sorted_chunk_ids.append(hash_to_index[chunk_id])
                sorted_chunk_scores.append(agent_chunk_scores[i] if i < len(agent_chunk_scores) else 0.0)

        # Convert to numpy arrays to match expected format
        sorted_chunk_ids = np.array(sorted_chunk_ids)
        sorted_chunk_scores = np.array(sorted_chunk_scores)

        return sorted_chunk_ids, sorted_chunk_scores, agent_result

    # Helper methods
    def _get_facts_from_chunk(self, chunk):
        """Helper function to extract temporal facts from a chunk."""
        facts = []
        if isinstance(chunk, dict):
            facts = chunk.get("facts", [])
        elif isinstance(chunk, TemporalRawOutput):
            facts = chunk.facts if chunk.facts else []
        else:
            logger.warning(f"Unexpected chunk type: {type(chunk)}.")
        return facts

    def _extract_entities_from_facts(self, facts):
        """Helper function to extract entities from temporal facts."""
        entities = set()
        for triple in facts:
            if isinstance(triple, dict):
                subject = triple.get("subject", "")
                obj = triple.get("object", "")
                if subject:
                    entities.add(subject)
                if obj:
                    entities.add(obj)
        return entities

    def _get_verbatim_from_chunk(self, chunk):
        """Helper function to extract verbatim text from a chunk."""
        if isinstance(chunk, dict):
            return chunk.get("verbatim", "")
        elif isinstance(chunk, TemporalRawOutput):
            return chunk.verbatim if hasattr(chunk, "verbatim") else ""
        return ""

    def _get_chunks_content_and_metadata(self, sorted_chunk_ids, return_chunk, num_to_retrieve):
        """Helper function to get chunk content and metadata."""
        if return_chunk == "verbatim":
            entry_keys = self.remem.entry_keys["verbatim"]
            store = self.remem.episodic_embedding_stores["verbatim"]

            # Limit processing to only what we need first
            limited_chunk_ids = sorted_chunk_ids[:num_to_retrieve]

            # Get hash_ids we need to fetch
            hash_ids_to_fetch = [entry_keys[idx] for idx in limited_chunk_ids]

            # Single batch call to get all required rows
            chunk_rows = store.get_rows(hash_ids_to_fetch)

            # Extract content and hash_ids from the batch result (single pass)
            top_k_chunks_content = []
            top_k_chunks_hash_id = []
            for hash_id in hash_ids_to_fetch:
                row = chunk_rows[hash_id]
                top_k_chunks_content.append(row["content"])
                top_k_chunks_hash_id.append(row["hash_id"])

            # Get metadata in another batch call if needed
            metadata_rows = store.get_rows(top_k_chunks_hash_id)
            top_k_chunks_metadata = [metadata_rows[hash_id].get("metadata", {}) for hash_id in top_k_chunks_hash_id]

        elif return_chunk == "facts":
            entry_keys = self.remem.entry_keys["facts"]
            store = self.remem.episodic_embedding_stores["facts"]

            # Limit processing to only what we need first
            limited_chunk_ids = sorted_chunk_ids[:num_to_retrieve]

            # Get hash_ids we need to fetch
            hash_ids_to_fetch = [entry_keys[idx] for idx in limited_chunk_ids]

            # Single batch call to get all required rows
            chunk_rows = store.get_rows(hash_ids_to_fetch)

            # Extract content from the batch result
            top_k_chunks_content = [chunk_rows[hash_id]["content"] for hash_id in hash_ids_to_fetch]
            top_k_chunks_metadata = []

        else:
            raise ValueError(f"Invalid return_chunk type: {return_chunk}")

        return top_k_chunks_content, top_k_chunks_metadata

    def _extract_gold_answer(self, gold_answers, q_idx):
        """Helper function to extract gold answer for a specific query index."""
        if not gold_answers or q_idx >= len(gold_answers):
            return None

        gold_answer = gold_answers[q_idx]
        if isinstance(gold_answer, list) and len(gold_answer) > 0:
            return gold_answer[0]
        elif isinstance(gold_answer, str):
            return gold_answer
        return None

    def _merge_temporal_extraction_results(self, all_openie_info, chunk_keys_to_save, temporal_results_dict):
        """Merge temporal extraction results into the overall OpenIE info."""
        chunk_key_to_openie_idx = {info["chunk_id"]: idx for idx, info in enumerate(all_openie_info)}

        for chunk_key in chunk_keys_to_save:
            if chunk_key in temporal_results_dict:
                temporal_result = temporal_results_dict[chunk_key]

                openie_entry = {
                    "chunk_id": chunk_key,
                    "facts": temporal_result.facts if hasattr(temporal_result, "facts") else [],
                    "verbatim": temporal_result.verbatim if hasattr(temporal_result, "verbatim") else "",
                }

                if chunk_key in chunk_key_to_openie_idx:
                    all_openie_info[chunk_key_to_openie_idx[chunk_key]] = openie_entry
                else:
                    all_openie_info.append(openie_entry)

    def _augment_temporal_graph(self):
        """Augment the temporal graph with additional nodes and edges using the episodic pattern."""
        logger.info("Augmenting temporal graph")
        # Add nodes first, then edges
        self._add_new_temporal_nodes()
        self._add_new_temporal_edges()

        try:

            # nodes_by_type = get_nodes_by_type(self.remem.graph)
            # edges = get_all_edge_properties(self.remem.graph)
            logger.info("Temporal graph construction completed!")
            print(self._get_graph_info())
        except Exception as e:
            logger.warning(f"Could not get graph properties: {e}")

    def _add_new_temporal_edges(self):
        """Add temporal edges directly from collected edge information."""
        if not hasattr(self.remem, "temporal_edges") or len(self.remem.temporal_edges) == 0:
            logger.warning("No temporal edges to add")
            return

        existing_edges = set()
        if self.remem.graph.ecount() > 0:
            try:
                for edge in self.remem.graph.es:
                    source_name = self.remem.graph.vs[edge.source]["name"]
                    target_name = self.remem.graph.vs[edge.target]["name"]
                    existing_edges.add((source_name, target_name))
            except (KeyError, IndexError):
                logger.warning("Could not build existing edges set")

        new_edges = []
        edge_weights = []
        edge_predicates = []
        edge_qualifiers = []
        edge_source_content = []
        edge_target_content = []
        edge_source_hash_id = []
        edge_target_hash_id = []

        # Create name to index mapping
        igraph_hash_id_to_idx = {}
        for idx, node in enumerate(self.remem.graph.vs):
            if "name" in node.attributes():
                igraph_hash_id_to_idx[node["name"]] = idx

        def get_entity_content_from_key(entity_key):
            """Get entity content from graph node using the entity key."""
            try:
                node_idx = igraph_hash_id_to_idx[entity_key]
                return self.remem.graph.vs[node_idx].get("content", entity_key)
            except (ValueError, KeyError):
                return entity_key

        # Process collected temporal edges
        for edge_info in self.remem.temporal_edges:
            source_hash_id = edge_info["source"]
            target_hash_id = edge_info["target"]

            if (source_hash_id, target_hash_id) not in existing_edges:
                if source_hash_id in igraph_hash_id_to_idx and target_hash_id in igraph_hash_id_to_idx:
                    source_idx = igraph_hash_id_to_idx[source_hash_id]
                    target_idx = igraph_hash_id_to_idx[target_hash_id]

                    new_edges.append((source_idx, target_idx))
                    edge_weights.append(edge_info.get("weight", 1.0))
                    edge_predicates.append(edge_info.get("predicate", ""))
                    edge_qualifiers.append(edge_info.get("qualifiers", {}))
                    edge_source_content.append(get_entity_content_from_key(source_hash_id))
                    edge_target_content.append(get_entity_content_from_key(target_hash_id))
                    edge_source_hash_id.append(source_hash_id)
                    edge_target_hash_id.append(target_hash_id)

        if new_edges:
            logger.info(f"Adding {len(new_edges)} temporal edges to graph")
            self.remem.graph.add_edges(
                new_edges,
                attributes={
                    "weight": edge_weights,
                    "predicate": edge_predicates,
                    "qualifiers": edge_qualifiers,
                    "source_content": edge_source_content,
                    "target_content": edge_target_content,
                    "source_hash_id": edge_source_hash_id,
                    "target_hash_id": edge_target_hash_id,
                },
            )

    def _add_new_temporal_nodes(self):
        """Add temporal-specific nodes to the graph using the episodic pattern."""
        existing_nodes = {v["name"]: v for v in self.remem.graph.vs if "name" in v.attributes()}

        # entries for temporal graph - verbatim, facts, entities
        entries = {"verbatim", "facts", "entity"}
        all_nodes = {}

        for entry in entries:
            if entry in self.remem.episodic_embedding_stores:
                store_nodes = self.remem.episodic_embedding_stores[
                    entry
                ].get_text_for_all_rows()  # Need deepcopy: data will be modified below
                all_nodes.update(store_nodes)

        # Process all nodes and add missing ones
        new_nodes = {}
        for node_hash_id, node in all_nodes.items():
            node["name"] = node_hash_id
            if node_hash_id not in existing_nodes:
                new_nodes[node_hash_id] = node

        if len(new_nodes) > 0:
            logger.info(f"Adding {len(new_nodes)} new temporal nodes to graph")

            all_attributes = set()
            for node_data in new_nodes.values():
                all_attributes.update(node_data.keys())

            node_attributes = {}
            for attr_name in all_attributes:
                node_attributes[attr_name] = [node_data.get(attr_name, "") for node_data in new_nodes.values()]

            # Check that each attribute list has the correct length
            for attr_name, attr_values in node_attributes.items():
                assert len(attr_values) == len(new_nodes), f"Attribute {attr_name} has wrong length"

            try:
                self.remem.graph.add_vertices(len(new_nodes), attributes=node_attributes)
            except Exception as e:
                logger.error(f"Error adding nodes to graph: {e}")

    def _prepare_retrieval_objects(self):
        """Prepare temporal-specific retrieval objects and data structures."""
        logger.info("Preparing temporal retrieval objects...")

        self.remem.query_to_embedding = {"triple": {}, "passage": {}}

        # Load keys for temporal entries
        temporal_entries = set(["verbatim", "facts", "entity"])
        self.remem.entry_keys = {}
        for entry in temporal_entries:
            if entry in self.remem.episodic_embedding_stores:
                self.remem.entry_keys[entry] = self.remem.episodic_embedding_stores[entry].get_all_ids()
            else:
                self.remem.entry_keys[entry] = []

        # Build node name to vertex index mapping if graph exists
        if hasattr(self.remem, "graph") and self.remem.graph is not None and self.remem.graph.vcount() > 0:
            self.remem.node_name_to_vertex_idx = {v["name"]: v.index for v in self.remem.graph.vs}
        else:
            self.remem.node_name_to_vertex_idx = {}

        # Load embeddings for each entry type
        logger.info("Loading temporal embeddings...")
        self.remem.entry_embeddings = {}
        for entry in self.remem.entry_keys:
            if len(self.remem.entry_keys[entry]) > 0:
                self.remem.entry_embeddings[entry] = self.remem.episodic_embedding_stores[entry].embeddings
            else:
                self.remem.entry_embeddings[entry] = np.array([])

        # Initialize rerank filter if specified
        if self.remem.global_config.rerank_dspy_file_path is not None:
            from remem.rerank import DSPyFilter

            self.remem.entry_filter = DSPyFilter(self.remem)
        else:
            self.remem.entry_filter = None

        self.remem.ready_to_retrieve = True

    def _get_graph_info(self):
        """Get information about the temporal graph structure."""
        graph_info = {}

        # get # of verbatim nodes
        verbatim_node_keys = self.remem.episodic_embedding_stores["verbatim"].get_all_ids()
        graph_info["num_verbatim_nodes"] = len(set(verbatim_node_keys))

        # get # of entity nodes
        entity_node_keys = (
            self.remem.episodic_embedding_stores["entity"].get_all_ids()
            if "entity" in self.remem.episodic_embedding_stores
            else []
        )
        graph_info["num_entity_nodes"] = len(set(entity_node_keys))

        # get # of temporal fact nodes
        facts_keys = self.remem.episodic_embedding_stores["facts"].get_all_ids()
        graph_info["num_temporal_fact_nodes"] = len(facts_keys)

        # get # of total nodes
        graph_info["num_total_nodes"] = (
            graph_info["num_verbatim_nodes"] + graph_info["num_entity_nodes"] + graph_info["num_temporal_fact_nodes"]
        )

        # Count different types of edges directly from igraph if available
        if hasattr(self.remem, "graph") and self.remem.graph is not None and self.remem.graph.ecount() > 0:
            graph_info["num_edges"] = self.remem.graph.ecount()
        else:
            graph_info["num_edges"] = 0

        # Add igraph vertex count for completeness
        if hasattr(self.remem, "graph") and self.remem.graph is not None:
            graph_info["igraph_vertex_count"] = self.remem.graph.vcount()
        else:
            graph_info["igraph_vertex_count"] = 0

        return graph_info
