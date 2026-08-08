import json
import logging
import os
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from tqdm import tqdm

from remem.agent.tools.base_tool import BaseTool
from remem.utils.misc_utils import EpisodeRawOutput, QuerySolution, compute_mdhash_id

from ..utils.chunk_utils import make_chunk_content
from .base_strategy import RAGStrategy

logger = logging.getLogger(__name__)


class EpisodicGistStrategy(RAGStrategy):
    """
    Strategy for episodic gist-based extraction and retrieval.
    """

    def __init__(self, remem_instance):
        super().__init__(remem_instance)
        self.concatenate_gists_per_chunk = remem_instance.global_config.concatenate_gists_per_chunk
        self.split_verbatim_per_chunk = remem_instance.global_config.split_verbatim_per_chunk

        # Set up datetime cache
        from remem.utils.datetime_utils import set_datetime_cache_path

        datetime_cache_path = os.path.join(remem_instance.working_dir, "datetime_cache.pkl")
        set_datetime_cache_path(datetime_cache_path)

    def index(self, docs: List) -> None:
        """Internal implementation of index_with_gist."""
        logger.info("Indexing Documents")
        logger.info("Performing Extraction")

        # Load datetime cache at the beginning of indexing
        from remem.utils.datetime_utils import load_datetime_cache

        load_datetime_cache()

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
                logger.info("Found existing graph, embeddings, and OpenIE results. Skipping indexing.")
                # Still need to prepare retrieval objects
                if not self.remem.ready_to_retrieve:
                    self.prepare_retrieval_objects()
                return

        # Continue with indexing process
        self.remem.add_chunk_and_embeddings(docs)
        chunk_dict = self.remem.chunk_embedding_store.hash_id_to_row

        all_openie_info, chunk_keys_to_process = self.remem.load_existing_openie(chunk_dict.keys())
        new_openie_rows = {k: chunk_dict[k] for k in chunk_keys_to_process}

        # Determine if we need to rebuild the graph
        need_graph_rebuild = force_rebuild or len(chunk_keys_to_process) > 0

        if len(chunk_keys_to_process) > 0:
            ie_results = self.remem.openie.batch_openie(new_openie_rows)
            self.merge_gist_extraction_results(all_openie_info, chunk_keys_to_process, ie_results)

        if self.remem.global_config.save_openie:
            self.remem.save_openie_results(all_openie_info)

        episode_results_dict = {openie_info["chunk_id"]: openie_info for openie_info in all_openie_info}
        print(f"# of chunks: {len(chunk_dict)}, # of episode results: {len(episode_results_dict)}")

        # prepare data_store
        element_to_encode = defaultdict(list)
        verbatim_metadata = []
        # Track mapping from original chunk to split verbatim nodes for graph construction
        chunk_id_to_verbatim_id = {}

        entries = ["verbatim", "facts", "gists"]
        for chunk_idx, chunk in enumerate(episode_results_dict.values()):
            chunk_id = chunk["chunk_id"]

            for entry in entries:
                if entry == "verbatim" and self.split_verbatim_per_chunk:
                    split_results = []
                    if isinstance(docs[chunk_idx], list):
                        for v in docs[chunk_idx]:
                            split_results.append([make_chunk_content("message", v), v])
                    elif isinstance(docs[chunk_idx], str):
                        split_results.append([docs[chunk_idx], {"content": docs[chunk_idx]}])
                    else:
                        raise ValueError(
                            f"Expected docs[chunk_idx] to be a list when splitting verbatim, but got type: {type(docs[chunk_idx])}"
                        )

                    verbatim_keys_for_chunk = []
                    for split_content, split_metadata in split_results:
                        element_to_encode[entry].append(split_content)
                        verbatim_metadata.append(split_metadata)
                        # Compute hash key for this split verbatim to use in graph construction
                        verbatim_key = compute_mdhash_id(split_content, prefix="verbatim-")
                        verbatim_keys_for_chunk.append(verbatim_key)

                    # Store mapping for graph construction
                    chunk_id_to_verbatim_id[chunk_id] = verbatim_keys_for_chunk

                elif entry in ["verbatim", "gists"]:
                    if isinstance(chunk[entry], str):
                        element_to_encode[entry].append(chunk[entry])
                    elif isinstance(chunk[entry], list):
                        if entry == "gists" and self.concatenate_gists_per_chunk:
                            # Concatenate all gists in this chunk into a single string
                            concatenated_gists = "\n".join(chunk[entry])
                            element_to_encode[entry].append(concatenated_gists)
                        else:
                            # Default behavior: each gist becomes a separate node
                            element_to_encode[entry].extend(chunk[entry])
                    if entry == "verbatim":
                        if isinstance(docs[chunk_idx], str):
                            if docs[chunk_idx].startswith("{") and docs[chunk_idx].endswith("}"):
                                metadata = json.loads(docs[chunk_idx])
                                # make a string from its "messages": "user: ... assistant: ..."
                                content = "\n\n".join(
                                    [f"{msg['role']}: {msg['content']}" for msg in metadata.get("messages", [])]
                                )
                                metadata["content"] = content  # session_metadata
                            else:
                                content = docs[chunk_idx]
                                metadata = {"content": content}
                        else:
                            chunk_id = chunk.get("chunk_id") if isinstance(chunk, dict) else chunk.chunk_id
                            metadata = chunk_dict[chunk_id]["metadata"]
                        verbatim_metadata.append(metadata)

                elif entry == "facts":
                    if chunk["facts"]:
                        valid_facts = []
                        for f in chunk["facts"]:
                            # Skip facts with empty strings in subject, predicate, or object
                            if isinstance(f, dict) and "subject" in f and "predicate" in f and "object" in f:
                                if f["subject"] and f["predicate"] and f["object"]:  # Check for non-empty strings
                                    valid_facts.append(f)
                                    element_to_encode["entity"].extend([f["subject"], f["object"]])
                            elif isinstance(f, list) and len(f) == 3:
                                if f[0] and f[1] and f[2]:  # Check for non-empty strings
                                    valid_facts.append(f)
                                    element_to_encode["entity"].extend([f[0], f[2]])
                        element_to_encode["facts"].extend(valid_facts)
                    else:
                        logger.warning(f"Chunk {chunk_idx} has no facts, skipping.")
                elif chunk[entry]:
                    element_to_encode[entry].extend(chunk[entry])
        # end for each type

        for entry in entries + ["entity"]:  # encoding each type
            logger.info(f"Encoding {entry}: {len(element_to_encode[entry])}")
            if entry == "facts":
                # Convert facts to strings for storage, handling both old and new formats
                # Prepare fact dictionaries for insert_chunk_dicts method
                fact_dicts = []
                for fact in element_to_encode["facts"]:
                    if isinstance(fact, dict) and "subject" in fact and "predicate" in fact and "object" in fact:
                        # Skip facts with empty strings before processing
                        if not (fact["subject"] and fact["predicate"] and fact["object"]):
                            logger.warning(f"Skipping fact with empty subject, predicate, or object: {fact}")
                            continue
                        # New dictionary format: use readable format for embedding
                        formatted_content = BaseTool.format_fact_dict_for_embedding(fact)
                        fact_dict = {
                            "content": formatted_content,
                            "subject": fact["subject"],
                            "predicate": fact["predicate"],
                            "object": fact["object"],
                            "qualifiers": fact.get("qualifiers", {}),
                        }
                    elif isinstance(fact, list) and len(fact) == 3:
                        # Skip facts with empty strings before processing
                        if not (fact[0] and fact[1] and fact[2]):
                            logger.warning(f"Skipping fact with empty subject, predicate, or object: {fact}")
                            continue
                        # Old list format: convert to readable format
                        fact_dict_temp = {"subject": fact[0], "predicate": fact[1], "object": fact[2], "qualifiers": {}}
                        formatted_content = BaseTool.format_fact_dict_for_embedding(fact_dict_temp)
                        fact_dict = {
                            "content": formatted_content,
                            "subject": fact[0],
                            "predicate": fact[1],
                            "object": fact[2],
                            "qualifiers": {},
                        }
                    else:
                        # Handle unexpected format - don't nest metadata
                        logger.warning(f"Unexpected fact format: {fact}")
                        continue
                    # Final check (this should not be needed anymore after the early checks above)
                    if not (fact_dict["subject"] and fact_dict["predicate"] and fact_dict["object"]):
                        logger.warning(f"Skipping fact with missing subject, predicate, or object: {fact_dict}")
                        continue
                    fact_dicts.append(fact_dict)

                self.remem.episodic_embedding_stores["facts"].insert_chunk_dicts(fact_dicts, "openie")
            elif entry == "verbatim":
                self.remem.episodic_embedding_stores[entry].insert_chunk_dicts(
                    verbatim_metadata, extract_method="message", embed=True
                )
            else:
                embed = False if entry == "entity" else True
                self.remem.episodic_embedding_stores[entry].insert_strings(element_to_encode[entry], embed=embed)

        logger.info("Constructing Graph...")

        # cache parsed datetime strings
        from remem.utils.datetime_utils import preparse_qualifiers_datetime, save_datetime_cache

        all_facts = [
            item["metadata"] for item in self.remem.episodic_embedding_stores.get("facts").hash_id_to_row.values()
        ]
        cached_count = preparse_qualifiers_datetime(all_facts)
        logger.info(f"Cached {cached_count} new datetime strings")
        save_datetime_cache()

        self.remem.node_to_node_count = {}
        self.remem.entity_node_to_num_chunk = {}

        for chunk in episode_results_dict.values():
            chunk_id = chunk["chunk_id"]
            verbatim = chunk["verbatim"]
            gists = chunk["gists"]
            facts = chunk["facts"]

            # Get all verbatim keys for this chunk (could be multiple if split was enabled)
            verbatim_keys = chunk_id_to_verbatim_id.get(chunk_id, [])
            if not verbatim_keys:
                # Fallback: compute verbatim key from original verbatim
                verbatim_key = compute_mdhash_id(verbatim, prefix="verbatim-")
                verbatim_keys = [verbatim_key]

            # verbatim -> gists (connect all verbatim nodes to gist nodes)
            if gists:
                if self.concatenate_gists_per_chunk and isinstance(gists, list):
                    # Concatenate all gists in this chunk into a single node
                    concatenated_gists = "\n".join(gists)
                    gist_keys = [compute_mdhash_id(concatenated_gists, prefix="gists-")]
                else:
                    # Default behavior: each gist becomes a separate node
                    gist_keys = (
                        [compute_mdhash_id(gist, prefix="gists-") for gist in gists]
                        if isinstance(gists, list)
                        else [compute_mdhash_id(gists, prefix="gists-")]
                    )

                # Connect each verbatim node to each gist node
                for verbatim_key in verbatim_keys:
                    for gist_key in gist_keys:
                        self.remem.node_to_node_count[(verbatim_key, gist_key)] = (
                            self.remem.node_to_node_count.get((verbatim_key, gist_key), 0) + 1
                        )
            else:
                gist_keys = []
                logger.warning(f"No gists found for chunk {chunk['chunk_id']}")

            entity_in_chunk = set()
            if facts:
                for fact in facts:
                    # Handle both old list format and new dictionary format
                    if isinstance(fact, dict):
                        if not (fact.get("subject") and fact.get("object")):
                            continue
                        subject = fact["subject"]
                        object_entity = fact["object"]
                        if not (isinstance(subject, str) and isinstance(object_entity, str)):
                            logger.warning(f"Invalid fact format: {fact}. Skipping this fact.")
                            continue
                        fact_content = BaseTool.format_fact_dict_for_embedding(fact)
                        fact_key = compute_mdhash_id(fact_content, prefix="facts-")
                    elif isinstance(fact, list):
                        if not (len(fact) == 3 and fact[0] and fact[2]):
                            continue
                        subject = fact[0]
                        object_entity = fact[2]
                        fact_dict_temp = {"subject": fact[0], "predicate": fact[1], "object": fact[2], "qualifiers": {}}
                        fact_content = BaseTool.format_fact_dict_for_embedding(fact_dict_temp)
                        fact_key = compute_mdhash_id(fact_content, prefix="facts-")
                    else:
                        continue

                    # subject -> object (only one direction)
                    subject_key = compute_mdhash_id(subject, prefix="entity-")
                    object_key = compute_mdhash_id(object_entity, prefix="entity-")
                    self.remem.node_to_node_count[(subject_key, object_key)] = (
                        self.remem.node_to_node_count.get((subject_key, object_key), 0) + 1
                    )

                    # verbatim -> subject/object (for all verbatim keys in this chunk)
                    for verbatim_key in verbatim_keys:
                        self.remem.node_to_node_count[(verbatim_key, subject_key)] = 1.0
                        self.remem.node_to_node_count[(verbatim_key, object_key)] = 1.0

                    # gist -> subject/object (for all gist keys)
                    for gist_key in gist_keys:
                        self.remem.node_to_node_count[(gist_key, object_key)] = (
                            self.remem.node_to_node_count.get((gist_key, object_key), 0) + 1
                        )
                        self.remem.node_to_node_count[(gist_key, subject_key)] = (
                            self.remem.node_to_node_count.get((gist_key, subject_key), 0) + 1
                        )

                        # gist -> facts (direct connection)
                        self.remem.node_to_node_count[(gist_key, fact_key)] = (
                            self.remem.node_to_node_count.get((gist_key, fact_key), 0) + 1
                        )

                    entity_in_chunk.add(subject)
                    entity_in_chunk.add(object_entity)

            for entity in entity_in_chunk:
                self.remem.entity_node_to_num_chunk[entity] = self.remem.entity_node_to_num_chunk.get(entity, 0) + 1

            # gist to other entries
            # entry_to_connect = ["space", "causality", "intent", "emotion"]
            entry_to_connect = []
            for entry in entry_to_connect:
                if entry in chunk and chunk[entry]:
                    for k in chunk[entry]:
                        if not isinstance(k, str):
                            continue
                        # verbatim_keys -> other entries (for all verbatim keys in this chunk)
                        other_key = compute_mdhash_id(k, prefix=f"{entry}-")
                        for verbatim_key in verbatim_keys:
                            self.remem.node_to_node_count[(verbatim_key, other_key)] = (
                                self.remem.node_to_node_count.get((verbatim_key, other_key), 0) + 1
                            )

        # end for each chunk

        # Only rebuild graph if we have new data or are forced to rebuild
        if need_graph_rebuild:
            logger.info("Expanding graph with synonymy edges...")
            # Add synonymy edges for gists (gist to gist only)
            entry_to_encode = ["gists"]
            for entry in entry_to_encode:
                if entry in self.remem.episodic_embedding_stores:
                    node_keys = self.remem.episodic_embedding_stores[entry].get_all_ids()
                    embeddings = list(self.remem.episodic_embedding_stores[entry].get_embeddings(node_keys))
                    logger.info(f"Adding synonymy edges for {len(node_keys)} {entry} nodes...")
                    self.remem.add_synonymy_edges(
                        embeddings, node_keys, self.remem.episodic_embedding_stores[entry].hash_id_to_row
                    )
                    logger.info(f"Completed synonymy edge addition for {entry}")
                else:
                    raise ValueError(f"Key {entry} not found in episodic embedding stores.")

            self._augment_episodic_graph()
            self.remem.save_igraph()
            logger.info("Graph has been rebuilt and saved!")
        else:
            logger.info("No new data to process, skipping graph rebuild")

    def _augment_episodic_graph(self):
        """Augment the episodic graph with nodes and edges."""
        self._add_new_nodes()
        self._add_new_edges()

        # nodes_by_type = get_nodes_by_type(self.remem.graph)
        # edges = get_all_edge_properties(self.remem.graph)
        logger.info("Graph construction completed!")
        print(self.get_graph_info())

    def _add_new_nodes(self):
        """Add new nodes to the graph."""
        existing_nodes = {v["name"]: v for v in self.remem.graph.vs if "name" in v.attributes()}
        facts = self.remem.episodic_embedding_stores["facts"].hash_id_to_row

        # entries = {"verbatim", "gist", "time", "space", "causality", "intent", "emotion"}
        entries = {"verbatim", "gists", "entity", "facts"}
        all_nodes = {}
        for entry in entries:
            if entry in self.remem.episodic_embedding_stores:
                all_nodes.update(self.remem.episodic_embedding_stores[entry].hash_id_to_row)

        for fact_key in facts:
            try:
                fact = facts[fact_key]["metadata"]
                if isinstance(fact, str):
                    fact = json.loads(fact)

                if isinstance(fact, dict):
                    if not (fact.get("subject") and fact.get("object")):
                        continue
                    subject = fact["subject"]
                    object_entity = fact["object"]
                elif isinstance(fact, list):
                    if not (len(fact) == 3 and fact[0] and fact[2]):
                        continue
                    subject = fact[0]
                    object_entity = fact[2]
                else:
                    raise ValueError(f"Unexpected fact format: {fact}")
            except Exception as e:
                logger.warning(f"Could not add fact {fact_key}: {e}")
                continue

            if not isinstance(subject, str) or not isinstance(object_entity, str):
                logger.warning(f"Invalid fact format: {fact}. Skipping this fact.")
                continue

            subject_key = compute_mdhash_id(subject, prefix="entity-")
            object_key = compute_mdhash_id(object_entity, prefix="entity-")
            all_nodes[subject_key] = {"hash_id": subject_key, "content": subject}
            all_nodes[object_key] = {"hash_id": object_key, "content": object_entity}

        new_nodes = {}
        for node_hash_id, node in all_nodes.items():
            node["name"] = node_hash_id
            # Add human-readable label based on content
            node["label"] = node.get("content", node_hash_id)
            if node_hash_id not in existing_nodes:
                for k, v in node.items():
                    if k not in new_nodes:
                        new_nodes[k] = []
                    new_nodes[k].append(v)

        if len(new_nodes) > 0:
            self.remem.graph.add_vertices(n=len(next(iter(new_nodes.values()))), attributes=new_nodes)

    def _add_new_edges(self):
        """Add new edges to the graph."""
        if len(self.remem.node_to_node_count) == 0:
            return

        existing_edges = set()
        if self.remem.graph.ecount() > 0:
            existing_edges = {
                (self.remem.graph.vs[e.source]["name"], self.remem.graph.vs[e.target]["name"])
                for e in self.remem.graph.es
            }

        # Build node name to vertex index mapping
        self.remem.node_name_to_vertex_idx = {v["name"]: v.index for v in self.remem.graph.vs}

        edges_to_add = []
        weights_to_add = []
        sources_to_add = []
        targets_to_add = []

        for (source_node, target_node), weight in self.remem.node_to_node_count.items():
            # Skip if edge already exists
            if (source_node, target_node) in existing_edges:
                continue

            # Skip if either node doesn't exist in the graph
            if (
                source_node not in self.remem.node_name_to_vertex_idx
                or target_node not in self.remem.node_name_to_vertex_idx
            ):
                continue

            source_idx = self.remem.node_name_to_vertex_idx[source_node]
            target_idx = self.remem.node_name_to_vertex_idx[target_node]

            edges_to_add.append((source_idx, target_idx))
            weights_to_add.append(weight)
            sources_to_add.append(source_node)
            targets_to_add.append(target_node)

        if edges_to_add:
            # Add edges with weights
            self.remem.graph.add_edges(edges_to_add, {"weight": weights_to_add})
            logger.info(f"Added {len(edges_to_add)} new edges to the graph")

    def rag_for_qa(
        self,
        queries: Union[List[str], List[QuerySolution]],
        num_to_retrieve: int = 5,
        gold_answers: Optional[List[List[str]]] = None,
        gold_docs: Optional[List[List[str]]] = None,
        metrics: Tuple[str, ...] = ("qa_em", "qa_f1", "retrieval_recall"),
        question_metadata: Optional[List[Dict]] = None,
        parallel: bool = True,
        max_workers: int = 8,
        return_chunk: str = "gists",
        evaluate_per_sample: bool = False,
        save_per_sample: bool = False,
        force_rag: bool = False,
        inference_type: str = "default",
        **kwargs,
    ) -> Tuple[List[QuerySolution], List[str], List[Dict], Dict, Dict]:
        """Perform QA using episodic gist approach.

        Args:
            evaluate_per_sample: If True, evaluates metrics for each sample immediately
                               after processing, rather than at the end. This allows for
                               real-time metric tracking during processing.
            save_per_sample: If True, saves each sample's result individually in a samples/
                           subdirectory as rag_results_question_{idx}_{inference_type}.json
            force_rag: If True, forces recomputation even if saved results exist
            inference_type: String identifier for the inference type, used in filenames
        """
        # Handle per-sample saving/loading if enabled
        if save_per_sample:
            samples_dir = os.path.join(self.remem.working_dir, "samples")
            if not os.path.exists(samples_dir):
                os.makedirs(samples_dir, exist_ok=True)

            # Pre-check which samples need processing and which can be loaded
            queries_to_process = []
            loaded_results = {}

            for q_idx, query in enumerate(queries):
                sample_file = os.path.join(samples_dir, f"rag_results_question_{q_idx}_{inference_type}.json")

                if os.path.exists(sample_file) and not force_rag:
                    try:
                        import json

                        with open(sample_file, "r") as f:
                            sample_data = json.load(f)

                        sample = sample_data["sample"] if "sample" in sample_data else sample_data

                        # Reconstruct QuerySolution from saved data
                        loaded_query_solution = QuerySolution(
                            question=sample.get("question", query),
                            docs=sample.get("docs", []),
                            question_metadata=sample.get("question_metadata", {}),
                            doc_scores=np.array(sample.get("doc_scores", [])),
                            doc_metadata=sample.get("doc_metadata", []),
                            graph_seeds=sample.get("graph_seeds", []),
                            gold_answers=sample.get("gold_answers", []),
                            gold_docs=sample.get("gold_docs", []),  # Add gold_docs from cached data
                            answer=sample.get("answer", ""),
                            qa_rationale=sample.get("qa_rationale", ""),
                            agent_session_logs=sample.get("agent_session_logs", None),
                            metrics=sample.get("sample_metrics", None),
                        )

                        # Construct agent_result_dict
                        agent_answer = sample.get("answer", "")
                        agent_result_dict = {
                            "agent_answer": agent_answer,
                            "reasoning_trace": sample.get("qa_rationale", ""),
                            "agent_session_logs": sample.get("agent_session_logs", None),
                        }

                        loaded_results[q_idx] = (loaded_query_solution, agent_result_dict, agent_answer)
                        logger.info(f"Loaded cached results for question {q_idx} from {sample_file}")
                    except Exception as e:
                        logger.warning(f"Failed to load cached result for question {q_idx}: {e}. Will recompute.")
                        queries_to_process.append((q_idx, query))
                else:
                    queries_to_process.append((q_idx, query))

            logger.info(f"Loaded {len(loaded_results)} cached results, processing {len(queries_to_process)} queries")

            # If all queries are cached, return the loaded results
            if len(queries_to_process) == 0:
                query_solutions = [loaded_results[i][0] for i in range(len(queries))]
                all_response_message = [loaded_results[i][1] for i in range(len(queries))]
                agent_answers = [loaded_results[i][2] for i in range(len(queries))]

                # Set gold_docs for loaded results if available and not already set
                if gold_docs:
                    for i, query_solution in enumerate(query_solutions):
                        if i < len(gold_docs) and (not query_solution.gold_docs or len(query_solution.gold_docs) == 0):
                            query_solution.gold_docs = gold_docs[i]

                # Compute overall metrics
                all_metadata = None
                overall_retrieval_metrics = {}
                qa_evaluators, retrieval_evaluators = self.remem.get_evaluators(gold_answers, gold_docs, metrics)
                overall_qa_metrics = {}
                if gold_answers is not None and len(agent_answers) > 0:
                    question_types = [q["type"] for q in question_metadata] if question_metadata else []
                    try:
                        for qa_evaluator in qa_evaluators:
                            overall_metric_result, example_metric_results = qa_evaluator.calculate_metric_scores(
                                gold_answers=gold_answers,
                                predicted_answers=agent_answers,
                                question_types=question_types,
                                questions=queries,
                            )
                            overall_qa_metrics.update(overall_metric_result)
                    except Exception as e:
                        logger.warning(f"Could not compute QA metrics: {e}")

                return (
                    query_solutions,
                    all_response_message,
                    all_metadata,
                    overall_retrieval_metrics,
                    overall_qa_metrics,
                )

        else:  # If not using per-sample saving, process all queries
            queries_to_process = [(i, q) for i, q in enumerate(queries)]
            loaded_results = {}
            samples_dir = None
        all_metadata = None
        overall_retrieval_metrics = {}
        question_types = [q.get("type", None) for q in question_metadata] if question_metadata else []

        # Thread-safe containers - initialize with loaded results
        query_solutions = []  # Will be populated in order
        all_response_message = []  # Will be populated in order
        agent_answers = []  # Will be populated in order

        # Pre-allocate with loaded results if any
        for i in range(len(queries)):
            if i in loaded_results:
                query_solution, agent_result_dict, agent_answer = loaded_results[i]
                query_solutions.append(query_solution)
                all_response_message.append(agent_result_dict)
                agent_answers.append(agent_answer)

                # Set gold_docs if available and not already set
                if (
                    gold_docs
                    and i < len(gold_docs)
                    and (not query_solution.gold_docs or len(query_solution.gold_docs) == 0)
                ):
                    query_solution.gold_docs = gold_docs[i]
            else:
                # Placeholder for queries that need processing
                query_solutions.append(None)
                all_response_message.append(None)
                agent_answers.append(None)

        # Lock for thread-safe logging/printing operations
        print_lock = threading.Lock()

        # Initialize evaluators once if per-sample evaluation is enabled
        evaluators = None
        running_metrics = {}  # Track running averages for per-sample evaluation
        if evaluate_per_sample and gold_answers is not None:
            qa_evaluators, retrieval_evaluators = self.remem.get_evaluators(gold_answers, gold_docs, metrics)
            evaluators = (qa_evaluators, retrieval_evaluators)
            logger.debug(f"Per-sample evaluation enabled with evaluators: {[type(e).__name__ for e in qa_evaluators]}")

        if parallel:
            logger.info(f"Processing {len(queries_to_process)} queries in parallel with {max_workers} workers")
            # Prepare arguments for parallel processing - only for queries that need processing
            args_list = []
            for q_idx, query in queries_to_process:
                current_gold_answer = None
                if gold_answers and q_idx < len(gold_answers):
                    # gold_answers is usually a list of lists, so we take the first answer if multiple exist
                    if isinstance(gold_answers[q_idx], list) and len(gold_answers[q_idx]) > 0:
                        current_gold_answer = gold_answers[q_idx][0]
                    elif isinstance(gold_answers[q_idx], str):
                        current_gold_answer = gold_answers[q_idx]

                question_metadata_item = (
                    question_metadata[q_idx] if question_metadata and q_idx < len(question_metadata) else {}
                )

                args_list.append(
                    (
                        q_idx,
                        query,
                        self.remem,
                        return_chunk,
                        num_to_retrieve,
                        current_gold_answer,
                        question_metadata_item,
                        evaluate_per_sample,
                        evaluators,
                        question_types,
                        print_lock,
                        running_metrics,
                        len(queries),
                        save_per_sample,
                        samples_dir,
                        inference_type,
                    )
                )

            # Process queries in parallel
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks
                future_to_idx = {executor.submit(self._process_single_query, args): args[0] for args in args_list}

                # Use tqdm with as_completed for progress tracking
                with print_lock:
                    total_to_process = len(queries_to_process)
                    pbar = tqdm(
                        total=total_to_process,
                        desc=f"Running RAG (Parallel, to run {total_to_process}, total {len(queries)})",
                    )

                for future in as_completed(future_to_idx):
                    try:
                        q_idx, query_solution, agent_result_dict, agent_answer = future.result()

                        # Set gold answers and gold docs if available
                        if gold_answers and q_idx < len(gold_answers):
                            query_solution.gold_answers = (
                                gold_answers[q_idx] if gold_answers and q_idx < len(gold_answers) else []
                            )
                        if gold_docs and q_idx < len(gold_docs):
                            query_solution.gold_docs = gold_docs[q_idx] if gold_docs and q_idx < len(gold_docs) else []

                        # Store results in the correct position to maintain order
                        query_solutions[q_idx] = query_solution
                        all_response_message[q_idx] = agent_result_dict
                        agent_answers[q_idx] = agent_answer

                        # Save per-sample result if enabled
                        if save_per_sample and samples_dir:
                            try:
                                from remem.utils.misc_utils import safe_dump_json, safe_serialize_query_solutions

                                sample_file = os.path.join(
                                    samples_dir, f"rag_results_question_{q_idx}_{inference_type}.json"
                                )
                                sample_data = safe_serialize_query_solutions([query_solution])[0]
                                safe_dump_json(sample_data, sample_file)
                            except Exception as e:
                                logger.warning(f"Failed to save sample {q_idx}: {e}")

                        # Thread-safe progress update
                        with print_lock:
                            pbar.update(1)

                    except Exception as e:
                        with print_lock:
                            logger.error(f"Error processing query: {str(e)}")
                            pbar.update(1)

                with print_lock:
                    pbar.close()

        else:
            # Sequential processing - only process queries that need processing
            for q_idx, query in tqdm(
                queries_to_process,
                desc=f"Running RAG (Sequential, to run {len(queries_to_process)}, total {len(queries)})",
                total=len(queries_to_process),
            ):
                # Get gold answer for this query if available
                current_gold_answer = None
                if gold_answers and q_idx < len(gold_answers):
                    # gold_answers is usually a list of lists, so we take the first answer if multiple exist
                    if isinstance(gold_answers[q_idx], list) and len(gold_answers[q_idx]) > 0:
                        current_gold_answer = gold_answers[q_idx][0]
                    elif isinstance(gold_answers[q_idx], str):
                        current_gold_answer = gold_answers[q_idx]

                question_metadata_item = (
                    question_metadata[q_idx] if question_metadata and q_idx < len(question_metadata) else {}
                )

                args = (
                    q_idx,
                    query,
                    self.remem,
                    return_chunk,
                    num_to_retrieve,
                    current_gold_answer,
                    question_metadata_item,
                    evaluate_per_sample,
                    evaluators,
                    question_types,
                    print_lock,
                    running_metrics,
                    len(queries),
                    save_per_sample,
                    samples_dir,
                    inference_type,
                )
                q_idx, query_solution, agent_result_dict, agent_answer = self._process_single_query(args)

                # Set gold answers and gold docs if available
                if gold_answers and q_idx < len(gold_answers):
                    query_solution.gold_answers = (
                        gold_answers[q_idx] if gold_answers and q_idx < len(gold_answers) else []
                    )
                if gold_docs and q_idx < len(gold_docs):
                    query_solution.gold_docs = gold_docs[q_idx] if gold_docs and q_idx < len(gold_docs) else []

                query_solutions[q_idx] = query_solution
                all_response_message[q_idx] = agent_result_dict
                agent_answers[q_idx] = agent_answer

                # Save per-sample result if enabled
                if save_per_sample and samples_dir:
                    try:
                        from remem.utils.misc_utils import safe_dump_json, safe_serialize_query_solutions

                        sample_file = os.path.join(samples_dir, f"rag_results_question_{q_idx}_{inference_type}.json")
                        sample_data = safe_serialize_query_solutions([query_solution])[0]
                        safe_dump_json(sample_data, sample_file)
                    except Exception as e:
                        logger.warning(f"Failed to save sample {q_idx}: {e}")

        # Validate that all positions have been filled - do not filter out None values
        # as this would destroy the correspondence between arrays
        self._fill_missing_query_solutions(query_solutions, all_response_message, agent_answers, queries)

        # Compute overall QA metrics if gold answers are provided and not evaluating per sample
        qa_evaluators, retrieval_evaluators = self.remem.get_evaluators(gold_answers, gold_docs, metrics)
        overall_qa_metrics = {}
        if gold_answers is not None and len(agent_answers) > 0 and not evaluate_per_sample:
            for qa_evaluator in qa_evaluators:
                overall_metric_result, example_metric_results = qa_evaluator.calculate_metric_scores(
                    gold_answers=gold_answers,
                    predicted_answers=agent_answers,
                    question_types=question_types,
                    questions=queries,
                )
                overall_qa_metrics.update(overall_metric_result)

                # Update individual query solution metrics
                for sample_idx, query_solution in enumerate(query_solutions):
                    if query_solution.metrics is None:
                        query_solution.metrics = {}
                    if sample_idx < len(example_metric_results):
                        query_solution.metrics.update(example_metric_results[sample_idx])

        elif evaluate_per_sample:
            # Aggregate metrics from individual samples
            for qa_evaluator in qa_evaluators:
                metric_name = getattr(qa_evaluator, "metric_name", qa_evaluator.__class__.__name__.lower())
                individual_scores = [
                    qs.metrics.get(metric_name, 0.0)
                    for qs in query_solutions
                    if qs.metrics and metric_name in qs.metrics
                ]
                if individual_scores:
                    overall_qa_metrics[metric_name] = np.mean(individual_scores)
                    print(
                        f"Overall {metric_name}: {overall_qa_metrics[metric_name]:.4f} (across {len(individual_scores)} samples)"
                    )
            print("=" * 40)

        return query_solutions, all_response_message, all_metadata, overall_retrieval_metrics, overall_qa_metrics

    def _process_single_query(self, args):
        """Helper function to process a single query. This function is designed to be thread-safe."""
        assert len(args) == 16
        (
            q_idx,
            query,
            remem,
            return_chunk,
            num_to_retrieve,
            current_gold_answer,
            question_metadata_item,
            evaluate_per_sample,
            evaluators,
            question_types,
            print_lock,
            running_metrics,
            total_queries,
            save_per_sample,
            samples_dir,
            inference_type,
        ) = args

        try:
            # Execute query retrieval
            sorted_chunk_ids, sorted_chunk_scores, agent_result = self._rag_each_query(
                remem, query, return_chunk, gold_answer=current_gold_answer, question_metadata=question_metadata_item
            )

            # Process chunks based on return_chunk type - OPTIMIZED VERSION
            if return_chunk == "verbatim":
                # Limit processing to only what we need first
                limited_chunk_ids = sorted_chunk_ids[:num_to_retrieve]

                # Get hash_ids we need to fetch
                hash_ids_to_fetch = [remem.entry_keys["verbatim"][idx] for idx in limited_chunk_ids]

                # Single batch call to get all required rows
                chunk_rows = remem.episodic_embedding_stores["verbatim"].get_rows(hash_ids_to_fetch)

                # Extract content and hash_ids from the batch result (single pass)
                top_k_chunks_content = []
                top_k_chunks_hash_id = []
                for hash_id in hash_ids_to_fetch:
                    row = chunk_rows[hash_id]
                    top_k_chunks_content.append(row["content"])
                    top_k_chunks_hash_id.append(row["hash_id"])

                # Get metadata in another batch call if needed
                metadata_rows = remem.episodic_embedding_stores["verbatim"].get_rows(top_k_chunks_hash_id)
                top_k_chunks_metadata = [
                    metadata_rows[hash_id].get("metadata", None) for hash_id in top_k_chunks_hash_id
                ]

            elif return_chunk == "gists":
                # Limit processing to only what we need first
                limited_chunk_ids = sorted_chunk_ids[:num_to_retrieve]

                # Get hash_ids we need to fetch
                hash_ids_to_fetch = [remem.entry_keys["gists"][idx] for idx in limited_chunk_ids]

                # Single batch call to get all required rows
                chunk_rows = remem.episodic_embedding_stores["gists"].get_rows(hash_ids_to_fetch)

                # Extract content from the batch result
                top_k_chunks_content = [chunk_rows[hash_id]["content"] for hash_id in hash_ids_to_fetch]
                top_k_chunks_metadata = []  # Empty list instead of None to fix type error
            else:
                raise ValueError(f"Invalid return_chunk type: {return_chunk}.")

            # Create QuerySolution with agent answer
            query_solution = QuerySolution(
                question=query,
                docs=top_k_chunks_content,
                question_metadata=question_metadata_item,
                doc_scores=sorted_chunk_scores,
                doc_metadata=top_k_chunks_metadata,
                graph_seeds=[],  # Empty list instead of None to fix type error
                gold_answers=[],  # Will be set from gold_answers parameter after query processing
                gold_docs=[],  # Will be set from gold_docs parameter after query processing
            )

            # Get agent result dictionary safely
            agent_result_dict = agent_result[2] if len(agent_result) > 2 and isinstance(agent_result[2], dict) else {}
            query_solution.answer = agent_result_dict.get("agent_answer", "")
            query_solution.qa_rationale = agent_result_dict.get("reasoning_trace", "")

            # Include agent session logs if available
            if agent_result_dict:
                query_solution.agent_session_logs = agent_result_dict.get("agent_session_logs")

            agent_answer = agent_result_dict.get("agent_answer", "")

            # Perform per-sample evaluation if enabled
            if evaluate_per_sample and evaluators and current_gold_answer is not None:
                qa_evaluators, retrieval_evaluators = evaluators
                try:
                    sample_metrics = {}
                    for qa_evaluator in qa_evaluators:
                        # Get question type for this sample
                        sample_question_type = question_types[q_idx] if q_idx < len(question_types) else None

                        # Calculate metric for this single sample
                        _, example_metric_results = qa_evaluator.calculate_metric_scores(
                            gold_answers=[[current_gold_answer]],  # Wrap in list for batch processing
                            predicted_answers=[agent_answer],
                            question_types=[sample_question_type] if sample_question_type else [],
                            questions=[query],
                        )

                        # Update query solution metrics
                        if query_solution.metrics is None:
                            query_solution.metrics = {}
                        if example_metric_results and len(example_metric_results) > 0:
                            query_solution.metrics.update(example_metric_results[0])
                            sample_metrics.update(example_metric_results[0])

                    # Thread-safe printing of current sample metrics and running averages
                    if sample_metrics and print_lock:
                        with print_lock:
                            # Update running metrics - ensure all sample metrics are tracked
                            for metric_name, score in sample_metrics.items():
                                # Initialize metric if not present (handles dynamic metric names)
                                if metric_name not in running_metrics:
                                    running_metrics[metric_name] = {"sum": 0.0, "count": 0}
                                running_metrics[metric_name]["sum"] += score
                                running_metrics[metric_name]["count"] += 1

                            # Print current sample metrics
                            sample_metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in sample_metrics.items()])

                            # Print running averages
                            running_avgs = {}
                            for metric_name, data in running_metrics.items():
                                if data["count"] > 0:
                                    running_avgs[metric_name] = data["sum"] / data["count"]

                            # Get total sample count for display
                            total_samples = (
                                max(data["count"] for data in running_metrics.values()) if running_metrics else 0
                            )

                            running_metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in running_avgs.items()])

                            # Use clean log format with flexible metric shortening

                            if total_samples > 1:
                                logger.info(
                                    f"📊 Sample {q_idx}: {sample_metrics_str} | Avg: {running_metrics_str} | Total: {total_samples}"
                                )
                            else:
                                logger.info(f"📊 {sample_metrics_str}")

                except Exception as e:
                    logger.warning(f"Warning: Could not compute per-sample QA metrics for query {q_idx}: {e}")

            return q_idx, query_solution, agent_result_dict, agent_answer

        except Exception as e:
            # Thread-safe error logging
            logger.error(f"Error processing query {q_idx}: {str(e)}")
            raise e
            # Return empty results to maintain consistency
            empty_query_solution = QuerySolution(
                question=query,
                docs=[],
                question_metadata=question_metadata_item,
                doc_scores=np.array([]),
                doc_metadata=[],
                graph_seeds=[],
                gold_answers=[],
                gold_docs=[],
                answer="",
                qa_rationale="",
                agent_session_logs=None,
                metrics={},
            )
            return q_idx, empty_query_solution, {}, ""

    def _rag_each_query(self, remem, query, return_chunk=None, gold_answer=None, question_metadata=None):
        """Retrieve chunks using the multi-turn agent framework."""

        if not remem.ready_to_retrieve:
            self.prepare_retrieval_objects()

        # Prepare node chunks dictionary
        node_chunks_dict = {}

        # Add verbatim chunks
        verbatim_store = remem.episodic_embedding_stores.get("verbatim")
        if verbatim_store:
            verbatim_chunks = verbatim_store.hash_id_to_row
            assert isinstance(
                verbatim_chunks, dict
            ), f"verbatim_chunks should be dict, got {type(verbatim_chunks)}: {verbatim_chunks}"
            for node_id, data in verbatim_chunks.items():
                node_chunks_dict[node_id] = data.get("content", "")

        # Add gist chunks
        gists_store = remem.episodic_embedding_stores.get("gists")
        if gists_store:
            gist_chunks = gists_store.hash_id_to_row
            assert isinstance(gist_chunks, dict), f"gist_chunks should be dict, got {type(gist_chunks)}: {gist_chunks}"
            for node_id, data in gist_chunks.items():
                node_chunks_dict[node_id] = data.get("content", "")

        # Add entity information
        entity_store = remem.episodic_embedding_stores.get("entity")
        if entity_store:
            entity_chunks = entity_store.hash_id_to_row
            assert isinstance(
                entity_chunks, dict
            ), f"entity_chunks should be dict, got {type(entity_chunks)}: {entity_chunks}"
            for node_id, data in entity_chunks.items():
                if isinstance(data.get("content"), dict):
                    entity_info = data["content"]
                    entity_repr = f"Entity: {entity_info.get('entity', 'Unknown')}"
                    if "context" in entity_info:
                        entity_repr += f" (Context: {entity_info['context']})"
                    node_chunks_dict[node_id] = entity_repr
                else:
                    node_chunks_dict[node_id] = str(data.get("content", ""))

        # Add fact information (triples)
        facts_store = remem.episodic_embedding_stores.get("facts")
        if facts_store:
            fact_chunks = facts_store.hash_id_to_row
            assert isinstance(fact_chunks, dict), f"fact_chunks should be dict, got {type(fact_chunks)}: {fact_chunks}"
            for node_id, data in fact_chunks.items():
                fact_content = data.get("content", "")
                fact_metadata = data.get("metadata", {})

                # Check if metadata contains fact information
                if isinstance(fact_metadata, dict) and all(
                    k in fact_metadata for k in ["subject", "predicate", "object"]
                ):
                    fact_info = fact_metadata
                    s, p, o = fact_info["subject"], fact_info["predicate"], fact_info["object"]

                    # Initialize qualifier string
                    qualifier_str = ""

                    # Check if there are qualifiers in metadata
                    qualifiers = fact_metadata.get("qualifiers", {})
                    if qualifiers and qualifiers != "{}":
                        if isinstance(qualifiers, str):
                            qualifiers = eval(qualifiers)

                        if isinstance(qualifiers, dict):
                            for qkey, qvalue in qualifiers.items():
                                if qkey in ["start_time", "end_time", "point_in_time"]:
                                    qualifier_str += f'"{qkey}": "{qvalue}", '
                            qualifier_str = qualifier_str.rstrip(", ")
                            if qualifier_str:
                                qualifier_str = "{" + qualifier_str + "}"

                    repr = {"triple": f"({s}, {p}, {o})"}
                    if qualifier_str:
                        repr["qualifiers"] = qualifier_str

                    node_chunks_dict[node_id] = json.dumps(repr)  # store as a string representation of the fact
                elif fact_content:
                    # Fallback: use the content directly if available
                    node_chunks_dict[node_id] = fact_content

        # Create a new GraphAgent instance for each query to ensure thread safety
        from remem.agent.graph_agent import GraphAgent

        graph_agent = GraphAgent(
            llm_model=remem.qa_llm, node_chunks_dict=node_chunks_dict, remem_instance=remem, logger=logger
        )

        agent_result = graph_agent.retrieve_with_agent(
            query=query,
            beam_size=remem.global_config.qa_top_k,
            return_chunk=return_chunk,
            gold_answer=gold_answer,
            question_metadata=question_metadata,
        )

        # Convert agent results to format compatible with traditional retrieval
        assert isinstance(agent_result, tuple) and len(agent_result) >= 2, "Agent result should be a tuple"
        agent_chunk_ids, agent_chunk_scores = agent_result[:2]

        # Convert chunk hash IDs to indices based on return_chunk type
        if return_chunk == "verbatim":
            entry_keys = remem.entry_keys["verbatim"]
        elif return_chunk == "gists":
            entry_keys = remem.entry_keys["gists"]
        else:
            # Default to gists if return_chunk is None or unspecified
            entry_keys = remem.entry_keys["gists"]

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

    def prepare_retrieval_objects(self):
        """Prepare for fast retrieval by loading keys and embeddings."""
        logger.info("Preparing for fast retrieval...")

        # Load datetime cache for inference
        from remem.utils.datetime_utils import load_datetime_cache

        load_datetime_cache()

        self.remem.query_to_embedding = {"triple": {}, "passage": {}}
        entries = {"verbatim", "facts", "gists"}
        self.remem.entry_keys = {}
        for entry in entries:
            self.remem.entry_keys[entry] = self.remem.episodic_embedding_stores[entry].get_all_ids()

        if "facts_and_sim" in self.remem.global_config.graph_type:
            igraph_name_to_idx = {node["name"]: idx for idx, node in enumerate(self.remem.graph.vs)}
            self.remem.node_name_to_vertex_idx = igraph_name_to_idx

        logger.info("Loading embeddings...")
        self.remem.entry_embeddings = {}
        for entry in self.remem.entry_keys:
            self.remem.entry_embeddings[entry] = np.array(
                list(self.remem.episodic_embedding_stores[entry].get_embeddings(self.remem.entry_keys[entry]))
            )

        if self.remem.global_config.rerank_dspy_file_path is not None:
            from remem.rerank import DSPyFilter

            self.remem.entry_filter = DSPyFilter(self.remem)
        else:
            self.remem.entry_filter = None
        self.remem.ready_to_retrieve = True

    def rank_entries(self, query: str, query_entry_scores: np.ndarray, entry: str, filter=False) -> List[str]:
        """Rank entries by query similarity scores."""
        query_entry_scores = query_entry_scores.flatten()
        sorted_indices = np.argsort(query_entry_scores)[::-1]
        sorted_keys = [self.remem.entry_keys[entry][i] for i in sorted_indices]
        all_rows = self.remem.episodic_embedding_stores[entry].hash_id_to_row
        sorted_items = [all_rows[k] for k in sorted_keys]

        # Apply filtering if enabled and a filter exists
        if filter:
            link_top_k: int = self.remem.global_config.linking_top_k
            candidate_entries: List[Tuple] = []
            for i in range(link_top_k):
                if isinstance(sorted_items[i], Dict) and "content" in sorted_items[i]:
                    candidate_entries.append(tuple(eval(sorted_items[i]["content"])))
                else:
                    raise ValueError(f"Invalid entry format for {entry}.")
            candidate_indices: List[int] = [int(sorted_indices[i]) for i in range(link_top_k)]

            filtered_indices, _, reranker_dict = self.remem.entry_filter(
                query, candidate_entries, candidate_indices, len_after_rerank=link_top_k
            )
            filtered_keys = [self.remem.entry_keys[entry][i] for i in filtered_indices]
            filtered_items = [all_rows[k] for k in filtered_keys]
            return filtered_indices, filtered_items

        return sorted_indices, sorted_items

    def retrieve_each_query_ppr(self, query, return_chunk=None):
        """Traditional PPR-based retrieval."""
        top_k_indices = {}
        top_k_entries = {}
        query_entry_scores = {}

        for entry in self.remem.entry_keys:
            if entry not in self.remem.entry_embeddings:
                raise ValueError(f"Entry {entry} is not indexed. Please index the data first.")
            from remem.utils.embed_utils import get_query_entry_scores

            query_cur_entry_scores = get_query_entry_scores(query, entry, self.remem, return_list=False)
            # Ensure we have np.ndarray for rank_entries method
            if not isinstance(query_cur_entry_scores, np.ndarray):
                query_cur_entry_scores = np.array(query_cur_entry_scores)
            filter = False
            if entry in ["facts"]:
                filter = True
            cur_top_k_indices, cur_top_k_entries = self.rank_entries(
                query, query_cur_entry_scores, entry, filter=filter
            )
            top_k_indices[entry], top_k_entries[entry] = cur_top_k_indices, cur_top_k_entries
            query_entry_scores[entry] = query_cur_entry_scores

        sorted_chunk_ids, sorted_chunk_scores = self.graph_search_with_nodes(
            query, top_k_indices, top_k_entries, query_entry_scores, return_chunk=return_chunk
        )

        graph_seeds = {}
        for entry in top_k_entries:
            graph_seeds[entry] = []
            selected = top_k_entries[entry] if len(top_k_entries[entry]) <= 10 else top_k_entries[entry][:5]
            for item in selected:
                graph_seeds[entry].append(item["content"])

        return sorted_chunk_ids, sorted_chunk_scores, graph_seeds

    def graph_search_with_nodes(
        self,
        query: str,
        top_k_indices: Dict,
        top_k_entries: Dict,
        query_entry_scores: Dict,
        link_top_k: int = 5,
        return_chunk=None,
    ):
        """Perform graph search with node weights."""
        node_weights = np.zeros(len(self.remem.graph.vs["name"]))
        linking_score_map = defaultdict(float)
        entries = ["facts", "verbatim", "gists"]
        weight_ratio = {"verbatim": 0.05, "gists": 0.01, "entity": 1.0}

        for entry in entries:
            if entry == "facts":
                for rank, f in enumerate(top_k_entries[entry]):
                    if rank >= link_top_k:
                        break
                    fact = eval(f["content"])
                    if len(fact) != 3:
                        continue
                    query_fact_score = query_entry_scores[entry][top_k_indices[entry][rank]]
                    subject_key = compute_mdhash_id(fact[0], prefix="entity-")
                    object_key = compute_mdhash_id(fact[2], prefix="entity-")

                    if subject_key in self.remem.node_name_to_vertex_idx:
                        cur_entity_node_score = (
                            weight_ratio["entity"] * query_fact_score / self.remem.entity_node_to_num_chunk[fact[0]]
                        )
                        node_weights[self.remem.node_name_to_vertex_idx[subject_key]] += cur_entity_node_score
                        linking_score_map[fact[0]] += cur_entity_node_score
                    if object_key in self.remem.node_name_to_vertex_idx:
                        cur_entity_node_score = (
                            weight_ratio["entity"] * query_fact_score / self.remem.entity_node_to_num_chunk[fact[2]]
                        )
                        node_weights[self.remem.node_name_to_vertex_idx[object_key]] += cur_entity_node_score
                        linking_score_map[fact[2]] += cur_entity_node_score

                node_weights, linking_score_map = self.remem.get_top_k_weights(
                    link_top_k, node_weights, linking_score_map, "entity"
                )
            else:
                if entry in ["verbatim", "gists"]:
                    cur_link_top_k = None  # use all verbatim as seeds
                else:
                    cur_link_top_k = link_top_k

                for rank, idx in enumerate(top_k_indices[entry][:cur_link_top_k]):
                    node_id = top_k_entries[entry][rank]["hash_id"]
                    query_cur_entry_score = query_entry_scores[entry][idx]
                    if node_id in self.remem.node_name_to_vertex_idx:
                        cur_entry_score = weight_ratio.get(entry, 1.0) * query_cur_entry_score
                        node_weights[self.remem.node_name_to_vertex_idx[node_id]] += cur_entry_score
                        linking_score_map[top_k_entries[entry][rank]["content"]] += cur_entry_score

        linking_score_map = {k: v for k, v in sorted(linking_score_map.items(), key=lambda item: item[1], reverse=True)}
        # Running PPR algorithm based on the entry weights
        ppr_sorted_chunk_ids, ppr_sorted_chunk_scores = self.run_ppr(
            node_weights, self.remem.global_config.damping, return_chunk=return_chunk
        )
        return ppr_sorted_chunk_ids, ppr_sorted_chunk_scores

    def run_ppr(self, reset_prob: np.ndarray, damping: float = 0.5, return_chunk=None) -> Tuple[np.ndarray, np.ndarray]:
        """Run PersonalizedPageRank algorithm."""
        if damping is None:
            damping = 0.5
        reset_prob = np.where(np.isnan(reset_prob) | (reset_prob < 0), 0, reset_prob)
        pagerank_scores = self.remem.graph.personalized_pagerank(
            vertices=range(len(self.remem.node_name_to_vertex_idx)),
            damping=damping,
            directed=False,
            weights="weight",
            reset=reset_prob,
            implementation="prpack",
        )

        if return_chunk == "verbatim":
            verbatim_node_hash = self.remem.episodic_embedding_stores["verbatim"].get_all_ids()
            chunk_scores = np.array(
                [pagerank_scores[self.remem.node_name_to_vertex_idx[hash_id]] for hash_id in verbatim_node_hash]
            )
        elif return_chunk == "gists":
            gist_node_hash = self.remem.episodic_embedding_stores["gists"].get_all_ids()
            chunk_scores = np.array(
                [pagerank_scores[self.remem.node_name_to_vertex_idx[hash_id]] for hash_id in gist_node_hash]
            )
        else:
            raise ValueError(f"Invalid return_chunk value: {return_chunk}.")

        sorted_chunk_ids = np.argsort(chunk_scores)[::-1]
        sorted_chunk_scores = chunk_scores[sorted_chunk_ids.tolist()]

        return sorted_chunk_ids, sorted_chunk_scores

    def get_graph_info(self):
        """Get comprehensive graph information."""
        logger.info("Gathering graph information...")
        graph_info = {}

        # get # of verbatim nodes
        verbatim_node_keys = self.remem.episodic_embedding_stores["verbatim"].get_all_ids()
        graph_info["num_verbatim_nodes"] = len(set(verbatim_node_keys))

        # get # of gist nodes
        gist_node_keys = self.remem.episodic_embedding_stores["gists"].get_all_ids()
        graph_info["num_gist_nodes"] = len(set(gist_node_keys))

        # get # of entity nodes
        entity_node_keys = self.remem.episodic_embedding_stores["entity"].get_all_ids()
        graph_info["num_entity_nodes"] = len(set(entity_node_keys))

        # get # of total nodes
        graph_info["num_total_nodes"] = (
            graph_info["num_verbatim_nodes"] + graph_info["num_gist_nodes"] + graph_info["num_entity_nodes"]
        )

        # Convert to sets for O(1) lookup instead of O(n) list lookup
        verbatim_node_set = set(verbatim_node_keys)
        gist_node_set = set(gist_node_keys)
        entity_node_set = set(entity_node_keys)

        # Single pass through node_to_node_count to compute all edge statistics
        num_facts = 0
        num_verbatim_to_entity_edges = 0
        num_gist_synonymy_edges = 0

        for node_pair in self.remem.node_to_node_count:
            node1, node2 = node_pair[0], node_pair[1]

            # Count entity-to-entity edges (facts)
            if node1 in entity_node_set and node2 in entity_node_set:
                num_facts += 1

            # Count verbatim-to-entity edges
            if node1 in verbatim_node_set and node2 in entity_node_set:
                num_verbatim_to_entity_edges += 1

            # Count gist-to-gist edges (synonymy)
            if node1 in gist_node_set and node2 in gist_node_set:
                num_gist_synonymy_edges += 1

        graph_info["num_facts (bidirectional)"] = num_facts
        graph_info["num_verbatim_to_entity_edges"] = num_verbatim_to_entity_edges
        graph_info["num_gist_synonymy_edges"] = num_gist_synonymy_edges

        # get # of total triples
        graph_info["num_total_edges"] = len(self.remem.node_to_node_count)

        return graph_info

    def merge_gist_extraction_results(
        self, all_openie_info: List[dict], chunk_keys_to_save: List[str], gist_results_dict: Dict[str, EpisodeRawOutput]
    ) -> List[dict]:
        """
        Merge Gist extraction results.
        """
        for chunk_key in chunk_keys_to_save:
            all_openie_info.append(gist_results_dict[chunk_key])
        return all_openie_info

    def _fill_missing_query_solutions(self, query_solutions, all_response_message, agent_answers, queries):
        """Fill missing QuerySolution objects to maintain array correspondence."""
        none_positions = [i for i, qs in enumerate(query_solutions) if qs is None]
        if none_positions:
            logger.warning(
                f"Found None values at positions {none_positions} in query_solutions. This may indicate processing errors."
            )
            # Create empty QuerySolution objects for None positions to maintain array correspondence
            for pos in none_positions:
                if pos < len(queries):
                    query_obj = queries[pos]
                    if isinstance(query_obj, str):
                        query_text = query_obj
                    else:
                        query_text = getattr(query_obj, "question", str(query_obj))

                    query_solutions[pos] = QuerySolution(
                        question=str(query_text),
                        docs=[],
                        question_metadata={},
                        doc_scores=np.array([]),
                        doc_metadata=[],
                        graph_seeds=[],
                        gold_answers=[],
                        gold_docs=[],
                        answer="",
                        qa_rationale="",
                        agent_session_logs=None,
                        metrics={},
                    )
                    all_response_message[pos] = {"agent_answer": "", "reasoning_trace": "", "agent_session_logs": None}
                    agent_answers[pos] = ""
