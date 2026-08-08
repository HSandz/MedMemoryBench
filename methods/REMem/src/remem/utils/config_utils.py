import os
from dataclasses import dataclass, field
from typing import Literal, Optional, Union

from remem.utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class BaseConfig:
    """One and only configuration."""

    # LLM specific attributes
    llm_name: str = field(default="gpt-4o-mini", metadata={"help": "Class name indicating which LLM model to use."})
    extract_llm_label: str = field(
        default=None, metadata={"help": "Label of the LLM model to use for information extraction."}
    )
    qa_llm_label: str = field(default=None, metadata={"help": "Label of the LLM model to use for QA."})
    llm_base_url: str = field(
        default=None, metadata={"help": "Base URL for the LLM model, if none, means using OPENAI service."}
    )
    max_new_tokens: Union[None, int] = field(
        default=2048, metadata={"help": "Max new tokens to generate in each inference."}
    )
    num_gen_choices: int = field(
        default=1, metadata={"help": "How many chat completion choices to generate for each input message."}
    )
    seed: Union[None, int] = field(default=None, metadata={"help": "Random seed."})
    temperature: float = field(default=0, metadata={"help": "Temperature for sampling in each inference."})
    extract_format: Union[None, Literal["json_object", "json_schema"]] = field(
        default=None, metadata={"help": "Specifying the response_format of extraction part."}
    )
    use_azure: bool = field(default=False, metadata={"help": "Whether to use Azure service or not."})
    max_num_seqs: int = field(default=256, metadata={"help": "Max number of sequences to generate for vLLM."})
    max_model_len: int = field(default=4096, metadata={"help": "Max number of tokens for the model context."})

    ## LLM specific attributes -> Async hyperparameters
    max_retries: int = field(
        default=10, metadata={"help": "Max number of retry attempts for an asynchronous API calling."}
    )
    # Storage specific attributes
    force_openie_from_scratch: bool = field(
        default=False,
        metadata={"help": "If set to True, will ignore all existing openie files and rebuild them from scratch."},
    )

    # Storage specific attributes
    force_index_from_scratch: bool = field(
        default=False,
        metadata={
            "help": "If set to True, will ignore all existing storage files and graph data and will rebuild from scratch."
        },
    )
    rerank_dspy_file_path: str = field(default=None, metadata={"help": "Path to the rerank dspy file."})
    passage_node_weight: float = field(
        default=0.05, metadata={"help": "Multiplicative factor that modified the passage node weights in PPR."}
    )
    save_openie: bool = field(default=True, metadata={"help": "If set to True, will save the OpenIE model to disk."})

    # Preprocessing specific attributes
    text_preprocessor_class_name: str = field(
        default="TextPreprocessor", metadata={"help": "Name of the text-based preprocessor to use in preprocessing."}
    )
    preprocess_encoder_name: str = field(
        default="gpt-4o",
        metadata={
            "help": "Name of the encoder to use in preprocessing (currently implemented specifically for doc chunking)."
        },
    )
    preprocess_chunk_overlap_token_size: int = field(
        default=128, metadata={"help": "Number of overlap tokens between neighbouring chunks."}
    )
    preprocess_chunk_max_token_size: int = field(
        default=None,
        metadata={
            "help": "Max number of tokens each chunk can contain. If set to None, the whole doc will treated as a single chunk."
        },
    )
    preprocess_chunk_func: str = field(default="by_token")

    # Information extraction specific attributes
    extract_method: Literal["openie", "episodic", "episodic_gist", "temporal"] = field(default="openie")
    llm_infer_mode: Literal["offline", "online"] = field(
        default="online", metadata={"help": "Mode of the OpenIE model to use."}
    )
    skip_graph: bool = field(
        default=False,
        metadata={
            "help": "Whether to skip graph construction or not. Set it to be true when running vllm offline indexing for the first time."
        },
    )
    vllm_tensor_parallel_size: int = field(
        default=2, metadata={"help": "Tensor parallel size for VLLM model offline mode."}
    )

    # Embedding specific attributes
    embedding_model_name: str = field(
        default="nvidia/NV-Embed-v2", metadata={"help": "Class name indicating which embedding model to use."}
    )
    embedding_batch_size: int = field(default=16, metadata={"help": "Batch size of calling embedding model."})
    embedding_return_as_normalized: bool = field(
        default=True, metadata={"help": "Whether to normalize encoded embeddings not."}
    )
    embedding_max_seq_len: int = field(default=2048, metadata={"help": "Max sequence length for the embedding model."})

    # Graph construction specific attributes
    concatenate_gists_per_chunk: bool = field(
        default=False,
        metadata={
            "help": "Whether to concatenate all gists in a chunk into a single gist node. "
            "If False, each gist in a chunk's gist list becomes a separate node. "
            "If True, all gists in a chunk are joined into one node for better context concentration."
        },
    )
    split_verbatim_per_chunk: bool = field(
        default=True,
        metadata={
            "help": "Whether to split long verbatim content into separate nodes for conversation-like data. "
            "If False, each verbatim chunk becomes a single node. "
            "If True, conversation sessions with multiple messages are split into individual message nodes, "
            "while single paragraph content remains as one node."
        },
    )
    synonymy_edge_topk: int = field(default=2047, metadata={"help": "k for knn retrieval in buiding synonymy edges."})
    synonymy_edge_query_batch_size: int = field(
        default=1000, metadata={"help": "Batch size for query embeddings for knn retrieval in buiding synonymy edges."}
    )
    synonymy_edge_key_batch_size: int = field(
        default=10000, metadata={"help": "Batch size for key embeddings for knn retrieval in buiding synonymy edges."}
    )
    synonymy_edge_sim_threshold: float = field(
        default=0.8, metadata={"help": "Similarity threshold to include candidate synonymy nodes."}
    )
    is_directed_graph: bool = field(default=False, metadata={"help": "Whether the graph is directed or not."})

    # Retrieval specific attributes
    linking_top_k: int = field(default=5, metadata={"help": "The number of linked nodes at each retrieval step"})
    retrieval_top_k: int = field(default=200, metadata={"help": "Retrieving k documents at each step"})
    damping: float = field(default=0.5, metadata={"help": "Damping factor for ppr algorithm."})

    # QA specific attributes
    qa_top_k: int = field(default=5, metadata={"help": "Feeding top k documents to the QA model for reading."})
    qa_passage_prefix: str = field(default="Wikipedia Title: ")
    qa_prompt_template: str = field(default=None, metadata={"help": "Template name for QA prompts"})
    qa_reader: Literal["remem", "tiser"] = field(default="remem", metadata={"help": "Which QA reader to use."})

    # Agent specific attributes (for episodic and temporal methods)
    agent_fixed_tools: bool = field(
        default=False,
        metadata={
            "help": "If True, agent uses only semantic_retrieve + output_answer. If False, uses full tool selection."
        },
    )
    agent_max_steps: int = field(
        default=5,
        metadata={
            "help": "Maximum number of reasoning steps for the agent. For fixed_tools: 1=semantic_retrieve only, 2=semantic_retrieve+output_answer."
        },
    )
    agent_fixed_retrieval_tool: str = field(
        default="semantic_retrieve",
        metadata={"help": "Which retrieval tool to use in fixed_tools mode: 'semantic_retrieve' or 'lexical_retrieve'"},
    )

    # Evaluation specific attributes
    do_eval_retrieval: bool = field(
        default=True, metadata={"help": "Whether perform evaluation on the retrieval results or not."}
    )
    do_eval_qa: bool = field(default=True, metadata={"help": "Whether perform evaluation on the QA results or not."})

    # Save dir (highest level directory)
    save_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Directory to save all related information. If it's given, will overwrite all default save_dir setups. If it's not given, then if we're not running specific datasets, default to `outputs`, otherwise, default to a dataset-customized output dir."
        },
    )

    # Dataset running specific attributes
    ## Dataset running specific attributes -> General
    dataset: Optional[str] = field(
        default=None,
        metadata={
            "help": "Dataset to use. If specified, it means we will run specific datasets. If not specified, it means we're running freely."
        },
    )
    ## Dataset running specific attributes -> Graph
    graph_type: Literal[
        "dpr_only",
        "facts_and_sim",
        "facts_and_sim_passage_node_unidirectional",
    ] = field(
        default="facts_and_sim_passage_node_unidirectional",
        metadata={"help": "Type of graph to use in the experiment."},
    )
    corpus_len: Optional[int] = field(default=None, metadata={"help": "Length of the corpus to use."})

    def __post_init__(self):
        """
        Create the highest level of save_dir.
        Returns:

        """

        if self.dataset is None:
            self.save_dir = "outputs"  # running freely
        else:
            # Map dataset to organized subdirectories
            prefix = {"locomo", "longmemeval", "tot_semantic", "realtalk"}
            # If the dataset starts with one of the prefixes (many graphs), save it in a specific subdirectory
            for p in prefix:
                if self.dataset.startswith(p):
                    self.save_dir = os.path.join("outputs", p, self.dataset)
                    break
            else:
                # For any other datasets, keep them at top level
                self.save_dir = os.path.join("outputs", self.dataset)

        # Add suffix for different modes
        if self.graph_type == "dpr_only":
            self.save_dir += "_dpr_only"

        if self.extract_llm_label is None:
            self.extract_llm_label = self.llm_name.replace("/", "_")
        if self.qa_llm_label is None:
            self.qa_llm_label = self.llm_name.replace("/", "_")

        os.makedirs(self.save_dir, exist_ok=True)
        logger.debug(f"Initializing the highest level of save_dir to be {self.save_dir}")
