import re
from argparse import ArgumentTypeError
from dataclasses import asdict, dataclass, field
from hashlib import md5
from typing import Any, Dict, List, Literal, Optional

import numpy as np

from remem.utils.llm_utils import filter_invalid_triples
from remem.utils.logging_utils import get_logger
from remem.utils.typing_utils import Triple

logger = get_logger(__name__)


@dataclass
class NerRawOutput:
    chunk_id: str
    response: str
    unique_entities: List[str]
    metadata: Dict[str, Any]


@dataclass
class TripleRawOutput:
    chunk_id: str
    response: str
    triples: List[List[str]]
    metadata: Dict[str, Any]


@dataclass
class ParaphraseRawOutput:
    chunk_id: str
    response: str
    paraphrases: List[str]
    metadata: Dict[str, Any]


@dataclass
class EpisodeRawOutput:
    chunk_id: str
    response: str
    metadata: Dict[str, Any]
    verbatim: str
    facts: Optional[List] = field(default_factory=list)
    gists: Optional[List] = None
    time: Optional[str] = None
    space: Optional[List] = field(default_factory=list)
    causality: Optional[List] = field(default_factory=list)
    intent: Optional[List] = field(default_factory=list)
    emotion: Optional[List] = field(default_factory=list)

    def __iter__(self):
        return iter(asdict(self).items())

    def __getitem__(self, key):
        return asdict(self)[key]


@dataclass
class LinkingOutput:
    score: np.ndarray
    type: Literal["node", "dpr"]


@dataclass
class QuerySolution:
    question: str
    docs: List[str]
    question_metadata: Dict[str, Any] = None
    doc_scores: np.ndarray = None
    doc_metadata: List[Dict[str, Any]] = None
    answer: str = None
    gold_answers: List[str] = None
    gold_docs: Optional[List[str]] = None
    qa_rationale: Optional[str] = None
    metrics: Dict = None
    graph_seeds: List = None
    agent_session_logs: Optional[Dict[str, Any]] = None

    def to_dict(self):
        # Safely handle doc_scores conversion
        doc_scores_list = None
        if self.doc_scores is not None:
            try:
                if hasattr(self.doc_scores, "tolist"):
                    doc_scores_list = [round(v, 4) for v in self.doc_scores.tolist()[:15]]
                elif isinstance(self.doc_scores, (list, tuple)):
                    doc_scores_list = [round(v, 4) for v in self.doc_scores[:15]]
                else:
                    # Convert single value or other types to list
                    doc_scores_list = [round(float(self.doc_scores), 4)]
            except (AttributeError, TypeError, ValueError) as e:
                logger.warning(f"Failed to convert doc_scores to serializable format: {e}")
                doc_scores_list = None

        # Safely handle agent_session_logs serialization using the safe_json_serializable function
        agent_logs_safe = None
        if self.agent_session_logs is not None:
            try:
                agent_logs_safe = safe_json_serializable(self.agent_session_logs)
            except Exception as e:
                logger.warning(f"Failed to safely serialize agent session logs: {e}")
                agent_logs_safe = {
                    "serialization_error": str(e),
                    "available_keys": (
                        list(self.agent_session_logs.keys())
                        if isinstance(self.agent_session_logs, dict)
                        else "Not a dict"
                    ),
                }

        res = {
            "question": self.question,
            "predicted_answer": self.answer,
            "gold_answers": self.gold_answers,
            "qa_rationale": self.qa_rationale,
            "docs": self.docs[:15],
            "doc_scores": doc_scores_list,
            "gold_docs": self.gold_docs,
            "sample_metrics": self.metrics,
        }
        if self.graph_seeds is not None:
            res["graph_seeds"] = self.graph_seeds
        if self.question_metadata is not None:
            res["question_metadata"] = self.question_metadata
        if self.doc_metadata is not None:
            res["doc_metadata"] = self.doc_metadata[:15]
            del res["docs"]
        if agent_logs_safe is not None:
            res["agent_session_logs"] = agent_logs_safe
        return res


def text_processing(text):
    if isinstance(text, list):
        return [text_processing(t) for t in text]
    if not isinstance(text, str):
        text = str(text)
    return re.sub("[^A-Za-z0-9 ]", " ", text.lower()).strip()


def reformat_openie_results(corpus_openie_results) -> (Dict[str, NerRawOutput], Dict[str, TripleRawOutput]):
    """
    Reformat the OpenIE results into NerRawOutput and TripleRawOutput dictionaries.
    """
    ner_output_dict = {}
    for chunk_item in corpus_openie_results:
        unique_entities = chunk_item["extracted_entities"]
        if len(unique_entities) == 0:
            continue
        if not isinstance(unique_entities[0], str):
            logger.warn(f"Extracted entities are not strings: {unique_entities}")
        else:
            ner_output_dict[chunk_item["idx"]] = NerRawOutput(
                chunk_id=chunk_item["idx"], response=None, metadata={}, unique_entities=list(np.unique(unique_entities))
            )

    triple_output_dict = {}
    for chunk_item in corpus_openie_results:
        triple_output_dict[chunk_item["idx"]] = TripleRawOutput(
            chunk_id=chunk_item["idx"],
            response=None,
            metadata={},
            triples=filter_invalid_triples(triples=chunk_item["extracted_triples"]),
        )

    paraphrase_output_dict = {}
    for chunk_item in corpus_openie_results:
        if "paraphrases" in chunk_item:
            paraphrase_output_dict[chunk_item["idx"]] = ParaphraseRawOutput(
                chunk_id=chunk_item["idx"], response=None, metadata={}, paraphrases=chunk_item["paraphrases"]
            )
    if len(paraphrase_output_dict):
        return ner_output_dict, triple_output_dict, paraphrase_output_dict

    return ner_output_dict, triple_output_dict


def extract_phrase_nodes(chunk_triples: List[List[Triple]]) -> (List[str], List[List[str]]):
    from itertools import chain

    chunk_triple_entities = []
    for triples in chunk_triples:
        flat_entities = [e for t in triples if len(t) == 3 for e in (t[0], t[2]) if e is not None]
        seen = set()
        unique_entities = [x for x in flat_entities if not (x in seen or seen.add(x))]
        chunk_triple_entities.append(unique_entities)
    graph_nodes = list(dict.fromkeys(chain.from_iterable(chunk_triple_entities)))
    return graph_nodes, chunk_triple_entities


def flatten_facts(chunk_triples: List[Triple]) -> List[Triple]:
    graph_triples = []  # a list of unique relation triple (in tuple) from all chunks
    for triples in chunk_triples:
        graph_triples.extend([tuple(t) for t in triples])
    graph_triples = list(set(graph_triples))
    return graph_triples


def min_max_normalize(x, axis=None):
    """
    Min-max normalization

    Args:
        x: Input array/matrix
        axis: None (global), 0 (column-wise), 1 (row-wise)

    Raises:
        ValueError: When all values are identical (no variation to normalize)
    """
    x = np.asarray(x)

    if axis is None:
        x_min = np.min(x)
        x_max = np.max(x)
        diff = x_max - x_min
        if diff == 0:
            logger.warning(f"No difference between min and max: {x}")
            return x
        return (x - x_min) / diff
    else:
        x_min = np.min(x, axis=axis, keepdims=True)
        x_max = np.max(x, axis=axis, keepdims=True)
        diff = x_max - x_min

        # Check if any axis has zero difference
        if np.any(diff == 0):
            zero_diff_axes = np.where(diff == 0)
            logger.warning(f"No difference between min and max: {x}")
            # Return the original values for those axes
            x = np.where(diff == 0, x, (x - x_min) / diff)
            return x

        return (x - x_min) / diff


def compute_mdhash_id(content: str, prefix: str = "") -> str:
    """
    Compute the MD5 hash of the given content string and optionally prepend a prefix.

    Args:
        content (str): The input string to be hashed.
        prefix (str, optional): A string to prepend to the resulting hash. Defaults to an empty string.

    Returns:
        str: A string consisting of the prefix followed by the hexadecimal representation of the MD5 hash.
    """
    assert content, "Content cannot be None"
    assert isinstance(content, str), f"Content must be a string, got {type(content)}"
    return prefix + md5(content.encode()).hexdigest()


def all_values_of_same_length(data: dict) -> bool:
    """
    Return True if all values in 'data' have the same length or data is an empty dict,
    otherwise return False.
    """
    # Get an iterator over the dictionary's values
    value_iter = iter(data.values())

    # Get the length of the first sequence (handle empty dict case safely)
    try:
        first_length = len(next(value_iter))
    except StopIteration:
        # If the dictionary is empty, treat it as all having "the same length"
        return True

    # Check that every remaining sequence has this same length
    return all(len(seq) == first_length for seq in value_iter)


def string_to_bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise ArgumentTypeError(
            f"Truthy value expected: got {v} but expected one of yes/no, true/false, t/f, y/n, 1/0 (case insensitive)."
        )


def describe_array(arr, *, quantiles=(0.25, 0.5, 0.75), ddof=0):
    """
    Compute basic descriptive statistics for a 1‑D array.

    Parameters
    ----------
    arr : array‑like
        Input data (anything that can be coerced to a NumPy array).
    quantiles : tuple of float, optional
        Probabilities between 0 and 1 for which to compute quantiles.
        Default is (0.25, 0.5, 0.75).
    ddof : int, optional
        "Delta Degrees of Freedom" for variance / standard deviation.
        * ddof=0 → population variance (default)
        * ddof=1 → sample variance

    Returns
    -------
    dict
        A dictionary containing:
        mean, variance, std, min, max, and a nested dict "quantiles".
    """
    data = np.asarray(arr)  # ensure NumPy array
    stats = {
        "len": len(data),
        "non_zero": np.count_nonzero(data),
        "mean": np.mean(data),
        "variance": np.var(data, ddof=ddof),
        "std": np.std(data, ddof=ddof),
        "min": np.min(data),
        "max": np.max(data),
        "quantiles": {q: v for q, v in zip(quantiles, np.quantile(data, quantiles))},
    }
    return stats


def skip_phrase_node(phrase: str) -> bool:
    if phrase in {"user", "assistant"}:
        return True
    phrase_split = phrase.split(": ")
    if phrase_split in ["user", "assistant"]:
        if len(phrase_split) > 1 and phrase_split[1].count(" ") == 0:
            return True
    return False


# Import parse_flexible_datetime from datetime_utils for backward compatibility
from datetime import datetime


@dataclass
class TemporalRawOutput:
    chunk_id: str
    response: str
    metadata: Dict[str, Any]
    verbatim: str
    facts: Optional[List] = field(default_factory=list)  # List of (subject, predicate, object, qualifiers) tuples

    def __iter__(self):
        return iter(asdict(self).items())

    def __getitem__(self, key):
        return asdict(self)[key]


def safe_json_serializable(obj, max_depth=5, current_depth=0):
    """
    Recursively convert an object to a JSON-serializable format.

    Args:
        obj: The object to convert
        max_depth: Maximum recursion depth to prevent infinite loops
        current_depth: Current recursion depth

    Returns:
        A JSON-serializable version of the object
    """
    if current_depth > max_depth:
        return f"<max_depth_exceeded_{type(obj).__name__}>"

    try:
        # Try direct JSON serialization first
        import json

        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        pass

    # Handle specific types
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    elif isinstance(obj, (list, tuple)):
        return [safe_json_serializable(item, max_depth, current_depth + 1) for item in obj]
    elif isinstance(obj, dict):
        return {str(k): safe_json_serializable(v, max_depth, current_depth + 1) for k, v in obj.items()}
    elif hasattr(obj, "__dict__"):
        # For objects with __dict__, try to serialize their attributes
        try:
            return {
                "type": type(obj).__name__,
                "attributes": safe_json_serializable(obj.__dict__, max_depth, current_depth + 1),
            }
        except:
            return f"<{type(obj).__name__}_object>"
    elif hasattr(obj, "to_dict"):
        # Try to use object's to_dict method
        try:
            return safe_json_serializable(obj.to_dict(), max_depth, current_depth + 1)
        except:
            return f"<{type(obj).__name__}_with_to_dict>"
    else:
        # Fallback: convert to string representation
        try:
            return str(obj)
        except:
            return f"<non_serializable_{type(obj).__name__}>"


def safe_dump_json(data, file_path, indent=4, fallback_on_error=True):
    """
    Safely dump data to JSON file with error handling.

    Args:
        data: Data to serialize
        file_path: Path to save the JSON file
        indent: JSON indentation (default: 4)
        fallback_on_error: Whether to save a simplified version on error (default: True)

    Returns:
        bool: True if successful, False if fallback was used
    """
    import json
    import os

    # Ensure directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    try:
        with open(file_path, "w") as f:
            json.dump(data, f, indent=indent)
        return True
    except (TypeError, ValueError) as e:
        logger.warning(f"JSON serialization failed for {file_path}: {e}")

        if not fallback_on_error:
            raise

        # Try to create a safe version of the data
        try:
            safe_data = safe_json_serializable(data)
            with open(file_path, "w") as f:
                json.dump(safe_data, f, indent=indent)
            logger.info(f"Saved safe version of data to {file_path}")
            return False
        except Exception as fallback_error:
            logger.error(f"Failed to save even safe version: {fallback_error}")
            # Save minimal error info
            error_data = {
                "serialization_error": str(e),
                "fallback_error": str(fallback_error),
                "data_type": str(type(data)),
                "timestamp": str(datetime.now()) if "datetime" in globals() else "unknown",
            }
            with open(file_path, "w") as f:
                json.dump(error_data, f, indent=indent)
            return False


def safe_serialize_query_solutions(query_solutions):
    """
    Safely convert a list of QuerySolution objects to serializable dictionaries.

    Args:
        query_solutions: List of QuerySolution objects

    Returns:
        list: List of serializable dictionaries
    """
    samples_dict = []

    for i, q in enumerate(query_solutions):
        try:
            samples_dict.append(q.to_dict())
        except Exception as individual_error:
            logger.warning(f"Error converting query {i} to dict: {individual_error}")
            # Create a minimal safe representation
            safe_query = {
                "error": f"Failed to serialize query {i}: {str(individual_error)}",
                "question": getattr(q, "question", "Unknown"),
                "predicted_answer": getattr(q, "answer", "Unknown"),
                "serialization_failed": True,
            }
            # Try to add some basic fields safely
            for field in ["gold_answers", "gold_docs", "qa_rationale"]:
                try:
                    value = getattr(q, field, None)
                    if value is not None:
                        safe_query[field] = str(value)[:500]  # Truncate if too long
                except:
                    pass
            samples_dict.append(safe_query)

    return samples_dict
