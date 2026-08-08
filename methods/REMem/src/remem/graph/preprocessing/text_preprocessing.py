import asyncio
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime
from functools import partial
from typing import Any, Dict, List, Optional

from remem.graph.preprocessing.base import BasePreprocessor, _get_text_encoder
from remem.utils.config_utils import BaseConfig
from remem.utils.logging_utils import get_logger

logger = get_logger(__name__)


def encode_string(content: str, encoder: Any = None) -> List[int]:
    tokens = encoder.encode(content)
    return tokens


def decode_tokens(tokens: List[int], encoder: Any = None) -> str:
    content = encoder.decode(tokens)
    return content


def chunk_by_token_count(
    content: str, overlap_token_size=128, max_token_size=1024, encoder: Any = None
) -> List[Dict[str, Any]]:
    """
    Splits the input content into chunks based on the specified token size limits.

    Args:
        content (str): The input text content to be chunked.
        overlap_token_size (int, optional): The number of tokens to overlap between consecutive chunks. Defaults to 128.
        max_token_size (int, optional): The maximum number of tokens per chunk. Defaults to 1024.
        encoder (Any, optional): The encoder to use for encoding and decoding the content. Defaults to None.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries, each representing a chunk with its metadata.
    """
    # Encode the content into tokens
    tokens = encode_string(content=content, encoder=encoder)

    # If max_token_size is None, return the entire content as a single chunk
    if max_token_size is None:
        return [{"num_tokens": len(tokens), "content": content, "chunk_order": 0}]

    results = []
    # Iterate over the tokens to create chunks
    for index, start in enumerate(range(0, len(tokens), max_token_size - overlap_token_size)):
        # Decode the tokens for the current chunk
        chunk_content = decode_tokens(tokens[start : start + max_token_size], encoder=encoder)
        # Append the chunk metadata and content to the results
        results.append(
            {
                "num_tokens": min(max_token_size, len(tokens) - start),
                "content": chunk_content,
                "chunk_order": index,
            }
        )
    return results


def chunk_by_word_count(
    content: str, max_words_per_chunk: int = 256, overlap_words: int = 50, keep_first_line: bool = True
) -> List[Dict[str, Any]]:
    """
    Splits the input content into chunks based on the specified word count and sentence boundaries.

    Args:
        content (str): The input text content to be chunked.
        max_words_per_chunk (int, optional): The maximum number of words per chunk. Defaults to 300.
        overlap_words (int, optional): The number of words to overlap between consecutive chunks. Defaults to 50.
        keep_first_line (bool, optional): Whether to keep the first line of the content in each chunk. Defaults to False.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries, each representing a chunk with its metadata.
    """
    # Tokenize the content into sentences using nltk's sentence tokenizer
    import nltk

    sentences = nltk.sent_tokenize(content)

    chunks = []
    current_chunk = []
    current_word_count = 0
    first_line = content.split("\n")[0]  # Get the first line of the content

    for sentence in sentences:
        words_in_sentence = sentence.split()
        word_count_in_sentence = len(words_in_sentence)

        # Check if adding this sentence exceeds the max word count
        if current_word_count + word_count_in_sentence > max_words_per_chunk:
            # If the chunk is too large, finalize the current chunk and start a new one
            chunks.append(
                {"num_words": current_word_count, "content": " ".join(current_chunk), "chunk_order": len(chunks)}
            )
            # Start a new chunk with the current sentence
            current_chunk = words_in_sentence
            current_word_count = word_count_in_sentence
        else:
            # Otherwise, just add the sentence to the current chunk
            current_chunk.extend(words_in_sentence)
            current_word_count += word_count_in_sentence

    # Add the last chunk if it has content
    if current_chunk:
        chunks.append({"num_words": current_word_count, "content": " ".join(current_chunk), "chunk_order": len(chunks)})

    res = chunks

    # Apply overlap
    if overlap_words > 0:
        overlapped_chunks = []
        for i in range(len(chunks)):
            chunk = chunks[i]
            if i > 0:
                # Add overlap between consecutive chunks
                overlap_content = " ".join(chunks[i - 1]["content"].split()[-overlap_words:])
                chunk["content"] = overlap_content + " " + chunk["content"]
            overlapped_chunks.append(chunk)
        res = overlapped_chunks

    if keep_first_line:
        # Keep the first line of the content in each chunk
        for chunk in res:
            chunk["content"] = first_line + "\n" + chunk["content"]
            chunk["num_words"] += len(first_line.split())
    return res


def chunk_by_message_and_token_count(content: str, encoder: Any = None, max_token=256) -> List[Dict[str, Any]]:
    """
    Split message content by message and the specified token size limits.
    Args:
        content: a json string, which can be loaded as {"date": str, "messages": List[Dict[str, str]]},
                 where each message is {"role": str, "content": str}
        max_token: maximum allowed token count per message or message chunk
        encoder: a tokenizer object that has encoding and decoding methods
    Returns:
        A list of messages (dict), where messages with content exceeding max_token tokens are split into smaller chunks.
    """
    if encoder is None:
        import tiktoken

        encoder = tiktoken.encoding_for_model("gpt-4o")
    # Load the session from JSON string
    session: Dict = json.loads(content)
    session_date = session.get("date", None)
    chunked_messages: List[Dict[str, Any]] = []

    # Iterate over each message in the session
    for message in session.get("messages", []):
        role = message.get("role", "user")
        text = message.get("content", "")
        message_date = message.get("date", session_date)
        # Encode the message content to get token list
        tokens = encoder.encode(text, allowed_special={"<|endoftext|>"})

        # If token count is within limit, add the message as is
        if len(tokens) <= max_token:
            chunked_messages.append({"role": role, "content": text, "date": message_date})
        else:
            # Split tokens into chunks of max_token tokens each
            for i in range(0, len(tokens), max_token):
                chunk_tokens = tokens[i : i + max_token]
                # Decode token chunk back to string
                chunk_text = encoder.decode(chunk_tokens)
                chunked_messages.append({"role": role, "content": chunk_text, "date": message_date})

    return chunked_messages


def chunk_by_session(session: List, **kwargs) -> List[Dict[str, Any]]:
    """
    Split a session into chunks based on the date of the messages.
    Args:
        session: a list of messages, where each message is a dict with keys "role", "content", and optionally "date".
                 Example: [{"role": "user", "content": "Hello", "date": "2023-10-01"}, {"role": "assistant", "content": "Hi"}]
        kwargs: additional arguments (not used in this function, but can be extended in the future)
    Returns:
        A list of dictionaries, where each dictionary contains the concatenated content of messages with the same date.
        If all messages have the same date, the content is concatenated into a single string
        with the date at the top. If messages have different dates, each message is prefixed with its date.
    """
    date_set = set([item.get("date", None) for item in session])
    date_set.discard(None)

    if len(date_set) == 1:
        date = date_set.pop()
        lines = [f"{msg['role']}: {msg['content']}" for msg in session]
        return [{"content": f"{date}\n" + "\n".join(lines), "date": date}]
    else:
        lines = [f"{msg['date']}\t{msg['role']}: {msg['content']}" for msg in session]
        return [{"content": "\n".join(lines)}]


def remem_text_processing(text):
    if not isinstance(text, str):
        text = str(text)
    return re.sub("[^A-Za-z0-9 ]", " ", text.lower()).strip()


class TextPreprocessor(BasePreprocessor):
    def __init__(
        self,
        working_dir: Optional[str] = None,
        global_config: Optional[BaseConfig] = None,
        text_preprocessing_func: Any = None,
    ) -> None:
        if global_config is None:
            logger.debug("global config is not given. Using the default ExperimentConfig instance.")
            self.global_config = BaseConfig()
        else:
            self.global_config = global_config
        logger.debug(f"Loading {self.__class__.__name__} with global_config: {asdict(self.global_config)}")

        if working_dir is None:
            logger.debug("working_dir is not given. Using the default working directory.")
            self.working_dir = os.path.join(
                self.global_config.save_dir, f"{self.__class__.__name__}_{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}"
            )
        else:
            self.working_dir = working_dir

        if not os.path.exists(self.working_dir):
            logger.info(f"Creating working directory: {self.working_dir}")
            os.makedirs(self.working_dir, exist_ok=True)

        self.encoder = None
        if self.global_config.preprocess_chunk_func == "by_word":
            chunking_func = chunk_by_word_count
            self.chunking_func = partial(chunking_func)
        elif self.global_config.preprocess_chunk_func == "by_token":
            self.encoder = _get_text_encoder(encoder_name=self.global_config.preprocess_encoder_name)
            chunking_func = chunk_by_token_count
            self.chunking_func = partial(
                chunking_func,
                overlap_token_size=self.global_config.preprocess_chunk_overlap_token_size,
                max_token_size=self.global_config.preprocess_chunk_max_token_size,
                encoder=self.encoder,
            )
        elif self.global_config.preprocess_chunk_func == "by_message":
            self.encoder = _get_text_encoder(encoder_name=self.global_config.preprocess_encoder_name)
            chunking_func = chunk_by_message_and_token_count
            self.chunking_func = partial(chunking_func)
        elif self.global_config.preprocess_chunk_func == "by_session":
            chunking_func = chunk_by_session
            self.chunking_func = partial(chunking_func)
        elif self.global_config.preprocess_chunk_func == "none":
            chunking_func = lambda content: [{"content": content, "metadata": None}]
            self.chunking_func = partial(chunking_func)
        else:
            raise ValueError(f"Invalid chunking function: {self.global_config.preprocess_chunk_func}")

        if text_preprocessing_func is None:
            logger.debug(
                f"Text preprocessing function is not specified for {self.__class__.__name__}. Using the default ReMem text preprocessing method."
            )
            self.text_preprocessing_func = remem_text_processing
        else:
            self.text_preprocessing_func = text_preprocessing_func

    async def apreprocess_doc(self, input: str) -> List[Dict[str, Any]]:
        """
        Preprocess a single document by chunking it and applying text processing to each chunk.

        Args:
            input (str): The raw input document as a string.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries representing the processed chunks. Each dictionary contains chunk metadata and the processed content.
        """
        # Chunking the doc
        doc_chunks: List[Dict[str, Any]] = self.chunking_func(content=input)

        # # Content Preprocessing
        # for doc_chunk_idx, doc_chunk in enumerate(doc_chunks):
        #     doc_chunk["content"] = self.text_preprocessing_func(text=doc_chunk["content"])

        return doc_chunks

    async def abatch_preprocess_doc(self, input: List[str]) -> List[List[Dict[str, Any]]]:
        """
        Preprocess a batch of documents by applying the `apreprocess_doc` function to each.

        Args:
            input (List[str]): A list of raw input documents as strings.

        Returns:
            List[List[Dict[str, Any]]]: A list of processed documents, where each document is represented as a list of processed chunks.
        """
        results: List[List[Dict[str, Any]]] = await asyncio.gather(
            *[self.apreprocess_doc(input=doc) for doc in input], return_exceptions=False
        )
        return results

    def preprocess_doc(self, input: str) -> List[Dict[str, Any]]:
        """
        Wrap a single input string with a list and call `batch_doc_preprocess`.

        Args:
            input (str): A single document string to preprocess.

        Returns:
            List[Dict[str, Any]]: The processed result for the single input string.
        """
        # Wrap the single input into a list and call batch_doc_preprocess
        results = self.batch_preprocess_doc([input])

        # Extract and return the first result since it's a single input
        return results[0]

    def batch_preprocess_doc(self, input: List[str], multithread=False) -> List[List[Dict[str, Any]]]:
        results = []

        if multithread is False:
            if self.global_config.preprocess_chunk_func == "by_round":
                doc_chunks = self.chunking_func(messages=input)
                return doc_chunks
            else:
                for doc_idx, doc in enumerate(input):
                    # Chunking the doc
                    doc_chunks: Dict[str, Any] = self.chunking_func(doc)
                    results.append(doc_chunks)
        else:  # multithread is True
            with ThreadPoolExecutor() as executor:
                results = list(executor.map(self.chunking_func, input))

        return results

    def apply_text_preprocessing(self, input: str) -> str:
        """
        Apply the text preprocessing techniques for any kind of input string. For ReMem, we will apply such text preprocessing on all ner and triple extraction outputs.


        Returns:
            str: the preprocessed string.
        """
        return self.text_preprocessing_func(text=input)

    def batch_apply_text_preprocessing(self, input: List[str]) -> List[str]:
        """
        Apply `self.apply_text_preprocessing` to a list of input strings respectively.

        Returns:
            List[str]: the list of preprocessed string corresponding to each input string.
        """
        return [self.apply_text_preprocessing(input=item) for item in input]
