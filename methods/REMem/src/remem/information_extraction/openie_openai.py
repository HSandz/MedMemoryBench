import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, TypedDict

from tqdm import tqdm

from remem.llm.openai_gpt import CacheOpenAI
from remem.prompts import PromptTemplateManager
from remem.utils.llm_utils import (
    filter_invalid_triples,
    fix_broken_generated_json,
)
from remem.utils.logging_utils import get_logger
from remem.utils.misc_utils import NerRawOutput, TripleRawOutput

logger = get_logger(__name__)


class ChunkInfo(TypedDict):
    num_tokens: int
    content: str
    chunk_order: List[Tuple]
    full_doc_ids: List[str]


@dataclass
class LLMInput:
    chunk_id: str
    input_message: List[Dict]


def _extract_value_from_json_response(real_response, key="named_entities"):
    pattern = r'\{[^{}]*"' + key + '"\s*:\s*\[[^\]]*\][^{}]*\}'
    match = re.search(pattern, real_response, re.DOTALL)
    return eval(match.group())[key]


class OpenIE:
    def __init__(self, llm_model: CacheOpenAI):
        # Init prompt template manager
        self.prompt_template_manager = PromptTemplateManager(
            role_mapping={"system": "system", "user": "user", "assistant": "assistant"}
        )
        self.llm_model = llm_model

    def ner(self, chunk_key: str, passage: str) -> NerRawOutput:
        # PREPROCESSING
        ner_input_message = self.prompt_template_manager.render(name="ner", passage=passage)
        raw_response = ""
        metadata = {}
        try:
            # LLM INFERENCE
            raw_response, metadata, cache_hit = self.llm_model.infer(messages=ner_input_message, response_format=None)
            metadata["cache_hit"] = cache_hit
            if metadata["finish_reason"] == "length":
                real_response = fix_broken_generated_json(raw_response)
            else:
                real_response = raw_response
            extracted_entities = _extract_value_from_json_response(real_response)
            unique_entities = list(dict.fromkeys(extracted_entities))  # deduplicate and keep the order
            # only keep strings in the unique entities
            unique_entities = [entity for entity in unique_entities if isinstance(entity, str)]

        except Exception as e:
            # For any other unexpected exceptions, log them and return with the error message
            logger.warning(f"NER exception {e}")
            metadata.update({"NER Error": str(e)})
            return NerRawOutput(
                chunk_id=chunk_key,
                response=raw_response,  # Store the error message in metadata
                unique_entities=[],
                metadata=metadata,  # Store the error message in metadata
            )

        return NerRawOutput(
            chunk_id=chunk_key, response=raw_response, unique_entities=unique_entities, metadata=metadata
        )

    def triple_extraction(self, chunk_key: str, passage: str, named_entities: List[str]) -> TripleRawOutput:

        # PREPROCESSING
        try:
            named_entities_str = json.dumps({"named_entities": named_entities})
        except Exception as e:
            named_entities_str = json.dumps({"named_entities": []})
            logger.info(f"Dumping NER exception during triple extraction {e}, {named_entities_str}")

        triple_extraction_input_message = self.prompt_template_manager.render(
            name="triple_extraction", passage=passage, named_entity_json=named_entities_str
        )

        raw_response = ""
        metadata = {}
        try:
            # LLM INFERENCE
            raw_response, metadata, cache_hit = self.llm_model.infer(
                messages=triple_extraction_input_message, response_format=None
            )
            metadata["cache_hit"] = cache_hit
            if metadata["finish_reason"] == "length":
                real_response = fix_broken_generated_json(raw_response)
            else:
                real_response = raw_response
            extracted_triples = _extract_value_from_json_response(real_response, "triples")
            triplets = filter_invalid_triples(triples=extracted_triples)

        except Exception as e:
            logger.warning(f"Triple extraction exception for chunk {chunk_key}: {e}")
            metadata.update({"Triple extraction error": str(e)})
            return TripleRawOutput(chunk_id=chunk_key, response=raw_response, metadata=metadata, triples=[])

        # Success
        return TripleRawOutput(chunk_id=chunk_key, response=raw_response, metadata=metadata, triples=triplets)

    def openie(self, chunk_key: str, passage: str) -> Dict[str, Any]:
        ner_output = self.ner(chunk_key=chunk_key, passage=passage)
        triple_output = self.triple_extraction(
            chunk_key=chunk_key, passage=passage, named_entities=ner_output.unique_entities
        )
        return {"ner": ner_output, "triplets": triple_output}

    def batch_openie(self, chunks: Dict[str, ChunkInfo]) -> Tuple[Dict[str, NerRawOutput], Dict[str, TripleRawOutput]]:
        """
        Conduct batch OpenIE synchronously using multi-threading which includes NER and triple extraction.

        Args:
            chunks (Dict[str, ChunkInfo]): chunks to be incorporated into graph. Each key is a hashed chunk
            and the corresponding value is the chunk info to insert.

        Returns:
            Tuple[Dict[str, NerRawOutput], Dict[str, TripleRawOutput]]:
                - A dict with keys as the chunk ids and values as the NER result instances.
                - A dict with keys as the chunk ids and values as the triple extraction result instances.
        """

        # Extract passages from the provided chunks
        chunk_passages = {chunk_key: chunk["content"] for chunk_key, chunk in chunks.items()}

        ner_results_list = []
        total_prompt_tokens = 0
        total_completion_tokens = 0
        num_cache_hit = 0

        with ThreadPoolExecutor() as executor:
            # Create NER futures for each chunk
            ner_futures = {
                executor.submit(self.ner, chunk_key, passage): chunk_key
                for chunk_key, passage in chunk_passages.items()
            }

            pbar = tqdm(as_completed(ner_futures), total=len(ner_futures), desc="NER")
            for future in pbar:
                result = future.result()
                ner_results_list.append(result)
                # Update metrics based on the metadata from the result
                metadata = result.metadata
                total_prompt_tokens += metadata.get("prompt_tokens", 0)
                total_completion_tokens += metadata.get("completion_tokens", 0)
                if metadata.get("cache_hit"):
                    num_cache_hit += 1

                pbar.set_postfix(
                    {
                        "total_prompt_tokens": total_prompt_tokens,
                        "total_completion_tokens": total_completion_tokens,
                        "num_cache_hit": num_cache_hit,
                    }
                )

        triple_results_list = []
        total_prompt_tokens, total_completion_tokens, num_cache_hit = 0, 0, 0
        with ThreadPoolExecutor() as executor:
            # Create triple extraction futures for each chunk
            re_futures = {
                executor.submit(
                    self.triple_extraction,
                    ner_result.chunk_id,
                    chunk_passages[ner_result.chunk_id],
                    ner_result.unique_entities,
                ): ner_result.chunk_id
                for ner_result in ner_results_list
            }
            # Collect triple extraction results with progress bar
            pbar = tqdm(as_completed(re_futures), total=len(re_futures), desc="Extracting triples")
            for future in pbar:
                result = future.result()
                triple_results_list.append(result)
                metadata = result.metadata
                total_prompt_tokens += metadata.get("prompt_tokens", 0)
                total_completion_tokens += metadata.get("completion_tokens", 0)
                if metadata.get("cache_hit"):
                    num_cache_hit += 1
                pbar.set_postfix(
                    {
                        "total_prompt_tokens": total_prompt_tokens,
                        "total_completion_tokens": total_completion_tokens,
                        "num_cache_hit": num_cache_hit,
                    }
                )

        ner_results_dict = {res.chunk_id: res for res in ner_results_list}
        triple_results_dict = {res.chunk_id: res for res in triple_results_list}

        return ner_results_dict, triple_results_dict
