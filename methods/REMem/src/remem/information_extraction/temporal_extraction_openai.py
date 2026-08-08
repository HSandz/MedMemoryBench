import json
from concurrent.futures import ThreadPoolExecutor
from typing import Dict

from tqdm import tqdm

from remem.information_extraction.openie_openai import ChunkInfo
from remem.llm import CacheOpenAI
from remem.prompts import PromptTemplateManager
from remem.utils.chunk_utils import make_chunk_content
from remem.utils.logging_utils import get_logger
from remem.utils.misc_utils import TemporalRawOutput

logger = get_logger(__name__)


class TemporalExtraction:
    def __init__(self, llm_model: CacheOpenAI, global_config=None):
        # Init prompt template manager
        self.prompt_template_manager = PromptTemplateManager(
            role_mapping={"system": "system", "user": "user", "assistant": "assistant"}
        )
        self.llm_model = llm_model
        self.global_config = global_config

    def temporal_extraction(self, chunk_key: str, passage: str, json_mode=True):
        """Extract temporal triples from a single chunk of text."""
        # Use temporal extraction template
        extraction_input_message = self.prompt_template_manager.render(name="temporal_extraction", prompt_user=passage)

        raw_response = ""
        metadata = {}

        response_format = None
        if json_mode:
            response_format = {"type": "json_object"}

        try:
            # LLM INFERENCE
            raw_response, metadata, cache_hit = self.llm_model.infer(
                messages=extraction_input_message,
                response_format=response_format,
                seed=self.global_config.seed if self.global_config else None,
                temperature=self.global_config.temperature if self.global_config else 0.0,
            )
            metadata["cache_hit"] = cache_hit

            if metadata["finish_reason"] == "length":
                from remem.utils.llm_utils import fix_broken_generated_json

                real_response = fix_broken_generated_json(raw_response)
            else:
                real_response = raw_response.replace("null", '""')

            generated_json_obj = json.loads(real_response)
            facts = generated_json_obj.get("facts", [])

            return TemporalRawOutput(
                chunk_id=chunk_key, response=str(raw_response), metadata=metadata, verbatim=passage, facts=facts
            )

        except Exception as e:
            # For any other unexpected exceptions, log them and return with the error message
            logger.warning(f"Temporal extraction exception {e}")
            metadata.update({"Extraction Error": str(e)})

            return TemporalRawOutput(
                chunk_id=chunk_key, response=str(raw_response), metadata=metadata, verbatim=passage, facts=[]
            )

    def batch_openie(self, chunks: Dict[str, ChunkInfo]) -> Dict[str, TemporalRawOutput]:
        """Conduct batch temporal extraction using multi-threading."""
        # Extract passages from the provided chunks
        chunk_passages = {
            chunk_key: make_chunk_content("temporal", chunk["metadata"]) for chunk_key, chunk in chunks.items()
        }

        temporal_results_list = []
        total_prompt_tokens = 0
        total_completion_tokens = 0
        num_cache_hit = 0

        with ThreadPoolExecutor(max_workers=8) as executor:
            # Create extraction futures for each chunk
            extraction_futures = {
                executor.submit(self.temporal_extraction, chunk_key, passage): chunk_key
                for chunk_key, passage in chunk_passages.items()
            }

            from concurrent.futures import as_completed

            pbar = tqdm(as_completed(extraction_futures), total=len(extraction_futures), desc="Temporal extraction")
            for future in pbar:
                result = future.result()
                temporal_results_list.append(result)

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

        result_dict = {item.chunk_id: item for item in temporal_results_list}
        return result_dict
