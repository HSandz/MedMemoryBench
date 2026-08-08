from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Tuple

from tqdm import tqdm

from remem.information_extraction.openie_openai import ChunkInfo, _extract_value_from_json_response
from remem.llm import CacheOpenAI
from remem.prompts import PromptTemplateManager
from remem.utils.logging_utils import get_logger
from remem.utils.misc_utils import NerRawOutput, ParaphraseRawOutput, TripleRawOutput

logger = get_logger(__name__)


class EpisodicExtraction:
    def __init__(self, llm_model: CacheOpenAI, global_config=None):
        # Init prompt template manager
        self.prompt_template_manager = PromptTemplateManager(
            role_mapping={"system": "system", "user": "user", "assistant": "assistant"}
        )
        self.llm_model = llm_model
        self.global_config = global_config

    def episodic_extraction(self, chunk_key: str, passage: str, json_mode=True, paraphrasing=False):
        name = "episodic_triple_extraction_longmemeval"
        for key in self.prompt_template_manager.templates:
            if key.startswith("episodic_triple_extraction"):
                if key.split("_")[-1] == self.global_config.dataset.split("_")[0]:
                    name = key
                    break
        extraction_input_message = self.prompt_template_manager.render(name=name, prompt_user=passage)
        raw_response = ""
        metadata = {}

        response_format = None
        if json_mode:
            response_format = {"type": "json_object"}
        try:
            # LLM INFERENCE
            raw_response, metadata, cache_hit = self.llm_model.infer(
                messages=extraction_input_message, response_format=response_format
            )
            metadata["cache_hit"] = cache_hit
            if metadata["finish_reason"] == "length":
                from remem.utils.llm_utils import fix_broken_generated_json

                real_response = fix_broken_generated_json(raw_response)
            else:
                real_response = raw_response.replace("null", '""')
        except Exception as e:
            # For any other unexpected exceptions, log them and return with the error message
            logger.warning(f"Extraction exception {e}")
            metadata.update({"Extraction Error": str(e)})

            ner_output = NerRawOutput(chunk_id=chunk_key, response=raw_response, unique_entities=[], metadata=metadata)
            triple_output = TripleRawOutput(chunk_id=chunk_key, response=raw_response, triples=[], metadata=metadata)

            if paraphrasing:
                paraphrase_output = ParaphraseRawOutput(
                    chunk_id=chunk_key, response=raw_response, paraphrases=[], metadata=metadata
                )
                return ner_output, triple_output, paraphrase_output
            return ner_output, triple_output

        triples = _extract_value_from_json_response(real_response, "triples")
        triple_raw_output = TripleRawOutput(
            chunk_id=chunk_key, response=real_response, triples=triples, metadata=metadata
        )
        unique_entities = set([t[0] for t in triples] + [t[2] for t in triples if len(t) == 3])
        unique_entities = list(unique_entities)
        ner_raw_output = NerRawOutput(
            chunk_id=chunk_key, response=real_response, unique_entities=unique_entities, metadata=metadata
        )

        if paraphrasing:
            paraphrases = _extract_value_from_json_response(real_response, "paraphrases")
            paraphrase_raw_output = ParaphraseRawOutput(
                chunk_id=chunk_key, response=real_response, paraphrases=paraphrases, metadata=metadata
            )
            return ner_raw_output, triple_raw_output, paraphrase_raw_output

        return ner_raw_output, triple_raw_output

    def batch_openie(self, chunks: Dict[str, ChunkInfo]) -> Tuple[Dict[str, NerRawOutput], Dict[str, TripleRawOutput]]:
        # Extract passages from the provided chunks
        from remem.utils.chunk_utils import make_chunk_content

        chunk_passages = {
            chunk_key: make_chunk_content("episodic", chunk["metadata"]) for chunk_key, chunk in chunks.items()
        }

        ner_results_list = []
        triple_results_list = []
        paraphrase_results_list = []
        total_prompt_tokens = 0
        total_completion_tokens = 0
        num_cache_hit = 0

        with ThreadPoolExecutor() as executor:
            # Create extraction futures for each chunk
            extraction_futures = {
                executor.submit(self.episodic_extraction, chunk_key, passage): chunk_key
                for chunk_key, passage in chunk_passages.items()
            }

            from concurrent.futures import as_completed

            pbar = tqdm(as_completed(extraction_futures), total=len(extraction_futures), desc="Episodic extraction")
            for future in pbar:
                result = future.result()
                ner_results_list.append(result[0])
                triple_results_list.append(result[1])
                if len(result) == 3:
                    paraphrase_results_list.append(result[2])

                # Update metrics based on the metadata from the result
                metadata = result[0].metadata
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
        if len(paraphrase_results_list) > 0:
            paraphrase_results_dict = {res.chunk_id: res for res in paraphrase_results_list}
            return ner_results_dict, triple_results_dict, paraphrase_results_dict

        return ner_results_dict, triple_results_dict
