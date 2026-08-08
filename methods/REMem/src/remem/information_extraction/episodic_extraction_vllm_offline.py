from typing import Dict, Tuple

from remem.information_extraction.openie_openai import ChunkInfo, _extract_value_from_json_response
from remem.llm.vllm_offline import VLLMOffline
from remem.utils.chunk_utils import make_chunk_content
from remem.utils.logging_utils import get_logger
from remem.utils.misc_utils import NerRawOutput, ParaphraseRawOutput, TripleRawOutput

logger = get_logger(__name__)


class VLLMOfflineEpisodicExtraction:
    def __init__(self, global_config, llm_model=None):
        from remem.prompts import PromptTemplateManager

        # Initialize prompt template manager with role mapping
        self.prompt_template_manager = PromptTemplateManager(
            role_mapping={"system": "system", "user": "user", "assistant": "assistant"}
        )
        if llm_model is not None:
            self.llm_model = llm_model
        else:
            self.llm_model = VLLMOffline(global_config)
        self.global_config = global_config

    def batch_openie(self, chunks: Dict[str, ChunkInfo]) -> Tuple[Dict[str, NerRawOutput], Dict[str, TripleRawOutput]]:
        results = self.batch_triple_extraction(chunks)
        return results

    def batch_triple_extraction(self, chunks, paraphrasing=False):
        # Prepare messages for batch inference
        chunk_passages = {
            chunk_key: make_chunk_content("episodic", chunk["metadata"]) for chunk_key, chunk in chunks.items()
        }
        template_name = "episodic_triple_extraction_longmemeval"
        for key in self.prompt_template_manager.templates:
            if key.startswith("episodic_triple_extraction"):
                if key.split("_")[-1] == self.global_config.dataset.split("_")[0]:
                    template_name = key
                    break
        input_messages = [
            self.prompt_template_manager.render(name=template_name, prompt_user=passage)
            for passage in chunk_passages.values()
        ]
        try:

            infer_results = self.llm_model.batch_infer(input_messages, json_template="episodic", max_tokens=2048)
            responses, metadata, _ = zip(*infer_results)
        except Exception as e:
            logger.warning(f"Batch episodic extraction exception: {e}")
            # Return empty dictionaries in case of batch failure.
            return {}, {}

        triple_raw_outputs = []
        ner_raw_outputs = []
        paraphrase_raw_outputs = []
        # Process results in the same order as the input messages
        for idx, raw_response in enumerate(responses):
            chunk_id = list(chunks.keys())[idx]
            cur_metadata = metadata[idx]
            paraphrases = None
            try:
                triples = _extract_value_from_json_response(raw_response, "triples")
                temporal_contexts = _extract_value_from_json_response(raw_response, "temporal_contexts")
                unique_triples = list(
                    {
                        tuple(item) if isinstance(item, list) else item: None for item in (triples + temporal_contexts)
                    }.keys()
                )

                if "paraphrases" in raw_response:
                    paraphrases = _extract_value_from_json_response(raw_response, "paraphrases")
            except Exception as e:
                logger.warning(f"Error extracting triples for chunk {chunk_id}: {e}")
                unique_triples = []
                cur_metadata = {**metadata, "Extraction Error": str(e)}
            triple_raw_outputs.append(
                TripleRawOutput(chunk_id=chunk_id, response=raw_response, triples=unique_triples, metadata=cur_metadata)
            )

            unique_entities = set()
            for t in unique_triples:
                if len(t) > 0:
                    unique_entities.add(t[0])
                if len(t) > 1:
                    unique_entities.add(t[1])
            unique_entities = list(unique_entities)
            ner_raw_outputs.append(
                NerRawOutput(chunk_id, response=raw_response, unique_entities=unique_entities, metadata=metadata)
            )

            if paraphrasing is not None:
                paraphrase_raw_outputs.append(
                    ParaphraseRawOutput(chunk_id, response=raw_response, paraphrases=paraphrases, metadata=metadata)
                )

        triple_results_dict = {res.chunk_id: res for res in triple_raw_outputs}
        ner_results_dict = {res.chunk_id: res for res in ner_raw_outputs}
        if paraphrasing:
            paraphrase_results_dict = {res.chunk_id: res for res in paraphrase_raw_outputs}
            return ner_results_dict, triple_results_dict, paraphrase_results_dict
        return ner_results_dict, triple_results_dict
