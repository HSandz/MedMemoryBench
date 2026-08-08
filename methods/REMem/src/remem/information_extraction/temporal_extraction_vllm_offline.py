import json
from typing import Dict

from remem.information_extraction.openie_openai import ChunkInfo
from remem.llm.vllm_offline import VLLMOffline
from remem.prompts import PromptTemplateManager
from remem.utils.chunk_utils import make_chunk_content
from remem.utils.logging_utils import get_logger
from remem.utils.misc_utils import TemporalRawOutput

logger = get_logger(__name__)


class VLLMOfflineTemporalExtraction:
    def __init__(self, global_config, llm_model=None):
        # Initialize prompt template manager with role mapping
        self.prompt_template_manager = PromptTemplateManager(
            role_mapping={"system": "system", "user": "user", "assistant": "assistant"}
        )
        if llm_model is not None:
            self.llm_model = llm_model
        else:
            self.llm_model = VLLMOffline(global_config)
        self.global_config = global_config

    def batch_openie(self, chunks: Dict[str, ChunkInfo]) -> Dict[str, TemporalRawOutput]:
        """Conduct batch temporal extraction using VLLM offline mode."""
        return self.batch_extraction(chunks)

    def batch_extraction(self, chunks: Dict[str, ChunkInfo]):
        """Extract temporal triples from chunks using batch inference."""
        # Prepare messages for batch inference
        chunk_passages = {
            chunk_key: make_chunk_content("temporal", chunk["metadata"]) for chunk_key, chunk in chunks.items()
        }

        # Use temporal extraction template
        input_messages = [
            self.prompt_template_manager.render(name="temporal_extraction", prompt_user=passage)
            for passage in chunk_passages.values()
        ]

        try:
            infer_results = self.llm_model.batch_infer(input_messages, json_template="temporal_facts", max_tokens=2048)
            responses, metadata, _ = zip(*infer_results)
        except Exception as e:
            logger.warning(f"Batch temporal extraction exception: {e}")
            # Return empty dictionaries in case of batch failure.
            return {}

        outputs = []
        # Process results in the same order as the input messages
        for idx, raw_response in enumerate(responses):
            chunk_id = list(chunks.keys())[idx]
            cur_metadata = metadata[idx]

            try:
                generated_json_obj = json.loads(raw_response)
                passage = chunk_passages[chunk_id]
                passage_split = passage.split("\n")
                if len(passage_split) == 2:
                    verbatim = passage_split[1]
                else:
                    verbatim = passage

                temporal_triples = generated_json_obj.get("facts", [])

                outputs.append(
                    TemporalRawOutput(
                        chunk_id=chunk_id,
                        response=raw_response,
                        metadata=cur_metadata,
                        verbatim=verbatim,
                        facts=temporal_triples,
                    )
                )

            except Exception as e:
                logger.warning(f"Error extracting temporal triples for chunk {chunk_id}: {e}")
                cur_metadata.update({"Extraction Error": str(e)})
                outputs.append(
                    TemporalRawOutput(
                        chunk_id=chunk_id, response=raw_response, metadata=cur_metadata, verbatim=verbatim, facts=[]
                    )
                )

        result_dict = {item.chunk_id: item for item in outputs}
        return result_dict
