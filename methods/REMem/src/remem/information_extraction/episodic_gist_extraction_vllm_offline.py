from typing import Dict, Optional

from remem.information_extraction.openie_openai import ChunkInfo
from remem.llm.vllm_offline import VLLMOffline
from remem.prompts import PromptTemplateManager
from remem.utils.chunk_utils import make_chunk_content
from remem.utils.logging_utils import get_logger
from remem.utils.misc_utils import EpisodeRawOutput

logger = get_logger(__name__)


class VLLMOfflineEpisodicGistExtraction:
    def __init__(self, global_config, llm_client=None):
        # Initialize prompt template manager with default role mapping
        self.prompt_template_manager = PromptTemplateManager()
        if llm_client is not None:
            self.llm_client = llm_client
        else:
            self.llm_client = VLLMOffline(global_config)
        self.global_config = global_config

    def batch_openie(self, chunks: Dict[str, ChunkInfo]) -> Dict[str, EpisodeRawOutput]:
        # Extract passages from the provided chunks
        chunk_passages = {
            chunk_key: make_chunk_content("episodic", chunk.get("metadata", chunk))
            for chunk_key, chunk in chunks.items()
        }

        # First extract gists for all chunks
        gist_outputs = self.batch_extraction(chunk_passages, template="episodic_gist_extraction", target="gists")

        # Create a mapping of chunk_id to extracted gists for use in fact extraction
        gist_map = {output["chunk_id"]: output.get("gists", []) for output in gist_outputs}

        # Then extract facts, leveraging the previously extracted gists
        fact_outputs = self.batch_extraction(
            chunk_passages, template="episodic_fact_extraction", target="facts", gist_map=gist_map
        )

        # Sort two lists of dictionaries by the keys of chunk_passages
        chunk_keys = list(chunk_passages.keys())
        fact_outputs = sorted(fact_outputs, key=lambda x: chunk_keys.index(x["chunk_id"]))
        gist_outputs = sorted(gist_outputs, key=lambda x: chunk_keys.index(x["chunk_id"]))

        results = []
        for fact_dict, gist_dict in zip(fact_outputs, gist_outputs):
            assert fact_dict["chunk_id"] == gist_dict["chunk_id"]
            results.append(
                EpisodeRawOutput(
                    chunk_id=fact_dict["chunk_id"],
                    verbatim=fact_dict["verbatim"],
                    facts=fact_dict.get("facts", []),
                    gists=gist_dict.get("gists", []),
                    response="",
                    metadata={},
                )
            )

        result_dict = {item.chunk_id: item for item in results}
        return result_dict

    def batch_extraction(
        self,
        chunk_passages: Dict[str, str],
        template: str = "episodic_fact_extraction",
        target: str = "facts",
        gist_map: Optional[Dict[str, list]] = None,
    ):
        # Prepare messages for batch inference
        template_name = f"{template}_longmemeval"
        for file_name in self.prompt_template_manager.templates:
            if file_name.startswith(f"{template}_"):
                if file_name.split("_")[-1] == self.global_config.dataset.split("_")[0]:
                    template_name = file_name
                    break
        logger.info(f"Using extraction template: {template_name}")

        input_messages = []
        chunk_keys = list(chunk_passages.keys())

        for chunk_key in chunk_keys:
            passage = chunk_passages[chunk_key]

            # If gists are provided and we're doing fact extraction, include them in the prompt
            if gist_map and target == "facts" and chunk_key in gist_map:
                gists = gist_map[chunk_key]
                if gists:
                    gist_context = "\n\nPreviously extracted gists for this session:\n" + "\n".join(
                        [f"- {gist}" for gist in gists]
                    )
                    enhanced_passage = passage + gist_context
                else:
                    enhanced_passage = passage
            else:
                enhanced_passage = passage

            message = self.prompt_template_manager.render(name=template_name, prompt_user=enhanced_passage)
            input_messages.append(message)

        try:
            if target == "facts":
                json_template = "temporal_facts"
            else:
                json_template = target
            infer_results = self.llm_client.batch_infer(input_messages, json_template=json_template, max_tokens=2048)
            responses, metadata, _ = zip(*infer_results)
        except Exception as e:
            logger.warning(f"Batch episodic extraction exception: {e}")
            # Return empty list in case of batch failure.
            return []

        outputs = []
        # Process results in the same order as the input messages
        for idx, raw_response in enumerate(responses):
            chunk_id = chunk_keys[idx]
            passage = chunk_passages[chunk_id]
            cur_metadata = metadata[idx]
            try:
                import json

                generated_json_obj = json.loads(raw_response)
                passage_split = passage.split("\n")
                if len(passage_split) == 2:
                    verbatim = passage_split[1]
                else:
                    verbatim = passage
                outputs.append({"chunk_id": chunk_id, "verbatim": verbatim, target: generated_json_obj.get(target, [])})
            except Exception as e:
                logger.warning(f"Error extracting triples for chunk {chunk_id}: {e}")
                cur_metadata.update({"Extraction Error": str(e)})
                passage_split = passage.split("\n")
                if len(passage_split) == 2:
                    verbatim = passage_split[1]
                else:
                    verbatim = passage
                outputs.append({"chunk_id": chunk_id, "verbatim": verbatim, target: []})

        return outputs
