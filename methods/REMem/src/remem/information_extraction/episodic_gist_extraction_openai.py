import json
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Tuple

from tqdm import tqdm

from remem.information_extraction.openie_openai import ChunkInfo
from remem.llm import CacheOpenAI
from remem.prompts import PromptTemplateManager
from remem.utils.logging_utils import get_logger
from remem.utils.misc_utils import EpisodeRawOutput

logger = get_logger(__name__)


class EpisodicGistExtraction:
    def __init__(self, llm_model: CacheOpenAI, global_config=None):
        # Init prompt template manager
        self.prompt_template_manager = PromptTemplateManager(
            role_mapping={"system": "system", "user": "user", "assistant": "assistant"}
        )
        self.llm_model = llm_model
        self.global_config = global_config

    def batch_openie(self, chunks: Dict[str, ChunkInfo]) -> Tuple[Dict[str, EpisodeRawOutput]]:
        # Extract passages from the provided chunks
        from remem.utils.chunk_utils import make_chunk_content

        chunk_passages = {
            chunk_key: make_chunk_content("episodic", chunk["metadata"]) for chunk_key, chunk in chunks.items()
        }

        # First extract gists for all chunks
        gist_outputs = self.batch_extraction(chunk_passages, template="episodic_gist_extraction", target="gists")

        # Create a mapping of chunk_id to extracted gists for use in fact extraction
        gist_map = {output["chunk_id"]: output.get("gists", []) for output in gist_outputs}

        # Then extract facts, leveraging the previously extracted gists
        fact_outputs = self.batch_extraction(
            chunk_passages, template="episodic_fact_extraction", target="facts", gist_map=gist_map
        )

        # sort two list of dictionaries by the keys of chunk_passages
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
                    response=None,
                    metadata=None,
                )
            )

        result_dict = {item.chunk_id: item for item in results}
        return result_dict

    def _extract_chunk(
        self,
        chunk_key: str,
        passage: str,
        json_mode=True,
        template: str = "episodic_fact_extraction",
        target: str = "facts",
        gists: list = None,
    ):
        template_name = f"{template}_locomo"
        selected = ["menatqa", "timeqa", "musique", "complex_tr", "2wikimultihopqa"]
        if any(self.global_config.dataset.startswith(prefix) for prefix in selected):
            template_name = f"{template}_wikipedia"
        else:
            for file_name in self.prompt_template_manager.templates:
                if file_name.startswith(f"{template}_"):
                    if file_name.split("_")[-1] == self.global_config.dataset.split("_")[0]:
                        template_name = file_name
                        break

        # If gists are provided and we're doing fact extraction, include them in the prompt
        if gists and target == "facts":
            gist_context = "\n\nPreviously extracted gists for this session:\n" + "\n".join(
                [f"- {gist}" for gist in gists]
            )
            enhanced_passage = passage + gist_context
        else:
            enhanced_passage = passage

        extraction_input_message = self.prompt_template_manager.render(name=template_name, prompt_user=enhanced_passage)
        passage_split = passage.split("\n")
        if len(passage_split) == 2:
            verbatim = passage_split[1]
        else:
            verbatim = passage
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
                seed=self.global_config.seed,
                temperature=self.global_config.temperature,
            )
            metadata["cache_hit"] = cache_hit
            if metadata["finish_reason"] == "length":
                from remem.utils.llm_utils import fix_broken_generated_json

                real_response = fix_broken_generated_json(raw_response)
            else:
                real_response = raw_response.replace("null", '""')

            generated_json_obj = json.loads(real_response)
            return {
                "chunk_id": chunk_key,
                "response": raw_response,
                "metadata": metadata,
                "verbatim": verbatim,
                target: generated_json_obj.get(target, []),
            }
        except Exception as e:
            # For any other unexpected exceptions, log them and return with the error message
            logger.warning(f"Extraction exception {e}")
            metadata.update({"Extraction Error": str(e)})
            return {
                "chunk_id": chunk_key,
                "response": raw_response,
                "metadata": metadata,
                "verbatim": verbatim,
                target: None,
            }

    def batch_extraction(
        self, chunk_passages, template: str, target: str, max_workers: int = 10, gist_map: Dict[str, list] = None
    ):
        episode_results_list = []
        total_prompt_tokens = 0
        total_completion_tokens = 0
        num_cache_hit = 0

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Create extraction futures for each chunk
            extraction_futures = {}
            for chunk_key, passage in chunk_passages.items():
                gists = gist_map.get(chunk_key, []) if gist_map else None
                future = executor.submit(
                    self._extract_chunk, chunk_key, passage, template=template, target=target, gists=gists
                )
                extraction_futures[future] = chunk_key

            from concurrent.futures import as_completed

            pbar = tqdm(as_completed(extraction_futures), total=len(extraction_futures), desc=f"Extracting {target}")
            for future in pbar:
                result = future.result()
                episode_results_list.append(result)

                # Update metrics based on the metadata from the result
                metadata = result.get("metadata", {})
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

        return episode_results_list
