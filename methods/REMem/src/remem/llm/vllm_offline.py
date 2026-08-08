from typing import Dict, List, Tuple

from remem.utils.llm_utils import JSON_SCHEMA, TextChatMessage
from remem.utils.logging_utils import get_logger

from .base import LLMConfig

logger = get_logger(__name__)
import os

from transformers import PreTrainedTokenizer
from vllm import LLM, SamplingParams


def convert_text_chat_messages_to_strings(messages: List[TextChatMessage], tokenizer: PreTrainedTokenizer) -> List[str]:
    return tokenizer.apply_chat_template(conversation=messages, tokenize=False)


def convert_text_chat_messages_to_input_ids(
    messages: List[TextChatMessage], tokenizer: PreTrainedTokenizer
) -> List[List[int]]:
    prompt = tokenizer.apply_chat_template(
        conversation=messages,
        chat_template=None,
        tokenize=False,
        add_generation_prompt=True,
        continue_final_message=False,
        tools=None,
        documents=None,
        enable_thinking=False,
    )
    encoded = tokenizer(prompt, add_special_tokens=False)
    return encoded["input_ids"]


class VLLMOffline:

    def _init_llm_config(self) -> None:
        self.llm_config = LLMConfig()

    def __init__(self, global_config, cache_dir=None, cache_filename=None, **kwargs):
        self.global_config = global_config
        if global_config is not None:
            model_name = kwargs.get("model_name", global_config.llm_name)
        else:
            model_name = kwargs["model_name"]
        pipeline_parallel_size = 1
        tensor_parallel_size = kwargs.get("num_gpus", global_config.vllm_tensor_parallel_size)
        if "8B" in model_name or "7B" in model_name or "4B" in model_name:
            tensor_parallel_size = 1
        if "bnb" in model_name:
            kwargs["quantization"] = "bitsandbytes"
            kwargs["load_format"] = "bitsandbytes"
            tensor_parallel_size = 1
            pipeline_parallel_size = kwargs.get("num_gpus", global_config.vllm_tensor_parallel_size)

        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        self.model_name = model_name

        self.client = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
            seed=kwargs.get("seed", 0),
            max_seq_len_to_capture=global_config.max_model_len,
            enable_prefix_caching=True,
            enforce_eager=False,
            gpu_memory_utilization=kwargs.get("gpu_memory_utilization", 0.93),
            max_model_len=global_config.max_model_len,
            quantization=kwargs.get("quantization", None),
            trust_remote_code=True,
            max_num_seqs=global_config.max_num_seqs,
        )

        self.tokenizer = self.client.get_tokenizer()
        if cache_filename is None:
            cache_filename = f'{model_name.replace("/", "_")}_cache.sqlite'
        if cache_dir is None:
            cache_dir = os.path.join(global_config.save_dir, "llm_cache")
        self.cache_file_name = os.path.join(cache_dir, cache_filename)

    def infer(self, messages: List[TextChatMessage], max_completion_tokens=2048, **kwargs) -> (str, Dict, bool):
        logger.info(f"Calling VLLM offline, # of messages {len(messages)}")
        messages_list = [messages]

        #     prompt = convert_text_chat_messages_to_strings(messages_list, self.tokenizer)
        prompt_ids = convert_text_chat_messages_to_input_ids(messages_list, self.tokenizer)
        vllm_output = self.client.generate(
            prompt_token_ids=prompt_ids, sampling_params=SamplingParams(max_tokens=max_completion_tokens, temperature=0)
        )

        response = vllm_output[0].outputs[0].text
        prompt_tokens = len(vllm_output[0].prompt_token_ids)
        completion_tokens = len(vllm_output[0].outputs[0].token_ids)
        metadata = {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "prompt": messages}
        return response, metadata, None  # None means no cache

    def batch_infer(
        self, messages_list: List[List[TextChatMessage]], max_tokens=2048, json_template=None
    ) -> List[Tuple[List[str], Dict, bool]]:
        if len(messages_list) > 1:
            logger.info(f"Calling VLLM offline, # of messages {len(messages_list)}")

        guided = None
        if json_template is not None:
            from vllm.model_executor.guided_decoding.guided_fields import GuidedDecodingRequest

            guided = GuidedDecodingRequest(guided_json=JSON_SCHEMA[json_template])

        try:
            all_prompt_ids = convert_text_chat_messages_to_input_ids(messages_list, self.tokenizer)
            vllm_output = self.client.generate(
                prompt_token_ids=all_prompt_ids,
                sampling_params=SamplingParams(max_tokens=max_tokens, temperature=0),
                guided_options_request=guided,
            )
        except Exception as e:
            logger.error("vllm offline batch infer error", str(e))

        all_responses = [completion.outputs[0].text for completion in vllm_output]
        all_prompt_tokens = [len(completion.prompt_token_ids) for completion in vllm_output]
        all_completion_tokens = [len(completion.outputs[0].token_ids) for completion in vllm_output]

        overall_metadata = {
            "prompt_tokens": sum(all_prompt_tokens),
            "completion_tokens": sum(all_completion_tokens),
            "num_request": len(messages_list),
            "prompt": messages_list,
        }

        results = []
        for idx, completion in enumerate(vllm_output):
            response = completion.outputs[0].text
            metadata = {
                "prompt_tokens": len(completion.prompt_token_ids),
                "completion_tokens": len(completion.outputs[0].token_ids),
                "prompt": messages_list[idx],
            }
            cache_hit = None
            results.append((response, metadata, cache_hit))
        return results
