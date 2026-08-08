import importlib
import os
import subprocess
import sys
from typing import Dict, Optional, List, Union, Any
from transformers import PreTrainedTokenizerBase

from lightmem.configs.pre_compressor.llmlingua_2 import LlmLingua2Config


def _load_prompt_compressor():
    """Load LLMLingua, installing its small Python package on first use."""
    try:
        from llmlingua import PromptCompressor
        return PromptCompressor
    except ModuleNotFoundError as error:
        if error.name != "llmlingua":
            raise

    if os.environ.get("LIGHTMEM_AUTO_INSTALL_LLMLINGUA", "1").lower() in {"0", "false", "no"}:
        raise ImportError(
            "LLMLingua is not installed. Install it with `pip install llmlingua>=0.2.2` "
            "or unset LIGHTMEM_AUTO_INSTALL_LLMLINGUA."
        )

    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "llmlingua>=0.2.2"],
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ImportError(
            "LLMLingua is required for pre-compression and automatic installation failed. "
            "Install it with `pip install llmlingua>=0.2.2`."
        ) from error

    importlib.invalidate_caches()
    try:
        from llmlingua import PromptCompressor
        return PromptCompressor
    except ModuleNotFoundError as error:
        raise ImportError(
            "LLMLingua installation completed, but its dependencies could not be imported. "
            "Try `pip install llmlingua>=0.2.2` manually."
        ) from error


class LlmLingua2Compressor:
    def __init__(self, config: Optional[LlmLingua2Config] = None):
        self.config = config or LlmLingua2Config()
        config = self.config

        try:
            PromptCompressor = _load_prompt_compressor()
            if config.llmlingua_config['use_llmlingua2'] is True:
                self._compressor = PromptCompressor(
                    model_name=config.llmlingua_config['model_name'],
                    device_map=config.llmlingua_config['device_map'],
                    use_llmlingua2=config.llmlingua_config['use_llmlingua2'],
                    llmlingua2_config=config.llmlingua2_config
                )
            else:
                self._compressor = PromptCompressor(
                    model_name=config.llmlingua_config['model_name'],
                    device_map=config.llmlingua_config['device_map']
                )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize LlmLingua2Compressor: {str(e)}")

    def compress(
        self,
        messages: List[Dict[str, str]],
        tokenizer: Union[PreTrainedTokenizerBase, Any, None],
    ) -> List[Dict[str, str]]:
        # TODO: Consider adding an extra field in the message, compressed_content, and put the compressed content in this field while keeping content unchanged.
        """
        Compress the content of each message.

        Args:
            messages: List of message dicts containing 'role' and 'content'.
            tokenizer: Tokenizer to check token length after compression.

        Returns:
            List of messages with compressed content.
        """
        for mes in messages:
            content = mes.get('content', '')
            if not content or not content.strip():
                # If content is empty, it doesn't need compression
                continue

            compress_config = {
                'context': [content],
                **self.config.compress_config
            }

            try:
                comp_content = self._compressor.compress_prompt(**compress_config)['compressed_prompt']
            except Exception as e:
                print(f"compress error, skip this message: {e}")
                comp_content = content  # Keep the original content if compression fails

            # Check if the compressed content is still too long
            if tokenizer is not None:
                try:
                    while len(tokenizer.encode(comp_content)) >= 512 and comp_content.strip():
                        new_compress_config = {
                            'context': comp_content,
                            **self.config.compress_config
                        }
                        comp_content = self._compressor.compress_prompt(**new_compress_config)['compressed_prompt']
                except Exception as e:
                    print(f"secondary compress error: {e}")
                    # If an error occurs, exit the loop and keep the current compression result
                    break

            # Update message
            if comp_content.strip():
                mes['content'] = comp_content.strip()

        return messages

    @property
    def inner_compressor(self):
        """
        Access the underlying PromptCompressor instance directly.
        """
        return self._compressor
