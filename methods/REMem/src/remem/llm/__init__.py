import os

from remem.utils.config_utils import BaseConfig
from remem.utils.logging_utils import get_logger

from .base import BaseLLM
from .openai_gpt import CacheOpenAI

logger = get_logger(__name__)


def _get_llm_class(config: BaseConfig):
    # Original logic for OpenAI/Azure
    if (config.llm_base_url is None) or ("localhost" in config.llm_base_url and os.getenv("OPENAI_API_KEY") is None):
        os.environ["OPENAI_API_KEY"] = "sk-"
    return CacheOpenAI.from_experiment_config(config)
