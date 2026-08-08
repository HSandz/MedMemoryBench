import os
from abc import ABC
from dataclasses import asdict
from datetime import datetime
from typing import Optional

import tiktoken

from remem.utils.config_utils import BaseConfig
from remem.utils.logging_utils import get_logger

logger = get_logger(__name__)


def _get_text_encoder(encoder_name: str = "gpt-4o"):
    if encoder_name == "gpt-4o":
        return tiktoken.encoding_for_model("gpt-4o")


class BasePreprocessor(ABC):
    """Abstract base class for all preprocessors."""

    global_config: BaseConfig
    working_dir: str

    def __init__(self, working_dir: Optional[str] = None, global_config: Optional[BaseConfig] = None) -> None:
        if global_config is None:
            logger.debug("global config is not given. Using the default BaseConfig instance.")
            self.global_config = BaseConfig()
        else:
            self.global_config = global_config
        logger.debug(f"Loading {self.__class__.__name__} with global_config: {asdict(self.global_config)}")

        if working_dir is None:
            logger.debug("working_dir is not given. Using the default working directory.")
            self.working_dir = os.path.join(
                self.global_config.save_dir, f"Preprocessor_{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}"
            )
        else:
            self.working_dir = working_dir

        if not os.path.exists(self.working_dir):
            logger.info(f"Creating working directory: {self.working_dir}")
            os.makedirs(self.working_dir, exist_ok=True)
