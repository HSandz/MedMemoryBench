from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

from remem.utils.config_utils import BaseConfig
from remem.utils.logging_utils import get_logger

logger = get_logger(__name__)


class BaseMetric:
    global_config: BaseConfig
    metric_name = "base"

    def __init__(self, global_config: Optional[BaseConfig] = None) -> None:
        if global_config is None:
            logger.debug("global config is not given. Using the default ExperimentConfig instance.")
            self.global_config = BaseConfig()
        else:
            self.global_config = global_config

        logger.debug(f"Loading {self.__class__.__name__} with global_config: {asdict(self.global_config)}")

    def calculate_metric_scores(self, *args: Any, **kwargs: Any) -> Tuple[Dict[str, Any], List[Any]]:
        """
        Calculate the total score under this metric and score for each individual example in the input.

        Subclasses should override this and may accept task-specific arguments, e.g.,
        (gold_answers, predicted_answers, aggregation_fn, ...).

        Returns:
            Tuple[Dict[str, Any], List[Any]]
        """
        return {}, []
