import os
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable, Dict, Any

from src.config import MethodConfig, DatasetConfig, ConfigLoader, PROJECT_ROOT, get_api_config
from src.result import EvaluationReport, ResultCollector
from utils.logger import get_eval_logger


# {dataset_name: evaluate_function}
DATASET_EVALUATOR_REGISTRY: Dict[str, Callable] = {}


def register_evaluator(dataset_name: str):
    """
    Example:
        @register_evaluator("medmemorybench")
        def evaluate_medmemorybench(method_config, dataset_config, **kwargs):
            ...
    """
    def decorator(func: Callable):
        DATASET_EVALUATOR_REGISTRY[dataset_name.lower()] = func
        return func
    return decorator


class Evaluator:

    def __init__(
        self,
        method_config: MethodConfig,
        dataset_config: DatasetConfig,
        output_dir: Optional[Path] = None,
        dry_run: bool = False,
        verbose: bool = True,
        resume: bool = False,
        rebuild_memory: bool = False,
        batch_api: bool = False,
        batch_gcs_uri: Optional[str] = None,
        batch_wait: bool = False,
    ):
        self.method_config = method_config
        self.dataset_config = dataset_config
        self.dry_run = dry_run
        self.verbose = verbose
        self.resume = resume
        self.rebuild_memory = rebuild_memory
        self.batch_api = batch_api
        self.batch_gcs_uri = batch_gcs_uri
        self.batch_wait = batch_wait

        self.output_dir = output_dir or (PROJECT_ROOT / "outputs")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.logger = get_eval_logger(
            method_config.method_name,
            dataset_config.dataset_name,
        )

        self.result_collector = ResultCollector()

    def _log(self, message: str, level: str = "INFO") -> None:
        if self.verbose:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] [{level}] {message}")
        self.logger.info(message)

    def run(self) -> EvaluationReport:
        _ensure_evaluators_registered()

        dataset_name = self.dataset_config.dataset_name.lower()

        evaluate_func = DATASET_EVALUATOR_REGISTRY.get(dataset_name)
        if evaluate_func is None:
            raise ValueError(
                f"No evaluator found for dataset '{dataset_name}', "
                f"available: {list(DATASET_EVALUATOR_REGISTRY.keys())}"
            )

        start_time = datetime.now()
        self._log("=" * 60)
        self._log("Starting evaluation")
        self._log(f"  Method: {self.method_config.method_name}")
        self._log(f"  Model: {self.method_config.model.name}")
        self._log(f"  Dataset: {self.dataset_config.dataset_name}")
        self._log(f"  Dry Run: {self.dry_run}")
        self._log(f"  Resume: {self.resume}")
        self._log(f"  Rebuild Memory: {self.rebuild_memory}")
        self._log(f"  Vertex Batch API: {self.batch_api}")
        self._log("=" * 60)

        if self.batch_api and not self.dry_run:
            self._validate_batch_eligibility()

        report = evaluate_func(
            method_config=self.method_config,
            dataset_config=self.dataset_config,
            output_dir=self.output_dir,
            dry_run=self.dry_run,
            verbose=self.verbose,
            logger=self.logger,
            resume=self.resume,
            rebuild_memory=self.rebuild_memory,
            batch_api=self.batch_api,
            batch_gcs_uri=self.batch_gcs_uri,
            batch_wait=self.batch_wait,
        )

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        self._log("=" * 60)
        self._log(f"Evaluation completed in {duration:.2f}s")
        self._log(f"  Total Queries: {report.summary.get('total', 0)}")
        self._log(f"  Accuracy: {report.summary.get('overall_accuracy', 0):.2%}")
        self._log("=" * 60)

        return report

    def _validate_batch_eligibility(self) -> None:
        """Fail early when --batch-api cannot reach any safe Gemini stage."""
        provider = self.method_config.model.provider.lower()
        method_name = self.method_config.method_name.lower()
        eligible_methods = (
            "long_context", "embedding_rag", "bm25_rag", "lightmem",
            "zep", "remem", "graph_rag", "amem", "mem0", "memos",
            "memrl", "mirix", "smart_mem0",
        )
        agent_params = getattr(self.method_config, "raw_config", {}).get("agent_params", {})
        has_method_stage = (
            (provider in {"gemini", "vertex", "vertex_ai"} or "gemini" in self.method_config.model.name.lower())
            and any(name in method_name for name in eligible_methods)
            and not (
                "mirix" in method_name
                and agent_params.get("use_native_query", True)
            )
        )
        has_medmemorybench_judge = (
            self.dataset_config.dataset_name.lower() == "medmemorybench"
            and (get_api_config().get_judge_provider().lower() in {"gemini", "vertex", "vertex_ai"} or "gemini" in get_api_config().get_judge_model().lower())
        )
        if not has_method_stage and not has_medmemorybench_judge:
            raise ValueError(
                "--batch-api requires at least one eligible Gemini stage. "
                "Use a supported Gemini method or configure a Gemini MedMemoryBench judge."
            )
        if not self.batch_gcs_uri and not os.environ.get("GOOGLE_BATCH_GCS_URI"):
            raise ValueError(
                "--batch-api requires --batch-gcs-uri or GOOGLE_BATCH_GCS_URI before evaluation starts."
            )


def _ensure_evaluators_registered():
    """Lazy import to avoid circular import."""
    if "medmemorybench" not in DATASET_EVALUATOR_REGISTRY:
        from benchmarks.medmemorybench.evaluator import evaluate_medmemorybench  # noqa: F401
    if "locomo" not in DATASET_EVALUATOR_REGISTRY:
        from benchmarks.locomo.evaluator import evaluate_locomo  # noqa: F401


def create_evaluator(
    method_config_name: str,
    dataset_name: str,
    config_loader: Optional[ConfigLoader] = None,
    **kwargs
) -> Evaluator:
    if config_loader is None:
        config_loader = ConfigLoader()

    method_config = config_loader.load_method_config(method_config_name)
    dataset_config = config_loader.load_dataset_config(dataset_name)

    dataset_overrides = method_config.raw_config.get("dataset_overrides", {})
    if dataset_overrides:
        allowed_overrides = {
            "persona_ids", "max_personas", "max_sessions_per_persona",
            "evaluation_interval", "inject_noise", "evaluation_mode",
        }
        unexpected = set(dataset_overrides) - allowed_overrides
        if unexpected:
            raise ValueError(
                f"Unsupported dataset overrides in {method_config_name}: "
                f"{', '.join(sorted(unexpected))}"
            )
        dataset_config = replace(dataset_config, **dataset_overrides)

    return Evaluator(
        method_config=method_config,
        dataset_config=dataset_config,
        **kwargs
    )


def list_available_evaluators() -> list:
    return list(DATASET_EVALUATOR_REGISTRY.keys())
