"""
Unified evaluation progress utility for ReMem

This module provides a consistent interface for tracking and displaying evaluation progress
across different baselines and examples, supporting both sequential and parallel processing.
"""

import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from tqdm import tqdm


@dataclass
class EvaluationConfig:
    """Configuration for evaluation progress tracking"""

    display_interval: int = 1  # How often to update progress bar description (every N samples)
    print_interval: int = 1  # How often to print detailed metrics (every N samples)
    max_description_metrics: int = 5  # Maximum number of metrics to show in progress bar
    show_individual_results: bool = True  # Whether to log individual sample results
    show_current_averages: bool = True  # Whether to show running averages
    metric_precision: int = 4  # Number of decimal places for metrics
    processing_mode: str = "sequential"  # "sequential" or "parallel"
    num_workers: int = 1  # Number of parallel workers
    enable_thread_safe_logging: bool = True  # Whether to use thread-safe logging


class ThreadSafeMetrics:
    """Thread-safe metrics tracking for evaluation"""

    def __init__(self, config: EvaluationConfig):
        self.lock = threading.Lock()
        self.config = config
        self.cumulative_metrics = defaultdict(float)
        self.num_samples = 0
        self.results = []
        self.failed_samples = []
        self.skipped_samples = []
        self.category_metrics = defaultdict(lambda: defaultdict(float))
        self.category_counts = defaultdict(int)

    def update(
        self,
        sample_idx: int,
        metrics_dict: Dict[str, float],
        result: Any = None,
        category: Optional[str] = None,
        failed: bool = False,
        skipped: bool = False,
        error_msg: Optional[str] = None,
    ):
        """Update metrics with new sample results"""
        with self.lock:
            self.num_samples += 1

            if failed:
                self.failed_samples.append({"sample_idx": sample_idx, "error": error_msg})
            elif skipped:
                self.skipped_samples.append(sample_idx)

            # Update metrics
            for key, value in metrics_dict.items():
                self.cumulative_metrics[key] += value
                if category:
                    self.category_metrics[category][key] += value

            if category:
                self.category_counts[category] += 1

            if result is not None:
                self.results.append(result)

    def _get_current_averages_unlocked(self) -> Tuple[Dict[str, float], int]:
        """Get current average metrics and sample count (assumes lock is already held)"""
        if self.num_samples == 0:
            return {}, 0

        averages = {
            key: round(value / self.num_samples, self.config.metric_precision)
            for key, value in self.cumulative_metrics.items()
        }
        return averages, self.num_samples

    def _get_category_averages_unlocked(self) -> Dict[str, Dict[str, float]]:
        """Get average metrics by category (assumes lock is already held)"""
        category_averages = {}
        for category, metrics in self.category_metrics.items():
            count = self.category_counts[category]
            if count > 0:
                category_averages[category] = {
                    key: round(value / count, self.config.metric_precision) for key, value in metrics.items()
                }
        return category_averages

    def get_current_averages(self) -> Tuple[Dict[str, float], int]:
        """Get current average metrics and sample count"""
        with self.lock:
            return self._get_current_averages_unlocked()

    def get_category_averages(self) -> Dict[str, Dict[str, float]]:
        """Get average metrics by category"""
        with self.lock:
            return self._get_category_averages_unlocked()

    def get_final_results(self) -> Dict[str, Any]:
        """Get final evaluation results"""
        with self.lock:
            averages, num_samples = self._get_current_averages_unlocked()
            category_averages = self._get_category_averages_unlocked()

            return {
                "num_samples": num_samples,
                "num_failed": len(self.failed_samples),
                "num_skipped": len(self.skipped_samples),
                "num_successful": num_samples - len(self.failed_samples),
                "overall_metrics": averages,
                "category_metrics": category_averages,
                "results": self.results.copy(),
                "failed_samples": self.failed_samples.copy(),
                "skipped_samples": self.skipped_samples.copy(),
            }


class ThreadSafeLogger:
    """Thread-safe logger for evaluation"""

    def __init__(self, config: EvaluationConfig):
        self.lock = threading.Lock()
        self.config = config

    def log_safe(self, message: str, use_tqdm: bool = True):
        """Thread-safe logging"""
        if self.config.enable_thread_safe_logging:
            with self.lock:
                if use_tqdm:
                    tqdm.write(message)
                else:
                    print(message)
        else:
            if use_tqdm:
                tqdm.write(message)
            else:
                print(message)

    def log_sample_result(self, sample_idx: int, status: str, details: str = ""):
        """Log individual sample result"""
        if self.config.show_individual_results:
            status_symbol = "✓" if status == "success" else "✗" if status == "failed" else "⚠"
            message = f"{status_symbol} Sample {sample_idx + 1}: {status}"
            if details:
                message += f" - {details}"
            self.log_safe(message)


class EvaluationProgressTracker:
    """Unified evaluation progress tracker for both sequential and parallel processing"""

    def __init__(
        self,
        total_samples: int,
        config: EvaluationConfig,
        description: str = "Processing samples",
        dataset_name: str = "Unknown",
    ):
        self.total_samples = total_samples
        self.config = config
        self.description = description
        self.dataset_name = dataset_name

        self.metrics = ThreadSafeMetrics(config)
        self.logger = ThreadSafeLogger(config)

        self.start_time = time.time()
        self.pbar = None

        # Track all metrics for display
        self.all_metrics = set()

    def _categorize_metrics(self, metrics_dict: Dict[str, float]):
        """Collect all metrics for display"""
        for key in metrics_dict.keys():
            self.all_metrics.add(key)

    def _get_progress_description(self, current_metrics: Dict[str, float], num_processed: int) -> str:
        """Generate progress bar description with key metrics"""
        if not current_metrics:
            return f"{self.description} [{num_processed}/{self.total_samples}]"

        # Show the most important metrics (up to max_description_metrics)
        priority_metrics = list(current_metrics.keys())[: self.config.max_description_metrics]
        metrics_str = " ".join(
            [f"{key}:{value:.3f}" for key, value in current_metrics.items() if key in priority_metrics]
        )

        return f"{self.description} [{num_processed}/{self.total_samples}] {metrics_str}"

    def _update_progress_bar(self, num_processed: int, current_metrics: Dict[str, float]):
        """Update progress bar with current metrics"""
        if self.pbar is None:
            return

        # Update description
        if num_processed % self.config.display_interval == 0 or num_processed == self.total_samples:
            desc = self._get_progress_description(current_metrics, num_processed)
            self.pbar.set_description(desc)

        # Update postfix for key metrics
        if current_metrics:
            postfix = {}
            for key, value in list(current_metrics.items())[:3]:  # Show top 3 in postfix
                postfix[key] = f"{value:.3f}"
            self.pbar.set_postfix(postfix)

    def _print_detailed_metrics(self, current_metrics: Dict[str, float], num_processed: int):
        """Print detailed metrics breakdown"""
        if not self.config.show_current_averages:
            return

        if num_processed % self.config.print_interval == 0 or num_processed == self.total_samples:
            self.logger.log_safe(f"\n=== Progress Update: {num_processed}/{self.total_samples} samples ===")

            # Display all metrics together
            if self.all_metrics:
                metrics_str = " ".join(
                    [
                        f"{key}:{current_metrics.get(key, 0):.{self.config.metric_precision}f}"
                        for key in sorted(self.all_metrics)
                        if key in current_metrics
                    ]
                )
                if metrics_str:
                    self.logger.log_safe(f"Metrics: {metrics_str}")

    def start_progress(self) -> tqdm:
        """Start progress tracking"""
        self.pbar = tqdm(
            total=self.total_samples,
            desc=self.description,
            unit="sample",
            leave=True,
            position=0 if self.config.processing_mode == "parallel" else None,
        )

        # Log initial setup
        self.logger.log_safe(f"Starting evaluation of {self.total_samples} samples...")
        self.logger.log_safe(f"Dataset: {self.dataset_name}")
        self.logger.log_safe(f"Processing mode: {self.config.processing_mode}")
        if self.config.processing_mode == "parallel":
            self.logger.log_safe(f"Number of workers: {self.config.num_workers}")

        return self.pbar

    def update_sample(
        self,
        sample_idx: int,
        metrics_dict: Dict[str, float],
        result: Any = None,
        category: Optional[str] = None,
        failed: bool = False,
        skipped: bool = False,
        error_msg: Optional[str] = None,
    ):
        """Update progress with new sample result"""

        # Categorize metrics for organized display
        self._categorize_metrics(metrics_dict)

        # Update thread-safe metrics
        self.metrics.update(sample_idx, metrics_dict, result, category, failed, skipped, error_msg)

        # Get current averages
        current_metrics, num_processed = self.metrics.get_current_averages()

        # Update progress bar
        self._update_progress_bar(num_processed, current_metrics)

        # Print detailed metrics
        self._print_detailed_metrics(current_metrics, num_processed)

        # Log individual sample result
        if failed:
            self.logger.log_sample_result(sample_idx, "failed", error_msg or "Unknown error")
        elif skipped:
            self.logger.log_sample_result(sample_idx, "skipped", "Results already exist")
        else:
            self.logger.log_sample_result(sample_idx, "success")

        # Update progress bar
        if self.pbar:
            self.pbar.update(1)

    def finish_progress(self) -> Dict[str, Any]:
        """Finish progress tracking and return final results"""
        if self.pbar:
            self.pbar.close()

        elapsed_time = time.time() - self.start_time
        final_results = self.metrics.get_final_results()

        # Print final summary
        self._print_final_summary(final_results, elapsed_time)

        return final_results

    def _print_final_summary(self, results: Dict[str, Any], elapsed_time: float):
        """Print final evaluation summary"""
        print(f"\n{'=' * 50}")
        print("FINAL EVALUATION RESULTS")
        print(f"{'=' * 50}")
        print(f"Dataset: {self.dataset_name}")
        print(f"Total samples: {results['num_samples']}")
        print(f"Successful: {results['num_successful']}")
        print(f"Failed: {results['num_failed']}")
        print(f"Skipped: {results['num_skipped']}")
        print(f"Processing time: {elapsed_time:.2f} seconds")
        print(f"Processing mode: {self.config.processing_mode}")
        if self.config.processing_mode == "parallel":
            print(f"Number of workers: {self.config.num_workers}")
        print()

        # Display all metrics together
        overall_metrics = results["overall_metrics"]

        if self.all_metrics:
            print("Overall Metrics:")
            for key in sorted(self.all_metrics):
                if key in overall_metrics:
                    print(f"  {key}: {overall_metrics[key]}")

        # Display category-specific metrics if available
        if results["category_metrics"]:
            print("\nCategory-specific Metrics:")
            for category, metrics in results["category_metrics"].items():
                print(f"  {category}:")
                for key, value in metrics.items():
                    print(f"    {key}: {value}")

        print(f"{'=' * 50}")


def create_evaluation_tracker(
    total_samples: int,
    description: str = "Processing samples",
    dataset_name: str = "Unknown",
    processing_mode: str = "sequential",
    num_workers: int = 1,
    display_interval: int = 5,
    print_interval: int = 10,
    max_description_metrics: int = 3,
    show_individual_results: bool = True,
    show_current_averages: bool = True,
    metric_precision: int = 4,
    enable_thread_safe_logging: Optional[bool] = None,
) -> EvaluationProgressTracker:
    """
    Create an evaluation progress tracker with given configuration

    Args:
        total_samples: Total number of samples to process
        description: Description for progress bar
        dataset_name: Name of the dataset being processed
        processing_mode: "sequential" or "parallel"
        num_workers: Number of parallel workers
        display_interval: How often to update progress bar description
        print_interval: How often to print detailed metrics
        max_description_metrics: Maximum metrics to show in progress bar
        show_individual_results: Whether to log individual sample results
        show_current_averages: Whether to show running averages
        metric_precision: Number of decimal places for metrics
        enable_thread_safe_logging: Whether to use thread-safe logging (auto-detect if None)

    Returns:
        EvaluationProgressTracker instance
    """
    # Auto-detect thread-safe logging need
    if enable_thread_safe_logging is None:
        enable_thread_safe_logging = processing_mode == "parallel"

    config = EvaluationConfig(
        display_interval=display_interval,
        print_interval=print_interval,
        max_description_metrics=max_description_metrics,
        show_individual_results=show_individual_results,
        show_current_averages=show_current_averages,
        metric_precision=metric_precision,
        processing_mode=processing_mode,
        num_workers=num_workers,
        enable_thread_safe_logging=enable_thread_safe_logging,
    )

    return EvaluationProgressTracker(
        total_samples=total_samples, config=config, description=description, dataset_name=dataset_name
    )
