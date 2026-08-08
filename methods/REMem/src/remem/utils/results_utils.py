"""
Results loading and processing utilities for ReMem evaluation scripts.

This module provides utilities for loading and processing evaluation results,
with support for both per-sample and aggregate result formats.
"""

import json
import os
from typing import Any, Dict, Optional


def get_inference_type(
    agent_fixed_tools: bool = False, agent_max_steps: int = 5, extract_method="eposodic_gist"
) -> str:
    """Generate inference type string based on agent configuration.

    Args:
        agent_fixed_tools: Whether agent uses fixed tools
        agent_max_steps: Maximum reasoning steps for agent

    Returns:
        String identifier for inference type
    """

    if extract_method == "openie":
        inference_type = "ppr"
    else:
        inference_type = (
            f"agent_max_step_{agent_max_steps}" if not agent_fixed_tools else f"agent_fixed_max_step_{agent_max_steps}"
        )
    return inference_type


def get_working_dir(output_dir: str, dataset_name: str, llm_name: str, embedding_name: str, extract_method=None) -> str:
    llm_label = llm_name.replace("/", "_")
    embedding_label = embedding_name.replace("/", "_")
    file_label = f"{dataset_name}_{llm_label}_{embedding_label}"
    if extract_method is not None and extract_method == "openie":
        file_label += "_openie"
    working_dir = os.path.join(output_dir, file_label)
    return working_dir


def load_per_sample_metrics(samples_dir: str, num_questions: int, inference_type: str) -> Optional[Dict[str, float]]:
    """Load metrics from per-sample files.

    Args:
        samples_dir: Directory containing per-sample files
        num_questions: Number of questions/samples to load
        inference_type: Inference type identifier used in filenames

    Returns:
        Dictionary of averaged metrics, or None if loading failed
    """
    if not os.path.exists(samples_dir):
        return None

    print(f"Found per-sample directory: {samples_dir}")
    sample_metrics_list = []

    for q_idx in range(num_questions):
        sample_file = os.path.join(samples_dir, f"rag_results_question_{q_idx}_{inference_type}.json")
        if os.path.exists(sample_file):
            try:
                with open(sample_file, "r") as f:
                    sample_data = json.load(f)

                # Extract metrics from per-sample file
                # Per-sample files have metrics at top level, not under a "sample" key
                sample_metrics = sample_data.get("sample_metrics", {})

                if isinstance(sample_metrics, dict) and sample_metrics:
                    sample_metrics_list.append(sample_metrics)
            except Exception as e:
                print(f"Warning: Failed to load per-sample file {sample_file}: {e}")
                continue

    # If we successfully loaded per-sample metrics, aggregate them
    if sample_metrics_list:
        print(f"Successfully loaded metrics from {len(sample_metrics_list)} per-sample files")
        num_samples = len(sample_metrics_list)
        total_metrics = {}

        for sample_metrics in sample_metrics_list:
            for key, value in sample_metrics.items():
                try:
                    # Ensure the value is numeric
                    numeric_value = float(value)
                    total_metrics[key] = total_metrics.get(key, 0.0) + numeric_value
                except (ValueError, TypeError):
                    print(f"Warning: Skipping non-numeric metric {key}={value}")
                    continue

        # Average the metrics by number of samples
        if num_samples > 0:
            total_metrics = {key: value / num_samples for key, value in total_metrics.items()}
            return total_metrics

    return None


def load_aggregate_metrics(existing_results: Dict[str, Any]) -> Optional[Dict[str, float]]:
    """Load metrics from aggregate results file.

    Args:
        existing_results: Dictionary loaded from main results file

    Returns:
        Dictionary of metrics, or None if loading failed
    """
    print("Per-sample metrics not found or failed to load, trying main results file")
    samples_list = existing_results.get("samples", [])
    if not isinstance(samples_list, list):
        samples_list = []
    num_samples = len(samples_list)

    if num_samples == 0:
        print("Warning: No samples found in existing results")
        return {}

    total_metrics = {}
    for sample in samples_list:
        if not isinstance(sample, dict):
            continue
        sample_metrics = sample.get("sample_metrics", {})
        if not isinstance(sample_metrics, dict):
            continue

        for key, value in sample_metrics.items():
            try:
                # Ensure the value is numeric
                numeric_value = float(value)
                total_metrics[key] = total_metrics.get(key, 0.0) + numeric_value
            except (ValueError, TypeError):
                print(f"Warning: Skipping non-numeric metric {key}={value}")
                continue

    # Average the metrics by number of samples
    if num_samples > 0:
        total_metrics = {key: value / num_samples for key, value in total_metrics.items()}
        return total_metrics
    else:
        return {}


def load_existing_results(
    results_file: str,
    num_questions: int,
    agent_fixed_tools: bool = False,
    agent_max_steps: int = 5,
    dataset_name: str = "unknown",
    llm_name: str = "unknown",
    embedding_name: str = "unknown",
) -> Dict[str, float]:
    """Load and process existing evaluation results.

    This function handles two scenarios:
    1. When save_per_sample=True was used, it loads metrics from individual sample files
       in the samples/ directory, which contain the correct per-sample metrics.
    2. When save_per_sample=False was used, it falls back to loading from the main results file.

    This addresses the issue where save_per_sample=True would save correct metrics to
    individual files, but the main results file would contain incorrect overall_metrics (0.0).

    Args:
        results_file: Path to the main results file
        num_questions: Number of questions in the dataset
        agent_fixed_tools: Whether agent uses fixed tools
        agent_max_steps: Maximum reasoning steps for agent
        dataset_name: Name of the dataset (for display)
        llm_name: Name of the LLM (for display)
        embedding_name: Name of the embedding model (for display)

    Returns:
        Dictionary of averaged metrics

    Raises:
        ValueError: If results file cannot be loaded
    """
    try:
        with open(results_file, "r") as f:
            existing_results = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        raise ValueError(f"Failed to load results file {results_file}: {e}")

    # Extract or recalculate metrics
    recalculate = True
    total_metrics = {}

    if existing_results.get("overall_metrics"):
        total_metrics = existing_results["overall_metrics"]
        # Check if we have valid metrics (not all None/empty)
        if total_metrics and any(v is not None and v != "" and v != 0.0 for v in total_metrics.values()):
            recalculate = False

    if recalculate:
        total_metrics = {}

        # First, try to load from per-sample files if they exist
        working_dir = os.path.dirname(results_file)
        samples_dir = os.path.join(working_dir, "samples")
        inference_type = get_inference_type(agent_fixed_tools, agent_max_steps)

        # Try to load per-sample metrics
        per_sample_metrics = load_per_sample_metrics(samples_dir, num_questions, inference_type)

        if per_sample_metrics is not None:
            total_metrics = per_sample_metrics
        else:
            # Fall back to loading from main results file
            aggregate_metrics = load_aggregate_metrics(existing_results)
            if aggregate_metrics is not None:
                total_metrics = aggregate_metrics

    return total_metrics


def print_evaluation_results(
    dataset_name: str,
    num_questions: int,
    llm_name: str,
    embedding_name: str,
    metrics: Dict[str, float],
    title: str = "Results",
) -> None:
    """Print formatted evaluation results.

    Args:
        dataset_name: Name of the dataset
        num_questions: Number of questions processed
        llm_name: Name of the LLM used
        embedding_name: Name of the embedding model used
        metrics: Dictionary of metrics to display
        title: Title for the results section
    """
    print(f"\n=== {title} ===")
    print(f"Dataset: {dataset_name}")
    print(f"Total # of questions: {num_questions}")
    print(f"LLM: {llm_name}")
    print(f"Embedding Model: {embedding_name}")
    print("\nAll Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {round(value, 4)}")


def print_existing_results_summary(
    results_file: str,
    dataset_name: str,
    num_questions: int,
    llm_name: str,
    embedding_name: str,
    metrics: Dict[str, float],
) -> None:
    """Print summary of loaded existing results.

    Args:
        results_file: Path to the results file
        dataset_name: Name of the dataset
        num_questions: Number of questions
        llm_name: Name of the LLM
        embedding_name: Name of the embedding model
        metrics: Dictionary of loaded metrics
    """
    print_evaluation_results(
        dataset_name=dataset_name,
        num_questions=num_questions,
        llm_name=llm_name,
        embedding_name=embedding_name,
        metrics=metrics,
        title="Existing Results (Loaded)",
    )

    print(f"\nResults loaded from {results_file}")
    print("Evaluation completed using existing results")


def check_and_load_existing_results(
    working_dir: str,
    inference_type: str,
    num_questions: int,
    agent_fixed_tools: bool = False,
    agent_max_steps: int = 5,
    dataset_name: str = "unknown",
    llm_name: str = "unknown",
    embedding_name: str = "unknown",
    force_rag: bool = False,
) -> Optional[Dict[str, float]]:
    """Check for existing results and load them if available.

    Args:
        working_dir: Working directory containing results
        inference_type: Inference type identifier
        num_questions: Number of questions in the dataset
        agent_fixed_tools: Whether agent uses fixed tools
        agent_max_steps: Maximum reasoning steps for agent
        dataset_name: Name of the dataset
        llm_name: Name of the LLM
        embedding_name: Name of the embedding model
        force_rag: Whether to force rerun even if results exist

    Returns:
        Dictionary of metrics if existing results found, None otherwise
    """
    results_file = os.path.join(working_dir, f"rag_results_{inference_type}.json")

    if os.path.exists(results_file) and not force_rag:
        print(f"RAG results already exist at {results_file}")
        print("Loading existing results... (use --force_rag/-fr to rerun)")

        try:
            metrics = load_existing_results(
                results_file=results_file,
                num_questions=num_questions,
                agent_fixed_tools=agent_fixed_tools,
                agent_max_steps=agent_max_steps,
                dataset_name=dataset_name,
                llm_name=llm_name,
                embedding_name=embedding_name,
            )

            print_existing_results_summary(
                results_file=results_file,
                dataset_name=dataset_name,
                num_questions=num_questions,
                llm_name=llm_name,
                embedding_name=embedding_name,
                metrics=metrics,
            )

            return metrics

        except Exception as e:
            print(f"Warning: Failed to load existing results ({e}), will rerun...")
            return None

    return None
