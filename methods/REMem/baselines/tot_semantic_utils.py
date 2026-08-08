#!/usr/bin/env python3
"""
Shared utility functions for ToT semantic evaluation scripts.
"""

from typing import Any, Dict, List


def calculate_question_type_stats(
    samples: List[Dict[str, Any]], results: List[Dict[str, Any]]
) -> Dict[str, Dict[str, int]]:
    """
    Calculate statistics by question type.

    Args:
        samples: List of samples from the dataset
        results: List of evaluation results

    Returns:
        Dictionary with question type statistics
    """
    question_type_stats = {}
    for i, sample in enumerate(samples):
        question_type = sample.get("question_type", "unknown")
        if question_type not in question_type_stats:
            question_type_stats[question_type] = {"total": 0, "correct": 0}

        question_type_stats[question_type]["total"] += 1
        if results[i]["sample_metrics"]["ExactMatch"] == 1:
            question_type_stats[question_type]["correct"] += 1

    return question_type_stats


def print_question_type_breakdown(
    question_type_stats: Dict[str, Dict[str, int]], title: str = "Accuracy by Question Type"
):
    """
    Print question type statistics in a formatted table.

    Args:
        question_type_stats: Dictionary with question type statistics
        title: Title for the statistics section
    """
    print(f"\n=== {title} ===")
    print(f"{'Question Type':<20} {'Count':<8} {'Correct':<8} {'Accuracy':<10}")
    print("-" * 50)

    for question_type in sorted(question_type_stats.keys()):
        stats = question_type_stats[question_type]
        type_accuracy = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        print(f"{question_type:<20} {stats['total']:<8} {stats['correct']:<8} {type_accuracy:<10.4f}")

    print("-" * 50)


def print_evaluation_summary(
    accuracy: float,
    correct_count: int,
    total_count: int,
    question_type_stats: Dict[str, Dict[str, int]],
    evaluation_type: str = "Evaluation",
):
    """
    Print a complete evaluation summary with overall and question type statistics.

    Args:
        accuracy: Overall accuracy
        correct_count: Number of correct predictions
        total_count: Total number of samples
        question_type_stats: Dictionary with question type statistics
        evaluation_type: Type of evaluation (e.g., "LLM", "TISER")
    """
    print(f"\n=== {evaluation_type} Results ===")
    print(f"Overall Accuracy: {accuracy:.4f} ({correct_count}/{total_count})")

    print_question_type_breakdown(question_type_stats)
