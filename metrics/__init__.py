"""Evaluation metrics module."""

import threading
from typing import Dict, List, Any, Type, Optional

from .base import BaseMetric, MetricResult
from .string_match import StringContainMetric, ExactMatchMetric, OptionMatchMetric, MQFixMetric
from .llm_judge import LLMJudgeMetric, EEMJudgeMetric, LLMJudgeMCDMetric
from .locomo_metrics import LoCoMoF1Metric, LoCoMoAdversarialMetric, LoCoMoTemporalMetric
from .retrieval_quality import (
    RETRIEVAL_QUALITY_GROUP,
    aggregate_session_retrieval_quality,
    compute_session_retrieval_quality,
)


METRIC_REGISTRY: Dict[str, Type[BaseMetric]] = {
    "string_contain": StringContainMetric,
    "exact_match": ExactMatchMetric,
    "option_match": OptionMatchMetric,
    "mq_fix": MQFixMetric,
    "llm_judge": LLMJudgeMetric,
    "eem_judge": EEMJudgeMetric,
    "llm_judge_mcd": LLMJudgeMCDMetric,
    "locomo_f1": LoCoMoF1Metric,
    "locomo_adversarial": LoCoMoAdversarialMetric,
    "locomo_temporal": LoCoMoTemporalMetric,
}


class MetricsCalculator:
    """Metrics calculator - auto selects metric based on query_type."""

    DEFAULT_METRIC_MAPPING = {
        "entity_exact_match": "string_contain",
        "entity_exact_match_judge": "eem_judge",
        "temporal_localization": "llm_judge",
        "state_update": "llm_judge",
        "multiple_choice": "option_match",
        "inference_generation": "llm_judge",
        "multi_hop_clinical_deduction": "llm_judge_mcd",
        "single_hop": "locomo_f1",
        "multi_hop": "locomo_f1",
        "temporal": "locomo_f1",
        "open_domain": "locomo_f1",
        "adversarial": "locomo_adversarial",
    }

    def __init__(self, custom_mapping: Optional[Dict[str, str]] = None, dataset: str = "medmemorybench",
                 judge_model: str = None, judge_api_key: str = None, judge_base_url: str = None,
                 judge_temperature: float = None, judge_reasoning_effort=None,
                 judge_client_max_tokens: int = None,
                 judge_max_tokens: int = None, judge_mcd_max_tokens: int = None,
                 language: str = "zh"):
        self.metric_mapping = self.DEFAULT_METRIC_MAPPING.copy()
        if custom_mapping:
            self.metric_mapping.update(custom_mapping)
        self._metric_instances: Dict[str, BaseMetric] = {}
        self._metric_lock = threading.RLock()
        self._dataset = dataset
        self._judge_model = judge_model
        self._judge_api_key = judge_api_key
        self._judge_base_url = judge_base_url
        self._judge_temperature = judge_temperature
        self._judge_reasoning_effort = judge_reasoning_effort
        self._judge_client_max_tokens = judge_client_max_tokens
        self._judge_max_tokens = judge_max_tokens
        self._judge_mcd_max_tokens = judge_mcd_max_tokens
        self._language = language

    def _get_metric(self, metric_name: str) -> BaseMetric:
        # Worker threads share this calculator; create each metric/client once.
        with self._metric_lock:
            if metric_name not in self._metric_instances:
                metric_class = METRIC_REGISTRY.get(metric_name)
                if metric_class is None:
                    raise ValueError(f"Unknown metric: {metric_name}, available: {list(METRIC_REGISTRY.keys())}")
                if metric_name in ("llm_judge", "eem_judge", "llm_judge_mcd"):
                    self._metric_instances[metric_name] = metric_class(
                        dataset=self._dataset,
                        judge_model=self._judge_model,
                        judge_api_key=self._judge_api_key,
                        judge_base_url=self._judge_base_url,
                        judge_temperature=self._judge_temperature,
                        judge_reasoning_effort=self._judge_reasoning_effort,
                        judge_client_max_tokens=self._judge_client_max_tokens,
                        judge_max_tokens=self._judge_max_tokens,
                        judge_mcd_max_tokens=self._judge_mcd_max_tokens,
                        language=self._language,
                    )
                else:
                    self._metric_instances[metric_name] = metric_class()
            return self._metric_instances[metric_name]

    def compute(
        self,
        query_id: str,
        query_type: str,
        model_output: str,
        expected_answers: List[str],
        question: str = "",
        metric_name: Optional[str] = None,
        **kwargs
    ) -> MetricResult:
        # Determine metric
        if metric_name is None:
            metric_name = self.metric_mapping.get(query_type, "string_contain")
        metric = self._get_metric(metric_name)
        return metric.compute(
            query_id=query_id,
            query_type=query_type,
            model_output=model_output,
            expected_answers=expected_answers,
            question=question,
            **kwargs
        )

    def get_metric_name(self, query_type: str, metric_name: Optional[str] = None) -> str:
        """Resolve the configured metric without evaluating it."""
        return metric_name or self.metric_mapping.get(query_type, "string_contain")

    def prepare_batch(
        self,
        query_id: str,
        query_type: str,
        model_output: str,
        expected_answers: List[str],
        question: str = "",
        metric_name: Optional[str] = None,
        **kwargs,
    ) -> Optional[Dict[str, Any]]:
        """Prepare an LLM-judge request, returning ``None`` for local metrics."""
        resolved_name = self.get_metric_name(query_type, metric_name)
        if resolved_name not in {"llm_judge", "eem_judge", "llm_judge_mcd"}:
            return None
        metric = self._get_metric(resolved_name)
        return {
            "metric_name": resolved_name,
            "prepared": metric.prepare_batch(
                query_id=query_id,
                query_type=query_type,
                model_output=model_output,
                expected_answers=expected_answers,
                question=question,
                **kwargs,
            ),
        }

    def get_batch_judge_client(
        self,
        query_type: str,
        metric_name: Optional[str] = None,
    ):
        """Return the judge's batch-capable client, or ``None`` for fallback."""
        resolved_name = self.get_metric_name(query_type, metric_name)
        if resolved_name not in {"llm_judge", "eem_judge", "llm_judge_mcd"}:
            return None
        metric = self._get_metric(resolved_name)
        return metric.get_batch_client()

    def finalize_batch(self, batch_payload: Dict[str, Any], result_text: str) -> MetricResult:
        """Turn a batch judge response back into the normal metric result."""
        metric = self._get_metric(batch_payload["metric_name"])
        return metric.finalize_batch(batch_payload["prepared"], result_text)


class MetricsAggregator:
    """Aggregates evaluation results."""

    def __init__(self):
        self.results: List[MetricResult] = []

    def add_result(self, result: MetricResult) -> None:
        self.results.append(result)

    def add_results(self, results: List[MetricResult]) -> None:
        self.results.extend(results)

    def get_summary(self) -> Dict[str, Any]:
        if not self.results:
            return {"total": 0, "message": "No results"}

        scored_results = [r for r in self.results if r.score is not None and r.is_correct is not None]
        total = len(scored_results)
        correct_count = sum(1 for r in scored_results if r.is_correct)
        total_score = sum(float(r.score) for r in scored_results)

        by_type: Dict[str, List[MetricResult]] = {}
        for result in scored_results:
            if result.query_type not in by_type:
                by_type[result.query_type] = []
            by_type[result.query_type].append(result)

        type_stats = {}
        for query_type, type_results in by_type.items():
            type_total = len(type_results)
            type_correct = sum(1 for r in type_results if r.is_correct)
            type_score = sum(float(r.score) for r in type_results)

            stats = {
                "total": type_total,
                "correct": type_correct,
                "accuracy": type_correct / type_total if type_total > 0 else 0.0,
                "avg_score": type_score / type_total if type_total > 0 else 0.0,
            }

            # MCD type extra stats
            if query_type == "multi_hop_clinical_deduction":
                ncr_scores = [r.details.get("ncr_score", 0.0) for r in type_results]
                crc_scores = [r.details.get("crc_score", 0.0) for r in type_results]
                cc_scores = [r.details.get("cc_score", 0.0) for r in type_results]

                stats["avg_ncr"] = sum(ncr_scores) / len(ncr_scores) if ncr_scores else 0.0
                stats["avg_crc"] = sum(crc_scores) / len(crc_scores) if crc_scores else 0.0
                stats["avg_cc"] = sum(cc_scores) / len(cc_scores) if cc_scores else 0.0

                total_nodes_validated = 0
                total_nodes_mentioned = 0
                total_causal_correct = 0
                for r in type_results:
                    node_validations = r.details.get("node_validations", [])
                    for nv in node_validations:
                        total_nodes_validated += 1
                        if nv.get("mentioned", False):
                            total_nodes_mentioned += 1
                        if nv.get("causal_link_correct", False):
                            total_causal_correct += 1

                if total_nodes_validated > 0:
                    stats["node_mention_rate"] = total_nodes_mentioned / total_nodes_validated
                    stats["node_causal_rate"] = total_causal_correct / total_nodes_validated
                    stats["total_nodes_validated"] = total_nodes_validated

            type_stats[query_type] = stats

        metric_stats: Dict[str, Dict[str, Any]] = {}
        for result in scored_results:
            configured_metrics = result.details.get("metrics")
            if not isinstance(configured_metrics, dict):
                metric_name = result.details.get("metric", "unknown")
                configured_metrics = {
                    metric_name: {
                        "score": result.score,
                        "is_correct": result.is_correct,
                    }
                }
            for metric_name, metric_result in configured_metrics.items():
                stats = metric_stats.setdefault(
                    metric_name,
                    {"total": 0, "correct": 0, "score": 0.0},
                )
                stats["total"] += 1
                stats["correct"] += 1 if metric_result.get("is_correct", False) else 0
                stats["score"] += float(metric_result.get("score", 0.0))

        for stats in metric_stats.values():
            total_for_metric = stats["total"]
            stats["accuracy"] = (
                stats["correct"] / total_for_metric if total_for_metric else 0.0
            )
            stats["avg_score"] = (
                stats["score"] / total_for_metric if total_for_metric else 0.0
            )
            del stats["score"]

        total_memory_time = sum(r.memory_construction_time for r in self.results)
        total_query_time = sum(r.query_time for r in self.results)

        efficiency_stats = {
            "total_memory_construction_time": total_memory_time,
            "total_query_time": total_query_time,
            "avg_memory_construction_time": total_memory_time / total if total > 0 else 0.0,
            "avg_query_time": total_query_time / total if total > 0 else 0.0,
        }

        retrieval_quality = aggregate_session_retrieval_quality(self.results)
        metric_groups = (
            {"retrieval_quality": retrieval_quality}
            if retrieval_quality else {}
        )

        summary = {
            "total": total,
            "executed": len(self.results),
            "unscored": len(self.results) - total,
            "correct": correct_count,
            "overall_accuracy": correct_count / total if total > 0 else 0.0,
            "overall_avg_score": total_score / total if total > 0 else 0.0,
            "by_type": type_stats,
            "by_metric": metric_stats,
            "efficiency": efficiency_stats,
        }
        if metric_groups:
            summary["metric_groups"] = metric_groups
        return summary

    def get_detailed_results(self) -> List[Dict[str, Any]]:
        return [r.to_dict() for r in self.results]

    def clear(self) -> None:
        self.results.clear()


__all__ = [
    "BaseMetric",
    "MetricResult",
    "StringContainMetric",
    "ExactMatchMetric",
    "OptionMatchMetric",
    "MQFixMetric",
    "LLMJudgeMetric",
    "EEMJudgeMetric",
    "LLMJudgeMCDMetric",
    "LoCoMoF1Metric",
    "LoCoMoAdversarialMetric",
    "LoCoMoTemporalMetric",
    "RETRIEVAL_QUALITY_GROUP",
    "compute_session_retrieval_quality",
    "aggregate_session_retrieval_quality",
    "MetricsCalculator",
    "MetricsAggregator",
    "METRIC_REGISTRY",
]
