"""Focused coverage for benchmark query-prompt protocol isolation."""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest

from benchmarks.locomo.dataset import LoCoMoQuery
from benchmarks.locomo.evaluator import LoCoMoEvaluator
from benchmarks.medmemorybench.checkpoint import (
    compute_memory_query_compatibility_hash,
    compute_query_config_hash,
)
from benchmarks.medmemorybench.dataset import MedQuery
from benchmarks.medmemorybench.evaluator import MedMemoryBenchEvaluator
from metrics import MetricResult
from src.config import DatasetConfig, MethodConfig
from src.evaluator import Evaluator
from src.result import EvaluationReport, ResultCollector
from utils.prompts_qa import NEUTRAL_QUERY_SYSTEM_PROMPTS, QA_TEMPLATES
from utils.templates import PromptManager


LOCOMO_TYPES = (
    "single_hop",
    "multi_hop",
    "temporal",
    "open_domain",
    "adversarial",
)
MED_TYPES = (
    "entity_exact_match",
    "temporal_localization",
    "state_update",
    "multiple_choice",
    "inference_generation",
    "multi_hop_clinical_deduction",
)

TYPE_AWARE_TEMPLATE_HASHES = {
    "locomo_single_hop_qa": "4fcc5620046e63fd8c642cffcd7529ceae844c96d6c77aa5612d025b94e1069f",
    "locomo_multi_hop_qa": "6f384340409a073c667788fac34a96fea8c4beb2db6d1f291fee58c665c32556",
    "locomo_temporal_qa": "790ffc9e55d10429e4873b1ca6813feaf8cd29f0601c2a2069fc25a8b2cea2a8",
    "locomo_open_domain_qa": "78cd9072573baf76bd8b36265d9d59d2c286ef4415e78d3cb8ff41ba7963f7af",
    "locomo_adversarial_qa": "a9f05b64d3d58d1879c8328f91d93bd376a9d8ca644374edd6995d4cd8e0b49b",
    "medmemorybench_en_entity_exact_match_qa": "5bdd49be3e67d1a12af614dea5368e387251ef4ac1d16670936c1bf3fc81bc40",
    "medmemorybench_en_temporal_localization_qa": "00772769bea98cd70fa8fc35f9b598e31e865545a64e1f3b29201794fb2c27ef",
    "medmemorybench_en_state_update_qa": "f7dde7bcb328007f1fea6ba7771eef93d88215ea03fba968b29c21428723079d",
    "medmemorybench_en_multiple_choice_qa": "de312f94724af3a2171745c33e8655c197ebb78bb149e21b78a5d20055ef8000",
    "medmemorybench_en_inference_generation_qa": "f863b5a62cef773fc03257af921a04c5ef0ba0492ee221dd7fca088cd299f5d4",
    "medmemorybench_en_multi_hop_clinical_deduction_qa": "ec9a3782ca5ebbbc72c3ca23277134d9164b0d56ab1926d4fa17ecf54cd43100",
    "medmemorybench_entity_exact_match_qa": "8e41e46a12164fca35307c6a63d39d4bb5d8d5ad393bc8bcf69d03f102208297",
    "medmemorybench_temporal_localization_qa": "2151886a073326c5fb4ef46469bfe854f60ee0508ab217705392acb7f80a02d2",
    "medmemorybench_state_update_qa": "83e95bfbef3744d019dff416258c6956436e774281ebec1edb8b7bb47fec4584",
    "medmemorybench_multiple_choice_qa": "d69f8c9f7a6b4e3aae7dbccf82c0ea3c46839f103e05cd17c1b0a31be068d6fe",
    "medmemorybench_inference_generation_qa": "d59fbd3b2859742f3d2483fb25733d0551e63c1412bf0c723fff1a226886afbf",
    "medmemorybench_multi_hop_clinical_deduction_qa": "a836d4a180f8175c6fc471872d14e6e19a458022e69b416a248fd7b7e1f7bf91",
}

FORBIDDEN_NEUTRAL_LABELS = (
    "single_hop",
    "multi_hop",
    "temporal",
    "open_domain",
    "adversarial",
    "entity_exact_match",
    "state_update",
    "multiple_choice",
    "inference_generation",
    "multi_hop_clinical_deduction",
    "query_type",
    "category",
    "gold answer",
    "gold evidence",
    "category 1",
    "category 2",
    "category 3",
    "category 4",
    "category 5",
)


def test_prompt_protocol_config_defaults_validates_and_snapshots() -> None:
    default = DatasetConfig.from_dict({"dataset_name": "locomo"})
    type_aware = DatasetConfig.from_dict({
        "dataset_name": "locomo",
        "evaluation": {"prompt_protocol": "type_aware"},
    })
    neutral = DatasetConfig.from_dict({
        "dataset_name": "locomo",
        "evaluation": {"prompt_protocol": "neutral"},
    })

    assert default.prompt_protocol == type_aware.prompt_protocol == "type_aware"
    assert neutral.prompt_protocol == "neutral"
    with pytest.raises(ValueError, match="evaluation.prompt_protocol must be one of"):
        DatasetConfig.from_dict({
            "dataset_name": "locomo",
            "evaluation": {"prompt_protocol": "unexpected"},
        })


def test_locomo_type_aware_prompts_are_unchanged() -> None:
    manager = PromptManager("locomo", method="embedding_rag", language="en")
    for query_type in LOCOMO_TYPES:
        actual = manager.format_query("What happened?", query_type)
        expected = QA_TEMPLATES[f"locomo_{query_type}_qa"].format(
            question="What happened?", memory_source="the retrieved relevant records"
        )
        assert actual == expected
    for key, expected_hash in TYPE_AWARE_TEMPLATE_HASHES.items():
        if key.startswith("locomo_"):
            assert hashlib.sha256(QA_TEMPLATES[key].encode()).hexdigest() == expected_hash


def test_locomo_neutral_prompts_are_shared_and_keep_the_user_message_minimal() -> None:
    manager = PromptManager("locomo", method="embedding_rag", language="en")
    user_prompts = {
        query_type: manager.format_query(
            "What happened?", query_type, prompt_protocol="neutral"
        )
        for query_type in LOCOMO_TYPES
    }
    system_prompts = {
        manager.get_query_system_prompt(prompt_protocol="neutral")
        for _query_type in LOCOMO_TYPES
    }

    assert len(set(user_prompts.values())) == len(system_prompts) == 1
    assert user_prompts["adversarial"] == (
        "the retrieved relevant records\n\nQuestion: What happened?\n\nAnswer:"
    )
    assert "Return only the minimal final answer" not in user_prompts["adversarial"]
    neutral = next(iter(system_prompts)) + "\n" + user_prompts["adversarial"]
    assert "No information available" not in neutral
    assert "20 February 2030" not in neutral
    assert all(label not in neutral.lower() for label in FORBIDDEN_NEUTRAL_LABELS)


def test_medmemorybench_type_aware_prompts_are_unchanged() -> None:
    english = PromptManager("medmemorybench", method="embedding_rag", language="en")
    chinese = PromptManager("medmemorybench", method="embedding_rag", language="zh")
    for query_type in MED_TYPES:
        english_actual = english.format_query("What changed?", query_type)
        english_expected = QA_TEMPLATES[f"medmemorybench_en_{query_type}_qa"].format(
            question="What changed?", memory_source="the retrieved relevant dialogue records"
        )
        chinese_actual = chinese.format_query("发生了什么变化？", query_type)
        chinese_expected = QA_TEMPLATES[f"medmemorybench_{query_type}_qa"].format(
            question="发生了什么变化？", memory_source="Retrieved related conversation records"
        )
        assert english_actual == english_expected
        assert chinese_actual == chinese_expected
    for key, expected_hash in TYPE_AWARE_TEMPLATE_HASHES.items():
        if key.startswith("medmemorybench"):
            assert hashlib.sha256(QA_TEMPLATES[key].encode()).hexdigest() == expected_hash


def test_medmemorybench_neutral_prompts_are_language_aware_and_shared() -> None:
    english = PromptManager("medmemorybench", method="embedding_rag", language="en")
    chinese = PromptManager("medmemorybench", method="embedding_rag", language="zh")
    english_user_prompts = {
        english.format_query("Which option applies?", query_type, "neutral")
        for query_type in MED_TYPES
    }
    chinese_user_prompts = {
        chinese.format_query("患者目前情况如何？", query_type, "neutral")
        for query_type in MED_TYPES
    }
    english_system_prompts = {
        english.get_query_system_prompt("neutral") for _query_type in MED_TYPES
    }
    chinese_system_prompts = {
        chinese.get_query_system_prompt("neutral") for _query_type in MED_TYPES
    }

    assert len(english_user_prompts) == len(chinese_user_prompts) == 1
    assert len(english_system_prompts) == len(chinese_system_prompts) == 1
    english_user_prompt = english_user_prompts.pop()
    chinese_user_prompt = chinese_user_prompts.pop()
    assert english_user_prompt == (
        "the retrieved relevant dialogue records\n\nQuestion: Which option applies?\n\nAnswer:"
    )
    assert chinese_user_prompt == "Retrieved related conversation records\n\n问题：患者目前情况如何？\n\n答案："
    assert "Return only the minimal final answer" not in english_user_prompt
    assert "只输出回答问题所必需的最简最终答案" not in chinese_user_prompt
    for prompt in (
        next(iter(english_system_prompts)),
        next(iter(chinese_system_prompts)),
        english_user_prompt,
        chinese_user_prompt,
    ):
        assert all(label not in prompt.lower() for label in FORBIDDEN_NEUTRAL_LABELS)


def test_neutral_system_prompts_define_the_output_contract_without_benchmark_leakage() -> None:
    locomo = NEUTRAL_QUERY_SYSTEM_PROMPTS["locomo"]
    med_en = NEUTRAL_QUERY_SYSTEM_PROMPTS["medmemorybench_en"]
    med_zh = NEUTRAL_QUERY_SYSTEM_PROMPTS["medmemorybench"]

    for prompt in (locomo, med_en):
        assert "shortest complete final answer" in prompt
        assert "restate the question" in prompt
        assert "mention retrieval or evidence" in prompt
        assert "all supported items needed for a complete answer" in prompt
        assert 'answer "Yes" only when the proposition is supported' in prompt
        assert '"No" only when its negation is supported' in prompt
        assert 'answer "Unknown"' in prompt
        assert "best concise" in prompt
        assert "most specific supported expression" in prompt
        assert "requested person-event-attribute relation" in prompt
        assert "resolve relative expressions" in prompt
        assert "human-readable dates" in prompt
        assert "Unknown" in prompt
        assert all(label not in prompt.lower() for label in FORBIDDEN_NEUTRAL_LABELS)
    assert "最短但完整的最终答案" in med_zh
    assert "重述问题" in med_zh
    assert "全部已支持项目" in med_zh
    assert "支持命题为真时回答“是”" in med_zh
    assert "支持命题为假时回答“否”" in med_zh
    assert "推断、可能性、预测、建议或含义" in med_zh
    assert "最具体且有支持的表达" in med_zh
    assert "人物、事件与属性之间的关系" in med_zh
    assert "相对时间换算为对应的绝对时间" in med_zh
    assert "不要使用 ISO 格式" in med_zh
    assert all(label not in med_zh.lower() for label in FORBIDDEN_NEUTRAL_LABELS)
    assert med_zh.endswith("Unknown")


class _RecordingAnswerManager:
    def __init__(self) -> None:
        self.calls = []

    def send_message(self, **kwargs):
        self.calls.append(kwargs)
        return {"output": "answer", "retrieved_memories": []}


def test_neutral_evaluator_keeps_raw_event_state_question_and_scores_by_type() -> None:
    evaluator = LoCoMoEvaluator.__new__(LoCoMoEvaluator)
    evaluator.prompt_protocol = "neutral"
    evaluator.prompt_manager = PromptManager("locomo", method="event_state", language="en")
    evaluator.dry_run = False
    evaluator.agent_manager = _RecordingAnswerManager()
    scored = []
    evaluator._score_agent_response = lambda query, response: scored.append((query, response)) or "score"
    query = LoCoMoQuery(
        query_id="q1",
        question="What was discussed?",
        query_type="temporal",
        expected_answers=["answer"],
        category=2,
    )

    assert evaluator._evaluate_query(query, context_id="conversation") == "score"
    answer_call = evaluator.agent_manager.calls[0]
    assert answer_call["raw_question"] == query.question
    assert "query_type" not in answer_call
    assert answer_call["query_system_prompt"] == evaluator.prompt_manager.get_query_system_prompt("neutral")
    assert "20 February 2030" not in answer_call["message"]
    assert scored[0][0].query_type == "temporal"

    evaluator.prompt_protocol = "type_aware"
    assert evaluator._answer_query_kwargs(query)["query_type"] == "temporal"


def test_neutral_medmemorybench_answer_excludes_gold_metadata_but_scorer_receives_it() -> None:
    evaluator = MedMemoryBenchEvaluator.__new__(MedMemoryBenchEvaluator)
    evaluator.prompt_protocol = "neutral"
    evaluator.prompt_manager = PromptManager("medmemorybench", method="event_state", language="en")
    evaluator.dry_run = False
    evaluator.method_config = SimpleNamespace(method_name="event_state")
    evaluator.agent_manager = _RecordingAnswerManager()
    evaluator._run_api_call = lambda function, *args, **kwargs: function(*args, **kwargs)
    evaluator._query_artifact_references = lambda **kwargs: {}
    scored = []
    evaluator._score_agent_response = lambda query, response, **kwargs: scored.append(query) or "score"
    query = MedQuery(
        query_id="q1",
        question="Which treatment applies?",
        query_type="multiple_choice",
        expected_answers=["GOLD_ANSWER"],
        answers_data=[{"is_correct": True, "content": "GOLD_ANSWER", "explanation": "GOLD_EXPLANATION"}],
        source_key_points=[{"content": "GOLD_SOURCE_KEY_POINT"}],
        metadata={"reasoning_chain": ["GOLD_CHAIN"], "trap_design": "GOLD_TRAP"},
    )

    assert evaluator._evaluate_query(query, context_id=1, unit_id=0) == "score"
    answer_call = evaluator.agent_manager.calls[0]
    assert answer_call["raw_question"] == query.question
    assert "query_type" not in answer_call
    assert answer_call["query_system_prompt"] == evaluator.prompt_manager.get_query_system_prompt("neutral")
    assert all(value not in answer_call["message"] for value in (
        "GOLD_ANSWER", "GOLD_EXPLANATION", "GOLD_SOURCE_KEY_POINT", "GOLD_CHAIN", "GOLD_TRAP",
    ))
    assert scored == [query]
    assert scored[0].answers_data[0]["explanation"] == "GOLD_EXPLANATION"
    assert scored[0].metadata["reasoning_chain"] == ["GOLD_CHAIN"]


def test_prompt_protocol_changes_query_identity_not_snapshot_compatibility() -> None:
    method = MethodConfig.from_dict({"method_name": "event_state", "method_type": "agentic_memory"})
    type_aware = DatasetConfig.from_dict({"dataset_name": "locomo"})
    neutral = DatasetConfig.from_dict({
        "dataset_name": "locomo",
        "evaluation": {"prompt_protocol": "neutral"},
    })

    assert compute_query_config_hash(method, type_aware) != compute_query_config_hash(method, neutral)
    assert compute_memory_query_compatibility_hash(method, type_aware) == compute_memory_query_compatibility_hash(method, neutral)


def test_artifacts_record_the_resolved_prompt_protocol(tmp_path) -> None:
    collector = ResultCollector()
    collector.add_result(MetricResult(
        query_id="q1", query_type="single_hop", score=1.0, is_correct=True,
        model_output="answer", expected_answer="answer", question="question",
    ))
    report = EvaluationReport(
        method_name="method", model_name="model", dataset_name="locomo",
        start_time="2026-01-01T00:00:00", end_time="2026-01-01T00:00:01",
        duration_seconds=1.0, summary={"total": 1, "overall_avg_score": 1.0},
        detailed_results=[collector.get_all_results()[0].to_dict()],
        metadata={"prompt_protocol": "neutral", "query_type_aware_prompting": False},
    )

    paths = collector.save_reports(report, tmp_path, [], use_method_subdir=False)
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["prompt_protocol"] == "neutral"
        assert payload["query_type_aware_prompting"] is False


def test_run_config_records_the_resolved_prompt_protocol(tmp_path) -> None:
    method = MethodConfig.from_dict({"method_name": "method", "method_type": "rag"})
    dataset = DatasetConfig.from_dict({
        "dataset_name": "locomo",
        "evaluation": {"prompt_protocol": "neutral"},
    })
    evaluator = Evaluator(
        method_config=method,
        dataset_config=dataset,
        output_dir=tmp_path,
        method_config_name="method",
        dataset_config_name="locomo",
        verbose=False,
    )

    payload = json.loads((evaluator.output_dir / "run_config.json").read_text(encoding="utf-8"))
    assert payload["prompt_protocol"] == "neutral"
    assert payload["query_type_aware_prompting"] is False
    assert payload["dataset_config"]["prompt_protocol"] == "neutral"
