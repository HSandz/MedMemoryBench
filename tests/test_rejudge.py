"""Offline tests for re-running MedMemoryBench LLM judges."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.medmemorybench import rejudge
from metrics import MetricResult
from src.config import ConfigLoader
from utils.vertex_batch import BatchChatResponse, VertexBatchPending


class _FakeCalculator:
    calls = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def get_metric_name(self, query_type):
        return "llm_judge" if query_type == "state_update" else "string_contain"

    def compute(self, **kwargs):
        self.calls.append(kwargs)
        return MetricResult(
            query_id=kwargs["query_id"],
            query_type=kwargs["query_type"],
            score=1.0,
            is_correct=True,
            model_output=kwargs["model_output"],
            expected_answer=kwargs["expected_answers"][0],
            question=kwargs["question"],
            details={"judge_reason": "new judgement", "metric": "llm_judge"},
        )


class _FakeBatchCalculator(_FakeCalculator):
    def prepare_batch(self, **kwargs):
        return {
            "metric_name": "llm_judge",
            "prepared": {
                "query_id": kwargs["query_id"],
                "query_type": kwargs["query_type"],
                "model_output": kwargs["model_output"],
                "expected_answer": kwargs["expected_answers"][0],
                "question": kwargs["question"],
                "judge_payload": {
                    "prompt": f"Judge: {kwargs['model_output']}",
                    "max_tokens": 500,
                },
            },
        }

    def get_batch_judge_client(self, query_type):
        return SimpleNamespace(model="judge-model")

    def finalize_batch(self, prepared_metric, result_text):
        prepared = prepared_metric["prepared"]
        return MetricResult(
            query_id=prepared["query_id"],
            query_type=prepared["query_type"],
            score=1.0,
            is_correct=True,
            model_output=prepared["model_output"],
            expected_answer=prepared["expected_answer"],
            question=prepared["question"],
            details={"judge_reason": result_text, "metric": "llm_judge"},
        )


class _FakeBatchClient:
    calls = []

    @classmethod
    def from_gemini_client(cls, client, **kwargs):
        cls.calls.append(kwargs)
        return cls()

    def run_stage(self, stage, requests):
        return {
            request.request_id: BatchChatResponse(
                request_id=request.request_id,
                content="batch judgement",
            )
            for request in requests
        }


def _write_dataset(project_root):
    eval_dir = project_root / "data" / "test_medmemorybench" / "persona_1" / "eval"
    eval_dir.mkdir(parents=True)
    (eval_dir / "generated_dialogues.json").write_text(
        json.dumps({"sessions": []}), encoding="utf-8"
    )
    (eval_dir / "generated_queries.json").write_text(
        json.dumps({
            "queries": [
                {
                    "query_id": "same-id",
                    "question": "What changed?",
                    "query_type": "state_update",
                    "answers": [
                        {
                            "content": "The dose increased",
                            "is_correct": True,
                            "explanation": "The latest session increased it.",
                        }
                    ],
                    "metadata": {"source": "latest session"},
                },
                {
                    "query_id": "local-id",
                    "question": "What drug?",
                    "query_type": "entity_exact_match",
                    "answers": [{"content": "Drug A", "is_correct": True}],
                },
            ]
        }),
        encoding="utf-8",
    )
    config_dir = project_root / "configs" / "dataset_config"
    config_dir.mkdir(parents=True)
    (config_dir / "medmemorybench.yaml").write_text(
        """dataset_name: medmemorybench
language: en
data:
  root_dir: data/test_medmemorybench
evaluation:
  mode: independent
  inject_noise: false
query_types: []
""",
        encoding="utf-8",
    )


def _write_source(path):
    path.write_text(
        json.dumps({
            "method_name": "method",
            "model_name": "answer-model",
            "dataset_name": "medmemorybench",
            "summary": {"correct_count": 0},
            "queries": [
                {
                    "query_id": "same-id",
                    "query_type": "state_update",
                    "question": "What changed?",
                    "expected_answer": "The dose increased",
                    "model_output": "It was increased.",
                    "score": 0.0,
                    "is_correct": False,
                    "retrieved_memories": [{"text": "memory"}],
                    "retrieved_count": 1,
                    "query_time": 2.5,
                    "evaluation_details": {"judge_reason": "old judgement"},
                },
                {
                    "query_id": "local-id",
                    "query_type": "entity_exact_match",
                    "question": "What drug?",
                    "expected_answer": "Drug A",
                    "model_output": "Drug A",
                    "score": 1.0,
                    "is_correct": True,
                    "retrieved_memories": [],
                    "retrieved_count": 0,
                    "query_time": 1.0,
                    "evaluation_details": {"metric": "string_contain"},
                },
            ],
        }),
        encoding="utf-8",
    )


def test_rejudge_reuses_answers_and_writes_incremented_file(tmp_path, monkeypatch):
    _FakeCalculator.calls = []
    _write_dataset(tmp_path)
    source_path = tmp_path / "run_query_answer.json"
    _write_source(source_path)
    (tmp_path / "run_query_answer_rejudge_1.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(rejudge, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(rejudge, "MetricsCalculator", _FakeCalculator)
    monkeypatch.setattr(
        rejudge,
        "get_api_config",
        lambda: SimpleNamespace(
            judge_model="new-judge",
            judge_api_key="key",
            judge_base_url="",
            get_judge_provider=lambda: "openai",
            get_judge_model=lambda: "new-judge",
        ),
    )

    output_path, output_data = rejudge.rejudge_medmemorybench(
        source_path,
        config_loader=ConfigLoader(project_root=tmp_path),
        verbose=False,
    )

    assert output_path.name == "run_query_answer_rejudge_2.json"
    assert output_data["queries"][0]["model_output"] == "It was increased."
    assert output_data["queries"][0]["evaluation_details"]["judge_reason"] == "new judgement"
    assert output_data["queries"][1]["evaluation_details"] == {"metric": "string_contain"}
    assert output_data["rejudge"]["rejudged_queries"] == 1
    assert output_data["rejudge"]["preserved_local_metric_queries"] == 1
    assert output_data["by_context"]["1"]["correct"] == 2
    assert _FakeCalculator.calls[0]["answers_data"][0]["explanation"].startswith("The latest")
    assert _FakeCalculator.calls[0]["metadata"] == {"source": "latest session"}


def test_batch_rejudge_uses_vertex_batch_and_preserves_query_order(tmp_path, monkeypatch):
    _write_dataset(tmp_path)
    source_path = tmp_path / "run_query_answer.json"
    _write_source(source_path)
    _FakeBatchClient.calls = []

    monkeypatch.setattr(rejudge, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(rejudge, "MetricsCalculator", _FakeBatchCalculator)
    monkeypatch.setattr(rejudge, "VertexBatchClient", _FakeBatchClient)
    monkeypatch.setattr(
        rejudge,
        "get_api_config",
        lambda: SimpleNamespace(
            judge_model="judge-model",
            judge_api_key="",
            judge_base_url="",
            get_judge_provider=lambda: "vertex",
            get_judge_model=lambda: "judge-model",
        ),
    )

    output_path, output_data = rejudge.rejudge_medmemorybench(
        source_path,
        config_loader=ConfigLoader(project_root=tmp_path),
        verbose=False,
        batch_api=True,
        batch_gcs_uri="gs://private-bucket/rejudge",
        batch_wait=True,
    )

    assert output_path.name == "run_query_answer_rejudge_1.json"
    assert [query["query_id"] for query in output_data["queries"]] == ["same-id", "local-id"]
    assert output_data["queries"][0]["evaluation_details"]["judge_reason"] == "batch judgement"
    assert output_data["queries"][1]["evaluation_details"] == {"metric": "string_contain"}
    assert output_data["rejudge"]["batch_api"] is True
    assert output_data["rejudge"]["batch_requests"] == 1
    assert _FakeBatchClient.calls[0]["wait"] is True
    assert not rejudge._batch_state_path(source_path.resolve()).exists()


def test_batch_rejudge_resume_reuses_state_and_manifest_scope(tmp_path, monkeypatch):
    _write_dataset(tmp_path)
    source_path = tmp_path / "run_query_answer.json"
    _write_source(source_path)
    calls = []

    class PendingBatchClient(_FakeBatchClient):
        @classmethod
        def from_gemini_client(cls, client, **kwargs):
            calls.append(kwargs)
            return cls()

        def run_stage(self, stage, requests):
            if len(calls) == 1:
                raise VertexBatchPending(stage, "jobs/one", calls[-1]["manifest_path"])
            return super().run_stage(stage, requests)

    monkeypatch.setattr(rejudge, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(rejudge, "MetricsCalculator", _FakeBatchCalculator)
    monkeypatch.setattr(rejudge, "VertexBatchClient", PendingBatchClient)
    monkeypatch.setattr(
        rejudge,
        "get_api_config",
        lambda: SimpleNamespace(
            judge_model="judge-model",
            judge_api_key="",
            judge_base_url="",
            get_judge_provider=lambda: "vertex",
            get_judge_model=lambda: "judge-model",
        ),
    )

    with pytest.raises(VertexBatchPending):
        rejudge.rejudge_medmemorybench(
            source_path,
            config_loader=ConfigLoader(project_root=tmp_path),
            verbose=False,
            batch_api=True,
            batch_gcs_uri="gs://private-bucket/rejudge",
        )

    state_path = rejudge._batch_state_path(source_path.resolve())
    state = json.loads(state_path.read_text())
    output_path, _ = rejudge.rejudge_medmemorybench(
        source_path,
        config_loader=ConfigLoader(project_root=tmp_path),
        verbose=False,
        batch_api=True,
        batch_gcs_uri="gs://private-bucket/rejudge",
        batch_wait=True,
        resume=True,
    )

    assert output_path == Path(state["output_path"])
    assert calls[0]["manifest_path"] == calls[1]["manifest_path"]
    assert not state_path.exists()


def test_rejudge_rejects_non_query_answer_file(tmp_path):
    source_path = tmp_path / "result.json"
    source_path.write_text(json.dumps({"dataset_name": "medmemorybench"}), encoding="utf-8")

    try:
        rejudge.rejudge_medmemorybench(source_path, verbose=False)
    except ValueError as exc:
        assert "queries list" in str(exc)
    else:
        raise AssertionError("Expected invalid rejudge input to fail")


def test_rejudge_hydrates_compact_batch_and_sidecar_retrieval(tmp_path):
    source_path = tmp_path / "run_query_answer.json"
    batch_dir = tmp_path / "batch"
    batch_dir.mkdir()
    manifest_path = batch_dir / "answers.json"
    manifest_path.write_text(json.dumps({
        "jobs": {
            "query-final": {
                "requests": [{
                    "request_id": "batch-request",
                    "metadata": {
                        "prepared_query": {
                            "retrieved_memories": [{"memory_id": "batch-memory"}],
                        }
                    },
                }]
            }
        }
    }), encoding="utf-8")
    records_path = tmp_path / "run_retrieval_records.json"
    records_path.write_text(json.dumps({
        "records": [{
            "query_id": "realtime-query",
            "retrieved_memories": [{"memory_id": "realtime-memory"}],
        }]
    }), encoding="utf-8")
    source_data = {
        "format": "medmemorybench.query_answers",
        "version": 2,
        "retrieval_records_path": records_path.name,
    }
    queries = [
        {
            "query_id": "batch-query",
            "retrieval_reference": {
                "source": "answer_batch_manifest",
                "manifest_path": "batch/answers.json",
                "request_id": "batch-request",
            },
        },
        {
            "query_id": "realtime-query",
            "retrieval_reference": {
                "source": "retrieval_records",
                "record_id": "realtime-query",
            },
        },
    ]

    rejudge._hydrate_retrieved_memories(source_path, source_data, queries)

    assert queries[0]["retrieved_memories"] == [{"memory_id": "batch-memory"}]
    assert queries[1]["retrieved_memories"] == [{"memory_id": "realtime-memory"}]


def test_rejudge_output_number_uses_original_stem(tmp_path):
    first_rejudge = tmp_path / "run_query_answer_rejudge_1.json"
    first_rejudge.write_text("{}", encoding="utf-8")

    output_path, run_number = rejudge._next_output_path(first_rejudge)

    assert output_path.name == "run_query_answer_rejudge_2.json"
    assert run_number == 2
