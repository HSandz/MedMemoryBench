"""Offline tests for Vertex Batch staging and judge result handling."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from metrics.llm_judge import LLMJudgeMCDMetric, LLMJudgeMetric
from benchmarks.medmemorybench.evaluator import MedMemoryBenchEvaluator
from benchmarks.medmemorybench.smart_mem0_batch_integration import (
    _evaluation_llm_usage,
    _method_llm_usage,
)
from src.evaluator import Evaluator
from utils.llm_client import get_usage_tracker
from utils.vertex_batch import (
    BatchChatRequest,
    PREPARED_QUERY_METADATA_KEY,
    VertexBatchClient,
    VertexBatchError,
    VertexBatchPending,
    make_request_id,
    restore_prepared_query,
    scoped_manifest_path,
    snapshot_prepared_query,
)


class _Blob:
    def __init__(self, storage, name: str, text: str = ""):
        self._storage = storage
        self.name = name
        self._text = text

    def upload_from_string(self, body: str, content_type: str) -> None:
        self._storage.uploads[self.name] = body

    def download_as_text(self, encoding: str) -> str:
        return self._text


class _Bucket:
    def __init__(self, storage):
        self._storage = storage

    def blob(self, name: str) -> _Blob:
        return _Blob(self._storage, name)


class _Storage:
    def __init__(self):
        self.uploads = {}
        self.outputs = {}

    def bucket(self, name: str) -> _Bucket:
        return _Bucket(self)

    def list_blobs(self, bucket_name: str, prefix: str):
        return [
            _Blob(self, f"{prefix}/predictions.jsonl", text)
            for output_prefix, text in self.outputs.items()
            if output_prefix == f"gs://{bucket_name}/{prefix}"
        ]


def _success_row(request: dict, content: str = "answer") -> dict:
    """Match Vertex's documented GCS output: request/status/response, no ID."""
    return {
        "status": "",
        "request": {"contents": request["contents"]},
        "response": {
            "candidates": [{"content": {"parts": [{"text": content}]}}],
            "usageMetadata": {"promptTokenCount": 7, "candidatesTokenCount": 3},
        },
    }


class _Batches:
    def __init__(self, storage: _Storage, partial_first_job: bool = False):
        self.storage = storage
        self.partial_first_job = partial_first_job
        self.created = []
        self.states = {}

    def create(self, *, model: str, src: str, config):
        name = f"jobs/{len(self.created) + 1}"
        self.created.append((name, model, src, config.dest))
        self.states[name] = "JOB_STATE_SUCCEEDED"
        request_rows = [json.loads(line) for line in self.storage.uploads[src[5:].split('/', 1)[1]].splitlines()]
        rows = request_rows[:1] if self.partial_first_job and len(self.created) == 1 else request_rows
        self.storage.outputs[config.dest] = "\n".join(
            json.dumps(_success_row(
                row["request"],
                f"answer-{row['request']['contents'][-1]['parts'][0]['text']}",
            ))
            for row in rows
        ) + "\n"
        return SimpleNamespace(name=name, state=self.states[name])

    def get(self, *, name: str):
        return SimpleNamespace(name=name, state=self.states[name])


class _GenAI:
    def __init__(self, batches: _Batches):
        self.batches = batches


def _client(
    tmp_path: Path,
    batches: _Batches,
    storage: _Storage,
    *,
    wait: bool = True,
    progress_callback=None,
):
    return VertexBatchClient(
        model="gemini-2.5-flash",
        project="test-project",
        location="global",
        credentials=object(),
        gcs_uri="gs://private-bucket/evaluation-batch",
        manifest_path=tmp_path / "batch_manifest.json",
        wait=wait,
        config_hash="config-hash",
        genai_client=_GenAI(batches),
        storage_client=storage,
        progress_callback=progress_callback,
    )


def _request(request_id: str, text: str = None) -> BatchChatRequest:
    return BatchChatRequest(
        request_id=request_id,
        messages=[{"role": "system", "content": "be concise"}, {"role": "user", "content": text or request_id}],
        temperature=0.2,
        max_tokens=55,
        response_format={"type": "json_object"},
        phase="query",
    )


def test_smart_mem0_usage_separates_method_and_evaluator_calls():
    evaluator = SimpleNamespace(
        aggregator=SimpleNamespace(
            results=[
                SimpleNamespace(
                    details={
                        "agent_telemetry": {
                            "method_llm_calls": {"controller": 1, "answer": 1, "total": 2},
                            "query_tokens": {"controller": 20, "answer": 30, "total": 50},
                        }
                    }
                ),
                SimpleNamespace(
                    details={
                        "agent_telemetry": {
                            "method_llm_calls": {"controller": 1, "answer": 0, "total": 1},
                            "query_tokens": {"controller": 25, "answer": 0, "total": 25},
                        }
                    }
                ),
            ]
        )
    )

    method = _method_llm_usage(evaluator)
    benchmark = _evaluation_llm_usage(
        {"query_phase": {"call_count": 5, "total_tokens": 100}}, method
    )

    assert method["total_calls"] == 3
    assert method["total_tokens"] == 75
    assert method["tokens_per_query"] == {
        "mean": 37.5,
        "median": 37.5,
        "p90": 50,
        "p95": 50,
        "max": 50,
    }
    assert benchmark["call_count"] == 2
    assert benchmark["total_tokens"] == 25


def test_jsonl_mapping_and_output_parsing(tmp_path):
    storage = _Storage()
    batches = _Batches(storage)
    client = _client(tmp_path, batches, storage)
    get_usage_tracker().reset()

    response = client.run_stage("query-unit-1", [_request("opaque-1")])["opaque-1"]

    uploaded = next(iter(storage.uploads.values()))
    record = json.loads(uploaded)
    assert set(record) == {"request"}
    assert record["request"]["systemInstruction"]["parts"][0]["text"] == "be concise"
    assert record["request"]["generationConfig"]["responseMimeType"] == "application/json"
    assert response.content == "answer-opaque-1"
    assert (response.input_tokens, response.output_tokens, response.status) == (7, 3, "")
    stats = get_usage_tracker().get_stats()["query_phase"]
    assert (stats["input_tokens"], stats["output_tokens"], stats["total_latency"]) == (7, 3, 0.0)


def test_json_schema_mapping_and_saved_request_restore(tmp_path):
    storage = _Storage()
    batches = _Batches(storage)
    client = _client(tmp_path, batches, storage)
    schema = {
        "type": "object",
        "properties": {"concepts_list": {"type": "array", "items": {"type": "string"}}},
        "required": ["concepts_list"],
    }
    request = BatchChatRequest(
        request_id="structured",
        messages=[{"role": "user", "content": "extract concepts"}],
        temperature=0.7,
        max_tokens=100,
        response_format={"type": "json_object", "response_schema": schema},
        metadata={"batch_request_time": "2026-08-02 01:02:03"},
    )

    client.run_stage("concepts", [request])

    uploaded = json.loads(next(iter(storage.uploads.values())))
    config = uploaded["request"]["generationConfig"]
    assert config["responseMimeType"] == "application/json"
    assert config["responseSchema"] == schema
    restored = client.get_saved_request("concepts", "structured")
    assert restored is not None
    assert restored.messages == request.messages
    assert restored.metadata == request.metadata


def test_progress_callback_reports_submission_and_collection(tmp_path):
    storage = _Storage()
    batches = _Batches(storage)
    progress = []
    client = _client(tmp_path, batches, storage, progress_callback=progress.append)

    client.run_stage("query-unit-1", [_request("opaque-1")])

    joined = "\n".join(progress)
    assert "preparing 1 request(s)" in joined
    assert "Uploading" in joined
    assert "submitted jobs/1" in joined
    assert "collected 1/1 output row(s)" in joined
    assert "finalized 1/1 request(s)" in joined


def test_progress_callback_reports_each_wait_poll(tmp_path):
    class PollingBatches(_Batches):
        def __init__(self, storage):
            super().__init__(storage)
            self.get_calls = 0

        def create(self, **kwargs):
            job = super().create(**kwargs)
            self.states[job.name] = "JOB_STATE_PENDING"
            return SimpleNamespace(name=job.name, state="JOB_STATE_PENDING")

        def get(self, *, name: str):
            self.get_calls += 1
            if self.get_calls >= 3:
                self.states[name] = "JOB_STATE_SUCCEEDED"
            return super().get(name=name)

    storage = _Storage()
    batches = PollingBatches(storage)
    progress = []
    client = _client(tmp_path, batches, storage, progress_callback=progress.append)
    client.poll_interval = 0

    client.run_stage("query-unit-1", [_request("opaque-1")])

    joined = "\n".join(progress)
    assert "polling every 0s" in joined
    assert "poll 1; jobs/1 is JOB_STATE_SUCCEEDED" in joined


def test_empty_stage_is_cloud_free(tmp_path):
    storage = _Storage()
    batches = _Batches(storage)
    client = _client(tmp_path, batches, storage)

    assert client.run_stage("unused", []) == {}
    assert batches.created == []
    assert storage.uploads == {}


@pytest.mark.parametrize("method_name", ["amem", "mem0", "memos", "memrl", "mirix"])
def test_new_gemini_batch_methods_pass_eligibility_check(method_name):
    evaluator = Evaluator.__new__(Evaluator)
    evaluator.method_config = SimpleNamespace(
        method_name=method_name,
        model=SimpleNamespace(provider="gemini"),
        raw_config={"agent_params": {"use_native_query": False}},
    )
    evaluator.dataset_config = SimpleNamespace(dataset_name="locomo")
    evaluator.batch_gcs_uri = "gs://private-bucket/evaluation-batch"

    evaluator._validate_batch_eligibility()


def test_native_mirix_is_not_an_eligible_final_answer_batch_stage():
    evaluator = Evaluator.__new__(Evaluator)
    evaluator.method_config = SimpleNamespace(
        method_name="mirix",
        model=SimpleNamespace(provider="gemini"),
        raw_config={"agent_params": {"use_native_query": True}},
    )
    evaluator.dataset_config = SimpleNamespace(dataset_name="locomo")
    evaluator.batch_gcs_uri = "gs://private-bucket/evaluation-batch"

    with pytest.raises(ValueError, match="at least one eligible Gemini stage"):
        evaluator._validate_batch_eligibility()


def test_submit_only_resume_does_not_resubmit(tmp_path):
    storage = _Storage()
    batches = _Batches(storage)
    client = _client(tmp_path, batches, storage, wait=False)
    batches.states["jobs/1"] = "JOB_STATE_PENDING" if batches.created else "JOB_STATE_PENDING"

    # The fake job starts as succeeded; force it pending as soon as it is created.
    original_create = batches.create

    def create_pending(**kwargs):
        job = original_create(**kwargs)
        batches.states[job.name] = "JOB_STATE_PENDING"
        return SimpleNamespace(name=job.name, state="JOB_STATE_PENDING")

    batches.create = create_pending
    with pytest.raises(VertexBatchPending):
        client.run_stage("query-unit-1", [_request("opaque-1")])

    assert len(batches.created) == 1
    batches.states["jobs/1"] = "JOB_STATE_SUCCEEDED"
    resumed = client.run_stage("query-unit-1", [_request("opaque-1")])
    assert len(batches.created) == 1
    assert resumed["opaque-1"].content == "answer-opaque-1"


def test_partial_output_retries_only_missing_request(tmp_path):
    storage = _Storage()
    batches = _Batches(storage, partial_first_job=True)
    client = _client(tmp_path, batches, storage)
    results = client.run_stage("query-unit-1", [_request("one"), _request("two")])

    assert set(results) == {"one", "two"}
    assert len(batches.created) == 2
    retry_input = storage.uploads[batches.created[1][2][5:].split('/', 1)[1]]
    assert [
        json.loads(line)["request"]["contents"][-1]["parts"][0]["text"]
        for line in retry_input.splitlines()
    ] == ["two"]
    manifest = json.loads((tmp_path / "batch_manifest.json").read_text())
    assert manifest["jobs"]["query-unit-1"]["missing_request_ids"] == []


def test_duplicate_message_content_is_rejected_before_submission(tmp_path):
    storage = _Storage()
    batches = _Batches(storage)
    client = _client(tmp_path, batches, storage)

    with pytest.raises(VertexBatchError, match="duplicate message content"):
        client.run_stage("query-unit-1", [_request("one", "same"), _request("two", "same")])

    assert batches.created == []
    assert storage.uploads == {}


def test_scoped_manifest_path_isolates_configs_and_reuses_matching_legacy_file(tmp_path):
    legacy = tmp_path / "shared_manifest.json"
    legacy.write_text(json.dumps({
        "model": "gemini-2.5-flash",
        "config_hash": "config-a",
    }))

    assert scoped_manifest_path(
        tmp_path, "shared_manifest", model="gemini-2.5-flash", config_hash="config-a"
    ) == legacy
    isolated = scoped_manifest_path(
        tmp_path, "shared_manifest", model="gemini-2.5-flash", config_hash="config-b"
    )
    assert isolated != legacy
    assert isolated.name == "shared_manifest-gemini-2.5-flash-configb.json"


def test_prepared_query_snapshot_is_json_safe_and_bound_to_staged_messages():
    prepared = {
        "messages": [{"role": "user", "content": "immutable prompt"}],
        "retrieved_count": 1,
        "retrieved_memories": [{"memory": "fact", "captured_at": object()}],
    }
    snapshot = snapshot_prepared_query(prepared)
    request = BatchChatRequest(
        request_id="saved",
        messages=prepared["messages"],
        temperature=0.3,
        max_tokens=100,
        metadata={PREPARED_QUERY_METADATA_KEY: snapshot},
    )

    assert restore_prepared_query(request) == snapshot
    request.messages = [{"role": "user", "content": "different prompt"}]
    assert restore_prepared_query(request) is None


def test_medmemorybench_resume_reuses_saved_prepared_query_without_retrieval():
    prepared = {
        "messages": [{"role": "user", "content": "original prompt"}],
        "retrieved_count": 1,
        "retrieved_memories": [{"memory": "original memory"}],
        "extra": {"method": "memrl"},
    }
    request_id = make_request_id("query", "memrl:1:q1")
    saved_request = BatchChatRequest(
        request_id=request_id,
        messages=prepared["messages"],
        temperature=0.3,
        max_tokens=100,
        metadata={PREPARED_QUERY_METADATA_KEY: snapshot_prepared_query(prepared)},
    )

    class _BatchClient:
        def __init__(self):
            self.submitted = []

        def get_saved_request(self, stage, request_id):
            assert (stage, request_id) == ("query-unit-1", saved_request.request_id)
            return saved_request

        def run_stage(self, stage, requests):
            self.submitted = requests
            return {
                saved_request.request_id: SimpleNamespace(
                    content="answer", status="", input_tokens=7, output_tokens=3
                )
            }

    class _AgentManager:
        def __init__(self):
            self.prepared = 0
            self.finalized = None

        def prepare_batch_query(self, *args, **kwargs):
            self.prepared += 1
            raise AssertionError("resume must not repeat retrieval")

        def finalize_batch_query(self, batch_prepared, content, **kwargs):
            self.finalized = (batch_prepared, content, kwargs)
            return SimpleNamespace(output=content)

    batch_client = _BatchClient()
    manager = _AgentManager()
    evaluator = MedMemoryBenchEvaluator.__new__(MedMemoryBenchEvaluator)
    evaluator._checkpoint_manager = None
    evaluator._is_deferred_judge_query = lambda query_id: False
    evaluator._get_batch_client = lambda: batch_client
    evaluator._log = lambda message: None
    evaluator.agent_manager = manager
    evaluator.method_config = SimpleNamespace(
        method_name="memrl",
        model=SimpleNamespace(temperature=0.3, max_completion_tokens=100, max_tokens=100),
    )
    evaluator._score_agent_response = lambda query, response, **kwargs: query.query_id
    query = SimpleNamespace(query_id="q1", question="question", query_type="state_update")
    unit = SimpleNamespace(unit_id=1, context_id=1, queries_to_evaluate=[query])

    results = evaluator._evaluate_batch_queries(unit, memory_time_per_query=0.0)

    assert results == ["q1"]
    assert manager.prepared == 0
    assert manager.finalized[0] == prepared
    assert manager.finalized[1] == "answer"
    assert manager.finalized[2] == {"input_tokens": 7, "output_tokens": 3}
    assert batch_client.submitted == [saved_request]


def test_deferred_judge_work_is_persisted_for_resume(tmp_path):
    evaluator = MedMemoryBenchEvaluator.__new__(MedMemoryBenchEvaluator)
    evaluator.output_dir = tmp_path
    evaluator.method_config = SimpleNamespace(raw_config={"method_name": "graph_rag"})
    evaluator.dataset_config = SimpleNamespace(raw_config={"dataset_name": "medmemorybench"})
    evaluator._deferred_judges = [{
        "query_id": "q1",
        "persona_id": 1,
        "prepared_metric": {"prepared": {"query_type": "state_update"}},
        "query_time": 0.0,
        "memory_construction_time": 0.0,
        "retrieved_memories": [],
        "retrieved_count": 0,
    }]
    evaluator._log = lambda message: None

    evaluator._save_deferred_judges(SimpleNamespace(model="gemini-2.5-flash-lite"))
    evaluator._deferred_judges = []
    evaluator._load_deferred_judges()

    assert evaluator._is_deferred_judge_query("q1")
    evaluator._clear_deferred_judges()
    assert evaluator._deferred_judges == []
    assert not list((tmp_path / "batch").glob("medmemorybench_deferred_judges-*.json"))


def test_judge_prepare_finalize_preserves_metric_schema():
    judge = LLMJudgeMetric(language="en")
    prepared = judge.prepare_batch(
        query_id="q1",
        query_type="state_update",
        model_output="The medication was stopped.",
        expected_answers=["It was stopped."],
        question="What changed?",
    )
    result = judge.finalize_batch(prepared, '{"is_correct": true, "reason": "matches"}')
    assert result.score == 1.0
    assert result.is_correct is True
    assert result.details["judge_reason"] == "matches"

    mcd = LLMJudgeMCDMetric(language="en")
    prepared_mcd = mcd.prepare_batch(
        query_id="q2",
        query_type="multi_hop_clinical_deduction",
        model_output="Answer",
        expected_answers=["Answer"],
        question="Question",
    )
    result_mcd = mcd.finalize_batch(
        prepared_mcd,
        json.dumps({
            "is_correct": True,
            "ncr_score": 1.0,
            "crc_score": 1.0,
            "cc_score": 1.0,
            "uses_patient_specific_info": True,
            "memory_retrieval_quality": "excellent",
        }),
    )
    assert result_mcd.score == 1.0
    assert result_mcd.details["metric"] == "llm_judge_mcd"
