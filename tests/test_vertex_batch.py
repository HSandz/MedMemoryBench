"""Offline tests for Vertex Batch staging and judge result handling."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from metrics.llm_judge import LLMJudgeMCDMetric, LLMJudgeMetric
from metrics.base import MetricResult
from benchmarks.medmemorybench.evaluator import MedMemoryBenchEvaluator
from src.evaluator import Evaluator
from utils.llm_client import LLMResponse, OpenRouterClient, get_usage_tracker
from utils.openrouter_batch import OpenRouterBatchClient, OpenRouterBatchPending
from utils.vertex_batch import (
    BatchChatRequest,
    BatchChatResponse,
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


def test_vertex_batch_usage_separates_thought_tokens_from_visible_output():
    response = VertexBatchClient._parse_row(
        {
            "response": {
                "usageMetadata": {
                    "promptTokenCount": 7,
                    "candidatesTokenCount": 10,
                    "thoughtsTokenCount": 6,
                },
                "candidates": [{"content": {"parts": [{"text": "answer"}]}}],
            }
        },
        "request-1",
    )

    assert (response.output_tokens, response.visible_output_tokens, response.thinking_tokens) == (10, 4, 6)


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


class _DirectClient:
    def __init__(self):
        self.calls = []

    def chat(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        text = messages[-1]["content"]
        return LLMResponse(
            content=f"direct-{text}",
            input_tokens=5,
            output_tokens=2,
            model="gemini-2.5-flash",
            raw_response={"transport": "direct"},
        )


def _client(
    tmp_path: Path,
    batches: _Batches,
    storage: _Storage,
    *,
    wait: bool = True,
    progress_callback=None,
    direct_client=None,
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
        direct_client=direct_client,
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


def test_batch_control_operations_use_vertex_credential_pool(tmp_path):
    attempts = []

    class CredentialPool:
        def _run_with_service_account_rotation(self, operation, operation_name):
            for project in ("project-one", "project-two"):
                try:
                    return operation(None, object(), project)
                except PermissionError:
                    continue
            raise AssertionError("credential pool was exhausted")

    class Batches:
        def __init__(self, project):
            self.project = project

        def get(self, *, name):
            attempts.append((self.project, name))
            if self.project == "project-one":
                raise PermissionError("permission denied")
            return SimpleNamespace(name=name, state="JOB_STATE_SUCCEEDED")

    client = VertexBatchClient(
        model="gemini-2.5-flash",
        project="project-one",
        location="global",
        credentials=object(),
        gcs_uri="gs://private-bucket/evaluation-batch",
        manifest_path=tmp_path / "batch_manifest.json",
        wait=True,
        credential_client=CredentialPool(),
    )
    client._new_batch_genai_client = lambda credentials, project: _GenAI(Batches(project))

    job = client._get_job("projects/test/locations/global/batchPredictionJobs/1")

    assert job.state == "JOB_STATE_SUCCEEDED"
    assert attempts == [
        ("project-one", "projects/test/locations/global/batchPredictionJobs/1"),
        ("project-two", "projects/test/locations/global/batchPredictionJobs/1"),
    ]
    assert client.project == "project-two"


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
    assert client.has_stage("concepts") is True
    assert [item.request_id for item in client.get_saved_requests("concepts")] == ["structured"]


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


def test_progress_callback_reports_only_poll_state_changes(tmp_path):
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
            if self.get_calls in {4, 5}:
                self.states[name] = "JOB_STATE_QUEUED"
            elif self.get_calls >= 6:
                self.states[name] = "JOB_STATE_SUCCEEDED"
            return super().get(name=name)

    storage = _Storage()
    batches = PollingBatches(storage)
    progress = []
    client = _client(tmp_path, batches, storage, progress_callback=progress.append)
    client.poll_interval = 0

    client.run_stage("query-unit-1", [_request("opaque-1")])

    joined = "\n".join(progress)
    poll_messages = [message for message in progress if ": poll " in message]
    assert "polling every 0s" in joined
    assert poll_messages == [
        "[Vertex Batch] Stage 'query-unit-1': poll 2; jobs/1 is JOB_STATE_QUEUED.",
        "[Vertex Batch] Stage 'query-unit-1': poll 4; jobs/1 is JOB_STATE_SUCCEEDED.",
    ]


def test_empty_stage_is_cloud_free(tmp_path):
    storage = _Storage()
    batches = _Batches(storage)
    client = _client(tmp_path, batches, storage)

    assert client.run_stage("unused", []) == {}
    assert batches.created == []
    assert storage.uploads == {}


def test_small_stage_uses_vertex_batch_when_explicit(tmp_path, monkeypatch):
    monkeypatch.delenv("VERTEX_BATCH_MIN_REQUESTS", raising=False)
    storage = _Storage()
    batches = _Batches(storage)
    direct_client = _DirectClient()
    client = _client(
        tmp_path,
        batches,
        storage,
        direct_client=direct_client,
    )
    requests = [_request("request-0")]

    responses = client.run_stage("small-stage", requests)

    assert len(batches.created) == 1
    assert storage.uploads
    assert (tmp_path / "batch_manifest.json").exists()
    assert direct_client.calls == []
    assert responses["request-0"].content == "answer-request-0"


def test_six_requests_create_a_batch_job(tmp_path, monkeypatch):
    monkeypatch.delenv("VERTEX_BATCH_MIN_REQUESTS", raising=False)
    storage = _Storage()
    batches = _Batches(storage)
    direct_client = _DirectClient()
    client = _client(tmp_path, batches, storage, direct_client=direct_client)

    responses = client.run_stage(
        "batch-stage",
        [_request(f"request-{index}") for index in range(6)],
    )

    assert len(batches.created) == 1
    assert direct_client.calls == []
    assert len(responses) == 6


def test_small_batch_retry_uses_another_batch_job(tmp_path, monkeypatch):
    monkeypatch.delenv("VERTEX_BATCH_MIN_REQUESTS", raising=False)
    storage = _Storage()
    batches = _Batches(storage, partial_first_job=True)
    direct_client = _DirectClient()
    client = _client(tmp_path, batches, storage, direct_client=direct_client)

    responses = client.run_stage(
        "partial-stage",
        [_request(f"request-{index}") for index in range(6)],
    )

    assert len(batches.created) == 2
    assert direct_client.calls == []
    assert len(responses) == 6
    assert responses["request-5"].content == "answer-request-5"


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


def test_openrouter_batch_eligibility_does_not_require_gcs(monkeypatch):
    evaluator = Evaluator.__new__(Evaluator)
    evaluator.method_config = SimpleNamespace(
        method_name="long_context",
        model=SimpleNamespace(provider="openrouter"),
        raw_config={},
    )
    evaluator.dataset_config = SimpleNamespace(dataset_name="locomo")
    evaluator.batch_gcs_uri = None
    monkeypatch.delenv("GOOGLE_BATCH_GCS_URI", raising=False)

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


def test_batch_request_serializes_gemini_thinking_settings():
    request = BatchChatRequest(
        request_id="thinking",
        messages=[{"role": "user", "content": "hello"}],
        temperature=0.2,
        max_tokens=100,
        reasoning_effort="high",
    )
    record = request.to_vertex_record(model="gemini-3-pro-preview")
    assert record["request"]["generationConfig"]["thinkingConfig"] == {
        "thinkingLevel": "HIGH"
    }

    budget_request = BatchChatRequest(
        request_id="budget",
        messages=[{"role": "user", "content": "hello"}],
        temperature=0.2,
        max_tokens=100,
        reasoning_effort=1024,
    )
    assert budget_request.to_vertex_request(model="gemini-2.5-flash")["generationConfig"][
        "thinkingConfig"
    ] == {"thinkingBudget": 1024}


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
        "planner": {"rounds_configured": 1, "retrieval_round_count": 1},
    }]
    evaluator._log = lambda message: None

    evaluator._save_deferred_judges(SimpleNamespace(model="gemini-2.5-flash-lite"))
    evaluator._deferred_judges = []
    evaluator._load_deferred_judges()

    assert evaluator._is_deferred_judge_query("q1")
    assert evaluator._deferred_judges[0]["planner"]["rounds_configured"] == 1
    evaluator._clear_deferred_judges()
    assert evaluator._deferred_judges == []
    assert not list((tmp_path / "batch").glob("medmemorybench_deferred_judges-*.json"))


def test_single_deferred_judge_uses_batch_stage():
    class BatchStageCalled(RuntimeError):
        pass

    captured = {}

    def run_stage(stage, requests):
        captured["stage"] = stage
        captured["requests"] = requests
        raise BatchStageCalled

    judge_client = SimpleNamespace(model="gemini-2.5-flash-lite")
    evaluator = MedMemoryBenchEvaluator.__new__(MedMemoryBenchEvaluator)
    evaluator._deferred_judges = [{
        "query_id": "q1",
        "persona_id": 1,
        "prepared_metric": {"prepared": {
            "query_id": "q1",
            "query_type": "state_update",
            "model_output": "answer",
            "question": "What changed?",
            "judge_payload": {
                "prompt": "judge this answer",
                "temperature": 0.2,
                "max_tokens": 100,
            },
        }},
    }]
    evaluator.method_config = SimpleNamespace(method_name="long_context")
    evaluator.metrics_calculator = SimpleNamespace(
        get_batch_judge_client=lambda query_type: judge_client,
    )
    evaluator._save_deferred_judges = lambda client: None
    evaluator._get_judge_batch_client = lambda client: SimpleNamespace(
        run_stage=run_stage,
    )
    evaluator._log = lambda message: None

    with pytest.raises(BatchStageCalled):
        evaluator._complete_deferred_judges()

    assert captured["stage"] == "judge-final"
    assert len(captured["requests"]) == 1


def test_completed_deferred_judge_restores_planner_artifact():
    judge_client = SimpleNamespace(model="gemini-2.5-flash-lite")
    evaluator = MedMemoryBenchEvaluator.__new__(MedMemoryBenchEvaluator)
    evaluator._deferred_judges = [{
        "query_id": "q1",
        "persona_id": 1,
        "prepared_metric": {"prepared": {
            "query_id": "q1",
            "query_type": "state_update",
            "judge_payload": {"prompt": "judge", "max_tokens": 100},
        }},
        "query_time": 1.0,
        "memory_construction_time": 2.0,
        "retrieved_memories": [],
        "retrieved_count": 0,
        "planner": {"rounds_configured": 1, "retrieval_round_count": 1},
    }]
    evaluator.method_config = SimpleNamespace(method_name="event_state")
    evaluator.metrics_calculator = SimpleNamespace(
        get_batch_judge_client=lambda query_type: judge_client,
        finalize_batch=lambda prepared, content: MetricResult("q1", "state_update", 1.0, True, "answer", "answer"),
    )
    evaluator._save_deferred_judges = lambda client: None
    evaluator._get_judge_batch_client = lambda client: SimpleNamespace(
        run_stage=lambda stage, requests: {
            requests[0].request_id: BatchChatResponse(requests[0].request_id, content="{}")
        }
    )
    evaluator._run_api_call = lambda operation, *args, **kwargs: operation(*args, **kwargs)
    evaluator._batch_artifact_reference = lambda client, request_id: {}
    evaluator._batch_response_usage = lambda response, transport: {}
    evaluator._attach_session_retrieval_quality = lambda result, quality: None
    evaluator._advance_query_progress = lambda *args, **kwargs: None
    evaluator._clear_deferred_judges = lambda: None
    evaluator._log = lambda *args, **kwargs: None

    finalized = evaluator._complete_deferred_judges()

    assert finalized[0]["result"].details["planner"] == {
        "rounds_configured": 1,
        "retrieval_round_count": 1,
    }


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


class _OpenRouterResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code
        self.text = json.dumps(payload)

    def json(self):
        return self._payload


class _OpenRouterHTTPClient:
    def __init__(self, handler):
        self.handler = handler
        self.calls = []

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        return self.handler(method, url, kwargs)


def _openrouter_client(*, routing=None, service_tier=None, direct_calls=None):
    client = object.__new__(OpenRouterClient)
    client.model = "openai/gpt-4o"
    client.temperature = 0.2
    client.max_tokens = 100
    client.provider_routing = routing
    client.service_tier = service_tier
    client.client = SimpleNamespace(
        base_url="https://openrouter.ai/api/v1/",
        api_key="test-key",
    )
    direct_calls = direct_calls if direct_calls is not None else []

    def chat(messages, **kwargs):
        direct_calls.append((messages, kwargs))
        return LLMResponse(
            content=f"direct-{messages[-1]['content']}",
            input_tokens=3,
            output_tokens=2,
            model=client.model,
        )

    client.chat = chat
    return client


def _openrouter_request(request_id, text):
    return BatchChatRequest(
        request_id=request_id,
        messages=[{"role": "user", "content": text}],
        temperature=0.1,
        max_tokens=50,
        phase="query",
    )


def test_openrouter_batch_submits_and_correlates_inline_results(tmp_path, monkeypatch):
    monkeypatch.setenv("VERTEX_BATCH_MIN_REQUESTS", "2")

    def handler(method, url, kwargs):
        if url.endswith("/models/openai/gpt-4o%3Abatch/endpoints"):
            return _OpenRouterResponse({
                "data": {"endpoints": [{"tag": "openai"}]}
            })
        if method == "POST":
            return _OpenRouterResponse(
                {"id": "batch_1", "status": "validating"}, 202
            )
        return _OpenRouterResponse({
            "id": "batch_1",
            "status": "completed",
            "results": [
                {
                    "custom_id": "req-2",
                    "response": {
                        "status_code": 200,
                        "body": {
                            "choices": [{"message": {"content": "answer-2"}}],
                            "usage": {"prompt_tokens": 7, "completion_tokens": 3},
                        },
                    },
                    "error": None,
                },
                {
                    "custom_id": "req-1",
                    "response": {
                        "status_code": 200,
                        "body": {
                            "choices": [{"message": {"content": "answer-1"}}],
                            "usage": {"prompt_tokens": 5, "completion_tokens": 2},
                        },
                    },
                    "error": None,
                },
            ],
        })

    http_client = _OpenRouterHTTPClient(handler)
    client = OpenRouterBatchClient(
        direct_client=_openrouter_client(
            routing={"order": ["openai", "azure"], "allow_fallbacks": False}
        ),
        manifest_path=tmp_path / "batch.json",
        wait=True,
        config_hash="hash",
        http_client=http_client,
    )

    responses = client.run_stage(
        "query-final",
        [
            _openrouter_request("req-1", "one"),
            _openrouter_request("req-2", "two"),
        ],
    )

    assert responses["req-1"].content == "answer-1"
    assert responses["req-2"].input_tokens == 7
    post = next(call for call in http_client.calls if call[0] == "POST")
    payload = json.loads(post[2]["content"])
    assert list(payload) == ["endpoint", "model", "requests"]
    assert payload["requests"][0]["body"]["provider"] == {
        "order": ["openai", "azure"],
        "allow_fallbacks": False,
    }


@pytest.mark.parametrize(
    ("routing", "service_tier"),
    [
        ({"only": ["azure"]}, None),
        (None, "priority"),
    ],
)
def test_openrouter_unsupported_route_uses_realtime_client(
    tmp_path, routing, service_tier
):
    direct_calls = []
    progress = []
    http_client = _OpenRouterHTTPClient(
        lambda method, url, kwargs: _OpenRouterResponse({
            "data": {"endpoints": [{"tag": "openai"}]}
        })
    )
    client = OpenRouterBatchClient(
        direct_client=_openrouter_client(
            routing=routing,
            service_tier=service_tier,
            direct_calls=direct_calls,
        ),
        manifest_path=tmp_path / "batch.json",
        wait=False,
        http_client=http_client,
        progress_callback=progress.append,
    )

    responses = client.run_stage(
        "query-final", [_openrouter_request("req-1", "one")]
    )

    assert responses["req-1"].content == "direct-one"
    assert len(direct_calls) == 1
    assert all(method != "POST" for method, _, _ in http_client.calls)
    assert any("ignoring --batch-api" in message for message in progress)


def test_openrouter_submit_only_raises_shared_pending_type(tmp_path, monkeypatch):
    monkeypatch.setenv("VERTEX_BATCH_MIN_REQUESTS", "2")

    def handler(method, url, kwargs):
        if url.endswith("/endpoints"):
            return _OpenRouterResponse({
                "data": {"endpoints": [{"tag": "openai"}]}
            })
        if method == "POST":
            return _OpenRouterResponse(
                {"id": "batch_pending", "status": "validating"}, 202
            )
        return _OpenRouterResponse({
            "id": "batch_pending", "status": "in_progress"
        })

    client = OpenRouterBatchClient(
        direct_client=_openrouter_client(),
        manifest_path=tmp_path / "batch.json",
        wait=False,
        http_client=_OpenRouterHTTPClient(handler),
    )

    with pytest.raises(OpenRouterBatchPending) as exc_info:
        client.run_stage(
            "query-final",
            [
                _openrouter_request("req-1", "one"),
                _openrouter_request("req-2", "two"),
            ],
        )

    assert isinstance(exc_info.value, VertexBatchPending)
    assert exc_info.value.job_name == "batch_pending"
