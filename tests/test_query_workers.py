"""Regression coverage for bounded real-time query concurrency."""

import threading
import time
from types import SimpleNamespace

from benchmarks.locomo.evaluator import LoCoMoEvaluator
from benchmarks.medmemorybench.evaluator import MedMemoryBenchEvaluator


def test_medmemorybench_workers_keep_answer_and_judge_together_and_ordered():
    evaluator = MedMemoryBenchEvaluator.__new__(MedMemoryBenchEvaluator)
    evaluator.workers = 2
    evaluator._log = lambda *args, **kwargs: None
    evaluator._supports_memory_snapshots = lambda: False

    active_workers = 0
    max_active_workers = 0
    events = []
    lock = threading.Lock()

    def evaluate_query(query, *args, **kwargs):
        nonlocal active_workers, max_active_workers
        with lock:
            active_workers += 1
            max_active_workers = max(max_active_workers, active_workers)
            events.append((query.query_id, "answer"))
        time.sleep(0.02)
        with lock:
            events.append((query.query_id, "judge"))
            active_workers -= 1
        return SimpleNamespace(query_id=query.query_id)

    evaluator._evaluate_query = evaluate_query
    queries = [SimpleNamespace(query_id=f"q{index}") for index in range(4)]

    completed = evaluator._evaluate_realtime_queries(
        queries,
        context_id=1,
        memory_time_per_query=0.0,
        unit_id=1,
    )

    assert max_active_workers == 2
    assert [query.query_id for query, _ in completed] == ["q0", "q1", "q2", "q3"]
    for query in queries:
        assert events.index((query.query_id, "answer")) < events.index(
            (query.query_id, "judge")
        )


def test_medmemorybench_query_progress_advances_for_serial_and_parallel_queries():
    class Progress:
        def __init__(self):
            self.updates = []

        def update(self, count):
            self.updates.append(count)

    evaluator = MedMemoryBenchEvaluator.__new__(MedMemoryBenchEvaluator)
    evaluator._log = lambda *args, **kwargs: None
    evaluator._supports_memory_snapshots = lambda: False
    evaluator._evaluate_query = lambda query, *args, **kwargs: query

    for worker_count in (1, 2):
        progress = Progress()
        evaluator.workers = worker_count
        evaluator._query_progress = progress
        evaluator._query_progress_lock = threading.Lock()
        evaluator._query_progress_completed = set()
        queries = [SimpleNamespace(query_id=f"q{index}") for index in range(3)]

        evaluator._evaluate_realtime_queries(
            queries,
            context_id=1,
            memory_time_per_query=0.0,
            unit_id=1,
        )

        assert progress.updates == [1, 1, 1]


def test_medmemorybench_query_progress_is_created_only_at_query_start():
    evaluator = MedMemoryBenchEvaluator.__new__(MedMemoryBenchEvaluator)
    evaluator.execution_stage = "all"
    evaluator.verbose = False
    evaluator._checkpoint_manager = None
    units = [
        SimpleNamespace(
            context_id=1,
            queries_to_evaluate=[SimpleNamespace(query_id="q1")],
        )
    ]

    evaluator._configure_query_progress(units)
    assert getattr(evaluator, "_query_progress", None) is None

    evaluator._start_query_progress()
    assert evaluator._query_progress is not None
    evaluator._finish_query_progress()


def test_locomo_workers_bound_real_time_query_concurrency_and_keep_order():
    evaluator = LoCoMoEvaluator.__new__(LoCoMoEvaluator)
    evaluator.workers = 2
    evaluator._log = lambda *args, **kwargs: None

    active_workers = 0
    max_active_workers = 0
    lock = threading.Lock()

    def evaluate_query(query, context_id):
        nonlocal active_workers, max_active_workers
        with lock:
            active_workers += 1
            max_active_workers = max(max_active_workers, active_workers)
        time.sleep(0.02)
        with lock:
            active_workers -= 1
        return query.query_id

    evaluator._evaluate_query = evaluate_query
    queries = [SimpleNamespace(query_id=f"q{index}") for index in range(4)]

    completed = evaluator._evaluate_realtime_queries(queries, context_id="sample")

    assert max_active_workers == 2
    assert completed == ["q0", "q1", "q2", "q3"]


def test_medmemorybench_batch_path_does_not_dispatch_real_time_workers():
    evaluator = MedMemoryBenchEvaluator.__new__(MedMemoryBenchEvaluator)
    evaluator.batch_api = True
    evaluator.dry_run = False
    evaluator._batch_fallback_logged = False
    evaluator._checkpoint_manager = None
    evaluator._deferred_judges = []
    evaluator._log = lambda *args, **kwargs: None
    evaluator._is_deferred_judge_query = lambda query_id: False
    evaluator._supports_batch_queries = lambda: True
    evaluator._get_batch_client = lambda: SimpleNamespace(has_stage=lambda stage: False)
    prepared = []
    evaluator._prepare_combined_batch_queries = (
        lambda unit, memory_time_per_query: prepared.append(unit.unit_id) or 2
    )
    evaluator._evaluate_realtime_queries = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("real-time workers must not run for an eligible batch stage")
    )
    unit = SimpleNamespace(
        unit_id=7,
        context_id=1,
        queries_to_evaluate=[
            SimpleNamespace(query_id="q1"),
            SimpleNamespace(query_id="q2"),
        ],
    )

    assert evaluator._evaluate_unit_queries(unit, total_memory_time=0.0) == []
    assert prepared == [7]
