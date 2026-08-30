"""Regression coverage for amem_test build worker boundaries."""

from types import SimpleNamespace

from methods.amem_test_agent import AMemTestAgent


def _agent_for_workers(system, *, original_evolution, typed_relations):
    agent = object.__new__(AMemTestAgent)
    agent._context_id = 7
    agent._memory_chunks = []
    agent._is_initialized = False
    agent.retrieve_num = 10
    agent.amem_chunk_size_tokens = 10240
    agent.amem_note_level = "turn"
    agent.amem_original_evolution = original_evolution
    agent.amem_typed_relations = typed_relations
    agent.amem_workers = 2
    agent.amem_relation_candidate_count = 5
    agent.amem_temporal_state = False
    agent.amem_provenance = False
    agent._llm_client = SimpleNamespace(
        count_tokens=lambda text: len(str(text).split())
    )
    agent._get_memory_system = lambda context_id: system
    return agent


def test_parallel_build_preserves_atomic_turn_order_and_timestamps():
    captured = []

    class System:
        def add_notes_parallel(self, specs, workers):
            captured.extend(specs)
            return [f"note-{index}" for index, _ in enumerate(specs)]

        def get_relation_audit(self, memory_id):
            return {}

    agent = _agent_for_workers(
        System(), original_evolution=False, typed_relations=True
    )

    result = agent.memorize(
        "formatted wrapper",
        timestamp="2025-01-02",
        memory_items=[
            {"role": "user", "content": "First turn."},
            {"role": "assistant", "content": "Second turn."},
        ],
    )

    assert [spec["content"] for spec in captured] == [
        "Speaker Patient says: First turn.",
        "Speaker Doctor says: Second turn.",
    ]
    assert [spec["time"] for spec in captured] == ["2025-01-02", "2025-01-02"]
    assert [entry["turn_index"] for entry in result.memory_entries] == [0, 1]
    assert result.extra["note_ids"] == ["note-0", "note-1"]


def test_metadata_workers_reuse_prepared_analysis_for_serial_evolution():
    prepared_contents = []
    added = []

    class System:
        def prepare_note_metadata(self, contents, workers):
            prepared_contents.extend(contents)
            return [
                {"keywords": [content], "context": "ctx", "tags": ["tag"]}
                for content in contents
            ]

        def add_note(self, **kwargs):
            added.append(kwargs)
            return f"note-{len(added)}"

    agent = _agent_for_workers(
        System(), original_evolution=True, typed_relations=False
    )

    agent.memorize(
        "formatted wrapper",
        timestamp="2025-01-02",
        memory_items=[
            {"role": "user", "content": "First turn."},
            {"role": "assistant", "content": "Second turn."},
        ],
    )

    assert prepared_contents == [
        "Speaker Patient says: First turn.",
        "Speaker Doctor says: Second turn.",
    ]
    assert [item["_prepared_analysis"]["context"] for item in added] == [
        "ctx",
        "ctx",
    ]
