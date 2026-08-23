"""Focused coverage for the experimental A-MEM Typed Memory Relations path."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from methods.amem_agent import AMemAgent
from methods.amem_fix_agent import AMemFixAgent
from methods.amem_test_agent import AMemTestAgent
from src.agent import AgentManager
from src.config import DatasetConfig, MethodConfig, method_config_from_snapshot
from benchmarks.medmemorybench.checkpoint import compute_config_hash


AMemTestAgent._load_amem_system_class(object())
from memory_layer_robust import RobustAgenticMemorySystem  # noqa: E402
from memory_layer_typed import (  # noqa: E402
    RELATION_TYPES,
    TypedRelationMemorySystem,
    detect_temporal_query,
    map_relation_predictions,
    parse_typed_relations,
)


class _Tokenizer:
    def encode(self, text: str):
        return (text or "").split()

    def decode(self, tokens):
        return " ".join(tokens)


class _LLMClient:
    def count_tokens(self, text: str) -> int:
        return len((text or "").split())


def _memory(content: str):
    return SimpleNamespace(
        content=content,
        context="General",
        keywords=[],
        tags=[],
        timestamp="2025-01-01",
        links=[],
    )


def _empty_system() -> TypedRelationMemorySystem:
    system = object.__new__(TypedRelationMemorySystem)
    system.memories = OrderedDict()
    system.typed_relations = []
    system._typed_edge_keys = set()
    return system


def _experimental_system(
    *,
    temporal: bool = True,
    provenance: bool = False,
) -> TypedRelationMemorySystem:
    system = _empty_system()
    system.temporal_state_enabled = temporal
    system.provenance_enabled = provenance
    system.temporal_min_confidence = 0.5
    system.temporal_audit = []
    system._temporal_audit_by_memory = {}
    system.evidence_store = {}
    system.provenance_audit = []
    system._provenance_audit_by_memory = {}
    return system


def _timestamped_memory(content: str, timestamp: str | None):
    memory = _memory(content)
    memory.timestamp = timestamp or ""
    memory.source_timestamp = timestamp
    return memory


def _edge(source: str, target: str, relation_type: str, confidence: float = 0.95):
    return {
        "source_id": source,
        "target_id": target,
        "relation_type": relation_type,
        "direction": "NEW_MEMORY_TO_EXISTING_MEMORY",
        "confidence": confidence,
        "reason": "test transition",
    }


def test_typed_relation_parser_accepts_every_relation_and_no_relation():
    response = "\n".join([
        "RELATION|0|SUPPORT|0.91|independent confirmation",
        "RELATION|1|REFINE|82%|adds precision",
        "RELATION|2|SUPERSEDE|1.4|explicit replacement",
        "RELATION|3|CONFLICT|-0.2|incompatible claims",
        "RELATION|4|RELATED|bad|same topic",
        "RELATION|5|NO_RELATION|0.9|not meaningful",
    ])

    parsed = parse_typed_relations(response, candidate_count=6)

    assert [item["relation_type"] for item in parsed] == list(RELATION_TYPES)
    assert [item["confidence"] for item in parsed] == [0.91, 0.82, 1.0, 0.0, 0.5]
    assert parse_typed_relations("NO_RELATIONS", 3) == []


def test_typed_relation_parser_handles_json_duplicates_and_malformed_output():
    response = """```json
    {"relations": [
      {"position": 0, "type": "support", "confidence": 0.4},
      {"position": 0, "type": "REFINE", "confidence": 0.8},
      {"position": 9, "type": "CONFLICT", "confidence": 0.9},
      {"position": 1, "type": "invented", "confidence": 0.9}
    ]}
    ```"""

    assert parse_typed_relations(response, 2) == [{
        "candidate_position": 0,
        "relation_type": "REFINE",
        "confidence": 0.8,
        "reason": "",
    }]
    assert parse_typed_relations("this is not parseable", 2) == []


def test_relation_positions_map_to_stable_ids_and_reject_self_links():
    edges = map_relation_predictions(
        "new-id",
        ["old-a", "new-id", "old-b"],
        [
            {"candidate_position": 2, "relation_type": "SUPERSEDE", "confidence": 0.9},
            {"candidate_position": 1, "relation_type": "CONFLICT", "confidence": 0.8},
        ],
    )

    assert edges == [{
        "source_id": "new-id",
        "target_id": "old-b",
        "relation_type": "SUPERSEDE",
        "direction": "NEW_MEMORY_TO_EXISTING_MEMORY",
        "confidence": 0.9,
        "reason": "",
        "candidate_position": 2,
    }]


def test_relation_storage_is_bidirectional_and_prevents_duplicates():
    system = _empty_system()
    system.memories["old"] = _memory("I work at Google")
    system.memories["new"] = _memory("I left Google and joined Microsoft")
    edge = {
        "source_id": "new",
        "target_id": "old",
        "relation_type": "SUPERSEDE",
        "direction": "NEW_MEMORY_TO_EXISTING_MEMORY",
        "confidence": 0.95,
        "reason": "explicit job replacement",
    }

    assert system.add_typed_relation(edge) is True
    assert system.add_typed_relation(edge) is False
    assert system.memories["new"].typed_relations_out[0]["target_id"] == "old"
    assert system.memories["old"].typed_relations_in[0]["source_id"] == "new"
    assert len(system.typed_relations) == 1

    conflicting_duplicate = dict(edge, relation_type="CONFLICT")
    assert system.add_typed_relation(conflicting_duplicate) is False


def test_temporal_supersession_preserves_old_memory_and_builds_chain():
    system = _experimental_system()
    system.memories["a"] = _timestamped_memory("I work at Google", "2024-01-01")
    system.memories["b"] = _timestamped_memory("I joined Microsoft", "2024-02-01")
    system.memories["c"] = _timestamped_memory("I joined Mozilla", "2024-03-01")

    assert system.add_typed_relation(_edge("b", "a", "SUPERSEDE")) is True
    assert system.add_typed_relation(_edge("c", "b", "SUPERSEDE")) is True

    assert system.memories["a"].content == "I work at Google"
    assert system.get_temporal_state("a")["status"] == "superseded"
    assert system.get_temporal_state("a")["valid_until"] == "2024-02-01"
    assert system.get_temporal_state("a")["superseded_by"] == ["b"]
    assert system.get_temporal_state("b")["supersedes"] == ["a"]
    assert system.get_temporal_state("b")["superseded_by"] == ["c"]
    assert system.get_temporal_state("c")["status"] == "current"


def test_refine_preserves_validity_and_conflict_preserves_uncertainty():
    system = _experimental_system()
    system.memories["old"] = _timestamped_memory("I moved to Bangkok", "2024-01-01")
    system.memories["detail"] = _timestamped_memory(
        "I moved to Bangkok in June", "2024-02-01"
    )
    system.memories["other"] = _timestamped_memory(
        "The meeting is Friday", "2024-03-01"
    )

    system.add_typed_relation(_edge("detail", "old", "REFINE"))
    assert system.get_temporal_state("old")["status"] == "current"
    assert system.get_temporal_state("old")["valid_until"] is None
    assert system.get_temporal_state("old")["refined_by"] == ["detail"]

    system.add_typed_relation(_edge("other", "detail", "CONFLICT"))
    assert system.get_temporal_state("detail")["status"] == "current"
    assert system.get_temporal_state("other")["status"] == "current"
    assert system.get_temporal_state("detail")["uncertainty"] == "conflicting"
    assert system.get_temporal_state("other")["uncertainty"] == "conflicting"


def test_invalid_or_low_confidence_temporal_transitions_do_not_mutate_state():
    system = _experimental_system()
    system.memories["old"] = _timestamped_memory("old", "2024-03-01")
    system.memories["earlier"] = _timestamped_memory("earlier", "2024-02-01")
    system.memories["weak"] = _timestamped_memory("weak", "2024-04-01")

    system.add_typed_relation(_edge("earlier", "old", "SUPERSEDE"))
    system.add_typed_relation(_edge("weak", "old", "SUPERSEDE", confidence=0.2))

    assert system.get_temporal_state("old")["status"] == "current"
    assert system.get_temporal_state("old")["superseded_by"] == []
    assert len(system.temporal_audit) == 2
    assert all(item["applied"] is False for item in system.temporal_audit)


def test_temporal_retrieval_current_historical_time_and_change():
    system = _experimental_system()
    system.memories["google"] = _timestamped_memory("Google", "2024-01-01")
    system.memories["microsoft"] = _timestamped_memory("Microsoft", "2024-02-01")
    system.memories["mozilla"] = _timestamped_memory("Mozilla", "2024-03-01")
    system.add_typed_relation(_edge("microsoft", "google", "SUPERSEDE"))
    system.add_typed_relation(_edge("mozilla", "microsoft", "SUPERSEDE"))

    current = system.select_temporal_memories(
        ["google"], "Where does the user work now?", expansion_budget=5
    )
    historical = system.select_temporal_memories(
        ["mozilla"], "Where did the user work before?", expansion_budget=5
    )
    historical_specific = system.select_temporal_memories(
        ["mozilla"], "Where did the user work before Microsoft?", expansion_budget=5
    )
    time_specific = system.select_temporal_memories(
        ["mozilla"], "Where did the user work in February 2024?", expansion_budget=5
    )
    month_only = system.select_temporal_memories(
        ["mozilla"], "Where did the user work in February?", expansion_budget=5
    )
    change = system.select_temporal_memories(
        ["microsoft"], "How did the user's job change?", expansion_budget=5
    )

    assert current["selected_memory_ids"] == ["mozilla", "microsoft", "google"]
    assert historical["selected_memory_ids"][0] == "microsoft"
    assert historical_specific["selected_memory_ids"][0] == "google"
    assert historical_specific["historical_pivot_ids"] == ["microsoft"]
    assert time_specific["selected_memory_ids"][0] == "microsoft"
    assert month_only["selected_memory_ids"][0] == "microsoft"
    assert month_only["query"]["target"]["year_inferred_from_timeline"] is True
    assert set(change["selected_memory_ids"]) == {"google", "microsoft", "mozilla"}
    assert change["query"]["intent"] == "change"


def test_temporal_retrieval_can_expand_without_reordering():
    system = _experimental_system()
    system.memories["google"] = _timestamped_memory("Google", "2024-01-01")
    system.memories["microsoft"] = _timestamped_memory("Microsoft", "2024-02-01")
    system.add_typed_relation(_edge("microsoft", "google", "SUPERSEDE"))

    result = system.select_temporal_memories(
        ["google"],
        "Where does the user work now?",
        expansion_budget=5,
        apply_ordering=False,
    )

    assert result["selected_memory_ids"] == ["google", "microsoft"]
    assert result["expanded_memories"][0]["added_memory_id"] == "microsoft"
    assert result["query"]["intent"] == "current"


def test_temporal_query_detection_supports_missing_and_relative_time():
    assert detect_temporal_query("What is true now?")["intent"] == "current"
    assert detect_temporal_query("What was true 2 months ago?")["intent"] == "time_specific"
    assert detect_temporal_query("Tell me the relevant facts")["intent"] == "none"

    system = _experimental_system()
    system.memories["unknown"] = _timestamped_memory("unknown time", None)
    result = system.select_temporal_memories(
        ["unknown"], "What was true in March 2024?", expansion_budget=1
    )
    assert result["selected_memory_ids"] == ["unknown"]
    assert result["states"]["unknown"]["valid_from"] is None


def test_temporal_state_orders_locomo_natural_language_timestamps():
    system = _experimental_system()
    system.memories["old"] = _timestamped_memory(
        "old state", "1:56 pm on 8 May, 2023"
    )
    system.memories["new"] = _timestamped_memory(
        "new state", "1:14 pm on 25 May, 2023"
    )

    system.add_typed_relation(_edge("new", "old", "SUPERSEDE"))

    assert system.get_temporal_state("old")["status"] == "superseded"
    assert system.get_temporal_state("old")["valid_until"] == "1:14 pm on 25 May, 2023"


def test_temporal_state_requires_typed_relations():
    with pytest.raises(ValueError, match="requires amem_typed_relations"):
        AMemTestAgent.__init__(
            object.__new__(AMemTestAgent),
            model="mock",
            amem_typed_relations=False,
            amem_temporal_state=True,
        )


def test_provenance_store_is_stable_immutable_and_deduplicated():
    system = _experimental_system(temporal=False, provenance=True)
    evidence = {
        "raw_text": "Exact source: 16.5 mmol/L on April 3.",
        "source_session_id": 7,
        "source_turn_id": 2,
        "source_timestamp": "2024-04-03",
    }

    first_id = system.register_evidence(evidence)
    second_id = system.register_evidence(dict(evidence))
    returned = system.get_evidence(first_id)
    returned["raw_text"] = "mutated"

    assert first_id == second_id
    assert len(system.evidence_store) == 1
    assert system.get_evidence(first_id)["raw_text"] == evidence["raw_text"]
    assert system.get_evidence(first_id)["source_timestamp"] == "2024-04-03"


def test_provenance_memorization_preserves_exact_raw_source_string():
    calls = []
    memory_system = SimpleNamespace(
        add_note=lambda **kwargs: calls.append(kwargs) or "note-1",
        get_relation_audit=lambda memory_id: {},
        get_provenance_audit=lambda memory_id: {
            "memory_id": memory_id,
            "evidence_ids": ["ev-1"],
            "error": "",
        },
    )
    agent = object.__new__(AMemTestAgent)
    agent._context_id = 1
    agent._memory_chunks = []
    agent._is_initialized = False
    agent.retrieve_num = 10
    agent.amem_chunk_size_tokens = 10240
    agent.amem_original_evolution = False
    agent.amem_typed_relations = False
    agent.amem_temporal_state = False
    agent.amem_provenance = True
    agent.amem_relation_candidate_count = 5
    agent._llm_client = _LLMClient()
    agent._get_memory_system = lambda context_id: memory_system

    agent.memorize(
        "wrapper",
        memory_items=[{
            "role": "user",
            "content": "  exact source text\n",
            "blip_caption": "  exact image caption  ",
        }],
    )

    assert calls[0]["content"] == (
        "Speaker Patient says: exact source text Shared image: exact image caption"
    )
    assert calls[0]["source_evidence"]["raw_text"] == "  exact source text\n"
    assert calls[0]["source_evidence"]["blip_caption"] == "  exact image caption  "


def test_provenance_survives_memory_evolution_and_missing_metadata():
    system = _experimental_system(temporal=False, provenance=True)
    memory = _timestamped_memory("structured summary", None)
    memory.id = "m1"
    system._attach_provenance(memory, {"raw_text": "raw exact statement"}, 0)
    system.memories["m1"] = memory
    evidence_id = memory.provenance["evidence_ids"][0]

    memory.content = "evolved summary"
    memory.context = "evolved context"

    assert system.evidence_for_memory("m1")[0]["raw_text"] == "raw exact statement"
    assert system.get_provenance_audit("m1")["source_session_id"] is None
    assert system.get_evidence(evidence_id)["source_timestamp"] is None


def test_experimental_missing_timestamp_does_not_invent_execution_time(monkeypatch):
    class _Note:
        def __init__(self, content, timestamp=None, **kwargs):
            self.id = "m1"
            self.content = content
            self.context = "General"
            self.keywords = []
            self.tags = []
            self.links = []
            self.timestamp = "2099-12-31"

    class _Retriever:
        def add_documents(self, documents):
            self.documents = documents

    import memory_layer_typed

    monkeypatch.setattr(memory_layer_typed, "RobustMemoryNote", _Note)
    system = _experimental_system(temporal=True, provenance=False)
    system.original_evolution_enabled = False
    system.typed_relations_enabled = False
    system.retriever = _Retriever()
    system.llm_controller = SimpleNamespace()
    system.max_context_chars = 1000
    system.evo_cnt = 0
    system.evo_threshold = 100

    note_id = system.add_note("unknown date", time=None, source_timestamp=None)

    assert system.memories[note_id].timestamp == ""
    assert system.get_temporal_state(note_id)["valid_from"] is None


def test_provenance_context_includes_raw_evidence_without_duplicate_text():
    system = _experimental_system(temporal=False, provenance=True)
    memory = _timestamped_memory("Speaker Patient says: exact statement", "2024-01-01")
    memory.id = "m1"
    system._attach_provenance(
        memory,
        {
            "raw_text": "exact statement",
            "source_session_id": 1,
            "source_turn_id": 0,
            "source_timestamp": "2024-01-01",
        },
        0,
    )
    system.memories["m1"] = memory
    agent = object.__new__(AMemTestAgent)
    agent.amem_provenance = True
    agent.amem_provenance_max_evidence = 5

    records = agent._provenance_context_records(system, ["m1"])
    context = agent._format_relation_context(
        system,
        ["m1"],
        ["m1"],
        [],
        [],
        typed_enabled=False,
        evidence_records=records,
    )

    assert len(records) == 1
    assert records[0]["raw_text_injected"] is False
    assert "Evidence ev_" in context
    assert "raw text injection disabled" in context


def test_provenance_raw_injection_flag_includes_raw_text():
    system = _experimental_system(temporal=False, provenance=True)
    memory = _timestamped_memory("structured summary", "2024-01-01")
    memory.id = "m1"
    system._attach_provenance(
        memory,
        {
            "raw_text": "raw exact statement",
            "source_session_id": 1,
            "source_turn_id": 0,
        },
        0,
    )
    system.memories["m1"] = memory
    agent = object.__new__(AMemTestAgent)
    agent.amem_provenance = True
    agent.amem_provenance_max_evidence = 5
    agent.amem_provenance_inject_raw_text = True

    records = agent._provenance_context_records(system, ["m1"])
    context = agent._format_relation_context(
        system,
        ["m1"],
        ["m1"],
        [],
        [],
        typed_enabled=False,
        evidence_records=records,
    )

    assert records[0]["raw_text_injected"] is True
    assert "raw text follows below" in context
    assert "[Raw Source Conversations]" in context
    assert context.count("raw exact statement") == 1


def test_provenance_raw_injection_false_is_strict_even_for_distinct_text():
    system = _experimental_system(temporal=False, provenance=True)
    memory = _timestamped_memory("structured summary", "2024-01-01")
    memory.id = "m1"
    system._attach_provenance(
        memory,
        {"raw_text": "different raw statement", "source_session_id": 1},
        0,
    )
    system.memories["m1"] = memory
    agent = object.__new__(AMemTestAgent)
    agent.amem_provenance = True
    agent.amem_provenance_max_evidence = 5
    agent.amem_provenance_inject_raw_text = False

    records = agent._provenance_context_records(system, ["m1"])
    context = agent._format_relation_context(
        system,
        ["m1"],
        ["m1"],
        [],
        [],
        typed_enabled=False,
        evidence_records=records,
    )

    assert records[0]["raw_text_injected"] is False
    assert "raw text injection disabled" in context
    assert "different raw statement" not in context


def test_provenance_raw_injection_deduplicates_repeated_source_text():
    system = _experimental_system(temporal=False, provenance=True)
    first = _timestamped_memory("summary one", "2024-01-01")
    second = _timestamped_memory("summary two", "2024-01-02")
    first.id = "m1"
    second.id = "m2"
    system._attach_provenance(
        first,
        {"raw_text": "same raw conversation", "source_session_id": 1},
        0,
    )
    system._attach_provenance(
        second,
        {"raw_text": "same raw conversation", "source_session_id": 2},
        0,
    )
    system.memories.update({"m1": first, "m2": second})
    agent = object.__new__(AMemTestAgent)
    agent.amem_provenance = True
    agent.amem_provenance_max_evidence = 5
    agent.amem_provenance_inject_raw_text = True

    records = agent._provenance_context_records(system, ["m1", "m2"])
    context = agent._format_relation_context(
        system,
        ["m1", "m2"],
        ["m1", "m2"],
        [],
        [],
        typed_enabled=False,
        evidence_records=records,
    )

    first_id = records[0]["evidence"]["evidence_id"]
    second_id = records[1]["evidence"]["evidence_id"]
    assert first_id != second_id
    assert records[0]["raw_text_injected"] is True
    assert records[1]["raw_text_injected"] is False
    assert records[1]["raw_text_duplicate_of"] == first_id
    assert f"Evidence {first_id} supports M1" in context
    assert f"Evidence {second_id} supports M2" in context
    assert f"raw text duplicates Evidence {first_id}; not repeated" in context
    assert context.count("same raw conversation") == 1


def test_provenance_keeps_shared_evidence_id_for_each_memory_note():
    system = _experimental_system(temporal=False, provenance=True)
    first = _timestamped_memory("summary one", "2024-01-01")
    second = _timestamped_memory("summary two", "2024-01-01")
    first.id = "m1"
    second.id = "m2"
    evidence = {
        "raw_text": "shared source turn",
        "source_session_id": 1,
        "source_turn_id": 0,
    }
    system._attach_provenance(first, evidence, 0)
    system._attach_provenance(second, evidence, 1)
    system.memories.update({"m1": first, "m2": second})
    agent = object.__new__(AMemTestAgent)
    agent.amem_provenance = True
    agent.amem_provenance_max_evidence = 5
    agent.amem_provenance_inject_raw_text = True

    records = agent._provenance_context_records(system, ["m1", "m2"])
    context = agent._format_relation_context(
        system,
        ["m1", "m2"],
        ["m1", "m2"],
        [],
        [],
        typed_enabled=False,
        evidence_records=records,
    )

    assert len(records) == 2
    assert records[0]["evidence"]["evidence_id"] == records[1]["evidence"]["evidence_id"]
    assert f"supports M1" in context
    assert f"supports M2" in context
    assert context.count("shared source turn") == 1


def test_provenance_disabled_does_not_add_memory_metadata():
    system = _experimental_system(temporal=False, provenance=False)
    memory = _timestamped_memory("plain", "2024-01-01")
    memory.id = "m1"
    system._attach_provenance(memory, {"raw_text": "plain"}, 0)
    assert not hasattr(memory, "provenance")
    assert system.evidence_store == {}


def test_provenance_can_be_enabled_without_typed_relations():
    agent = AMemTestAgent.__new__(AMemTestAgent)
    agent.amem_original_evolution = True
    agent.amem_typed_relations = False
    agent.amem_temporal_state = False
    agent.amem_provenance = True
    agent.amem_relation_candidate_count = 5
    agent.amem_typed_expansion_count = 5
    agent.amem_temporal_expansion_count = 5
    agent.amem_provenance_max_evidence = 5
    assert agent.amem_provenance is True
    assert agent.amem_temporal_state is False


def test_snapshot_feature_mismatch_is_rejected_before_restore():
    agent = object.__new__(AMemTestAgent)
    agent.amem_original_evolution = False
    agent.amem_typed_relations = True
    agent.amem_temporal_state = True
    agent.amem_provenance = False
    state = {
        "system_state": {
            "original_evolution_enabled": False,
            "typed_relations_enabled": True,
            "temporal_state_enabled": False,
            "provenance_enabled": False,
        }
    }
    with pytest.raises(ValueError, match="temporal-state flag"):
        agent.import_memory_state(state, context_id=1)


@pytest.mark.parametrize(
    "field_name, state_value, error_pattern",
    [
        ("original_evolution_enabled", True, "original-evolution flag"),
        ("typed_relations_enabled", False, "typed-relations flag"),
    ],
)
def test_snapshot_rejects_core_experimental_feature_mismatch(
    field_name,
    state_value,
    error_pattern,
):
    agent = object.__new__(AMemTestAgent)
    agent.amem_original_evolution = False
    agent.amem_typed_relations = True
    agent.amem_temporal_state = False
    agent.amem_provenance = False
    system_state = {
        "original_evolution_enabled": False,
        "typed_relations_enabled": True,
        "temporal_state_enabled": False,
        "provenance_enabled": False,
    }
    system_state[field_name] = state_value

    with pytest.raises(ValueError, match=error_pattern):
        agent.import_memory_state({"system_state": system_state}, context_id=1)


def test_typed_expansion_traverses_incoming_edges_with_budget_and_confidence():
    system = _empty_system()
    for memory_id in ("old", "new", "detail", "weak", "topic"):
        system.memories[memory_id] = _memory(memory_id)
    for source, target, relation_type, confidence in [
        ("new", "old", "SUPERSEDE", 0.95),
        ("detail", "new", "REFINE", 0.85),
        ("weak", "old", "SUPPORT", 0.2),
        ("topic", "old", "RELATED", 0.99),
    ]:
        system.add_typed_relation({
            "source_id": source,
            "target_id": target,
            "relation_type": relation_type,
            "direction": "NEW_MEMORY_TO_EXISTING_MEMORY",
            "confidence": confidence,
            "reason": "test",
        })

    selected, expansions = system.expand_typed_relations(
        ["old"], expansion_budget=2, min_confidence=0.5, include_related=False
    )

    assert selected == ["old", "new", "detail"]
    assert expansions[0]["traversal_direction"] == "incoming"
    assert [item["relation"]["relation_type"] for item in expansions] == [
        "SUPERSEDE", "REFINE"
    ]

    selected_related, _ = system.expand_typed_relations(
        ["old"], expansion_budget=3, min_confidence=0.5, include_related=True
    )
    assert "topic" in selected_related
    assert "weak" not in selected_related

def test_hybrid_candidate_retrieval_fuses_deterministic_channels():
    system = _experimental_system(temporal=False)
    system.memories["dense"] = _timestamped_memory(
        "general nutrition note", "2024-01-01"
    )
    system.memories["match"] = _timestamped_memory(
        "metformin medication dose", "2024-02-10"
    )
    system.memories["other"] = _timestamped_memory(
        "blood pressure reading", "2024-03-01"
    )
    system.retriever = SimpleNamespace(search=lambda query, k: [0, 1, 2])
    weights = {
        "dense": 0.25,
        "bm25": 1.0,
        "entity_attribute": 1.0,
        "timestamp": 1.0,
        "state": 0.0,
        "graph": 0.0,
    }

    first = system.hybrid_candidate_retrieval(
        "metformin medication",
        "Which medication was used in February 2024?",
        k=2,
        candidate_count=3,
        rrf_k=60.0,
        channel_weights=weights,
        use_typed_relations=False,
        use_ordinary_links=False,
    )
    second = system.hybrid_candidate_retrieval(
        "metformin medication",
        "Which medication was used in February 2024?",
        k=2,
        candidate_count=3,
        rrf_k=60.0,
        channel_weights=weights,
        use_typed_relations=False,
        use_ordinary_links=False,
    )

    assert first == second
    assert first["selected_memory_ids"][0] == "match"
    assert first["channel_rankings"]["dense"][0] == "dense"
    assert first["channel_rankings"]["bm25"][0] == "match"
    assert first["channel_rankings"]["timestamp"] == ["match"]
    assert first["memory_scores"][0]["ranks"] == {
        "dense": 2,
        "bm25": 1,
        "entity_attribute": 1,
        "timestamp": 1,
    }
    assert sum(first["seed_scores"].values()) == pytest.approx(1.0)

def test_query_conditioned_typed_ppr_prefers_current_superseder():
    system = _experimental_system()
    system.memories["old"] = _timestamped_memory("old treatment", "2024-01-01")
    system.memories["new"] = _timestamped_memory("current treatment", "2024-02-01")
    system.memories["detail"] = _timestamped_memory("supporting detail", "2024-02-02")
    system.add_typed_relation(_edge("new", "old", "SUPERSEDE"))
    system.add_typed_relation(_edge("detail", "old", "SUPPORT"))
    relation_weights = {
        "SUPERSEDE": 1.25,
        "CONFLICT": 0.75,
        "REFINE": 1.15,
        "SUPPORT": 1.0,
        "RELATED": 0.5,
    }

    untyped = system.query_conditioned_graph_rank(
        {"old": 1.0},
        "What is the current treatment now?",
        mode="untyped_ppr",
        expansion_budget=2,
        alpha=0.85,
        iterations=100,
        tolerance=1e-12,
        relation_weights=relation_weights,
    )
    typed = system.query_conditioned_graph_rank(
        {"old": 1.0},
        "What is the current treatment now?",
        mode="typed_ppr",
        expansion_budget=2,
        alpha=0.85,
        iterations=300,
        tolerance=1e-12,
        relation_weights=relation_weights,
    )

    assert untyped["scores"]["new"] == pytest.approx(untyped["scores"]["detail"])
    assert typed["scores"]["new"] > typed["scores"]["detail"]
    assert typed["expanded_memories"][0]["added_memory_id"] == "new"
    assert typed["temporal_query"]["intent"] == "current"
    assert typed["converged"] is True

def test_query_conditioned_graph_rank_rejects_unknown_mode():
    system = _empty_system()
    system.memories["seed"] = _memory("seed")

    with pytest.raises(ValueError, match="Unsupported graph ranking mode"):
        system.query_conditioned_graph_rank(
            {"seed": 1.0},
            "question",
            mode="unknown",
            expansion_budget=1,
            alpha=0.85,
            iterations=20,
            tolerance=1e-6,
            relation_weights={},
        )

def test_typed_ppr_weights_relation_type_from_query_intent():
    system = _experimental_system(temporal=False)
    for memory_id in ("seed", "conflict", "detail"):
        system.memories[memory_id] = _memory(memory_id)
    system.add_typed_relation(_edge("conflict", "seed", "CONFLICT"))
    system.add_typed_relation(_edge("detail", "seed", "REFINE"))
    relation_weights = {relation_type: 1.0 for relation_type in RELATION_TYPES}

    result = system.query_conditioned_graph_rank(
        {"seed": 1.0},
        "Which statements conflict or contradict each other?",
        mode="typed_ppr",
        expansion_budget=2,
        alpha=0.85,
        iterations=200,
        tolerance=1e-9,
        relation_weights=relation_weights,
    )

    assert result["scores"]["conflict"] > result["scores"]["detail"]
    assert result["temporal_query"]["relation_intents"] == ["conflict"]


def test_typed_add_note_stores_memory_when_relation_inference_fails(monkeypatch):
    class _Note:
        def __init__(self, content, **kwargs):
            self.id = "new"
            self.content = content
            self.context = "General"
            self.keywords = []
            self.tags = []
            self.timestamp = "2025-01-02"
            self.links = []

    class _Retriever:
        def search(self, query, k):
            return [0]

        def add_documents(self, documents):
            self.documents = documents

    import memory_layer_typed

    monkeypatch.setattr(memory_layer_typed, "RobustMemoryNote", _Note)
    system = _empty_system()
    system.original_evolution_enabled = True
    system.typed_relations_enabled = True
    system.relation_candidate_count = 5
    system.relation_audit = []
    system._relation_audit_by_memory = {}
    system.memories["old"] = _memory("old state")
    system.retriever = _Retriever()
    system.process_memory = lambda note: (False, note)
    system.llm_controller = SimpleNamespace(
        llm=SimpleNamespace(
            get_completion=lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("bad relation call"))
        )
    )
    system.max_context_chars = 1000
    system.evo_cnt = 0
    system.evo_threshold = 100

    note_id = system.add_note("new state")

    assert note_id == "new"
    assert system.memories["new"].content == "new state"
    assert system.typed_relations == []
    assert "bad relation call" in system.get_relation_audit("new")["relation_inference_error"]


def test_feature_disabled_delegates_to_original_memory_add_note(monkeypatch):
    calls = []
    monkeypatch.setattr(
        RobustAgenticMemorySystem,
        "add_note",
        lambda self, content, time=None, **kwargs: calls.append((content, time, kwargs)) or "base-id",
    )
    system = object.__new__(TypedRelationMemorySystem)
    system.original_evolution_enabled = True
    system.typed_relations_enabled = False

    assert system.add_note("unchanged baseline input", time="2025-01-01") == "base-id"
    assert calls == [("unchanged baseline input", "2025-01-01", {})]


def test_original_amem_loader_and_query_path_remain_separate():
    assert AMemAgent._load_amem_system_class(object()).__name__ == (
        "RobustAgenticMemorySystem"
    )
    assert AMemTestAgent._load_amem_system_class(object()).__name__ == (
        "TypedRelationMemorySystem"
    )

    original = object.__new__(AMemAgent)
    original._context_id = 3
    original.retrieve_num = 5
    original.amem_max_tokens = 100
    original.amem_max_context_tokens = 2000
    original._tokenizer = _Tokenizer()
    original._llm_client = _LLMClient()
    original._get_memory_system = lambda context_id: SimpleNamespace(
        find_related_memories=lambda question, k: ("original note", [0])
    )

    prepared = original.prepare_batch_query("What changed?", system_message="system")

    assert prepared["extra"] == {"method": "amem", "context_id": 3}
    assert prepared["retrieved_memories"][0]["type"] == "amem_retrieval"


def test_original_evolution_disabled_preserves_notes_and_builds_typed_edges(monkeypatch):
    class _Note:
        next_id = 0

        def __init__(self, content, **kwargs):
            type(self).next_id += 1
            self.id = f"new-{self.next_id}"
            self.content = content
            self.context = "Initial context"
            self.keywords = ["initial"]
            self.tags = ["initial"]
            self.timestamp = kwargs.get("timestamp") or "2025-01-02"
            self.links = []

    class _Retriever:
        def search(self, query, k):
            return [0]

        def add_documents(self, documents):
            self.documents = documents

    import memory_layer_typed

    monkeypatch.setattr(memory_layer_typed, "RobustMemoryNote", _Note)
    system = _empty_system()
    system.original_evolution_enabled = False
    system.typed_relations_enabled = True
    system.relation_candidate_count = 5
    system.relation_audit = []
    system._relation_audit_by_memory = {}
    old_memory = _memory("old state")
    old_memory.context = "Old context"
    old_memory.tags = ["old-tag"]
    system.memories["old"] = old_memory
    system.retriever = _Retriever()
    system.process_memory = lambda note: (_ for _ in ()).throw(
        AssertionError("original evolution must be skipped")
    )
    system._infer_relations = lambda note, candidate_ids: ([{
        "source_id": note.id,
        "target_id": candidate_ids[0],
        "relation_type": "REFINE",
        "direction": "NEW_MEMORY_TO_EXISTING_MEMORY",
        "confidence": 0.9,
        "reason": "adds detail",
        "candidate_position": 0,
    }], "raw")
    system.llm_controller = SimpleNamespace()
    system.max_context_chars = 1000
    system.evo_cnt = 0
    system.evo_threshold = 100

    note_id = system.add_note("new state")

    assert note_id == "new-1"
    assert system.memories[note_id].links == []
    assert system.memories[note_id].context == "Initial context"
    assert old_memory.content == "old state"
    assert old_memory.context == "Old context"
    assert old_memory.tags == ["old-tag"]
    assert old_memory.links == []
    assert system.typed_relations[0]["source_id"] == note_id
    assert system.typed_relations[0]["target_id"] == "old"


def test_feature_disabled_keeps_original_query_messages():
    memories = OrderedDict([
        ("stable-id", _memory("stored memory")),
        ("linked-id", _memory("linked memory")),
    ])
    memories["stable-id"].links = [1]
    memory_system = SimpleNamespace(
        memories=memories,
        retriever=SimpleNamespace(search=lambda question, k: [0]),
    )

    def make_agent(agent_class):
        agent = object.__new__(agent_class)
        agent._context_id = 1
        agent.retrieve_num = 10
        agent.amem_query_keywords = False
        agent.amem_expand_links = True
        agent.amem_max_tokens = 100
        agent.amem_max_context_tokens = 2000
        agent.max_tokens = 100
        agent._llm_client = _LLMClient()
        agent._tokenizer = _Tokenizer()
        agent._get_memory_system = lambda context_id: memory_system
        return agent

    baseline = make_agent(AMemFixAgent)
    experimental = make_agent(AMemTestAgent)
    experimental.amem_typed_relations = False

    baseline_prepared = baseline.prepare_batch_query(
        "formatted question", system_message="system", raw_question="raw question"
    )
    test_prepared = experimental.prepare_batch_query(
        "formatted question", system_message="system", raw_question="raw question"
    )

    assert test_prepared["messages"] == baseline_prepared["messages"]
    assert test_prepared["retrieved_count"] == baseline_prepared["retrieved_count"]
    assert test_prepared["extra"]["semantic_seed_ids"] == ["stable-id"]
    assert test_prepared["extra"]["final_memory_ids"] == ["stable-id", "linked-id"]
    assert test_prepared["extra"]["retrieval_query"] == baseline_prepared["extra"]["retrieval_query"]


def test_amem_test_memorization_is_amem_fix_atomic_flow():
    calls = []
    memory_system = SimpleNamespace(
        add_note=lambda **kwargs: calls.append(kwargs) or f"note-{len(calls)}"
    )
    agent = object.__new__(AMemTestAgent)
    agent._context_id = 7
    agent._memory_chunks = []
    agent._is_initialized = False
    agent.retrieve_num = 10
    agent.amem_chunk_size_tokens = 10240
    agent.amem_original_evolution = True
    agent.amem_typed_relations = False
    agent.amem_relation_candidate_count = 5
    agent._llm_client = _LLMClient()
    agent._get_memory_system = lambda context_id: memory_system

    result = agent.memorize(
        "formatted wrapper",
        timestamp="2025-01-02",
        memory_items=[
            {"role": "user", "content": "First turn."},
            {"role": "assistant", "content": "Second turn."},
        ],
    )

    assert calls == [
        {"content": "Speaker Patient says: First turn.", "time": "2025-01-02"},
        {"content": "Speaker Doctor says: Second turn.", "time": "2025-01-02"},
    ]
    assert result.method == "amem_test"
    assert result.extra["turns_received"] == 2
    assert result.extra["notes_created"] == 2
    assert "formatted wrapper" not in result.stored_content


def test_amem_test_provenance_memorization_passes_real_source_metadata():
    calls = []

    class _MemorySystem:
        def add_note(self, **kwargs):
            calls.append(deepcopy(kwargs))
            return "note-1"

        def get_relation_audit(self, memory_id):
            return {}

        def get_provenance_audit(self, memory_id):
            return {
                "memory_id": memory_id,
                "evidence_ids": ["ev_test"],
                "source_session_id": 12,
                "source_timestamp": "2024-04-03",
                "error": "",
            }

    from copy import deepcopy

    memory_system = _MemorySystem()
    agent = object.__new__(AMemTestAgent)
    agent._context_id = 7
    agent._memory_chunks = []
    agent._is_initialized = False
    agent.retrieve_num = 10
    agent.amem_chunk_size_tokens = 10240
    agent.amem_original_evolution = False
    agent.amem_typed_relations = False
    agent.amem_temporal_state = False
    agent.amem_provenance = True
    agent.amem_relation_candidate_count = 5
    agent._llm_client = _LLMClient()
    agent._get_memory_system = lambda context_id: memory_system

    result = agent.memorize(
        "formatted wrapper",
        timestamp="2024-04-03",
        source_session_id=12,
        source_session_index=4,
        source_event_id="event-12",
        memory_items=[
            {"role": "user", "turn": 3, "content": "My value was exactly 16.5 mmol/L."},
        ],
    )

    assert calls[0]["time"] == "2024-04-03"
    assert calls[0]["source_evidence"] == {
        "raw_text": "My value was exactly 16.5 mmol/L.",
        "source_context_id": 7,
        "source_session_id": 12,
        "source_session_index": 4,
        "source_event_id": "event-12",
        "source_turn_id": 3,
        "source_timestamp": "2024-04-03",
        "speaker": "Patient",
        "role": "user",
        "blip_caption": None,
        "source_text_scope": "source_turn",
    }
    assert result.memory_entries[0]["provenance"]["evidence_ids"] == ["ev_test"]
    assert result.extra["evidence_count"] == 1


def test_chunked_memories_share_stable_source_evidence_with_part_references():
    calls = []
    memory_system = SimpleNamespace(
        add_note=lambda **kwargs: calls.append(kwargs) or f"note-{len(calls)}",
        get_relation_audit=lambda memory_id: {},
        get_provenance_audit=lambda memory_id: {
            "memory_id": memory_id,
            "evidence_ids": [f"ev-{memory_id}"],
            "error": "",
        },
    )
    agent = object.__new__(AMemTestAgent)
    agent._context_id = 1
    agent._memory_chunks = []
    agent._is_initialized = False
    agent.retrieve_num = 10
    agent.amem_chunk_size_tokens = 2
    agent.amem_original_evolution = False
    agent.amem_typed_relations = False
    agent.amem_temporal_state = False
    agent.amem_provenance = True
    agent.amem_relation_candidate_count = 5
    agent._llm_client = _LLMClient()
    agent._get_memory_system = lambda context_id: memory_system
    agent._split_text_into_chunks = lambda text, max_tokens: ["part one", "part two"]

    agent.memorize(
        "wrapper",
        memory_items=[{"role": "user", "content": "whole source turn"}],
    )

    assert [call["source_evidence"]["raw_text"] for call in calls] == [
        "whole source turn",
        "whole source turn",
    ]
    assert [call["provenance_part_index"] for call in calls] == [0, 1]
    assert all(
        call["source_evidence"]["source_text_scope"] == "source_turn"
        for call in calls
    )


def test_relation_aware_context_explicitly_shows_supersession():
    system = _empty_system()
    system.memories["old"] = _memory("I work at Google")
    system.memories["new"] = _memory("I left Google and joined Microsoft")
    relation = {
        "source_id": "new",
        "target_id": "old",
        "relation_type": "SUPERSEDE",
        "direction": "NEW_MEMORY_TO_EXISTING_MEMORY",
        "confidence": 0.95,
        "reason": "explicit replacement",
    }

    context = AMemTestAgent._format_relation_context(
        system,
        final_ids=["old", "new"],
        seed_ids=["old"],
        expansions=[{"added_memory_id": "new"}],
        relations=[relation],
    )

    assert "M2 --SUPERSEDE (confidence=0.95)--> M1" in context
    assert "I work at Google" in context
    assert "I left Google and joined Microsoft" in context


def test_typed_query_extends_amem_fix_keyword_and_link_retrieval():
    keyword_prompts = []
    search_queries = []
    system = _empty_system()
    system.memories["m0"] = _memory("semantic seed")
    system.memories["m1"] = _memory("original A-MEM linked note")
    system.memories["m2"] = _memory("new current state")
    system.memories["m0"].links = [1]
    system.retriever = SimpleNamespace(
        search=lambda query, k: search_queries.append((query, k)) or [0]
    )
    system.llm_controller = SimpleNamespace(
        llm=SimpleNamespace(
            get_completion=lambda prompt: keyword_prompts.append(prompt)
            or '{"keywords": "current state"}'
        )
    )
    system.add_typed_relation({
        "source_id": "m2",
        "target_id": "m1",
        "relation_type": "SUPERSEDE",
        "direction": "NEW_MEMORY_TO_EXISTING_MEMORY",
        "confidence": 0.95,
        "reason": "replacement",
    })

    agent = object.__new__(AMemTestAgent)
    agent._context_id = 3
    agent.retrieve_num = 10
    agent.amem_query_keywords = True
    agent.amem_expand_links = True
    agent.amem_typed_relations = True
    agent.amem_typed_expansion_count = 5
    agent.amem_expand_related = False
    agent.amem_relation_min_confidence = 0.5
    agent.amem_max_context_tokens = 2000
    agent.max_tokens = 100
    agent._tokenizer = _Tokenizer()
    agent._llm_client = _LLMClient()
    agent._get_memory_system = lambda context_id: system

    prepared = agent.prepare_batch_query(
        "formatted question",
        system_message="system",
        raw_question="What is the current state?",
    )

    assert "What is the current state?" in keyword_prompts[0]
    assert search_queries == [("current state", 10)]
    assert prepared["extra"]["direct_indices"] == [0]
    assert prepared["extra"]["amem_fix_expanded_indices"] == [0, 1]
    assert prepared["extra"]["final_memory_ids"] == ["m0", "m1", "m2"]
    assert prepared["retrieved_memories"][1]["linked_expansion"] is True
    assert prepared["retrieved_memories"][2]["typed_expansion"] is not None
    assert "M3 --SUPERSEDE (confidence=0.95)--> M2" in prepared["messages"][-1]["content"]


def test_combined_typed_temporal_provenance_retrieval():
    system = _experimental_system(temporal=True, provenance=True)
    system.memories["old"] = _timestamped_memory("I work at Google", "2024-01-01")
    system.memories["new"] = _timestamped_memory(
        "Speaker Patient says: I left Google and joined Microsoft.",
        "2024-02-01",
    )
    system.memories["old"].id = "old"
    system.memories["new"].id = "new"
    system._attach_provenance(
        system.memories["new"],
        {
            "raw_text": "I left Google and joined Microsoft.",
            "source_session_id": 2,
            "source_turn_id": 0,
            "source_timestamp": "2024-02-01",
        },
        0,
    )
    system.add_typed_relation(_edge("new", "old", "SUPERSEDE"))
    system.retriever = SimpleNamespace(search=lambda query, k: [0])
    system.llm_controller = SimpleNamespace(llm=SimpleNamespace())

    agent = object.__new__(AMemTestAgent)
    agent._context_id = 3
    agent.retrieve_num = 1
    agent.amem_query_keywords = False
    agent.amem_expand_links = False
    agent.amem_typed_relations = True
    agent.amem_typed_expansion_count = 0
    agent.amem_expand_related = False
    agent.amem_relation_min_confidence = 0.5
    agent.amem_temporal_state = True
    agent.amem_temporal_expansion_count = 5
    agent.amem_provenance = True
    agent.amem_provenance_max_evidence = 5
    agent.amem_max_context_tokens = 2000
    agent.max_tokens = 100
    agent._tokenizer = _Tokenizer()
    agent._llm_client = _LLMClient()
    agent._get_memory_system = lambda context_id: system

    prepared = agent.prepare_batch_query(
        "formatted question",
        system_message="system",
        raw_question="Where does the user work now?",
    )

    assert prepared["extra"]["semantic_seed_ids"] == ["old"]
    assert prepared["extra"]["temporal_expanded_memories"][0]["added_memory_id"] == "new"
    assert prepared["extra"]["final_memory_ids"][0] == "new"
    assert prepared["extra"]["temporal_states"]["old"]["status"] == "superseded"
    assert prepared["extra"]["final_evidence_ids"]
    assert prepared["retrieved_memories"][0]["memory_id"] == "new"
    context = prepared["messages"][-1]["content"]
    assert "status=current" in context
    assert "source_session_id" not in context
    assert "session=2" in context
    assert "raw text injection disabled" in context


def test_retrieval_switches_disable_built_features_without_rebuilding():
    system = _experimental_system(temporal=True, provenance=True)
    system.memories["old"] = _timestamped_memory("I work at Google", "2024-01-01")
    system.memories["new"] = _timestamped_memory(
        "I left Google and joined Microsoft.", "2024-02-01"
    )
    system.memories["old"].id = "old"
    system.memories["new"].id = "new"
    system._attach_provenance(
        system.memories["new"],
        {"raw_text": "I left Google and joined Microsoft."},
        0,
    )
    system.add_typed_relation(_edge("new", "old", "SUPERSEDE"))
    system.retriever = SimpleNamespace(search=lambda query, k: [0])
    system.llm_controller = SimpleNamespace(llm=SimpleNamespace())

    agent = object.__new__(AMemTestAgent)
    agent._context_id = 3
    agent.retrieve_num = 1
    agent.amem_query_keywords = False
    agent.amem_expand_links = False
    agent.amem_typed_relations = True
    agent.amem_typed_retrieval = False
    agent.amem_typed_expansion_count = 5
    agent.amem_expand_related = False
    agent.amem_relation_min_confidence = 0.5
    agent.amem_temporal_state = True
    agent.amem_temporal_retrieval = False
    agent.amem_temporal_ordering = True
    agent.amem_temporal_expansion_count = 5
    agent.amem_provenance = True
    agent.amem_provenance_retrieval = False
    agent.amem_provenance_max_evidence = 5
    agent.amem_max_context_tokens = 2000
    agent.max_tokens = 100
    agent._tokenizer = _Tokenizer()
    agent._llm_client = _LLMClient()
    agent._get_memory_system = lambda context_id: system

    prepared = agent.prepare_batch_query(
        "formatted question",
        system_message="system",
        raw_question="Where does the user work now?",
    )

    assert prepared["extra"]["final_memory_ids"] == ["old"]
    assert prepared["extra"]["typed_relations"] == []
    assert prepared["extra"]["temporal_expanded_memories"] == []
    assert prepared["extra"]["provenance_evidence"] == []
    context = prepared["messages"][-1]["content"]
    assert "[Typed Memory Relations]" not in context
    assert "[Temporal State / Validity]" not in context
    assert "[Immutable Source Evidence]" not in context

@pytest.mark.parametrize(
    "graph_mode, expected_ids, typed_context",
    [
        ("none", ["old"], False),
        ("fixed_bfs", ["old", "new"], True),
        ("untyped_ppr", ["old", "new"], False),
        ("typed_ppr", ["old", "new"], True),
    ],
)
def test_query_pipeline_supports_all_graph_ranking_modes(
    graph_mode,
    expected_ids,
    typed_context,
):
    system = _experimental_system()
    system.memories["old"] = _timestamped_memory("old state", "2024-01-01")
    system.memories["new"] = _timestamped_memory("new current state", "2024-02-01")
    system.add_typed_relation(_edge("new", "old", "SUPERSEDE"))
    system.retriever = SimpleNamespace(search=lambda query, k: [0])
    system.llm_controller = SimpleNamespace(llm=SimpleNamespace())

    agent = object.__new__(AMemTestAgent)
    agent._context_id = 3
    agent.retrieve_num = 1
    agent.amem_query_keywords = False
    agent.amem_expand_links = False
    agent.amem_typed_relations = True
    agent.amem_typed_retrieval = True
    agent.amem_typed_expansion_count = 1
    agent.amem_expand_related = False
    agent.amem_relation_min_confidence = 0.5
    agent.amem_temporal_state = True
    agent.amem_temporal_retrieval = False
    agent.amem_temporal_ordering = False
    agent.amem_provenance = False
    agent.amem_provenance_retrieval = False
    agent.amem_hybrid_retrieval = False
    agent.amem_graph_ranking_mode = graph_mode
    agent.amem_graph_alpha = 0.85
    agent.amem_graph_iterations = 100
    agent.amem_graph_tolerance = 1e-12
    agent.amem_graph_relation_weights = {
        "SUPERSEDE": 1.25,
        "CONFLICT": 0.75,
        "REFINE": 1.15,
        "SUPPORT": 1.0,
        "RELATED": 0.5,
    }
    agent.amem_max_context_tokens = 2000
    agent.max_tokens = 100
    agent._tokenizer = _Tokenizer()
    agent._llm_client = _LLMClient()
    agent._get_memory_system = lambda context_id: system

    prepared = agent.prepare_batch_query(
        "formatted question",
        system_message="system",
        raw_question="What is the current state now?",
    )

    assert prepared["extra"]["graph_ranking_mode"] == graph_mode
    assert prepared["extra"]["final_memory_ids"] == expected_ids
    context = prepared["messages"][-1]["content"]
    assert ("[Typed Memory Relations]" in context) is typed_context

def test_hybrid_retrieval_audit_records_rrf_channels_and_scores():
    system = _experimental_system(temporal=False)
    system.memories["dense"] = _memory("general note")
    system.memories["match"] = _memory("metformin medication")
    system.retriever = SimpleNamespace(search=lambda query, k: [0, 1])
    system.llm_controller = SimpleNamespace(llm=SimpleNamespace())

    agent = object.__new__(AMemTestAgent)
    agent._context_id = 3
    agent.retrieve_num = 1
    agent.amem_query_keywords = False
    agent.amem_expand_links = False
    agent.amem_typed_relations = False
    agent.amem_typed_retrieval = False
    agent.amem_typed_expansion_count = 0
    agent.amem_expand_related = False
    agent.amem_relation_min_confidence = 0.5
    agent.amem_temporal_state = False
    agent.amem_temporal_retrieval = False
    agent.amem_provenance = False
    agent.amem_provenance_retrieval = False
    agent.amem_hybrid_retrieval = True
    agent.amem_hybrid_candidate_count = 2
    agent.amem_hybrid_rrf_k = 60.0
    agent.amem_hybrid_channel_weights = {
        "dense": 0.1,
        "bm25": 1.0,
        "entity_attribute": 1.0,
        "timestamp": 0.0,
        "state": 0.0,
        "graph": 0.0,
    }
    agent.amem_graph_ranking_mode = "none"
    agent.amem_max_context_tokens = 2000
    agent.max_tokens = 100
    agent._tokenizer = _Tokenizer()
    agent._llm_client = _LLMClient()
    agent._get_memory_system = lambda context_id: system

    prepared = agent.prepare_batch_query(
        "formatted question",
        system_message="system",
        raw_question="Which metformin medication?",
    )

    assert prepared["extra"]["final_memory_ids"] == ["match"]
    assert prepared["extra"]["hybrid_channel_rankings"]["dense"] == [
        "dense", "match"
    ]
    assert prepared["extra"]["hybrid_channel_rankings"]["bm25"] == ["match"]
    assert prepared["retrieved_memories"][0]["hybrid_retrieval"]["ranks"] == {
        "dense": 2,
        "bm25": 1,
        "entity_attribute": 1,
    }
    assert "[Typed Memory Relations]" not in prepared["messages"][-1]["content"]

def test_chain_selector_adds_complete_path_atomically_with_exact_budget():
    system = _experimental_system(temporal=False)
    system.memories["start"] = _memory("alpha finding")
    system.memories["bridge"] = _memory("intermediate supporting evidence")
    system.memories["end"] = _memory("gamma outcome")
    system.add_typed_relation(_edge("bridge", "start", "SUPPORT"))
    system.add_typed_relation(_edge("end", "bridge", "SUPPORT"))

    selected = system.select_chain_preserving_evidence(
        ["start", "bridge", "end"],
        "How did alpha lead to gamma?",
        token_budget=30,
        token_cost=lambda memory_ids: len(memory_ids) * 10,
        max_hops=2,
        max_groups=1,
        relevance_weight=0.0,
        coverage_weight=0.0,
        connectivity_weight=0.0,
        path_weight=1.0,
        temporal_weight=0.0,
        redundancy_weight=0.0,
    )
    too_small = system.select_chain_preserving_evidence(
        ["start", "bridge", "end"],
        "How did alpha lead to gamma?",
        token_budget=29,
        token_cost=lambda memory_ids: len(memory_ids) * 10,
        max_hops=2,
        max_groups=1,
        relevance_weight=0.0,
        coverage_weight=0.0,
        connectivity_weight=0.0,
        path_weight=1.0,
        temporal_weight=0.0,
        redundancy_weight=0.0,
    )

    assert selected["selected_memory_ids"] == ["start", "bridge", "end"]
    assert selected["selected_tokens"] == 30
    assert selected["selection_steps"][0]["action_type"] == "path"
    assert selected["selection_steps"][0]["added_memory_ids"] == [
        "start", "bridge", "end"
    ]
    assert too_small["selected_memory_ids"] == ["start", "bridge"]
    assert too_small["selected_tokens"] == 20
    assert too_small["selection_steps"][0]["action_type"] == "path"
    assert "end" not in too_small["selection_steps"][0]["added_memory_ids"]

def test_chain_selector_transition_coverage_requires_both_edge_endpoints():
    system = _experimental_system(temporal=False)
    system.memories["old"] = _memory("old treatment")
    system.memories["new"] = _memory("new treatment")
    system.add_typed_relation(_edge("new", "old", "SUPERSEDE"))

    result = system.select_chain_preserving_evidence(
        ["old", "new"],
        "What changed?",
        token_budget=20,
        token_cost=lambda memory_ids: len(memory_ids) * 10,
        max_hops=1,
        max_groups=1,
        relevance_weight=0.0,
        coverage_weight=1.0,
        connectivity_weight=0.0,
        path_weight=0.0,
        temporal_weight=0.0,
        redundancy_weight=0.0,
    )
    singleton = system.select_chain_preserving_evidence(
        ["old", "new"],
        "What changed?",
        token_budget=10,
        token_cost=lambda memory_ids: len(memory_ids) * 10,
        max_hops=1,
        max_groups=1,
        relevance_weight=0.0,
        coverage_weight=1.0,
        connectivity_weight=0.0,
        path_weight=0.0,
        temporal_weight=0.0,
        redundancy_weight=0.0,
    )

    assert result["selected_memory_ids"] == ["old", "new"]
    assert result["utility"]["coverage"] > singleton["utility"]["coverage"]
    assert singleton["utility"]["coverage"] < result["utility"]["coverage"]
    assert len(singleton["selected_memory_ids"]) == 1
    assert "transition" in result["question_facets"]["structural"]

def test_chain_selector_reaches_general_evidence_target_without_forced_ids():
    system = _experimental_system(temporal=False)
    candidate_ids = [f"memory-{index}" for index in range(40)]
    for index, memory_id in enumerate(candidate_ids):
        system.memories[memory_id] = _memory(
            f"clinical evidence item {index}"
        )

    result = system.select_chain_preserving_evidence(
        candidate_ids,
        "Which clinical evidence is relevant?",
        candidate_rankings={
            "upstream": candidate_ids[:20],
            "dense": list(reversed(candidate_ids[20:])),
        },
        token_budget=300,
        token_cost=lambda memory_ids: len(memory_ids) * 10,
        candidate_count=40,
        evidence_count=30,
        max_groups=1,
        redundancy_weight=1.0,
    )

    assert len(result["selected_memory_ids"]) == 30
    assert result["selected_tokens"] == 300
    assert result["target_evidence_count"] == 30
    assert any(
        memory_id not in candidate_ids[:20]
        for memory_id in result["selected_memory_ids"]
    )
    assert {item["reason"] for item in result["rejected_candidates"]} == {
        "evidence_count"
    }

def test_chain_selector_uses_soft_group_preference_not_hard_rejection():
    system = _experimental_system(temporal=False)
    candidate_ids = [f"memory-{index}" for index in range(5)]
    for memory_id in candidate_ids:
        system.memories[memory_id] = _memory(memory_id)

    result = system.select_chain_preserving_evidence(
        candidate_ids,
        "List the relevant memories",
        token_budget=50,
        token_cost=lambda memory_ids: len(memory_ids) * 10,
        evidence_count=5,
        max_groups=1,
    )

    assert result["selected_memory_ids"] == candidate_ids
    assert result["utility"]["component_count"] == 5
    assert result["utility"]["group_penalty"] > 0.0

def test_chain_selector_gates_temporal_state_and_matches_direct_dates():
    system = _experimental_system(temporal=False)
    system.memories["old"] = _timestamped_memory("old note", "2024-01-05")
    system.memories["new"] = _timestamped_memory("new note", "2024-02-05")

    historical = system.select_chain_preserving_evidence(
        ["old", "new"],
        "What happened previously?",
        token_budget=20,
        token_cost=lambda memory_ids: len(memory_ids) * 10,
        evidence_count=2,
        temporal_weight=1.0,
        temporal_state_enabled=False,
    )
    dated = system.select_chain_preserving_evidence(
        ["old", "new"],
        "What happened on 2024-01-05?",
        token_budget=20,
        token_cost=lambda memory_ids: len(memory_ids) * 10,
        evidence_count=2,
        temporal_weight=1.0,
        temporal_state_enabled=False,
    )

    assert "historical_state" not in historical["question_facets"]["structural"]
    assert historical["utility"]["temporal"] == 0.0
    assert dated["candidate_scores"]["old"]["facet_matches"]["target_time"] == 1.0
    assert dated["candidate_scores"]["new"]["facet_matches"]["target_time"] == 0.0
    assert dated["utility"]["temporal"] == 1.0

def test_chain_selector_fuses_channels_before_candidate_limit():
    system = _experimental_system(temporal=False)
    candidate_ids = [f"memory-{index}" for index in range(8)]
    for memory_id in candidate_ids:
        system.memories[memory_id] = _memory(memory_id)

    result = system.select_chain_preserving_evidence(
        candidate_ids,
        "memory evidence",
        candidate_rankings={
            "upstream": candidate_ids[:4],
            "dense": candidate_ids[4:],
        },
        token_budget=40,
        token_cost=lambda memory_ids: len(memory_ids) * 10,
        candidate_count=4,
        evidence_count=4,
    )

    assert set(result["candidate_memory_ids"]) != set(candidate_ids[:4])
    assert any(
        memory_id in candidate_ids[4:]
        for memory_id in result["candidate_memory_ids"]
    )

def test_chain_selector_reserves_budget_for_feasible_evidence_target():
    system = _experimental_system(temporal=False)
    candidate_ids = ["large", "small-a", "small-b", "small-c"]
    for memory_id in candidate_ids:
        system.memories[memory_id] = _memory(memory_id)
    costs = {"large": 25, "small-a": 10, "small-b": 10, "small-c": 10}

    result = system.select_chain_preserving_evidence(
        candidate_ids,
        "large evidence",
        candidate_rankings={"dense": candidate_ids},
        token_budget=30,
        token_cost=lambda memory_ids: sum(costs[item] for item in memory_ids),
        candidate_count=4,
        evidence_count=3,
    )

    assert result["selected_memory_ids"] == ["small-a", "small-b", "small-c"]
    assert result["selected_tokens"] == 30

def test_chain_selector_requires_resolved_bounds_for_relative_dates():
    system = _experimental_system(temporal=False)
    system.memories["january"] = _timestamped_memory(
        "January evidence", "2024-01-05"
    )
    system.memories["march"] = _timestamped_memory(
        "March evidence", "2024-03-05"
    )
    unresolved_query = detect_temporal_query("What happened two months ago?")
    unresolved_query["intent"] = "time_specific"

    unresolved = system.select_chain_preserving_evidence(
        ["january", "march"],
        "What happened two months ago?",
        token_budget=20,
        token_cost=lambda memory_ids: len(memory_ids) * 10,
        evidence_count=2,
        temporal_state_enabled=False,
        temporal_query=unresolved_query,
    )
    resolved_query = detect_temporal_query("What happened two months ago?")
    resolved_query["intent"] = "time_specific"
    resolved_query["target"].update({
        "start": "2024-01-01T00:00:00",
        "end": "2024-02-01T00:00:00",
    })
    resolved = system.select_chain_preserving_evidence(
        ["january", "march"],
        "What happened two months ago?",
        token_budget=20,
        token_cost=lambda memory_ids: len(memory_ids) * 10,
        evidence_count=2,
        temporal_state_enabled=False,
        temporal_query=resolved_query,
    )

    assert unresolved["candidate_scores"]["january"]["facet_matches"][
        "target_time"
    ] == 0.0
    assert unresolved["candidate_scores"]["march"]["facet_matches"][
        "target_time"
    ] == 0.0
    assert resolved["candidate_scores"]["january"]["facet_matches"][
        "target_time"
    ] == 1.0
    assert resolved["candidate_scores"]["march"]["facet_matches"][
        "target_time"
    ] == 0.0

def test_chain_selector_uses_all_bounded_candidates_as_path_anchors():
    system = _experimental_system(temporal=False)
    candidate_ids = [f"memory-{index}" for index in range(9)]
    for index, memory_id in enumerate(candidate_ids):
        content = "alpha finding" if index == 0 else f"unrelated note {index}"
        if index == 8:
            content = "gamma outcome"
        system.memories[memory_id] = _memory(content)
    system.add_typed_relation(_edge("memory-8", "memory-0", "SUPPORT"))

    result = system.select_chain_preserving_evidence(
        candidate_ids,
        "How did alpha lead to gamma?",
        token_budget=20,
        token_cost=lambda memory_ids: len(memory_ids) * 10,
        candidate_count=9,
        max_hops=1,
        max_groups=1,
        relevance_weight=0.0,
        coverage_weight=0.0,
        connectivity_weight=0.0,
        path_weight=1.0,
        temporal_weight=0.0,
        redundancy_weight=0.0,
    )
    empty = system.select_chain_preserving_evidence(
        candidate_ids,
        "How did alpha lead to gamma?",
        token_budget=0,
        token_cost=lambda memory_ids: len(memory_ids) * 10,
    )

    assert result["selected_memory_ids"] == ["memory-0", "memory-8"]
    assert result["selection_steps"][0]["action_type"] == "path"
    assert empty["evidence_graph_edges"] == []

def test_query_pipeline_chain_selection_widens_candidates_and_preserves_path():
    system = _experimental_system(temporal=False)
    system.memories["start"] = _memory("alpha finding")
    system.memories["bridge"] = _memory("intermediate supporting evidence")
    system.memories["end"] = _memory("gamma outcome")
    system.add_typed_relation(_edge("bridge", "start", "SUPPORT"))
    system.add_typed_relation(_edge("end", "bridge", "SUPPORT"))
    system.retriever = SimpleNamespace(
        search=lambda query, k: list(range(min(k, len(system.memories))))
    )
    system.llm_controller = SimpleNamespace(llm=SimpleNamespace())

    agent = object.__new__(AMemTestAgent)
    agent._context_id = 3
    agent.retrieve_num = 1
    agent.amem_query_keywords = False
    agent.amem_expand_links = False
    agent.amem_typed_relations = True
    agent.amem_typed_retrieval = True
    agent.amem_typed_expansion_count = 0
    agent.amem_expand_related = False
    agent.amem_relation_min_confidence = 0.5
    agent.amem_temporal_state = False
    agent.amem_temporal_retrieval = False
    agent.amem_temporal_ordering = False
    agent.amem_provenance = False
    agent.amem_provenance_retrieval = False
    agent.amem_hybrid_retrieval = False
    agent.amem_graph_ranking_mode = "fixed_bfs"
    agent.amem_graph_relation_weights = {
        "SUPERSEDE": 1.25,
        "CONFLICT": 0.75,
        "REFINE": 1.15,
        "SUPPORT": 1.0,
        "RELATED": 0.5,
    }
    agent.amem_chain_selection = True
    agent.amem_chain_candidate_count = 3
    agent.amem_chain_max_hops = 2
    agent.amem_chain_max_groups = 1
    agent.amem_chain_weights = {
        "relevance": 0.0,
        "coverage": 0.0,
        "connectivity": 0.0,
        "path": 1.0,
        "temporal": 0.0,
        "redundancy": 0.0,
    }
    agent.amem_max_context_tokens = 2000
    agent.max_tokens = 100
    agent._tokenizer = _Tokenizer()
    agent._llm_client = _LLMClient()
    agent._get_memory_system = lambda context_id: system

    prepared = agent.prepare_batch_query(
        "formatted question",
        system_message="system",
        raw_question="How did alpha lead to gamma?",
    )

    audit = prepared["extra"]["chain_selection"]
    context = prepared["extra"]["relation_aware_context"]
    assert audit["enabled"] is True
    assert audit["candidate_memory_ids"] == ["start", "bridge", "end"]
    assert audit["selected_memory_ids"] == ["start", "bridge", "end"]
    assert audit["selected_tokens"] == agent._llm_client.count_tokens(context)
    assert audit["selected_tokens"] <= audit["token_budget"]
    assert prepared["extra"]["final_memory_ids"] == [
        "start", "bridge", "end"
    ]
    assert "id=bridge | chain-selection candidate" in context
    assert "id=end | chain-selection candidate" in context
    assert "M2 --SUPPORT" in context
    assert "M3 --SUPPORT" in context


def test_retrieval_config_does_not_change_build_snapshot_hash():
    base = {
        "method_name": "amem_test",
        "method_type": "agentic_memory",
        "model": {"name": "test-model"},
        "build_config": {
            "amem_embedding_model": "all-MiniLM-L6-v2",
            "amem_typed_relations": True,
            "amem_temporal_state": True,
            "amem_provenance": True,
        },
        "retrieval_config": {
            "amem_typed_retrieval": True,
            "amem_temporal_retrieval": True,
            "amem_provenance_retrieval": True,
            "amem_temporal_ordering": True,
            "amem_hybrid_retrieval": False,
            "amem_graph_ranking_mode": "fixed_bfs",
            "amem_chain_selection": False,
        },
    }
    ablated = {**base, "retrieval_config": {
        **base["retrieval_config"],
        "amem_typed_retrieval": False,
        "amem_temporal_retrieval": False,
        "amem_provenance_retrieval": False,
        "amem_temporal_ordering": False,
        "amem_hybrid_retrieval": True,
        "amem_graph_ranking_mode": "typed_ppr",
        "amem_chain_selection": True,
    }}
    dataset = DatasetConfig(dataset_name="test")

    assert compute_config_hash(MethodConfig.from_dict(base), dataset) == compute_config_hash(
        MethodConfig.from_dict(ablated), dataset
    )


def test_build_context_and_temporal_transition_threshold_change_snapshot_hash():
    base = {
        "method_name": "amem_test",
        "method_type": "agentic_memory",
        "build_config": {
            "amem_embedding_model": "all-MiniLM-L6-v2",
            "amem_build_max_context_tokens": 200000,
            "amem_temporal_transition_min_confidence": 0.5,
        },
        "retrieval_config": {
            "amem_max_context_tokens": 200000,
            "amem_relation_min_confidence": 0.5,
        },
    }
    dataset = DatasetConfig(dataset_name="test")
    changed_context = {
        **base,
        "build_config": {
            **base["build_config"],
            "amem_build_max_context_tokens": 100000,
        },
    }
    changed_threshold = {
        **base,
        "build_config": {
            **base["build_config"],
            "amem_temporal_transition_min_confidence": 0.9,
        },
    }

    base_hash = compute_config_hash(MethodConfig.from_dict(base), dataset)
    assert compute_config_hash(MethodConfig.from_dict(changed_context), dataset) != base_hash
    assert compute_config_hash(MethodConfig.from_dict(changed_threshold), dataset) != base_hash


def test_legacy_amem_config_preserves_shared_build_and_retrieval_values():
    config = MethodConfig.from_dict({
        "method_name": "amem_test",
        "method_type": "agentic_memory",
        "model": {"provider": "vertex", "name": "actual-legacy-build-model"},
        "agent_params": {
            "amem_backend": "gemini",
            "amem_model": "ignored-legacy-build-model",
            "amem_max_context_tokens": 12345,
            "amem_relation_min_confidence": 0.8,
        },
    })

    assert config.build_config["amem_backend"] == "vertex"
    assert config.build_config["amem_model"] == "actual-legacy-build-model"
    assert config.build_config["amem_build_max_context_tokens"] == 12345
    assert config.build_config["amem_temporal_transition_min_confidence"] == 0.8
    assert config.retrieval_config["amem_max_context_tokens"] == 12345
    assert config.retrieval_config["amem_relation_min_confidence"] == 0.8


def test_split_snapshot_without_new_build_aliases_is_normalized():
    raw_config = {
        "method_name": "amem_test",
        "method_type": "agentic_memory",
        "build_config": {"amem_typed_relations": True},
        "retrieval_config": {
            "amem_max_context_tokens": 12345,
            "amem_relation_min_confidence": 0.8,
        },
    }
    config = method_config_from_snapshot({
        "raw_config": raw_config,
        "build_config": dict(raw_config["build_config"]),
        "retrieval_config": dict(raw_config["retrieval_config"]),
        "agent_params": {
            **raw_config["build_config"],
            **raw_config["retrieval_config"],
        },
    })

    assert config.build_config["amem_build_max_context_tokens"] == 12345
    assert config.build_config["amem_temporal_transition_min_confidence"] == 0.8


def test_split_amem_config_rejects_cross_owned_settings():
    with pytest.raises(ValueError, match="retrieval settings in build_config"):
        MethodConfig.from_dict({
            "method_name": "amem_test",
            "method_type": "agentic_memory",
            "build_config": {"amem_max_context_tokens": 1000},
        })


def test_amem_test_registration_creation_and_baseline_mapping(monkeypatch):
    assert list(AgentManager.SUPPORTED_METHODS).index("amem_test") < list(
        AgentManager.SUPPORTED_METHODS
    ).index("amem")
    assert AgentManager.SUPPORTED_METHODS["amem_fix"] == (
        "methods.amem_fix_agent", "AMemFixAgent"
    )
    assert issubclass(AMemTestAgent, AMemFixAgent)

    captured = {}

    def fake_init(self, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(AMemTestAgent, "__init__", fake_init)
    manager = object.__new__(AgentManager)
    manager.method_config = MethodConfig.from_dict({
        "method_name": "amem_test",
        "method_type": "agentic_memory",
        "model": {"provider": "openai", "name": "mock-model"},
        "agent_params": {
            "amem_original_evolution": False,
            "amem_typed_relations": False,
            "amem_typed_expansion_count": 7,
            "amem_hybrid_retrieval": True,
            "amem_graph_ranking_mode": "none",
        },
    })
    manager.dataset_config = DatasetConfig(dataset_name="medmemorybench")
    manager._api_config = SimpleNamespace(openai_api_key=None, openai_base_url=None)
    manager._batch_api = False

    agent = manager._create_agent_instance(
        "methods.amem_test_agent", "AMemTestAgent", "amem_test"
    )

    assert isinstance(agent, AMemTestAgent)
    assert captured["amem_original_evolution"] is False
    assert captured["amem_typed_relations"] is False
    assert captured["amem_typed_expansion_count"] == 7
    assert captured["amem_hybrid_retrieval"] is True
    assert captured["amem_graph_ranking_mode"] == "none"
    assert captured["amem_chain_selection"] is False
    assert captured["amem_chain_candidate_count"] == 50
    assert captured["amem_chain_evidence_count"] == 30
    assert captured["amem_chain_max_hops"] == 2
    assert captured["amem_chain_max_groups"] == 3
    assert captured["amem_chain_relevance_weight"] == 1.0
    assert captured["amem_chain_coverage_weight"] == 1.0
    assert captured["amem_chain_connectivity_weight"] == 0.35
    assert captured["amem_chain_path_weight"] == 0.75
    assert captured["amem_chain_temporal_weight"] == 0.5
    assert captured["amem_chain_redundancy_weight"] == 0.25
    assert captured["amem_temporal_state"] is False
    assert captured["amem_temporal_expansion_count"] == 5
    assert captured["amem_provenance"] is False
    assert captured["amem_provenance_max_evidence"] == 10
    assert captured["amem_provenance_inject_raw_text"] is False
    assert captured["retrieve_num"] == 10
    assert captured["amem_query_keywords"] is True
    assert captured["amem_expand_links"] is True

def test_invalid_graph_ranking_configuration_is_rejected():
    with pytest.raises(ValueError, match="amem_graph_ranking_mode must be one of"):
        AMemTestAgent.__init__(
            object.__new__(AMemTestAgent),
            model="mock",
            amem_graph_ranking_mode="invalid",
        )

    with pytest.raises(
        ValueError,
        match="requires amem_typed_relations=true and amem_typed_retrieval=true",
    ):
        AMemTestAgent.__init__(
            object.__new__(AMemTestAgent),
            model="mock",
            amem_typed_relations=True,
            amem_typed_retrieval=False,
            amem_graph_ranking_mode="typed_ppr",
        )

    with pytest.raises(ValueError, match="requires amem_typed_relations=true"):
        AMemTestAgent.__init__(
            object.__new__(AMemTestAgent),
            model="mock",
            amem_typed_relations=False,
            amem_typed_retrieval=True,
            amem_graph_ranking_mode="typed_ppr",
        )


def test_amem_fix_and_amem_test_initialize_with_mocked_dependencies(monkeypatch):
    import methods.amem_agent as amem_agent_module

    class _Client:
        pass

    class _System:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(amem_agent_module, "create_llm_client", lambda **kwargs: _Client())
    monkeypatch.setattr(AMemAgent, "_load_amem_system_class", lambda self: _System)
    baseline = AMemFixAgent(model="mock", api_key="test-key")
    baseline_system = baseline._get_memory_system(0)

    monkeypatch.setattr(AMemTestAgent, "_load_amem_system_class", lambda self: _System)
    typed = AMemTestAgent(model="mock", api_key="test-key", amem_typed_relations=True)
    typed_only = AMemTestAgent(
        model="mock",
        api_key="test-key",
        amem_original_evolution=False,
        amem_typed_relations=True,
    )
    control = AMemTestAgent(model="mock", api_key="test-key", amem_typed_relations=False)
    all_features = AMemTestAgent(
        model="mock",
        api_key="test-key",
        amem_typed_relations=True,
        amem_temporal_state=True,
        amem_provenance=True,
    )
    typed_system = typed._get_memory_system(1)
    typed_only_system = typed_only._get_memory_system(2)
    control_system = control._get_memory_system(3)
    all_features_system = all_features._get_memory_system(4)

    assert baseline_system.kwargs.get("typed_relations_enabled") is None
    assert typed_system.kwargs["original_evolution_enabled"] is True
    assert typed_system.kwargs["typed_relations_enabled"] is True
    assert typed_only_system.kwargs["original_evolution_enabled"] is False
    assert typed_only_system.kwargs["typed_relations_enabled"] is True
    assert control_system.kwargs["typed_relations_enabled"] is False
    assert all_features_system.kwargs["temporal_state_enabled"] is True
    assert all_features_system.kwargs["provenance_enabled"] is True


def test_gemini_answer_model_does_not_override_amem_build_model(monkeypatch):
    import methods.amem_agent as amem_agent_module

    class _System:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(
        amem_agent_module,
        "create_llm_client",
        lambda **kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(AMemTestAgent, "_load_amem_system_class", lambda self: _System)
    agent = AMemTestAgent(
        model="gemini-3-flash-preview",
        provider="gemini",
        api_key="test-key",
        amem_backend="gemini",
        amem_model="gemini-2.5-flash",
    )

    system = agent._get_memory_system(0)

    assert system.kwargs["llm_backend"] == "gemini"
    assert system.kwargs["llm_model"] == "gemini-2.5-flash"


@pytest.mark.parametrize(
    "typed, temporal, provenance",
    [
        (True, False, False),
        (True, True, False),
        (True, False, True),
        (True, True, True),
        (False, False, True),
    ],
)
def test_amem_test_feature_combinations_reach_memory_layer(
    monkeypatch,
    typed,
    temporal,
    provenance,
):
    import methods.amem_agent as amem_agent_module

    class _System:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(
        amem_agent_module,
        "create_llm_client",
        lambda **kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(AMemTestAgent, "_load_amem_system_class", lambda self: _System)
    agent = AMemTestAgent(
        model="mock",
        api_key="test-key",
        amem_original_evolution=False,
        amem_typed_relations=typed,
        amem_temporal_state=temporal,
        amem_provenance=provenance,
    )

    system = agent._get_memory_system(0)

    assert system.kwargs["typed_relations_enabled"] is typed
    assert system.kwargs["temporal_state_enabled"] is temporal
    assert system.kwargs["provenance_enabled"] is provenance


def test_amem_test_preserves_vertex_batch_query_and_usage_flow():
    manager = object.__new__(AgentManager)
    manager.method_config = MethodConfig.from_dict({
        "method_name": "amem_test",
        "method_type": "agentic_memory",
        "model": {"provider": "gemini", "name": "gemini-2.5-flash"},
    })
    manager._agent = SimpleNamespace(
        prepare_batch_query=lambda *args, **kwargs: {},
        finalize_batch_query=lambda prepared, content: SimpleNamespace(
            output=content, extra={}
        ),
        record_batch_query_usage=AMemTestAgent.record_batch_query_usage,
    )

    assert manager.supports_batch_queries() is True
    response = manager.finalize_batch_query({}, "answer", input_tokens=12, output_tokens=3)
    assert response.extra["tokens_used"] == {"input": 12, "output": 3}


@pytest.mark.parametrize("config_name, enabled", [
    ("amem_test_gemini", True),
    ("amem_test_off_gemini", False),
])
def test_amem_test_configs_load(config_name, enabled):
    from src.config import ConfigLoader

    config = ConfigLoader().load_method_config(config_name)
    assert config.method_name == "amem_test"
    assert config.agent_params["amem_typed_relations"] is enabled
    assert config.model.temperature == 0.0
    assert config.model.max_completion_tokens == 2000
    assert config.agent_params["retrieve_num"] == 10
    assert config.agent_params["amem_embedding_model"] == "all-MiniLM-L6-v2"
    assert config.agent_params["amem_query_keywords"] is True
    assert config.agent_params["amem_expand_links"] is True
    assert config.agent_params["amem_original_evolution"] is True
    if config_name == "amem_test_gemini":
        assert config.retrieval_config["amem_hybrid_retrieval"] is False
        assert config.retrieval_config["amem_graph_ranking_mode"] == "fixed_bfs"
        assert config.retrieval_config["amem_hybrid_graph_weight"] == 1.0
    assert isinstance(config.retrieval_config["amem_chain_selection"], bool)
    assert config.retrieval_config["amem_chain_candidate_count"] == 50
    assert config.retrieval_config["amem_chain_evidence_count"] == 30

def test_all_amem_test_configs_declare_complete_retrieval_controls():
    config_root = Path("configs/method_config")
    paths = sorted(config_root.glob("amem_test*.yaml"))
    paths.extend(sorted((config_root / "persona_1").glob("amem_test*.yaml")))
    expected_keys = {
        "retrieve_num",
        "amem_max_context_tokens",
        "amem_query_keywords",
        "amem_expand_links",
        "amem_typed_retrieval",
        "amem_typed_expansion_count",
        "amem_expand_related",
        "amem_relation_min_confidence",
        "amem_temporal_expansion_count",
        "amem_temporal_retrieval",
        "amem_provenance_max_evidence",
        "amem_provenance_retrieval",
        "amem_provenance_inject_raw_text",
        "amem_temporal_ordering",
        "amem_hybrid_retrieval",
        "amem_hybrid_candidate_count",
        "amem_hybrid_rrf_k",
        "amem_hybrid_dense_weight",
        "amem_hybrid_bm25_weight",
        "amem_hybrid_entity_weight",
        "amem_hybrid_timestamp_weight",
        "amem_hybrid_state_weight",
        "amem_hybrid_graph_weight",
        "amem_graph_ranking_mode",
        "amem_graph_alpha",
        "amem_graph_iterations",
        "amem_graph_tolerance",
        "amem_graph_supersede_weight",
        "amem_graph_conflict_weight",
        "amem_graph_refine_weight",
        "amem_graph_support_weight",
        "amem_graph_related_weight",
        "amem_chain_selection",
        "amem_chain_candidate_count",
        "amem_chain_evidence_count",
        "amem_chain_max_hops",
        "amem_chain_max_groups",
        "amem_chain_relevance_weight",
        "amem_chain_coverage_weight",
        "amem_chain_connectivity_weight",
        "amem_chain_path_weight",
        "amem_chain_temporal_weight",
        "amem_chain_redundancy_weight",
    }

    assert len(paths) == 16
    for path in paths:
        retrieval_config = yaml.safe_load(
            path.read_text(encoding="utf-8")
        )["retrieval_config"]
        assert set(retrieval_config) == expected_keys, path
        assert isinstance(retrieval_config["amem_hybrid_retrieval"], bool), path
        assert retrieval_config["amem_graph_ranking_mode"] in {
            "none", "fixed_bfs", "untyped_ppr", "typed_ppr"
        }, path
        assert isinstance(retrieval_config.get("amem_chain_selection"), bool), path


def test_all_amem_test_configs_declare_explicit_original_evolution_mode():
    config_root = Path("configs/method_config")
    paths = sorted(config_root.glob("amem_test*.yaml"))
    paths.extend(sorted((config_root / "persona_1").glob("amem_test*.yaml")))

    assert paths
    for path in paths:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        build_config = config["build_config"]
        assert "amem_original_evolution" in build_config, path
        assert isinstance(build_config["amem_original_evolution"], bool), path


def test_persona_amem_test2_gemini_disables_original_evolution():
    path = Path("configs/method_config/persona_1/amem_test2_gemini.yaml")
    config = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert config["build_config"]["amem_original_evolution"] is False


@pytest.mark.parametrize(
    "config_name, temporal, provenance",
    [
        ("amem_test2_gemini", False, False),
        ("amem_test2_temporal_gemini", True, False),
        ("amem_test2_provenance_gemini", False, True),
        ("amem_test2_temporal_provenance_gemini", True, True),
    ],
)
def test_amem_test_ablation_configs_load(config_name, temporal, provenance):
    from src.config import ConfigLoader

    config = ConfigLoader().load_method_config(config_name)
    assert config.method_name == "amem_test"
    assert config.agent_params["amem_typed_relations"] is True
    assert config.agent_params["amem_original_evolution"] is False
    assert config.agent_params["amem_expand_links"] is False
    assert config.agent_params["amem_temporal_state"] is temporal
    assert config.agent_params["amem_provenance"] is provenance
    assert config.agent_params["amem_provenance_inject_raw_text"] is provenance
    assert config.retrieval_config["amem_typed_retrieval"] is True
    assert config.retrieval_config["amem_temporal_retrieval"] is temporal
    assert config.retrieval_config["amem_provenance_retrieval"] is provenance
