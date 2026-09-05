"""Regression coverage for immutable Event-State turn retrieval."""

from datetime import date
from types import SimpleNamespace

from benchmarks.locomo.dataset import LoCoMoQuery
from benchmarks.locomo.evaluator import LoCoMoEvaluator
from methods.event_state.schemas import (
    EPISODE_RETRIEVAL_MAX_CHARS,
    Episode,
    TurnEvidence,
    select_episode_retrieval_turns,
)
from methods.event_state.temporal import parse_temporal_query
from methods.event_state_agent import EventStateAgent
from src.config import ConfigLoader


class KeywordEmbedder:
    def embed_documents(self, texts):
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text):
        return [1.0, 0.0] if "needle" in text.casefold() else [0.0, 1.0]


class EmptyExtractionLLM:
    def chat(self, messages, **kwargs):
        if "Extract conversational memory" in messages[0]["content"]:
            return SimpleNamespace(content='{"episode_summary":"summary", "claims":[]}')
        return SimpleNamespace(content="answer")


class ClaimedNeedleLLM:
    def chat(self, messages, **kwargs):
        if "Extract conversational memory" in messages[0]["content"]:
            return SimpleNamespace(content=(
                '{"episode_summary":"summary", "claims":[{"subject":"A",'
                '"predicate":"detail","value":"needle","source_turn_ids":["t"]}]}'
            ))
        return SimpleNamespace(content='{"operation":"NEW","confidence":1}')


def test_episode_retrieval_sampling_spans_source_order_and_keeps_captions():
    turns = [
        TurnEvidence(
            turn_id=f"t{index}", speaker="A", role="user",
            text=f"turn-{index}", image_caption="final-caption" if index == 19 else None,
        )
        for index in range(20)
    ]
    sampled = select_episode_retrieval_turns(turns)
    assert [turn.turn_id for turn in select_episode_retrieval_turns(turns[:8])] == [f"t{index}" for index in range(8)]
    assert sampled[0].turn_id == "t0"
    assert sampled[-1].turn_id == "t19"
    assert any(turn.turn_id in {"t9", "t10"} for turn in sampled)
    assert [turn.turn_id for turn in sampled] == sorted(
        (turn.turn_id for turn in sampled), key=lambda value: int(value[1:])
    )
    assert [turn.turn_id for turn in sampled] == [turn.turn_id for turn in select_episode_retrieval_turns(turns)]

    episode = Episode("E", "ctx", "s", 0, None, None, [], None, "", "summary", turns)
    assert "final-caption" in episode.retrieval_text()
    oversized = Episode("E2", "ctx", "s2", 1, None, None, [], None, "", "summary", [
        TurnEvidence(str(index), "A", "user", "x" * 1000) for index in range(20)
    ])
    assert len(oversized.retrieval_text()) <= EPISODE_RETRIEVAL_MAX_CHARS


def test_temporal_parser_accepts_comma_day_month_dates_conservatively():
    assert parse_temporal_query("What happened on 1 February, 2023?").target_date == date(2023, 2, 1)
    assert parse_temporal_query("What happened on 1 February 2023?").target_date == date(2023, 2, 1)
    assert parse_temporal_query("What happened on 13 Oct, 2023?").target_date == date(2023, 10, 13)
    assert parse_temporal_query("What happened on October 13, 2023?").target_date == date(2023, 10, 13)
    assert parse_temporal_query("What happened in February 2023?") is None
    assert parse_temporal_query("What happened last February?") is None


def test_global_turn_index_recovers_late_unclaimed_turn_and_snapshot_restore():
    agent = EventStateAgent(
        llm_client=EmptyExtractionLLM(), memory_llm_client=EmptyExtractionLLM(),
        embedding_client=KeywordEmbedder(), retrieve_claims=False,
        retrieve_episodes=False, retrieve_turns=True, candidate_count=4,
        evidence_count=1, max_episode_source_excerpts_total=2,
    )
    agent.set_context_id("ctx")
    agent.memorize("", memory_items=[
        {
            "source_session_id": "session-1", "source_turn_id": str(index),
            "speaker": "A", "content": "needle detail" if index == 19 else "ordinary discussion",
        }
        for index in range(20)
    ])
    prepared = agent.prepare_batch_query("Where was the needle detail?", raw_question="needle detail")
    selected = prepared["retrieved_memories"]
    assert selected[0]["type"] == "immutable_turn"
    assert selected[0]["source_turn_id"] == "19"
    assert selected[0]["source_session_id"] == "session-1"
    assert not agent._store().claims
    assert selected[0]["included_in_context"] is True
    assert prepared["extra"]["configured_global_raw_excerpt_budget"] == 2
    assert prepared["extra"]["selected_raw_turn_count"] == 1

    legacy_state = agent.export_memory_state()
    legacy_state["schema_version"] = 4
    legacy_state.pop("turn_embeddings")
    legacy_state.pop("turn_metadata")
    restored = EventStateAgent(
        llm_client=EmptyExtractionLLM(), memory_llm_client=EmptyExtractionLLM(),
        embedding_client=KeywordEmbedder(), retrieve_claims=False,
        retrieve_episodes=False, retrieve_turns=True, candidate_count=4,
        evidence_count=1, max_episode_source_excerpts_total=8,
    )
    restored.import_memory_state(legacy_state, context_id="ctx")
    restored_prepared = restored.prepare_batch_query("needle detail", raw_question="needle detail")
    assert restored_prepared["retrieved_memories"][0]["source_turn_id"] == "19"
    assert restored_prepared["extra"]["configured_global_raw_excerpt_budget"] == 8


def test_configured_raw_excerpt_budget_reaches_effective_agent():
    config = ConfigLoader().load_method_config("event_state_gemini")
    agent = EventStateAgent(
        llm_client=EmptyExtractionLLM(), memory_llm_client=EmptyExtractionLLM(),
        embedding_client=KeywordEmbedder(), **config.agent_params,
    )
    assert config.retrieval_config["max_episode_source_excerpts_total"] == 2
    assert agent.max_episode_source_excerpts_total == 2
    assert agent._retrieval_config["max_episode_source_excerpts_total"] == 2


def test_claim_provenance_and_direct_turn_retrieval_render_one_copy():
    agent = EventStateAgent(
        llm_client=ClaimedNeedleLLM(), memory_llm_client=ClaimedNeedleLLM(),
        embedding_client=KeywordEmbedder(), candidate_count=4, evidence_count=2,
        retrieve_turns=True,
    )
    agent.memorize("", memory_items=[{
        "source_session_id": "session-1", "source_turn_id": "t",
        "speaker": "A", "content": "needle detail",
    }])
    prepared = agent.prepare_batch_query("needle detail", raw_question="needle detail")
    records = prepared["retrieved_memories"]
    assert any(record["type"] == "state_claim" for record in records)
    assert not any(record["type"] == "immutable_turn" for record in records)
    claim = next(record for record in records if record["type"] == "state_claim")
    assert claim["included_provenance_evidence"][0]["evidence"]["source_turn_ids"] == ["t"]


def test_locomo_stage_telemetry_remains_evaluator_only():
    evaluator = LoCoMoEvaluator.__new__(LoCoMoEvaluator)
    query = LoCoMoQuery(
        query_id="q", question="", query_type="single_hop", expected_answers=[""],
        evidence=["D2:4"],
    )
    quality = evaluator._locomo_retrieval_quality(
        query,
        [
            {
                "type": "episode", "source_session_id": 2,
                "episode_archive_turn_ids": ["4"], "included_in_context": False,
                "episode_evidence_turn_ids": [],
            },
            {
                "type": "immutable_turn", "source_session_id": 2,
                "episode_evidence_turn_ids": ["4"], "included_in_context": True,
            },
        ],
        {
            "retrieval_stage_candidates": {
                "pre_candidate_truncation_fused": [{"source_session_ids": ["1"]}],
                "post_candidate_count": [{"source_session_ids": ["2"]}],
                "final_memory_object_selection": [{"source_session_ids": ["2"]}],
            }
        },
    )
    assert quality["pre_candidate_truncation_fused_session"]["hit"] is False
    assert quality["post_candidate_count_session"]["hit"] is True
    assert quality["selected_memory_session"]["hit"] is True
    assert quality["selected_episode_archive_exact_turn"]["hit"] is True
    assert quality["answer_visible_exact_turn"]["hit"] is True
