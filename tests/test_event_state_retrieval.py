from types import SimpleNamespace

from methods.event_state.retrieval import EventStateRetriever
from methods.event_state.schemas import Claim, Episode, EvidenceRef
from methods.event_state.store import EventStateStore
from methods.event_state.temporal import parse_temporal_query


class Embedder:
    def embed_query(self, text):
        return [1.0, 0.0]


def _episode(identifier, recorded_at, vector):
    return Episode(identifier, "ctx", identifier, 0, None, recorded_at, ["User"], "primary_user", "", identifier, []), vector


def test_temporal_parser_is_conservative_and_supports_bounded_iso_forms():
    assert parse_temporal_query("What did we discuss in the record dated 2025-03-15?").kind == "exact_record_time"
    assert parse_temporal_query("Where was I living as of 2025/03/15?").kind == "as_of"
    interval = parse_temporal_query("What happened between 2025-01-05 and 2025-01-15?")
    assert interval.start_date.isoformat() == "2025-01-05"
    assert parse_temporal_query("What happened around the 5th?") is None


def test_temporal_episode_channel_adds_exact_record_date_candidate():
    store = EventStateStore("ctx")
    for episode, vector in (_episode("A", "2025-01-01", [0.0, 1.0]), _episode("B", "2025-03-15", [0.0, 1.0]), _episode("C", "2025-06-01", [1.0, 0.0])):
        store.add_episode(episode, vector)
    retriever = EventStateRetriever(store, Embedder(), retrieve_claims=False, episode_top_k=1, candidate_count=1, evidence_count=1)
    selected, extra = retriever.retrieve("What did we discuss in the record dated 2025-03-15?")
    assert extra["temporal_constraint_detected"] is True
    assert extra["temporal_episode_candidate_count"] == 1
    assert selected[0]["id"] == "B"
    assert selected[0]["temporal_match_type"] == "exact_record_time"


def test_as_of_retrieval_exposes_historical_state_but_hides_current_version():
    store = EventStateStore("ctx")
    store.add_episode(Episode("E1", "ctx", "s1", 0, None, "2025-01-01", ["User"], "primary_user", "", "", []), [1.0, 0.0])
    store.add_episode(Episode("E2", "ctx", "s2", 1, None, "2025-03-15", ["User"], "primary_user", "", "", []), [1.0, 0.0])
    old = Claim("A", "User", "primary_user", "lives_in", "Boston", state_slot="residence_location", recorded_at="2025-01-01", valid_from="2025-01-01", valid_to="2025-03-01", status="superseded", evidence=[EvidenceRef("E1", "s1", [])])
    current = Claim("B", "User", "primary_user", "lives_in", "Tokyo", state_slot="residence_location", recorded_at="2025-03-15", valid_from="2025-03-01", status="active", evidence=[EvidenceRef("E2", "s2", [])])
    store.add_claim(old, [1.0, 0.0])
    store.add_claim(current, [1.0, 0.0])
    retriever = EventStateRetriever(store, Embedder(), retrieve_episodes=False, claim_top_k=10, candidate_count=10, evidence_count=2)
    selected, extra = retriever.retrieve("Where was the user living as of 2025-02-15?")
    assert [item["id"] for item in selected] == ["A"]
    assert extra["temporal_historical_state_candidate_count"] == 1
    assert extra["temporal_future_state_filtered_count"] == 1
    assert old.status == "superseded" and current.status == "active"


def test_non_temporal_retrieval_does_not_activate_temporal_channel():
    store = EventStateStore("ctx")
    for episode, vector in (_episode("A", "2025-01-01", [1.0, 0.0]), _episode("B", "2025-03-15", [0.0, 1.0])):
        store.add_episode(episode, vector)
    config = dict(retrieve_claims=False, episode_top_k=2, candidate_count=2, evidence_count=2)
    selected_a, extra_a = EventStateRetriever(store, Embedder(), **config, temporal_retrieval_enabled=False).retrieve("Where does the user live?")
    selected_b, extra_b = EventStateRetriever(store, Embedder(), **config, temporal_retrieval_enabled=True).retrieve("Where does the user live?")
    assert [item["id"] for item in selected_a] == [item["id"] for item in selected_b]
    assert [(item["fusion_score"], item["final_score"]) for item in selected_a] == [(item["fusion_score"], item["final_score"]) for item in selected_b]
    assert extra_b["temporal_constraint_detected"] is False


def test_planner_merge_recomputes_final_score_and_clears_stale_selection_metadata():
    store = EventStateStore("ctx")
    retriever = EventStateRetriever(
        store,
        Embedder(),
        planner_merge_mode="coverage_interleave",
        candidate_count=3,
        evidence_count=1,
        selector_mode="topk",
    )
    merged = retriever.merge_rank_channels([
        [{"id": "A", "type": "episode", "final_score": 0.99, "selection_score": 0.99, "selected_rank": 7}],
        [{"id": "C", "type": "episode", "final_score": 0.01, "selection_score": -3.0, "selected_rank": 8}],
        [{"id": "B", "type": "episode", "final_score": 0.5}],
    ])
    by_id = {item["id"]: item for item in merged}
    assert by_id["A"]["final_score"] == 1.0
    assert by_id["C"]["final_score"] == 1.0
    assert "selection_score" not in by_id["A"] and "selected_rank" not in by_id["A"]
    selected, _ = retriever.select_candidates(merged)
    assert selected[0]["id"] == "C"


def test_ppr_is_bounded_and_conserves_personalized_mass():
    store = EventStateStore("ctx")
    store.episodes = {
        "E1": Episode("E1", "ctx", 1, 0, None, None, ["User"], "primary_user", "", "one"),
        "E2": Episode("E2", "ctx", 2, 1, None, None, ["User"], "primary_user", "", "two"),
        "E3": Episode("E3", "ctx", 3, 2, None, None, ["User"], "primary_user", "", "three"),
    }
    store.episode_embeddings = {key: [1.0, 0.0] for key in store.episodes}
    store.add_edge("E1", "E2", "EPISODE_SUPPORTS_CLAIM")
    store.add_edge("E2", "E3", "EPISODE_SUPPORTS_CLAIM")
    retriever = EventStateRetriever(store, Embedder(), ppr_expand_hops=1, ppr_max_iterations=100, ppr_tolerance=1e-12)
    candidates = [
        {"id": "E1", "type": "episode", "score": 0.5},
        {"id": "E2", "type": "episode", "score": 0.25},
    ]
    result = retriever._ppr_impl(candidates)
    assert {item["id"] for item in result} == {"E1", "E2", "E3"}
    assert abs(sum(item["ppr_score"] for item in result) - 1.0) < 1e-9


def test_ppr_expansion_does_not_escape_hop_bound():
    store = EventStateStore("ctx")
    store.claims = {key: SimpleNamespace() for key in ("C1", "C2", "C3")}
    store.claim_embeddings = {key: [1.0, 0.0] for key in store.claims}
    store.add_edge("C1", "C2", "REFINES")
    store.add_edge("C2", "C3", "REFINES")
    retriever = EventStateRetriever(store, Embedder(), ppr_expand_hops=1)
    result = retriever._ppr_impl([{"id": "C1", "type": "state_claim", "score": 1.0}])
    assert {item["id"] for item in result} == {"C1", "C2"}


def test_ppr_follows_claim_episode_claim_and_excludes_disconnected_component():
    store = EventStateStore("ctx")
    store.claims = {key: SimpleNamespace() for key in ("C1", "C2", "C99")}
    store.episodes = {key: SimpleNamespace(source_session_id=key) for key in ("E1", "E99")}
    store.claim_embeddings = {key: [1.0, 0.0] for key in store.claims}
    store.episode_embeddings = {key: [1.0, 0.0] for key in store.episodes}
    store.add_edge("C1", "E1", "CLAIM_SUPPORTED_BY_EPISODE")
    store.add_edge("E1", "C2", "EPISODE_SUPPORTS_CLAIM")
    store.add_edge("C99", "E99", "CLAIM_SUPPORTED_BY_EPISODE")
    retriever = EventStateRetriever(store, Embedder(), ppr_expand_hops=2)
    result = retriever._ppr_impl([{"id": "C1", "type": "state_claim", "score": 1.0}])
    assert {item["id"] for item in result} == {"C1", "E1", "C2"}


def test_retrieval_hides_superseded_state_versions_but_keeps_history_and_reports_statuses():
    store = EventStateStore("ctx")
    store.claims = {
        "OLD": Claim("OLD", "Alice", "alice", "lives_in", "Boston", persistence="state", status="superseded", evidence=[EvidenceRef("E1", "s1", ["1"])]),
        "CURRENT": Claim("CURRENT", "Alice", "alice", "lives_in", "Tokyo", persistence="state", status="active", evidence=[EvidenceRef("E2", "s2", ["1"])]),
        "HISTORY": Claim("HISTORY", "Alice", "alice", "worked_at", "Acme", persistence="history", status="standalone", evidence=[EvidenceRef("E3", "s3", ["1"])]),
    }
    store.claim_embeddings = {key: [1.0, 0.0] for key in store.claims}
    store.add_edge("CURRENT", "OLD", "SUPERSEDES")
    retriever = EventStateRetriever(store, Embedder(), claim_top_k=10, episode_top_k=0, candidate_count=10, evidence_count=10, retrieve_episodes=False)
    selected, extra = retriever.retrieve("where does Alice live")
    ids = {item["id"] for item in selected}
    assert "CURRENT" in ids and "HISTORY" in ids and "OLD" not in ids
    assert extra["hidden_prior_state_candidate_count"] == 1
    assert extra["claim_candidate_status_counts"]["active"] == 1
