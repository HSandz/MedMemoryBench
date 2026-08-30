from types import SimpleNamespace

from methods.event_state.retrieval import EventStateRetriever
from methods.event_state.schemas import Episode
from methods.event_state.store import EventStateStore


class Embedder:
    def embed_query(self, text):
        return [1.0, 0.0]


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
