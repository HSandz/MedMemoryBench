from types import SimpleNamespace

from methods.event_state.compiler import StateCompiler
from methods.event_state.embeddings import cosine
from methods.event_state.schemas import Claim, EvidenceRef
from methods.event_state.store import EventStateStore


class FakeEmbedder:
    def embed_documents(self, texts):
        return [[1.0, 0.0] for _ in texts]


def claim(store, episode, value, subject="Alice", predicate="dose"):
    return Claim(store.stable_id("C", [episode, value, subject]), subject, subject.casefold(), predicate, value, evidence=[EvidenceRef(episode, episode, [1])])


def test_stable_ids_and_snapshot_round_trip_preserve_vectors():
    store = EventStateStore("patient")
    assert store.stable_id("E", [1, "x"]) == store.stable_id("E", [1, "x"])
    c = claim(store, "s1", "500 mg")
    store.add_claim(c, [1.0, 0.0])
    restored = EventStateStore.from_export(store.export())
    assert restored.context_id == "patient"
    assert restored.claims[c.claim_id].value == "500 mg"
    assert restored.claim_embeddings[c.claim_id] == [1.0, 0.0]


def test_same_session_duplicate_and_later_corroboration_share_one_claim():
    store = EventStateStore("p")
    compiler = StateCompiler(store, FakeEmbedder(), SimpleNamespace(chat=lambda messages: SimpleNamespace(content='{}')))
    first = claim(store, "s1", "500 mg")
    assert compiler.apply(first, "s1", [1.0, 0.0]) == "NEW"
    duplicate = claim(store, "s1", "500 mg", subject="Alice")
    assert compiler.apply(duplicate, "s1", [1.0, 0.0]) == "DUPLICATE"
    later = claim(store, "s2", "500 mg")
    assert compiler.apply(later, "s2", [1.0, 0.0]) == "CORROBORATE"
    assert len(store.claims) == 1
    assert len(store.claims[first.claim_id].evidence) == 2


def test_classifier_operations_preserve_history_and_conflicts():
    decisions = iter([
        '{"matched_claim_id":"OLD","operation":"SUPERSEDE","confidence":0.9,"rationale":"explicit change"}',
        '{"matched_claim_id":"OLD2","operation":"CONFLICT","confidence":0.9,"rationale":"incompatible"}',
    ])
    llm = SimpleNamespace(chat=lambda messages: SimpleNamespace(content=next(decisions)))
    store = EventStateStore("p")
    compiler = StateCompiler(store, FakeEmbedder(), llm, min_similarity=0.1)
    old = Claim("OLD", "Alice", "alice", "dose", "500 mg", evidence=[EvidenceRef("s1", "s1", [1])])
    store.add_claim(old, [1.0, 0.0])
    new = Claim("NEW", "Alice", "alice", "dose", "850 mg", evidence=[EvidenceRef("s2", "s2", [2])])
    assert compiler.apply(new, "s2", [1.0, 0.0]) == "SUPERSEDE"
    assert old.status == "superseded"
    assert any(edge["relation_type"] == "SUPERSEDES" for edge in store.edges)
    old2 = Claim("OLD2", "Alice", "alice", "dose", "100 mg", evidence=[EvidenceRef("s3", "s3", [3])])
    store.add_claim(old2, [1.0, 0.0])
    conflict = Claim("NEW2", "Alice", "alice", "dose", "2000 mg", evidence=[EvidenceRef("s4", "s4", [4])])
    assert compiler.apply(conflict, "s4", [1.0, 0.0]) == "CONFLICT"
    assert old2.status == "contested"


def test_subject_scope_and_non_observation_claims_are_not_merged():
    store = EventStateStore("p")
    compiler = StateCompiler(store, FakeEmbedder(), SimpleNamespace(chat=lambda messages: SimpleNamespace(content='{}')))
    user = Claim("U", "Alice", "alice", "hypertension", "false", evidence=[EvidenceRef("s1", "s1", [1])])
    father = Claim("F", "Father", "father", "hypertension", "true", evidence=[EvidenceRef("s2", "s2", [2])])
    plan = Claim("P", "Alice", "alice", "dose", "850 mg", modality="planned", evidence=[EvidenceRef("s3", "s3", [3])])
    assert compiler.apply(user, "s1", [1.0, 0.0]) == "NEW"
    assert compiler.apply(father, "s2", [1.0, 0.0]) == "NEW"
    assert compiler.apply(plan, "s3", [1.0, 0.0]) == "EPISODIC"
    assert store.claims["U"].status == "active"
    assert store.claims["F"].status == "active"
    assert store.claims["P"].status == "standalone"
