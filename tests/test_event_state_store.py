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


def test_state_reversion_creates_new_version_instead_of_reactivating_history():
    decisions = iter([
        '{"matched_claim_id":"C500","operation":"SUPERSEDE","confidence":0.9}',
        '{"matched_claim_id":"C850","operation":"SUPERSEDE","confidence":0.9}',
    ])
    store = EventStateStore("p")
    compiler = StateCompiler(store, FakeEmbedder(), SimpleNamespace(chat=lambda messages: SimpleNamespace(content=next(decisions))), min_similarity=0.1)
    c500 = Claim("C500", "Alice", "alice", "dose", "500 mg", evidence=[EvidenceRef("s1", "s1", [1])])
    c850 = Claim("C850", "Alice", "alice", "dose", "850 mg", recorded_at="2024-03-01", evidence=[EvidenceRef("s2", "s2", [2])])
    c500_again = Claim("C500B", "Alice", "alice", "dose", "500 mg", recorded_at="2024-06-01", evidence=[EvidenceRef("s3", "s3", [3])])
    assert compiler.apply(c500, "s1", [1.0, 0.0]) == "NEW"
    assert compiler.apply(c850, "s2", [1.0, 0.0]) == "SUPERSEDE"
    result = compiler.apply(c500_again, "s3", [1.0, 0.0])
    assert result.operation == "SUPERSEDE"
    assert len(store.claims) == 3
    assert store.claims["C500"].status == "superseded"
    assert store.claims["C850"].status == "superseded"
    assert store.claims["C500B"].status == "active"


def test_supersede_only_closes_validity_when_valid_from_is_explicit():
    store = EventStateStore("p")
    decisions = iter([
        '{"matched_claim_id":"OLD","operation":"SUPERSEDE","confidence":0.9}',
        '{"matched_claim_id":"OLD2","operation":"SUPERSEDE","confidence":0.9}',
    ])
    compiler = StateCompiler(store, FakeEmbedder(), SimpleNamespace(chat=lambda messages: SimpleNamespace(content=next(decisions))), min_similarity=0.1)
    old = Claim("OLD", "Alice", "alice", "city", "Boston", evidence=[EvidenceRef("s1", "s1", [1])])
    old2 = Claim("OLD2", "Alice", "alice", "city", "Seattle", evidence=[EvidenceRef("s2", "s2", [2])])
    store.add_claim(old, [1.0, 0.0]); store.add_claim(old2, [1.0, 0.0])
    explicit = Claim("NEW1", "Alice", "alice", "city", "Toronto", recorded_at="2024-06-20", valid_from="2024-06-17", evidence=[EvidenceRef("s3", "s3", [3])])
    unknown = Claim("NEW2", "Alice", "alice", "city", "Boston", recorded_at="2024-06-20", evidence=[EvidenceRef("s4", "s4", [4])])
    compiler.apply(explicit, "s3", [1.0, 0.0]); compiler.apply(unknown, "s4", [1.0, 0.0])
    assert old.valid_to == "2024-06-17"
    assert old2.valid_to is None


def test_llm_classified_corroboration_and_duplicate_keep_graph_edges():
    store = EventStateStore("p")
    decisions = iter([
        '{"matched_claim_id":"OLD","operation":"CORROBORATE","confidence":0.9}',
        '{"matched_claim_id":"OLD","operation":"DUPLICATE","confidence":0.9}',
    ])
    compiler = StateCompiler(store, FakeEmbedder(), SimpleNamespace(chat=lambda messages: SimpleNamespace(content=next(decisions))), min_similarity=0.1)
    old = Claim("OLD", "Alice", "alice", "dose", "500 mg", evidence=[EvidenceRef("s1", "s1", [1])])
    store.add_claim(old, [1.0, 0.0])
    for cid, session, value in (("C2", "s2", "500 milligrams"), ("C3", "s3", "500mg")):
        result = compiler.apply(Claim(cid, "Alice", "alice", "dose", value, evidence=[EvidenceRef(session, session, [2])]), session, [1.0, 0.0])
        assert result.operation in {"CORROBORATE", "DUPLICATE"}
    assert len(store.claims["OLD"].evidence) == 3
    for session in ("s1", "s2", "s3"):
        assert {edge["relation_type"] for edge in store.edges if edge["source_id"] == "OLD" and edge["target_id"] == session} == {"CLAIM_SUPPORTED_BY_EPISODE"}
        assert {edge["relation_type"] for edge in store.edges if edge["source_id"] == session and edge["target_id"] == "OLD"} == {"EPISODE_SUPPORTS_CLAIM"}


def test_low_confidence_classifier_falls_back_to_new_with_reason():
    store = EventStateStore("p")
    old = Claim("OLD", "Alice", "alice", "dose", "500 mg", evidence=[EvidenceRef("s1", "s1", [1])])
    store.add_claim(old, [1.0, 0.0])
    llm = SimpleNamespace(chat=lambda messages: SimpleNamespace(content='{"matched_claim_id":"OLD","operation":"SUPERSEDE","confidence":0.3}'))
    compiler = StateCompiler(store, FakeEmbedder(), llm, min_similarity=0.1, min_confidence=0.55)
    result = compiler.apply(Claim("NEW", "Alice", "alice", "dose", "850 mg", evidence=[EvidenceRef("s2", "s2", [2])]), "s2", [1.0, 0.0])
    assert result.operation == "NEW"
    assert result.fallback_reason == "below_confidence_threshold"
    assert store.claims["OLD"].status == "active"


def test_classifier_cannot_reactivate_historical_claim_via_duplicate():
    store = EventStateStore("p")
    old = Claim("OLD", "Alice", "alice", "dose", "500 mg", status="superseded", evidence=[EvidenceRef("s1", "s1", [1])])
    store.add_claim(old, [1.0, 0.0])
    llm = SimpleNamespace(chat=lambda messages: SimpleNamespace(content='{"matched_claim_id":"OLD","operation":"CORROBORATE","confidence":0.9}'))
    compiler = StateCompiler(store, FakeEmbedder(), llm, min_similarity=0.1)
    result = compiler.apply(Claim("NEW", "Alice", "alice", "dose", "500 mg", evidence=[EvidenceRef("s2", "s2", [2])]), "s2", [1.0, 0.0])
    assert result.operation == "NEW"
    assert len(store.claims["OLD"].evidence) == 1


def test_store_invariant_checker_reports_missing_nodes_and_embeddings():
    store = EventStateStore("p")
    store.claims["C"] = Claim("C", "Alice", "alice", "dose", "500 mg", evidence=[EvidenceRef("MISSING", "s", [1])])
    store.add_edge("C", "MISSING", "CLAIM_SUPPORTED_BY_EPISODE")
    errors = store.validate_state_invariants()
    assert any("missing node" in error for error in errors)
    assert any("missing claim embedding" in error for error in errors)
    assert any("missing evidence episode" in error for error in errors)
