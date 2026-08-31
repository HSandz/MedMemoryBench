from types import SimpleNamespace

import pytest

from methods.event_state.compiler import StateCompiler
from methods.event_state.embeddings import cosine
from methods.event_state.schemas import Claim, Episode, EvidenceRef, TurnEvidence
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
    c.state_slot = "medication_dose"
    store.add_claim(c, [1.0, 0.0], [0.0, 1.0])
    restored = EventStateStore.from_export(store.export())
    assert restored.context_id == "patient"
    assert restored.claims[c.claim_id].value == "500 mg"
    assert restored.claim_embeddings[c.claim_id] == [1.0, 0.0]
    assert restored.claims[c.claim_id].state_slot == "medication_dose"
    assert restored.claim_slot_embeddings[c.claim_id] == [0.0, 1.0]


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


def test_same_session_corroboration_is_downgraded_to_duplicate():
    store = EventStateStore("p")
    store.add_episode(Episode("s1", "p", "s1", 0, None, None, ["Alice"], "primary_user", "", "", [TurnEvidence("1", "Alice", "user", "I work remotely.") , TurnEvidence("2", "Alice", "user", "My job is fully remote.")]), [1.0, 0.0])
    old = Claim("OLD", "Alice", "alice", "work_style", "remote", evidence=[EvidenceRef("s1", "s1", ["1"])])
    store.add_claim(old, [1.0, 0.0])
    llm = SimpleNamespace(chat=lambda messages: SimpleNamespace(content='{"matched_claim_id":"OLD","operation":"CORROBORATE","same_state_dimension":true,"same_episode_relation":"restatement","confidence":1}'))
    result = StateCompiler(store, FakeEmbedder(), llm, min_similarity=0.1).apply(Claim("NEW", "Alice", "alice", "work_style", "fully remote", evidence=[EvidenceRef("s1", "s1", ["2"])]), "s1", [1.0, 0.0])
    assert result.operation == "DUPLICATE"
    assert old.status == "active" and len(old.evidence) == 2


def test_invalid_classifier_fallback_supplies_default_same_episode_relation():
    store = EventStateStore("p")
    old = Claim("OLD", "Alice", "alice", "work_style", "office", evidence=[EvidenceRef("s1", "s1", [1])])
    store.add_claim(old, [1.0, 0.0])
    compiler = StateCompiler(store, FakeEmbedder(), SimpleNamespace(chat=lambda messages: SimpleNamespace(content="not json")), min_similarity=0.1)
    result = compiler.apply(Claim("NEW", "Alice", "alice", "work_style", "hybrid", evidence=[EvidenceRef("s2", "s2", [2])]), "s2", [1.0, 0.0])
    assert result.operation == "NEW"
    assert compiler.update_parse_failures == 1


def test_malformed_classifier_output_gets_one_bounded_structured_repair():
    responses = iter([
        "unfinished reasoning without JSON",
        '{"matched_claim_id":"OLD","operation":"SUPERSEDE","same_state_dimension":true,"confidence":0.9}',
    ])

    class RepairingLLM:
        def __init__(self):
            self.calls = []

        def chat(self, messages, **kwargs):
            self.calls.append((messages, kwargs))
            return SimpleNamespace(content=next(responses))

    store = EventStateStore("p")
    old = Claim("OLD", "Alice", "alice", "city", "Boston", state_slot="residence_location", evidence=[EvidenceRef("s1", "s1", [1])])
    store.add_claim(old, [1.0, 0.0], [1.0, 0.0])
    llm = RepairingLLM()
    compiler = StateCompiler(store, FakeEmbedder(), llm, min_similarity=0.1)
    result = compiler.apply(
        Claim("NEW", "Alice", "alice", "city", "Tokyo", state_slot="residence_location", evidence=[EvidenceRef("s2", "s2", [2])]),
        "s2",
        [1.0, 0.0],
        [1.0, 0.0],
    )

    assert result.operation == "SUPERSEDE"
    assert compiler.update_parse_failures == 1
    assert compiler.update_repair_calls == 1
    assert compiler.update_repair_successes == 1
    assert compiler.update_repair_failures == 0
    assert len(llm.calls) == 2
    assert "unfinished reasoning" in llm.calls[1][0][-1]["content"]
    assert "new_claim" not in llm.calls[1][0][-1]["content"]


def test_malformed_classifier_and_repair_fall_back_to_new_without_mutation():
    class BrokenLLM:
        def __init__(self):
            self.calls = 0

        def chat(self, messages, **kwargs):
            self.calls += 1
            return SimpleNamespace(content="not JSON")

    store = EventStateStore("p")
    old = Claim("OLD", "Alice", "alice", "city", "Boston", state_slot="residence_location", evidence=[EvidenceRef("s1", "s1", [1])])
    store.add_claim(old, [1.0, 0.0], [1.0, 0.0])
    llm = BrokenLLM()
    compiler = StateCompiler(store, FakeEmbedder(), llm, min_similarity=0.1)
    result = compiler.apply(
        Claim("NEW", "Alice", "alice", "city", "Tokyo", state_slot="residence_location", evidence=[EvidenceRef("s2", "s2", [2])]),
        "s2",
        [1.0, 0.0],
        [1.0, 0.0],
    )

    assert result.operation == "NEW"
    assert old.status == "active"
    assert compiler.update_parse_failures == 1
    assert compiler.update_repair_calls == 1
    assert compiler.update_repair_successes == 0
    assert compiler.update_repair_failures == 1
    assert len(compiler.invalid_update_output_previews) == 1
    assert len(compiler.invalid_update_output_sha256) == 1
    assert llm.calls == 2


def test_same_session_transition_requires_explicit_relation_and_later_evidence():
    store = EventStateStore("p")
    store.add_episode(Episode("s1", "p", "s1", 0, None, None, ["Alice"], "primary_user", "", "", [TurnEvidence("1", "Alice", "user", "I live in Boston."), TurnEvidence("2", "Alice", "user", "I moved to Tokyo.")]), [1.0, 0.0])
    old = Claim("OLD", "Alice", "alice", "lives_in", "Boston", evidence=[EvidenceRef("s1", "s1", ["1"])])
    store.add_claim(old, [1.0, 0.0])
    llm = SimpleNamespace(chat=lambda messages: SimpleNamespace(content='{"matched_claim_id":"OLD","operation":"SUPERSEDE","same_state_dimension":true,"same_episode_relation":"none","confidence":1}'))
    result = StateCompiler(store, FakeEmbedder(), llm, min_similarity=0.1).apply(Claim("NEW", "Alice", "alice", "lives_in", "Tokyo", evidence=[EvidenceRef("s1", "s1", ["2"])]), "s1", [1.0, 0.0])
    assert result.operation == "NEW" and old.status == "active"


def test_classifier_operations_preserve_history_and_conflicts():
    decisions = iter([
        '{"matched_claim_id":"OLD","operation":"SUPERSEDE","same_state_dimension":true,"confidence":0.9,"rationale":"explicit change"}',
        '{"matched_claim_id":"OLD2","operation":"CONFLICT","same_state_dimension":true,"confidence":0.9,"rationale":"incompatible"}',
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
        '{"matched_claim_id":"C500","operation":"SUPERSEDE","same_state_dimension":true,"confidence":0.9}',
        '{"matched_claim_id":"C850","operation":"SUPERSEDE","same_state_dimension":true,"confidence":0.9}',
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
        '{"matched_claim_id":"OLD","operation":"SUPERSEDE","same_state_dimension":true,"confidence":0.9}',
        '{"matched_claim_id":"OLD2","operation":"SUPERSEDE","same_state_dimension":true,"confidence":0.9}',
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
        '{"matched_claim_id":"OLD","operation":"CORROBORATE","same_state_dimension":true,"confidence":0.9}',
        '{"matched_claim_id":"OLD","operation":"DUPLICATE","same_state_dimension":true,"confidence":0.9}',
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


def test_historical_supersede_target_falls_back_to_new_without_mutating_history():
    store = EventStateStore("p")
    old_boston = Claim("BOSTON_OLD", "Alice", "alice", "city", "Boston", status="superseded", evidence=[EvidenceRef("s1", "s1", [1])])
    tokyo = Claim("TOKYO", "Alice", "alice", "city", "Tokyo", evidence=[EvidenceRef("s2", "s2", [2])])
    store.add_claim(old_boston, [1.0, 0.0])
    store.add_claim(tokyo, [0.0, 1.0])
    llm = SimpleNamespace(chat=lambda messages: SimpleNamespace(content='{"matched_claim_id":"BOSTON_OLD","operation":"SUPERSEDE","confidence":0.9}'))
    compiler = StateCompiler(store, FakeEmbedder(), llm, candidate_top_k=2, current_candidate_top_k=1, min_similarity=0.9)

    result = compiler.apply(Claim("BOSTON_NEW", "Alice", "alice", "city", "Boston", evidence=[EvidenceRef("s3", "s3", [3])]), "s3", [1.0, 0.0])

    assert result.operation == "NEW"
    assert result.matched_claim_id is None
    assert result.fallback_reason == "unknown_matched_claim"
    assert old_boston.status == "superseded"
    assert tokyo.status == "active"
    assert store.claims["BOSTON_NEW"].status == "active"
    assert not any(
        edge["source_id"] == "BOSTON_NEW"
        and edge["relation_type"] in {"SUPERSEDES", "REFINES", "CONFLICTS_WITH"}
        for edge in store.edges
    )
    assert store.operations[-1].matched_claim_id is None


@pytest.mark.parametrize(
    ("operation", "historical_status"),
    [("REFINE", "refined"), ("CONFLICT", "standalone")],
)
def test_historical_transition_targets_are_rejected(operation, historical_status):
    store = EventStateStore("p")
    old = Claim("OLD", "Alice", "alice", "city", "Boston", status=historical_status, evidence=[EvidenceRef("s1", "s1", [1])])
    store.add_claim(old, [1.0, 0.0])
    llm = SimpleNamespace(chat=lambda messages: SimpleNamespace(content=f'{{"matched_claim_id":"OLD","operation":"{operation}","same_state_dimension":true,"confidence":0.9}}'))
    compiler = StateCompiler(store, FakeEmbedder(), llm, min_similarity=0.1)

    result = compiler.apply(Claim("NEW", "Alice", "alice", "city", "Paris", evidence=[EvidenceRef("s2", "s2", [2])]), "s2", [1.0, 0.0])

    assert result.operation == "NEW"
    assert result.matched_claim_id is None
    assert result.fallback_reason == "no_candidate"
    assert old.status == historical_status
    assert store.claims["NEW"].status == "active"
    assert not any(
        edge["relation_type"] in {"SUPERSEDES", "REFINES", "CONFLICTS_WITH"}
        for edge in store.edges
    )


def test_store_invariant_checker_reports_missing_nodes_and_embeddings():
    store = EventStateStore("p")
    store.claims["C"] = Claim("C", "Alice", "alice", "dose", "500 mg", evidence=[EvidenceRef("MISSING", "s", [1])])
    store.add_edge("C", "MISSING", "CLAIM_SUPPORTED_BY_EPISODE")
    errors = store.validate_state_invariants()
    assert any("missing node" in error for error in errors)
    assert any("missing claim embedding" in error for error in errors)
    assert any("missing evidence episode" in error for error in errors)


def test_current_state_candidate_is_rejected_below_slot_similarity_threshold():
    captured = []
    class LLM:
        def chat(self, messages, **kwargs):
            captured.append(messages[-1]["content"])
            return SimpleNamespace(content='{"matched_claim_id":"TOKYO","operation":"SUPERSEDE","same_state_dimension":true,"confidence":0.9}')
    store = EventStateStore("p")
    store.add_claim(Claim("BOSTON", "Alice", "alice", "city", "Boston", status="superseded", evidence=[EvidenceRef("s1", "s1", [1])]), [1.0, 0.0], [1.0, 0.0])
    store.add_claim(Claim("TOKYO", "Alice", "alice", "city", "Tokyo", status="active", evidence=[EvidenceRef("s2", "s2", [2])]), [0.0, 1.0], [0.0, 1.0])
    compiler = StateCompiler(store, FakeEmbedder(), LLM(), candidate_top_k=2, current_candidate_top_k=1, min_similarity=0.9)
    result = compiler.apply(Claim("NEW", "Alice", "alice", "preferred_language", "English", evidence=[EvidenceRef("s3", "s3", [3])]), "s3", [1.0, 0.0], [1.0, 0.0])
    assert result.operation == "NEW"
    assert not captured


def _slot_claim(cid, slot, value, session, claim_vector=(1.0, 0.0), slot_vector=(1.0, 0.0)):
    return Claim(
        cid, "Alice", "alice", slot, value, state_slot=slot,
        evidence=[EvidenceRef(session, session, [1])],
    ), list(claim_vector), list(slot_vector)


def test_related_employer_and_job_role_cannot_form_destructive_chain():
    store = EventStateStore("p")
    employer, emb, slot_emb = _slot_claim("EMP", "employer", "Acme", "s1")
    store.add_claim(employer, emb, slot_emb)
    llm = SimpleNamespace(chat=lambda messages, **kwargs: SimpleNamespace(
        content='{"matched_claim_id":"EMP","operation":"SUPERSEDE","same_state_dimension":false,"confidence":1}'
    ))
    compiler = StateCompiler(store, FakeEmbedder(), llm, min_similarity=0.1)
    role, emb, slot_emb = _slot_claim("ROLE", "job_role", "backend engineer", "s2", slot_vector=(1.0, 0.0))
    result = compiler.apply(role, "s2", emb, slot_emb)
    assert result.operation == "NEW"
    assert employer.status == "active"
    assert compiler.different_state_dimension_guard_count == 1
    assert not any(edge["relation_type"] == "SUPERSEDES" for edge in store.edges)


def test_similar_state_slots_allow_predicate_drift_transition():
    store = EventStateStore("p")
    old = Claim("BOS", "Alice", "alice", "lives_in", "Boston", state_slot="residence_location", evidence=[EvidenceRef("s1", "s1", [1])])
    store.add_claim(old, [1.0, 0.0], [1.0, 0.0])
    llm = SimpleNamespace(chat=lambda messages, **kwargs: SimpleNamespace(
        content='{"matched_claim_id":"BOS","operation":"SUPERSEDE","same_state_dimension":true,"confidence":1}'
    ))
    compiler = StateCompiler(store, FakeEmbedder(), llm, min_similarity=0.9)
    new = Claim("TOK", "Alice", "alice", "current_city", "Tokyo", state_slot="current_residence", evidence=[EvidenceRef("s2", "s2", [1])])
    result = compiler.apply(new, "s2", [0.0, 1.0], [1.0, 0.0])
    assert result.operation == "SUPERSEDE"
    assert old.status == "superseded"
    assert new.status == "active"


def test_exact_slot_value_repeat_corrobates_without_classifier():
    store = EventStateStore("p")
    old = Claim("BOS", "Alice", "alice", "lives_in", "Boston", state_slot="residence_location", evidence=[EvidenceRef("s1", "s1", [1])])
    store.add_claim(old, [1.0, 0.0], [1.0, 0.0])
    calls = []
    compiler = StateCompiler(store, FakeEmbedder(), SimpleNamespace(chat=lambda *args, **kwargs: calls.append(args)))
    repeated = Claim("BOS2", "Alice", "alice", "current_city", "Boston", state_slot="residence_location", evidence=[EvidenceRef("s2", "s2", [1])])
    result = compiler.apply(repeated, "s2", [0.0, 1.0], [1.0, 0.0])
    assert result.operation == "CORROBORATE"
    assert not calls
    assert {ref.source_session_id for ref in old.evidence} == {"s1", "s2"}


def test_unrelated_state_slots_skip_classifier_until_same_slot_is_seen():
    calls = []
    llm = SimpleNamespace(chat=lambda *args, **kwargs: calls.append(args) or SimpleNamespace(
        content='{"matched_claim_id":"S0","operation":"SUPERSEDE","same_state_dimension":true,"confidence":1}'
    ))
    store = EventStateStore("p")
    compiler = StateCompiler(store, FakeEmbedder(), llm, min_similarity=0.9)

    for index in range(10):
        slot_vector = [1.0 if position == index else 0.0 for position in range(10)]
        item = Claim(
            f"S{index}", "Alice", "alice", f"attribute_{index}", "value",
            state_slot=f"attribute_{index}",
            evidence=[EvidenceRef(f"s{index}", f"s{index}", [1])],
        )
        assert compiler.apply(item, f"s{index}", [1.0, 0.0], slot_vector).operation == "NEW"
    assert not calls

    same_slot = Claim(
        "S10", "Alice", "alice", "attribute_0", "new value",
        state_slot="attribute_0",
        evidence=[EvidenceRef("s10", "s10", [1])],
    )
    assert compiler.apply(same_slot, "s10", [1.0, 0.0], [1.0] + [0.0] * 9).operation == "SUPERSEDE"
    assert len(calls) == 1


def test_no_information_values_are_rejected_but_explicit_negative_values_remain():
    from methods.event_state.validation import is_no_information_value
    assert is_no_information_value(" not specified ")
    assert is_no_information_value("N/A")
    assert not is_no_information_value("no")
    assert not is_no_information_value("none")
