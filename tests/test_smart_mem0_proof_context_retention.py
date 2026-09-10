from methods.smart_mem0.read_proof_context_retention import (
    ReadProofContextRetentionMixin,
)


class _Base:
    def __init__(self):
        self._last_requirement_proof_support = {}
        self._last_requirement_context_candidates = {}
        self._last_requirement_status = {}
        self._last_candidate_propositions = {}
        self._last_proposition_probe_coverage = {}
        self._last_candidate_local_coverage = {}

    @staticmethod
    def _semantic_context_slot(slot):
        return str(slot.get("evidence_role") or "").upper() in {
            "REQUIREMENT",
            "COMPARAND",
        }

    def _role_aware_support_ids(self, slots, slot_support, candidate_order, limit):
        del slots, slot_support
        return list(candidate_order[:limit])

    def _run_query_retrieval(self, *args, **kwargs):
        del args, kwargs
        return {}

    def prepare_batch_query(self, *args, **kwargs):
        del args, kwargs
        return {"extra": {}}


class _Harness(ReadProofContextRetentionMixin, _Base):
    pass


def test_rescues_candidate_lane_without_growing_context():
    harness = _Harness()
    harness._last_candidate_propositions = {
        "A": "a",
        "B": "b",
        "C": "c",
        "D": "d",
    }
    harness._last_proposition_probe_coverage = {
        "A": ["m24", "m46"],
        "B": ["m49", "m82"],
        "C": ["m49", "m7"],
        "D": ["m74", "m24"],
    }
    order = ["m24", "m46", "m7", "m74", "m82", "m85", "m49"]
    selected = harness._role_aware_support_ids([], {}, order, 6)

    assert len(selected) == 6
    assert "m49" in selected
    assert harness._last_context_retention["budget_unchanged"] is True
    assert "m49" in harness._last_context_retention["rescued_ids"]


def test_reserves_requirement_proof_if_baseline_would_drop_it():
    harness = _Harness()
    harness._last_requirement_proof_support = {"r1": ["proof"]}
    harness._last_requirement_status = {"r1": "FOUND"}
    slot = {"id": "r1", "type": "DIRECT", "evidence_role": "REQUIREMENT"}
    order = ["a", "b", "c", "d", "proof"]

    selected = harness._role_aware_support_ids(
        [slot], {"r1": ["proof"]}, order, 4
    )

    assert selected[0] == "proof"
    assert len(selected) == 4


def test_does_not_admit_unauthorized_probe_id():
    harness = _Harness()
    harness._last_candidate_propositions = {"A": "a"}
    harness._last_proposition_probe_coverage = {"A": ["outside", "inside"]}

    selected = harness._role_aware_support_ids(
        [], {}, ["inside", "x"], 2
    )

    assert selected == ["inside", "x"]
    assert "outside" not in selected


def test_selector_slot_is_not_reinterpreted_by_retention():
    harness = _Harness()
    harness._last_requirement_proof_support = {"r1": ["proof"]}
    harness._last_requirement_status = {"r1": "FOUND"}
    slot = {"id": "r1", "type": "TEMPORAL", "evidence_role": "REQUIREMENT"}

    selected = harness._role_aware_support_ids(
        [slot], {"r1": ["proof"]}, ["a", "proof"], 1
    )

    assert selected == ["a"]


def test_shared_probe_memory_represents_multiple_propositions_once():
    harness = _Harness()
    harness._last_candidate_propositions = {"A": "a", "B": "b"}
    harness._last_proposition_probe_coverage = {
        "A": ["shared"],
        "B": ["shared"],
    }

    selected = harness._role_aware_support_ids(
        [], {}, ["x", "shared", "y"], 2
    )

    assert selected[0] == "shared"
    assert selected.count("shared") == 1
    assert len(selected) == 2
