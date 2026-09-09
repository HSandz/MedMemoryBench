from methods.smart_mem0.read_reasoning_completion import ReadReasoningCompletionMixin


SLOT = {
    "id": "r1",
    "type": "DIRECT",
    "evidence_role": "REQUIREMENT",
    "target_surface": "target fact",
    "proof_anchor": "target fact",
}


class _BaseHarness:
    def __init__(self):
        self._memories = []
        self.calls = []
        self._last_retrieval_viability = {}

    @staticmethod
    def _semantic_context_slot(slot):
        return str(slot.get("evidence_role") or "").upper() in {"REQUIREMENT", "COMPARAND"}

    @staticmethod
    def _requirement_target_proof(slot, memory):
        target = str(slot.get("proof_anchor") or slot.get("target_surface") or "").lower()
        return bool(target and target in str(memory.get("claim") or "").lower())

    @staticmethod
    def _query_visible_memory(memory, include_history=False):
        del include_history
        return not memory.get("hidden")

    @staticmethod
    def _rc_owner_match(slot, memory):
        del slot
        return memory.get("owner", "u") == "u"

    def _compile_gap_operations(self, slots, question, budget_tier="MEDIUM", plan=None):
        del question, budget_tier, plan
        return [{"op": "SEARCH_FAMILY", "produces": [slots[0]["id"]]}]

    def _execute_operation(self, operation, outputs, seeds, frame=None):
        del outputs, seeds, frame
        query = operation.get("query", "primary")
        self.calls.append(query)
        rows = {
            "primary": [{"id": "d", "claim": "topical decoy", "capsule_id": "c1", "atom_id": "s1:w0:a0"}],
            "keys": [{"id": "p", "claim": "this is the target fact", "capsule_id": "c2", "atom_id": "s2:w0:a0"}],
            "family": [{"id": "f", "claim": "another fact"}],
            "question": [],
        }
        return [dict(row) for row in rows.get(query, [])], [], []

    @staticmethod
    def _retrieval_status(plan, slot_support, selected, relations):
        del plan, slot_support, selected, relations
        return {"r1": "FOUND"}, {}, True

    def _run_query_retrieval(self, *args, **kwargs):
        del args, kwargs
        return {}

    def prepare_batch_query(self, *args, **kwargs):
        del args, kwargs
        return {"extra": {}}


class _Harness(ReadReasoningCompletionMixin, _BaseHarness):
    pass


def _operation(*, reasoning=False):
    return {
        "op": "SEARCH_FAMILY",
        "query": "primary",
        "produces": ["r1"],
        "_acquisition_requirement": dict(SLOT),
        "_reasoning_acquisition": reasoning,
        "retrieval_views": [
            {"kind": "obligation", "query": "primary"},
            {"kind": "keys", "query": "keys"},
            {"kind": "family", "query": "family"},
            {"kind": "question", "query": "question"},
        ],
    }


def test_compile_marks_reasoning_requirement_without_query_type_routing():
    harness = _Harness()
    plan = {
        "query_spec": {"requires_inference": True},
        "required_slots": [dict(SLOT)],
        "semantic_relations": [{"type": "INFER", "from": "r1", "to": "ANSWER"}],
    }
    operations = harness._compile_gap_operations([dict(SLOT)], "q", plan=plan)
    assert operations[0]["_reasoning_acquisition"] is True
    assert operations[0]["_acquisition_requirement"]["id"] == "r1"


def test_proof_miss_opens_auxiliary_view_and_promotes_target_proof():
    harness = _Harness()
    rows, _, _ = harness._execute_operation(_operation(), [], [], None)
    assert [row["id"] for row in rows[:2]] == ["p", "d"]
    assert harness.calls == ["primary", "keys"]
    stats = harness._rcm_stats()
    assert stats["proof_gate_misses"] == 1
    assert stats["auxiliary_views_opened"] == 1
    assert stats["auxiliary_candidate_additions"] == 1


def test_target_proof_miss_changes_structural_found_to_empty_for_recovery():
    harness = _Harness()
    status, _, complete = harness._retrieval_status(
        {"required_slots": [dict(SLOT)]},
        {"r1": ["d"]},
        [{"id": "d", "claim": "topical decoy"}],
        [],
    )
    assert status == {"r1": "EMPTY"}
    assert complete is False
    assert harness._last_retrieval_viability["candidate_viability_requirements"] == {"r1": "FOUND"}


def test_option_context_is_not_converted_into_truth_gate():
    harness = _Harness()
    option_slot = dict(SLOT, evidence_role="OPTION_CONTEXT")
    status, _, complete = harness._retrieval_status(
        {"required_slots": [option_slot]},
        {"r1": ["d"]},
        [{"id": "d", "claim": "topical decoy"}],
        [],
    )
    assert status == {"r1": "FOUND"}
    assert complete is True


def test_reasoning_endpoint_keeps_bounded_same_capsule_neighbor():
    harness = _Harness()
    harness._memories = [
        {"id": "p", "claim": "this is the target fact", "capsule_id": "c2", "session_idx": 2, "atom_id": "s2:w0:a0"},
        {"id": "n1", "claim": "adjacent participant premise", "capsule_id": "c2", "session_idx": 2, "atom_id": "s2:w0:a1"},
        {"id": "n2", "claim": "second adjacent premise", "capsule_id": "c2", "session_idx": 2, "atom_id": "s2:w0:a2"},
        {"id": "far", "claim": "unrelated source", "capsule_id": "c9", "session_idx": 9, "atom_id": "s9:w0:a0"},
    ]
    op = _operation(reasoning=True)
    op["query"] = "keys"
    op["retrieval_views"][0]["query"] = "keys"
    rows, _, _ = harness._execute_operation(op, [], [], None)
    ids = [row["id"] for row in rows]
    assert "p" in ids
    assert "n1" in ids and "n2" in ids
    assert "far" not in ids
    neighbors = [row for row in rows if row.get("_source_neighbor_context")]
    assert len(neighbors) <= harness.ACQUISITION_SOURCE_NOVEL


def test_candidate_world_remains_hard_bounded():
    harness = _Harness()
    harness.ACQUISITION_HARD_CAP = 3
    harness._memories = [
        {"id": f"n{i}", "claim": f"neighbor {i}", "capsule_id": "c2", "session_idx": 2, "atom_id": f"s2:w0:a{i+1}"}
        for i in range(10)
    ]
    rows, _, _ = harness._execute_operation(_operation(reasoning=True), [], [], None)
    assert len(rows) <= 3


class _ColdHarness(_Harness):
    def _retrieval_tier_ids(self, frame, tier, include_history=False):
        del frame, include_history
        return {"cold-proof"} if tier == "COLD" else set()

    def _hybrid_search(self, query, top_k, candidate_ids=None):
        del query, top_k
        if candidate_ids and "cold-proof" in set(candidate_ids):
            return [{"id": "cold-proof", "claim": "target fact from cold", "owner": "u"}]
        return []

    def _execute_operation(self, operation, outputs, seeds, frame=None):
        if operation.get("retrieval_views"):
            return ReadReasoningCompletionMixin._execute_operation(self, operation, outputs, seeds, frame)
        query = operation.get("query", "primary")
        self.calls.append(query)
        return ([{"id": "d", "claim": "topical decoy"}] if query == "primary" else []), [], []


def test_proof_miss_can_open_cold_despite_topically_strong_hot_path():
    harness = _ColdHarness()
    op = _operation()
    op["retrieval_views"] = [
        {"kind": "obligation", "query": "primary"},
        {"kind": "keys", "query": "no-key-hit"},
        {"kind": "family", "query": "no-family-hit"},
    ]
    rows, _, _ = ReadReasoningCompletionMixin._execute_operation(harness, op, [], [], None)
    assert "cold-proof" in [row["id"] for row in rows]
    stats = harness._rcm_stats()
    assert stats["proof_cold_expansions"] == 1
    assert stats["proof_cold_additions"] == 1


def test_reasoning_prompt_is_strengthened_without_an_extra_model_stage():
    class _PromptBase(_BaseHarness):
        def prepare_batch_query(self, *args, **kwargs):
            del args, kwargs
            return {
                "messages": [
                    {"role": "system", "content": "base"},
                    {"role": "user", "content": "q"},
                ],
                "extra": {
                    "plan": {
                        "query_spec": {"requires_inference": True},
                        "semantic_relations": [{"type": "INFER", "from": "r1", "to": "ANSWER"}],
                    }
                },
            }

    class _PromptHarness(ReadReasoningCompletionMixin, _PromptBase):
        pass

    prepared = _PromptHarness().prepare_batch_query("q")
    assert "REASONING COMPLETION" in prepared["messages"][0]["content"]
    assert prepared["extra"]["reasoning_completion"]["reasoning_prompt_strengthened"] is True


def test_temporal_selector_contract_is_not_replaced_by_direct_target_gate():
    harness = _Harness()
    temporal_slot = dict(SLOT, type="TEMPORAL")
    status, _, complete = harness._retrieval_status(
        {"required_slots": [temporal_slot]},
        {"r1": ["d"]},
        [{"id": "d", "claim": "topical decoy"}],
        [],
    )
    assert status == {"r1": "FOUND"}
    assert complete is True
