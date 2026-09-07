"""Regression tests for the generalized SmartMem0 evidence policy."""

from copy import deepcopy

from methods.smart_mem0.read_evidence_policy import ReadEvidencePolicyMixin


class _Base:
    HARD_MEMORY_LIMIT = 8

    @staticmethod
    def _rc_text(value):
        return " ".join(str(value or "").casefold().split())

    @staticmethod
    def _question_stem(question):
        return str(question).split("\nA.")[0].strip()

    @staticmethod
    def _normalize_candidate_propositions(values):
        return dict(values or {})

    def _rc_search_query(self, slot, question):
        del question
        return str(slot.get("evidence_family") or "")

    def _rc_bundle_query(self, slots, question):
        del question
        return " | ".join(str(slot.get("evidence_family") or "") for slot in slots)

    def _build_candidate_proposition_pack(self, propositions, frame=None, seeds=None, question=""):
        del frame, seeds, question
        self.last_pack_input = dict(propositions)
        self._last_candidate_propositions = dict(propositions)
        pack = {"candidate_set": dict(propositions), "propositions": dict(propositions)}
        self._last_candidate_proposition_pack = deepcopy(pack)
        return pack

    def _semantic_operation_search(self, query, top_k, strategy, frame=None, option_queries=None):
        del top_k, frame
        self.last_semantic_search = {
            "query": query,
            "strategy": strategy,
            "option_queries": deepcopy(option_queries or []),
        }
        return []

    def _controller_plan(self, ir, question, frame):
        del question, frame
        return {
            "budget_tier": "SMALL",
            "max_memories": 3,
            "required_slots": deepcopy(ir.get("required_slots") or []),
            "candidate_propositions": deepcopy(ir.get("candidate_propositions") or {}),
            "reasoning_bridges": deepcopy(ir.get("reasoning_bridges") or []),
            "semantic_ir": deepcopy(ir.get("semantic_ir") or {}),
            "operations": deepcopy(ir.get("operations") or []),
        }

    def _retrieval_status(self, plan, slot_support, selected, relations):
        del plan, slot_support, selected, relations
        return {"r1": "EMPTY"}, {"CAUSES:r1:r2": "UNPROVEN"}, False

    def _role_aware_support_ids(self, slots, slot_support, candidate_order, limit):
        del slots, slot_support
        return list(candidate_order)[:limit]

    def _reconstruct_beliefs(self, memories, limit):
        output = list(memories)[:limit]
        output.append({"id": "expanded", "claim": "must be removed"})
        return output, [
            {"source_id": output[0]["id"], "target_id": "expanded", "type": "RELATED"}
        ]


class _Harness(ReadEvidencePolicyMixin, _Base):
    pass


def test_semantic_recall_keeps_original_question_predicate():
    harness = _Harness()
    query = harness._rc_search_query(
        {"evidence_family": "medication suitability"},
        "Should I take this medicine given my history?",
    )
    assert "medication suitability" in query
    assert "Should I take this medicine given my history?" in query


def test_multilingual_question_surface_survives_recall_expansion():
    harness = _Harness()
    question = "Tôi có nên tăng liều thuốc buổi tối không?"
    query = harness._rc_search_query(
        {"evidence_family": "điều chỉnh liều thuốc"}, question
    )
    assert "điều chỉnh liều thuốc" in query
    assert question in query


def test_candidate_pack_conditions_each_proposition_on_shared_predicate():
    harness = _Harness()
    pack = harness._build_candidate_proposition_pack(
        {"A": "increase dose", "B": "keep dose"},
        question="What is the safest action?\nA. increase dose\nB. keep dose",
    )
    assert harness.last_pack_input["A"].startswith("What is the safest action? | ")
    assert harness.last_pack_input["B"].startswith("What is the safest action? | ")
    assert pack["candidate_set"] == {"A": "increase dose", "B": "keep dose"}
    assert pack["retrieval_query_semantics"] == "shared_predicate_plus_candidate"


def test_shared_option_physical_search_is_predicate_conditioned():
    harness = _Harness()
    harness._semantic_operation_search(
        "late-night glucose management",
        8,
        "SHARED_OPTIONS",
        option_queries=[{"label": "A", "query": "increase insulin"}],
    )
    conditioned = harness.last_semantic_search["option_queries"][0]["query"]
    assert "late-night glucose management" in conditioned
    assert "increase insulin" in conditioned


def test_coverage_floor_does_not_change_compute_tier():
    harness = _Harness()
    ir = {
        "required_slots": [{"id": "r1"}],
        "candidate_propositions": {"A": "a", "B": "b", "C": "c", "D": "d"},
        "reasoning_bridges": [],
        "operations": [{"op": "SEARCH_FAMILY", "family_mode": "candidate_set", "top_k": 3}],
    }
    plan = harness._controller_plan(ir, "q", None)
    assert plan["budget_tier"] == "SMALL"
    assert plan["max_memories"] >= 5
    assert plan["context_coverage_floor"] >= 5
    assert plan["operations"][0]["top_k"] >= 5


def test_reasoning_bridge_has_richer_floor_without_query_type_branch():
    harness = _Harness()
    plan = harness._controller_plan(
        {
            "required_slots": [{"id": "r1"}],
            "reasoning_bridges": [{"type": "INFER", "from": "r1", "to": "ANSWER"}],
        },
        "q",
        None,
    )
    assert plan["budget_tier"] == "SMALL"
    assert plan["max_memories"] >= harness.MIN_BRIDGE_CONTEXT


def test_nonempty_typed_support_is_viable_without_certificate():
    harness = _Harness()
    statuses, relations, complete = harness._retrieval_status(
        {"required_slots": [{"id": "r1"}]},
        {"r1": ["m1"]},
        [{"id": "m1"}],
        [],
    )
    assert statuses == {"r1": "FOUND"}
    assert relations == {"CAUSES:r1:r2": "UNPROVEN"}
    assert complete is True


def test_temporal_selector_support_is_first_in_context():
    harness = _Harness()
    slots = [{"id": "r1", "time_relation": "EARLIEST"}]
    selected = harness._role_aware_support_ids(
        slots,
        {"r1": ["selector_winner"]},
        ["family_root", "selector_winner", "other"],
        3,
    )
    assert selected[0] == "selector_winner"


def test_arbitration_cannot_expand_authorized_memory_ids():
    harness = _Harness()
    beliefs, relations = harness._reconstruct_beliefs(
        [{"id": "m1", "claim": "authorized"}], 3
    )
    assert [memory["id"] for memory in beliefs] == ["m1"]
    assert relations == []


def test_bridge_block_materializes_endpoints_and_permissions():
    harness = _Harness()
    block = harness._reasoning_bridge_block(
        {
            "semantic_ir": {
                "requirements": [
                    {"id": "r1", "answer_obligation": "night glucose spike"},
                    {"id": "r2", "answer_obligation": "morning symptoms"},
                ]
            },
            "reasoning_bridges": [
                {
                    "type": "POSSIBLE_CAUSE",
                    "from": "r1",
                    "to": "r2",
                    "world_knowledge_allowed": True,
                    "requires_stored_relation": False,
                    "goal": "explain the mechanism",
                }
            ],
        }
    )
    assert "night glucose spike" in block
    assert "morning symptoms" in block
    assert "general_domain_knowledge=AUTHORIZED" in block
    assert "explain the mechanism" in block
