from methods.smart_mem0.agent import SmartMem0Agent
from methods.smart_mem0.read_requirement_projection_runtime import (
    ReadRequirementProjectionRuntimeMixin,
)
from methods.smart_mem0.read_requirement_resolution_runtime import (
    ReadRequirementResolutionRuntimeMixin,
)


class _SlotBase:
    def _requirement_slot(self, requirement, ir, compiled_mode):
        del compiled_mode
        return {
            "id": requirement["id"],
            "evidence_role": "REQUIREMENT",
            "requested_projection": (ir.get("goal") or {}).get("projection", "TEXT"),
        }

    @staticmethod
    def _rg_requested_projection(slot):
        return slot.get("requested_projection", "TEXT")


class _SlotHarness(ReadRequirementResolutionRuntimeMixin, _SlotBase):
    @staticmethod
    def _date_for(memory, axis):
        return memory.get(axis, "")


class _MaterialBase:
    @staticmethod
    def _rg_material_binding(slot, memory, frame=None):
        del slot, memory, frame
        return {
            "level": "STRONG",
            "proof": False,
            "resolved_key": False,
            "views": 3,
            "need": 0.4,
            "local_retrieval": True,
        }

    @staticmethod
    def _rc_text(value):
        return " ".join(str(value or "").replace("_", " ").lower().split())

    @classmethod
    def _rc_owner(cls, value):
        text = cls._rc_text(value)
        return "primary_user" if text in {"primary user", "patient"} else text

    @classmethod
    def _rq_surface_similarity(cls, left, right):
        left_terms = set(cls._rc_text(left).split())
        right_terms = set(cls._rc_text(right).split())
        return len(left_terms & right_terms) / len(left_terms) if left_terms else 0.0


class _MaterialHarness(ReadRequirementResolutionRuntimeMixin, _MaterialBase):
    pass


class _StatusBase:
    def __init__(self):
        self.captured_support = {}

    @staticmethod
    def _semantic_context_slot(slot):
        return str(slot.get("evidence_role") or "").upper() == "REQUIREMENT"

    @staticmethod
    def _context_candidate_eligible(slot, memory):
        del slot, memory
        return True

    @staticmethod
    def _requirement_target_proof(slot, memory):
        del slot
        return bool(memory.get("proof"))

    @staticmethod
    def _slot_structure_covered(slot, support_ids, selected, relations):
        del slot, support_ids, selected, relations
        return True

    @staticmethod
    def _rg_selector_relation(slot):
        return str((slot.get("selector") or {}).get("relation") or "")

    @staticmethod
    def _rg_answerability_key(slot, memory, original_index):
        del slot, memory
        return (original_index,)

    def _retrieval_status(self, plan, slot_support, selected, relations):
        del plan, selected, relations
        self.captured_support = {
            key: list(value) for key, value in slot_support.items()
        }
        complete = bool(slot_support.get("r1"))
        return {"r1": "FOUND" if complete else "EMPTY"}, {}, complete

    @staticmethod
    def _date_for(memory, axis):
        return memory.get(axis, "")


class _StatusHarness(ReadRequirementResolutionRuntimeMixin, _StatusBase):
    pass


class _ContextBase:
    def __init__(self):
        self._last_requirement_proof_support = {}
        self._last_query_memory_alignment = {}
        self.memories = {}

    @staticmethod
    def _semantic_context_slot(slot):
        return str(slot.get("evidence_role") or "").upper() == "REQUIREMENT"

    def _alignment_memory(self, memory_id):
        return self.memories.get(memory_id, {})

    @staticmethod
    def _rg_selector_relation(slot):
        return str((slot.get("selector") or {}).get("relation") or "")

    @staticmethod
    def _date_for(memory, axis):
        return memory.get(axis, "")

    @staticmethod
    def _rg_answerability_key(slot, memory, original_index):
        del slot
        return (-float(memory.get("semantic_score", 0.0)), original_index)

    @staticmethod
    def _rg_answer_signature(slot, memory):
        del slot
        return (str(memory.get("value") or memory.get("event_time") or ""),)

    @staticmethod
    def _role_aware_support_ids(slots, slot_support, candidate_order, limit):
        del slots, slot_support
        return list(candidate_order[:limit])


class _ContextHarness(ReadRequirementResolutionRuntimeMixin, _ContextBase):
    pass


def test_resolution_runtime_precedes_projection_runtime_in_active_mro():
    mro = SmartMem0Agent.mro()
    assert mro.index(ReadRequirementResolutionRuntimeMixin) < mro.index(
        ReadRequirementProjectionRuntimeMixin
    )


def test_goal_option_set_projection_does_not_make_each_requirement_an_option_set():
    harness = _SlotHarness()
    slot = harness._requirement_slot(
        {"id": "r1"},
        {"goal": {"projection": "OPTION_SET"}, "requirements": [{"id": "r1"}]},
        "MULTI_OPTION",
    )
    assert slot["goal_projection"] == "OPTION_SET"
    assert slot["premise_projection"] == "FACT"
    evidence = harness._rg_projection_evidence(
        slot, {"id": "m1", "claim": "Participant-specific premise"}
    )
    assert evidence["projection"] == "FACT"
    assert evidence["capable"] is True


def test_multi_requirement_goal_projection_is_not_imposed_on_each_premise():
    harness = _SlotHarness()
    slot = harness._requirement_slot(
        {"id": "r1"},
        {
            "goal": {"projection": "VALUE"},
            "requirements": [{"id": "r1"}, {"id": "r2"}],
        },
        "DIRECT",
    )
    assert slot["goal_projection"] == "VALUE"
    assert slot["premise_projection"] == "FACT"


def test_single_direct_requirement_keeps_goal_projection():
    harness = _SlotHarness()
    slot = harness._requirement_slot(
        {"id": "r1"},
        {"goal": {"projection": "VALUE"}, "requirements": [{"id": "r1"}]},
        "DIRECT",
    )
    assert slot["premise_projection"] == "VALUE"


def test_owner_anchor_alone_cannot_create_strong_material_binding():
    harness = _MaterialHarness()
    binding = harness._rg_material_binding(
        {"material_anchors": ["primary_user"]},
        {
            "id": "m1",
            "subject_id": "primary_user",
            "subject": "patient",
            "scope": "symptom",
            "entities": ["patient"],
        },
    )
    assert binding["level"] != "STRONG"
    assert binding["topic_anchor_guard"] == "DOWNGRADED_OWNER_OR_ROLE_ONLY"


def test_topic_anchor_can_still_create_strong_material_binding():
    harness = _MaterialHarness()
    binding = harness._rg_material_binding(
        {"material_anchors": ["primary_user", "medical_history"]},
        {
            "id": "m1",
            "subject_id": "primary_user",
            "subject": "patient",
            "scope": "medical_history",
            "entities": ["patient"],
        },
    )
    assert binding["level"] == "STRONG"
    assert binding["topic_anchor_guard"] == "PASS"


def test_strict_proof_is_admitted_from_retrieved_candidate_world_not_only_old_slot_lane():
    harness = _StatusHarness()
    support = {"r1": []}
    status, _, complete = harness._retrieval_status(
        {"required_slots": [{"id": "r1", "evidence_role": "REQUIREMENT"}]},
        support,
        [{"id": "noise", "proof": False}, {"id": "proof", "proof": True}],
        [],
    )
    assert support["r1"] == ["proof"]
    assert harness.captured_support["r1"] == ["proof"]
    assert status == {"r1": "FOUND"}
    assert complete is True


def test_earliest_selector_overrides_relevance_order_before_compaction():
    harness = _ContextHarness()
    harness.memories = {
        "late": {
            "id": "late",
            "event_time": "2024-01-07",
            "value": "later state",
            "semantic_score": 1.0,
        },
        "early": {
            "id": "early",
            "event_time": "2024-01-06",
            "value": "earlier state",
            "semantic_score": 0.5,
        },
    }
    harness._last_requirement_proof_support = {"r1": ["late", "early"]}
    selected = harness._role_aware_support_ids(
        [
            {
                "id": "r1",
                "evidence_role": "REQUIREMENT",
                "selector": {"relation": "EARLIEST", "axis": "event_time"},
            }
        ],
        {"r1": ["late", "early"]},
        ["late", "early"],
        2,
    )
    assert selected[0] == "early"
    assert (
        harness._last_query_memory_alignment["requirement_resolution"]["r1"][
            "reason"
        ]
        == "EARLIEST"
    )


def test_competing_proof_surfaces_are_preserved_instead_of_collapsed_to_first_proof():
    harness = _ContextHarness()
    harness.memories = {
        "a": {"id": "a", "value": "A", "semantic_score": 3.0},
        "b": {"id": "b", "value": "B", "semantic_score": 2.0},
        "c": {"id": "c", "value": "C", "semantic_score": 1.0},
    }
    harness._last_requirement_proof_support = {"r1": ["a", "b", "c"]}
    selected = harness._role_aware_support_ids(
        [{"id": "r1", "evidence_role": "REQUIREMENT", "selector": {}}],
        {"r1": ["a", "b", "c"]},
        ["a", "b", "c"],
        3,
    )
    assert selected == ["a", "b", "c"]
    resolution = harness._last_query_memory_alignment["requirement_resolution"]["r1"]
    assert resolution["resolved"] is False
    assert resolution["reason"] == "COMPETING_PROOF_SURFACES"
