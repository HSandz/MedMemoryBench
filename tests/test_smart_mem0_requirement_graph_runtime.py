from methods.smart_mem0.agent import SmartMem0Agent
from methods.smart_mem0.read_requirement_graph import ReadRequirementGraphMixin
from methods.smart_mem0.read_requirement_graph_runtime import ReadRequirementGraphRuntimeMixin
from methods.smart_mem0.read_query_memory_alignment import ReadQueryMemoryAlignmentMixin


class _SlotBase:
    def _requirement_slot(self, requirement, ir, compiled_mode):
        del compiled_mode
        return {
            "id": requirement["id"],
            "need": requirement.get("need", ""),
            "subject_id": "",
            "subject": "",
            "evidence_role": "OPTION_CONTEXT" if ir.get("visible_options") else "REQUIREMENT",
            "material_anchors": list(requirement.get("material_anchors") or []),
            "resolved_keys": [],
        }

    @staticmethod
    def _rc_resolve_target_keys(target, subject_id=""):
        del subject_id
        return ["resolved canonical"] if target else []

    @staticmethod
    def _rg_unique_text(values, limit=8):
        output = []
        for value in values:
            if value and value not in output:
                output.append(value)
        return output[:limit]


class _SlotHarness(ReadRequirementGraphRuntimeMixin, _SlotBase):
    pass


class _CurrentBase:
    @staticmethod
    def _rg_selector_compatible(slot, memory):
        del slot, memory
        return True

    @staticmethod
    def _rg_selector_relation(slot):
        return (slot.get("selector") or {}).get("relation", "")

    @staticmethod
    def _is_state_head(memory):
        return bool(memory.get("head"))


class _CurrentHarness(ReadRequirementGraphRuntimeMixin, _CurrentBase):
    pass


def test_active_mro_puts_runtime_and_requirement_graph_before_old_alignment():
    mro = SmartMem0Agent.mro()
    assert mro.index(ReadRequirementGraphRuntimeMixin) < mro.index(ReadRequirementGraphMixin)
    assert mro.index(ReadRequirementGraphMixin) < mro.index(ReadQueryMemoryAlignmentMixin)


def test_runtime_derives_canonical_addresses_when_llm_anchor_is_missing():
    harness = _SlotHarness()
    slot = harness._requirement_slot(
        {"id": "r1", "need": "participant variable", "material_anchors": []},
        {},
        "DIRECT",
    )
    assert slot["material_anchors"] == ["resolved canonical"]
    assert slot["canonical_anchor_source"]["runtime_resolved"] == ["resolved canonical"]


def test_candidate_set_requirement_stays_shared_participant_evidence():
    harness = _SlotHarness()
    slot = harness._requirement_slot(
        {"id": "r1", "need": "participant constraint"},
        {"visible_options": {"A": "a", "B": "b"}},
        "MULTI_OPTION",
    )
    assert slot["evidence_role"] == "REQUIREMENT"


def test_current_material_binding_accepts_only_actual_state_head():
    harness = _CurrentHarness()
    slot = {"selector": {"relation": "CURRENT"}}
    assert harness._rg_selector_compatible(slot, {"id": "old", "head": False}) is False
    assert harness._rg_selector_compatible(slot, {"id": "new", "head": True}) is True
