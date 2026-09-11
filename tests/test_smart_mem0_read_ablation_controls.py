from methods.smart_mem0.agent import SmartMem0Agent
from methods.smart_mem0.read_ablation_controls import ReadAblationControlsMixin
from methods.smart_mem0.read_reasoning_completion import ReadReasoningCompletionMixin
from methods.smart_mem0.read_requirement_identity_runtime import (
    ReadRequirementIdentityRuntimeMixin,
)


class _ControlBase:
    QUERY_ORCHESTRATOR_VERSION = "orchestrator-test"
    CONTROLLER_SCHEMA_VERSION = "controller-test"
    REQUIREMENT_GRAPH_VERSION = "graph-test"
    REQUIREMENT_GRAPH_RUNTIME_VERSION = "runtime-test"
    PROJECTION_PROOF_VERSION = "proof-test"
    REQUIREMENT_IDENTITY_VERSION = "identity-test"
    REQUIREMENT_RESOLUTION_VERSION = "resolution-test"
    REASONING_COMPLETION_VERSION = "reasoning-test"

    def __init__(self):
        self.enable_reasoning_completion = True
        self.enable_proof_status_gate = True
        self.enable_proof_expansion = True
        self.enable_reasoning_source_neighbors = True
        self.enable_reasoning_prompt_strengthening = True
        self.enable_zero_result_recovery = True
        self.base_slot_covered_calls = 0
        self.structural_slot_covered_calls = 0
        self.base_proof_calls = 0
        self.base_retrieval_status_calls = 0

    def _slot_covered(self, slot, support_ids, selected, relations):
        self.base_slot_covered_calls += 1
        return False

    def _slot_structure_covered(self, slot, support_ids, selected, relations):
        self.structural_slot_covered_calls += 1
        return bool(support_ids)

    def _rcm_requires_proof(self, slot):
        self.base_proof_calls += 1
        return True

    def _retrieval_status(self, plan, slot_support, selected, relations):
        self.base_retrieval_status_calls += 1
        return {"r1": "BASE"}, {}, False

    @staticmethod
    def _rcm_reasoning_ids(plan):
        return {"r1"}

    @staticmethod
    def _rcm_source_neighbors(slot, rows, existing_ids, limit):
        return [{"id": "m-neighbor"}]

    @staticmethod
    def _rcm_strengthen_prompt(prepared):
        return True

    @staticmethod
    def prepare_batch_query(question, system_message=None, **kwargs):
        return {"question": question, "extra": {}}


class _Harness(ReadAblationControlsMixin, _ControlBase):
    pass


def _structural_plan():
    return {
        "required_slots": [
            {
                "id": "r1",
                "type": "DIRECT",
                "evidence_role": "REQUIREMENT",
            }
        ],
        "semantic_relations": [],
    }


def test_ablation_controls_precede_identity_and_reasoning_completion_in_active_mro():
    mro = SmartMem0Agent.mro()
    assert mro.index(ReadAblationControlsMixin) < mro.index(
        ReadRequirementIdentityRuntimeMixin
    )
    assert mro.index(ReadAblationControlsMixin) < mro.index(
        ReadReasoningCompletionMixin
    )


def test_default_controls_preserve_existing_status_path():
    harness = _Harness()
    status, _, complete = harness._retrieval_status(
        _structural_plan(), {"r1": ["m1"]}, [{"id": "m1"}], []
    )
    assert status == {"r1": "BASE"}
    assert complete is False
    assert harness.base_retrieval_status_calls == 1


def test_disabling_proof_status_gate_uses_structural_retrieval_completion():
    harness = _Harness()
    harness.enable_proof_status_gate = False
    status, relations, complete = harness._retrieval_status(
        _structural_plan(), {"r1": ["m1"]}, [{"id": "m1"}], []
    )
    assert status == {"r1": "FOUND"}
    assert relations == {}
    assert complete is True
    assert harness.base_retrieval_status_calls == 0
    assert harness.structural_slot_covered_calls == 1
    assert harness._last_requirement_graph_stop_guard["proof_status_gate_enabled"] is False


def test_disabling_proof_expansion_does_not_disable_status_path():
    harness = _Harness()
    harness.enable_proof_expansion = False
    assert harness._rcm_requires_proof({}) is False
    status, _, complete = harness._retrieval_status(
        _structural_plan(), {"r1": ["m1"]}, [{"id": "m1"}], []
    )
    assert status == {"r1": "BASE"}
    assert complete is False
    assert harness.base_retrieval_status_calls == 1


def test_disabling_reasoning_completion_disables_completion_side_effects():
    harness = _Harness()
    harness.enable_reasoning_completion = False
    assert harness._rcm_requires_proof({}) is False
    assert harness._rcm_reasoning_ids({}) == set()
    assert harness._rcm_source_neighbors({}, [], set(), 2) == []
    assert harness._rcm_strengthen_prompt({}) is False
    status, _, complete = harness._retrieval_status(
        _structural_plan(), {"r1": ["m1"]}, [{"id": "m1"}], []
    )
    assert status == {"r1": "FOUND"}
    assert complete is True


def test_ablation_telemetry_reports_controls_and_active_version_stack():
    harness = _Harness()
    harness.enable_proof_status_gate = False
    harness.enable_proof_expansion = False
    prepared = harness.prepare_batch_query("question")
    extra = prepared["extra"]

    controls = extra["read_ablation_controls"]
    assert controls["version"] == "read-ablation-controls-v1"
    assert controls["enable_proof_status_gate"] is False
    assert controls["enable_proof_expansion"] is False
    assert controls["enable_zero_result_recovery"] is True

    versions = extra["active_read_version_stack"]
    assert versions == {
        "orchestrator": "orchestrator-test",
        "controller_schema": "controller-test",
        "requirement_graph": "graph-test",
        "requirement_graph_runtime": "runtime-test",
        "proof": "proof-test",
        "requirement_identity": "identity-test",
        "requirement_resolution": "resolution-test",
        "reasoning_completion": "reasoning-test",
    }
