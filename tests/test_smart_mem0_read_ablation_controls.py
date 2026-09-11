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
        self.status_phase_proof_required = None

    def _slot_covered(self, slot, support_ids, selected, relations):
        self.base_slot_covered_calls += 1
        return False

    def _slot_structure_covered(self, slot, support_ids, selected, relations):
        self.structural_slot_covered_calls += 1
        return True

    def _rcm_requires_proof(self, slot):
        self.base_proof_calls += 1
        return True

    def _retrieval_status(self, plan, slot_support, selected, relations):
        self.status_phase_proof_required = self._rcm_requires_proof({})
        return {}, {}, bool(self.status_phase_proof_required)

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


def test_ablation_controls_precede_identity_and_reasoning_completion_in_active_mro():
    mro = SmartMem0Agent.mro()
    assert mro.index(ReadAblationControlsMixin) < mro.index(
        ReadRequirementIdentityRuntimeMixin
    )
    assert mro.index(ReadAblationControlsMixin) < mro.index(
        ReadReasoningCompletionMixin
    )


def test_default_controls_preserve_existing_proof_status_gate():
    harness = _Harness()
    assert harness._slot_covered({}, [], [], []) is False
    assert harness.base_slot_covered_calls == 1
    assert harness.structural_slot_covered_calls == 0
    _, _, complete = harness._retrieval_status({}, {}, [], [])
    assert complete is True
    assert harness.status_phase_proof_required is True


def test_disabling_proof_status_gate_uses_structural_coverage_and_disables_status_proof():
    harness = _Harness()
    harness.enable_proof_status_gate = False
    assert harness._slot_covered({}, [], [], []) is True
    assert harness.base_slot_covered_calls == 0
    assert harness.structural_slot_covered_calls == 1
    _, _, complete = harness._retrieval_status({}, {}, [], [])
    assert complete is False
    assert harness.status_phase_proof_required is False


def test_disabling_proof_expansion_does_not_disable_status_gate():
    harness = _Harness()
    harness.enable_proof_expansion = False
    assert harness._rcm_requires_proof({}) is False
    _, _, complete = harness._retrieval_status({}, {}, [], [])
    assert complete is True
    assert harness.status_phase_proof_required is True


def test_disabling_reasoning_completion_disables_completion_side_effects():
    harness = _Harness()
    harness.enable_reasoning_completion = False
    assert harness._rcm_requires_proof({}) is False
    assert harness._rcm_reasoning_ids({}) == set()
    assert harness._rcm_source_neighbors({}, [], set(), 2) == []
    assert harness._rcm_strengthen_prompt({}) is False
    _, _, complete = harness._retrieval_status({}, {}, [], [])
    assert complete is False
    assert harness.status_phase_proof_required is False


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
