from methods.smart_mem0.agent import SmartMem0Agent
from methods.smart_mem0.read_requirement_identity_runtime import (
    ReadRequirementIdentityRuntimeMixin,
)
from methods.smart_mem0.read_requirement_resolution_runtime import (
    ReadRequirementResolutionRuntimeMixin,
)


class _SemanticBase:
    def __init__(self, memories=None):
        self._memories = list(memories or [])

    @staticmethod
    def _rc_text(value):
        return " ".join(str(value or "").replace("_", " ").lower().split())

    @classmethod
    def _rc_terms(cls, value):
        return [term for term in cls._rc_text(value).split() if term]

    @classmethod
    def _rq_surface_similarity(cls, target, surface):
        left = "".join(ch for ch in cls._rc_text(target) if ch.isalnum())
        right = "".join(ch for ch in cls._rc_text(surface) if ch.isalnum())
        if not left or not right:
            return 0.0
        if left == right or left in right:
            return 1.0
        width = 3
        left_grams = (
            {left}
            if len(left) <= width
            else {left[index : index + width] for index in range(len(left) - width + 1)}
        )
        right_grams = (
            {right}
            if len(right) <= width
            else {right[index : index + width] for index in range(len(right) - width + 1)}
        )
        char_coverage = len(left_grams & right_grams) / len(left_grams)
        left_terms = list(dict.fromkeys(cls._rc_terms(target)))
        right_terms = set(cls._rc_terms(surface))
        token_coverage = sum(term in right_terms for term in left_terms) / len(left_terms)
        return max(char_coverage, token_coverage)

    @staticmethod
    def _rc_memory_target_text(memory):
        return " ".join(
            str(memory.get(field) or "")
            for field in (
                "claim",
                "value",
                "verbatim_value",
                "scope",
                "state_key",
                "object_anchor",
            )
        )

    @staticmethod
    def _rg_requested_projection(slot):
        return str(
            slot.get("premise_projection")
            or slot.get("requested_projection")
            or "TEXT"
        )

    @staticmethod
    def _rg_material_topic_surfaces(memory):
        values = [
            memory.get("scope"),
            memory.get("state_key"),
            memory.get("object_anchor"),
            memory.get("evidence_family"),
            *(memory.get("entities") or []),
            *(memory.get("scope_entities") or []),
        ]
        blocked = {"patient", "primary user", "primary_user"}
        output = []
        for value in values:
            text = " ".join(str(value or "").replace("_", " ").lower().split())
            if text and text not in blocked and text not in output:
                output.append(text)
        return output


class _IdentityHarness(ReadRequirementIdentityRuntimeMixin, _SemanticBase):
    pass


class _UnsupportedCurrentBase(_SemanticBase):
    def _requirement_slot(self, requirement, ir, compiled_mode):
        del requirement, ir, compiled_mode
        return {
            "id": "r1",
            "type": "CURRENT_STATE",
            "evidence_role": "REQUIREMENT",
            "selector": {"relation": "CURRENT"},
            "semantic_relation_types": ["CURRENT"],
            "material_anchors": ["past_medical_history", "metabolic_disease"],
            "resolved_keys": [],
        }


class _UnsupportedCurrentHarness(
    ReadRequirementIdentityRuntimeMixin, _UnsupportedCurrentBase
):
    pass


class _SupportedCurrentBase(_SemanticBase):
    def _requirement_slot(self, requirement, ir, compiled_mode):
        del requirement, ir, compiled_mode
        return {
            "id": "r1",
            "type": "CURRENT_STATE",
            "evidence_role": "REQUIREMENT",
            "selector": {"relation": "CURRENT"},
            "semantic_relation_types": ["CURRENT"],
            "material_anchors": ["heart_rate"],
            "resolved_keys": [],
        }


class _SupportedCurrentHarness(ReadRequirementIdentityRuntimeMixin, _SupportedCurrentBase):
    pass


def test_identity_runtime_precedes_resolution_runtime_in_active_mro():
    mro = SmartMem0Agent.mro()
    assert mro.index(ReadRequirementIdentityRuntimeMixin) < mro.index(
        ReadRequirementResolutionRuntimeMixin
    )


def test_generic_participant_surface_is_not_question_identity_proof():
    harness = _IdentityHarness()
    identity, meta = harness._rg_question_owned_identity(
        {
            "requested_projection": "TEXT",
            "proof_need": "the symptom experienced by the patient starting from the specified date",
            "proof_question_span": "what symptom did the patient begin to experience",
        },
        {
            "id": "m1",
            "claim": "The patient was instructed to book a hospital appointment.",
            "scope": "general",
            "entities": ["patient"],
        },
    )
    assert identity is False
    assert meta["generic_substring_overlap_is_proof"] is False
    assert meta["source"] == "no_question_owned_identity"


def test_discriminative_topic_surface_can_establish_question_identity():
    harness = _IdentityHarness()
    identity, meta = harness._rg_question_owned_identity(
        {
            "requested_projection": "VALUE",
            "proof_need": "HbA1c",
            "proof_question_span": "HbA1c",
        },
        {
            "id": "m1",
            "claim": "HbA1c increased to 8.8%.",
            "state_key": "hba1c",
            "value": "8.8%",
        },
    )
    assert identity is True
    assert meta["concept_coverage"] >= harness.PROOF_CONCEPT_COVERAGE_MIN


def test_current_without_durable_state_address_is_downgraded_to_direct_retrieval():
    harness = _UnsupportedCurrentHarness(
        [
            {
                "id": "m1",
                "kind": "STATE",
                "state_key": "metabolic_stress",
                "object_anchor": "metabolic_stress",
            }
        ]
    )
    slot = harness._requirement_slot({"id": "r1"}, {}, "STATE")
    assert slot["selector"] == {}
    assert slot["type"] == "DIRECT"
    assert slot["semantic_relation_types"] == []
    assert slot["current_selector_guard"]["status"] == "DOWNGRADED_NO_DURABLE_STATE_ADDRESS"


def test_current_with_matching_durable_state_address_is_preserved():
    harness = _SupportedCurrentHarness(
        [
            {
                "id": "m1",
                "kind": "STATE",
                "state_key": "morning_heart_rate",
                "object_anchor": "heart_rate",
            }
        ]
    )
    slot = harness._requirement_slot({"id": "r1"}, {}, "STATE")
    assert slot["selector"] == {"relation": "CURRENT"}
    assert slot["type"] == "CURRENT_STATE"
    assert slot["current_selector_guard"]["status"] == "PASS"
    assert slot["current_selector_guard"]["matched_surface"] in {
        "morning_heart_rate",
        "heart_rate",
    }
