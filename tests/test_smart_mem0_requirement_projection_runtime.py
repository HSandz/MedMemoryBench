from methods.smart_mem0.agent import SmartMem0Agent
from methods.smart_mem0.read_requirement_graph_runtime import ReadRequirementGraphRuntimeMixin
from methods.smart_mem0.read_requirement_projection_runtime import (
    ReadRequirementProjectionRuntimeMixin,
)


class _ProofBase:
    @staticmethod
    def _semantic_context_slot(slot):
        return str(slot.get("evidence_role") or "").upper() == "REQUIREMENT"

    @staticmethod
    def _rg_selector_compatible(slot, memory):
        del slot, memory
        return True

    @staticmethod
    def _date_for(memory, axis):
        return memory.get(axis, "")

    @staticmethod
    def _rc_text(value):
        return " ".join(str(value or "").replace("_", " ").lower().split())

    @classmethod
    def _rc_terms(cls, value):
        return [term for term in cls._rc_text(value).split() if term]

    @classmethod
    def _rq_surface_similarity(cls, target, surface):
        target_terms = cls._rc_terms(target)
        surface_terms = set(cls._rc_terms(surface))
        if not target_terms:
            return 0.0
        return sum(term in surface_terms for term in target_terms) / len(target_terms)

    @staticmethod
    def _rc_memory_target_text(memory):
        return " ".join(
            [
                str(memory.get("claim") or ""),
                str(memory.get("value") or ""),
                str(memory.get("verbatim_value") or ""),
                str(memory.get("scope") or ""),
                str(memory.get("state_key") or ""),
                str(memory.get("object_anchor") or ""),
                " ".join(memory.get("entities") or []),
                " ".join(memory.get("scope_entities") or []),
            ]
        )

    @staticmethod
    def _rq_memory_concept_surfaces(memory):
        return [
            str(value).replace("_", " ")
            for value in [
                memory.get("scope"),
                memory.get("state_key"),
                memory.get("object_anchor"),
                *(memory.get("entities") or []),
                *(memory.get("scope_entities") or []),
            ]
            if value
        ]


class _ProofHarness(ReadRequirementProjectionRuntimeMixin, _ProofBase):
    pass


class _ReservationBase(_ProofBase):
    def __init__(self):
        self._last_requirement_context_candidates = {
            "r1": ["generic", "specific", "noise"]
        }
        self._last_requirement_proof_support = {"r1": []}
        self._last_requirement_status = {"r1": "EMPTY"}
        self._rg_runtime_requirement_status = {"r1": "EMPTY"}
        self._rg_runtime_relation_status = {}
        self._last_query_memory_alignment = {}
        self.memories = {
            "generic": {
                "id": "generic",
                "claim": "Patient has severe cephalosporin allergy",
                "scope": "allergy",
                "entities": ["cephalosporins"],
                "value": "cephalosporin allergy",
            },
            "specific": {
                "id": "specific",
                "claim": "Patient has known allergy to cefuroxime",
                "scope": "allergy",
                "entities": ["cefuroxime"],
                "value": "cefuroxime allergy",
            },
            "noise": {
                "id": "noise",
                "claim": "Unrelated monitoring instruction",
                "entities": ["monitoring"],
                "value": "monitoring",
            },
        }

    def _alignment_memory(self, memory_id):
        return self.memories.get(memory_id, {})

    @staticmethod
    def _role_aware_support_ids(slots, slot_support, candidate_order, limit):
        del slots, slot_support
        return list(candidate_order[:limit])


class _ReservationHarness(ReadRequirementProjectionRuntimeMixin, _ReservationBase):
    pass


def _entity_slot():
    return {
        "id": "r1",
        "evidence_role": "REQUIREMENT",
        "requested_projection": "ENTITY",
        "proof_need": "name of the antibiotic causing systemic allergy",
        "proof_question_span": "antibiotic",
        "proof_spec": {"status": "UNSPECIFIED"},
    }


def test_projection_runtime_precedes_old_runtime_in_active_mro():
    mro = SmartMem0Agent.mro()
    assert mro.index(ReadRequirementProjectionRuntimeMixin) < mro.index(
        ReadRequirementGraphRuntimeMixin
    )


def test_generic_entity_topic_match_is_not_specific_entity_proof():
    harness = _ProofHarness()
    memory = {
        "id": "generic",
        "claim": "Patient has severe cephalosporin allergy",
        "scope": "allergy",
        "entities": ["cephalosporins"],
        "value": "cephalosporin allergy",
    }
    assert harness._requirement_target_proof(_entity_slot(), memory) is False
    check = harness._last_requirement_graph_proof_checks["r1"]["generic"]
    assert check["projection"]["capable"] is True
    assert check["identity"]["source"] == "entity_requires_specific_grounding"


def test_retrieval_only_keys_cannot_become_proof_authority():
    harness = _ProofHarness()
    slot = {
        "id": "r1",
        "evidence_role": "REQUIREMENT",
        "requested_projection": "VALUE",
        "proof_need": "unmatched participant variable",
        "proof_question_span": "",
        "material_anchors": ["hba1c"],
        "resolved_keys": ["hba1c"],
        "search_aliases": ["hba1c"],
        "proof_spec": {"status": "UNSPECIFIED"},
    }
    memory = {
        "id": "hba1c",
        "claim": "HbA1c increased to 8.8%",
        "state_key": "hba1c",
        "value": "8.8%",
    }
    assert harness._requirement_target_proof(slot, memory) is False


def test_projection_aware_value_proof_accepts_question_owned_canonical_identity():
    harness = _ProofHarness()
    slot = {
        "id": "r1",
        "evidence_role": "REQUIREMENT",
        "requested_projection": "VALUE",
        "proof_need": "current HbA1c measurement",
        "proof_question_span": "HbA1c",
        "proof_spec": {"status": "UNSPECIFIED"},
    }
    memory = {
        "id": "hba1c",
        "claim": "HbA1c increased to 8.8%",
        "state_key": "hba1c",
        "object_anchor": "hba1c",
        "value": "8.8%",
        "verbatim_value": "8.8%",
    }
    assert harness._requirement_target_proof(slot, memory) is True


def test_projection_aware_date_proof_rejects_topic_without_date_field():
    harness = _ProofHarness()
    slot = {
        "id": "r1",
        "evidence_role": "REQUIREMENT",
        "requested_projection": "DATE",
        "proof_need": "date of weight change",
        "proof_question_span": "weight change",
        "proof_spec": {"status": "UNSPECIFIED"},
    }
    memory = {
        "id": "weight",
        "claim": "Weight changed recently",
        "state_key": "weight_change",
        "value": "lost weight",
    }
    assert harness._requirement_target_proof(slot, memory) is False


def test_unresolved_requirement_reserves_two_distinct_answer_surfaces():
    harness = _ReservationHarness()
    selected = harness._role_aware_support_ids(
        [_entity_slot()],
        {"r1": []},
        ["generic", "specific", "noise"],
        2,
    )
    assert selected == ["generic", "specific"]
    assert harness._last_query_memory_alignment["answerability_reservations"]["r1"] == [
        "generic",
        "specific",
    ]
