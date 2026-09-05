"""Regression tests for SmartMem0 P0 target binding and seed preservation."""

from copy import deepcopy

from methods.smart_mem0.read_execution_contract import ReadExecutionContractMixin
from methods.smart_mem0.read_p0_contract import ReadP0ContractMixin


class _TargetHarness(ReadP0ContractMixin, ReadExecutionContractMixin):
    def __init__(self, memories=None):
        self._memories = list(memories or [])
        self._belief_status = {
            memory["id"]: memory.get("_status", "active")
            for memory in self._memories
        }
        self._state_heads = {}
        self._last_option_probe_coverage = {}

    @staticmethod
    def _memory_value(memory):
        return memory.get("value") or memory.get("verbatim_value") or ""

    @staticmethod
    def _snapshot(value):
        return deepcopy(value)

    @staticmethod
    def _rc_text(value):
        return " ".join(str(value or "").casefold().replace("_", " ").split())

    @classmethod
    def _rc_terms(cls, value):
        return [token for token in cls._rc_text(value).replace("-", " ").split() if token]

    @classmethod
    def _rc_content_terms(cls, value):
        noise = {"what", "which", "the", "is", "was", "latest", "result", "mentioned"}
        return [term for term in cls._rc_terms(value) if term not in noise]

    @classmethod
    def _rc_token_sequence_present(cls, needle, haystack):
        wanted = cls._rc_terms(needle)
        available = cls._rc_terms(haystack)
        width = len(wanted)
        return bool(
            wanted
            and any(
                available[index : index + width] == wanted
                for index in range(len(available) - width + 1)
            )
        )

    @classmethod
    def _rc_owner(cls, value):
        return "_".join(cls._rc_terms(value))

    def _rc_owner_match(self, slot, memory):
        wanted = self._rc_owner(slot.get("subject_id") or slot.get("subject") or "")
        actual = self._rc_owner(memory.get("subject_id") or memory.get("subject") or "")
        return not wanted or wanted == actual

    def _rc_memory_concept_keys(self, memory):
        output = []
        for value in (
            memory.get("scope"),
            memory.get("state_key"),
            memory.get("object_anchor"),
            *(memory.get("entities") or []),
            *(memory.get("scope_entities") or []),
        ):
            terms = self._rc_content_terms(value)
            if terms:
                key = " ".join(terms)
                if key not in output:
                    output.append(key)
        return output

    def _hybrid_search(self, _query, top_k, candidate_ids=None):
        allowed = set(candidate_ids or [])
        return [
            deepcopy(memory)
            for memory in self._memories
            if not allowed or memory["id"] in allowed
        ][:top_k]

    @staticmethod
    def _date_for(memory, axis="event_time"):
        return str(memory.get(axis) or "")

    @staticmethod
    def _is_state_head(memory):
        return bool(memory.get("is_state_head"))


def _memory(memory_id, value, claim, **extra):
    return {
        "id": memory_id,
        "subject": "primary_user",
        "kind": "FACT",
        "claim": claim,
        "value": value,
        "verbatim_value": "",
        "assertion_mode": "DIRECT",
        "entities": extra.pop("entities", []),
        "scope_entities": [],
        "scope": extra.pop("scope", ""),
        "state_key": extra.pop("state_key", ""),
        "object_anchor": extra.pop("object_anchor", ""),
        "event_time": extra.pop("event_time", ""),
        "document_time": extra.pop("document_time", ""),
        **extra,
    }


def _slot(slot_type="DIRECT", target="HbA1c", **extra):
    return {
        "id": "r1",
        "type": slot_type,
        "evidence_role": "REQUIREMENT",
        "subject": "primary_user",
        "subject_id": "primary_user",
        "target_surface": target,
        "retrieval_hint": target,
        "resolved_keys": extra.pop("resolved_keys", []),
        "required_fields": extra.pop("required_fields", []),
        **extra,
    }


def test_direct_requirement_rejects_same_owner_wrong_target():
    wrong = _memory(
        "m_wrong",
        "no meter",
        "The patient does not have a home blood glucose meter.",
        object_anchor="blood_glucose_meter",
    )
    right = _memory(
        "m_right",
        "8.1%",
        "The latest HbA1c result was 8.1%.",
        state_key="hba1c",
        entities=["HbA1c"],
    )
    harness = _TargetHarness([wrong, right])
    supported = harness._operation_slot_support(_slot(), [wrong, right], [])
    assert [memory["id"] for memory in supported] == ["m_right"]


def test_temporal_requirement_matches_target_before_date():
    wrong = _memory(
        "m_wrong",
        "120/80",
        "Blood pressure was 120/80.",
        event_time="2025-06-01",
        state_key="blood_pressure",
    )
    right = _memory(
        "m_right",
        "8.1%",
        "HbA1c dropped to 8.1%.",
        event_time="2025-05-01",
        state_key="hba1c",
        entities=["HbA1c"],
    )
    harness = _TargetHarness([wrong, right])
    slot = _slot(
        "TEMPORAL",
        time_axis="event_time",
        temporal_relation="LATEST",
        required_fields=["event_time"],
    )
    supported = harness._operation_slot_support(slot, [wrong, right], [])
    assert [memory["id"] for memory in supported] == ["m_right"]
    assert harness._slot_covered(slot, ["m_wrong"], [wrong], []) is False
    assert harness._slot_covered(slot, ["m_right"], [right], []) is True


def test_generic_requirement_quota_is_temporarily_one():
    first = _memory("m1", "8.1%", "HbA1c was 8.1%.", state_key="hba1c")
    second = _memory("m2", "9.2%", "HbA1c was 9.2%.", state_key="hba1c")
    harness = _TargetHarness([first, second])
    supported = harness._operation_slot_support(_slot(), [first, second], [])
    assert len(supported) == 1


def test_target_derived_resolved_key_can_authorize_same_concept_family():
    diabetes = _memory(
        "m_diabetes",
        "Diabetes",
        "The patient has diabetes mellitus.",
        state_key="diabetes",
        entities=["diabetes mellitus"],
    )
    harness = _TargetHarness([diabetes])
    slot = _slot(
        target="chronic metabolic disease",
        resolved_keys=["diabetes"],
    )
    assert harness._slot_contract_match(slot, diabetes, False) is True


class _RunBase:
    def _run_query_retrieval(
        self,
        question,
        initial_seeds,
        frame,
        fast_supports,
        gate,
        planning_seeds=None,
        planning_context=None,
    ):
        del question, initial_seeds, frame, fast_supports, gate, planning_seeds, planning_context
        return {
            "fast_supports": None,
            "plan": {
                "required_slots": [
                    {
                        "id": "r1",
                        "type": "DIRECT",
                        "evidence_role": "REQUIREMENT",
                        "subject": "primary_user",
                        "subject_id": "primary_user",
                        "target_surface": "cefuroxime",
                        "resolved_keys": ["cefuroxime"],
                        "required_fields": [],
                    }
                ]
            },
            "replan": None,
            "slot_support": {"r1": []},
            "requirement_status": {"r1": "EMPTY"},
            "retrieval_complete": False,
            "planning_seeds": [],
        }


class _SeedHarness(ReadP0ContractMixin, _RunBase):
    def __init__(self):
        self._belief_status = {}

    @staticmethod
    def _memory_value(memory):
        return memory.get("value") or ""

    @staticmethod
    def _snapshot(value):
        return deepcopy(value)

    @staticmethod
    def _slot_contract_match(slot, memory, strict_targets=None):
        del strict_targets
        return (
            slot.get("subject") == memory.get("subject")
            and slot.get("target_surface", "").casefold()
            in (memory.get("claim", "") + " " + memory.get("value", "")).casefold()
        )


def test_reserved_seed_is_context_only_not_retrieval_proof():
    seed = _memory(
        "m_cefuroxime",
        "cefuroxime",
        "Known allergic reaction to cefuroxime.",
    )
    harness = _SeedHarness()
    run = harness._run_query_retrieval(
        "Which antibiotic caused the allergic reaction?",
        [seed],
        object(),
        None,
        {},
    )
    assert run["slot_support"]["r1"] == ["m_cefuroxime"]
    assert run["reserved_seed_context"] == {"r1": "m_cefuroxime"}
    assert run["planning_seeds"][0]["id"] == "m_cefuroxime"
    assert run["requirement_status"]["r1"] == "EMPTY"
    assert run["retrieval_complete"] is False
