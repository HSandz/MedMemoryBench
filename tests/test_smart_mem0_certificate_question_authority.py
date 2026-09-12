"""Regression tests for question-owned certificate authority."""

from methods.smart_mem0.read_evidence_certificate import EvidenceCertificateMixin


class CertificateHarness(EvidenceCertificateMixin):
    subject_aliases = {"patient": "primary_user"}

    def __init__(self, memories):
        self._memories = memories
        self._evidence = [{"id": "ev1", "text": "linked"}]
        self._belief_status = {}
        self._state_heads = {}

    @staticmethod
    def _date_for(memory, axis):
        return memory.get(axis)

    @staticmethod
    def _parse_date(value):
        return str(value or "")

    @staticmethod
    def _date_matches(value, target):
        return value == target

    @staticmethod
    def _is_state_head(memory):
        return bool(memory.get("head"))

    @staticmethod
    def _normalised_value(memory):
        return str(memory.get("value") or "").casefold()


def memory(claim, *, anchor="Cefuroxime", value="cefuroxime", **extra):
    return {
        "id": "m1",
        "claim": claim,
        "value": value,
        "object_anchor": anchor,
        "subject_id": "primary_user",
        "subject": "doctor",
        "evidence_ids": ["ev1"],
        "assertion_mode": "DIRECT",
        "stance": "AFFIRM",
        **extra,
    }


def advisory(*, focus=None, hypothesis="Cefuroxime"):
    return {
        "projection_hint": "ENTITY",
        "answer_hypothesis": hypothesis,
        "focus_spans": list(focus or []),
        "semantic_hints": [],
        "selector_hint": {},
        "selector_status": "NOT_REQUESTED",
        "relation_hints": ["VERIFY_SOURCE"],
    }


def test_advisor_focus_span_cannot_authorize_wrong_predicate():
    item = memory("Cefuroxime was prescribed")
    agent = CertificateHarness([item])
    result = agent._evidence_certificate(
        "Was Cefuroxime contraindicated?",
        [dict(item)],
        advisory(focus=["was"]),
    )
    assert result["status"] == "INSUFFICIENT"
    assert not result["terminal"]["closed"]
    assert result["binding"]["m1"]["advisor_fields_are_proof"] is False


def test_direct_stored_predicate_can_terminal_even_with_unhelpful_focus_span():
    item = memory("The patient was instructed to avoid Cefuroxime")
    agent = CertificateHarness([item])
    result = agent._evidence_certificate(
        "Which drug was I instructed to avoid?",
        [dict(item)],
        advisory(focus=["drug"]),
    )
    assert result["status"] == "SUPPORTED_UNIQUE"
    assert result["terminal"]["closed"]
    assert result["answer"] == "Cefuroxime"


def test_hypothesis_and_entity_name_alone_never_establish_question_predicate():
    item = memory("Allergy to Cefuroxime")
    agent = CertificateHarness([item])
    result = agent._evidence_certificate(
        "Which drug was I instructed to avoid?",
        [dict(item)],
        advisory(focus=["Cefuroxime"], hypothesis="Cefuroxime"),
    )
    assert result["hypothesis_status"] == "UNSUPPORTED"
    assert not result["terminal"]["closed"]


def test_vocative_speaker_is_not_a_durable_owner_constraint():
    item = memory("Dose is 500 mg", anchor="dose", value="500 mg", state_key="dose")
    agent = CertificateHarness([item])
    atomic = {
        "projection_hint": "VALUE",
        "answer_hypothesis": "500 mg",
        "focus_spans": [],
        "semantic_hints": [],
        "selector_hint": {},
        "selector_status": "NOT_REQUESTED",
        "relation_hints": [],
    }
    result = agent._evidence_certificate(
        "Doctor, what dose is recorded for the patient?",
        [dict(item)],
        atomic,
    )
    assert "m1" not in result["rejected"]
    assert result["terminal"]["closed"]
