"""Phase-4 regressions for candidate viability and unified ProofContext."""

from methods.smart_mem0.agent import SmartMem0Agent
from methods.smart_mem0.read_answerability_contract import ReadAnswerabilityContractMixin
from methods.smart_mem0.read_proof_context import UnifiedProofContextMixin
from methods.smart_mem0.read_proof_context_owner_contract import (
    ReadProofContextOwnerContractMixin,
)


def test_zero_viable_candidate_requires_recovery():
    action = ReadAnswerabilityContractMixin._answerability_action(
        {"r1": "FOUND"},
        {"CAUSES:r1:r2": "UNPROVEN"},
        viable_candidate_counts={"r1": 0},
    )
    assert action == "RECOVER"


def test_viable_evidence_synthesizes_even_without_structural_relation_certificate():
    action = ReadAnswerabilityContractMixin._answerability_action(
        {"r1": "FOUND", "r2": "FOUND"},
        {"CAUSES:r1:r2": "UNPROVEN"},
        viable_candidate_counts={"r1": 2, "r2": 1},
    )
    assert action == "SYNTHESIZE"


def test_missing_requirement_still_requires_recovery():
    action = ReadAnswerabilityContractMixin._answerability_action(
        {"r1": "FOUND", "r2": "EMPTY"},
        {},
        viable_candidate_counts={"r1": 1, "r2": 0},
    )
    assert action == "RECOVER"


def test_precomputed_strict_certificate_is_terminal():
    action = ReadAnswerabilityContractMixin._answerability_action(
        {"r1": "FOUND"},
        {},
        viable_candidate_counts={"r1": 1},
        terminal=True,
    )
    assert action == "TERMINAL"


def test_smartmem0_has_one_direct_proof_context_owner():
    direct_bases = set(SmartMem0Agent.__bases__)
    assert UnifiedProofContextMixin in direct_bases
    assert ReadProofContextOwnerContractMixin not in direct_bases
    assert getattr(
        UnifiedProofContextMixin, "PROOF_CONTEXT_VERSION", ""
    ) == "proof-context-v2"


def test_unified_proof_context_retains_owner_compatibility_internally():
    assert issubclass(
        UnifiedProofContextMixin, ReadProofContextOwnerContractMixin
    )
