from copy import deepcopy
from types import SimpleNamespace

import numpy as np

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
            "evidence_role": "OPTION_CONTEXT"
            if ir.get("visible_options")
            else "REQUIREMENT",
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


class _StatusBase:
    @staticmethod
    def _slot_covered(slot, support_ids, selected, relations):
        del slot, relations
        selected_ids = {memory.get("id") for memory in selected}
        return "proof" in set(support_ids) and "proof" in selected_ids

    @staticmethod
    def _relation_status_map(plan, slot_support, selected, relations):
        del plan, slot_support, selected, relations
        return {}


class _StatusHarness(ReadRequirementGraphRuntimeMixin, _StatusBase):
    pass


class _FakeBM25:
    @staticmethod
    def get_scores(tokens):
        del tokens
        return np.asarray([10.0, 0.0, 1.0], dtype=float)


class _FakeEmbedder:
    @staticmethod
    def encode(texts, show_progress_bar=False):
        del texts, show_progress_bar
        return np.asarray([[0.0, 1.0]], dtype=float)


class _FusionBase:
    RRF_K = 60

    def __init__(self):
        self._memories = [
            {"id": "lexical", "claim": "lexical"},
            {"id": "dense", "claim": "dense"},
            {"id": "semantic", "claim": "semantic"},
        ]
        self._belief_status = {}
        self._bm25 = _FakeBM25()
        self._embedder = _FakeEmbedder()
        self._embedding_matrix = np.asarray(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.7, 0.7],
            ],
            dtype=float,
        )

    @staticmethod
    def _fusion_multiview(operation, outputs, seeds, frame):
        del operation, outputs, seeds, frame
        return [
            {
                "id": "semantic",
                "claim": "semantic",
                "_alignment_recall_sources": ["semantic_ir"],
            }
        ], [], []

    @staticmethod
    def _fusion_views(views):
        del views
        return {"question": {"kind": "question", "query": "raw question"}}

    @staticmethod
    def _memory_satisfies_frame(memory, frame, include_entities=False):
        del memory, frame, include_entities
        return True

    @staticmethod
    def _query_visible_memory(memory, include_history=False):
        del memory, include_history
        return True

    @staticmethod
    def _snapshot(memory):
        return deepcopy(memory)

    @staticmethod
    def _tokenize(text):
        return str(text or "").lower().split()

    @staticmethod
    def _memory_text(memory):
        return str(memory.get("claim") or "")

    @staticmethod
    def _refresh_index():
        return None


class _FusionHarness(ReadRequirementGraphRuntimeMixin, _FusionBase):
    pass


class _CompactBase:
    def __init__(self):
        self._active_query_links = []
        self._last_candidate_propositions = {}
        self._last_requirement_proof_support = {
            "r1": ["proof1"],
            "r2": ["proof2"],
        }
        self._rg_runtime_requirement_status = {
            "r1": "FOUND",
            "r2": "FOUND",
        }
        self._rg_runtime_relation_status = {}
        self._last_query_memory_alignment = {}

    @staticmethod
    def _role_aware_support_ids(slots, slot_support, candidate_order, limit):
        del slots, slot_support
        return list(candidate_order[:limit])

    @staticmethod
    def _semantic_context_slot(slot):
        return str(slot.get("evidence_role") or "").upper() == "REQUIREMENT"


class _CompactHarness(ReadRequirementGraphRuntimeMixin, _CompactBase):
    pass


def test_active_mro_puts_runtime_and_requirement_graph_before_old_alignment():
    mro = SmartMem0Agent.mro()
    assert mro.index(ReadRequirementGraphRuntimeMixin) < mro.index(
        ReadRequirementGraphMixin
    )
    assert mro.index(ReadRequirementGraphMixin) < mro.index(
        ReadQueryMemoryAlignmentMixin
    )


def test_runtime_derives_canonical_addresses_when_llm_anchor_is_missing():
    harness = _SlotHarness()
    slot = harness._requirement_slot(
        {"id": "r1", "need": "participant variable", "material_anchors": []},
        {},
        "DIRECT",
    )
    assert slot["material_anchors"] == ["resolved canonical"]
    assert slot["canonical_anchor_source"]["runtime_resolved"] == [
        "resolved canonical"
    ]
    assert (
        slot["canonical_anchor_source"]["authority"]
        == "retrieval_and_materiality_only_not_proof"
    )


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
    assert (
        harness._rg_selector_compatible(slot, {"id": "old", "head": False})
        is False
    )
    assert (
        harness._rg_selector_compatible(slot, {"id": "new", "head": True})
        is True
    )


def test_material_candidate_cannot_promote_requirement_to_found_or_stop():
    harness = _StatusHarness()
    plan = {"required_slots": [{"id": "r1"}]}
    status, relations, complete = harness._retrieval_status(
        plan,
        {"r1": ["material"]},
        [{"id": "material"}],
        [],
    )
    assert status == {"r1": "EMPTY"}
    assert relations == {}
    assert complete is False
    assert (
        harness._last_requirement_graph_stop_guard[
            "materiality_promotes_found"
        ]
        is False
    )


def test_structurally_proven_support_can_close_requirement():
    harness = _StatusHarness()
    plan = {"required_slots": [{"id": "r1"}]}
    status, _, complete = harness._retrieval_status(
        plan,
        {"r1": ["proof"]},
        [{"id": "proof"}],
        [],
    )
    assert status == {"r1": "FOUND"}
    assert complete is True


def test_raw_bm25_and_dense_heads_survive_before_fused_truncation():
    harness = _FusionHarness()
    rows, _, _ = harness._fusion_multiview(
        {
            "top_k": 3,
            "retrieval_views": [
                {"kind": "question", "query": "raw question"}
            ],
        },
        [],
        [],
        SimpleNamespace(hard_entities=()),
    )
    assert [memory["id"] for memory in rows] == [
        "semantic",
        "lexical",
        "dense",
    ]
    assert "raw_lexical" in rows[1]["_alignment_recall_sources"]
    assert "raw_dense" in rows[2]["_alignment_recall_sources"]


def test_proof_complete_context_stops_before_answer_context_cap():
    harness = _CompactHarness()
    slots = [
        {"id": "r1", "evidence_role": "REQUIREMENT"},
        {"id": "r2", "evidence_role": "REQUIREMENT"},
    ]
    selected = harness._role_aware_support_ids(
        slots,
        {"r1": ["proof1"], "r2": ["proof2"]},
        ["proof1", "extra1", "proof2", "extra2"],
        4,
    )
    assert selected == ["proof1", "proof2"]
    assert (
        harness._last_query_memory_alignment["selection_semantics"]
        == "smallest_proof_complete_evidence_set"
    )
    assert (
        harness._last_query_memory_alignment["sufficient_context_compacted"]
        is True
    )


def test_structural_relation_keeps_extra_context_instead_of_over_compacting():
    harness = _CompactHarness()
    harness._active_query_links = [
        {"type": "COMPARE", "from": "r1", "to": "r2"}
    ]
    slots = [
        {"id": "r1", "evidence_role": "REQUIREMENT"},
        {"id": "r2", "evidence_role": "REQUIREMENT"},
    ]
    selected = harness._role_aware_support_ids(
        slots,
        {"r1": ["proof1"], "r2": ["proof2"]},
        ["proof1", "extra1", "proof2", "extra2"],
        4,
    )
    assert selected == ["proof1", "extra1", "proof2", "extra2"]
