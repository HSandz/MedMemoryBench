from methods.smart_mem0.read_requirement_graph import ReadRequirementGraphMixin


class _RequirementGraphBase:
    ANSWER_CONTEXT_HARD_CAP = 8
    CANDIDATE_WORLD_HARD_CAP = 16

    def __init__(self):
        self._memories = []
        self._relations = []
        self._belief_status = {}
        self._last_requirement_binding_views = {}
        self._last_requirement_context_candidates = {}
        self._last_requirement_proof_support = {}
        self._last_proposition_probe_coverage = {}
        self._active_requirement_slots = {}
        self._active_query_links = []
        self._alignment_candidate_meta = {}

    @staticmethod
    def _rc_text(value):
        return " ".join(str(value or "").replace("_", " ").lower().split())

    @classmethod
    def _rq_surface_similarity(cls, left, right):
        left_text, right_text = cls._rc_text(left), cls._rc_text(right)
        if not left_text or not right_text:
            return 0.0
        if left_text == right_text or left_text in right_text or right_text in left_text:
            return 1.0
        left_terms, right_terms = set(left_text.split()), set(right_text.split())
        return len(left_terms & right_terms) / max(1, len(left_terms))

    @classmethod
    def _rc_question_span(cls, value, question):
        value = " ".join(str(value or "").split())
        return value if value and cls._rc_text(value) in cls._rc_text(question) else ""

    @staticmethod
    def _question_stem(question):
        return str(question or "").strip()

    @staticmethod
    def _memory_value(memory):
        return memory.get("value") or memory.get("claim") or ""

    @staticmethod
    def _date_for(memory, axis):
        return memory.get(axis) or ""

    @staticmethod
    def _date_matches(value, anchor):
        return str(value or "") == str(anchor or "") or str(value or "").startswith(str(anchor or ""))

    @staticmethod
    def _query_visible_memory(memory, include_history=False):
        del memory, include_history
        return True

    @staticmethod
    def _rc_owner_match(slot, memory):
        del slot, memory
        return True

    @staticmethod
    def _context_resolved_key_match(slot, memory):
        del slot, memory
        return False

    @staticmethod
    def _rc_memory_target_text(memory):
        return " ".join(
            str(memory.get(key) or "")
            for key in (
                "claim",
                "value",
                "verbatim_value",
                "scope",
                "state_key",
                "object_anchor",
                "evidence_family",
            )
        )

    def _alignment_memory(self, memory_id):
        return self._alignment_candidate_meta.get(memory_id) or next(
            (memory for memory in self._memories if memory.get("id") == memory_id),
            {},
        )

    @classmethod
    def _alignment_signature(cls, memory):
        return (
            cls._rc_text(memory.get("subject_id") or memory.get("subject")),
            cls._rc_text(memory.get("state_key") or memory.get("evidence_family") or memory.get("scope")),
            cls._rc_text(memory.get("object_anchor")),
            cls._rc_text(memory.get("value") or memory.get("verbatim_value")),
            cls._rc_text(memory.get("stance")),
            str(memory.get("event_time") or ""),
            str(memory.get("document_time") or ""),
        )


class _Harness(ReadRequirementGraphMixin, _RequirementGraphBase):
    pass


def test_sparse_selector_keeps_missing_semantics_absent():
    assert _Harness._rg_sparse_selector({}) == {}
    assert _Harness._rg_sparse_selector(
        {"axis": "", "relation": "", "anchor": "", "end": ""}
    ) == {}
    assert _Harness._rg_sparse_selector({"relation": "CURRENT"}) == {
        "relation": "CURRENT"
    }
    assert _Harness._rg_sparse_selector(
        {"relation": "EARLIEST", "axis": "event_time"}
    ) == {"relation": "EARLIEST", "axis": "event_time"}


def test_query_shape_does_not_treat_empty_selector_object_as_temporal():
    shape = _Harness._derive_query_shape(
        {
            "goal": {"projection": "TEXT", "directive": "resolve the answer"},
            "requirements": [
                {"id": "r1", "selector": {"axis": "", "relation": ""}}
            ],
            "relations": [],
        }
    )
    assert shape["has_temporal_selector"] is False


def test_current_selector_is_compiled_as_runtime_state_resolution_not_public_link():
    harness = _Harness()
    legacy, graph = harness._rg_to_legacy(
        {
            "goal": {"projection": "VALUE", "directive": "resolve current status"},
            "requirements": [
                {
                    "id": "r1",
                    "need": "medication status",
                    "question_span": "medication status",
                    "anchors": ["medication"],
                    "selector": {"relation": "CURRENT"},
                    "aliases": ["current treatment status"],
                }
            ],
        },
        "What is the medication status?",
    )
    assert legacy["requirements"][0]["selector"] == {}
    assert {"type": "CURRENT", "from": "r1", "to": ""} in legacy["bridges"]
    assert graph["requirements"]["r1"]["selector"] == {"relation": "CURRENT"}
    assert legacy["requirements"][0]["evidence_family"] == "medication status"
    assert "current treatment status" not in legacy["requirements"][0]["evidence_family"]


def test_broad_context_membership_alone_is_not_material_binding():
    harness = _Harness()
    memory = {
        "id": "topical",
        "claim": "metabolic stress is increased",
        "value": "increased",
        "scope": "clinical assessment",
        "evidence_family": "metabolic stress",
        "subject_id": "primary_user",
    }
    slot = {
        "id": "r1",
        "need": "chronic disease in medical history",
        "material_anchors": ["diagnosis", "medical history"],
        "resolved_keys": [],
        "selector": {},
    }
    harness._last_requirement_context_candidates = {"r1": ["topical"]}
    binding = harness._rg_material_binding(slot, memory)
    assert binding["level"] == "NONE"


def test_canonical_memory_address_can_create_strong_material_binding():
    harness = _Harness()
    memory = {
        "id": "diagnosis_memory",
        "claim": "A chronic condition is documented.",
        "value": "condition present",
        "scope": "diagnosis",
        "evidence_family": "medical history",
        "subject_id": "primary_user",
    }
    slot = {
        "id": "r1",
        "need": "chronic disease in medical history",
        "material_anchors": ["diagnosis", "medical history"],
        "resolved_keys": [],
        "selector": {},
    }
    binding = harness._rg_material_binding(slot, memory)
    assert binding["level"] == "STRONG"
    assert binding["anchor_exact_hits"] >= 1


def test_selection_preserves_unique_material_requirement_before_global_rank():
    harness = _Harness()
    harness._alignment_candidate_meta = {
        "common1": {
            "id": "common1",
            "claim": "topic alpha",
            "value": "a",
            "scope": "alpha",
        },
        "common2": {
            "id": "common2",
            "claim": "topic alpha repeated",
            "value": "a2",
            "scope": "alpha",
        },
        "unique": {
            "id": "unique",
            "claim": "topic beta",
            "value": "b",
            "scope": "beta",
        },
    }
    slots = [
        {"id": "r1", "need": "alpha", "material_anchors": ["alpha"], "selector": {}},
        {"id": "r2", "need": "beta", "material_anchors": ["beta"], "selector": {}},
    ]
    selected = harness._role_aware_support_ids(
        slots,
        {},
        ["common1", "common2", "unique"],
        2,
    )
    assert "unique" in selected
    assert len(selected) == 2
    assert harness._last_query_memory_alignment["requirement_material_state"] == {
        "r1": "STRONG",
        "r2": "STRONG",
    }


def test_candidate_option_probe_is_not_a_material_lane():
    harness = _Harness()
    harness._alignment_candidate_meta = {
        "shared": {
            "id": "shared",
            "claim": "shared participant constraint",
            "value": "constraint",
            "scope": "constraint",
        },
        "option_only": {
            "id": "option_only",
            "claim": "option topical memory",
            "value": "topic",
            "scope": "option topic",
        },
    }
    harness._last_proposition_probe_coverage = {"A": ["option_only"]}
    slots = [
        {
            "id": "r1",
            "need": "participant constraint",
            "material_anchors": ["constraint"],
            "selector": {},
        }
    ]
    selected = harness._role_aware_support_ids(
        slots,
        {},
        ["option_only", "shared"],
        1,
    )
    assert selected == ["shared"]
    assert harness._last_query_memory_alignment["candidate_probe_semantics"] == "retrieval_only_not_material_lane"


def test_stored_relation_expansion_requires_target_material_binding():
    harness = _Harness()
    source = {
        "id": "source",
        "claim": "source fact",
        "value": "x",
        "scope": "source_topic",
    }
    relevant_neighbor = {
        "id": "relevant",
        "claim": "target fact",
        "value": "y",
        "scope": "target_topic",
    }
    noisy_neighbor = {
        "id": "noise",
        "claim": "unrelated fact",
        "value": "z",
        "scope": "other_topic",
    }
    harness._memories = [source, relevant_neighbor, noisy_neighbor]
    harness._relations = [
        {
            "source_id": "source",
            "target_id": "relevant",
            "type": "RELATED",
            "confidence": 0.9,
        },
        {
            "source_id": "source",
            "target_id": "noise",
            "type": "RELATED",
            "confidence": 1.0,
        },
    ]
    harness._active_requirement_slots = {
        "r1": {
            "id": "r1",
            "need": "source topic",
            "material_anchors": ["source topic"],
            "selector": {},
        },
        "r2": {
            "id": "r2",
            "need": "target topic",
            "material_anchors": ["target topic"],
            "selector": {},
        },
    }
    harness._active_query_links = [
        {"type": "DEPENDS_ON", "from": "r1", "to": "r2"}
    ]
    candidates = harness._rg_relation_neighbor_candidates(
        [source], ["r1"], None, {"source"}
    )
    assert [item[1]["id"] for item in candidates] == ["relevant"]
    assert candidates[0][3] == "r2"
