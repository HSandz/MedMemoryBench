from methods.smart_mem0.read_evidence_resolve import ReadEvidenceResolveMixin
from methods.smart_mem0.read_lean_execution_adapter import ReadLeanExecutionAdapterMixin


class _PhysicalSearchHarness:
    @classmethod
    def _rc_text(cls, value):
        return " ".join(str(value or "").lower().split())

    def _question_stem(self, question):
        return question

    def _compile_gap_operations(self, slots, question, budget_tier="MEDIUM", plan=None):
        return [
            {
                "op": "SEARCH_FAMILY",
                "query": "legacy joined query",
                "top_k": 4,
                "family_mode": "semantic",
                "produces": [slots[0]["id"]],
            }
        ]

    def _execute_operation(self, operation, outputs, seeds, frame=None):
        rows = {
            "missed insulin doses": [
                {"id": "m_good", "claim": "Missed insulin doses during schedule chaos"},
                {"id": "m_other", "claim": "Other insulin note"},
            ],
            "insulin_regimen_adherence": [
                {"id": "m_good", "claim": "Missed insulin doses during schedule chaos"},
                {"id": "m_adherent", "claim": "Usually adherent"},
            ],
            "insulin adherence": [
                {"id": "m_good", "claim": "Missed insulin doses during schedule chaos"}
            ],
            "full question symptoms and missed insulin": [
                {"id": "m_symptom", "claim": "Morning nausea"}
            ],
        }
        return rows.get(operation.get("query"), []), [], []


class _Harness(
    ReadEvidenceResolveMixin,
    ReadLeanExecutionAdapterMixin,
    _PhysicalSearchHarness,
):
    pass


def test_multiview_search_family_survives_legacy_alias_conversion():
    harness = _Harness()
    slot = {
        "id": "r2",
        "type": "DIRECT",
        "answer_obligation": "missed insulin doses",
        "evidence_family": "insulin_regimen_adherence",
        "resolved_keys": ["insulin adherence"],
    }
    operation = harness._compile_gap_operations(
        [slot], "full question symptoms and missed insulin"
    )[0]

    internal = harness._legacy_control_operation(operation)
    assert internal["op"] == "SEARCH_FAMILY"
    assert internal["_lean_op"] == "SEARCH_FAMILY"
    assert internal["_multiview_dispatch_preserved"] is True

    harness._last_requirement_view_coverage = {}
    harness._last_requirement_binding_scores = {}
    harness._last_requirement_binding_views = {}
    rows, _, _ = harness._execute_operation(internal, [], [], None)

    assert rows[0]["id"] == "m_good"
    assert {"obligation", "family", "keys"}.issubset(rows[0]["_binding_views"])
    assert harness._last_requirement_view_coverage["r2"]["question"] == ["m_symptom"]
    assert harness._last_requirement_binding_scores["r2"]["m_good"] > harness._last_requirement_binding_scores["r2"]["m_symptom"]


def test_plain_search_family_keeps_legacy_physical_alias():
    operation = {
        "op": "SEARCH_FAMILY",
        "query": "plain semantic query",
        "family_mode": "semantic",
    }
    internal = ReadLeanExecutionAdapterMixin._legacy_control_operation(operation)
    assert internal["op"] == "SEMANTIC_SEARCH"
    assert internal["_lean_op"] == "SEARCH_FAMILY"
    assert "_multiview_dispatch_preserved" not in internal


def test_temporal_search_family_keeps_anchor_alias():
    operation = {
        "op": "SEARCH_FAMILY",
        "query": "temporal query",
        "family_mode": "temporal_extremum",
    }
    internal = ReadLeanExecutionAdapterMixin._legacy_control_operation(operation)
    assert internal["op"] == "LOCATE_ANCHOR"
    assert internal["_lean_op"] == "SEARCH_FAMILY"
