from methods.smart_mem0.read_evidence_resolve import ReadEvidenceResolveMixin


class _BaseHarness:
    HARD_MEMORY_LIMIT = 8

    @classmethod
    def _rc_text(cls, value):
        return " ".join(str(value or "").lower().split())

    def _rc_question_span(self, value, question):
        value = " ".join(str(value or "").split()).strip()
        return value if value and value in question else ""

    def _question_stem(self, question):
        return question

    def _aop_vnext_to_legacy(self, parsed, question):
        requirements = []
        for index, raw in enumerate(parsed.get("requirements") or []):
            obligation = raw.get("answer_obligation") or ""
            requirements.append({"id": raw.get("id") or f"r{index + 1}", "grounding_kind": raw.get("grounding_kind") or "QUESTION", "focus_span": obligation, "target": raw.get("evidence_family") or obligation, "retrieval_hint": raw.get("evidence_family") or obligation, "time_constraint": dict(raw.get("selector") or {})})
        return {"requirements": requirements, "_vnext_metadata": {}}

    def _rc_normalize_ir(self, parsed, question, frame):
        return dict(parsed)

    def _compile_gap_operations(self, slots, question, budget_tier="MEDIUM", plan=None):
        return [{"op": "SEARCH_FAMILY", "query": "legacy joined query", "top_k": 4, "family_mode": "semantic", "produces": [slots[0]["id"]]}]

    def _execute_operation(self, operation, outputs, seeds, frame=None):
        rows = {
            "missed insulin doses": [{"id": "m_good", "claim": "Missed insulin doses during schedule chaos"}, {"id": "m_other", "claim": "Other insulin note"}],
            "insulin_regimen_adherence": [{"id": "m_good", "claim": "Missed insulin doses during schedule chaos"}, {"id": "m_adherent", "claim": "Usually adherent"}],
            "insulin adherence": [{"id": "m_good", "claim": "Missed insulin doses during schedule chaos"}],
            "full question symptoms and missed insulin": [{"id": "m_symptom", "claim": "Morning nausea"}],
        }
        return rows.get(operation.get("query"), []), [], []

    def _role_aware_support_ids(self, slots, slot_support, candidate_order, limit):
        return list(candidate_order)[:limit]

    def _terminal_render_answer(self, question, answer_type, memory, *, candidate_answer="", answer_field=""):
        return candidate_answer

    def _terminal_candidate_is_scalar_surface(self, candidate, memory):
        return candidate in {str(memory.get("value", "")), str(memory.get("verbatim_value", ""))}

    @staticmethod
    def _rq_surface_chars(value):
        return "".join(str(value or "").split())

    def _controller_plan(self, ir, question, frame):
        return {"compiled_mode": "DIRECT", "query_mode": "DIRECT", "query_spec": {}, "semantic_ir": {}}

    def _semantic_controller(self, question, seeds, frame, context_map=None):
        if context_map == "direct":
            return [seeds[0]], {}, {"route": "DIRECT"}
        return None, {"program_shape": "ATOMIC"}, {"route": "PLAN"}

    def prepare_batch_query(self, question, system_message=None, **kwargs):
        return {"precomputed_answer": kwargs.get("precomputed_answer"), "retrieved_memories": kwargs.get("retrieved_memories", []), "messages": [{"role": "system", "content": "base"}], "extra": kwargs.get("extra", {})}


class _Harness(ReadEvidenceResolveMixin, _BaseHarness):
    pass


def test_paraphrased_question_requirement_is_bound_not_deleted():
    harness = _Harness()
    question = "Recently I feel a bit nauseated in the morning and my chest feels tight. Could this come from missed insulin doses?"
    parsed = {"subject_span": "missed insulin doses", "requirements": [{"id": "r1", "grounding_kind": "QUESTION", "answer_obligation": "morning nausea and chest tightness", "evidence_family": "morning_symptoms"}]}
    requirement = harness._aop_vnext_to_legacy(parsed, question)["requirements"][0]
    assert requirement["focus_span"] == "missed insulin doses"
    assert requirement["_semantic_focus_binding"] == "PARAPHRASED_QUESTION_OBLIGATION"


def test_valid_and_partial_graph_states_are_preserved():
    harness = _Harness()
    valid = harness._rc_normalize_ir({"requirements": [{"id": "r1"}], "graph_validation": {"valid": True}, "normalization_status": "VALID"}, "q", None)
    partial = harness._rc_normalize_ir({"requirements": [{"id": "r1"}], "graph_validation": {"valid": False, "orphan_requirements": ["r1"]}, "normalization_status": "VALID"}, "q", None)
    assert valid["graph_state"] == "VALID"
    assert partial["graph_state"] == "PARTIAL"


def test_old_direct_b_is_exposed_as_evidence_resolve():
    harness = _Harness()
    plan = harness._controller_plan({"graph_state": "VALID", "graph_integrity_semantics": "x"}, "q", None)
    assert plan["resolution_path"] == "EVIDENCE_RESOLVE"
    assert plan["program_shape"] == "ATOMIC"
    assert plan["legacy_compiled_mode"] == "DIRECT"


def test_certified_direct_and_evidence_resolve_are_distinct():
    harness = _Harness()
    seed = {"id": "m1"}
    supports, _, telemetry = harness._semantic_controller("q", [seed], None, context_map="direct")
    assert supports
    assert telemetry["resolution_path"] == "CERTIFIED_DIRECT"
    assert telemetry["program_shape"] == "CERTIFIED_ATOMIC"
    supports, plan, telemetry = harness._semantic_controller("q", [seed], None)
    assert supports is None
    assert plan["program_shape"] == "ATOMIC"
    assert telemetry["resolution_path"] == "EVIDENCE_RESOLVE"


def test_requirement_recall_uses_independent_views_and_rrf_binding():
    harness = _Harness()
    slot = {"id": "r2", "type": "DIRECT", "answer_obligation": "missed insulin doses", "evidence_family": "insulin_regimen_adherence", "resolved_keys": ["insulin adherence"]}
    operations = harness._compile_gap_operations([slot], "full question symptoms and missed insulin")
    assert [view["kind"] for view in operations[0]["retrieval_views"]] == ["obligation", "family", "keys", "question"]
    harness._last_requirement_view_coverage = {}
    harness._last_requirement_binding_scores = {}
    harness._last_requirement_binding_views = {}
    rows, _, _ = harness._execute_operation(operations[0], [], [], None)
    assert rows[0]["id"] == "m_good"
    assert {"obligation", "family", "keys"}.issubset(rows[0]["_binding_views"])
    assert harness._last_requirement_view_coverage["r2"]["question"] == ["m_symptom"]


def test_requirement_binding_reserves_endpoints_round_robin():
    harness = _Harness()
    harness._last_requirement_binding_scores = {"r1": {"m1": 0.5, "m2": 0.4}, "r2": {"m3": 0.6, "m4": 0.3}}
    harness._last_candidate_local_coverage = {}
    harness._last_candidate_propositions = {}
    slots = [{"id": "r1", "type": "DIRECT"}, {"id": "r2", "type": "DIRECT"}]
    selected = harness._role_aware_support_ids(slots, {}, ["m1", "m2", "m3", "m4", "m5"], 5)
    assert selected[:4] == ["m1", "m3", "m2", "m4"]


def test_candidate_local_quota_is_fair_before_second_round():
    harness = _Harness()
    harness._last_requirement_binding_scores = {}
    harness._last_candidate_propositions = {"A": "a", "B": "b", "C": "c", "D": "d"}
    harness._last_candidate_local_coverage = {"A": ["a1", "a2"], "B": ["b1", "b2"], "C": ["c1", "c2"], "D": ["d1", "d2"]}
    selected = harness._role_aware_support_ids([], {}, ["a1", "a2", "b1", "b2", "c1", "c2", "d1", "d2"], 7)
    assert selected == ["a1", "b1", "c1", "d1", "a2", "b2", "c2"]


def test_certified_direct_qualitative_value_renders_complete_claim():
    harness = _Harness()
    memory = {"value": "stabilized", "verbatim_value": "basically stopped", "claim": "Patient's weight stopped dropping continuously this month and stabilized."}
    assert harness._terminal_render_answer("q", "VALUE", memory, candidate_answer="stabilized").startswith("Patient's weight stopped")
    numeric = {"value": "18.4 mmol/L", "verbatim_value": "18.4 mmol/L", "claim": "Fasting glucose was 18.4 mmol/L."}
    assert harness._terminal_render_answer("q", "VALUE", numeric, candidate_answer="18.4 mmol/L") == "18.4 mmol/L"


def test_two_stage_architecture_adds_no_model_stage():
    assert ReadEvidenceResolveMixin.CERTIFIED_DIRECT_PATH == "CERTIFIED_DIRECT"
    assert ReadEvidenceResolveMixin.EVIDENCE_RESOLVE_PATH == "EVIDENCE_RESOLVE"
    assert ReadEvidenceResolveMixin.REQUIREMENT_BINDING_VERSION == "requirement-binding-v1"
