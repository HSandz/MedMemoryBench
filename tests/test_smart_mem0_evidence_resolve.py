from methods.smart_mem0.read_evidence_resolve import ReadEvidenceResolveMixin


class _BaseHarness:
    def _rc_question_span(self, value, question):
        value = " ".join(str(value or "").split()).strip()
        return value if value and value in question else ""

    def _question_stem(self, question):
        return question

    def _aop_vnext_to_legacy(self, parsed, question):
        requirements = []
        for index, raw in enumerate(parsed.get("requirements") or []):
            obligation = raw.get("answer_obligation") or ""
            requirements.append(
                {
                    "id": raw.get("id") or f"r{index + 1}",
                    "grounding_kind": raw.get("grounding_kind") or "QUESTION",
                    "focus_span": obligation,
                    "target": raw.get("evidence_family") or obligation,
                    "retrieval_hint": raw.get("evidence_family") or obligation,
                    "time_constraint": dict(raw.get("selector") or {}),
                }
            )
        return {"requirements": requirements, "_vnext_metadata": {}}

    def _rc_normalize_ir(self, parsed, question, frame):
        return dict(parsed)

    def _controller_plan(self, ir, question, frame):
        return {
            "compiled_mode": "DIRECT",
            "query_mode": "DIRECT",
            "query_spec": {},
            "semantic_ir": {},
        }

    def _semantic_controller(self, question, seeds, frame, context_map=None):
        if context_map == "direct":
            return [seeds[0]], {}, {"route": "DIRECT"}
        return None, {"program_shape": "ATOMIC"}, {"route": "PLAN"}

    def prepare_batch_query(self, question, system_message=None, **kwargs):
        return {
            "precomputed_answer": kwargs.get("precomputed_answer"),
            "extra": kwargs.get("extra", {}),
        }


class _Harness(ReadEvidenceResolveMixin, _BaseHarness):
    pass


def test_paraphrased_question_requirement_is_bound_not_deleted():
    harness = _Harness()
    question = (
        "Recently I feel a bit nauseated in the morning and my chest feels tight. "
        "Could this come from missed insulin doses?"
    )
    parsed = {
        "subject_span": "missed insulin doses",
        "requirements": [
            {
                "id": "r1",
                "grounding_kind": "QUESTION",
                "answer_obligation": "morning nausea and chest tightness",
                "evidence_family": "morning_symptoms",
            }
        ],
    }
    legacy = harness._aop_vnext_to_legacy(parsed, question)
    requirement = legacy["requirements"][0]
    assert requirement["focus_span"] == "missed insulin doses"
    assert requirement["_semantic_focus_binding"] == "PARAPHRASED_QUESTION_OBLIGATION"


def test_valid_and_partial_graph_states_are_preserved():
    harness = _Harness()
    valid = harness._rc_normalize_ir(
        {
            "requirements": [{"id": "r1"}],
            "graph_validation": {"valid": True},
            "normalization_status": "VALID",
        },
        "q",
        None,
    )
    partial = harness._rc_normalize_ir(
        {
            "requirements": [{"id": "r1"}],
            "graph_validation": {"valid": False, "orphan_requirements": ["r1"]},
            "normalization_status": "VALID",
        },
        "q",
        None,
    )
    assert valid["graph_state"] == "VALID"
    assert partial["graph_state"] == "PARTIAL"


def test_old_direct_b_is_exposed_as_evidence_resolve():
    harness = _Harness()
    plan = harness._controller_plan(
        {"graph_state": "VALID", "graph_integrity_semantics": "x"}, "q", None
    )
    assert plan["resolution_path"] == "EVIDENCE_RESOLVE"
    assert plan["program_shape"] == "ATOMIC"
    assert plan["legacy_compiled_mode"] == "DIRECT"


def test_certified_direct_and_evidence_resolve_are_distinct():
    harness = _Harness()
    seed = {"id": "m1"}
    supports, _, telemetry = harness._semantic_controller(
        "q", [seed], None, context_map="direct"
    )
    assert supports
    assert telemetry["resolution_path"] == "CERTIFIED_DIRECT"
    assert telemetry["program_shape"] == "CERTIFIED_ATOMIC"

    supports, plan, telemetry = harness._semantic_controller("q", [seed], None)
    assert supports is None
    assert plan["program_shape"] == "ATOMIC"
    assert telemetry["resolution_path"] == "EVIDENCE_RESOLVE"


def test_two_stage_architecture_adds_no_model_stage():
    assert ReadEvidenceResolveMixin.CERTIFIED_DIRECT_PATH == "CERTIFIED_DIRECT"
    assert ReadEvidenceResolveMixin.EVIDENCE_RESOLVE_PATH == "EVIDENCE_RESOLVE"
