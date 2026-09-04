"""Offline contract tests for SmartMem0's minimal semantic IR."""

import json
import re
from copy import deepcopy
from types import SimpleNamespace

from methods.smart_mem0.contracts import QueryFrame
from methods.smart_mem0.execution import ExecutionMixin
from methods.smart_mem0.read_controller import SEMANTIC_IR_POLICY, ReadContractMixin
from methods.smart_mem0.read_execution_contract import ReadExecutionContractMixin
from methods.smart_mem0.read_option_contract import ReadOptionContractMixin
from methods.smart_mem0.read_plan_contract import ReadPlanContractMixin
from methods.smart_mem0.read_usage_contract import ReadUsageContractMixin


class _JsonClient:
    def __init__(self, payload):
        self.payload = payload
        self.calls = 0

    def chat(self, *_args, **_kwargs):
        self.calls += 1
        return SimpleNamespace(content=json.dumps(self.payload))


class _MinimalHarness(
    ReadContractMixin,
    ReadPlanContractMixin,
    ReadExecutionContractMixin,
):
    subject_aliases = {"patient": "primary_user"}

    def __init__(self, memories=None):
        self._memories = list(memories or [])
        self._belief_status = {
            memory["id"]: memory.get("_status", "active") for memory in self._memories
        }
        self._state_heads = {}
        self._active_controller_seeds = self._memories[:3]
        self._relations = []
        self._last_option_probe_coverage = {}
        self._llm_client = _JsonClient({})

    @staticmethod
    def _question_stem(question):
        pattern = r"(?m)^\s*(?:\(?[A-Ha-h]\)|[A-Ha-h][.)])\s+"
        return re.split(pattern, question)[0].strip()

    @staticmethod
    def _question_options(question):
        return {
            match.group(1).upper(): match.group(2).strip()
            for match in re.finditer(
                r"(?m)^\s*\(?([A-Ha-h])\)?[.)]\s+(.+?)\s*$", question
            )
        }

    @staticmethod
    def _memory_value(memory):
        return memory.get("value") or memory.get("verbatim_value") or ""

    @staticmethod
    def _response_usage(_response, _prompt):
        return {"total_tokens": 17, "latency": 0.01}

    @staticmethod
    def _parse_json(content):
        return json.loads(content)

    def _validate_fast_support(self, reference, seeds, frame):
        match = re.fullmatch(r"\$seed([0-2])", str(reference or ""))
        if not match or int(match.group(1)) >= len(seeds):
            return None
        memory = seeds[int(match.group(1))]
        if (
            memory.get("assertion_mode", "DIRECT") != "DIRECT"
            or not self._memory_value(memory)
            or memory.get("_status", "active") in {"superseded", "conflicting"}
        ):
            return None
        if frame.dates and memory.get("document_time") not in frame.dates:
            return None
        return [deepcopy(memory)]

    @staticmethod
    def _is_state_head(memory):
        return bool(memory.get("is_state_head"))


class _OptionBase:
    @staticmethod
    def _coverage_map(plan, slot_support, selected, relations):
        del selected, relations
        return {
            slot["id"]: bool(slot_support.get(slot["id"]))
            for slot in plan.get("required_slots", [])
        }


class _OptionHarness(ReadOptionContractMixin, _OptionBase):
    pass


class _UsageBase:
    def prepare_batch_query(self, question, system_message=None, **kwargs):
        del question, system_message, kwargs
        return deepcopy(self.prepared)


class _UsageHarness(ReadUsageContractMixin, _UsageBase):
    pass


def _memory(memory_id="m1", value="cefuroxime", **extra):
    return {
        "id": memory_id,
        "claim": extra.pop("claim", f"The participant took {value}."),
        "value": value,
        "verbatim_value": "",
        "kind": extra.pop("kind", "FACT"),
        "assertion_mode": "DIRECT",
        "subject": "primary_user",
        "entities": [value],
        "scope_entities": [],
        "evidence_ids": ["ev1"],
        "document_time": "2024-01-01",
        **extra,
    }


def _ir(requirements, relations=None, candidate=None, answer_type="TEXT"):
    return {
        "answer_type": answer_type,
        "subject_span": "",
        "requirements": requirements,
        "relations": relations or [],
        "candidate": candidate,
        "visible_options": {},
        "_resolved_subject_id": "primary_user",
    }


def test_controller_schema_does_not_ask_llm_for_route_or_operations():
    forbidden = (
        '"route":',
        '"operator":',
        '"query_mode":',
        '"budget_tier":',
        '"operations":',
    )
    assert all(token not in SEMANTIC_IR_POLICY for token in forbidden)


def test_focus_span_is_question_owned_but_hint_may_be_seed_conditioned():
    harness = _MinimalHarness()
    question = (
        "After taking which antibiotic did I develop a systemic allergic reaction?"
    )
    ir = harness._rc_normalize_ir(
        {
            "requirements": [
                {
                    "id": "r1",
                    "focus_span": "cefuroxime allergy",
                    "retrieval_hint": "antibiotic exposure associated with systemic allergy",
                }
            ]
        },
        question,
        QueryFrame(),
    )
    requirement = ir["requirements"][0]
    assert requirement["focus_span"] == harness._question_stem(question)
    assert requirement["retrieval_hint"] == (
        "antibiotic exposure associated with systemic allergy"
    )


def test_legacy_route_and_operator_fields_have_no_semantic_effect():
    harness = _MinimalHarness()
    ir = harness._rc_normalize_ir(
        {
            "route": "ANSWER",
            "operator": "CAUSAL",
            "requirements": [{"id": "r1", "focus_span": "which antibiotic"}],
        },
        "After which antibiotic did the reaction occur?",
        QueryFrame(),
    )
    public = harness._rc_public_ir(ir)
    assert "route" not in public
    assert "operator" not in public
    assert public["relations"] == []


def test_atomic_candidate_authorization_is_structural_not_lexical():
    seed = _memory(claim="Cefuroxime was followed by generalized hives.")
    harness = _MinimalHarness([seed])
    ir = _ir(
        [
            {
                "id": "r1",
                "focus_span": "which antibiotic",
                "retrieval_hint": "systemic allergy exposure",
                "time_constraint": {},
            }
        ],
        candidate={"answer": "cefuroxime", "support_ref": "$seed0"},
        answer_type="ENTITY",
    )
    supports, reason = harness._authorize_controller_answer(ir, [seed], QueryFrame())
    assert reason == "AUTHORIZED"
    assert [memory["id"] for memory in supports] == ["m1"]


def test_explicit_date_blocks_atomic_candidate():
    seed = _memory()
    harness = _MinimalHarness([seed])
    ir = _ir(
        [{"id": "r1", "focus_span": "which antibiotic", "time_constraint": {}}],
        candidate={"answer": "cefuroxime", "support_ref": "$seed0"},
    )
    supports, reason = harness._authorize_controller_answer(
        ir, [seed], QueryFrame(dates=("2024-01-01",))
    )
    assert supports is None
    assert reason == "TEMPORAL_CONSTRAINT_REQUIRES_PLAN"


def test_current_candidate_requires_a_real_state_head():
    seed = _memory(kind="STATE")
    harness = _MinimalHarness([seed])
    ir = _ir(
        [{"id": "r1", "focus_span": "currently taking", "time_constraint": {}}],
        relations=[{"type": "CURRENT", "from": "r1", "to": ""}],
        candidate={"answer": "cefuroxime", "support_ref": "$seed0"},
    )
    supports, reason = harness._authorize_controller_answer(ir, [seed], QueryFrame())
    assert supports is None
    assert reason == "CURRENT_CANDIDATE_IS_NOT_STATE_HEAD"

    seed["is_state_head"] = True
    supports, reason = harness._authorize_controller_answer(ir, [seed], QueryFrame())
    assert reason == "AUTHORIZED"
    assert supports


def test_time_relation_without_axis_is_not_implicitly_event_time():
    harness = _MinimalHarness()
    constraint = harness._rc_normalize_time_constraint(
        {"relation": "LATEST"}, QueryFrame(), "What happened latest?"
    )
    assert constraint == {"axis": "", "relation": "", "anchor": "", "end": ""}


def test_requirement_graph_derives_mode_budget_and_operations():
    harness = _MinimalHarness()
    ir = _ir(
        [
            {
                "id": "r1",
                "focus_span": "January dose",
                "retrieval_hint": "baseline dose",
                "time_constraint": {},
            },
            {
                "id": "r2",
                "focus_span": "March dose",
                "retrieval_hint": "later dose",
                "time_constraint": {},
            },
        ],
        relations=[{"type": "COMPARE", "from": "r1", "to": "r2"}],
    )
    plan = harness._controller_plan(
        ir, "Compare January dose with March dose.", QueryFrame()
    )
    assert plan["compiled_mode"] == "COMPARISON"
    assert plan["budget_tier"] == "MEDIUM"
    assert plan["seed_coverage"] == []
    assert [operation["op"] for operation in plan["operations"]] == [
        "SEMANTIC_SEARCH",
        "SEMANTIC_SEARCH",
    ]
    assert {slot["evidence_role"] for slot in plan["required_slots"]} == {"REQUIREMENT"}


def test_possible_cause_derives_general_knowledge_bridge():
    harness = _MinimalHarness()
    ir = _ir(
        [
            {"id": "r1", "focus_span": "missed doses", "time_constraint": {}},
            {"id": "r2", "focus_span": "current symptoms", "time_constraint": {}},
        ],
        relations=[{"type": "POSSIBLE_CAUSE", "from": "r1", "to": "r2"}],
    )
    plan = harness._controller_plan(
        ir, "Could missed doses explain current symptoms?", QueryFrame()
    )
    assert plan["query_spec"]["requires_inference"] is True
    assert plan["query_spec"]["world_knowledge_bridge_allowed"] is True


def test_current_relation_compiles_state_resolution():
    harness = _MinimalHarness()
    ir = _ir(
        [{"id": "r1", "focus_span": "currently taking", "time_constraint": {}}],
        relations=[{"type": "CURRENT", "from": "r1", "to": ""}],
    )
    plan = harness._controller_plan(ir, "What am I currently taking?", QueryFrame())
    assert plan["required_slots"][0]["type"] == "CURRENT_STATE"
    assert plan["operations"][0]["op"] == "RESOLVE_STATE"


def test_temporal_axis_is_preserved_without_fallback():
    harness = _MinimalHarness()
    ir = _ir(
        [
            {
                "id": "r1",
                "focus_span": "documented",
                "retrieval_hint": "weight change documentation",
                "time_constraint": {
                    "axis": "document_time",
                    "relation": "LOCATE",
                    "anchor": "",
                    "end": "",
                },
            }
        ]
    )
    plan = harness._controller_plan(ir, "When was the change documented?", QueryFrame())
    operation = plan["operations"][0]
    assert operation["op"] == "TEMPORAL_FILTER"
    assert operation["axis"] == "document_time"
    assert operation["fallback_axis"] == ""


def test_visible_options_compile_one_shared_physical_operation():
    harness = _MinimalHarness()
    ir = _ir(
        [{"id": "r1", "focus_span": "Which action is supported", "time_constraint": {}}]
    )
    ir["visible_options"] = {"A": "Keep monitoring", "B": "Add testing"}
    plan = harness._controller_plan(
        ir,
        "Which action is supported?\nA. Keep monitoring\nB. Add testing",
        QueryFrame(),
    )
    assert plan["query_spec"]["answer_type"] == "OPTION_SET"
    assert len(plan["operations"]) == 1
    operation = plan["operations"][0]
    assert operation["strategy"] == "SHARED_OPTIONS"
    assert operation["produces"] == ["r1"]
    assert [item["label"] for item in operation["option_queries"]] == ["A", "B"]


def test_option_coverage_requires_every_visible_option_to_be_probed():
    harness = _OptionHarness()
    plan = {
        "visible_options": {"A": "one", "B": "two"},
        "required_slots": [{"id": "r1"}],
    }
    harness._last_option_probe_coverage = {"A": ["m1"]}
    assert harness._coverage_map(plan, {"r1": ["m1"]}, [], []) == {"r1": False}
    harness._last_option_probe_coverage["B"] = []
    assert harness._coverage_map(plan, {"r1": ["m1"]}, [], []) == {"r1": True}


def test_deterministic_recovery_broadens_only_soft_hint():
    harness = _MinimalHarness()
    slot = {
        "id": "r1",
        "type": "DIRECT",
        "evidence_role": "REQUIREMENT",
        "target_surface": "which antibiotic",
        "retrieval_hint": "antibiotic exposure associated with allergy",
        "resolved_keys": [],
    }
    initial = {
        "query_spec": {},
        "query_mode": "DIRECT",
        "required_slots": [slot],
        "semantic_relations": [],
        "visible_options": {},
        "budget_tier": "SMALL",
        "operations": harness._compile_gap_operations([slot], "question", "SMALL"),
    }
    recovery = harness._make_deterministic_recovery_plan([slot], "question", initial)
    assert recovery is not None
    assert recovery["required_slots"][0]["target_surface"] == "which antibiotic"
    assert recovery["required_slots"][0]["retrieval_hint"] == ""


def test_minimal_ir_path_never_requests_middle_llm_validation():
    assert ExecutionMixin._plan_requires_semantic_validation({}) is False


def test_two_stage_usage_has_hard_two_call_budget():
    harness = _UsageHarness()
    harness.prepared = {
        "precomputed_answer": "",
        "extra": {
            "semantic_controller": {"called": True},
            "planner_called": True,
            "replan_called": False,
            "slot_validation": [],
            "query_tokens": {"controller": 100, "planner": 0, "replan": 0},
        },
    }
    prepared = harness.prepare_batch_query("q")
    calls = prepared["extra"]["method_llm_calls"]
    assert calls["total"] == 2
    assert calls["middle"] == 0
    assert prepared["extra"]["deterministic_plan_compiled"] is True
    assert prepared["extra"]["planner_llm_called"] is False

    harness.prepared["extra"]["slot_validation"] = [{"called": True}]
    prepared = harness.prepare_batch_query("q")
    assert prepared["extra"]["method_llm_calls"]["two_stage_budget_violation"] is True
