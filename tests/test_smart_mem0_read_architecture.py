"""Offline contract tests for SmartMem0's minimal semantic IR."""

import json
import re
from copy import deepcopy
from types import SimpleNamespace

import pytest

from methods.smart_mem0.contracts import QueryFrame
from methods.smart_mem0.agent import SmartMem0Agent
from methods.smart_mem0.canonicalization import (
    is_state_projection_eligible,
    state_identity,
)
from methods.smart_mem0.execution import ExecutionMixin
from methods.smart_mem0.core import CoreMemoryMixin
from methods.smart_mem0.query import QueryMixin
from methods.smart_mem0.read_controller import SEMANTIC_IR_POLICY, ReadContractMixin
from methods.smart_mem0.read_execution_contract import ReadExecutionContractMixin
from methods.smart_mem0.read_option_contract import ReadOptionContractMixin
from methods.smart_mem0.read_plan_contract import ReadPlanContractMixin
from methods.smart_mem0.read_temporal_contract import ReadTemporalContractMixin
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
    ReadTemporalContractMixin,
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

    @staticmethod
    def _parse_date(value):
        value = str(value or "")
        return value if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value) else ""

    @staticmethod
    def _date_for(memory, axis="event_time"):
        value = str(memory.get(axis) or "")
        return "" if value == "UNKNOWN" else value

    @staticmethod
    def _valid_causal_relation(relation, by_id):
        source = by_id.get(relation.get("source_id"))
        target = by_id.get(relation.get("target_id"))
        provenance = set(relation.get("provenance_evidence_ids") or [])
        return bool(
            relation.get("type") == "CAUSES"
            and source
            and target
            and source.get("evidence_ids")
            and target.get("evidence_ids")
            and provenance
            and provenance.issubset(
                set(source.get("evidence_ids") or [])
                | set(target.get("evidence_ids") or [])
            )
        )


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


class _ExecutionHarness(ReadExecutionContractMixin, ExecutionMixin):
    HARD_MEMORY_LIMIT = 8

    def __init__(self):
        self._relations = []
        self._belief_status = {}
        self._memories = []
        self._operation_results = {}

    @staticmethod
    def _snapshot(value):
        return deepcopy(value)

    @staticmethod
    def _slot_seed_support(plan, seeds):
        del seeds
        return {slot["id"]: [] for slot in plan["required_slots"]}

    def _execute_operation(self, operation, outputs, seeds, frame):
        del outputs, seeds, frame
        return deepcopy(self._operation_results[operation["query"]]), [], []

    @staticmethod
    def _operation_slot_support(slot, result, relations):
        del slot, relations
        return result[:1]

    @staticmethod
    def _reconstruct_beliefs(memories, limit, prefer_active=True):
        del prefer_active
        return deepcopy(memories[:limit]), []

    @staticmethod
    def _merge_relations(*groups, **_kwargs):
        return [relation for group in groups for relation in group]

    @staticmethod
    def _slot_covered(slot, support_ids, selected, relations):
        del slot, relations
        selected_ids = {memory["id"] for memory in selected}
        return bool(set(support_ids) & selected_ids)


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


def test_atomic_candidate_rejects_incidental_entity_in_rich_seed():
    seed = _memory(
        claim="Cefuroxime caused hives, while NSAIDs were contraindicated.",
        value="contraindicated",
        verbatim_value="NSAIDs were contraindicated",
        object_anchor="NSAIDs",
        entities=["cefuroxime", "NSAIDs"],
    )
    harness = _MinimalHarness([seed])
    ir = _ir(
        [
            {
                "id": "r1",
                "focus_span": "which contraindicated medication",
                "time_constraint": {},
            }
        ],
        candidate={"answer": "cefuroxime", "support_ref": "$seed0"},
        answer_type="ENTITY",
    )
    supports, reason = harness._authorize_controller_answer(ir, [seed], QueryFrame())
    assert supports is None
    assert reason == "ANSWER_NOT_IN_ATOMIC_SURFACE"


def test_target_matching_uses_tokens_not_substrings():
    harness = _MinimalHarness()
    memory = _memory(value="current condition", claim="diagnosis is current")
    assert not harness._rc_memory_matches_target({"target_surface": "rent"}, memory)
    assert not harness._rc_memory_matches_target({"target_surface": "ia"}, memory)
    assert not harness._rc_answer_grounded("rent", [memory])
    assert not harness._rc_answer_grounded("ia", [memory])
    assert harness._rc_memory_matches_target(
        {"target_surface": "current condition"}, memory
    )


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


def test_exact_date_filter_can_authorize_matching_atomic_candidate():
    seed = _memory(event_time="2024-01-01")
    harness = _MinimalHarness([seed])
    ir = _ir(
        [
            {
                "id": "r1",
                "focus_span": "which antibiotic",
                "time_constraint": {
                    "axis": "event_time",
                    "relation": "EXACT",
                    "anchor": "2024-01-01",
                },
            }
        ],
        candidate={"answer": "cefuroxime", "support_ref": "$seed0"},
        answer_type="ENTITY",
    )
    supports, reason = harness._authorize_controller_answer(
        ir, [seed], QueryFrame(dates=("2024-01-01",))
    )
    assert reason == "AUTHORIZED"
    assert [memory["id"] for memory in supports] == ["m1"]


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


def test_temporal_order_parser_requires_an_explicit_supported_relation():
    harness = _MinimalHarness()
    question = "Which medication started after the blood pressure increase?"
    base = {
        "requirements": [
            {"id": "r1", "focus_span": "blood pressure increase"},
            {"id": "r2", "focus_span": "Which medication"},
        ],
        "relations": [
            {
                "type": "TEMPORAL_ORDER",
                "from": "r2",
                "to": "r1",
                "relation": "AFTER",
            },
            {"type": "TEMPORAL_ORDER", "from": "r1", "to": "r2"},
        ],
    }
    ir = harness._rc_normalize_ir(base, question, QueryFrame())
    assert ir["relations"] == [
        {
            "type": "TEMPORAL_ORDER",
            "from": "r2",
            "to": "r1",
            "relation": "AFTER",
        }
    ]


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
    assert {slot["evidence_role"] for slot in plan["required_slots"]} == {"COMPARAND"}


def test_relative_temporal_order_compiles_anchor_then_filter():
    harness = _MinimalHarness()
    ir = _ir(
        [
            {
                "id": "r1",
                "focus_span": "blood pressure began increasing",
                "time_constraint": {},
            },
            {"id": "r2", "focus_span": "which medication", "time_constraint": {}},
        ],
        relations=[
            {
                "type": "TEMPORAL_ORDER",
                "from": "r2",
                "to": "r1",
                "relation": "AFTER",
            }
        ],
        answer_type="ENTITY",
    )
    plan = harness._controller_plan(
        ir,
        "Which medication did I start after my blood pressure began increasing?",
        QueryFrame(),
    )
    assert plan["compiled_mode"] == "TEMPORAL"
    assert plan["query_spec"]["requires_inference"] is True
    assert plan["required_slots"][1]["time_axis"] == "event_time"
    assert plan["required_slots"][1]["relative_to_requirement"] == "r1"
    assert [operation["op"] for operation in plan["operations"]] == [
        "SEMANTIC_SEARCH",
        "TEMPORAL_FILTER",
    ]
    assert plan["operations"][1]["anchor"] == "$0"
    assert plan["operations"][1]["anchor_requirement"] == "r1"
    assert plan["operations"][1]["relation"] == "AFTER"
    assert plan["operations"][1]["fallback_axis"] == ""


def test_relative_temporal_overlap_requires_the_resolved_anchor_date():
    harness = _MinimalHarness()
    memory = _memory(event_time="2024-04-03")
    slot = {
        "id": "r2",
        "type": "TEMPORAL",
        "evidence_role": "REQUIREMENT",
        "subject_id": "primary_user",
        "required_fields": ["event_time"],
        "time_axis": "event_time",
        "temporal_relation": "OVERLAPS",
        "resolved_time_anchor": "2024-04-03",
    }
    assert harness._slot_covered(slot, ["m1"], [memory], [])
    slot["resolved_time_anchor"] = "2024-04-04"
    assert not harness._slot_covered(slot, ["m1"], [memory], [])


def test_temporal_overlap_accepts_month_and_day_intervals():
    harness = _MinimalHarness()
    assert harness._temporal_relation_holds("2024-04", "2024-04-03", "OVERLAPS")
    assert not harness._temporal_relation_holds("2024-05", "2024-04-03", "OVERLAPS")


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
    assert [
        slot["evidence_role"] for slot in plan["required_slots"]
    ] == ["REQUIREMENT", "REQUIREMENT"]


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
    assert plan["need_evidence"] is False
    assert all(op["op"] != "VERIFY_EVIDENCE" for op in plan["operations"])


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
    assert plan["required_slots"][0]["evidence_role"] == "OPTION_CONTEXT"


def test_focus_months_bind_to_their_own_comparison_requirements():
    harness = _MinimalHarness()
    question = "How did the January dose compare with the March dose?"
    ir = harness._rc_normalize_ir(
        {
            "answer_type": "VALUE",
            "requirements": [
                {"id": "r1", "focus_span": "January dose"},
                {"id": "r2", "focus_span": "March dose"},
            ],
            "relations": [{"type": "COMPARE", "from": "r1", "to": "r2"}],
        },
        question,
        QueryFrame(dates=("*-01", "*-03")),
    )
    assert ir["requirements"][0]["time_constraint"]["anchor"] == "*-01"
    assert ir["requirements"][1]["time_constraint"]["anchor"] == "*-03"


def test_non_temporal_answer_drops_meaningless_locate_and_keeps_current():
    harness = _MinimalHarness()
    question = "What medication am I currently taking?"
    ir = harness._rc_normalize_ir(
        {
            "answer_type": "ENTITY",
            "requirements": [
                {
                    "id": "r1",
                    "focus_span": "medication",
                    "time_constraint": {
                        "axis": "event_time",
                        "relation": "LOCATE",
                    },
                }
            ],
            "relations": [{"type": "CURRENT", "from": "r1"}],
        },
        question,
        QueryFrame(),
    )
    assert ir["requirements"][0]["time_constraint"]["axis"] == ""
    plan = harness._controller_plan(ir, question, QueryFrame())
    assert plan["required_slots"][0]["type"] == "CURRENT_STATE"
    assert plan["operations"][0]["op"] == "RESOLVE_STATE"


def test_atomic_candidate_removes_redundant_infer_and_overdecomposition():
    harness = _MinimalHarness()
    question = "What was the latest HbA1c result on the follow-up test?"
    ir = harness._rc_normalize_ir(
        {
            "answer_type": "VALUE",
            "requirements": [
                {
                    "id": "r1",
                    "focus_span": "latest HbA1c result",
                    "time_constraint": {
                        "axis": "event_time",
                        "relation": "LATEST",
                    },
                },
                {"id": "r2", "focus_span": "follow-up test"},
            ],
            "relations": [
                {"type": "DEPENDS_ON", "from": "r1", "to": "r2"},
                {"type": "INFER", "from": "r1", "to": "ANSWER"},
            ],
            "candidate": {"answer": "8.1%", "support_ref": "$seed0"},
        },
        question,
        QueryFrame(),
    )
    assert len(ir["requirements"]) == 1
    assert ir["requirements"][0]["time_constraint"]["relation"] == "LATEST"
    assert ir["relations"] == []


def test_retrieval_status_rejects_zero_hop_cause_and_accepts_real_edge():
    first = _memory("m1", value="missed dose", claim="The participant missed a dose.")
    second = _memory("m2", value="high glucose", claim="Glucose rose.", evidence_ids=["ev2"])
    harness = _MinimalHarness([first, second])
    plan = {
        "required_slots": [
            {"id": "r1", "type": "DIRECT", "evidence_role": "REQUIREMENT"},
            {"id": "r2", "type": "DIRECT", "evidence_role": "REQUIREMENT"},
        ],
        "semantic_relations": [{"type": "CAUSES", "from": "r1", "to": "r2"}],
    }
    statuses = harness._relation_status_map(
        plan, {"r1": ["m1"], "r2": ["m1"]}, [first], []
    )
    assert statuses["CAUSES:r1:r2"] == "UNPROVEN"
    edge = {
        "type": "CAUSES",
        "source_id": "m1",
        "target_id": "m2",
        "provenance_evidence_ids": ["ev1"],
    }
    statuses = harness._relation_status_map(
        plan, {"r1": ["m1"], "r2": ["m2"]}, [first, second], [edge]
    )
    assert statuses["CAUSES:r1:r2"] == "PROVEN"


def test_requirement_status_is_found_empty_and_relation_aware():
    first = _memory("m1", value="left")
    second = _memory("m2", value="right", evidence_ids=["ev2"])
    harness = _MinimalHarness([first, second])
    plan = {
        "required_slots": [
            {"id": "r1", "type": "DIRECT", "evidence_role": "REQUIREMENT"},
            {"id": "r2", "type": "DIRECT", "evidence_role": "REQUIREMENT"},
        ],
        "semantic_relations": [{"type": "COMPARE", "from": "r1", "to": "r2"}],
    }
    requirement, relations, complete = harness._retrieval_status(
        plan,
        {"r1": ["m1"], "r2": []},
        [first],
        [],
    )
    assert requirement == {"r1": "FOUND", "r2": "EMPTY"}
    assert relations == {"COMPARE:r1:r2": "UNPROVEN"}
    assert complete is False
    requirement, relations, complete = harness._retrieval_status(
        plan,
        {"r1": ["m1"], "r2": ["m2"]},
        [first, second],
        [],
    )
    assert requirement == {"r1": "FOUND", "r2": "FOUND"}
    assert relations == {"COMPARE:r1:r2": "PROVEN"}
    assert complete is True


def test_planned_execution_runs_every_bounded_operation_after_early_found():
    harness = _ExecutionHarness()
    harness._operation_results = {
        "first": [_memory("m1", "first")],
        "second": [_memory("m2", "second")],
    }
    plan = {
        "required_slots": [
            {"id": "r1", "type": "DIRECT"},
            {"id": "r2", "type": "DIRECT"},
        ],
        "seed_coverage": [],
        "semantic_relations": [],
        "operations": [
            {
                "op": "SEMANTIC_SEARCH",
                "query": "first",
                "top_k": 1,
                "produces": ["r1"],
            },
            {
                "op": "SEMANTIC_SEARCH",
                "query": "second",
                "top_k": 1,
                "produces": ["r2"],
            },
        ],
        "max_memories": 2,
    }
    result = harness._execute_plan(plan, [], question="q")
    assert [item["operation"] for item in result["trace"]] == [
        "SEMANTIC_SEARCH",
        "SEMANTIC_SEARCH",
    ]
    assert result["requirement_status"] == {"r1": "FOUND", "r2": "FOUND"}
    assert result["retrieval_complete"] is True


def test_state_identity_includes_scope_and_recap_cannot_become_head():
    medication = _memory(
        "m1",
        "500 mg",
        kind="STATE",
        scope="medication",
        state_key="dose",
        object_anchor="metformin",
        memory_tier="HOT",
    )
    nutrition = dict(medication, id="m2", scope="nutrition")
    assert state_identity(medication) != state_identity(nutrition)
    assert is_state_projection_eligible(medication)
    assert not is_state_projection_eligible(
        dict(medication, assertion_mode="RECAP")
    )


def test_measurement_container_is_canonical_across_lab_and_test_wording():
    lab = CoreMemoryMixin._normalise_memory(
        {
            "claim": "HbA1c was 8.1%.",
            "kind": "STATE",
            "semantic_role": "MEASUREMENT",
            "scope": "lab",
            "state_key": "hba1c_level",
            "value": "8.1%",
        }
    )
    test = CoreMemoryMixin._normalise_memory(
        {
            "claim": "The HbA1c test result was 9.2%.",
            "kind": "STATE",
            "semantic_role": "MEASUREMENT",
            "scope": "test",
            "state_key": "hba1c",
            "value": "9.2%",
        }
    )
    assert lab["scope"] == test["scope"] == "measurement"
    assert state_identity(lab) == state_identity(test)


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


@pytest.mark.parametrize("relation_type", ["CAUSES", "POSSIBLE_CAUSE"])
def test_causal_endpoints_accept_structural_paraphrase_support(relation_type):
    memory = _memory(value="high-carb takeout at 1-2 a.m.")
    harness = _MinimalHarness([memory])
    ir = _ir(
        [
            {"id": "r1", "focus_span": "those occasional late-night delivery orders"},
            {"id": "r2", "focus_span": "morning symptoms"},
        ],
        relations=[{"type": relation_type, "from": "r1", "to": "r2"}],
    )
    plan = harness._controller_plan(ir, "Could those orders explain symptoms?", QueryFrame())
    assert all(slot["evidence_role"] == "REQUIREMENT" for slot in plan["required_slots"])
    assert harness._slot_covered(plan["required_slots"][0], ["m1"], [memory], [])
    statuses = harness._relation_status_map(plan, {"r1": ["m1"], "r2": ["m1"]}, [memory], [])
    assert statuses == ({"CAUSES:r1:r2": "UNPROVEN"} if relation_type == "CAUSES" else {})


@pytest.mark.parametrize("verify_source", [False, True])
def test_document_time_does_not_imply_source_verification(verify_source):
    harness = _MinimalHarness()
    ir = harness._rc_normalize_ir(
        {
            "answer_type": "DATE",
            "requirements": [{
                "id": "r1", "focus_span": "HbA1c mentioned",
                "time_constraint": {"axis": "document_time", "relation": "LATEST"},
            }],
            "relations": [{"type": "VERIFY_SOURCE", "from": "r1"}] if verify_source else [],
        },
        "When was HbA1c mentioned most recently?",
        QueryFrame(),
    )
    assert any(r["type"] == "VERIFY_SOURCE" for r in ir["relations"]) == verify_source
    before = deepcopy(ir)
    plan = harness._controller_plan(ir, "When was HbA1c mentioned most recently?", QueryFrame())
    assert ir == before
    assert plan["need_evidence"] == verify_source
    assert any(op["op"] == "VERIFY_EVIDENCE" for op in plan["operations"]) == verify_source


@pytest.mark.parametrize("edge", [
    {"type": "TEMPORAL_ORDER", "from": "r2", "to": "r1", "relation": "OVERLAPS"},
    {"type": "DEPENDS_ON", "from": "r2", "to": "r1"},
])
def test_locate_survives_as_intermediate_anchor_for_entity_answer(edge):
    harness = _MinimalHarness()
    question = "Which medication was I taking when the rash first appeared?"
    ir = harness._rc_normalize_ir(
        {
            "answer_type": "ENTITY",
            "requirements": [
                {"id": "r1", "focus_span": "rash first appeared", "time_constraint": {
                    "axis": "event_time", "relation": "LOCATE",
                }},
                {"id": "r2", "focus_span": "Which medication"},
            ],
            "relations": [edge],
        }, question, QueryFrame(),
    )
    assert ir["requirements"][0]["time_constraint"]["relation"] == "LOCATE"
    plan = harness._controller_plan(ir, question, QueryFrame())
    anchor = plan["required_slots"][0]
    assert anchor["type"] == "TEMPORAL"
    assert not harness._slot_covered(anchor, ["m1"], [_memory(event_time="UNKNOWN")], [])


def test_temporal_order_status_keeps_distinct_predicates():
    first = _memory("m1", event_time="2024-01-01")
    second = _memory("m2", event_time="2024-02-01")
    harness = _MinimalHarness([first, second])
    plan = {"semantic_relations": [
        {"type": "TEMPORAL_ORDER", "from": "r1", "to": "r2", "relation": order}
        for order in ("BEFORE", "AFTER", "OVERLAPS")
    ]}
    assert harness._relation_status_map(
        plan, {"r1": ["m1"], "r2": ["m2"]}, [first, second], [],
    ) == {
        "TEMPORAL_ORDER:r1:r2:BEFORE": "PROVEN",
        "TEMPORAL_ORDER:r1:r2:AFTER": "UNPROVEN",
        "TEMPORAL_ORDER:r1:r2:OVERLAPS": "UNPROVEN",
    }


@pytest.mark.parametrize("subject_span", ["my mother", ""])
def test_unresolved_subject_is_not_inferred_from_seed_owners(subject_span):
    harness = _MinimalHarness([_memory(subject_id="primary_user")])
    ir = harness._rc_normalize_ir(
        {"subject_span": subject_span, "requirements": [{"id": "r1", "focus_span": "medication"}]},
        "What medication does my mother take?", QueryFrame(),
    )
    assert ir["_resolved_subject_id"] == ""
    plan = harness._controller_plan(ir, "What medication does my mother take?", QueryFrame())
    assert plan["required_slots"][0]["subject_id"] == ""


def test_question_resolved_subject_remains_a_hard_owner():
    harness = _MinimalHarness([_memory(subject_id="primary_user"), _memory("m2", subject_id="mother")])
    harness.subject_aliases = {"my mother": "mother"}
    ir = harness._rc_normalize_ir(
        {"subject_span": "my mother", "requirements": [{"id": "r1", "focus_span": "medication"}]},
        "What medication does my mother take?", QueryFrame(),
    )
    assert ir["_resolved_subject_id"] == "mother"


def test_measurement_predicate_preserves_fasting_and_postprandial_families():
    base = _memory(semantic_role="MEASUREMENT", scope="measurement", object_anchor="blood_glucose")
    fasting = dict(base, state_key="fasting_blood_glucose_level")
    postprandial = dict(base, state_key="postprandial_blood_glucose_level")
    assert state_identity(fasting) != state_identity(postprandial)
    assert state_identity(fasting) == state_identity(dict(fasting, state_key="fasting_blood_glucose_result"))
    assert state_identity(dict(base, state_key="")) == state_identity(dict(base, state_key="measurement"))


def test_supplementary_packing_preserves_slot_provenance_across_recovery(monkeypatch):
    agent = object.__new__(SmartMem0Agent)
    first = _memory("m1", "initial observation")
    second = _memory("m2", "treatment decision")
    unrelated = _memory("m3", "decision alternative")
    recovered = _memory("m4", "independent observation")
    candidates = [unrelated, recovered, first, second]
    agent._memories = candidates
    agent._evidence = []
    agent._relations = []
    agent._belief_status = {}
    agent._tokenizer = SimpleNamespace(encode=list, decode="".join)
    agent.max_context_tokens = 32000
    agent.max_question_tokens = 1200
    agent.enable_two_stage_controller = True
    monkeypatch.setattr(agent, "_constraint_first_search", lambda *_a, **_k: [])
    monkeypatch.setattr(agent, "_select_initial_seeds", lambda rows: rows)
    monkeypatch.setattr(agent, "_planning_seed_set", lambda _q, rows: rows)
    monkeypatch.setattr(agent, "_effective_runtime_config", lambda: {})
    slots = [
        {"id": sid, "description": sid, "type": "DIRECT", "evidence_role": "REQUIREMENT"}
        for sid in ("r1", "r2")
    ]
    plan = {
        "required_slots": slots, "operations": [], "max_memories": 4,
        "budget_tier": "MEDIUM", "query_spec": {"requires_inference": True},
    }
    run = {
        "query_tokens": {}, "plan": plan, "replan": {**plan, "required_slots": [slots[0]]},
        "planner_called": True, "replan_called": True, "slot_validation": [],
        "trace": [
            {"retrieval_round": 1, "operation_index": 0, "operation": "SEMANTIC_SEARCH",
             "produces": ["r2"], "output_ids": ["m2", "m3"]},
            {"retrieval_round": 2, "operation_index": 0, "operation": "SEMANTIC_SEARCH",
             "produces": ["r1"], "output_ids": ["m1", "m4"]},
        ],
        "operation_output_ids": {m["id"] for m in candidates},
        "operation_candidates": candidates, "evidence_refs": [],
        "slot_support": {"r1": ["m1"], "r2": ["m2"]},
        "slot_coverage": {"r1": True, "r2": True},
        "requirement_status": {"r1": "FOUND", "r2": "FOUND"},
        "relation_status": {}, "retrieval_complete": True,
        "relations": [], "beliefs": [first, second],
    }
    monkeypatch.setattr(agent, "_run_query_retrieval", lambda *_a, **_k: deepcopy(run))

    prepared = QueryMixin.prepare_batch_query(agent, "Explain the decision using the observations.")

    assert set(prepared["extra"]["final_memory_ids"]) == {"m1", "m2", "m3", "m4"}
    provenance = {item["memory_id"]: item for item in prepared["extra"]["retrieval_provenance"]}
    assert provenance["m3"]["slot_ids"] == ["r2"]
    assert provenance["m4"]["slot_ids"] == ["r1"]
    assert provenance["m3"]["retrieval_round"] == 1
    assert provenance["m4"]["retrieval_round"] == 2
    assert not prepared["extra"]["boundary_violation"]
    assert not prepared["extra"]["arbitration_expansion_violation"]
