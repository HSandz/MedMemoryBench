"""Control-flow compatibility regressions for the lean retrieval executor."""

from copy import deepcopy

from methods.smart_mem0.read_lean_execution_adapter import ReadLeanExecutionAdapterMixin


class _Base:
    def __init__(self):
        self.internal_plan = None
        self.operation_seen = None

    def _execute_plan(self, plan, seeds, *args, **kwargs):
        del seeds, args, kwargs
        self.internal_plan = deepcopy(plan)
        return {
            "trace": [
                {
                    "operation_index": index,
                    "operation": operation["op"],
                }
                for index, operation in enumerate(plan.get("operations") or [])
            ]
        }

    def _execute_operation(self, operation, outputs, seeds, frame):
        del outputs, seeds, frame
        self.operation_seen = deepcopy(operation)
        return [], [], []


class _Harness(ReadLeanExecutionAdapterMixin, _Base):
    pass


def test_public_plan_is_lean_but_internal_control_flow_sees_legacy_aliases():
    harness = _Harness()
    plan = {
        "operations": [
            {
                "op": "SEARCH_FAMILY",
                "family_mode": "temporal_extremum",
                "query": "dose",
                "produces": ["r1"],
            },
            {
                "op": "SELECT",
                "relation": "LATEST",
                "produces": ["r1"],
            },
            {
                "op": "VERIFY_SOURCE",
                "produces": ["r1"],
            },
        ]
    }
    before = deepcopy(plan)
    result = harness._execute_plan(plan, [])

    assert plan == before
    assert [item["op"] for item in harness.internal_plan["operations"]] == [
        "LOCATE_ANCHOR",
        "TEMPORAL_FILTER",
        "VERIFY_EVIDENCE",
    ]
    assert [item["operation"] for item in result["trace"]] == [
        "SEARCH_FAMILY",
        "SELECT",
        "VERIFY_SOURCE",
    ]


def test_internal_marker_restores_lean_primitive_for_physical_executor():
    harness = _Harness()
    operation = {
        "op": "LOCATE_ANCHOR",
        "_lean_op": "SEARCH_FAMILY",
        "family_mode": "temporal_extremum",
        "query": "dose",
        "produces": ["r1"],
    }
    harness._execute_operation(operation, [], [], None)
    assert harness.operation_seen["op"] == "SEARCH_FAMILY"
    assert "_lean_op" not in harness.operation_seen
