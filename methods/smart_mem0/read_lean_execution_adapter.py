"""Internal compatibility adapter for the lean retrieval program.

Public SmartMem0 plans expose only the Phase-3 retrieval primitives. ExecutionMixin still
contains a few bounded-control branches keyed by the historical operation names, so this
adapter supplies those aliases only inside execution. The public plan and returned trace
remain lean.
"""

from copy import deepcopy


class ReadLeanExecutionAdapterMixin:
    LEAN_EXECUTION_ADAPTER_VERSION = "lean-execution-adapter-v1"

    @staticmethod
    def _legacy_control_operation(operation):
        current = deepcopy(operation)
        lean = str(current.get("op") or "")
        current["_lean_op"] = lean
        if lean == "SEARCH_FAMILY":
            mode = str(current.get("family_mode") or "semantic")
            current["op"] = (
                "LOCATE_ANCHOR"
                if mode in {"anchor", "temporal_extremum"}
                else "SEMANTIC_SEARCH"
            )
        elif lean == "SELECT":
            current["op"] = "TEMPORAL_FILTER"
        elif lean == "EXPAND_RELATION":
            current["op"] = "FOLLOW_CAUSES"
        elif lean == "VERIFY_SOURCE":
            current["op"] = "VERIFY_EVIDENCE"
        return current

    def _execute_operation(self, operation, outputs, seeds, frame):
        lean = str(operation.get("_lean_op") or "")
        if not lean:
            return super()._execute_operation(operation, outputs, seeds, frame)
        public = deepcopy(operation)
        public["op"] = lean
        public.pop("_lean_op", None)
        return super()._execute_operation(public, outputs, seeds, frame)

    def _execute_plan(self, plan, seeds, *args, **kwargs):
        public_operations = [deepcopy(item) for item in plan.get("operations") or []]
        internal = deepcopy(plan)
        internal["operations"] = [
            self._legacy_control_operation(operation)
            for operation in public_operations
        ]
        result = super()._execute_plan(internal, seeds, *args, **kwargs)
        for item in result.get("trace") or []:
            try:
                index = int(item.get("operation_index", -1))
            except (TypeError, ValueError):
                index = -1
            if 0 <= index < len(public_operations):
                item["operation"] = str(public_operations[index].get("op") or "")
        result["execution_adapter_version"] = self.LEAN_EXECUTION_ADAPTER_VERSION
        return result
