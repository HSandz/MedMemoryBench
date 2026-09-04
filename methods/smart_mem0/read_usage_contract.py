"""Method-local LLM accounting for SmartMem0 reads.

Benchmark-level trackers also count evaluator judges. This telemetry records
only calls implied by the SmartMem0 prepared query so resource claims can be
audited without subtracting benchmark infrastructure calls afterward.
"""


class ReadUsageContractMixin:
    TWO_STAGE_MAX_LLM_CALLS = 2

    def prepare_batch_query(self, question, system_message=None, **kwargs):
        prepared = super().prepare_batch_query(
            question, system_message=system_message, **kwargs
        )
        extra = prepared.setdefault("extra", {})
        controller = extra.get("semantic_controller") or {}
        query_tokens = extra.get("query_tokens") or {}
        controller_calls = int(bool(controller.get("called")))

        # Count any middle-stage LLM activity explicitly. It should be zero in
        # the active two-stage architecture, but keeping it visible prevents a
        # future flag/config regression from being hidden by accounting.
        validation_events = extra.get("slot_validation") or []
        slot_validation_calls = sum(
            1
            for item in validation_events
            if isinstance(item, dict)
            and item.get("called")
            and not item.get("cache_hit")
        )
        legacy_gate_calls = int(
            not controller_calls and bool((extra.get("fast_gate") or {}).get("called"))
        )
        legacy_planner_calls = int(
            not controller_calls
            and bool(extra.get("planner_called"))
            and int(query_tokens.get("planner", 0) or 0) > 0
        )
        legacy_replan_calls = int(
            bool(extra.get("replan_called"))
            and int(query_tokens.get("replan", 0) or 0) > 0
        )
        middle_calls = (
            slot_validation_calls
            + legacy_gate_calls
            + legacy_planner_calls
            + legacy_replan_calls
        )

        # At this point all output-contract handling has already run. A
        # surviving precomputed answer therefore really skips the final model;
        # otherwise one final answer generation is required.
        answer_calls = 0 if str(prepared.get("precomputed_answer") or "").strip() else 1
        total_calls = controller_calls + middle_calls + answer_calls
        two_stage_active = bool(controller_calls)
        budget_violation = bool(
            two_stage_active
            and (middle_calls > 0 or total_calls > self.TWO_STAGE_MAX_LLM_CALLS)
        )

        # planner_called is retained for historical telemetry compatibility: in
        # the two-stage path it means a deterministic plan was compiled, not an
        # LLM planner call. Expose that distinction explicitly.
        extra["deterministic_plan_compiled"] = bool(
            controller_calls and extra.get("planner_called")
        )
        extra["planner_llm_called"] = bool(legacy_planner_calls)
        extra["two_stage_llm_budget"] = {
            "max_calls_per_query": self.TWO_STAGE_MAX_LLM_CALLS,
            "violation": budget_violation,
        }
        extra["method_llm_calls"] = {
            "controller": controller_calls,
            "middle": middle_calls,
            "slot_validation": slot_validation_calls,
            "legacy_gate": legacy_gate_calls,
            "legacy_planner": legacy_planner_calls,
            "legacy_replan": legacy_replan_calls,
            "answer": answer_calls,
            "total": total_calls,
            "two_stage_budget_violation": budget_violation,
            "excludes_evaluator_judges": True,
        }
        return prepared
