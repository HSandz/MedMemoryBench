"""Two-stage-only LLM accounting for SmartMem0 READ.

The active architecture permits exactly one semantic-controller call and, unless a strict
terminal certificate closes the query, one final answer call. Legacy gate/planner/replan
activity is a contract violation rather than an alternative accounting branch.
"""


class ReadUsageContractMixin:
    TWO_STAGE_MAX_LLM_CALLS = 2
    MIDDLE_TOKEN_STAGES = ("fast_gate", "planner", "slot_validation", "replan")

    def prepare_batch_query(self, question, system_message=None, **kwargs):
        prepared = super().prepare_batch_query(
            question, system_message=system_message, **kwargs
        )
        extra = prepared.setdefault("extra", {})
        controller = extra.get("semantic_controller") or {}
        query_tokens = extra.get("query_tokens") or {}
        controller_calls = int(bool(controller.get("called")))
        terminal = prepared.get("precomputed_answer") not in (None, "")
        answer_calls = 0 if terminal else 1

        middle_tokens = {
            stage: int(query_tokens.get(stage, 0) or 0)
            for stage in self.MIDDLE_TOKEN_STAGES
        }
        validation_events = extra.get("slot_validation") or []
        validation_calls = sum(
            1
            for item in validation_events
            if isinstance(item, dict)
            and item.get("called")
            and not item.get("cache_hit")
        )
        middle_activity = bool(any(middle_tokens.values()) or validation_calls)
        total_calls = controller_calls + answer_calls + validation_calls
        budget_violation = bool(
            controller_calls != 1
            or middle_activity
            or total_calls > self.TWO_STAGE_MAX_LLM_CALLS
        )

        extra["precomputed_answer_present"] = terminal
        extra["answer_llm_called"] = False
        extra["answer_llm_planned"] = bool(answer_calls)
        extra["direct_generation_violation"] = False
        extra["deterministic_plan_compiled"] = bool(extra.get("planner_called"))
        extra["planner_llm_called"] = False
        extra["two_stage_llm_budget"] = {
            "max_calls_per_query": self.TWO_STAGE_MAX_LLM_CALLS,
            "violation": budget_violation,
        }
        extra["method_llm_calls"] = {
            "controller": controller_calls,
            "middle": validation_calls,
            "answer": answer_calls,
            "total": total_calls,
            "two_stage_budget_violation": budget_violation,
            "middle_token_stages": middle_tokens,
            "excludes_evaluator_judges": True,
        }
        return prepared
