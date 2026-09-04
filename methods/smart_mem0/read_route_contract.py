"""Active retrieval strategy for the two-stage SmartMem0 read path.

The controller's semantic operator is an initial routing judgment, not an
immutable executor command. Once a query is in PLAN, execution follows the
normalized evidence contract. A failed/insufficient DIRECT proposal therefore
falls back to neutral focal retrieval instead of staying locked to a fast-path
label or being promoted to a fabricated MULTI_HOP plan.
"""

from .contracts import RETRIEVAL_BUDGETS


class ReadRouteContractMixin:
    def _controller_gaps(self, decision, question, frame):
        gaps = super()._controller_gaps(decision, question, frame)
        if str(decision.get("operator") or "").upper() == "MULTI_OPTION":
            # OPTION_CONTEXT is an exploration surface, not a participant fact
            # that must exist. Real shared requirements (FOCAL_STATE,
            # CONSTRAINT, etc.) remain required; a synthetic option-context-only
            # fallback must never make an MC query insufficient merely because
            # some choices have no personal-memory hit.
            if gaps and all(str(gap.role).upper() == "OPTION_CONTEXT" for gap in gaps):
                for gap in gaps:
                    gap.required = False
        return gaps

    def _controller_plan(self, decision, question, frame):
        plan = super()._controller_plan(decision, question, frame)
        initial_route = str(decision.get("operator") or "DIRECT").upper()
        semantic_mode = str(plan.get("query_mode") or initial_route).upper()
        slot_types = {str(slot.get("type") or "DIRECT").upper() for slot in plan.get("required_slots", [])}

        active_strategy = semantic_mode
        route_lock_released = False
        if semantic_mode == "DIRECT" and str(decision.get("route") or "PLAN").upper() == "PLAN":
            # DIRECT is only the cheap first attempt. Once planning is required,
            # typed requirements/constraints own execution. A temporal slot can
            # select its temporal primitive; otherwise use neutral focal recall.
            active_strategy = "TEMPORAL" if "TEMPORAL" in slot_types else "FOCAL"
            route_lock_released = True
            # A neutral recovery from a failed fast path needs enough room to
            # expose competing evidence, but it is still a single bounded plan.
            if plan.get("budget_tier") == "SMALL":
                plan["budget_tier"] = "MEDIUM"
                plan["max_memories"] = RETRIEVAL_BUDGETS["MEDIUM"]["max_memories"]

        plan["initial_route"] = initial_route
        plan["active_strategy"] = active_strategy
        plan["route_lock_released"] = route_lock_released
        spec = plan.setdefault("query_spec", {})
        spec["initial_route"] = initial_route
        spec["active_strategy"] = active_strategy
        return plan
