"""Active retrieval strategy for the two-stage SmartMem0 read path.

The controller's semantic operator is an initial routing judgment, not an
immutable executor command. Once a query is in PLAN, execution follows the
normalized evidence contract. A failed/insufficient DIRECT proposal therefore
falls back to neutral focal retrieval instead of staying locked to a fast-path
label or being promoted to a fabricated MULTI_HOP plan.
"""


class ReadRouteContractMixin:
    def _controller_plan(self, decision, question, frame):
        plan = super()._controller_plan(decision, question, frame)
        initial_route = str(decision.get("operator") or "DIRECT").upper()
        semantic_mode = str(plan.get("query_mode") or initial_route).upper()
        slot_types = {
            str(slot.get("type") or "DIRECT").upper()
            for slot in plan.get("required_slots", [])
        }

        # SHARED_OPTIONS must execute once for every visible label, but probe
        # completeness is telemetry/exploration, not proof that an option is
        # true. The OPTION_CONTEXT slot remains required so execution cannot
        # short-circuit before probing; ReadOptionContractMixin separately
        # requires actual participant evidence before marking it covered.
        if semantic_mode == "MULTI_OPTION":
            expected_labels = sorted(
                str(label) for label in (plan.get("visible_options") or {})
            )
            for slot in plan.get("required_slots", []):
                if str(slot.get("evidence_role") or "").upper() == "OPTION_CONTEXT":
                    slot["required"] = True
                    slot["option_labels"] = list(expected_labels)

        active_strategy = semantic_mode
        route_lock_released = False
        if (
            semantic_mode == "DIRECT"
            and str(decision.get("route") or "PLAN").upper() == "PLAN"
        ):
            # DIRECT is only the cheap first attempt. Once planning is required,
            # typed requirements/constraints own execution. A temporal slot can
            # select its temporal primitive; otherwise use neutral focal recall.
            # Keep the one-gap plan's original SMALL budget: failed authorization
            # is not evidence that a larger evidence bundle is necessary.
            active_strategy = "TEMPORAL" if "TEMPORAL" in slot_types else "FOCAL"
            route_lock_released = True

        plan["initial_route"] = initial_route
        plan["active_strategy"] = active_strategy
        plan["route_lock_released"] = route_lock_released
        spec = plan.setdefault("query_spec", {})
        spec["initial_route"] = initial_route
        spec["active_strategy"] = active_strategy
        return plan

    def _coverage_map(self, plan, slot_support, selected, relations):
        coverage = super()._coverage_map(plan, slot_support, selected, relations)
        # Optional gaps may enrich answer context but can never block execution
        # or trigger deterministic recovery.
        for slot in plan.get("required_slots", []):
            if slot.get("required") is False:
                coverage[slot["id"]] = True
        return coverage
