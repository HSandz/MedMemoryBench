"""Compile minimal semantic requirements into deterministic read operations."""

from .contracts import RETRIEVAL_BUDGETS


class ReadPlanContractMixin:
    @staticmethod
    def _ir_relation_types(ir):
        return {
            str(relation.get("type") or "").upper()
            for relation in ir.get("relations") or []
        }

    @staticmethod
    def _ir_relations_for(requirement_id, ir):
        return [
            relation
            for relation in ir.get("relations") or []
            if relation.get("from") == requirement_id
            or relation.get("to") == requirement_id
        ]

    def _derive_compiled_mode(self, ir):
        """Derive telemetry only; this value is not semantic source of truth."""
        if ir.get("visible_options"):
            return "MULTI_OPTION"
        relation_types = self._ir_relation_types(ir)
        requirements = ir.get("requirements") or []
        if "CAUSES" in relation_types or "POSSIBLE_CAUSE" in relation_types:
            return "CAUSAL"
        if "TEMPORAL_ORDER" in relation_types:
            return "TEMPORAL"
        if "COMPARE" in relation_types:
            return "COMPARISON"
        if any(
            (requirement.get("time_constraint") or {}).get("relation")
            or (requirement.get("time_constraint") or {}).get("axis")
            for requirement in requirements
        ):
            return "TEMPORAL"
        if "CURRENT" in relation_types:
            return "STATE"
        if len(requirements) > 1 or "DEPENDS_ON" in relation_types:
            return "MULTI_HOP"
        return "DIRECT"

    def _requirement_slot(self, requirement, ir, compiled_mode):
        requirement_id = requirement["id"]
        time_constraint = requirement.get("time_constraint") or {}
        axis = str(time_constraint.get("axis") or "")
        relation = str(time_constraint.get("relation") or "")
        relations = self._ir_relations_for(requirement_id, ir)
        relation_types = {str(item.get("type") or "").upper() for item in relations}
        relative_order = next(
            (
                item
                for item in relations
                if item.get("type") == "TEMPORAL_ORDER"
                and item.get("from") == requirement_id
            ),
            None,
        )
        slot_type = "DIRECT"
        if axis:
            slot_type = "TEMPORAL"
        elif relative_order:
            slot_type = "TEMPORAL"
            axis = "event_time"
            relation = str(relative_order.get("relation") or "")
        elif "CURRENT" in relation_types:
            slot_type = "CURRENT_STATE"

        side_label = None
        comparison = next(
            (
                item
                for item in ir.get("relations") or []
                if item.get("type") == "COMPARE"
                and requirement_id in {item.get("from"), item.get("to")}
            ),
            None,
        )
        if comparison:
            side_label = "LEFT" if comparison.get("from") == requirement_id else "RIGHT"

        focus = str(requirement.get("focus_span") or "").strip()
        hint = str(requirement.get("retrieval_hint") or "").strip()
        subject_id = str(ir.get("_resolved_subject_id") or "")
        return {
            "id": requirement_id,
            "type": slot_type,
            "evidence_role": "REQUIREMENT",
            "description": hint or focus or "question evidence",
            "required": True,
            "resolution_strategy": "RETRIEVE",
            "subject": subject_id,
            "subject_id": subject_id,
            "target_surface": focus,
            "retrieval_hint": hint,
            "resolved_keys": self._rc_resolve_target_keys(focus, subject_id),
            "side_label": side_label,
            "time_axis": axis,
            "time_relation": relation,
            "time_anchor": str(time_constraint.get("anchor") or ""),
            "time_end": str(time_constraint.get("end") or ""),
            "required_fields": [axis] if axis else [],
            "temporal_relation": relation,
            "relative_to_requirement": (
                str(relative_order.get("to") or "") if relative_order else ""
            ),
            "semantic_relation_types": sorted(relation_types),
            "option_labels": list((ir.get("visible_options") or {}).keys()),
        }

    def _controller_plan(self, ir, question, frame):
        del frame
        compiled_mode = self._derive_compiled_mode(ir)
        answer_type = (
            "OPTION_SET"
            if ir.get("visible_options")
            else ir.get("answer_type") or "TEXT"
        )
        slots = [
            self._requirement_slot(requirement, ir, compiled_mode)
            for requirement in ir.get("requirements") or []
        ]
        relation_types = self._ir_relation_types(ir)
        physical_cost = len(slots)
        physical_cost += sum(
            1
            for slot in slots
            if slot.get("type") == "TEMPORAL"
            and slot.get("time_relation") in {"EARLIEST", "LATEST"}
        )
        physical_cost += int("CAUSES" in relation_types)
        physical_cost += int("TEMPORAL_ORDER" in relation_types)
        if ir.get("visible_options") or physical_cost >= 4:
            tier = "LARGE"
        elif physical_cost >= 2 or relation_types:
            tier = "MEDIUM"
        else:
            tier = "SMALL"

        inferential = bool(
            relation_types.intersection(
                {
                    "COMPARE",
                    "CAUSES",
                    "POSSIBLE_CAUSE",
                    "DEPENDS_ON",
                    "TEMPORAL_ORDER",
                    "INFER",
                }
            )
        )
        world_bridge = bool(relation_types.intersection({"POSSIBLE_CAUSE", "INFER"}))
        need_source = "VERIFY_SOURCE" in relation_types
        plan = {
            "query_spec": {
                "answer_type": answer_type,
                "requires_inference": inferential,
                "world_knowledge_bridge_allowed": world_bridge,
                "semantic_ir_version": "minimal-v1",
            },
            # These are compiled telemetry labels, not controller outputs.
            "query_mode": compiled_mode,
            "compiled_mode": compiled_mode,
            "required_slots": slots,
            "semantic_relations": [dict(item) for item in ir.get("relations") or []],
            "seed_coverage": [],
            "operations": [],
            "need_evidence": need_source,
            "need_raw_evidence": need_source,
            "budget_tier": tier,
            "max_memories": RETRIEVAL_BUDGETS[tier]["max_memories"],
            "planner_fallback": False,
            "fallback_reason": "",
            "valid": bool(slots),
            "option_coverage": [],
            "visible_options": dict(ir.get("visible_options") or {}),
            "semantic_ir": {
                "answer_type": answer_type,
                "subject_span": ir.get("subject_span") or "",
                "requirements": [dict(item) for item in ir.get("requirements") or []],
                "relations": [dict(item) for item in ir.get("relations") or []],
            },
        }
        plan["operations"] = self._compile_gap_operations(
            slots, question, tier, plan=plan
        )
        return plan
