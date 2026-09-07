"""Reasoning Bridge contract for SmartMem0 READ.

A bridge never invents participant facts. It records grounded-memory endpoints, the
semantic relation still required by the question, whether general world knowledge is
authorized, and whether a stored relation must be followed. Graph expansion is bounded to
one hop.
"""

from copy import deepcopy


class ReadReasoningBridgeMixin:
    REASONING_BRIDGE_VERSION = "reasoning-bridge-v1"
    BRIDGE_TYPES = frozenset(
        {
            "COMPARE",
            "CAUSES",
            "POSSIBLE_CAUSE",
            "DEPENDS_ON",
            "TEMPORAL_ORDER",
            "INFER",
        }
    )

    @classmethod
    def _reasoning_bridges(cls, relations):
        output = []
        for relation in relations or []:
            relation_type = str(relation.get("type") or "").upper()
            if relation_type not in cls.BRIDGE_TYPES:
                continue
            bridge = {
                "type": relation_type,
                "from": str(relation.get("from") or ""),
                "to": str(relation.get("to") or ""),
                "world_knowledge_allowed": relation_type
                in {"POSSIBLE_CAUSE", "INFER"},
                "requires_stored_relation": relation_type == "CAUSES",
                "max_graph_hops": 1 if relation_type == "CAUSES" else 0,
            }
            order = str(relation.get("relation") or "").upper()
            if order:
                bridge["relation"] = order
            goal = " ".join(
                str(
                    relation.get("bridge_goal")
                    or relation.get("goal")
                    or ""
                ).split()
            )[:220]
            if goal:
                bridge["goal"] = goal
            output.append(bridge)
        return output

    def _controller_plan(self, ir, question, frame):
        plan = super()._controller_plan(ir, question, frame)
        bridges = self._reasoning_bridges(
            plan.get("semantic_relations")
            or ir.get("relations")
            or []
        )
        plan["reasoning_bridges"] = deepcopy(bridges)
        plan["reasoning_bridge_version"] = self.REASONING_BRIDGE_VERSION
        spec = plan.setdefault("query_spec", {})
        spec["world_knowledge_bridge_allowed"] = any(
            bridge["world_knowledge_allowed"] for bridge in bridges
        )
        spec["stored_relation_required"] = any(
            bridge["requires_stored_relation"] for bridge in bridges
        )
        plan.setdefault("semantic_ir", {})["reasoning_bridges"] = deepcopy(
            bridges
        )
        return plan

    def prepare_batch_query(self, question, system_message=None, **kwargs):
        prepared = super().prepare_batch_query(
            question, system_message=system_message, **kwargs
        )
        extra = prepared.setdefault("extra", {})
        source_plan = extra.get("plan") or {}
        extra["reasoning_bridge_version"] = self.REASONING_BRIDGE_VERSION
        extra["reasoning_bridges"] = deepcopy(
            source_plan.get("reasoning_bridges") or []
        )
        return prepared
