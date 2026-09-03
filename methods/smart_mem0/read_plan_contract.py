"""Minimal semantic requirement -> EvidenceGap plan normalization."""

from .contracts import RETRIEVAL_BUDGETS
from .p1b_execution import EvidenceGap, EvidenceLattice


class ReadPlanContractMixin:
    def _rc_gap(self, gid, role, decision, target_surface="", side_label="", **kwargs):
        subject_id = decision.get("subject_id") or ""
        target_surface = str(target_surface or decision.get("target") or "").strip()
        gap = EvidenceGap(
            id=gid,
            role=role,
            required=True,
            subject_id=subject_id,
            target_surface=target_surface,
            resolved_keys=self._rc_resolve_target_keys(target_surface, subject_id),
            temporal_axis=str(kwargs.pop("temporal_axis", "") or ""),
            temporal_relation=str(kwargs.pop("temporal_relation", "") or ""),
            temporal_anchor=str(kwargs.pop("temporal_anchor", "") or ""),
            temporal_end=str(kwargs.pop("temporal_end", "") or ""),
            side_label=str(side_label or ""),
        )
        gap.qrf_operator = decision["operator"]
        gap.description = target_surface or f"{role} evidence"
        gap.required_fields = [gap.temporal_axis] if gap.temporal_axis else []
        return gap

    @staticmethod
    def _rc_req_by_role(requirements, role):
        return next((req for req in requirements if req.get("role") == role), None)

    def _rc_gap_from_req(self, decision, req, fallback_id):
        temporal = decision.get("temporal") or {}
        return self._rc_gap(
            str(req.get("id") or fallback_id),
            str(req.get("role") or "GENERIC_EVIDENCE"),
            decision,
            target_surface=req.get("target") or decision.get("target") or "",
            side_label=req.get("side") or "",
            temporal_axis=req.get("temporal_axis") or (temporal.get("axis") if decision["operator"] == "TEMPORAL" else "") or "",
            temporal_relation=req.get("temporal_relation") or (temporal.get("relation") if decision["operator"] == "TEMPORAL" else "") or "",
            temporal_anchor=req.get("temporal_anchor") or (temporal.get("anchor") if decision["operator"] == "TEMPORAL" else "") or "",
            temporal_end=req.get("temporal_end") or (temporal.get("end") if decision["operator"] == "TEMPORAL" else "") or "",
        )

    def _controller_gaps(self, decision, question, frame):
        del frame
        operator = decision["operator"]
        requirements = decision.get("requirements") or []

        if operator == "MULTI_OPTION":
            shared = [req for req in requirements if req.get("role") != "COMPARAND"][:3]
            if not shared:
                shared = [{"id": "r_options", "role": "OPTION_CONTEXT", "target": decision.get("target") or ""}]
            return [self._rc_gap_from_req(decision, req, f"g_option_ctx_{i}") for i, req in enumerate(shared)]

        if operator == "TEMPORAL":
            req = requirements[0] if requirements else {"id": "r_temporal", "role": "GENERIC_EVIDENCE", "target": decision.get("target") or ""}
            gap = self._rc_gap_from_req(decision, req, "g_temporal")
            if not gap.temporal_relation:
                gap.temporal_relation = decision["temporal"].get("relation") or "LOCATE"
            if not gap.temporal_axis:
                gap.temporal_axis = decision["temporal"].get("axis") or "event_time"
            gap.required_fields = [gap.temporal_axis]
            return [gap]

        if operator == "COMPARISON":
            gaps = []
            for i, side in enumerate(("LEFT", "RIGHT")):
                req = requirements[i] if i < len(requirements) else {"id": f"r_{side.lower()}", "role": "COMPARAND", "side": side, "target": decision.get("target") or ""}
                gap = self._rc_gap_from_req(decision, req, f"g_{side.lower()}")
                gap.role = "COMPARAND"
                gap.side_label = side
                gaps.append(gap)
            return gaps

        if operator == "CAUSAL":
            roles = ["FOCAL_TRIGGER", "CAUSAL_BRIDGE", "OUTCOME"] if decision.get("causal_mode") == "STORED" else ["FOCAL_TRIGGER", "PRIOR_TRAJECTORY", "OUTCOME"]
            gaps = []
            for i, role in enumerate(roles):
                req = self._rc_req_by_role(requirements, role) or {"id": f"r_{i}", "role": role, "target": decision.get("target") or ""}
                gaps.append(self._rc_gap_from_req(decision, req, f"g_{i}"))
            return gaps

        if operator == "DECISION":
            chosen = [req for req in requirements if req.get("role") in {"FOCAL_STATE", "CONSTRAINT", "ACTION_RULE", "PRIOR_TRAJECTORY"}][:4]
            if not chosen:
                chosen = [
                    {"id": "r_focal", "role": "FOCAL_STATE", "target": decision.get("target") or ""},
                    {"id": "r_constraint", "role": "CONSTRAINT", "target": decision.get("target") or ""},
                ]
            return [self._rc_gap_from_req(decision, req, f"g_{i}") for i, req in enumerate(chosen)]

        if operator == "MULTI_HOP":
            chosen = requirements[:4]
            if not chosen:
                chosen = [
                    {"id": "r1", "role": "GENERIC_EVIDENCE", "target": decision.get("target") or ""},
                    {"id": "r2", "role": "PRIOR_TRAJECTORY", "target": decision.get("target") or ""},
                ]
            elif len(chosen) == 1:
                chosen.append({"id": "r2", "role": "PRIOR_TRAJECTORY", "target": decision.get("target") or chosen[0].get("target") or ""})
            return [self._rc_gap_from_req(decision, req, f"g_{i}") for i, req in enumerate(chosen)]

        req = requirements[0] if requirements else {"id": "r1", "role": "GENERIC_EVIDENCE" if operator == "STATE" else "ANSWER", "target": decision.get("target") or ""}
        return [self._rc_gap_from_req(decision, req, "g_state" if operator == "STATE" else "g_direct")]

    def _controller_plan(self, decision, question, frame):
        lattice = EvidenceLattice()
        gaps = self._controller_gaps(decision, question, frame)
        for gap in gaps:
            lattice.add_gap(gap)
        self._evidence_lattice = lattice
        slots = lattice.to_legacy_slots()
        tier = "LARGE" if decision["operator"] in {"MULTI_OPTION", "CAUSAL", "MULTI_HOP"} or len(slots) >= 3 else ("MEDIUM" if decision["operator"] in {"DECISION", "COMPARISON"} or len(slots) >= 2 else "SMALL")
        top_target = decision.get("target") or ""
        plan = {
            "query_spec": {
                "target_surface": top_target,
                "resolved_keys": self._rc_resolve_target_keys(top_target, decision.get("subject_id") or ""),
                "target_entities": [],
                "target_property": "",
                "answer_type": decision["answer_slot"],
                "reasoning": "SYNTHESIS" if decision["requires_inference"] else "NONE",
                "requires_inference": decision["requires_inference"],
                "temporal": dict(decision["temporal"]),
                "causal_mode": decision.get("causal_mode") or "",
            },
            "query_mode": decision["operator"],
            "required_slots": slots,
            "seed_coverage": [],
            "operations": [],
            "need_evidence": True,
            "budget_tier": tier,
            "max_memories": RETRIEVAL_BUDGETS[tier]["max_memories"],
            "planner_fallback": False,
            "fallback_reason": "",
            "valid": True,
            "option_coverage": [],
            "visible_options": dict(decision["visible_options"]),
        }
        plan["operations"] = self._compile_gap_operations(slots, question, tier, plan=plan)
        return plan
