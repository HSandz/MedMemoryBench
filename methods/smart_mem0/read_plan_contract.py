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

    def _rc_gap_from_req(self, decision, req, fallback_id, frame=None):
        role = str(req.get("role") or "GENERIC_EVIDENCE")
        temporal = decision.get("temporal") or {}
        target_surface = req.get("target") or decision.get("target") or ""
        explicit_axis = req.get("temporal_axis") or ""
        explicit_relation = req.get("temporal_relation") or ""
        explicit_anchor = req.get("temporal_anchor") or ""
        explicit_end = req.get("temporal_end") or ""

        if role == "PRIOR_TRAJECTORY":
            axis, relation, anchor, end = explicit_axis, explicit_relation, explicit_anchor, explicit_end
        else:
            axis = explicit_axis or temporal.get("axis") or ""
            relation = explicit_relation or temporal.get("relation") or ""
            anchor = explicit_anchor or temporal.get("anchor") or ""
            end = explicit_end or temporal.get("end") or ""
            local_dates = []
            if target_surface:
                try:
                    local_dates = list((self._query_frame(target_surface).dates or ()))
                except Exception:
                    local_dates = []
            frame_dates = list((getattr(frame, "dates", ()) or ())) if frame is not None else []
            # Never invent event_time/document_time merely because a date is
            # present. If the controller supplied an axis, attach the date to
            # that axis; otherwise QueryFrame will enforce a broad hard-date
            # constraint over the existing write-time date fields.
            if axis and not anchor and len(local_dates) == 1:
                relation, anchor = relation or "EXACT", local_dates[0]
            elif axis and not anchor and len(frame_dates) == 1 and decision["operator"] not in {"TEMPORAL", "COMPARISON"}:
                relation, anchor = relation or "EXACT", frame_dates[0]

        return self._rc_gap(
            str(req.get("id") or fallback_id),
            role,
            decision,
            target_surface=target_surface,
            side_label=req.get("side") or "",
            temporal_axis=axis,
            temporal_relation=relation,
            temporal_anchor=anchor,
            temporal_end=end,
        )

    def _controller_gaps(self, decision, question, frame):
        operator = decision["operator"]
        requirements = decision.get("requirements") or []

        if operator == "MULTI_OPTION":
            option_texts = {self._rc_text(text) for text in (decision.get("visible_options") or {}).values() if str(text).strip()}
            shared_roles = {"FOCAL_STATE", "PRIOR_TRAJECTORY", "ACTION_RULE", "CONSTRAINT", "OPTION_CONTEXT", "GENERIC_EVIDENCE"}
            shared = []
            for req in requirements:
                if req.get("role") not in shared_roles:
                    continue
                req_target = self._rc_text(req.get("target") or "")
                if req_target and any(req_target == option or req_target in option or option in req_target for option in option_texts):
                    continue
                shared.append(req)
            shared = shared[:3]
            if not shared:
                shared = [{"id": "r_options", "role": "OPTION_CONTEXT", "target": decision.get("target") or ""}]
            return [self._rc_gap_from_req(decision, req, f"g_option_ctx_{i}", frame=frame) for i, req in enumerate(shared)]

        if operator == "TEMPORAL":
            req = requirements[0] if requirements else {"id": "r_temporal", "role": "GENERIC_EVIDENCE", "target": decision.get("target") or ""}
            gap = self._rc_gap_from_req(decision, req, "g_temporal", frame=frame)
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
                gap = self._rc_gap_from_req(decision, req, f"g_{side.lower()}", frame=frame)
                gap.role, gap.side_label = "COMPARAND", side
                gaps.append(gap)
            return gaps

        if operator == "CAUSAL":
            roles = ["FOCAL_TRIGGER", "CAUSAL_BRIDGE", "OUTCOME"] if decision.get("causal_mode") == "STORED" else ["FOCAL_TRIGGER", "PRIOR_TRAJECTORY", "OUTCOME"]
            gaps = []
            for i, role in enumerate(roles):
                req = self._rc_req_by_role(requirements, role) or {"id": f"r_{i}", "role": role, "target": decision.get("target") or ""}
                gaps.append(self._rc_gap_from_req(decision, req, f"g_{i}", frame=frame))
            return gaps

        if operator == "DECISION":
            chosen = [req for req in requirements if req.get("role") in {"FOCAL_STATE", "CONSTRAINT", "ACTION_RULE", "PRIOR_TRAJECTORY"}][:4]
            if not chosen:
                chosen = [
                    {"id": "r_focal", "role": "FOCAL_STATE", "target": decision.get("target") or ""},
                    {"id": "r_constraint", "role": "CONSTRAINT", "target": decision.get("target") or ""},
                ]
            return [self._rc_gap_from_req(decision, req, f"g_{i}", frame=frame) for i, req in enumerate(chosen)]

        if operator == "MULTI_HOP":
            chosen = requirements[:4]
            if len(chosen) < 2:
                chosen = chosen or [{"id": "r1", "role": "ANSWER", "target": decision.get("target") or ""}]
            return [self._rc_gap_from_req(decision, req, f"g_{i}", frame=frame) for i, req in enumerate(chosen)]

        req = requirements[0] if requirements else {"id": "r1", "role": "GENERIC_EVIDENCE" if operator == "STATE" else "ANSWER", "target": decision.get("target") or ""}
        return [self._rc_gap_from_req(decision, req, "g_state" if operator == "STATE" else "g_direct", frame=frame)]

    def _controller_plan(self, decision, question, frame):
        lattice = EvidenceLattice()
        gaps = self._controller_gaps(decision, question, frame)
        effective_operator = decision["operator"]
        if effective_operator == "MULTI_HOP" and len(gaps) < 2:
            effective_operator = "DIRECT"
            decision = dict(decision)
            decision["operator"] = "DIRECT"
            decision["requires_inference"] = False
            for gap in gaps:
                gap.qrf_operator = "DIRECT"
        # A dated STATE with no known temporal axis must not resolve the global
        # current head. QueryFrame already knows the date and will constrain a
        # bounded semantic search across event/document/origin dates.
        if effective_operator == "STATE" and getattr(frame, "dates", ()) and not any(gap.temporal_axis for gap in gaps):
            effective_operator = "DIRECT"
            for gap in gaps:
                gap.qrf_operator = "DIRECT"
        for gap in gaps:
            lattice.add_gap(gap)
        self._evidence_lattice = lattice
        slots = lattice.to_legacy_slots()
        tier = "LARGE" if effective_operator in {"MULTI_OPTION", "CAUSAL", "MULTI_HOP"} or len(slots) >= 3 else ("MEDIUM" if effective_operator in {"DECISION", "COMPARISON"} or len(slots) >= 2 else "SMALL")
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
                "world_knowledge_bridge_allowed": any(
                    bool(requirement.get("world_knowledge_bridge"))
                    for requirement in decision.get("requirements") or []
                ),
            },
            "query_mode": effective_operator,
            "required_slots": slots,
            "seed_coverage": [],
            "operations": [],
            "need_evidence": bool(decision.get("need_raw_evidence", False)),
            "need_raw_evidence": bool(decision.get("need_raw_evidence", False)),
            "budget_tier": tier,
            "max_memories": RETRIEVAL_BUDGETS[tier]["max_memories"],
            "planner_fallback": False,
            "fallback_reason": "",
            "valid": True,
            "option_coverage": [],
            "visible_options": dict(decision["visible_options"]),
        }
        seeds = list(decision.get("_seed_candidates") or [])[:3]
        requirements = {
            str(requirement.get("id") or ""): requirement
            for requirement in decision.get("requirements") or []
        }
        seed_by_ref = {f"$seed{index}": memory for index, memory in enumerate(seeds)}
        seed_by_id = {memory["id"]: memory for memory in seeds}
        seed_ids = set(seed_by_id)
        seed_relations = [
            relation
            for relation in self._relations
            if relation.get("source_id") in seed_ids
            and relation.get("target_id") in seed_ids
        ]

        missing_slots = []
        for slot in slots:
            requirement = requirements.get(str(slot.get("id") or ""), {})
            slot["world_knowledge_bridge"] = bool(
                requirement.get("world_knowledge_bridge", False)
            )
            refs = list(requirement.get("support_refs") or [])
            if requirement.get("coverage") != "COVERED":
                missing_slots.append(slot)
                continue
            if effective_operator == "MULTI_OPTION" or str(slot.get("time_relation") or "").upper() in {"EARLIEST", "LATEST"}:
                missing_slots.append(slot)
                continue
            memories = [seed_by_ref[ref] for ref in refs if ref in seed_by_ref]
            if len(memories) != len(refs) or not memories:
                missing_slots.append(slot)
                continue
            if any(not self._memory_satisfies_frame(memory, frame) for memory in memories):
                missing_slots.append(slot)
                continue

            memory_ids = list(dict.fromkeys(memory["id"] for memory in memories))
            slot["controller_seed_ids"] = memory_ids
            if not self._slot_covered(slot, memory_ids, memories, seed_relations):
                slot.pop("controller_seed_ids", None)
                missing_slots.append(slot)
                continue
            plan["seed_coverage"].append(
                {"slot_id": slot["id"], "refs": list(dict.fromkeys(refs))}
            )

        plan["operations"] = self._compile_gap_operations(
            missing_slots, question, tier, plan=plan
        )
        plan["controller_coverage"] = {
            "covered_requirement_ids": [
                item["slot_id"] for item in plan["seed_coverage"]
            ],
            "missing_requirement_ids": [slot["id"] for slot in missing_slots],
            "covered_count": len(plan["seed_coverage"]),
            "missing_count": len(missing_slots),
        }
        return plan
