"""Resolution-aware proof admission and context preservation for RequirementGraph READ.

This layer repairs the boundary between premise proof and final-answer resolution.  A
requirement is a participant-memory fact; the goal projection is the shape of the final
answer.  They coincide only for a single direct requirement.  Proof coverage therefore
must not be used as permission to collapse competing answer-bearing memories unless a
selector or identical projected surface makes that collapse deterministic.

The runtime remains domain-, language-, dataset- and benchmark-agnostic. Retrieval aliases,
resolver keys, material anchors and retrieval scores may admit/rank candidates, but never
become truth.  Candidate-world proof is widened only by strict certification over memories
that were actually retrieved and are hard-eligible for the requirement.
"""

from copy import deepcopy

from .contracts import VALID_TEMPORAL_AXES


class ReadRequirementResolutionRuntimeMixin:
    REQUIREMENT_GRAPH_RUNTIME_VERSION = "requirement-graph-runtime-v4"
    REQUIREMENT_RESOLUTION_VERSION = "requirement-resolution-v1"
    AMBIGUOUS_PROOF_RESERVATION = 3

    def _requirement_slot(self, requirement, ir, compiled_mode):
        slot = super()._requirement_slot(requirement, ir, compiled_mode)
        goal_projection = str(
            slot.get("requested_projection")
            or ir.get("answer_type")
            or (ir.get("goal") or {}).get("projection")
            or "TEXT"
        ).upper()
        requirement_count = len(ir.get("requirements") or [])

        # Final-answer projection is not automatically the value type of every premise.
        # Candidate selection and multi-premise reasoning need participant facts, not one
        # OPTION_SET/TEXT artifact per requirement.
        premise_projection = (
            "FACT"
            if goal_projection == "OPTION_SET" or requirement_count > 1
            else goal_projection
        )
        slot["goal_projection"] = goal_projection
        slot["premise_projection"] = premise_projection
        return slot

    def _rg_requested_projection(self, slot):
        projection = str(slot.get("premise_projection") or "").upper()
        return projection or super()._rg_requested_projection(slot)

    def _rg_projection_evidence(self, slot, memory):
        if self._rg_requested_projection(slot) != "FACT":
            return super()._rg_projection_evidence(slot, memory)

        fields, values = [], []

        def add(field, value):
            text = " ".join(str(value or "").replace("_", " ").split())
            if text and text.casefold() not in {item.casefold() for item in values}:
                fields.append(field)
                values.append(text)

        # FACT means "this memory can supply a participant premise".  It deliberately
        # excludes retrieval metadata and ranking surfaces.
        add("value", memory.get("value"))
        add("verbatim_value", memory.get("verbatim_value"))
        add("object_anchor", memory.get("object_anchor"))
        add("claim", memory.get("claim"))
        if not values:
            for axis in VALID_TEMPORAL_AXES:
                add(axis, self._date_for(memory, axis))
        return {
            "projection": "FACT",
            "capable": bool(values),
            "fields": fields,
            "values": values,
        }

    def _rg_material_topic_surfaces(self, memory):
        """Canonical materiality surfaces with owner/role metadata removed."""
        raw = [
            memory.get("scope"),
            memory.get("state_key"),
            memory.get("object_anchor"),
            memory.get("evidence_family"),
            *(memory.get("entities") or []),
            *(memory.get("scope_entities") or []),
            *(memory.get("planning_tags") or []),
        ]
        non_topic = [
            memory.get("subject_id"),
            memory.get("subject"),
            memory.get("subject_class"),
            memory.get("semantic_role"),
            memory.get("kind"),
            *(memory.get("source_speakers") or []),
        ]
        normalize = getattr(
            self,
            "_rc_text",
            lambda value: " ".join(str(value or "").lower().split()),
        )
        blocked = {normalize(value) for value in non_topic if normalize(value)}
        owner_fn = getattr(self, "_rc_owner", None)
        memory_owner = (
            owner_fn(memory.get("subject_id") or memory.get("subject") or "")
            if callable(owner_fn)
            else ""
        )
        output, seen = [], set()
        for value in raw:
            text = " ".join(str(value or "").replace("_", " ").split()).strip()
            key = normalize(text)
            if not text or not key or key in blocked or key in seen:
                continue
            if callable(owner_fn) and memory_owner:
                try:
                    if owner_fn(text) == memory_owner:
                        continue
                except Exception:
                    pass
            output.append(text)
            seen.add(key)
        return output[:24]

    def _rg_material_binding(self, slot, memory, frame=None):
        """Do not let an owner/role anchor alone create a STRONG material lane."""
        binding = dict(super()._rg_material_binding(slot, memory, frame=frame) or {})
        if binding.get("level") != "STRONG":
            return binding
        if binding.get("proof") or binding.get("resolved_key"):
            return binding

        normalize = getattr(
            self,
            "_rc_text",
            lambda value: " ".join(str(value or "").lower().split()),
        )
        owner_fn = getattr(self, "_rc_owner", None)
        memory_owner = (
            owner_fn(memory.get("subject_id") or memory.get("subject") or "")
            if callable(owner_fn)
            else ""
        )
        metadata = {
            normalize(value)
            for value in [
                memory.get("subject_id"),
                memory.get("subject"),
                memory.get("subject_class"),
                memory.get("semantic_role"),
                memory.get("kind"),
                *(memory.get("source_speakers") or []),
            ]
            if normalize(value)
        }
        anchors = []
        for anchor in [
            *(slot.get("material_anchors") or []),
            *(slot.get("resolved_keys") or []),
        ]:
            text = " ".join(str(anchor or "").replace("_", " ").split()).strip()
            key = normalize(text)
            if not text or not key or key in metadata:
                continue
            if callable(owner_fn) and memory_owner:
                try:
                    if owner_fn(text) == memory_owner:
                        continue
                except Exception:
                    pass
            if key not in {normalize(item) for item in anchors}:
                anchors.append(text)

        surfaces = self._rg_material_topic_surfaces(memory)
        similarity = getattr(self, "_rq_surface_similarity", None)
        exact_hits = 0
        scores = []
        for anchor in anchors:
            anchor_key = normalize(anchor)
            best = 0.0
            for surface in surfaces:
                surface_key = normalize(surface)
                if anchor_key and surface_key and (
                    anchor_key == surface_key
                    or (len(anchor_key) >= 4 and anchor_key in surface_key)
                    or (len(surface_key) >= 4 and surface_key in anchor_key)
                ):
                    score = 1.0
                    exact_hits += 1
                else:
                    score = (
                        float(similarity(anchor, surface))
                        if callable(similarity)
                        else 0.0
                    )
                best = max(best, score)
            scores.append(best)
        high_hits = sum(score >= 0.82 for score in scores)
        medium_hits = sum(score >= 0.65 for score in scores)
        topic_strong = bool(
            exact_hits > 0
            or high_hits >= 2
            or (
                high_hits >= 1
                and int(binding.get("views") or 0) >= 2
                and float(binding.get("need") or 0.0) >= 0.35
            )
        )
        if topic_strong:
            binding["topic_anchor_guard"] = "PASS"
            binding["topic_anchor_exact_hits"] = exact_hits
            binding["topic_anchor_high_hits"] = high_hits
            return binding

        local_retrieval = bool(binding.get("local_retrieval"))
        plausible = local_retrieval and (
            medium_hits > 0
            or int(binding.get("views") or 0) >= 2
            or float(binding.get("need") or 0.0) >= 0.55
        )
        binding["level"] = "PLAUSIBLE" if plausible else "NONE"
        binding["topic_anchor_guard"] = "DOWNGRADED_OWNER_OR_ROLE_ONLY"
        binding["topic_anchor_exact_hits"] = exact_hits
        binding["topic_anchor_high_hits"] = high_hits
        return binding

    def _rg_temporal_support_value(self, slot, memory):
        selector = slot.get("selector") if isinstance(slot.get("selector"), dict) else {}
        axis = str(selector.get("axis") or slot.get("time_axis") or "").lower()
        if axis not in VALID_TEMPORAL_AXES:
            return ""
        return " ".join(str(self._date_for(memory, axis) or "").split())

    def _rg_order_proof_ids(self, slot, proof_ids, selected_by_id):
        proof_ids = [
            str(memory_id)
            for memory_id in proof_ids or []
            if str(memory_id) in selected_by_id
        ]
        if not proof_ids:
            return []
        original_index = {memory_id: index for index, memory_id in enumerate(proof_ids)}
        relation = str(self._rg_selector_relation(slot) or "").upper()

        def semantic_key(memory_id):
            return self._rg_answerability_key(
                slot,
                selected_by_id[memory_id],
                original_index.get(memory_id, 10**9),
            )

        if relation in {"EARLIEST", "LATEST"}:
            dated = [
                memory_id
                for memory_id in proof_ids
                if self._rg_temporal_support_value(slot, selected_by_id[memory_id])
            ]
            undated = [memory_id for memory_id in proof_ids if memory_id not in dated]
            # Stable two-pass ordering keeps semantic quality ascending inside equal
            # timestamps while reversing only the temporal key for LATEST.
            dated.sort(key=semantic_key)
            dated.sort(
                key=lambda memory_id: self._rg_temporal_support_value(
                    slot, selected_by_id[memory_id]
                ),
                reverse=(relation == "LATEST"),
            )
            undated.sort(key=semantic_key)
            return dated + undated

        return sorted(proof_ids, key=semantic_key)

    def _retrieval_status(self, plan, slot_support, selected, relations):
        """Strictly certify the whole retrieved candidate world, not only an early lane."""
        selected_by_id = {
            str(memory.get("id") or ""): memory
            for memory in selected or []
            if memory and memory.get("id")
        }
        semantic_slot = getattr(self, "_semantic_context_slot", None)
        eligible = getattr(self, "_context_candidate_eligible", None)
        structure = getattr(self, "_slot_structure_covered", None)
        for slot in (plan or {}).get("required_slots") or []:
            if callable(semantic_slot) and not semantic_slot(slot):
                continue
            rid = str(slot.get("id") or "")
            if not rid:
                continue
            certified = []
            for memory_id, memory in selected_by_id.items():
                if callable(eligible) and not eligible(slot, memory):
                    continue
                if not self._requirement_target_proof(slot, memory):
                    continue
                if callable(structure) and not structure(
                    slot, [memory_id], [memory], relations
                ):
                    continue
                certified.append(memory_id)
            if certified:
                slot_support[rid] = self._rg_order_proof_ids(
                    slot, certified, selected_by_id
                )

        return super()._retrieval_status(plan, slot_support, selected, relations)

    def _rg_proof_lane(self, slot, proof_ids, candidate_order):
        allowed = set(str(memory_id) for memory_id in candidate_order or [])
        selected_by_id = {
            memory_id: self._alignment_memory(memory_id)
            for memory_id in allowed
            if self._alignment_memory(memory_id)
        }
        ordered = self._rg_order_proof_ids(
            slot,
            [memory_id for memory_id in proof_ids or [] if memory_id in allowed],
            selected_by_id,
        )
        if not ordered:
            return {"resolved": False, "reason": "NO_PROOF", "ids": []}

        relation = str(self._rg_selector_relation(slot) or "").upper()
        if relation in {"EARLIEST", "LATEST"}:
            first = ordered[0]
            if self._rg_temporal_support_value(slot, selected_by_id[first]):
                return {
                    "resolved": True,
                    "reason": relation,
                    "ids": [first],
                }

        signatures = []
        for memory_id in ordered:
            signature = self._rg_answer_signature(slot, selected_by_id[memory_id])
            if signature and signature not in signatures:
                signatures.append(signature)
        if len(signatures) <= 1:
            return {
                "resolved": True,
                "reason": "SAME_PROJECTED_SURFACE",
                "ids": [ordered[0]],
            }
        return {
            "resolved": False,
            "reason": "COMPETING_PROOF_SURFACES",
            "ids": ordered[: self.AMBIGUOUS_PROOF_RESERVATION],
        }

    def _role_aware_support_ids(self, slots, slot_support, candidate_order, limit):
        baseline = list(
            super()._role_aware_support_ids(
                slots, slot_support, candidate_order, limit
            )
        )
        bounded_limit = max(0, int(limit))
        if not bounded_limit:
            return baseline

        semantic_slot = getattr(self, "_semantic_context_slot", None)
        proof_map = getattr(self, "_last_requirement_proof_support", {}) or {}
        lanes = {}
        for slot in slots or []:
            if callable(semantic_slot) and not semantic_slot(slot):
                continue
            rid = str(slot.get("id") or "")
            if not rid:
                continue
            lane = self._rg_proof_lane(
                slot,
                proof_map.get(rid) or (slot_support or {}).get(rid) or [],
                candidate_order,
            )
            if lane.get("ids"):
                lanes[rid] = (slot, lane)

        # No strict proof lane: v3's unresolved answerability reservation remains owner.
        if not lanes:
            return baseline

        selected = []

        def add(memory_id):
            if (
                memory_id
                and memory_id in set(candidate_order or [])
                and memory_id not in selected
                and len(selected) < bounded_limit
            ):
                selected.append(memory_id)
                return True
            return False

        # Coverage first: one strict proof support per requirement.
        for _, lane in lanes.values():
            if lane.get("ids"):
                add(lane["ids"][0])

        # Then preserve competing proof surfaces round-robin.  They are not redundant until
        # a selector or identical projected answer makes them substitutable.
        for depth in range(1, self.AMBIGUOUS_PROOF_RESERVATION):
            for _, lane in lanes.values():
                ids = lane.get("ids") or []
                if depth < len(ids):
                    add(ids[depth])
                if len(selected) >= bounded_limit:
                    break
            if len(selected) >= bounded_limit:
                break

        for memory_id in baseline:
            add(memory_id)
            if len(selected) >= bounded_limit:
                break

        telemetry = dict(getattr(self, "_last_query_memory_alignment", {}) or {})
        telemetry.update(
            {
                "selected_ids": list(selected),
                "selection_semantics": "proof_resolution_then_coverage_then_competing_surface_preservation",
                "requirement_resolution": {
                    rid: {
                        "resolved": bool(lane.get("resolved")),
                        "reason": str(lane.get("reason") or ""),
                        "reserved_ids": list(lane.get("ids") or []),
                    }
                    for rid, (_, lane) in lanes.items()
                },
                "proof_semantics": "coverage_is_not_answer_resolution",
            }
        )
        self._last_query_memory_alignment = telemetry
        return selected[:bounded_limit]

    def prepare_batch_query(self, question, system_message=None, **kwargs):
        prepared = super().prepare_batch_query(
            question, system_message=system_message, **kwargs
        )
        extra = prepared.setdefault("extra", {})
        extra["requirement_graph_runtime_version"] = self.REQUIREMENT_GRAPH_RUNTIME_VERSION
        extra["requirement_resolution_contract"] = {
            "version": self.REQUIREMENT_RESOLUTION_VERSION,
            "semantics": "premise_proof_is_not_final_answer_resolution",
            "goal_projection_is_requirement_projection": False,
            "ambiguous_proof_reservation": self.AMBIGUOUS_PROOF_RESERVATION,
        }
        return prepared
