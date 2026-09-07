"""Deterministic proof and bounded answer-context selection.

Proof and context are separate decisions. Context eligibility is deliberately broader than
proof: retrieval candidates must satisfy hard structural constraints, while semantic target
agreement only promotes or certifies them. This keeps future roles extensible without letting
weak language similarity become proof.
"""

from typing import Any, Dict, List, Sequence

from .contracts import VALID_TEMPORAL_AXES


class ProofContextContractMixin:
    CONTEXT_POOL_KEY = "__answer_context_candidates__"
    SEMANTIC_CONTEXT_ROLES = frozenset({"REQUIREMENT", "COMPARAND"})

    @classmethod
    def _semantic_context_slot(cls, slot: Dict[str, Any]) -> bool:
        return str(slot.get("evidence_role") or "").upper() in cls.SEMANTIC_CONTEXT_ROLES

    def _rc_memory_matches_target(self, slot, memory):
        target = str(slot.get("proof_anchor") or slot.get("target_surface") or "").strip()
        if not target:
            return False
        text = self._rc_memory_target_text(memory)
        if self._rc_token_sequence_present(target, text):
            return True
        similarity = getattr(self, "_rq_surface_similarity", None)
        return bool(callable(similarity) and similarity(target, text) >= 0.86)

    def _requirement_target_proof(self, slot, memory):
        if not self._semantic_context_slot(slot):
            return True
        if slot.get("degraded"):
            return False
        return self._rc_memory_matches_target(slot, memory)

    def _slot_covered(self, slot, support_ids, selected, relations):
        slot_type = str(slot.get("type") or "DIRECT").upper()
        if (
            not self._semantic_context_slot(slot)
            or slot_type not in {"DIRECT", "TEMPORAL", "CURRENT_STATE"}
        ):
            return super()._slot_covered(slot, support_ids, selected, relations)
        support_set = set(support_ids or [])
        proof_ids = [
            memory["id"]
            for memory in selected
            if memory.get("id") in support_set
            and self._requirement_target_proof(slot, memory)
        ]
        if not proof_ids:
            return False
        return super()._slot_covered(slot, proof_ids, selected, relations)

    def _relation_status_map(self, plan, slot_support, selected, relations):
        selected_by_id = {memory.get("id"): memory for memory in selected}
        filtered_support = {
            str(slot_id): list(memory_ids or [])
            for slot_id, memory_ids in (slot_support or {}).items()
        }
        for slot in plan.get("required_slots") or []:
            if not self._semantic_context_slot(slot):
                continue
            slot_id = str(slot.get("id") or "")
            filtered_support[slot_id] = [
                memory_id
                for memory_id in filtered_support.get(slot_id, [])
                if memory_id in selected_by_id
                and self._requirement_target_proof(slot, selected_by_id[memory_id])
            ]
        return super()._relation_status_map(
            plan, filtered_support, selected, relations
        )

    def _context_hard_eligible(self, slot, memory) -> bool:
        """Eligibility is broad recall with only schema-level hard exclusions.

        Target similarity, evidence-role semantics and certificate success are NOT
        eligibility predicates. They belong to promotion/proof. This is what lets a
        COMPARAND retain a correctly time-filtered memory even when its claim does not
        lexically restate the whole comparison target.
        """
        if not memory or not memory.get("id") or not self._memory_value(memory):
            return False
        if not self._rc_owner_match(slot, memory):
            return False
        if callable(getattr(self, "_query_visible_memory", None)):
            try:
                if not self._query_visible_memory(memory):
                    return False
            except Exception:
                pass
        for field in slot.get("required_fields") or []:
            if field in VALID_TEMPORAL_AXES:
                if not self._date_for(memory, field):
                    return False
            elif field and not memory.get(field):
                return False
        status = memory.get(
            "_status", self._belief_status.get(memory.get("id"), "active")
        )
        if not slot.get("history") and status == "superseded":
            return False
        return True

    def _context_candidate_eligible(self, slot, memory):
        if self._semantic_context_slot(slot):
            return self._context_hard_eligible(slot, memory)
        if not self._context_hard_eligible(slot, memory):
            return False
        return self._memory_matches_slot_role(slot, memory)

    def _context_resolved_key_match(self, slot, memory) -> bool:
        wanted = {
            self._rc_text(key)
            for key in (slot.get("resolved_keys") or [])
            if str(key or "").strip()
        }
        if not wanted:
            return False
        surfaces_fn = getattr(self, "_rq_memory_concept_surfaces", None)
        actual = {
            self._rc_text(key)
            for key in (
                surfaces_fn(memory)
                if callable(surfaces_fn)
                else self._rc_memory_concept_keys(memory)
            )
            if str(key or "").strip()
        }
        return bool(wanted & actual)

    def _context_surface_match(self, slot, memory) -> bool:
        target = str(
            slot.get("target_surface") or slot.get("retrieval_target") or ""
        ).strip()
        if not target:
            return False
        text = self._rc_memory_target_text(memory)
        if self._rc_token_sequence_present(target, text):
            return True
        similarity = getattr(self, "_rq_surface_similarity", None)
        return bool(callable(similarity) and similarity(target, text) >= 0.90)

    def _context_promotion_level(self, slot, memory) -> int:
        certificate_fn = getattr(self, "_certificate_result", None)
        if callable(certificate_fn):
            try:
                if certificate_fn(slot, memory)[0]:
                    return 3
            except Exception:
                pass
        if self._context_resolved_key_match(slot, memory):
            return 2
        if self._context_surface_match(slot, memory):
            return 1
        return 0

    @staticmethod
    def _unique_ids(values):
        output, seen = [], set()
        for value in values:
            if value and value not in seen:
                output.append(value)
                seen.add(value)
        return output

    def _rank_context_candidates(
        self, slot, ordered, candidate_by_id, relations, views=None
    ):
        del relations
        views = [
            [
                memory_id
                for memory_id in view
                if memory_id in candidate_by_id and memory_id in ordered
            ]
            for view in (views or [])
        ]
        views = [self._unique_ids(view) for view in views if view]
        baseline = self._unique_ids(
            [memory_id for view in views for memory_id in view] + list(ordered)
        )
        original_index = {
            memory_id: index for index, memory_id in enumerate(baseline)
        }
        promoted = [
            memory_id
            for memory_id in baseline
            if self._context_promotion_level(slot, candidate_by_id[memory_id]) > 0
        ]
        promoted.sort(
            key=lambda memory_id: (
                -self._context_promotion_level(
                    slot, candidate_by_id[memory_id]
                ),
                original_index[memory_id],
            )
        )
        ranked = self._unique_ids(promoted)
        max_depth = max((len(view) for view in views), default=0)
        for depth in range(max_depth):
            for view in views:
                if depth < len(view) and view[depth] not in ranked:
                    ranked.append(view[depth])
        for memory_id in baseline:
            if memory_id not in ranked:
                ranked.append(memory_id)
        ordered[:] = ranked
        return ordered

    @staticmethod
    def _unique_slots(
        slots: Sequence[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        output, seen = [], set()
        for slot in slots:
            slot_id = str(slot.get("id") or "")
            if slot_id and slot_id not in seen:
                output.append(slot)
                seen.add(slot_id)
        return output

    def _slot_trace_views(
        self, slot, trace, candidate_by_id, operation_output_ids
    ) -> List[List[str]]:
        slot_id = str(slot.get("id") or "")
        indexed = list(enumerate(trace or []))
        indexed.sort(
            key=lambda pair: (
                int(pair[1].get("retrieval_round") or 0),
                int(pair[1].get("operation_index") or pair[0]),
                pair[0],
            ),
            reverse=True,
        )
        views = []
        for _, item in indexed:
            if slot_id not in (item.get("produces") or []):
                continue
            view = []
            for memory_id in item.get("output_ids") or []:
                memory = candidate_by_id.get(memory_id)
                if (
                    memory_id in operation_output_ids
                    and memory
                    and self._context_candidate_eligible(slot, memory)
                    and memory_id not in view
                ):
                    view.append(memory_id)
            if view:
                views.append(view)
        return views

    def _prepare_requirement_context_state(self, run, initial_seeds):
        self._last_reserved_seed_context = []
        self._last_requirement_proof_support = {}
        self._last_requirement_context_candidates = {}
        self._last_requirement_context_views = {}
        self._last_requirement_status = dict(run.get("requirement_status") or {})
        if run.get("fast_supports") is not None:
            run["requirement_proof_support"] = {}
            run["requirement_context_candidates"] = {}
            run["requirement_context_views"] = {}
            run["reserved_seed_context"] = []
            return run
        plan = run.get("plan") or {}
        semantic_slots = [
            slot
            for slot in self._unique_slots(plan.get("required_slots") or [])
            if self._semantic_context_slot(slot)
        ]
        if not semantic_slots:
            run["requirement_proof_support"] = {}
            run["requirement_context_candidates"] = {}
            run["requirement_context_views"] = {}
            run["reserved_seed_context"] = []
            return run
        operation_output_ids = set(run.get("operation_output_ids") or [])
        candidate_by_id = {}
        for memory in (
            *(run.get("operation_candidates") or []),
            *(run.get("beliefs") or []),
            *(run.get("planning_seeds") or []),
            *(initial_seeds or []),
        ):
            if memory and memory.get("id"):
                candidate_by_id[memory["id"]] = memory
        trace = list(run.get("trace") or [])
        slot_support = run.setdefault("slot_support", {})
        original_support = {
            str(slot_id): list(memory_ids or [])
            for slot_id, memory_ids in slot_support.items()
        }
        relations = run.get("relations") or []
        proof_support, context_candidates, context_views = {}, {}, {}
        for slot in semantic_slots:
            slot_id = str(slot.get("id") or "")
            views = self._slot_trace_views(
                slot, trace, candidate_by_id, operation_output_ids
            )
            support_view = [
                memory_id
                for memory_id in original_support.get(slot_id, [])
                if memory_id in candidate_by_id
                and self._context_candidate_eligible(
                    slot, candidate_by_id[memory_id]
                )
            ]
            seed_view = [
                memory.get("id")
                for memory in initial_seeds or []
                if memory.get("id") in candidate_by_id
                and self._context_candidate_eligible(slot, memory)
            ]
            ordered = self._unique_ids(
                [memory_id for view in views for memory_id in view]
                + support_view
                + seed_view
            )
            self._rank_context_candidates(
                slot, ordered, candidate_by_id, relations, views=views
            )
            context_candidates[slot_id] = ordered
            context_views[slot_id] = [list(view) for view in views]
            certified = []
            for memory_id in original_support.get(slot_id, []) or []:
                memory = candidate_by_id.get(memory_id)
                if not memory or not self._requirement_target_proof(slot, memory):
                    continue
                if self._slot_covered(
                    slot, [memory_id], [memory], relations
                ):
                    certified.append(memory_id)
            proof_support[slot_id] = certified
            slot_support[slot_id] = list(certified)
        pool = []
        for slot in semantic_slots:
            for memory_id in context_candidates.get(
                str(slot.get("id") or ""), []
            ):
                if memory_id not in pool:
                    pool.append(memory_id)
        slot_support[self.CONTEXT_POOL_KEY] = pool
        certified_ids = {
            memory_id for values in proof_support.values() for memory_id in values
        }
        for memory in [
            *(run.get("operation_candidates") or []),
            *(run.get("planning_seeds") or []),
        ]:
            if (
                memory.get("id") in pool
                and memory.get("id") not in certified_ids
            ):
                memory["_supplementary_context"] = True
        self._last_requirement_proof_support = proof_support
        self._last_requirement_context_candidates = context_candidates
        self._last_requirement_context_views = context_views
        run["requirement_proof_support"] = {
            key: list(value) for key, value in proof_support.items()
        }
        run["requirement_context_candidates"] = {
            key: list(value) for key, value in context_candidates.items()
        }
        run["requirement_context_views"] = {
            key: [list(view) for view in views]
            for key, views in context_views.items()
        }
        run["reserved_seed_context"] = []
        return run

    def _reserve_initial_requirement_context(self, run, initial_seeds):
        return self._prepare_requirement_context_state(run, initial_seeds)

    def _run_query_retrieval(
        self,
        question,
        initial_seeds,
        frame,
        fast_supports,
        gate,
        planning_seeds=None,
        planning_context=None,
    ):
        run = super()._run_query_retrieval(
            question,
            initial_seeds,
            frame,
            fast_supports,
            gate,
            planning_seeds=planning_seeds,
            planning_context=planning_context,
        )
        return self._prepare_requirement_context_state(run, initial_seeds)

    def _role_aware_support_ids(
        self,
        slots: List[Dict[str, Any]],
        slot_support: Dict[str, List[str]],
        candidate_order: List[str],
        limit: int,
    ) -> List[str]:
        bounded_limit = max(0, int(limit))
        if not bounded_limit:
            return []
        semantic_slots = [
            slot
            for slot in self._unique_slots(slots)
            if self._semantic_context_slot(slot)
        ]
        if not semantic_slots or not getattr(
            self, "_last_requirement_context_candidates", None
        ):
            return super()._role_aware_support_ids(
                slots, slot_support, candidate_order, bounded_limit
            )
        allowed, selected = set(candidate_order), []

        def add(memory_id: str) -> bool:
            if (
                memory_id
                and memory_id in allowed
                and memory_id not in selected
                and len(selected) < bounded_limit
            ):
                selected.append(memory_id)
                return True
            return False

        candidates = (
            getattr(self, "_last_requirement_context_candidates", {}) or {}
        )
        proofs = getattr(self, "_last_requirement_proof_support", {}) or {}
        statuses = getattr(self, "_last_requirement_status", {}) or {}
        for slot in semantic_slots:
            slot_id = str(slot.get("id") or "")
            proof_id = (
                next(
                    (
                        memory_id
                        for memory_id in proofs.get(slot_id, [])
                        if memory_id in allowed
                    ),
                    "",
                )
                if statuses.get(slot_id) == "FOUND"
                else ""
            )
            if proof_id:
                add(proof_id)
            else:
                add(
                    next(
                        (
                            memory_id
                            for memory_id in candidates.get(slot_id, [])
                            if memory_id in allowed
                        ),
                        "",
                    )
                )
        legacy = super()._role_aware_support_ids(
            slots,
            {
                key: value
                for key, value in slot_support.items()
                if key != self.CONTEXT_POOL_KEY
            },
            candidate_order,
            bounded_limit,
        )
        while len(selected) < bounded_limit:
            progressed = False
            for slot in semantic_slots:
                values = [
                    memory_id
                    for memory_id in candidates.get(
                        str(slot.get("id") or ""), []
                    )
                    if memory_id in allowed and memory_id not in selected
                ]
                if values and add(values[0]):
                    progressed = True
            if not progressed:
                break
        for memory_id in legacy:
            add(memory_id)
        return selected

    @staticmethod
    def _reasoning_obligation_lines(plan: Dict[str, Any]) -> List[str]:
        semantic_ir = (plan or {}).get("semantic_ir") or {}
        requirements = semantic_ir.get("requirements") or []
        relations = semantic_ir.get("relations") or (
            plan or {}
        ).get("semantic_relations") or []
        lines = []
        for requirement in requirements:
            lines.append(
                "- requirement "
                + str(requirement.get("id") or "?")
                + " ["
                + str(requirement.get("grounding_kind") or "QUESTION")
                + "]: "
                + str(
                    requirement.get("retrieval_hint")
                    or requirement.get("evidence_target")
                    or requirement.get("focus_span")
                    or "evidence"
                )
            )
        for relation in relations:
            line = (
                f"- bridge {relation.get('from','')} "
                f"-[{relation.get('type','')}]-> {relation.get('to','')}"
            )
            goal = str(relation.get("bridge_goal") or "").strip()
            if goal:
                line += f": {goal}"
            lines.append(line)
        return lines

    def prepare_batch_query(self, question, system_message=None, **kwargs):
        prepared = super().prepare_batch_query(
            question, system_message=system_message, **kwargs
        )
        extra = prepared.setdefault("extra", {})
        proof_support = {
            key: list(value)
            for key, value in (
                getattr(self, "_last_requirement_proof_support", {}) or {}
            ).items()
        }
        context_candidates = {
            key: list(value)
            for key, value in (
                getattr(self, "_last_requirement_context_candidates", {}) or {}
            ).items()
        }
        context_views = {
            key: [list(view) for view in views]
            for key, views in (
                getattr(self, "_last_requirement_context_views", {}) or {}
            ).items()
        }
        extra["requirement_proof_support"] = proof_support
        extra["requirement_context_candidates"] = context_candidates
        extra["requirement_context_views"] = context_views
        extra["reserved_seed_context"] = []
        extra["read_contract_version"] = "minimal-ir-v5-broad-context-strict-proof"
        final_ids = set(extra.get("final_memory_ids") or [])
        certified_ids = {
            memory_id for values in proof_support.values() for memory_id in values
        }
        assigned_ids = {
            memory_id
            for values in context_candidates.values()
            for memory_id in values
        }
        unverified = set(extra.get("unverified_context_ids") or [])
        unverified.update((final_ids & assigned_ids) - certified_ids)
        extra["unverified_context_ids"] = sorted(unverified & final_ids)
        extra["requirement_context_selected"] = {
            slot_id: [
                memory_id for memory_id in values if memory_id in final_ids
            ]
            for slot_id, values in context_candidates.items()
        }
        extra["candidate_lifecycle"] = [
            {
                "slot_id": slot_id,
                "memory_id": memory_id,
                "context_rank": rank + 1,
                "selected": memory_id in final_ids,
                "drop_reason": ""
                if memory_id in final_ids
                else "CONTEXT_BUDGET_OR_ARBITRATION",
            }
            for slot_id, values in context_candidates.items()
            for rank, memory_id in enumerate(values)
        ]
        for item in extra.get("retrieval_provenance") or []:
            item["slot_ids"] = [
                slot_id
                for slot_id in item.get("slot_ids") or []
                if slot_id != self.CONTEXT_POOL_KEY
            ]
        source_plan = extra.get("plan") or extra.get("replan") or {}
        extra["graph_validation"] = source_plan.get("graph_validation") or {}
        slots = source_plan.get("required_slots") or []
        extra["requirement_diagnostics"] = {
            str(slot["id"]): {
                "focus_span": slot.get("focus_span", ""),
                "retrieval_target": slot.get(
                    "retrieval_target", slot.get("target_surface", "")
                ),
                "proof_anchor": slot.get(
                    "proof_anchor", slot.get("target_surface", "")
                ),
                "resolved_keys": list(slot.get("resolved_keys") or []),
                "proof_result": bool(proof_support.get(str(slot["id"]))),
                "context_candidate_count": len(
                    context_candidates.get(str(slot["id"]), [])
                ),
                "proof_miss_reason": ""
                if proof_support.get(str(slot["id"]))
                else "DEGRADED_TARGET"
                if slot.get("degraded")
                else "NO_CANDIDATE"
                if not context_candidates.get(str(slot["id"]))
                else "NO_CONSERVATIVE_TARGET_PROOF",
            }
            for slot in slots
            if self._semantic_context_slot(slot)
        }
        options = source_plan.get("visible_options") or {}
        probed = getattr(self, "_last_option_probe_coverage", {}) or {}
        extra["option_probe_complete"] = bool(options) and set(options).issubset(
            probed
        )
        extra["retrieval_complete_semantics"] = (
            "OPTION_EXPLORATION"
            if options
            else "DETERMINISTIC_EVIDENCE_CONTRACT"
        )
        obligation_lines = self._reasoning_obligation_lines(source_plan)
        if obligation_lines:
            addendum = (
                "\n\n=== SEMANTIC EVIDENCE GRAPH ===\n"
                + "\n".join(obligation_lines)
                + "\nContext eligibility is broad and structural; target proof is "
                "strict and controls proof state only. A context candidate may be "
                "useful without being certified. resolved_keys are optional schema "
                "hints, not proof. Compiler evidence-role names do not change semantic "
                "obligation priority. DEPENDS_ON means the target is a prerequisite "
                "for the source, not a causal edge."
            )
            graph = extra["graph_validation"]
            if graph.get("orphan_requirements"):
                addendum += (
                    "\nINCOMPLETE CONTROLLER GRAPH: unconnected requirements "
                    + ", ".join(graph["orphan_requirements"])
                    + ". Keep them as evidence candidates; do not manufacture a relation."
                )
            for message in prepared.get("messages") or []:
                if message.get("role") == "system":
                    message["content"] = (
                        str(message.get("content") or "") + addendum
                    )
                    break
        return prepared
