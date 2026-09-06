"""Deterministic requirement proof and bounded answer-context selection.

Proof and context are deliberately separate decisions:
- target proof is a conservative certificate used by FOUND/EMPTY and structural relation proof;
- context packing is recall-oriented and may retain bounded, authorized candidates that
  are useful to the final answer model even when deterministic target proof fails.
"""

from typing import Any, Dict, List, Sequence


class ProofContextContractMixin:
    """Keep proof strict without letting proof failures erase useful context."""

    CONTEXT_POOL_KEY = "__answer_context_candidates__"

    def _requirement_target_proof(self, slot, memory):
        """Proof uses proof_anchor; semantic target stays recall-oriented."""
        if str(slot.get("evidence_role") or "").upper() != "REQUIREMENT":
            return True
        if slot.get("degraded"):
            return False
        target = str(slot.get("proof_anchor") or slot.get("target_surface") or "").strip()
        if not target:
            return False
        text = self._rc_memory_target_text(memory)
        if self._rc_token_sequence_present(target, text):
            return True
        target_terms = list(dict.fromkeys(self._rc_content_terms(target)))
        if not target_terms:
            return False
        text_terms = set(self._rc_content_terms(text))
        overlap = sum(term in text_terms for term in target_terms)
        if len(target_terms) == 1:
            return overlap == 1
        if len(target_terms) == 2:
            return overlap == 2
        return overlap >= max(2, (3 * len(target_terms) + 4) // 5)

    def _slot_covered(self, slot, support_ids, selected, relations):
        """Candidate availability is not deterministic requirement proof."""
        role = str(slot.get("evidence_role") or "").upper()
        slot_type = str(slot.get("type") or "DIRECT").upper()
        if role != "REQUIREMENT" or slot_type not in {"DIRECT", "TEMPORAL", "CURRENT_STATE"}:
            return super()._slot_covered(slot, support_ids, selected, relations)

        support_set = set(support_ids or [])
        proof_ids = [
            memory["id"]
            for memory in selected
            if memory.get("id") in support_set and self._requirement_target_proof(slot, memory)
        ]
        if not proof_ids:
            return False
        return super()._slot_covered(slot, proof_ids, selected, relations)

    def _relation_status_map(self, plan, slot_support, selected, relations):
        """Structural relations may use only target-certified endpoints."""
        selected_by_id = {memory.get("id"): memory for memory in selected}
        filtered_support = {
            str(slot_id): list(memory_ids or [])
            for slot_id, memory_ids in (slot_support or {}).items()
        }
        for slot in plan.get("required_slots") or []:
            if str(slot.get("evidence_role") or "").upper() != "REQUIREMENT":
                continue
            slot_id = str(slot.get("id") or "")
            filtered_support[slot_id] = [
                memory_id
                for memory_id in filtered_support.get(slot_id, [])
                if memory_id in selected_by_id
                and self._requirement_target_proof(slot, selected_by_id[memory_id])
            ]
        return super()._relation_status_map(plan, filtered_support, selected, relations)

    def _context_candidate_eligible(self, slot, memory):
        """Cheap deterministic guard for candidate context, not target proof."""
        if not memory or not memory.get("id") or not self._memory_value(memory):
            return False
        if not self._rc_owner_match(slot, memory):
            return False
        if not self._memory_matches_slot_role(slot, memory):
            return False
        status = memory.get("_status", self._belief_status.get(memory.get("id"), "active"))
        return bool(slot.get("history")) or status != "superseded"

    def _context_target_score(self, slot, memory) -> float:
        """Recall ranking only: semantic target relevance is never proof."""
        target = str(slot.get("target_surface") or slot.get("retrieval_target") or "").strip()
        if not target:
            return 0.0
        text = self._rc_memory_target_text(memory)
        if self._rc_token_sequence_present(target, text):
            return 1.0
        target_terms = list(dict.fromkeys(self._rc_content_terms(target)))
        if not target_terms:
            return 0.0
        text_terms = set(self._rc_content_terms(text))
        return sum(term in text_terms for term in target_terms) / len(target_terms)

    def _context_rank_signals(self, slot, memory, relations):
        """Rank answer context by obligation value; provenance is only a tie-break."""
        try:
            structural = bool(
                self._slot_structure_covered(slot, [memory["id"]], [memory], relations)
            )
        except Exception:
            structural = False
        certificate = False
        certificate_fn = getattr(self, "_certificate_result", None)
        if callable(certificate_fn):
            try:
                certificate = bool(certificate_fn(slot, memory)[0])
            except Exception:
                certificate = False
        wanted_keys = {self._rc_text(key) for key in (slot.get("resolved_keys") or []) if key}
        memory_keys = {self._rc_text(key) for key in self._rc_memory_concept_keys(memory) if key}
        resolved_key_match = bool(wanted_keys & memory_keys)
        answer_bearing = bool(
            self._memory_value(memory)
            or memory.get("claim")
            or memory.get("verbatim_value")
        )
        return {
            "structural": structural,
            "certificate": certificate,
            "resolved_key_match": resolved_key_match,
            "target_score": self._context_target_score(slot, memory),
            "answer_bearing": answer_bearing,
            "retrieval_score": float(memory.get("_score") or 0.0),
        }

    def _rank_context_candidates(self, slot, ordered, candidate_by_id, relations):
        """Answer relevance outranks retrieval round while preserving stable ties."""
        original_index = {memory_id: index for index, memory_id in enumerate(ordered)}
        signals = {
            memory_id: self._context_rank_signals(slot, candidate_by_id[memory_id], relations)
            for memory_id in ordered
            if memory_id in candidate_by_id
        }
        ordered.sort(
            key=lambda memory_id: (
                signals.get(memory_id, {}).get("structural", False),
                signals.get(memory_id, {}).get("certificate", False),
                signals.get(memory_id, {}).get("resolved_key_match", False),
                signals.get(memory_id, {}).get("target_score", 0.0),
                signals.get(memory_id, {}).get("answer_bearing", False),
                signals.get(memory_id, {}).get("retrieval_score", 0.0),
                -original_index.get(memory_id, 10**6),
            ),
            reverse=True,
        )
        return ordered

    @staticmethod
    def _unique_slots(slots: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        output, seen = [], set()
        for slot in slots:
            slot_id = str(slot.get("id") or "")
            if slot_id and slot_id not in seen:
                output.append(slot)
                seen.add(slot_id)
        return output

    def _prepare_requirement_context_state(self, run, initial_seeds):
        """Build strict proof and recall-oriented context lanes.

        Retrieval round/provenance determines which candidates enter the pool, but once
        admitted, obligation relevance ranks them. Recovery recency is only a stable
        tie-break and can never outrank a clearly answer-bearing earlier candidate.
        """
        self._last_reserved_seed_context = []
        self._last_requirement_proof_support = {}
        self._last_requirement_context_candidates = {}
        self._last_requirement_status = dict(run.get("requirement_status") or {})

        if run.get("fast_supports") is not None:
            run["requirement_proof_support"] = {}
            run["requirement_context_candidates"] = {}
            run["reserved_seed_context"] = []
            return run

        plan = run.get("plan") or {}
        requirement_slots = [
            slot
            for slot in self._unique_slots(plan.get("required_slots") or [])
            if str(slot.get("evidence_role") or "").upper() == "REQUIREMENT"
        ]
        if not requirement_slots:
            run["requirement_proof_support"] = {}
            run["requirement_context_candidates"] = {}
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
        indexed_trace = list(enumerate(trace))
        indexed_trace.sort(
            key=lambda pair: (int(pair[1].get("retrieval_round") or 0), pair[0]),
            reverse=True,
        )

        slot_support = run.setdefault("slot_support", {})
        original_support = {
            str(slot_id): list(memory_ids or [])
            for slot_id, memory_ids in slot_support.items()
        }
        relations = run.get("relations") or []
        proof_support: Dict[str, List[str]] = {}
        context_candidates: Dict[str, List[str]] = {}

        for slot in requirement_slots:
            slot_id = str(slot.get("id") or "")
            ordered: List[str] = []
            grouped = {}
            for _, item in indexed_trace:
                if slot_id not in (item.get("produces") or []):
                    continue
                round_id = int(item.get("retrieval_round") or 0)
                grouped.setdefault(round_id, [])
                for memory_id in item.get("output_ids") or []:
                    if memory_id not in grouped[round_id]:
                        grouped[round_id].append(memory_id)
            for round_id in sorted(grouped, reverse=True):
                for memory_id in grouped[round_id]:
                    memory = candidate_by_id.get(memory_id)
                    if (
                        memory_id in operation_output_ids
                        and memory
                        and memory_id not in ordered
                        and self._context_candidate_eligible(slot, memory)
                    ):
                        ordered.append(memory_id)

            for memory_id in original_support.get(slot_id, []) or []:
                memory = candidate_by_id.get(memory_id)
                if memory and memory_id not in ordered and self._context_candidate_eligible(slot, memory):
                    ordered.append(memory_id)

            for memory in initial_seeds or []:
                memory_id = memory.get("id")
                if memory_id and memory_id not in ordered and self._context_candidate_eligible(slot, memory):
                    ordered.append(memory_id)

            self._rank_context_candidates(slot, ordered, candidate_by_id, relations)
            context_candidates[slot_id] = ordered

            certified = []
            for memory_id in original_support.get(slot_id, []) or []:
                memory = candidate_by_id.get(memory_id)
                if not memory or not self._requirement_target_proof(slot, memory):
                    continue
                if self._slot_covered(slot, [memory_id], [memory], relations):
                    certified.append(memory_id)
            proof_support[slot_id] = certified
            slot_support[slot_id] = list(certified)

        pool = []
        for slot in requirement_slots:
            for memory_id in context_candidates.get(str(slot.get("id") or ""), []):
                if memory_id not in pool:
                    pool.append(memory_id)
        slot_support[self.CONTEXT_POOL_KEY] = pool

        certified_ids = {memory_id for values in proof_support.values() for memory_id in values}
        for memory in run.get("operation_candidates") or []:
            if memory.get("id") in pool and memory.get("id") not in certified_ids:
                memory["_supplementary_context"] = True
        for memory in run.get("planning_seeds") or []:
            if memory.get("id") in pool and memory.get("id") not in certified_ids:
                memory["_supplementary_context"] = True

        self._last_requirement_proof_support = proof_support
        self._last_requirement_context_candidates = context_candidates
        run["requirement_proof_support"] = {key: list(value) for key, value in proof_support.items()}
        run["requirement_context_candidates"] = {key: list(value) for key, value in context_candidates.items()}
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
        """Pack requirements fairly; proof reserves space but does not own ranking."""
        bounded_limit = max(0, int(limit))
        if not bounded_limit:
            return []

        requirement_slots = [
            slot
            for slot in self._unique_slots(slots)
            if str(slot.get("evidence_role") or "").upper() == "REQUIREMENT"
        ]
        if not requirement_slots or not getattr(self, "_last_requirement_context_candidates", None):
            return super()._role_aware_support_ids(slots, slot_support, candidate_order, bounded_limit)

        allowed = set(candidate_order)
        selected: List[str] = []

        def add(memory_id: str) -> bool:
            if memory_id and memory_id in allowed and memory_id not in selected and len(selected) < bounded_limit:
                selected.append(memory_id)
                return True
            return False

        candidates = getattr(self, "_last_requirement_context_candidates", {}) or {}
        proofs = getattr(self, "_last_requirement_proof_support", {}) or {}
        statuses = getattr(self, "_last_requirement_status", {}) or {}

        for slot in requirement_slots:
            slot_id = str(slot.get("id") or "")
            if statuses.get(slot_id) == "FOUND":
                proof_id = next((mid for mid in proofs.get(slot_id, []) if mid in allowed), "")
                if proof_id:
                    add(proof_id)
                    continue
            candidate_id = next((mid for mid in candidates.get(slot_id, []) if mid in allowed), "")
            add(candidate_id)

        non_requirement_ids = {
            memory_id
            for slot in self._unique_slots(slots)
            if str(slot.get("evidence_role") or "").upper() != "REQUIREMENT"
            for memory_id in slot_support.get(str(slot.get("id") or ""), [])
        }
        legacy = super()._role_aware_support_ids(
            slots,
            {key: value for key, value in slot_support.items() if key != self.CONTEXT_POOL_KEY},
            candidate_order,
            bounded_limit,
        )
        for memory_id in legacy:
            if memory_id in non_requirement_ids:
                add(memory_id)

        while len(selected) < bounded_limit:
            progressed = False
            for slot in requirement_slots:
                slot_id = str(slot.get("id") or "")
                values = [mid for mid in candidates.get(slot_id, []) if mid in allowed and mid not in selected]
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
        relations = semantic_ir.get("relations") or (plan or {}).get("semantic_relations") or []
        if not requirements and not relations:
            return []
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
            relation_type = str(relation.get("type") or "")
            source = str(relation.get("from") or "")
            target = str(relation.get("to") or "")
            goal = str(relation.get("bridge_goal") or "").strip()
            line = f"- bridge {source} -[{relation_type}]-> {target}"
            if goal:
                line += f": {goal}"
            lines.append(line)
        return lines

    def prepare_batch_query(self, question, system_message=None, **kwargs):
        prepared = super().prepare_batch_query(question, system_message=system_message, **kwargs)
        extra = prepared.setdefault("extra", {})
        proof_support = {
            key: list(value)
            for key, value in (getattr(self, "_last_requirement_proof_support", {}) or {}).items()
        }
        context_candidates = {
            key: list(value)
            for key, value in (getattr(self, "_last_requirement_context_candidates", {}) or {}).items()
        }
        extra["requirement_proof_support"] = proof_support
        extra["requirement_context_candidates"] = context_candidates
        extra["reserved_seed_context"] = []
        extra["read_contract_version"] = "minimal-ir-v2-proof-context-bridges"

        final_ids = set(extra.get("final_memory_ids") or [])
        certified_ids = {memory_id for values in proof_support.values() for memory_id in values}
        assigned_ids = {memory_id for values in context_candidates.values() for memory_id in values}
        unverified = set(extra.get("unverified_context_ids") or [])
        unverified.update((final_ids & assigned_ids) - certified_ids)
        extra["unverified_context_ids"] = sorted(unverified & final_ids)
        extra["requirement_context_selected"] = {
            slot_id: [memory_id for memory_id in values if memory_id in final_ids]
            for slot_id, values in context_candidates.items()
        }
        extra["candidate_lifecycle"] = [
            {
                "slot_id": slot_id,
                "memory_id": memory_id,
                "context_rank": rank + 1,
                "selected": memory_id in final_ids,
                "drop_reason": "" if memory_id in final_ids else "CONTEXT_BUDGET_OR_ARBITRATION",
            }
            for slot_id, values in context_candidates.items()
            for rank, memory_id in enumerate(values)
        ]
        for item in extra.get("retrieval_provenance") or []:
            item["slot_ids"] = [
                slot_id for slot_id in item.get("slot_ids") or [] if slot_id != self.CONTEXT_POOL_KEY
            ]

        plan = extra.get("replan") or extra.get("plan") or {}
        source_plan = extra.get("plan") or plan
        extra["graph_validation"] = source_plan.get("graph_validation") or {}
        slots = source_plan.get("required_slots") or []
        extra["requirement_diagnostics"] = {
            str(slot["id"]): {
                "focus_span": slot.get("focus_span", ""),
                "retrieval_target": slot.get("retrieval_target", slot.get("target_surface", "")),
                "proof_anchor": slot.get("proof_anchor", slot.get("target_surface", "")),
                "proof_result": bool(proof_support.get(str(slot["id"]))),
                "proof_miss_reason": "" if proof_support.get(str(slot["id"])) else
                    "DEGRADED_TARGET" if slot.get("degraded") else
                    "NO_CANDIDATE" if not context_candidates.get(str(slot["id"])) else
                    "NO_TARGET_AND_TYPED_CONTRACT_CERTIFICATE",
            }
            for slot in slots
            if slot.get("evidence_role") == "REQUIREMENT"
        }
        options = source_plan.get("visible_options") or {}
        probed = getattr(self, "_last_option_probe_coverage", {}) or {}
        extra["option_probe_complete"] = bool(options) and set(options).issubset(probed)
        extra["retrieval_complete_semantics"] = (
            "OPTION_EXPLORATION" if options else "DETERMINISTIC_EVIDENCE_CONTRACT"
        )
        obligation_lines = self._reasoning_obligation_lines(source_plan)
        if obligation_lines:
            addendum = (
                "\n\n=== SEMANTIC EVIDENCE GRAPH ===\n"
                + "\n".join(obligation_lines)
                + "\nTarget certification is a retrieval/control-flow certificate, not a guarantee "
                "that one certified memory is the final answer. Evaluate all supplied "
                "authorized memories and use bridge goals only to connect grounded facts."
                " DEPENDS_ON means the target is a prerequisite for the source, not a causal edge."
                " Use grounded DERIVED values wherever the reasoning depends on them; never"
                " replace a measured participant value with a generic assumption."
            )
            graph = extra["graph_validation"]
            if graph.get("orphan_requirements"):
                addendum += (
                    "\nINCOMPLETE CONTROLLER GRAPH: unconnected requirements "
                    + ", ".join(graph["orphan_requirements"])
                    + ". These remain evidence candidates, not proven causal links."
                    " Do not manufacture a relation to repair the graph."
                )
            for message in prepared.get("messages") or []:
                if message.get("role") == "system":
                    message["content"] = str(message.get("content") or "") + addendum
                    break
        return prepared
