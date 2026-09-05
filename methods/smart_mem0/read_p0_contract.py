"""P0 retrieval guards for the minimal-IR SmartMem0 read path.

These guards deliberately stay below semantic answerability.  They tighten the
structural meaning of REQUIREMENT support and preserve one useful initial seed
for final answer context without turning that seed into retrieval proof.
"""

from typing import Any, Dict, List, Optional

from .contracts import QueryFrame


class ReadP0ContractMixin:
    """Repair target authorization and bounded seed-to-context preservation."""

    def _slot_contract_match(self, slot, memory, strict_targets=None):
        """A REQUIREMENT may be FOUND only on its question-owned target surface.

        Older callers explicitly passed ``False`` for REQUIREMENT temporal and
        current-state slots.  That made any dated/current same-owner value a
        structural success.  Override that exception whenever the compiler has
        a question-owned target surface; empty legacy targets retain the prior
        behavior rather than becoming impossible to satisfy.
        """
        role = str(slot.get("evidence_role") or "").upper()
        if role == "REQUIREMENT" and str(slot.get("target_surface") or "").strip():
            strict_targets = True
        return super()._slot_contract_match(slot, memory, strict_targets)

    def _operation_slot_support(self, slot, result, relations):
        """Keep generic DIRECT requirements at one candidate until P0 is proven."""
        supported = super()._operation_slot_support(slot, result, relations)
        role = str(slot.get("evidence_role") or "").upper()
        slot_type = str(slot.get("type") or "DIRECT").upper()
        if role == "REQUIREMENT" and slot_type == "DIRECT":
            return supported[:1]
        return supported

    def _run_query_retrieval(
        self,
        question: str,
        initial_seeds: List[Dict[str, Any]],
        frame: QueryFrame,
        fast_supports: Optional[List[Dict[str, Any]]],
        gate: Dict[str, Any],
        planning_seeds: Optional[List[Dict[str, Any]]] = None,
        planning_context: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Preserve at most one target-compatible Top-3 seed per requirement.

        The base executor computes FOUND/EMPTY before this method returns.  We
        therefore add the seed only to the context support map after execution;
        ``requirement_status`` and ``retrieval_complete`` remain unchanged.  In
        particular, a preserved seed is useful answer context, not retroactive
        proof that an operation satisfied the requirement.
        """
        run = super()._run_query_retrieval(
            question,
            initial_seeds,
            frame,
            fast_supports,
            gate,
            planning_seeds=planning_seeds,
            planning_context=planning_context,
        )
        run["reserved_seed_context"] = {}
        if run.get("fast_supports") is not None:
            return run

        slots = []
        seen_slot_ids = set()
        for candidate_plan in (run.get("plan") or {}, run.get("replan") or {}):
            for slot in candidate_plan.get("required_slots") or []:
                slot_id = str(slot.get("id") or "")
                if slot_id and slot_id not in seen_slot_ids:
                    slots.append(slot)
                    seen_slot_ids.add(slot_id)

        slot_support = run.setdefault("slot_support", {})
        planning_seed_list = run.setdefault("planning_seeds", [])
        planning_seed_ids = {
            str(memory.get("id") or "") for memory in planning_seed_list
        }
        reserved = run["reserved_seed_context"]

        for slot in slots:
            if str(slot.get("evidence_role") or "").upper() != "REQUIREMENT":
                continue
            if not str(slot.get("target_surface") or "").strip():
                continue
            slot_id = str(slot.get("id") or "")
            allow_history = bool(slot.get("history"))
            for seed in initial_seeds[:3]:
                memory_id = str(seed.get("id") or "")
                if not memory_id or not self._memory_value(seed):
                    continue
                status = str(
                    seed.get(
                        "_status",
                        self._belief_status.get(memory_id, "active"),
                    )
                    or "active"
                ).lower()
                if not allow_history and status == "superseded":
                    continue
                if not self._slot_contract_match(slot, seed, True):
                    continue

                supports = slot_support.setdefault(slot_id, [])
                if memory_id not in supports:
                    supports.append(memory_id)
                reserved[slot_id] = memory_id

                # QueryMixin's boundary treats planning_seeds as the authorized
                # seed set. Initial Top-3 seeds are already retrieval-authorized;
                # make that provenance explicit if the planning subset omitted
                # the one we reserved.
                if memory_id not in planning_seed_ids:
                    planning_seed_list.append(self._snapshot(seed))
                    planning_seed_ids.add(memory_id)
                break

        return run
