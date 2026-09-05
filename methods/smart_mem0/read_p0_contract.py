"""P0 retrieval guards for the minimal-IR SmartMem0 read path.

These guards deliberately stay below semantic answerability. They tighten the
structural meaning of REQUIREMENT support and preserve one useful initial seed
for final answer context without turning that seed into retrieval proof.
"""

from typing import Any, Dict, List, Optional

from .contracts import QueryFrame


class ReadP0ContractMixin:
    """Repair target authorization and bounded seed-to-context preservation."""

    def _rc_memory_matches_target(self, slot, memory):
        """Match the question surface or a deterministic target-derived family.

        ``resolved_keys`` are produced from the immutable question focus by the
        compiler. They may therefore identify the same durable concept family,
        but arbitrary retrieval hints or seed text never authorize a proof.
        """
        if super()._rc_memory_matches_target(slot, memory):
            return True
        if str(slot.get("evidence_role") or "").upper() != "REQUIREMENT":
            return False
        resolved_keys = [
            " ".join(self._rc_content_terms(key))
            for key in slot.get("resolved_keys") or []
            if self._rc_content_terms(key)
        ]
        if not resolved_keys:
            return False
        memory_keys = set(self._rc_memory_concept_keys(memory))
        memory_key_terms = [set(self._rc_content_terms(key)) for key in memory_keys]
        for resolved in resolved_keys:
            resolved_terms = set(self._rc_content_terms(resolved))
            if not resolved_terms:
                continue
            if any(
                resolved_terms == terms
                or resolved_terms.issubset(terms)
                or terms.issubset(resolved_terms)
                for terms in memory_key_terms
                if terms
            ):
                return True
        return False

    def _slot_contract_match(self, slot, memory, strict_targets=None):
        """A REQUIREMENT may be FOUND only on its question-owned target family.

        Older callers explicitly passed ``False`` for REQUIREMENT temporal and
        current-state slots. That made any dated/current same-owner value a
        structural success. Override that exception whenever the compiler has a
        question-owned target surface; empty legacy targets retain prior behavior.
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

    def _seed_is_context_eligible(self, slot, seed):
        memory_id = str(seed.get("id") or "")
        if not memory_id or not self._memory_value(seed):
            return False
        status = str(
            seed.get(
                "_status",
                self._belief_status.get(memory_id, "active"),
            )
            or "active"
        ).lower()
        if not bool(slot.get("history")) and status == "superseded":
            return False
        return self._rc_owner_match(slot, seed)

    def _reserve_seed_context(self, run, initial_seeds):
        """Reserve one bounded seed without changing retrieval proof state.

        Strict target compatibility is preferred. For a single-requirement
        query only, if metadata cannot express a semantic family relation (for
        example ``antibiotic to avoid`` -> ``cefuroxime allergy``), preserve the
        top-ranked RRF seed as *unverified context*. That fallback is deliberately
        forbidden from changing FOUND/EMPTY and is not used for multi-requirement
        plans where a top seed cannot be assigned safely to a particular slot.
        """
        run["reserved_seed_context"] = {}
        run["reserved_seed_context_mode"] = {}
        if run.get("fast_supports") is not None:
            return

        slots = []
        seen_slot_ids = set()
        for candidate_plan in (run.get("plan") or {}, run.get("replan") or {}):
            for slot in candidate_plan.get("required_slots") or []:
                slot_id = str(slot.get("id") or "")
                if slot_id and slot_id not in seen_slot_ids:
                    slots.append(slot)
                    seen_slot_ids.add(slot_id)

        requirements = [
            slot
            for slot in slots
            if str(slot.get("evidence_role") or "").upper() == "REQUIREMENT"
            and str(slot.get("target_surface") or "").strip()
        ]
        slot_support = run.setdefault("slot_support", {})
        planning_seed_list = run.setdefault("planning_seeds", [])
        planning_seed_ids = {
            str(memory.get("id") or "") for memory in planning_seed_list
        }

        for slot in requirements:
            slot_id = str(slot.get("id") or "")
            chosen = None
            mode = ""
            for seed in initial_seeds[:3]:
                if not self._seed_is_context_eligible(slot, seed):
                    continue
                if self._slot_contract_match(slot, seed, True):
                    chosen = seed
                    mode = "strict_target"
                    break

            if chosen is None and len(requirements) == 1:
                chosen = next(
                    (
                        seed
                        for seed in initial_seeds[:1]
                        if self._seed_is_context_eligible(slot, seed)
                    ),
                    None,
                )
                if chosen is not None:
                    mode = "top1_unverified"

            if chosen is None:
                continue
            memory_id = str(chosen.get("id") or "")
            supports = slot_support.setdefault(slot_id, [])
            if memory_id not in supports:
                supports.append(memory_id)
            run["reserved_seed_context"][slot_id] = memory_id
            run["reserved_seed_context_mode"][slot_id] = mode

            # QueryMixin's boundary treats planning_seeds as the authorized seed
            # set. Initial Top-3 seeds are already retrieval-authorized; make the
            # provenance explicit when the planning subset omitted the reserved
            # seed. This affects final context only, never requirement_status.
            if memory_id not in planning_seed_ids:
                planning_seed_list.append(self._snapshot(chosen))
                planning_seed_ids.add(memory_id)

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
        """Run the base proof first, then preserve bounded seed context."""
        run = super()._run_query_retrieval(
            question,
            initial_seeds,
            frame,
            fast_supports,
            gate,
            planning_seeds=planning_seeds,
            planning_context=planning_context,
        )
        self._reserve_seed_context(run, initial_seeds)
        self._last_reserved_seed_context = dict(run.get("reserved_seed_context") or {})
        self._last_reserved_seed_context_mode = dict(
            run.get("reserved_seed_context_mode") or {}
        )
        return run

    def prepare_batch_query(self, question, system_message=None, **kwargs):
        prepared = super().prepare_batch_query(
            question, system_message=system_message, **kwargs
        )
        extra = prepared.setdefault("extra", {})
        extra["reserved_seed_context"] = dict(
            getattr(self, "_last_reserved_seed_context", {}) or {}
        )
        extra["reserved_seed_context_mode"] = dict(
            getattr(self, "_last_reserved_seed_context_mode", {}) or {}
        )
        return prepared
