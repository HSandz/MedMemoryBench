"""Public SmartMem0 facade assembled from cohesive pipeline components."""

from methods.base import BaseAgent
from .capture import CaptureMixin
from .consolidation import ConsolidationMixin
from .core import CoreMemoryMixin
from .execution import ExecutionMixin
from .planning import PlanningMixin
from .query import QueryMixin
from .read_controller import ReadContractMixin
from .read_temporal_contract import ReadTemporalContractMixin
from .read_option_contract import ReadOptionContractMixin
from .read_execution_contract import ReadExecutionContractMixin
from .read_plan_contract import ReadPlanContractMixin
from .read_usage_contract import ReadUsageContractMixin
from .retrieval import RetrievalOperationsMixin
from .write import WriteLifecycleMixin


class SmartMem0Agent(
    ReadContractMixin,
    ReadTemporalContractMixin,
    ReadPlanContractMixin,
    ReadUsageContractMixin,
    ReadOptionContractMixin,
    ReadExecutionContractMixin,
    QueryMixin,
    ExecutionMixin,
    RetrievalOperationsMixin,
    PlanningMixin,
    WriteLifecycleMixin,
    ConsolidationMixin,
    CaptureMixin,
    CoreMemoryMixin,
    BaseAgent,
):
    """Compact evidence-grounded long-term memory for conversational agents."""

    MAX_TWO_STAGE_READ_LLM_CALLS = 2

    def __init__(self, *args, **kwargs):
        enable_two_stage = bool(kwargs.pop("enable_two_stage_controller", True))
        super().__init__(*args, **kwargs)
        self.enable_two_stage_controller = enable_two_stage
        self.max_read_llm_calls = self.MAX_TWO_STAGE_READ_LLM_CALLS
        if enable_two_stage:
            self.enable_slot_support_validation = False
            self.enable_replan = False
            self.enable_planner_repair = False

    def _requirement_target_proof(self, slot, memory):
        """Authorize requirement coverage without narrowing the candidate set.

        Retrieval hints and ranking are deliberately excluded from this proof.
        A generic REQUIREMENT may keep several candidates for the final answer
        model, but only a candidate compatible with the question-owned target
        surface may turn that requirement into FOUND.
        """
        if str(slot.get("evidence_role") or "").upper() != "REQUIREMENT":
            return True
        return bool(slot.get("target_surface")) and self._rc_memory_matches_target(
            slot, memory
        )

    def _slot_covered(self, slot, support_ids, selected, relations):
        """Separate candidate availability from deterministic requirement proof."""
        role = str(slot.get("evidence_role") or "").upper()
        slot_type = str(slot.get("type") or "DIRECT").upper()
        if role != "REQUIREMENT" or slot_type not in {
            "DIRECT",
            "TEMPORAL",
            "CURRENT_STATE",
        }:
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
        # Reuse the existing type-specific structural checks (date relation,
        # current-state head, value/assertion mode, history status) on the same
        # target-compatible candidates. This prevents one candidate from
        # satisfying the target while another independently satisfies time.
        return super()._slot_covered(slot, proof_ids, selected, relations)

    def _relation_status_map(self, plan, slot_support, selected, relations):
        """Do not prove relations from candidates that cannot prove endpoints."""
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
        return super()._relation_status_map(
            plan, filtered_support, selected, relations
        )

    def _seed_context_eligible(self, slot, memory):
        if not memory or not memory.get("id") or not self._memory_value(memory):
            return False
        if not self._rc_owner_match(slot, memory):
            return False
        status = memory.get(
            "_status", self._belief_status.get(memory.get("id"), "active")
        )
        return bool(slot.get("history")) or status != "superseded"

    def _reserve_initial_requirement_context(self, run, initial_seeds):
        """Preserve one useful Top-3 seed as context, never as retrieval proof."""
        self._last_reserved_seed_context = []
        if run.get("fast_supports") is not None:
            return run

        plan = run.get("plan") or {}
        requirement_slots = [
            slot
            for slot in plan.get("required_slots") or []
            if str(slot.get("evidence_role") or "").upper() == "REQUIREMENT"
        ]
        empty_slots = [
            slot
            for slot in requirement_slots
            if (run.get("requirement_status") or {}).get(str(slot.get("id") or ""))
            == "EMPTY"
        ]
        if not empty_slots:
            return run

        planning_seeds = run.setdefault("planning_seeds", [])
        for slot in empty_slots:
            eligible = [
                memory
                for memory in initial_seeds[:3]
                if self._seed_context_eligible(slot, memory)
            ]
            chosen = next(
                (
                    memory
                    for memory in eligible
                    if self._requirement_target_proof(slot, memory)
                ),
                None,
            )
            mode = "strict_target"
            # A semantic seed can contain the answer without repeating the
            # question's abstraction (e.g. "cefuroxime allergy" for "which
            # antibiotic was instructed to avoid"). For one atomic requirement
            # only, preserve Top-1 as explicitly unverified context. It cannot
            # change FOUND/EMPTY or retrieval_complete.
            if chosen is None and len(requirement_slots) == 1 and eligible:
                chosen = eligible[0]
                mode = "top1_unverified"
            if chosen is None:
                continue

            slot_id = str(slot.get("id") or "")
            support = run.setdefault("slot_support", {}).setdefault(slot_id, [])
            support[:] = [
                chosen["id"], *[memory_id for memory_id in support if memory_id != chosen["id"]]
            ]

            snapshot = self._snapshot(chosen)
            if mode == "top1_unverified":
                snapshot["_supplementary_context"] = True
            planning_seeds[:] = [
                memory
                for memory in planning_seeds
                if memory.get("id") != chosen["id"]
            ]
            planning_seeds.insert(0, snapshot)
            self._last_reserved_seed_context.append(
                {
                    "slot_id": slot_id,
                    "memory_id": chosen["id"],
                    "mode": mode,
                }
            )

        run["reserved_seed_context"] = list(self._last_reserved_seed_context)
        return run

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
        return self._reserve_initial_requirement_context(run, initial_seeds)

    def prepare_batch_query(self, question, system_message=None, **kwargs):
        prepared = super().prepare_batch_query(
            question, system_message=system_message, **kwargs
        )
        extra = prepared.setdefault("extra", {})
        reservations = list(getattr(self, "_last_reserved_seed_context", []) or [])
        extra["reserved_seed_context"] = reservations
        extra["read_contract_version"] = "minimal-ir-v1-target-bound"

        final_ids = set(extra.get("final_memory_ids") or [])
        unverified = set(extra.get("unverified_context_ids") or [])
        unverified.update(
            item["memory_id"]
            for item in reservations
            if item.get("mode") == "top1_unverified"
            and item.get("memory_id") in final_ids
        )
        extra["unverified_context_ids"] = sorted(unverified)
        return prepared
