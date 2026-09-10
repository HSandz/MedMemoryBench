"""Coverage-preserving final-context retention for SmartMem0 READ.

This layer does not widen retrieval and does not increase the answer-context budget.  It sits
at the outer edge of ProofContext arbitration and protects already-authorized evidence lanes
from being lost when downstream ranking reorders a compact context.  The policy is structural:
requirement proof/support and CandidateSet probe coverage are retention signals, never truth
labels.
"""

from copy import deepcopy
from typing import Any, Dict, List


class ReadProofContextRetentionMixin:
    """Keep the smallest sufficient authorized context instead of the smallest possible one."""

    PROOF_CONTEXT_VERSION = "proof-context-v3-coverage-retention"
    CONTEXT_OWNER_VERSION = PROOF_CONTEXT_VERSION
    CONTEXT_RETENTION_VERSION = "coverage-preserving-context-v1"

    @staticmethod
    def _pcr_has_selector(slot: Dict[str, Any]) -> bool:
        selector = slot.get("selector") or {}
        relation = str(
            slot.get("time_relation")
            or slot.get("temporal_relation")
            or selector.get("relation")
            or ""
        ).upper()
        return bool(relation) or str(slot.get("type") or "").upper() in {
            "TEMPORAL",
            "CURRENT_STATE",
        }

    def _pcr_semantic_slot(self, slot: Dict[str, Any]) -> bool:
        checker = getattr(self, "_semantic_context_slot", None)
        return bool(callable(checker) and checker(slot))

    @staticmethod
    def _pcr_unique_allowed(values, allowed) -> List[str]:
        output, seen = [], set()
        for value in values or []:
            memory_id = str(value or "")
            if memory_id and memory_id in allowed and memory_id not in seen:
                output.append(memory_id)
                seen.add(memory_id)
        return output

    def _pcr_requirement_reservations(
        self,
        slots: List[Dict[str, Any]],
        slot_support: Dict[str, List[str]],
        allowed: set,
    ) -> List[str]:
        proof_support = getattr(self, "_last_requirement_proof_support", {}) or {}
        context_candidates = (
            getattr(self, "_last_requirement_context_candidates", {}) or {}
        )
        statuses = getattr(self, "_last_requirement_status", {}) or {}
        reserved: List[str] = []
        seen_slots = set()
        for slot in slots or []:
            slot_id = str(slot.get("id") or "")
            if not slot_id or slot_id in seen_slots:
                continue
            seen_slots.add(slot_id)
            if not self._pcr_semantic_slot(slot) or self._pcr_has_selector(slot):
                continue

            preferred = self._pcr_unique_allowed(
                proof_support.get(slot_id) or [], allowed
            )
            if not preferred and str(statuses.get(slot_id) or "") == "FOUND":
                preferred = self._pcr_unique_allowed(
                    [
                        *(context_candidates.get(slot_id) or []),
                        *(slot_support.get(slot_id) or []),
                    ],
                    allowed,
                )
            if preferred and preferred[0] not in reserved:
                reserved.append(preferred[0])
        return reserved

    def _pcr_proposition_reservations(self, allowed: set) -> Dict[str, str]:
        propositions = getattr(self, "_last_candidate_propositions", {}) or {}
        probe = getattr(self, "_last_proposition_probe_coverage", {}) or {}
        local = getattr(self, "_last_candidate_local_coverage", {}) or {}
        if not propositions:
            return {}

        lanes: Dict[str, List[str]] = {}
        local_membership: Dict[str, set] = {}
        for proposition_id in propositions:
            pid = str(proposition_id)
            local_ids = self._pcr_unique_allowed(local.get(pid) or [], allowed)
            probe_ids = self._pcr_unique_allowed(probe.get(pid) or [], allowed)
            lanes[pid] = self._pcr_unique_allowed([*local_ids, *probe_ids], allowed)
            local_membership[pid] = set(local_ids)

        frequency: Dict[str, int] = {}
        for values in lanes.values():
            for memory_id in set(values):
                frequency[memory_id] = frequency.get(memory_id, 0) + 1

        reservations: Dict[str, str] = {}
        for proposition_id, values in lanes.items():
            if not values:
                continue
            indexed = {memory_id: index for index, memory_id in enumerate(values)}
            chosen = min(
                values,
                key=lambda memory_id: (
                    0 if memory_id in local_membership[proposition_id] else 1,
                    indexed[memory_id],
                    frequency.get(memory_id, 10**6),
                    memory_id,
                ),
            )
            reservations[proposition_id] = chosen
        return reservations

    def _role_aware_support_ids(self, slots, slot_support, candidate_order, limit):
        baseline = list(
            super()._role_aware_support_ids(
                slots, slot_support, candidate_order, limit
            )
        )
        bounded_limit = max(0, int(limit))
        if not bounded_limit:
            self._last_context_retention = {
                "version": self.CONTEXT_RETENTION_VERSION,
                "limit": 0,
                "retained_ids": [],
                "semantics": "coverage_preserving_not_truth",
            }
            return []

        allowed = set(str(memory_id) for memory_id in candidate_order or [])
        requirement_reserved = self._pcr_requirement_reservations(
            list(slots or []), slot_support or {}, allowed
        )
        proposition_reserved = self._pcr_proposition_reservations(allowed)

        retained: List[str] = []

        def add(memory_id: str) -> None:
            memory_id = str(memory_id or "")
            if (
                memory_id
                and memory_id in allowed
                and memory_id not in retained
                and len(retained) < bounded_limit
            ):
                retained.append(memory_id)

        # Proof/requirement lanes are strongest structural reservations.
        for memory_id in requirement_reserved:
            add(memory_id)

        # One discriminative representative per proposition.  A shared memory can satisfy
        # multiple lanes without consuming duplicate slots.
        for proposition_id in proposition_reserved:
            add(proposition_reserved[proposition_id])

        # Preserve the existing ProofContext ranking for the remaining capacity.  This is the
        # critical bug fix: retention signals may rescue an omitted evidence lane, but they do
        # not grow the final context or replace the owner's normal ordering wholesale.
        for memory_id in baseline:
            add(memory_id)
        for memory_id in candidate_order or []:
            add(memory_id)

        self._last_context_retention = {
            "version": self.CONTEXT_RETENTION_VERSION,
            "limit": bounded_limit,
            "baseline_ids": list(baseline),
            "requirement_reserved_ids": list(requirement_reserved),
            "proposition_reserved_ids": dict(proposition_reserved),
            "rescued_ids": [
                memory_id for memory_id in retained if memory_id not in baseline
            ],
            "dropped_baseline_ids": [
                memory_id for memory_id in baseline if memory_id not in retained
            ],
            "retained_ids": list(retained),
            "final_count": len(retained),
            "budget_unchanged": len(retained) <= bounded_limit,
            "semantics": "coverage_preserving_not_truth",
        }
        return retained

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
        self._last_context_retention = {}
        return super()._run_query_retrieval(
            question,
            initial_seeds,
            frame,
            fast_supports,
            gate,
            planning_seeds=planning_seeds,
            planning_context=planning_context,
        )

    def prepare_batch_query(self, question, system_message=None, **kwargs):
        prepared = super().prepare_batch_query(
            question, system_message=system_message, **kwargs
        )
        extra = prepared.setdefault("extra", {})
        extra["context_retention_version"] = self.CONTEXT_RETENTION_VERSION
        extra["context_retention"] = deepcopy(
            getattr(self, "_last_context_retention", {}) or {}
        )
        extra["proof_context_version"] = self.PROOF_CONTEXT_VERSION
        extra["context_selection_owner"] = self.PROOF_CONTEXT_VERSION
        extra["proof_context_single_owner"] = True
        return prepared
