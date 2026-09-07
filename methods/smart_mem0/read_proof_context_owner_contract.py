"""Final answer-context ownership and evidence lineage for SmartMem0 READ.

Retrieval may be broad, proof remains conservative, and this layer is the single final
ordering policy for the bounded memory context passed to LLM #2. It also preserves
requirement/proposition lineage so evidence that was successfully retrieved cannot become
an anonymous global candidate before context arbitration.
"""

from copy import deepcopy
from typing import Any, Dict, Iterable, List, Sequence


class ReadProofContextOwnerContractMixin:
    CONTEXT_OWNER_VERSION = "semantic-lineage-context-v1"

    @staticmethod
    def _lineage_unique(values: Iterable[str]) -> List[str]:
        output, seen = [], set()
        for value in values:
            value = str(value or "")
            if value and value not in seen:
                output.append(value)
                seen.add(value)
        return output

    def _proposition_priority_map(self) -> Dict[str, List[str]]:
        relations = getattr(self, "_last_proposition_relation_views", {}) or {}
        coverage = getattr(self, "_last_proposition_probe_coverage", {}) or {}
        propositions = getattr(self, "_last_candidate_propositions", {}) or {}
        priority: Dict[str, List[str]] = {}
        for proposition_id in propositions:
            items = list(relations.get(proposition_id) or [])
            strong = [
                str(item.get("memory_id") or "")
                for item in items
                if item.get("accepted")
                and item.get("relation") in {"SUPPORTS", "CONTRADICTS"}
            ]
            contextual = [
                str(item.get("memory_id") or "")
                for item in items
                if item.get("accepted") and item.get("relation") == "CONTEXT_FOR"
            ]
            unknown = [
                str(item.get("memory_id") or "")
                for item in items
                if item.get("relation") == "UNKNOWN"
            ]
            priority[proposition_id] = self._lineage_unique(
                [*strong, *contextual, *unknown, *(coverage.get(proposition_id) or [])]
            )
        return priority

    def _safe_proposition_context_ids(
        self, run: Dict[str, Any], initial_seeds: Sequence[Dict[str, Any]]
    ) -> List[str]:
        allowed = set(run.get("operation_output_ids") or [])
        for memory in (
            *(run.get("planning_seeds") or []),
            *(initial_seeds or []),
        ):
            if memory and memory.get("id"):
                allowed.add(str(memory["id"]))
        priority = self._proposition_priority_map()
        ordered: List[str] = []
        max_depth = max((len(values) for values in priority.values()), default=0)
        for depth in range(max_depth):
            for proposition_id in priority:
                values = priority[proposition_id]
                if depth < len(values):
                    memory_id = values[depth]
                    if memory_id in allowed and memory_id not in ordered:
                        ordered.append(memory_id)
        return ordered[: self.HARD_MEMORY_LIMIT]

    def _build_evidence_lineage(
        self, run: Dict[str, Any], proposition_context_ids: Sequence[str]
    ) -> Dict[str, Any]:
        plan = run.get("plan") or {}
        trace = list(run.get("trace") or [])
        requirement_status = dict(run.get("requirement_status") or {})
        context_candidates = dict(run.get("requirement_context_candidates") or {})
        proof_support = dict(run.get("requirement_proof_support") or {})
        slot_support = dict(run.get("slot_support") or {})

        requirements: Dict[str, Dict[str, Any]] = {}
        memory_index: Dict[str, Dict[str, Any]] = {}

        def memory_entry(memory_id: str) -> Dict[str, Any]:
            return memory_index.setdefault(
                memory_id,
                {
                    "requirement_ids": [],
                    "proposition_ids": [],
                    "retrieval_rounds": [],
                    "operations": [],
                    "relations": [],
                },
            )

        for slot in plan.get("required_slots") or []:
            slot_id = str(slot.get("id") or "")
            views = []
            for item in trace:
                if slot_id not in (item.get("produces") or []):
                    continue
                output_ids = self._lineage_unique(item.get("output_ids") or [])
                views.append(
                    {
                        "retrieval_round": item.get("retrieval_round"),
                        "operation": item.get("operation"),
                        "output_ids": output_ids,
                    }
                )
                for memory_id in output_ids:
                    entry = memory_entry(memory_id)
                    if slot_id not in entry["requirement_ids"]:
                        entry["requirement_ids"].append(slot_id)
                    if item.get("retrieval_round") not in entry["retrieval_rounds"]:
                        entry["retrieval_rounds"].append(item.get("retrieval_round"))
                    operation = str(item.get("operation") or "")
                    if operation and operation not in entry["operations"]:
                        entry["operations"].append(operation)
            requirements[slot_id] = {
                "status": requirement_status.get(slot_id, ""),
                "proof_ids": self._lineage_unique(proof_support.get(slot_id) or []),
                "context_candidate_ids": self._lineage_unique(
                    context_candidates.get(slot_id) or []
                ),
                "support_ids": self._lineage_unique(slot_support.get(slot_id) or []),
                "views": views,
            }

        proposition_relations = (
            getattr(self, "_last_proposition_relation_views", {}) or {}
        )
        propositions: Dict[str, Dict[str, Any]] = {}
        priority = self._proposition_priority_map()
        for proposition_id, proposition_text in (
            getattr(self, "_last_candidate_propositions", {}) or {}
        ).items():
            relations = [
                dict(item) for item in proposition_relations.get(proposition_id) or []
            ]
            candidate_ids = self._lineage_unique(priority.get(proposition_id) or [])
            propositions[proposition_id] = {
                "text": proposition_text,
                "candidate_ids": candidate_ids,
                "relations": relations,
            }
            for item in relations:
                memory_id = str(item.get("memory_id") or "")
                if not memory_id:
                    continue
                entry = memory_entry(memory_id)
                if proposition_id not in entry["proposition_ids"]:
                    entry["proposition_ids"].append(proposition_id)
                descriptor = {
                    "proposition_id": proposition_id,
                    "relation": item.get("relation", "UNKNOWN"),
                    "proposed_relation": item.get("proposed_relation", "UNKNOWN"),
                    "confidence": item.get("confidence", 0.0),
                    "accepted": bool(item.get("accepted")),
                }
                if descriptor not in entry["relations"]:
                    entry["relations"].append(descriptor)

        return {
            "version": self.CONTEXT_OWNER_VERSION,
            "requirements": requirements,
            "propositions": propositions,
            "proposition_context_ids": list(proposition_context_ids),
            "memories": memory_index,
        }

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
        proposition_ids = self._safe_proposition_context_ids(run, initial_seeds)
        if proposition_ids and run.get("fast_supports") is None:
            slot_support = run.setdefault("slot_support", {})
            pool_key = getattr(self, "CONTEXT_POOL_KEY", "__answer_context_candidates__")
            pool = slot_support.setdefault(pool_key, [])
            for memory_id in proposition_ids:
                if memory_id not in pool:
                    pool.append(memory_id)
        lineage = self._build_evidence_lineage(run, proposition_ids)
        self._last_evidence_lineage = deepcopy(lineage)
        self._last_proposition_context_candidate_ids = list(proposition_ids)
        run["evidence_lineage"] = lineage
        run["proposition_context_candidates"] = list(proposition_ids)
        run["context_selection_owner"] = self.CONTEXT_OWNER_VERSION
        return run

    def _role_aware_support_ids(
        self,
        slots: List[Dict[str, Any]],
        slot_support: Dict[str, List[str]],
        candidate_order: List[str],
        limit: int,
    ) -> List[str]:
        """Single final ordering policy for answer-memory selection."""
        bounded_limit = max(0, int(limit))
        if not bounded_limit:
            return []
        allowed = set(candidate_order)
        selected: List[str] = []

        def add(memory_id: str) -> bool:
            memory_id = str(memory_id or "")
            if (
                memory_id
                and memory_id in allowed
                and memory_id not in selected
                and len(selected) < bounded_limit
            ):
                selected.append(memory_id)
                return True
            return False

        unique_slots, seen_slots = [], set()
        for slot in slots:
            slot_id = str(slot.get("id") or "")
            if slot_id and slot_id not in seen_slots:
                unique_slots.append(slot)
                seen_slots.add(slot_id)

        requirement_candidates = (
            getattr(self, "_last_requirement_context_candidates", {}) or {}
        )
        proof_support = getattr(self, "_last_requirement_proof_support", {}) or {}
        statuses = getattr(self, "_last_requirement_status", {}) or {}

        # 1) One answer-bearing candidate per semantic requirement/comparand.
        for slot in unique_slots:
            slot_id = str(slot.get("id") or "")
            semantic_slot = bool(
                callable(getattr(self, "_semantic_context_slot", None))
                and self._semantic_context_slot(slot)
            )
            if not semantic_slot:
                continue
            preferred = (
                list(proof_support.get(slot_id) or [])
                if statuses.get(slot_id) == "FOUND"
                else []
            )
            preferred.extend(requirement_candidates.get(slot_id) or [])
            add(next((memory_id for memory_id in preferred if memory_id in allowed), ""))

        # 2) One candidate per explicit proposition, preferring accepted stance.
        proposition_priority = self._proposition_priority_map()
        for proposition_id in proposition_priority:
            add(
                next(
                    (
                        memory_id
                        for memory_id in proposition_priority[proposition_id]
                        if memory_id in allowed
                    ),
                    "",
                )
            )

        # 3) Preserve typed supports for non-semantic slots too.
        for slot in unique_slots:
            slot_id = str(slot.get("id") or "")
            add(
                next(
                    (
                        memory_id
                        for memory_id in (slot_support.get(slot_id) or [])
                        if memory_id in allowed
                    ),
                    "",
                )
            )

        # 4) Fair round-robin over remaining requirement views.
        remaining_requirement_views = {
            str(slot.get("id") or ""): [
                memory_id
                for memory_id in requirement_candidates.get(
                    str(slot.get("id") or ""), []
                )
                if memory_id in allowed
            ]
            for slot in unique_slots
        }
        max_requirement_depth = max(
            (len(values) for values in remaining_requirement_views.values()), default=0
        )
        for depth in range(1, max_requirement_depth):
            for slot in unique_slots:
                values = remaining_requirement_views.get(str(slot.get("id") or ""), [])
                if depth < len(values):
                    add(values[depth])

        # 5) Then additional proposition evidence, still round-robin.
        max_prop_depth = max(
            (len(values) for values in proposition_priority.values()), default=0
        )
        for depth in range(1, max_prop_depth):
            for proposition_id in proposition_priority:
                values = proposition_priority[proposition_id]
                if depth < len(values):
                    add(values[depth])

        # 6) Finally preserve any typed supports and authorized candidate order.
        for slot in unique_slots:
            for memory_id in slot_support.get(str(slot.get("id") or ""), []) or []:
                add(memory_id)
        for memory_id in candidate_order:
            add(memory_id)
        return selected

    def prepare_batch_query(self, question, system_message=None, **kwargs):
        prepared = super().prepare_batch_query(
            question, system_message=system_message, **kwargs
        )
        extra = prepared.setdefault("extra", {})
        final_ids = set(extra.get("final_memory_ids") or [])
        lineage = deepcopy(getattr(self, "_last_evidence_lineage", {}) or {})
        for memory_id, entry in (lineage.get("memories") or {}).items():
            entry["selected_for_answer"] = memory_id in final_ids
        extra["evidence_lineage"] = lineage
        extra["context_selection_owner"] = self.CONTEXT_OWNER_VERSION
        extra["proposition_context_candidates"] = list(
            getattr(self, "_last_proposition_context_candidate_ids", []) or []
        )
        extra["context_owner_selected_ids"] = sorted(final_ids)
        extra["family_recall_contract_version"] = getattr(
            self, "FAMILY_RECALL_VERSION", ""
        )
        return prepared
