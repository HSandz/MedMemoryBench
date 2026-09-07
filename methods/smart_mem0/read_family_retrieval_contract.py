"""Selector-neutral evidence-family recall for SmartMem0 reads.

EARLIEST/LATEST are selectors, not retrieval intent. This contract marks only temporal
extremum anchor operations for bounded family recall, gathers plausible durable families,
preserves endpoints on the requested time axis, and then lets the existing TEMPORAL_FILTER
choose the extremum. Other LOCATE_ANCHOR uses (causal anchors, trajectories, legacy flows)
retain their existing behavior.
"""

import re
from typing import Any, Dict, List, Tuple

from .canonicalization import state_identity
from .contracts import QueryFrame, VALID_TEMPORAL_AXES


class ReadFamilyRetrievalContractMixin:
    FAMILY_RECALL_VERSION = "selector-neutral-family-v1"
    FAMILY_ROOT_LIMIT = 3
    FAMILY_BASE_K = 24
    FAMILY_OUTPUT_LIMIT = 16

    @classmethod
    def _family_number_tokens(cls, value: Any) -> set:
        return {
            token.replace(",", ".")
            for token in re.findall(r"(?<!\d)\d+(?:[.,]\d+)?", str(value or ""))
        }

    def _family_memory_text(self, memory: Dict[str, Any]) -> str:
        return " ".join(
            str(value or "")
            for value in (
                memory.get("claim"),
                self._memory_value(memory),
                memory.get("verbatim_value"),
                memory.get("scope"),
                memory.get("state_key"),
                memory.get("object_anchor"),
                " ".join(memory.get("entities") or []),
                " ".join(memory.get("scope_entities") or []),
            )
        )

    def _family_identity(self, memory: Dict[str, Any]) -> Tuple[str, ...]:
        """Return a durable, language-neutral family identity when one exists."""
        identity = state_identity(memory)
        if identity:
            return ("STATE", str(identity))
        owner = self._rc_owner(
            memory.get("subject_id") or memory.get("subject") or ""
        )
        scope = self._rc_text(memory.get("scope") or "")
        object_anchor = self._rc_text(memory.get("object_anchor") or "")
        state_key = self._rc_text(memory.get("state_key") or "")
        if object_anchor:
            return ("OBJECT", owner, scope, object_anchor)
        if state_key:
            return ("STATE_KEY", owner, scope, state_key)
        return ("MEMORY", str(memory.get("id") or ""))

    def _family_root_related(
        self,
        root: Dict[str, Any],
        best: Dict[str, Any],
        *,
        best_dense: float,
        best_overlap: int,
    ) -> bool:
        if root.get("id") == best.get("id"):
            return True
        root_scope = self._rc_text(root.get("scope") or "")
        best_scope = self._rc_text(best.get("scope") or "")
        root_object = self._rc_text(root.get("object_anchor") or "")
        best_object = self._rc_text(best.get("object_anchor") or "")
        if (
            root_object
            and best_object
            and root_object == best_object
            and root_scope == best_scope
        ):
            return True
        root_key = self._rc_text(root.get("state_key") or "")
        best_key = self._rc_text(best.get("state_key") or "")
        if root_key and best_key and root_key == best_key and root_scope == best_scope:
            return True
        dense = float(root.get("_dense_score", 0.0) or 0.0)
        overlap = int(root.get("_overlap", 0) or 0)
        # Scores are comparable because every root came from this same retrieval view.
        return dense >= best_dense - 0.10 and overlap >= max(0, best_overlap - 1)

    def _compile_gap_operations(self, slots, question, budget_tier="MEDIUM", plan=None):
        operations = super()._compile_gap_operations(
            slots, question, budget_tier, plan=plan
        )
        # Mark only LOCATE_ANCHOR operations that feed an EARLIEST/LATEST filter.
        for index, operation in enumerate(operations):
            if operation.get("op") != "LOCATE_ANCHOR":
                continue
            ref = f"${index}"
            selector = next(
                (
                    candidate
                    for candidate in operations
                    if candidate.get("op") == "TEMPORAL_FILTER"
                    and ref in (candidate.get("candidate_refs") or [])
                    and str(candidate.get("relation") or "").upper()
                    in {"EARLIEST", "LATEST"}
                ),
                None,
            )
            if selector:
                operation["family_recall"] = True
                operation["family_selector"] = str(
                    selector.get("relation") or ""
                ).upper()
                operation["family_axis"] = str(
                    selector.get("axis") or "event_time"
                ).lower()
        return operations

    def _locate_temporal_family(
        self,
        query: str,
        frame: QueryFrame,
        axis: str,
    ) -> List[Dict[str, Any]]:
        """Recall a bounded semantic family without applying the extremum selector."""
        axis = axis if axis in VALID_TEMPORAL_AXES else "event_time"
        eligible = [
            memory
            for memory in self._memories
            if self._memory_satisfies_frame(
                memory,
                frame,
                include_dates=False,
                include_entities=bool(getattr(frame, "hard_entities", ()) or ()),
            )
            and self._query_visible_memory(memory, include_history=True)
        ]
        if not eligible:
            return []

        eligible_ids = {memory["id"] for memory in eligible if memory.get("id")}
        ranked = self._hybrid_search(
            query,
            top_k=min(self.FAMILY_BASE_K, len(eligible_ids)),
            candidate_ids=eligible_ids,
        )
        if not ranked:
            return []

        query_numbers = self._family_number_tokens(query)
        if query_numbers:
            numeric_ranked = [
                memory
                for memory in ranked
                if query_numbers.issubset(
                    self._family_number_tokens(self._family_memory_text(memory))
                )
            ]
            if numeric_ranked:
                ranked = numeric_ranked

        best = ranked[0]
        best_dense = float(best.get("_dense_score", 0.0) or 0.0)
        best_overlap = int(best.get("_overlap", 0) or 0)
        root_families: List[Tuple[str, ...]] = []
        root_by_family: Dict[Tuple[str, ...], Dict[str, Any]] = {}
        for memory in ranked[:8]:
            if not self._family_root_related(
                memory,
                best,
                best_dense=best_dense,
                best_overlap=best_overlap,
            ):
                continue
            family = self._family_identity(memory)
            if family in root_by_family:
                continue
            root_families.append(family)
            root_by_family[family] = memory
            if len(root_families) >= self.FAMILY_ROOT_LIMIT:
                break
        if not root_families:
            root_families = [self._family_identity(best)]
            root_by_family[root_families[0]] = best

        members: Dict[Tuple[str, ...], List[Dict[str, Any]]] = {
            family: [] for family in root_families
        }
        for memory in eligible:
            family = self._family_identity(memory)
            if family not in members:
                continue
            if str(memory.get("assertion_mode") or "DIRECT").upper() == "INFERRED":
                continue
            if query_numbers and not query_numbers.issubset(
                self._family_number_tokens(self._family_memory_text(memory))
            ):
                continue
            members[family].append(memory)

        def axis_date(memory: Dict[str, Any]) -> str:
            return self._date_for(memory, axis) or ""

        for family in root_families:
            members[family].sort(
                key=lambda memory: (
                    axis_date(memory) or "9999-99-99",
                    str(memory.get("id") or ""),
                )
            )

        selected: List[Dict[str, Any]] = []
        selected_ids = set()

        def add(memory: Dict[str, Any]) -> None:
            memory_id = str((memory or {}).get("id") or "")
            if (
                memory_id
                and memory_id not in selected_ids
                and len(selected) < self.FAMILY_OUTPUT_LIMIT
            ):
                selected.append(self._snapshot(memory))
                selected_ids.add(memory_id)

        # Preserve both temporal ends across plausible families before topical rank.
        # The downstream TEMPORAL_FILTER decides EARLIEST versus LATEST.
        for family in root_families:
            dated = [memory for memory in members[family] if axis_date(memory)]
            if dated:
                add(dated[0])
        for family in root_families:
            dated = [memory for memory in members[family] if axis_date(memory)]
            if dated:
                add(dated[-1])
        for family in root_families:
            add(root_by_family[family])
        for memory in ranked:
            add(memory)
        return selected

    def _execute_operation(self, operation, outputs, seeds, frame=QueryFrame()):
        if operation.get("op") == "LOCATE_ANCHOR" and operation.get("family_recall"):
            result = self._locate_temporal_family(
                str(operation.get("query") or ""),
                frame,
                str(operation.get("family_axis") or "event_time"),
            )
            excluded = {
                str(memory_id) for memory_id in operation.get("exclude_ids", [])
            }
            if excluded:
                result = [
                    memory for memory in result if memory.get("id") not in excluded
                ]
            return result, [], []
        return super()._execute_operation(operation, outputs, seeds, frame)
