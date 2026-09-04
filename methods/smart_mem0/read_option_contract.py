"""Multiple-choice retrieval invariants for the two-stage read path.

Visible options are probes into one shared participant-evidence bundle. They are
never required memory facts and their probe ordering must survive support
packing; otherwise a broad stem such as "current condition" can replace useful
option-specific candidates with topical noise.
"""

from typing import Any, Dict, List

from .contracts import QueryFrame


class ReadOptionContractMixin:
    def _semantic_operation_search(
        self, query, top_k, strategy, frame=None, option_queries=None
    ):
        strategy = str(strategy or "FOCAL").upper()
        if strategy != "SHARED_OPTIONS":
            return super()._semantic_operation_search(
                query, top_k, strategy, frame=frame, option_queries=option_queries
            )

        frame = frame or QueryFrame()
        eligible_ids = {
            memory["id"]
            for memory in self._memories
            if self._memory_satisfies_frame(
                memory, frame, include_entities=bool(frame.hard_entities)
            )
            and self._query_visible_memory(memory)
        }
        if not eligible_ids:
            # The probe was still executed for every visible option. Empty lists
            # mean no personal-memory candidate, not an unprobed/false option.
            self._last_option_probe_coverage = {
                str(item.get("label") if isinstance(item, dict) else index): []
                for index, item in enumerate(option_queries or [])
            }
            return []

        base = self._hybrid_search(
            query,
            top_k=min(max(int(top_k) * 3, 12), len(eligible_ids)),
            candidate_ids=eligible_ids,
        )
        representatives: List[Dict[str, Any]] = []
        option_hits: List[Dict[str, Any]] = []
        representative_ids = set()
        coverage: Dict[str, List[str]] = {}

        for index, item in enumerate(option_queries or []):
            if isinstance(item, dict):
                label = str(item.get("label") or index)
                option_text = str(item.get("query") or item.get("text") or "").strip()
            else:
                label, option_text = str(index), str(item or "").strip()
            if not option_text:
                coverage[label] = []
                continue

            # Probe the proposition itself. A broad stem prefix causes different
            # options to collapse onto the same generic memories.
            hits = self._hybrid_search(
                option_text,
                top_k=min(4, len(eligible_ids)),
                candidate_ids=eligible_ids,
            )
            coverage[label] = [memory["id"] for memory in hits[:3]]
            option_hits.extend(hits)
            representative = next(
                (memory for memory in hits if memory["id"] not in representative_ids),
                None,
            )
            if representative is not None:
                representatives.append(representative)
                representative_ids.add(representative["id"])

        self._last_option_probe_coverage = coverage

        # Preserve one distinct representative per option before the shared
        # stem and redundant option hits. Do not sort by "number of options
        # matched": generic noise often matches many probes and would be
        # promoted by that heuristic.
        selected: List[Dict[str, Any]] = []
        selected_ids = set()
        for memory in (*representatives, *base, *option_hits):
            if memory["id"] in selected_ids:
                continue
            selected.append(self._snapshot(memory))
            selected_ids.add(memory["id"])
            if len(selected) >= int(top_k):
                break
        return selected

    def _slot_covered(self, slot, support_ids, selected, relations):
        role = str(slot.get("evidence_role") or "").upper()
        if role != "OPTION_CONTEXT":
            return super()._slot_covered(slot, support_ids, selected, relations)

        # Exploration completeness and evidence coverage are different facts.
        # All visible labels must be probed, but probing alone cannot make the
        # evidence contract sufficient. At least one participant-specific
        # candidate returned by the shared operation must survive support
        # packing. An individual option is still allowed to have zero hits.
        expected = {str(label) for label in (slot.get("option_labels") or [])}
        probed = {
            str(label)
            for label in (getattr(self, "_last_option_probe_coverage", {}) or {})
        }
        if not expected or not expected.issubset(probed):
            return False
        support = set(support_ids or [])
        return any(memory.get("id") in support for memory in selected)

    def _coverage_map(self, plan, slot_support, selected, relations):
        coverage = super()._coverage_map(plan, slot_support, selected, relations)
        options = plan.get("visible_options") or {}
        if not options:
            return coverage
        expected = {str(label) for label in options}
        probed = {
            str(label)
            for label in (getattr(self, "_last_option_probe_coverage", {}) or {})
        }
        if not expected.issubset(probed):
            return {slot_id: False for slot_id in coverage}
        return coverage

    def _operation_slot_support(self, slot, result, relations):
        role = str(slot.get("evidence_role") or "").upper()
        if role != "OPTION_CONTEXT":
            return super()._operation_slot_support(slot, result, relations)
        if not result:
            return []

        # Preserve the SHARED_OPTIONS operation order. Re-ranking by a broad
        # target such as "current condition" would undo per-option diversity.
        ordered = []
        seen = set()
        for memory in result:
            if memory["id"] in seen or not self._rc_owner_match(slot, memory):
                continue
            if not self._rc_option_probe_labels_for_memory(memory["id"]):
                continue
            ordered.append(memory)
            seen.add(memory["id"])
        return ordered[:6]
