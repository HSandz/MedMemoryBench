"""Multiple-choice retrieval invariants for the two-stage read path.

Visible options are probes into one shared participant-evidence bundle. Probe hits are
NOT verdicts: an option with zero personal-memory support can still be correct after
world-knowledge reasoning or elimination. The only deterministic guarantee here is fair,
label-preserving evidence exploration.
"""

from typing import Any, Dict, List

from .contracts import QueryFrame


class ReadOptionContractMixin:
    def _semantic_operation_search(self, query, top_k, strategy, frame=None, option_queries=None):
        strategy = str(strategy or "FOCAL").upper()
        if strategy != "SHARED_OPTIONS":
            return super()._semantic_operation_search(query, top_k, strategy, frame=frame, option_queries=option_queries)
        frame = frame or QueryFrame()
        eligible_ids = {memory["id"] for memory in self._memories if self._memory_satisfies_frame(memory, frame, include_entities=bool(frame.hard_entities)) and self._query_visible_memory(memory)}
        labels = []
        for index, item in enumerate(option_queries or []):
            label = str(item.get("label") if isinstance(item, dict) else index)
            labels.append(label)
        if not eligible_ids:
            self._last_option_probe_coverage = {label: [] for label in labels}
            return []
        base = self._hybrid_search(query, top_k=min(max(int(top_k) * 3, 12), len(eligible_ids)), candidate_ids=eligible_ids)
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
            hits = self._hybrid_search(option_text, top_k=min(4, len(eligible_ids)), candidate_ids=eligible_ids)
            coverage[label] = [memory["id"] for memory in hits[:3]]
            option_hits.extend(hits)
            representative = next((memory for memory in hits if memory["id"] not in representative_ids), None)
            if representative is not None:
                representatives.append(representative)
                representative_ids.add(representative["id"])
        self._last_option_probe_coverage = coverage
        selected, selected_ids = [], set()
        for memory in (*representatives, *base, *option_hits):
            if memory["id"] in selected_ids:
                continue
            selected.append(self._snapshot(memory))
            selected_ids.add(memory["id"])
            if len(selected) >= int(top_k):
                break
        return selected

    def _slot_covered(self, slot, support_ids, selected, relations):
        if str(slot.get("evidence_role") or "").upper() != "OPTION_CONTEXT":
            return super()._slot_covered(slot, support_ids, selected, relations)
        expected = {str(label) for label in (slot.get("option_labels") or [])}
        probed = {str(label) for label in (getattr(self, "_last_option_probe_coverage", {}) or {})}
        if not expected or not expected.issubset(probed):
            return False
        # Exploration completeness is not a verdict. At least one personal-memory
        # candidate must survive when such evidence exists, but any individual label may
        # legitimately have an empty hit list.
        support = set(support_ids or [])
        return any(memory.get("id") in support for memory in selected) or all(not ids for ids in (getattr(self, "_last_option_probe_coverage", {}) or {}).values())

    def _coverage_map(self, plan, slot_support, selected, relations):
        coverage = super()._coverage_map(plan, slot_support, selected, relations)
        options = plan.get("visible_options") or {}
        if not options:
            return coverage
        probed = {str(label) for label in (getattr(self, "_last_option_probe_coverage", {}) or {})}
        if not {str(label) for label in options}.issubset(probed):
            return {slot_id: False for slot_id in coverage}
        return coverage

    def _operation_slot_support(self, slot, result, relations):
        if str(slot.get("evidence_role") or "").upper() != "OPTION_CONTEXT":
            return super()._operation_slot_support(slot, result, relations)
        ordered, seen = [], set()
        for memory in result or []:
            if memory["id"] in seen or not self._rc_owner_match(slot, memory):
                continue
            if not self._rc_option_probe_labels_for_memory(memory["id"]):
                continue
            ordered.append(memory)
            seen.add(memory["id"])
        return ordered[:8]

    def prepare_batch_query(self, question, system_message=None, **kwargs):
        prepared = super().prepare_batch_query(question, system_message=system_message, **kwargs)
        extra = prepared.setdefault("extra", {})
        coverage = {str(label): list(ids or []) for label, ids in (getattr(self, "_last_option_probe_coverage", {}) or {}).items()}
        extra["option_evidence_views"] = coverage
        if coverage:
            lines = ["\n=== OPTION EVIDENCE VIEWS ===", "Each label maps to participant-memory candidates only. An empty list means no personal-memory support was found; it does NOT mean the option is false. Evaluate every option against the question using all authorized memory plus any permitted world-knowledge bridge before choosing labels."]
            for label, memory_ids in coverage.items():
                lines.append(f"- {label}: {memory_ids if memory_ids else '[]'}")
            addendum = "\n".join(lines)
            for message in prepared.get("messages") or []:
                if message.get("role") == "system":
                    message["content"] = str(message.get("content") or "") + addendum
                    break
        return prepared
