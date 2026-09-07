"""Multiple-choice retrieval invariants for the two-stage read path.

Visible options are probes into one shared participant-evidence bundle. Probe hits are
NOT verdicts: an option with zero personal-memory evidence can still be correct after
world-knowledge reasoning or elimination. This layer preserves label-specific retrieval
relations and confidence without adding an LLM call.
"""

from typing import Any, Dict, List

from .contracts import QueryFrame


class ReadOptionContractMixin:
    def _option_memory_relation(
        self, option_text: str, memory: Dict[str, Any], rank: int
    ) -> Dict[str, Any]:
        text = self._rc_memory_target_text(memory)
        explicit = self._rc_token_sequence_present(option_text, text)
        similarity_fn = getattr(self, "_rq_surface_similarity", None)
        similarity = (
            float(similarity_fn(option_text, text))
            if callable(similarity_fn)
            else 0.0
        )
        semantic_role = str(memory.get("semantic_role") or "").upper()
        if explicit and semantic_role == "SAFETY_CONSTRAINT":
            relation, confidence = "SAFETY_CONSTRAINT_ON_OPTION", 1.0
        elif explicit:
            relation, confidence = "EXPLICIT_OPTION_MENTION", 0.96
        else:
            relation = "RELATED_NEIGHBOR"
            confidence = max(0.20, min(0.80, max(similarity, 0.55 - 0.08 * rank)))
        return {
            "memory_id": str(memory.get("id") or ""),
            "relation": relation,
            "confidence": round(confidence, 3),
        }

    def _semantic_operation_search(
        self, query, top_k, strategy, frame=None, option_queries=None
    ):
        strategy = str(strategy or "FOCAL").upper()
        if strategy != "SHARED_OPTIONS":
            return super()._semantic_operation_search(
                query,
                top_k,
                strategy,
                frame=frame,
                option_queries=option_queries,
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
        labels = []
        for index, item in enumerate(option_queries or []):
            label = str(item.get("label") if isinstance(item, dict) else index)
            labels.append(label)
        if not eligible_ids:
            self._last_option_probe_coverage = {label: [] for label in labels}
            self._last_option_probe_relations = {
                label: [{"relation": "NO_PERSONAL_MEMORY", "confidence": 0.0}]
                for label in labels
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
        relation_map: Dict[str, List[Dict[str, Any]]] = {}
        for index, item in enumerate(option_queries or []):
            if isinstance(item, dict):
                label = str(item.get("label") or index)
                option_text = str(
                    item.get("query") or item.get("text") or ""
                ).strip()
            else:
                label, option_text = str(index), str(item or "").strip()
            if not option_text:
                coverage[label] = []
                relation_map[label] = [
                    {"relation": "NO_PERSONAL_MEMORY", "confidence": 0.0}
                ]
                continue
            hits = self._hybrid_search(
                option_text,
                top_k=min(4, len(eligible_ids)),
                candidate_ids=eligible_ids,
            )
            probe_hits = hits[:3]
            coverage[label] = [memory["id"] for memory in probe_hits]
            relation_map[label] = [
                self._option_memory_relation(option_text, memory, rank)
                for rank, memory in enumerate(probe_hits)
            ] or [{"relation": "NO_PERSONAL_MEMORY", "confidence": 0.0}]
            option_hits.extend(hits)
            representative = next(
                (
                    memory
                    for memory in hits
                    if memory["id"] not in representative_ids
                ),
                None,
            )
            if representative is not None:
                representatives.append(representative)
                representative_ids.add(representative["id"])
        self._last_option_probe_coverage = coverage
        self._last_option_probe_relations = relation_map
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
        probed = {
            str(label)
            for label in (
                getattr(self, "_last_option_probe_coverage", {}) or {}
            )
        }
        if not expected or not expected.issubset(probed):
            return False
        support = set(support_ids or [])
        return any(memory.get("id") in support for memory in selected) or all(
            not ids
            for ids in (
                getattr(self, "_last_option_probe_coverage", {}) or {}
            ).values()
        )

    def _coverage_map(self, plan, slot_support, selected, relations):
        coverage = super()._coverage_map(
            plan, slot_support, selected, relations
        )
        options = plan.get("visible_options") or {}
        if not options:
            return coverage
        probed = {
            str(label)
            for label in (
                getattr(self, "_last_option_probe_coverage", {}) or {}
            )
        }
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
        prepared = super().prepare_batch_query(
            question, system_message=system_message, **kwargs
        )
        extra = prepared.setdefault("extra", {})
        coverage = {
            str(label): list(ids or [])
            for label, ids in (
                getattr(self, "_last_option_probe_coverage", {}) or {}
            ).items()
        }
        relations = {
            str(label): [dict(item) for item in items or []]
            for label, items in (
                getattr(self, "_last_option_probe_relations", {}) or {}
            ).items()
        }
        extra["option_evidence_views"] = coverage
        extra["option_relation_views"] = relations
        if coverage:
            lines = [
                "\n=== OPTION RETRIEVAL RELATIONS ===",
                "These are retrieval relations, NOT option verdicts. "
                "SAFETY_CONSTRAINT_ON_OPTION means a structured safety memory explicitly "
                "mentions the option; EXPLICIT_OPTION_MENTION means direct textual/entity "
                "presence; RELATED_NEIGHBOR means only semantic retrieval proximity. "
                "Confidence measures relation-to-memory strength, NOT probability that the "
                "option is correct. NO_PERSONAL_MEMORY does NOT mean FALSE. Evaluate every "
                "option against the question, all authorized participant evidence, and any "
                "permitted world-knowledge bridge before choosing labels.",
            ]
            for label in coverage:
                items = relations.get(label) or [
                    {"relation": "NO_PERSONAL_MEMORY", "confidence": 0.0}
                ]
                rendered = ", ".join(
                    (
                        f"{item.get('memory_id','') or '-'}:"
                        f"{item.get('relation','RELATED_NEIGHBOR')}"
                        f"@{float(item.get('confidence', 0.0)):.2f}"
                    )
                    for item in items
                )
                lines.append(f"- {label}: {rendered}")
            addendum = "\n".join(lines)
            for message in prepared.get("messages") or []:
                if message.get("role") == "system":
                    message["content"] = (
                        str(message.get("content") or "") + addendum
                    )
                    break
        return prepared
