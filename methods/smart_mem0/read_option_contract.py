"""Multiple-choice evidential relations for the two-stage read path.

Visible options are propositions evaluated against one normalized selection predicate.
Retrieval neighbors are not support. This layer assigns deterministic SUPPORTS or
CONTRADICTS only when a retrieved memory both binds the option strongly and has a
canonical semantic role that the controller declared evidential for that predicate.
Ambiguous evidence remains CONTEXT_FOR or UNKNOWN. No extra LLM call is used.
"""

from typing import Any, Dict, List

from .contracts import QueryFrame


class ReadOptionContractMixin:
    @staticmethod
    def _option_semantic_roles(semantics: Dict[str, Any]):
        semantics = semantics if isinstance(semantics, dict) else {}
        support = {
            str(role or "").upper()
            for role in (semantics.get("support_roles") or [])
            if str(role or "").strip()
        }
        contradict = {
            str(role or "").upper()
            for role in (semantics.get("contradict_roles") or [])
            if str(role or "").strip()
        }
        overlap = support & contradict
        return support - overlap, contradict - overlap

    def _option_identity_strength(
        self, option_text: str, memory: Dict[str, Any]
    ) -> float:
        option = str(option_text or "").strip()
        if not option:
            return 0.0
        atomic_surfaces = [
            memory.get("value"),
            memory.get("verbatim_value"),
            str(memory.get("object_anchor") or "").replace("_", " "),
            *(memory.get("entities") or []),
            *(memory.get("scope_entities") or []),
        ]
        for surface in atomic_surfaces:
            if surface and (
                self._rc_token_sequence_present(option, surface)
                or self._rc_token_sequence_present(surface, option)
            ):
                return 1.0
        claim = str(memory.get("claim") or "")
        if claim and self._rc_token_sequence_present(option, claim):
            return 0.94
        similarity_fn = getattr(self, "_rq_surface_similarity", None)
        if callable(similarity_fn):
            similarity = float(
                similarity_fn(option, self._rc_memory_target_text(memory))
            )
            if similarity >= 0.90:
                return 0.86
            if similarity >= 0.75:
                return 0.70
        return 0.0

    def _option_memory_relation(
        self,
        option_text: str,
        memory: Dict[str, Any],
        rank: int,
        semantics: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        semantics = (
            semantics
            if isinstance(semantics, dict)
            else getattr(self, "_last_option_semantics", {}) or {}
        )
        identity = self._option_identity_strength(option_text, memory)
        semantic_role = str(memory.get("semantic_role") or "").upper()
        support_roles, contradict_roles = self._option_semantic_roles(semantics)
        stance = str(memory.get("stance") or "AFFIRM").upper()
        assertion = str(memory.get("assertion_mode") or "DIRECT").upper()
        relation = "UNKNOWN"
        confidence = max(0.10, min(0.55, 0.42 - 0.07 * int(rank)))

        strong_identity = identity >= 0.86
        direct_assertion = assertion in {"DIRECT", "RECAP"}
        if strong_identity and direct_assertion:
            supports = semantic_role in support_roles
            contradicts = semantic_role in contradict_roles
            if supports ^ contradicts:
                relation = "SUPPORTS" if supports else "CONTRADICTS"
                if stance == "NEGATE":
                    relation = (
                        "CONTRADICTS" if relation == "SUPPORTS" else "SUPPORTS"
                    )
                confidence = min(
                    0.99,
                    0.64 + 0.25 * identity + (0.08 if semantic_role else 0.0),
                )
            else:
                relation = "CONTEXT_FOR"
                confidence = min(0.90, 0.55 + 0.30 * identity)
        elif identity >= 0.70:
            relation = "CONTEXT_FOR"
            confidence = min(0.78, 0.46 + 0.32 * identity)

        return {
            "memory_id": str(memory.get("id") or ""),
            "relation": relation,
            "confidence": round(float(confidence), 3),
            "identity_strength": round(float(identity), 3),
            "semantic_role": semantic_role,
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
        semantics = getattr(self, "_last_option_semantics", {}) or {}
        if not eligible_ids:
            self._last_option_probe_coverage = {label: [] for label in labels}
            self._last_option_probe_relations = {
                label: [{"relation": "NO_PERSONAL_MEMORY", "confidence": 0.0}]
                for label in labels
            }
            self._last_option_support_views = {label: [] for label in labels}
            self._last_option_contradict_views = {label: [] for label in labels}
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
        support_map: Dict[str, List[Dict[str, Any]]] = {}
        contradict_map: Dict[str, List[Dict[str, Any]]] = {}

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
                support_map[label] = []
                contradict_map[label] = []
                continue

            hits = self._hybrid_search(
                option_text,
                top_k=min(4, len(eligible_ids)),
                candidate_ids=eligible_ids,
            )
            probe_hits = hits[:3]
            coverage[label] = [memory["id"] for memory in probe_hits]
            relations = [
                self._option_memory_relation(
                    option_text, memory, rank, semantics=semantics
                )
                for rank, memory in enumerate(probe_hits)
            ]
            relation_map[label] = relations or [
                {"relation": "NO_PERSONAL_MEMORY", "confidence": 0.0}
            ]
            support_map[label] = [
                dict(item)
                for item in relations
                if item.get("relation") == "SUPPORTS"
            ]
            contradict_map[label] = [
                dict(item)
                for item in relations
                if item.get("relation") == "CONTRADICTS"
            ]
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
        self._last_option_support_views = support_map
        self._last_option_contradict_views = contradict_map

        stance_ids = []
        for label in labels:
            for item in [
                *(support_map.get(label, []) or []),
                *(contradict_map.get(label, []) or []),
            ]:
                memory_id = str(item.get("memory_id") or "")
                if memory_id and memory_id not in stance_ids:
                    stance_ids.append(memory_id)
        by_id = {memory["id"]: memory for memory in (*option_hits, *base)}
        selected, selected_ids = [], set()
        ordered = [
            *(by_id[memory_id] for memory_id in stance_ids if memory_id in by_id),
            *representatives,
            *base,
            *option_hits,
        ]
        for memory in ordered:
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
        stance_ids = {
            str(item.get("memory_id") or "")
            for items in [
                *((getattr(self, "_last_option_support_views", {}) or {}).values()),
                *((getattr(self, "_last_option_contradict_views", {}) or {}).values()),
            ]
            for item in items
            if str(item.get("memory_id") or "")
        }
        for memory in result or []:
            if memory.get("id") in stance_ids and memory.get("id") not in seen:
                if self._rc_owner_match(slot, memory):
                    ordered.append(memory)
                    seen.add(memory["id"])
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
        supports = {
            str(label): [dict(item) for item in items or []]
            for label, items in (
                getattr(self, "_last_option_support_views", {}) or {}
            ).items()
        }
        contradictions = {
            str(label): [dict(item) for item in items or []]
            for label, items in (
                getattr(self, "_last_option_contradict_views", {}) or {}
            ).items()
        }
        semantics = dict(getattr(self, "_last_option_semantics", {}) or {})
        extra["option_evidence_views"] = coverage
        extra["option_relation_views"] = relations
        extra["option_support_views"] = supports
        extra["option_contradict_views"] = contradictions
        extra["option_semantics"] = semantics

        if coverage:
            predicate = str(semantics.get("predicate") or "").strip()
            lines = [
                "\n=== OPTION EVIDENTIAL STANCE ===",
                (
                    f"Normalized selection predicate: {predicate}"
                    if predicate
                    else "Normalized selection predicate: unspecified; treat all deterministic stances conservatively."
                ),
                "SUPPORTS/CONTRADICTS are deterministic high-confidence relations between "
                "a memory and an option AS AN ANSWER TO THE normalized predicate. They "
                "require strong option identity plus a controller-authorized canonical "
                "memory role. CONTEXT_FOR and UNKNOWN are not verdicts. Confidence is "
                "confidence in the memory↔option evidential relation, NOT probability that "
                "the option is correct. An empty support list or NO_PERSONAL_MEMORY does "
                "NOT mean FALSE; world knowledge or elimination may still make the option "
                "correct. Evaluate every option before outputting labels.",
            ]
            for label in coverage:
                relation_items = relations.get(label) or [
                    {"relation": "NO_PERSONAL_MEMORY", "confidence": 0.0}
                ]
                support_items = supports.get(label) or []
                contradict_items = contradictions.get(label) or []
                rendered_relations = ", ".join(
                    (
                        f"{item.get('memory_id','') or '-'}:"
                        f"{item.get('relation','UNKNOWN')}"
                        f"@{float(item.get('confidence', 0.0)):.2f}"
                    )
                    for item in relation_items
                )
                rendered_support = ", ".join(
                    (
                        f"{item.get('memory_id','')}:"
                        f"SUPPORTS@{float(item.get('confidence', 0.0)):.2f}"
                    )
                    for item in support_items
                ) or "[]"
                rendered_contradict = ", ".join(
                    (
                        f"{item.get('memory_id','')}:"
                        f"CONTRADICTS@{float(item.get('confidence', 0.0)):.2f}"
                    )
                    for item in contradict_items
                ) or "[]"
                lines.append(
                    f"- {label}: supports=[{rendered_support}] ; "
                    f"contradicts=[{rendered_contradict}] ; "
                    f"context=[{rendered_relations}]"
                )
            addendum = "\n".join(lines)
            for message in prepared.get("messages") or []:
                if message.get("role") == "system":
                    message["content"] = str(message.get("content") or "") + addendum
                    break
        return prepared
