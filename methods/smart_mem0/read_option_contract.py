"""Generic CandidateSet retrieval for SmartMem0 READ.

CandidateSet is structural: candidates are propositions to evaluate, never memory facts.
This layer provides bounded per-candidate recall and coverage lineage only. It does not ask
LLM #1 to label SUPPORTS/CONTRADICTS, attach confidence, or choose a final candidate.
LLM #2 evaluates candidates against the raw evidence that survives ProofContext.
"""

from copy import deepcopy
from typing import Any, Dict, List

from .contracts import QueryFrame


class ReadOptionContractMixin:
    """Compatibility-named mixin implementing the generic CandidateSet primitive."""

    PROPOSITION_PACK_PER_CANDIDATE = 2
    PROPOSITION_PACK_GLOBAL_CAP = 8

    @classmethod
    def _normalize_candidate_propositions(cls, values: Any) -> Dict[str, str]:
        if not isinstance(values, dict):
            return {}
        output: Dict[str, str] = {}
        for raw_id, raw_text in values.items():
            proposition_id = str(raw_id or "").strip()[:48]
            text = " ".join(str(raw_text or "").split()).strip()
            if proposition_id and text and proposition_id not in output:
                output[proposition_id] = text[:320]
        return output

    def _candidate_propositions_from_visible_options(
        self, options: Dict[str, Any]
    ) -> Dict[str, str]:
        """Visible options are only one adapter into the generic CandidateSet."""
        return self._normalize_candidate_propositions(options)

    def _candidate_proposition_memory_payload(
        self, memory: Dict[str, Any], ref: str, proposition_ids: List[str]
    ) -> Dict[str, Any]:
        return {
            "ref": ref,
            "memory_id": str(memory.get("id") or ""),
            "proposition_ids": list(proposition_ids),
            "kind": memory.get("kind"),
            "claim": str(memory.get("claim") or "")[:260],
            "value": str(self._memory_value(memory) or "")[:140],
            "object_anchor": str(memory.get("object_anchor") or "")[:120],
            "event_time": memory.get("event_time"),
            "document_time": memory.get("document_time"),
            "status": memory.get(
                "_status", self._belief_status.get(memory.get("id"), "active")
            ),
        }

    def _build_candidate_proposition_pack(
        self,
        propositions: Dict[str, Any],
        frame=None,
        seeds=None,
        question: str = "",
    ) -> Dict[str, Any]:
        """Bounded structural recall. Packet memories are never sent to LLM #1."""
        del question
        propositions = self._normalize_candidate_propositions(propositions)
        self._last_candidate_propositions = dict(propositions)
        self._last_candidate_proposition_pack = {}
        self._last_proposition_probe_coverage = {
            proposition_id: [] for proposition_id in propositions
        }
        if not propositions:
            return {}

        frame = frame or QueryFrame()
        seed_ids = {
            str(memory.get("id") or "")
            for memory in (seeds or [])[:3]
            if memory.get("id")
        }
        eligible_ids = {
            memory["id"]
            for memory in getattr(self, "_memories", []) or []
            if memory.get("id")
            and memory.get("id") not in seed_ids
            and self._memory_satisfies_frame(
                memory,
                frame,
                include_entities=False,
            )
            and self._query_visible_memory(memory)
        }

        per_candidate_hits: Dict[str, List[Dict[str, Any]]] = {}
        for proposition_id, proposition_text in propositions.items():
            if not eligible_ids:
                per_candidate_hits[proposition_id] = []
                continue
            hits = self._hybrid_search(
                proposition_text,
                top_k=min(
                    self.PROPOSITION_PACK_PER_CANDIDATE + 1,
                    len(eligible_ids),
                ),
                candidate_ids=eligible_ids,
            )
            per_candidate_hits[proposition_id] = list(
                hits[: self.PROPOSITION_PACK_PER_CANDIDATE]
            )

        selected_by_id: Dict[str, Dict[str, Any]] = {}
        ref_by_memory_id: Dict[str, str] = {}
        candidate_refs = {candidate_id: [] for candidate_id in propositions}
        coverage = {candidate_id: [] for candidate_id in propositions}

        for depth in range(self.PROPOSITION_PACK_PER_CANDIDATE):
            for candidate_id in propositions:
                hits = per_candidate_hits.get(candidate_id, [])
                if depth >= len(hits):
                    continue
                memory = hits[depth]
                memory_id = str(memory.get("id") or "")
                if not memory_id:
                    continue
                if memory_id not in ref_by_memory_id:
                    if len(ref_by_memory_id) >= self.PROPOSITION_PACK_GLOBAL_CAP:
                        continue
                    ref = f"$prop{len(ref_by_memory_id)}"
                    ref_by_memory_id[memory_id] = ref
                    selected_by_id[memory_id] = memory
                ref = ref_by_memory_id[memory_id]
                if ref not in candidate_refs[candidate_id]:
                    candidate_refs[candidate_id].append(ref)
                if memory_id not in coverage[candidate_id]:
                    coverage[candidate_id].append(memory_id)

        candidate_ids_by_memory: Dict[str, List[str]] = {}
        for candidate_id, memory_ids in coverage.items():
            for memory_id in memory_ids:
                candidate_ids_by_memory.setdefault(memory_id, []).append(candidate_id)

        candidates = [
            self._candidate_proposition_memory_payload(
                selected_by_id[memory_id],
                ref_by_memory_id[memory_id],
                candidate_ids_by_memory.get(memory_id, []),
            )
            for memory_id in ref_by_memory_id
        ]
        pack = {
            "version": "candidate-set-recall-v1",
            "candidate_set": dict(propositions),
            "propositions": dict(propositions),
            "candidate_refs": {
                candidate_id: list(refs)
                for candidate_id, refs in candidate_refs.items()
            },
            # Keep the historical packet key as a structural retrieval-view alias.
            "candidates": candidates,
            "retrieval_views": candidates,
            "limits": {
                "top_per_candidate": self.PROPOSITION_PACK_PER_CANDIDATE,
                "top_per_proposition": self.PROPOSITION_PACK_PER_CANDIDATE,
                "global_cap": self.PROPOSITION_PACK_GLOBAL_CAP,
                "seed_budget": 3,
            },
        }
        self._last_candidate_proposition_pack = deepcopy(pack)
        self._last_proposition_probe_coverage = {
            candidate_id: list(memory_ids)
            for candidate_id, memory_ids in coverage.items()
        }
        return pack

    def _option_memory_relation(
        self,
        option_text: str,
        memory: Dict[str, Any],
        rank: int = 0,
        semantics: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """Legacy compatibility: retrieval proximity never implies evidence stance."""
        del option_text, rank, semantics
        return {
            "memory_id": str(memory.get("id") or ""),
            "relation": "UNJUDGED",
            "accepted": False,
            "status": "CANDIDATESET_RETRIEVAL_ONLY",
        }

    @staticmethod
    def _evidence_memory_ids(*views: Dict[str, List[Dict[str, Any]]]) -> List[str]:
        output, seen = [], set()
        for view in views:
            for items in (view or {}).values():
                for item in items or []:
                    memory_id = str(item.get("memory_id") or "")
                    if memory_id and memory_id not in seen:
                        output.append(memory_id)
                        seen.add(memory_id)
        return output

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
        propositions = {}
        for index, item in enumerate(option_queries or []):
            if isinstance(item, dict):
                label = str(item.get("label") or index)
                text = str(item.get("query") or item.get("text") or "").strip()
            else:
                label, text = str(index), str(item or "").strip()
            if text:
                propositions[label] = text
        propositions = self._normalize_candidate_propositions(propositions)
        labels = list(propositions)

        eligible_ids = {
            memory["id"]
            for memory in getattr(self, "_memories", []) or []
            if memory.get("id")
            and self._memory_satisfies_frame(
                memory, frame, include_entities=False
            )
            and self._query_visible_memory(memory)
        }
        precoverage = getattr(self, "_last_proposition_probe_coverage", {}) or {}
        coverage = {
            label: list(precoverage.get(label, []))
            for label in labels
        }
        if not eligible_ids:
            self._last_option_probe_coverage = coverage
            self._last_proposition_probe_coverage = deepcopy(coverage)
            return []

        base = self._hybrid_search(
            query,
            top_k=min(max(int(top_k) * 3, 12), len(eligible_ids)),
            candidate_ids=eligible_ids,
        )
        representatives: List[Dict[str, Any]] = []
        candidate_hits: List[Dict[str, Any]] = []
        representative_ids = set()

        for label, proposition_text in propositions.items():
            hits = self._hybrid_search(
                proposition_text,
                top_k=min(4, len(eligible_ids)),
                candidate_ids=eligible_ids,
            )
            current_ids = list(coverage.get(label, []))
            for memory in hits[:3]:
                if memory.get("id") and memory["id"] not in current_ids:
                    current_ids.append(memory["id"])
            coverage[label] = current_ids
            candidate_hits.extend(hits)
            representative = next(
                (
                    memory
                    for memory in hits
                    if memory.get("id") not in representative_ids
                ),
                None,
            )
            if representative is not None:
                representatives.append(representative)
                representative_ids.add(representative["id"])

        self._last_option_probe_coverage = deepcopy(coverage)
        self._last_proposition_probe_coverage = deepcopy(coverage)

        packet_ids = [
            str(item.get("memory_id") or "")
            for item in (
                (getattr(self, "_last_candidate_proposition_pack", {}) or {}).get(
                    "retrieval_views"
                )
                or []
            )
            if str(item.get("memory_id") or "")
        ]
        by_id = {
            memory["id"]: memory
            for memory in getattr(self, "_memories", []) or []
            if memory.get("id")
        }

        selected, selected_ids = [], set()
        ordered = [
            *(by_id[memory_id] for memory_id in packet_ids if memory_id in by_id),
            *representatives,
            *base,
            *candidate_hits,
        ]
        for memory in ordered:
            memory_id = str(memory.get("id") or "")
            if not memory_id or memory_id in selected_ids or memory_id not in eligible_ids:
                continue
            selected.append(self._snapshot(memory))
            selected_ids.add(memory_id)
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
        coverage = super()._coverage_map(plan, slot_support, selected, relations)
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
            memory_id = str(memory.get("id") or "")
            if (
                not memory_id
                or memory_id in seen
                or not self._rc_owner_match(slot, memory)
            ):
                continue
            if not self._rc_option_probe_labels_for_memory(memory_id):
                continue
            ordered.append(memory)
            seen.add(memory_id)
        return ordered[:8]

    def prepare_batch_query(self, question, system_message=None, **kwargs):
        prepared = super().prepare_batch_query(
            question, system_message=system_message, **kwargs
        )
        extra = prepared.setdefault("extra", {})
        propositions = dict(
            getattr(self, "_last_candidate_propositions", {}) or {}
        )
        pack = deepcopy(
            getattr(self, "_last_candidate_proposition_pack", {}) or {}
        )
        coverage = deepcopy(
            getattr(self, "_last_proposition_probe_coverage", {}) or {}
        )

        candidate_set = {
            "version": "candidate-set-v1",
            "candidates": propositions,
            "coverage": coverage,
            "coverage_semantics": "retrieval_only_not_verdict",
        }
        extra["candidate_set"] = deepcopy(candidate_set)
        # Historical telemetry aliases; semantic relation views are intentionally empty.
        extra["candidate_propositions"] = propositions
        extra["candidate_proposition_pack"] = pack
        extra["proposition_relation_views"] = {}
        extra["proposition_support_views"] = {}
        extra["proposition_contradict_views"] = {}
        extra["proposition_context_views"] = {}
        extra["proposition_unknown_views"] = {}
        extra["proposition_semantics"] = {}

        if self._question_options(question) or {}:
            extra["option_evidence_views"] = {
                str(label): list(ids or [])
                for label, ids in (
                    getattr(self, "_last_option_probe_coverage", {}) or {}
                ).items()
            }
            extra["option_relation_views"] = {}
            extra["option_support_views"] = {}
            extra["option_contradict_views"] = {}
            extra["option_semantics"] = {}

        if not propositions:
            return prepared

        final_ids = set(extra.get("final_memory_ids") or [])
        lines = [
            "\n=== CANDIDATE SET ===",
            (
                "Candidates are propositions to evaluate against the QUESTION predicate. "
                "retrieved_memory_ids are retrieval coverage only, never support/"
                "contradiction labels. Empty coverage does not make a candidate false. "
                "Evaluate every candidate from the raw evidence that follows."
            ),
        ]
        for candidate_id, proposition_text in propositions.items():
            ids = list(coverage.get(candidate_id, []))
            if final_ids:
                ids = [memory_id for memory_id in ids if memory_id in final_ids]
            lines.append(
                f"- {candidate_id} ({proposition_text}): "
                f"retrieved_memory_ids={ids}"
            )
        addendum = "\n".join(lines)
        for message in prepared.get("messages") or []:
            if message.get("role") == "system":
                message["content"] = str(message.get("content") or "") + addendum
                break
        return prepared
