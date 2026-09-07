"""Candidate-proposition evidence for the two-stage read path.

Visible options are only one producer of candidate propositions. The generic primitive is
a bounded proposition->memory candidate packet plus LLM #1 semantic annotations. Retrieval
neighbors are never deterministic support. LLM #1 proposes SUPPORTS / CONTRADICTS /
CONTEXT_FOR / UNKNOWN with confidence that the relation label itself is correct; low
confidence abstains to UNKNOWN. LLM #2 remains the final proposition/option judge.
"""

from copy import deepcopy
from typing import Any, Dict, List

from .contracts import QueryFrame


class ReadOptionContractMixin:
    PROPOSITION_PACK_PER_CANDIDATE = 2
    PROPOSITION_PACK_GLOBAL_CAP = 8
    PROPOSITION_MIN_RELATION_CONFIDENCE = 0.50
    PROPOSITION_STRONG_RELATION_CONFIDENCE = 0.70
    VALID_PROPOSITION_RELATIONS = frozenset(
        {"SUPPORTS", "CONTRADICTS", "CONTEXT_FOR", "UNKNOWN"}
    )

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
            "semantic_role": str(memory.get("semantic_role") or ""),
            "stance": str(memory.get("stance") or "AFFIRM"),
            "assertion_mode": str(memory.get("assertion_mode") or "DIRECT"),
            "object_anchor": str(memory.get("object_anchor") or ""),
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

        per_proposition_hits: Dict[str, List[Dict[str, Any]]] = {}
        for proposition_id, proposition_text in propositions.items():
            if not eligible_ids:
                per_proposition_hits[proposition_id] = []
                continue
            hits = self._hybrid_search(
                proposition_text,
                top_k=min(
                    self.PROPOSITION_PACK_PER_CANDIDATE + 1, len(eligible_ids)
                ),
                candidate_ids=eligible_ids,
            )
            per_proposition_hits[proposition_id] = list(
                hits[: self.PROPOSITION_PACK_PER_CANDIDATE]
            )

        selected_by_id: Dict[str, Dict[str, Any]] = {}
        ref_by_memory_id: Dict[str, str] = {}
        proposition_refs = {proposition_id: [] for proposition_id in propositions}
        proposition_memory_ids = {
            proposition_id: [] for proposition_id in propositions
        }

        for depth in range(self.PROPOSITION_PACK_PER_CANDIDATE):
            for proposition_id in propositions:
                hits = per_proposition_hits.get(proposition_id, [])
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
                if ref not in proposition_refs[proposition_id]:
                    proposition_refs[proposition_id].append(ref)
                if memory_id not in proposition_memory_ids[proposition_id]:
                    proposition_memory_ids[proposition_id].append(memory_id)

        proposition_ids_by_memory: Dict[str, List[str]] = {}
        for proposition_id, memory_ids in proposition_memory_ids.items():
            for memory_id in memory_ids:
                proposition_ids_by_memory.setdefault(memory_id, []).append(
                    proposition_id
                )

        candidates = [
            self._candidate_proposition_memory_payload(
                selected_by_id[memory_id],
                ref_by_memory_id[memory_id],
                proposition_ids_by_memory.get(memory_id, []),
            )
            for memory_id in ref_by_memory_id
        ]
        pack = {
            "version": "candidate-proposition-pack-v1",
            "propositions": dict(propositions),
            "candidate_refs": {
                proposition_id: list(refs)
                for proposition_id, refs in proposition_refs.items()
            },
            "candidates": candidates,
            "limits": {
                "top_per_proposition": self.PROPOSITION_PACK_PER_CANDIDATE,
                "global_cap": self.PROPOSITION_PACK_GLOBAL_CAP,
                "seed_budget": 3,
            },
        }
        self._last_candidate_proposition_pack = deepcopy(pack)
        self._last_proposition_probe_coverage = {
            proposition_id: list(memory_ids)
            for proposition_id, memory_ids in proposition_memory_ids.items()
        }
        return pack

    @staticmethod
    def _bounded_confidence(value: Any) -> float:
        try:
            confidence = float(value)
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, min(1.0, confidence))

    def _normalize_candidate_proposition_evidence(
        self,
        raw_evidence: Any,
        pack: Dict[str, Any],
        seeds,
        propositions: Dict[str, Any],
    ) -> Dict[str, List[Dict[str, Any]]]:
        propositions = self._normalize_candidate_propositions(propositions)
        raw_evidence = raw_evidence if isinstance(raw_evidence, dict) else {}

        ref_to_memory_id = {}
        refs_by_proposition = {
            proposition_id: set()
            for proposition_id in propositions
        }
        for candidate in (pack or {}).get("candidates") or []:
            ref = str(candidate.get("ref") or "")
            memory_id = str(candidate.get("memory_id") or "")
            if ref and memory_id:
                ref_to_memory_id[ref] = memory_id
        for proposition_id, refs in ((pack or {}).get("candidate_refs") or {}).items():
            if proposition_id in refs_by_proposition:
                refs_by_proposition[proposition_id].update(
                    str(ref) for ref in (refs or []) if str(ref)
                )

        seed_refs = {}
        for index, memory in enumerate((seeds or [])[:3]):
            memory_id = str(memory.get("id") or "")
            if memory_id:
                seed_refs[f"$seed{index}"] = memory_id
        ref_to_memory_id.update(seed_refs)

        normalized: Dict[str, List[Dict[str, Any]]] = {
            proposition_id: [] for proposition_id in propositions
        }
        for proposition_id in propositions:
            entries = raw_evidence.get(proposition_id)
            if not isinstance(entries, list):
                continue
            seen_refs = set()
            for raw in entries[:8]:
                if not isinstance(raw, dict):
                    continue
                memory_ref = str(
                    raw.get("memory_ref") or raw.get("ref") or ""
                ).strip()
                if not memory_ref or memory_ref in seen_refs:
                    continue
                if memory_ref not in ref_to_memory_id:
                    continue
                if (
                    memory_ref not in seed_refs
                    and memory_ref not in refs_by_proposition.get(proposition_id, set())
                ):
                    continue
                seen_refs.add(memory_ref)

                proposed = str(raw.get("relation") or "UNKNOWN").upper()
                if proposed not in self.VALID_PROPOSITION_RELATIONS:
                    proposed = "UNKNOWN"
                confidence = self._bounded_confidence(raw.get("confidence"))

                relation = proposed
                accepted = False
                if proposed == "UNKNOWN":
                    relation = "UNKNOWN"
                    status = "ABSTAIN_UNKNOWN"
                elif confidence < self.PROPOSITION_MIN_RELATION_CONFIDENCE:
                    relation = "UNKNOWN"
                    status = "ABSTAIN_LOW_CONFIDENCE"
                elif (
                    proposed in {"SUPPORTS", "CONTRADICTS"}
                    and confidence < self.PROPOSITION_STRONG_RELATION_CONFIDENCE
                ):
                    relation = "UNKNOWN"
                    status = "ABSTAIN_TENTATIVE_STRONG_RELATION"
                else:
                    relation = proposed
                    accepted = True
                    status = "ACCEPTED"

                normalized[proposition_id].append(
                    {
                        "memory_ref": memory_ref,
                        "memory_id": ref_to_memory_id[memory_ref],
                        "proposed_relation": proposed,
                        "relation": relation,
                        "confidence": round(confidence, 3),
                        "accepted": accepted,
                        "status": status,
                    }
                )
        return normalized

    def _activate_candidate_proposition_evidence(
        self,
        propositions: Dict[str, Any],
        evidence: Dict[str, List[Dict[str, Any]]],
        *,
        visible_options: bool = False,
        predicate: str = "",
    ) -> None:
        propositions = self._normalize_candidate_propositions(propositions)
        evidence = {
            proposition_id: [dict(item) for item in (evidence.get(proposition_id) or [])]
            for proposition_id in propositions
        }
        self._last_candidate_propositions = dict(propositions)
        self._last_proposition_relation_views = deepcopy(evidence)
        self._last_proposition_support_views = {
            proposition_id: [
                dict(item)
                for item in items
                if item.get("relation") == "SUPPORTS" and item.get("accepted")
            ]
            for proposition_id, items in evidence.items()
        }
        self._last_proposition_contradict_views = {
            proposition_id: [
                dict(item)
                for item in items
                if item.get("relation") == "CONTRADICTS" and item.get("accepted")
            ]
            for proposition_id, items in evidence.items()
        }
        self._last_proposition_context_views = {
            proposition_id: [
                dict(item)
                for item in items
                if item.get("relation") == "CONTEXT_FOR" and item.get("accepted")
            ]
            for proposition_id, items in evidence.items()
        }
        self._last_proposition_unknown_views = {
            proposition_id: [
                dict(item)
                for item in items
                if item.get("relation") == "UNKNOWN"
            ]
            for proposition_id, items in evidence.items()
        }
        self._last_proposition_semantics = {"predicate": str(predicate or "").strip()}

        if visible_options:
            self._last_option_probe_relations = deepcopy(
                self._last_proposition_relation_views
            )
            self._last_option_support_views = deepcopy(
                self._last_proposition_support_views
            )
            self._last_option_contradict_views = deepcopy(
                self._last_proposition_contradict_views
            )
            self._last_option_semantics = deepcopy(self._last_proposition_semantics)
            self._last_option_probe_coverage = {
                proposition_id: list(
                    (getattr(self, "_last_proposition_probe_coverage", {}) or {}).get(
                        proposition_id, []
                    )
                )
                for proposition_id in propositions
            }

    def _option_memory_relation(
        self,
        option_text: str,
        memory: Dict[str, Any],
        rank: int = 0,
        semantics: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """Compatibility lookup: never infer evidence stance deterministically."""
        del rank, semantics
        proposition_id = next(
            (
                key
                for key, text in (
                    getattr(self, "_last_candidate_propositions", {}) or {}
                ).items()
                if self._rc_text(text) == self._rc_text(option_text)
            ),
            "",
        )
        memory_id = str(memory.get("id") or "")
        for item in (
            (getattr(self, "_last_proposition_relation_views", {}) or {}).get(
                proposition_id, []
            )
            if proposition_id
            else []
        ):
            if str(item.get("memory_id") or "") == memory_id:
                return dict(item)
        return {
            "memory_id": memory_id,
            "proposed_relation": "UNKNOWN",
            "relation": "UNKNOWN",
            "confidence": 0.0,
            "accepted": False,
            "status": "UNANNOTATED",
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
            return []

        base = self._hybrid_search(
            query,
            top_k=min(max(int(top_k) * 3, 12), len(eligible_ids)),
            candidate_ids=eligible_ids,
        )
        representatives: List[Dict[str, Any]] = []
        option_hits: List[Dict[str, Any]] = []
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
            option_hits.extend(hits)
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

        self._last_option_probe_coverage = coverage
        strong_ids = self._evidence_memory_ids(
            getattr(self, "_last_proposition_support_views", {}) or {},
            getattr(self, "_last_proposition_contradict_views", {}) or {},
        )
        context_ids = self._evidence_memory_ids(
            getattr(self, "_last_proposition_context_views", {}) or {}
        )
        packet_ids = [
            str(item.get("memory_id") or "")
            for item in (
                (getattr(self, "_last_candidate_proposition_pack", {}) or {}).get(
                    "candidates"
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
            *(by_id[memory_id] for memory_id in strong_ids if memory_id in by_id),
            *(by_id[memory_id] for memory_id in context_ids if memory_id in by_id),
            *(by_id[memory_id] for memory_id in packet_ids if memory_id in by_id),
            *representatives,
            *base,
            *option_hits,
        ]
        for memory in ordered:
            memory_id = str(memory.get("id") or "")
            if not memory_id or memory_id in selected_ids:
                continue
            if memory_id not in eligible_ids:
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
        priority_ids = self._evidence_memory_ids(
            getattr(self, "_last_proposition_support_views", {}) or {},
            getattr(self, "_last_proposition_contradict_views", {}) or {},
            getattr(self, "_last_proposition_context_views", {}) or {},
        )
        by_id = {memory.get("id"): memory for memory in (result or []) if memory.get("id")}
        for memory_id in priority_ids:
            memory = by_id.get(memory_id)
            if memory and self._rc_owner_match(slot, memory):
                ordered.append(memory)
                seen.add(memory_id)
        for memory in result or []:
            memory_id = str(memory.get("id") or "")
            if memory_id in seen or not self._rc_owner_match(slot, memory):
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
        pack = deepcopy(getattr(self, "_last_candidate_proposition_pack", {}) or {})
        relations = deepcopy(
            getattr(self, "_last_proposition_relation_views", {}) or {}
        )
        supports = deepcopy(
            getattr(self, "_last_proposition_support_views", {}) or {}
        )
        contradictions = deepcopy(
            getattr(self, "_last_proposition_contradict_views", {}) or {}
        )
        contexts = deepcopy(
            getattr(self, "_last_proposition_context_views", {}) or {}
        )
        unknowns = deepcopy(
            getattr(self, "_last_proposition_unknown_views", {}) or {}
        )
        semantics = dict(
            getattr(self, "_last_proposition_semantics", {}) or {}
        )
        extra["candidate_propositions"] = propositions
        extra["candidate_proposition_pack"] = pack
        extra["proposition_relation_views"] = relations
        extra["proposition_support_views"] = supports
        extra["proposition_contradict_views"] = contradictions
        extra["proposition_context_views"] = contexts
        extra["proposition_unknown_views"] = unknowns
        extra["proposition_semantics"] = semantics

        if self._question_options(question) or {}:
            extra["option_evidence_views"] = {
                str(label): list(ids or [])
                for label, ids in (
                    getattr(self, "_last_option_probe_coverage", {}) or {}
                ).items()
            }
            extra["option_relation_views"] = deepcopy(relations)
            extra["option_support_views"] = deepcopy(supports)
            extra["option_contradict_views"] = deepcopy(contradictions)
            extra["option_semantics"] = dict(semantics)

        if not propositions:
            return prepared

        final_ids = set(extra.get("final_memory_ids") or [])
        predicate = str(semantics.get("predicate") or "").strip()
        lines = [
            "\n=== CANDIDATE PROPOSITION EVIDENCE ===",
            (
                f"Normalized selection predicate: {predicate}"
                if predicate
                else "Normalized selection predicate: unspecified."
            ),
            "These relations were proposed by semantic controller LLM #1 from the "
            "bounded candidate packet and Top-3 seeds. Confidence is P(the relation "
            "label is correct), NOT P(the proposition is correct). SUPPORTS and "
            "CONTRADICTS are accepted only at high confidence; lower-confidence strong "
            "claims abstain to UNKNOWN. Re-check every annotation against the raw memory. "
            "CONTEXT_FOR/UNKNOWN and an empty support list are not verdicts, and no "
            "personal-memory support does NOT mean a proposition is false.",
        ]
        for proposition_id, proposition_text in propositions.items():
            items = relations.get(proposition_id) or []
            if final_ids:
                items = [
                    item
                    for item in items
                    if str(item.get("memory_id") or "") in final_ids
                ]
            rendered = ", ".join(
                (
                    f"{item.get('memory_id','') or '-'}:"
                    f"{item.get('relation','UNKNOWN')}"
                    f"@{float(item.get('confidence', 0.0)):.2f}"
                    + (
                        f"(proposed={item.get('proposed_relation')})"
                        if item.get("proposed_relation") != item.get("relation")
                        else ""
                    )
                )
                for item in items
            ) or "[]"
            lines.append(
                f"- {proposition_id} ({proposition_text}): evidence=[{rendered}]"
            )

        addendum = "\n".join(lines)
        for message in prepared.get("messages") or []:
            if message.get("role") == "system":
                message["content"] = str(message.get("content") or "") + addendum
                break
        return prepared
