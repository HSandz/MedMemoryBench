"""Deterministic evidence-precision policy for the locked SmartMem0 READ path.

This Stage-D layer does not add a model call and does not classify query types. It preserves
relevance order for non-temporal synthesis, gives CandidateSet probes both the shared predicate
and the proposition while keeping shared/local lanes separate, broadens numeric temporal
anchors without broadening final context, and materializes a compact obligation map for LLM #2.
"""

import re
from copy import deepcopy
from typing import Any, Dict, List, Sequence


class ReadEvidencePrecisionMixin:
    PRECISION_POLICY_VERSION = "evidence-precision-v1"
    NUMERIC_TEMPORAL_BACKUP_K = 32
    NUMERIC_TEMPORAL_OUTPUT_CAP = 20

    @classmethod
    def _precision_unique_text(cls, values) -> List[str]:
        output = []
        seen = set()
        for value in values:
            text = " ".join(str(value or "").split()).strip()
            if not text:
                continue
            key = cls._rc_text(text)
            if key and key not in seen:
                output.append(text)
                seen.add(key)
        return output

    @staticmethod
    def _context_time_axis(slots: List[Dict[str, Any]]):
        """Chronology is a requested projection, not the default answer-context order."""
        axes = list(
            dict.fromkeys(
                str(slot.get("time_axis") or "")
                for slot in slots
                if slot.get("type") == "TEMPORAL" and slot.get("time_axis")
            )
        )
        return axes[0] if len(axes) == 1 else None

    def _semantic_operation_search(
        self, query, top_k, strategy, frame=None, option_queries=None
    ):
        """Probe each candidate with shared predicate + proposition, then let lanes dedupe."""
        if str(strategy or "").upper() != "SHARED_OPTIONS" or not option_queries:
            return super()._semantic_operation_search(
                query,
                top_k,
                strategy,
                frame=frame,
                option_queries=option_queries,
            )

        conditioned = []
        for index, item in enumerate(option_queries or []):
            if isinstance(item, dict):
                current = dict(item)
                label = str(current.get("label") or index)
                proposition = str(
                    current.get("query") or current.get("text") or ""
                ).strip()
            else:
                label = str(index)
                proposition = str(item or "").strip()
                current = {"label": label}
            current["label"] = label
            current["query"] = " | ".join(
                self._precision_unique_text((query, proposition))
            )
            conditioned.append(current)

        return super()._semantic_operation_search(
            query,
            top_k,
            strategy,
            frame=frame,
            option_queries=conditioned,
        )

    @staticmethod
    def _precision_numbers(value: Any) -> set:
        return {
            token.replace(",", ".")
            for token in re.findall(r"(?<!\d)\d+(?:[.,]\d+)?", str(value or ""))
        }

    def _locate_temporal_family(self, query, frame, axis):
        """Add numeric semantic aliases to an extremum pool, never to final context directly.

        Doses, measurements and counts often have multiple atom identities even though they
        denote the same real-world variable. Numeric anchors are structural enough to recall
        those aliases without introducing a query-type or language-specific rule.
        """
        base = list(super()._locate_temporal_family(query, frame, axis))
        query_numbers = self._precision_numbers(query)
        if not query_numbers:
            return base

        eligible = [
            memory
            for memory in getattr(self, "_memories", []) or []
            if self._memory_satisfies_frame(
                memory,
                frame,
                include_dates=False,
                include_entities=bool(getattr(frame, "hard_entities", ()) or ()),
            )
            and self._query_visible_memory(memory, include_history=True)
            and self._date_for(memory, axis)
        ]
        eligible_ids = {
            str(memory.get("id") or "") for memory in eligible if memory.get("id")
        }
        if not eligible_ids:
            return base

        ranked = self._hybrid_search(
            query,
            top_k=min(self.NUMERIC_TEMPORAL_BACKUP_K, len(eligible_ids)),
            candidate_ids=eligible_ids,
        )
        numeric = []
        for memory in ranked:
            text = " ".join(
                str(value or "")
                for value in (
                    memory.get("claim"),
                    self._memory_value(memory),
                    memory.get("verbatim_value"),
                    memory.get("object_anchor"),
                    memory.get("state_key"),
                )
            )
            if query_numbers.issubset(self._precision_numbers(text)):
                numeric.append(memory)
        if not numeric:
            return base

        best_dense = max(float(memory.get("_dense_score", 0.0) or 0.0) for memory in numeric)
        best_overlap = max(int(memory.get("_overlap", 0) or 0) for memory in numeric)
        gated = [
            memory
            for memory in numeric
            if float(memory.get("_dense_score", 0.0) or 0.0) >= best_dense - 0.16
            and int(memory.get("_overlap", 0) or 0) >= max(1, best_overlap - 2)
        ] or numeric[:1]

        ordered = []
        seen = set()

        def add(memory):
            memory_id = str((memory or {}).get("id") or "")
            if memory_id and memory_id not in seen:
                ordered.append(self._snapshot(memory))
                seen.add(memory_id)

        for memory in base:
            add(memory)
        for memory in gated:
            add(memory)
        return ordered[: self.NUMERIC_TEMPORAL_OUTPUT_CAP]

    @staticmethod
    def _precision_selector_text(requirement: Dict[str, Any]) -> str:
        selector = dict(
            requirement.get("selector")
            or requirement.get("time_constraint")
            or {}
        )
        relation = str(selector.get("relation") or "").upper()
        axis = str(selector.get("axis") or "")
        anchor = str(selector.get("anchor") or "")
        end = str(selector.get("end") or "")
        if not relation and not axis:
            return "NONE"
        parts = [relation or "LOCATE", axis or "event_time"]
        if anchor:
            parts.append(f"anchor={anchor}")
        if end:
            parts.append(f"end={end}")
        return " ".join(parts)

    def _evidence_obligation_block(self, prepared: Dict[str, Any]) -> str:
        extra = prepared.get("extra") or {}
        plan = extra.get("replan") or extra.get("plan") or {}
        semantic_ir = plan.get("semantic_ir") or {}
        requirements = list(semantic_ir.get("requirements") or [])
        if not requirements:
            return ""

        final = {
            str(memory.get("id") or ""): memory
            for memory in (prepared.get("retrieved_memories") or [])
            if memory.get("id")
        }
        selected_map = (
            extra.get("requirement_context_selected")
            or extra.get("requirement_context_candidates")
            or {}
        )
        lines = [
            "=== EVIDENCE OBLIGATION MAP ===",
            "Requirements describe evidence to evaluate, not facts that are automatically true.",
            "For non-temporal synthesis, evidence order is relevance/role order; do not prefer a "
            "memory merely because it is newer. Use chronology only when a selector or bridge asks for it.",
        ]
        for requirement in requirements:
            rid = str(requirement.get("id") or "")
            obligation = " ".join(
                str(
                    requirement.get("answer_obligation")
                    or requirement.get("focus_span")
                    or requirement.get("target")
                    or rid
                ).split()
            )[:220]
            selector_text = self._precision_selector_text(requirement)
            lines.append(f"- {rid}: {obligation}; selector={selector_text}")
            ids = [
                str(memory_id)
                for memory_id in (selected_map.get(rid) or [])
                if str(memory_id) in final
            ]
            for memory_id in ids[:2]:
                claim = " ".join(str(final[memory_id].get("claim") or "").split())[:240]
                lines.append(f"  evidence: {claim}")

        lines.extend(
            [
                "Selector semantics: LOCATE means choose the evidence whose proposition best "
                "matches the obligation on the requested axis; it does NOT mean latest or earliest. "
                "EARLIEST/LATEST are the only extremum selectors.",
                "For decision/inference, combine all answer-sensitive grounded requirements before "
                "applying an authorized bridge. Do not replace a missing participant variable with "
                "generic advice or a generic domain story.",
            ]
        )
        if (extra.get("candidate_set") or {}).get("candidates"):
            lines.append(
                "For multi-select CandidateSet questions, evaluate every candidate independently "
                "and select all satisfying candidates. Shared and candidate-local evidence may be "
                "combined; no single memory must cover every clause of a candidate."
            )
        return "\n".join(lines)

    def prepare_batch_query(self, question, system_message=None, **kwargs):
        prepared = super().prepare_batch_query(
            question, system_message=system_message, **kwargs
        )
        extra = prepared.setdefault("extra", {})
        extra["precision_policy_version"] = self.PRECISION_POLICY_VERSION
        explicit_axis = self._context_time_axis(
            [
                slot
                for candidate_plan in (
                    extra.get("plan") or {},
                    extra.get("replan") or {},
                )
                for slot in candidate_plan.get("required_slots", [])
            ]
        )
        extra["context_order_semantics"] = (
            f"explicit_temporal_axis:{explicit_axis}"
            if explicit_axis
            else "relevance_unless_explicit_temporal_axis"
        )

        block = self._evidence_obligation_block(prepared)
        if block:
            for message in prepared.get("messages") or []:
                if str(message.get("role") or "").lower() == "system":
                    message["content"] = (
                        str(message.get("content") or "").rstrip()
                        + "\n\n"
                        + block
                    )
                    break
            extra["evidence_obligation_map_materialized"] = True
        else:
            extra["evidence_obligation_map_materialized"] = False
        return prepared
