"""One read-only evidence checkpoint over a frozen candidate world.

This certifies support in stored evidence, not real-world truth. It cannot acquire,
drop, or mutate memories. Ambiguity is retained for the synthesis model.
"""

import re
import unicodedata

from .canonicalization import state_identity
from .contracts import VALID_TEMPORAL_AXES


class EvidenceCertificateMixin:
    @staticmethod
    def _ec_surface(value):
        return " ".join(
            unicodedata.normalize("NFKC", str(value or "")).casefold().split()
        )

    @staticmethod
    def _ec_text(value):
        return " ".join(
            re.findall(
                r"\w+(?:[./%+-]\w+)*",
                unicodedata.normalize("NFKC", str(value or ""))
                .casefold()
                .replace("_", " "),
            )
        )

    def _ec_identity(self, question, memory, advisory):
        """Bind exact question predicates to stored assertions, never retrieval scores.

        A name/value alone is not a predicate. Unknown paraphrases remain synthesis
        work: we do not equate allergy with an instruction to avoid a drug.
        """
        normalize = self._ec_text
        predicate = normalize(memory.get("state_key") or memory.get("predicate"))
        assertion = normalize(memory.get("claim"))
        excluded = {
            normalize(v)
            for v in [
                memory.get("subject"),
                memory.get("subject_id"),
                memory.get("scope"),
                memory.get("object_anchor"),
                memory.get("value"),
                memory.get("verbatim_value"),
                *(memory.get("entities") or []),
            ]
            if v
        }
        spans = [normalize(s) for s in advisory.get("focus_spans", []) if s in question]
        if not spans and normalize(question) == predicate:
            spans = [predicate]
        if not spans or not all(spans):
            return False, "NO_QUESTION_PREDICATE_BINDING"

        def contains(text, span):
            return bool(
                span and re.search(r"(?<!\w)" + re.escape(span) + r"(?!\w)", text)
            )

        object_bound = contains(
            normalize(question), normalize(memory.get("object_anchor"))
        )
        binding = any(
            s == predicate
            or (
                not any(contains(surface, s) for surface in excluded)
                and (
                    contains(assertion, s) or (object_bound and contains(predicate, s))
                )
            )
            for s in spans
        )
        qualifiers = all(
            contains(assertion, s) or contains(predicate, s) for s in spans
        )
        return bool(binding and qualifiers), (
            "EXACT_PREDICATE_AND_QUALIFIERS"
            if binding and qualifiers
            else "UNRESOLVED_PREDICATE_OR_QUALIFIER"
        )

    def _ec_projections(self, memory, projection, selector):
        if projection == "DATE":
            axis = selector.get("axis")
            return [self._date_for(memory, axis)] if axis in VALID_TEMPORAL_AXES else []
        if projection == "VALUE":
            return [str(memory.get("verbatim_value") or memory.get("value") or "")]
        if projection == "ENTITY":
            # Multiple possible entity fields are competition, not license to pick the
            # first entity (which is often the subject, not the requested object).
            anchor = memory.get("object_anchor")
            # Prefer the durable object for ENTITY; its dose/value is not a second entity.
            values = [anchor] if anchor else [memory.get("value")]
            return list(dict.fromkeys(str(v).replace("_", " ") for v in values if v))
        return []

    def _ec_selector(self, memories, selector):
        relation = selector.get("relation", "")
        if not relation:
            return memories, "NOT_REQUESTED"
        if relation == "CURRENT":
            heads = [
                m for m in memories if state_identity(m) and self._is_state_head(m)
            ]
            return heads, "RESOLVED" if heads else "NO_DURABLE_STATE_HEAD"
        axis = selector.get("axis")
        if axis not in VALID_TEMPORAL_AXES:
            return [], "MISSING_AXIS"
        dated = [(m, self._date_for(m, axis)) for m in memories]
        if any(not date for _, date in dated):
            # Missing timestamps cannot silently lose an extremum competition.
            if relation in {"EARLIEST", "LATEST"}:
                return [], "INCOMPLETE_CHRONOLOGY"
        dated = [(m, date) for m, date in dated if date]
        anchor = self._parse_date(selector.get("anchor", ""))
        end = self._parse_date(selector.get("end", ""))
        if relation in {"EARLIEST", "LATEST"} and dated:
            if len({len(date) for _, date in dated}) != 1:
                return [], "MIXED_TIME_PRECISION"
            target = (min if relation == "EARLIEST" else max)(date for _, date in dated)
            dated = [(m, date) for m, date in dated if date == target]
        elif relation == "EXACT":
            dated = [
                (m, date)
                for m, date in dated
                if anchor and self._date_matches(date, anchor)
            ]
        elif relation in {"BEFORE", "AFTER", "BETWEEN"}:
            if not anchor or any(len(date) != len(anchor) for _, date in dated):
                return [], "UNRESOLVED_ANCHOR_PRECISION"
            dated = [
                (m, date)
                for m, date in dated
                if (relation == "BEFORE" and date < anchor)
                or (relation == "AFTER" and date > anchor)
                or (
                    relation == "BETWEEN"
                    and end
                    and len(end) == len(date)
                    and anchor <= date <= end
                )
            ]
        return [m for m, _ in dated], "RESOLVED" if dated else "NO_MATCHING_TIME"

    def _evidence_certificate(self, question, world, advisory):
        projection = advisory["projection_hint"]
        selector = advisory["selector_hint"]
        evidence_ids = {e["id"] for e in self._evidence}
        known_subjects = {
            str(m.get("subject_id") or m.get("subject") or "") for m in self._memories
        }
        explicit_subjects = {
            s
            for s in known_subjects
            if s
            and re.search(
                r"(?<!\w)" + re.escape(s.replace("_", " ")) + r"(?!\w)", question, re.I
            )
        }
        for alias, subject in (getattr(self, "subject_aliases", {}) or {}).items():
            if subject in known_subjects and re.search(
                r"(?<!\w)" + re.escape(alias.replace("_", " ")) + r"(?!\w)",
                question,
                re.I,
            ):
                explicit_subjects.add(subject)
        eligible, rejected, contradicted = [], {}, False
        historical = bool(
            selector.get("relation") and selector.get("relation") != "CURRENT"
        )
        for memory in world:
            mid = memory["id"]
            status = self._belief_status.get(mid, memory.get("_status", "active"))
            reason = ""
            if (
                explicit_subjects
                and str(memory.get("subject_id") or memory.get("subject") or "")
                not in explicit_subjects
            ):
                reason = "EXPLICIT_SUBJECT_MISMATCH"
            elif not set(memory.get("evidence_ids") or []) & evidence_ids:
                reason = "NO_STORED_LINKED_EVIDENCE"
            elif memory.get("assertion_mode", "DIRECT") != "DIRECT":
                reason = "NON_DIRECT_ASSERTION"
            elif status == "superseded" and not historical:
                reason = "SUPERSEDED"
            else:
                identity, _ = self._ec_identity(question, memory, advisory)
                if not identity:
                    reason = "QUESTION_PREDICATE_NOT_ESTABLISHED"
                elif memory.get("stance", "AFFIRM") != "AFFIRM":
                    contradicted = True
                    reason = "NON_AFFIRMATIVE_ASSERTION"
            if reason:
                rejected[mid] = reason
            else:
                eligible.append(memory)
        selected, selector_status = self._ec_selector(eligible, selector)
        if advisory.get("selector_status", "INVALID") == "INVALID":
            selected, selector_status = [], "INVALID_SELECTOR"
        groups = {}
        for memory in selected:
            for value in self._ec_projections(memory, projection, selector):
                key = self._ec_surface(value)
                if key and key != "unknown":
                    group = groups.setdefault(key, {"value": value, "support_ids": []})
                    if memory["id"] not in group["support_ids"]:
                        group["support_ids"].append(memory["id"])
        support_ids = list(
            dict.fromkeys(mid for g in groups.values() for mid in g["support_ids"])
        )
        conflicting = contradicted or any(
            self._belief_status.get(m["id"], m.get("_status")) == "conflicting"
            for m in selected
        )
        # State-head metadata is inspected, never injected into the candidate world.
        by_id = {m["id"]: m for m in self._memories}
        for memory in selected:
            identity = state_identity(memory)
            if identity and not historical:
                values = {
                    self._normalised_value(by_id[mid])
                    for mid in self._state_heads.get(identity, [])
                    if mid in by_id
                }
                conflicting = conflicting or len(values) > 1
        status = (
            "SUPPORTED_COMPETING"
            if len(groups) > 1 or conflicting
            else "SUPPORTED_UNIQUE" if len(groups) == 1 else "INSUFFICIENT"
        )
        hypothesis = advisory.get("answer_hypothesis")
        hypothesis_key = self._ec_surface(hypothesis)
        hypothesis_status = (
            status
            if hypothesis_key in groups
            else "UNSUPPORTED" if hypothesis else "NOT_PROPOSED"
        )
        synthesis = len(explicit_subjects) > 1 or projection not in {
            "ENTITY",
            "VALUE",
            "DATE",
        }
        terminal = status == "SUPPORTED_UNIQUE" and not synthesis
        return {
            "status": status,
            "hypothesis_status": hypothesis_status,
            "supported_surfaces": list(groups.values()),
            "support_ids": support_ids,
            "competitor_surfaces": (
                [g["value"] for g in groups.values()] if len(groups) > 1 else []
            ),
            "selector_resolution": selector_status,
            "rejected": rejected,
            "terminal": {
                "eligible": not synthesis,
                "closed": terminal,
                "answer_source": "memory_projection" if terminal else "",
                "reason": (
                    "UNIQUE_STORED_SUPPORT"
                    if terminal
                    else "SYNTHESIS_REQUIRED" if synthesis else status
                ),
            },
            "answer": next(iter(groups.values()))["value"] if terminal else None,
        }
