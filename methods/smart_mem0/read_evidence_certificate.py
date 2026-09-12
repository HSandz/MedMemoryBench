"""One read-only support checkpoint over a frozen candidate world.

The certificate can establish that stored participant evidence supports a projected
answer under the original question's constraints. It is not a real-world truth
oracle and cannot acquire, drop, rerank, or mutate memories.
"""

import math
import re
import unicodedata

from .canonicalization import state_identity
from .contracts import VALID_TEMPORAL_AXES


class EvidenceCertificateMixin:
    CERTIFICATE_VERSION = "question-owned-support-v3"
    PREDICATE_RARE_DF_RATIO = 0.10
    PREDICATE_MEDIUM_DF_RATIO = 0.25

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

    @classmethod
    def _ec_terms(cls, value):
        return list(
            dict.fromkeys(
                token
                for token in re.findall(r"\w+", cls._ec_text(value), flags=re.UNICODE)
                if len(token) > 1
            )
        )

    def _ec_assertion_text(self, memory):
        # Only stored assertion wording participates in lexical predicate binding.
        # Scope/entity/address metadata can help retrieval but cannot prove predicate
        # semantics such as prescribed vs contraindicated.
        return " ".join(
            str(value)
            for value in (memory.get("claim"), memory.get("verbatim_value"))
            if value
        )

    def _ec_direct_candidate_terms(self, memory, advisory):
        projection = advisory.get("projection_hint")
        selector = advisory.get("selector_hint") or {}
        terms = set()
        if projection in {"ENTITY", "VALUE", "DATE"}:
            for value in self._ec_projections(memory, projection, selector):
                terms.update(self._ec_terms(value))
        return terms

    def _ec_identity(self, question, memory, advisory):
        """Deterministic question-predicate binding with no LLM proof authority.

        Advisor focus spans, hints and hypotheses are deliberately ignored. Exact
        Unicode terms from the original question are matched only against the
        stored assertion/predicate. Candidate-answer tokens are removed first, so
        mentioning an entity/value alone cannot certify a different predicate.
        Unknown paraphrases remain synthesis work rather than fuzzy proof.
        """
        question_terms = self._ec_terms(question)
        if not question_terms:
            return False, {"source": "NO_QUESTION_TERMS"}

        assertion_terms = set(self._ec_terms(self._ec_assertion_text(memory)))
        candidate_terms = self._ec_direct_candidate_terms(memory, advisory)
        shared = [
            term
            for term in question_terms
            if term in assertion_terms and term not in candidate_terms
        ]

        corpus_terms = [
            set(self._ec_terms(self._ec_assertion_text(item))) for item in self._memories
        ]
        total = max(1, len(corpus_terms))
        df = {term: sum(term in terms for terms in corpus_terms) for term in question_terms}
        rare_limit = max(1, math.ceil(total * self.PREDICATE_RARE_DF_RATIO))
        medium_limit = max(2, math.ceil(total * self.PREDICATE_MEDIUM_DF_RATIO))
        rare = [
            term
            for term in shared
            if df.get(term, total) <= rare_limit
            and (len(term) >= 4 or any(ch.isdigit() for ch in term) or not term.isascii())
        ]
        medium = [
            term
            for term in shared
            if df.get(term, total) <= medium_limit and len(term) >= 3
        ]

        predicate = self._ec_text(memory.get("state_key") or memory.get("predicate"))
        question_text = self._ec_text(question)
        predicate_exact = bool(
            predicate
            and re.search(r"(?<!\w)" + re.escape(predicate) + r"(?!\w)", question_text)
        )
        established = bool(predicate_exact or rare or len(set(medium)) >= 2)
        matched = list(dict.fromkeys([*rare, *medium]))
        return established, {
            "source": (
                "EXACT_STORED_PREDICATE"
                if predicate_exact
                else "RARE_QUESTION_PREDICATE_TERM"
                if rare
                else "MULTI_QUESTION_PREDICATE_TERMS"
                if established
                else "UNRESOLVED_QUESTION_PREDICATE"
            ),
            "matched_predicate_terms": matched[:12],
            "candidate_terms_removed": sorted(candidate_terms)[:12],
            "rare_df_limit": rare_limit,
            "medium_df_limit": medium_limit,
            "advisor_fields_are_proof": False,
        }

    def _ec_projections(self, memory, projection, selector):
        if projection == "DATE":
            axis = selector.get("axis")
            return [self._date_for(memory, axis)] if axis in VALID_TEMPORAL_AXES else []
        if projection == "VALUE":
            value = str(memory.get("verbatim_value") or memory.get("value") or "").strip()
            return [value] if value else []
        if projection == "ENTITY":
            # Prefer a durable answer object. A dose/status value is not a second entity.
            anchor = memory.get("object_anchor")
            values = [anchor] if anchor else [memory.get("value")]
            return list(dict.fromkeys(str(v).replace("_", " ") for v in values if v))
        return []

    def _ec_selector(self, memories, selector):
        relation = selector.get("relation", "")
        if not relation:
            return memories, "NOT_REQUESTED"
        if relation == "CURRENT":
            heads = [m for m in memories if state_identity(m) and self._is_state_head(m)]
            return heads, "RESOLVED" if heads else "NO_DURABLE_STATE_HEAD"
        axis = selector.get("axis")
        if axis not in VALID_TEMPORAL_AXES:
            return [], "MISSING_AXIS"
        dated = [(m, self._date_for(m, axis)) for m in memories]
        if any(not date for _, date in dated) and relation in {"EARLIEST", "LATEST"}:
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

    def _ec_explicit_subjects(self, question):
        # Prefer durable subject_id. Only legacy stores with no subject_id at all
        # fall back to conversational subject labels, avoiding "Doctor, ..." as an
        # accidental owner constraint in modern snapshots.
        durable = {
            str(m.get("subject_id") or "").strip()
            for m in self._memories
            if str(m.get("subject_id") or "").strip()
        }
        fallback = not durable
        if fallback:
            durable = {
                str(m.get("subject") or "").strip()
                for m in self._memories
                if str(m.get("subject") or "").strip()
            }
        explicit = {
            subject
            for subject in durable
            if subject
            and re.search(
                r"(?<!\w)" + re.escape(subject.replace("_", " ")) + r"(?!\w)",
                question,
                re.I,
            )
        }
        for alias, subject in (getattr(self, "subject_aliases", {}) or {}).items():
            if subject in durable and re.search(
                r"(?<!\w)" + re.escape(str(alias).replace("_", " ")) + r"(?!\w)",
                question,
                re.I,
            ):
                explicit.add(subject)
        return explicit, fallback

    @classmethod
    def _ec_surface_matches(cls, left, right):
        a, b = cls._ec_surface(left), cls._ec_surface(right)
        if not a or not b:
            return False
        if a == b:
            return True
        return bool(
            re.search(r"(?<!\w)" + re.escape(a) + r"(?!\w)", b)
            or re.search(r"(?<!\w)" + re.escape(b) + r"(?!\w)", a)
        )

    def _evidence_certificate(self, question, world, advisory):
        projection = advisory["projection_hint"]
        selector = advisory["selector_hint"]
        selector_contract = advisory.get("selector_status", "INVALID")
        evidence_ids = {e["id"] for e in self._evidence}
        explicit_subjects, subject_fallback = self._ec_explicit_subjects(question)

        eligible, rejected, contradicted, binding = [], {}, False, {}
        historical = bool(
            selector.get("relation") and selector.get("relation") != "CURRENT"
        )
        for memory in world:
            mid = memory["id"]
            status = self._belief_status.get(mid, memory.get("_status", "active"))
            owner = str(
                memory.get("subject_id")
                or (memory.get("subject") if subject_fallback else "")
                or ""
            )
            reason = ""
            if explicit_subjects and owner and owner not in explicit_subjects:
                reason = "EXPLICIT_SUBJECT_MISMATCH"
            elif not set(memory.get("evidence_ids") or []) & evidence_ids:
                reason = "NO_STORED_LINKED_EVIDENCE"
            elif memory.get("assertion_mode", "DIRECT") != "DIRECT":
                reason = "NON_DIRECT_ASSERTION"
            elif status == "superseded" and not historical:
                reason = "SUPERSEDED"
            else:
                identity, detail = self._ec_identity(question, memory, advisory)
                binding[mid] = detail
                if not identity:
                    reason = "QUESTION_PREDICATE_NOT_ESTABLISHED"
                elif memory.get("stance", "AFFIRM") != "AFFIRM":
                    contradicted = True
                    reason = "NON_AFFIRMATIVE_ASSERTION"
            if reason:
                rejected[mid] = reason
            else:
                eligible.append(memory)

        selected, selector_resolution = self._ec_selector(eligible, selector)
        if selector_contract == "INVALID":
            selected, selector_resolution = [], "INVALID_SELECTOR"

        groups = {}
        for memory in selected:
            for value in self._ec_projections(memory, projection, selector):
                key = self._ec_surface(value)
                if key and key != "unknown":
                    group = groups.setdefault(key, {"value": value, "support_ids": []})
                    if memory["id"] not in group["support_ids"]:
                        group["support_ids"].append(memory["id"])
        support_ids = list(
            dict.fromkeys(mid for group in groups.values() for mid in group["support_ids"])
        )
        conflicting = contradicted or any(
            self._belief_status.get(m["id"], m.get("_status")) == "conflicting"
            for m in selected
        )

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
        hypothesis_supported = bool(
            hypothesis
            and any(
                self._ec_surface_matches(hypothesis, group["value"])
                for group in groups.values()
            )
        )
        hypothesis_status = (
            status
            if hypothesis_supported
            else "UNSUPPORTED"
            if hypothesis
            else "NOT_PROPOSED"
        )

        atomic = projection in {"ENTITY", "VALUE", "DATE"}
        # Exact focus spans can only veto an optimization, never authorize proof.
        # Multiple spans conservatively indicate possible composition even when the
        # advisor mislabeled the projection as atomic.
        composition_veto = atomic and len(advisory.get("focus_spans") or []) > 1
        synthesis = not atomic or composition_veto
        terminal = (
            status == "SUPPORTED_UNIQUE"
            and not synthesis
            and selector_contract != "INVALID"
        )
        return {
            "version": self.CERTIFICATE_VERSION,
            "status": status,
            "hypothesis_status": hypothesis_status,
            "supported_surfaces": list(groups.values()),
            "support_ids": support_ids,
            "competitor_surfaces": (
                [group["value"] for group in groups.values()] if len(groups) > 1 else []
            ),
            "selector_status": selector_contract,
            "selector_resolution": selector_resolution,
            "binding": binding,
            "rejected": rejected,
            "terminal": {
                "eligible": atomic and not composition_veto and selector_contract != "INVALID",
                "closed": terminal,
                "answer_source": "memory_projection" if terminal else "",
                "reason": (
                    "UNIQUE_STORED_SUPPORT"
                    if terminal
                    else "SYNTHESIS_REQUIRED"
                    if synthesis
                    else "INVALID_SELECTOR"
                    if selector_contract == "INVALID"
                    else status
                ),
            },
            "answer": next(iter(groups.values()))["value"] if terminal else None,
        }
