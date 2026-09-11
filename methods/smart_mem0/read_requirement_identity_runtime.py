"""Discriminative proof identity and guarded CURRENT-state selection.

This layer hardens two semantic authorities that must remain conservative:

1. canonical concept overlap may help retrieval, but a generic shared surface must not by
   itself certify requirement identity;
2. CURRENT may select an active durable state only when the memory store exposes a matching
   state address. Otherwise retrieval falls back to ordinary requirement search.

The implementation is deterministic and domain-, dataset-, benchmark-, and language-agnostic.
It never promotes retrieval aliases, resolver keys, material scores, BM25, or dense similarity
into proof.
"""

from copy import deepcopy


class ReadRequirementIdentityRuntimeMixin:
    REQUIREMENT_GRAPH_RUNTIME_VERSION = "requirement-graph-runtime-v5"
    PROJECTION_PROOF_VERSION = "projection-aware-proof-v2-discriminative"
    REQUIREMENT_IDENTITY_VERSION = "discriminative-question-identity-v1"
    PROOF_CONCEPT_COVERAGE_MIN = 0.72
    CURRENT_STATE_FORWARD_MIN = 0.90
    CURRENT_STATE_REVERSE_MIN = 0.50

    def _rg_current_state_address(self, slot):
        """Return a durable state-address match for CURRENT, if one actually exists.

        Anchors remain retrieval-only.  This check authorizes only the retrieval operator
        (state resolution versus family search); it never authorizes proof or FOUND/STOP.
        """
        normalize = getattr(
            self,
            "_rc_text",
            lambda value: " ".join(str(value or "").replace("_", " ").lower().split()),
        )
        similarity = getattr(self, "_rq_surface_similarity", None)
        anchors = []
        for value in [
            *(slot.get("material_anchors") or []),
            *(slot.get("resolved_keys") or []),
        ]:
            text = " ".join(str(value or "").replace("_", " ").split()).strip()
            key = normalize(text)
            if text and key and key not in {normalize(item) for item in anchors}:
                anchors.append(text)

        if not anchors:
            return False, {"matched_anchor": "", "matched_surface": "", "memory_id": ""}

        best = None
        for memory in getattr(self, "_memories", []) or []:
            state_surfaces = []
            state_key = str(memory.get("state_key") or "").strip()
            object_anchor = str(memory.get("object_anchor") or "").strip()
            if state_key:
                state_surfaces.append(state_key)
            if str(memory.get("kind") or "").upper() == "STATE" and object_anchor:
                state_surfaces.append(object_anchor)
            if not state_surfaces:
                continue

            for anchor in anchors:
                for surface in state_surfaces:
                    anchor_key = normalize(anchor)
                    surface_key = normalize(surface)
                    if not anchor_key or not surface_key:
                        continue
                    exact = anchor_key == surface_key
                    forward = (
                        float(similarity(anchor, surface))
                        if callable(similarity)
                        else float(exact)
                    )
                    reverse = (
                        float(similarity(surface, anchor))
                        if callable(similarity)
                        else float(exact)
                    )
                    supported = bool(
                        exact
                        or (
                            forward >= self.CURRENT_STATE_FORWARD_MIN
                            and reverse >= self.CURRENT_STATE_REVERSE_MIN
                        )
                    )
                    if not supported:
                        continue
                    rank = (int(exact), min(forward, reverse), forward, reverse)
                    if best is None or rank > best[0]:
                        best = (
                            rank,
                            {
                                "matched_anchor": anchor,
                                "matched_surface": surface,
                                "memory_id": str(memory.get("id") or ""),
                                "exact": bool(exact),
                                "forward_similarity": round(forward, 6),
                                "reverse_similarity": round(reverse, 6),
                            },
                        )
        if best is None:
            return False, {"matched_anchor": "", "matched_surface": "", "memory_id": ""}
        return True, best[1]

    def _requirement_slot(self, requirement, ir, compiled_mode):
        slot = super()._requirement_slot(requirement, ir, compiled_mode)
        selector = slot.get("selector") if isinstance(slot.get("selector"), dict) else {}
        if str(selector.get("relation") or "").upper() != "CURRENT":
            return slot

        supported, detail = self._rg_current_state_address(slot)
        slot["current_selector_guard"] = {
            "status": "PASS" if supported else "DOWNGRADED_NO_DURABLE_STATE_ADDRESS",
            **detail,
            "authority": "retrieval_operator_only_not_proof",
        }
        if supported:
            return slot

        # CURRENT without a durable state address is not allowed to force RESOLVE_STATE.
        # Preserve the semantic requirement and retrieve it as an ordinary evidence family.
        slot["selector"] = {}
        if str(slot.get("type") or "").upper() == "CURRENT_STATE":
            slot["type"] = "DIRECT"
        relation_types = [
            item
            for item in (slot.get("semantic_relation_types") or [])
            if str(item or "").upper() != "CURRENT"
        ]
        slot["semantic_relation_types"] = relation_types
        return slot

    def _rg_question_owned_identity(self, slot, memory):
        """Proof identity requires discriminative question-owned coverage.

        A memory surface such as ``patient``, ``symptom`` or ``medication`` may be useful for
        retrieval but is not sufficient proof merely because it occurs inside the question.
        Canonical surfaces participate in proof only when they cover a substantial fraction
        of the question-owned concept.  High full-obligation/span similarity and exact ledger
        certificates remain independent proof paths in the lower projection runtime.
        """
        question_span = str(
            slot.get("proof_question_span")
            or slot.get("question_span")
            or slot.get("focus_span")
            or ""
        ).strip()
        need = str(
            slot.get("proof_need")
            or slot.get("need")
            or slot.get("answer_obligation")
            or ""
        ).strip()
        memory_text_fn = getattr(self, "_rc_memory_target_text", None)
        memory_text = (
            memory_text_fn(memory)
            if callable(memory_text_fn)
            else " ".join(
                str(memory.get(field) or "")
                for field in ("claim", "value", "verbatim_value", "object_anchor")
            )
        )
        similarity = getattr(self, "_rq_surface_similarity", None)
        need_score = float(similarity(need, memory_text)) if callable(similarity) and need else 0.0
        span_score = (
            float(similarity(question_span, memory_text))
            if callable(similarity) and question_span
            else 0.0
        )

        topic_surfaces_fn = getattr(self, "_rg_material_topic_surfaces", None)
        if callable(topic_surfaces_fn):
            surfaces = list(topic_surfaces_fn(memory) or [])
        else:
            surfaces_fn = getattr(self, "_rq_memory_concept_surfaces", None)
            surfaces = list(surfaces_fn(memory) or []) if callable(surfaces_fn) else []

        concept_score = 0.0
        concept_surface = ""
        concept_owned = ""
        if callable(similarity):
            for owned in (question_span, need):
                if not owned:
                    continue
                for surface in surfaces:
                    score = float(similarity(owned, surface))
                    if score > concept_score:
                        concept_score = score
                        concept_surface = str(surface)
                        concept_owned = owned

        projection = self._rg_requested_projection(slot)
        if projection == "ENTITY":
            # Concrete entity category membership still requires stronger grounding than a
            # lexical/topic match. Keep the conservative entity rule from v3.
            terms = getattr(self, "_rc_terms", lambda value: [])(need)
            identity = bool(need and len(terms) >= 3 and need_score >= 0.88)
            source = (
                "full_obligation_match"
                if identity
                else "entity_requires_specific_grounding"
            )
        else:
            concept_identity = bool(concept_score >= self.PROOF_CONCEPT_COVERAGE_MIN)
            identity = bool(
                need_score >= 0.78
                or span_score >= 0.82
                or concept_identity
            )
            source = (
                "full_obligation_match"
                if need_score >= 0.78
                else "question_span_match"
                if span_score >= 0.82
                else "discriminative_concept_match"
                if concept_identity
                else "no_question_owned_identity"
            )

        return identity, {
            "source": source,
            "need_similarity": round(need_score, 6),
            "question_span_similarity": round(span_score, 6),
            "canonical_question_overlap": bool(
                concept_score >= self.PROOF_CONCEPT_COVERAGE_MIN
            ),
            "concept_coverage": round(concept_score, 6),
            "concept_surface": concept_surface,
            "concept_owned_surface": concept_owned,
            "concept_coverage_min": self.PROOF_CONCEPT_COVERAGE_MIN,
            "generic_substring_overlap_is_proof": False,
        }

    def prepare_batch_query(self, question, system_message=None, **kwargs):
        prepared = super().prepare_batch_query(
            question, system_message=system_message, **kwargs
        )
        extra = prepared.setdefault("extra", {})
        extra["requirement_graph_runtime_version"] = self.REQUIREMENT_GRAPH_RUNTIME_VERSION
        proof_contract = dict(extra.get("requirement_graph_proof_contract") or {})
        proof_contract.update(
            {
                "version": self.PROJECTION_PROOF_VERSION,
                "semantics": "discriminative_question_owned_identity_plus_projection_plus_selector",
                "generic_substring_overlap_is_proof": False,
            }
        )
        extra["requirement_graph_proof_contract"] = proof_contract
        stop_guard = dict(extra.get("requirement_graph_stop_guard") or {})
        stop_guard["proof_version"] = self.PROJECTION_PROOF_VERSION
        extra["requirement_graph_stop_guard"] = stop_guard
        extra["requirement_identity_contract"] = {
            "version": self.REQUIREMENT_IDENTITY_VERSION,
            "proof_concept_coverage_min": self.PROOF_CONCEPT_COVERAGE_MIN,
            "current_requires_durable_state_address": True,
            "current_state_forward_min": self.CURRENT_STATE_FORWARD_MIN,
            "current_state_reverse_min": self.CURRENT_STATE_REVERSE_MIN,
            "anchors_are_proof": False,
        }
        return prepared
