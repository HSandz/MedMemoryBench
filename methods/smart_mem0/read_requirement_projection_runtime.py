"""Projection-aware proof and evidence preservation for RequirementGraph READ.

This layer is deliberately domain-, language-, dataset-, and benchmark-agnostic. It
separates three concepts that retrieval must not collapse:

1. materiality: a memory is worth considering for a requirement;
2. answerability: the memory carries a durable field compatible with the requested output;
3. proof: question-owned semantic identity + answerability + selector compatibility.

Retrieval aliases, resolver keys, dense/BM25 scores and material-binding strength may rank
or admit candidates, but they never authorize FOUND/STOP. When proof is unavailable, the
runtime conservatively preserves distinct answer-bearing alternatives instead of allowing a
single high-ranked generic candidate to evict unique relevant evidence.
"""

from copy import deepcopy

from .contracts import VALID_TEMPORAL_AXES


class ReadRequirementProjectionRuntimeMixin:
    REQUIREMENT_GRAPH_RUNTIME_VERSION = "requirement-graph-runtime-v3"
    PROJECTION_PROOF_VERSION = "projection-aware-proof-v1"
    UNRESOLVED_ANSWERABILITY_RESERVATION = 2

    def _requirement_slot(self, requirement, ir, compiled_mode):
        slot = super()._requirement_slot(requirement, ir, compiled_mode)
        slot["requested_projection"] = str(
            ir.get("answer_type")
            or (ir.get("goal") or {}).get("projection")
            or "TEXT"
        ).upper()

        # These are the only semantic surfaces allowed to participate in automatic proof.
        # They come from QUESTION semantics, never from retrieval-only expansion.
        slot["proof_need"] = str(
            slot.get("need")
            or requirement.get("need")
            or requirement.get("evidence_family")
            or requirement.get("answer_obligation")
            or ""
        ).strip()
        slot["proof_question_span"] = str(
            slot.get("question_span")
            or requirement.get("question_span")
            or slot.get("focus_span")
            or requirement.get("focus_span")
            or ""
        ).strip()
        return slot

    def _rg_requested_projection(self, slot):
        projection = str(slot.get("requested_projection") or "").upper()
        if projection:
            return projection
        graph = getattr(self, "_active_requirement_graph", {}) or {}
        return str((graph.get("goal") or {}).get("projection") or "TEXT").upper()

    def _rg_projection_evidence(self, slot, memory):
        """Return durable fields capable of supplying the requested projection.

        This is structural answerability, not truth. Retrieval aliases, material anchors,
        resolved keys and ranking scores are intentionally absent.
        """
        projection = self._rg_requested_projection(slot)
        fields, values = [], []

        def add(field, value):
            text = " ".join(str(value or "").replace("_", " ").split())
            if text and text.casefold() not in {item.casefold() for item in values}:
                fields.append(field)
                values.append(text)

        selector = slot.get("selector") if isinstance(slot.get("selector"), dict) else {}
        axis = str(selector.get("axis") or slot.get("time_axis") or "").lower()
        if projection in {"DATE", "RELATIVE_TIME"}:
            axes = [axis] if axis in VALID_TEMPORAL_AXES else sorted(VALID_TEMPORAL_AXES)
            for candidate_axis in axes:
                add(candidate_axis, self._date_for(memory, candidate_axis))
        elif projection == "VALUE":
            add("value", memory.get("value"))
            add("verbatim_value", memory.get("verbatim_value"))
        elif projection == "ENTITY":
            add("object_anchor", memory.get("object_anchor"))
            for entity in memory.get("entities") or []:
                add("entities", entity)
            for entity in memory.get("scope_entities") or []:
                add("scope_entities", entity)
            # WRITE sometimes stores the concrete entity in value. This is an answer-bearing
            # fallback only; value presence never establishes semantic identity by itself.
            add("value", memory.get("value"))
        elif projection == "OPTION_SET":
            # A single memory can support options but cannot certify a complete option set.
            pass
        else:
            add("value", memory.get("value"))
            add("verbatim_value", memory.get("verbatim_value"))
            add("claim", memory.get("claim"))

        return {
            "projection": projection,
            "capable": bool(values),
            "fields": fields,
            "values": values,
        }

    def _rg_exact_certificate(self, slot, memory):
        """Use an explicit ledger certificate when one exists; never synthesize one."""
        proof_spec = slot.get("proof_spec") if isinstance(slot.get("proof_spec"), dict) else {}
        if not proof_spec or str(proof_spec.get("status") or "").upper() == "UNSPECIFIED":
            return False
        certificate = getattr(self, "_certificate_result", None)
        if not callable(certificate):
            return False
        try:
            result = certificate(slot, memory)
            return bool(result[0] if isinstance(result, (tuple, list)) else result)
        except Exception:
            return False

    def _rg_question_owned_identity(self, slot, memory):
        """Conservative identity using only QUESTION-owned/semantic obligation surfaces."""
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

        # Canonical overlap is allowed only between the immutable question/need and surfaces
        # physically written on the memory. Resolver-generated keys and aliases are excluded.
        surfaces_fn = getattr(self, "_rq_memory_concept_surfaces", None)
        surfaces = list(surfaces_fn(memory) or []) if callable(surfaces_fn) else []
        rc_text = getattr(
            self,
            "_rc_text",
            lambda value: " ".join(str(value or "").lower().split()),
        )
        canonical_overlap = False
        for owned in (question_span, need):
            owned_key = rc_text(owned)
            if not owned_key:
                continue
            for surface in surfaces:
                surface_key = rc_text(surface)
                if not surface_key:
                    continue
                if surface_key == owned_key or (
                    len(surface_key) >= 4 and surface_key in owned_key
                ):
                    canonical_overlap = True
                    break
            if canonical_overlap:
                break

        projection = self._rg_requested_projection(slot)
        if projection == "ENTITY":
            # Runtime has no ontology proving that a retrieved class/category is the concrete
            # entity instance requested by the question. Therefore a short generic match such
            # as "antibiotic" -> "cephalosporin" is never enough for FOUND/STOP.
            terms = getattr(self, "_rc_terms", lambda value: [])(need)
            identity = bool(need and len(terms) >= 3 and need_score >= 0.88)
            source = (
                "full_obligation_match"
                if identity
                else "entity_requires_specific_grounding"
            )
        else:
            identity = bool(canonical_overlap or need_score >= 0.78 or span_score >= 0.82)
            source = (
                "question_canonical_overlap"
                if canonical_overlap
                else "full_obligation_match"
                if need_score >= 0.78
                else "question_span_match"
                if span_score >= 0.82
                else "no_question_owned_identity"
            )
        return identity, {
            "source": source,
            "need_similarity": round(need_score, 6),
            "question_span_similarity": round(span_score, 6),
            "canonical_question_overlap": bool(canonical_overlap),
        }

    def _rg_record_proof_check(self, slot, memory, payload):
        state = getattr(self, "_last_requirement_graph_proof_checks", None)
        if not isinstance(state, dict):
            state = {}
            self._last_requirement_graph_proof_checks = state
        rid = str(slot.get("id") or "")
        mid = str(memory.get("id") or "")
        if rid and mid:
            state.setdefault(rid, {})[mid] = deepcopy(payload)

    def _requirement_target_proof(self, slot, memory):
        """Proof = identity + projection capability + selector compatibility."""
        semantic_slot = getattr(self, "_semantic_context_slot", None)
        if callable(semantic_slot) and not semantic_slot(slot):
            return True
        if slot.get("degraded") or not memory or not memory.get("id"):
            return False

        projection = self._rg_projection_evidence(slot, memory)
        selector_ok = bool(self._rg_selector_compatible(slot, memory))
        exact_certificate = self._rg_exact_certificate(slot, memory)
        identity, identity_meta = self._rg_question_owned_identity(slot, memory)
        proven = bool(
            projection.get("capable")
            and selector_ok
            and (exact_certificate or identity)
        )
        self._rg_record_proof_check(
            slot,
            memory,
            {
                "version": self.PROJECTION_PROOF_VERSION,
                "proven": proven,
                "exact_certificate": bool(exact_certificate),
                "selector_compatible": selector_ok,
                "projection": projection,
                "identity": identity_meta,
                "forbidden_authorities": [
                    "material_anchors",
                    "resolved_keys",
                    "search_aliases",
                    "retrieval_scores",
                    "material_binding",
                ],
            },
        )
        return proven

    def _rg_answer_signature(self, slot, memory):
        evidence = self._rg_projection_evidence(slot, memory)
        normalizer = getattr(
            self,
            "_rc_text",
            lambda value: " ".join(str(value or "").lower().split()),
        )
        return tuple(
            sorted(
                {
                    normalizer(value)
                    for value in evidence.get("values") or []
                    if normalizer(value)
                }
            )
        )

    def _rg_answerability_key(self, slot, memory, original_index):
        evidence = self._rg_projection_evidence(slot, memory)
        need = str(slot.get("proof_need") or slot.get("need") or "").strip()
        memory_text_fn = getattr(self, "_rc_memory_target_text", None)
        memory_text = (
            memory_text_fn(memory)
            if callable(memory_text_fn)
            else str(memory.get("claim") or "")
        )
        similarity = getattr(self, "_rq_surface_similarity", None)
        semantic_score = float(similarity(need, memory_text)) if callable(similarity) and need else 0.0
        proof = self._requirement_target_proof(slot, memory)
        return (
            -int(proof),
            -int(bool(evidence.get("capable"))),
            -semantic_score,
            -len(evidence.get("fields") or []),
            original_index,
            str(memory.get("id") or ""),
        )

    def _run_query_retrieval(
        self,
        question,
        initial_seeds,
        frame,
        fast_supports,
        gate,
        planning_seeds=None,
        planning_context=None,
    ):
        self._last_requirement_graph_proof_checks = {}
        return super()._run_query_retrieval(
            question,
            initial_seeds,
            frame,
            fast_supports,
            gate,
            planning_seeds=planning_seeds,
            planning_context=planning_context,
        )

    def _rg_reserve_answerable_candidates(self, slots, candidate_order, limit):
        """Reserve unique answer-bearing alternatives for unresolved requirements."""
        bounded_limit = max(0, int(limit))
        if not bounded_limit:
            return [], {}
        semantic_slot_checker = getattr(self, "_semantic_context_slot", None)
        unique_slots, seen_slots = [], set()
        for slot in slots or []:
            rid = str(slot.get("id") or "")
            if not rid or rid in seen_slots:
                continue
            if callable(semantic_slot_checker) and not semantic_slot_checker(slot):
                continue
            unique_slots.append(slot)
            seen_slots.add(rid)

        allowed = []
        for memory_id in candidate_order or []:
            memory_id = str(memory_id or "")
            if memory_id and memory_id not in allowed:
                allowed.append(memory_id)
        allowed_set = set(allowed)
        original_index = {memory_id: index for index, memory_id in enumerate(allowed)}
        contexts = getattr(self, "_last_requirement_context_candidates", {}) or {}
        proofs = getattr(self, "_last_requirement_proof_support", {}) or {}
        reserved, per_requirement, ranked = [], {}, {}

        def add(memory_id):
            if (
                memory_id
                and memory_id in allowed_set
                and memory_id not in reserved
                and len(reserved) < bounded_limit
            ):
                reserved.append(memory_id)
                return True
            return False

        # Coverage first: every semantic requirement gets one answer-bearing candidate.
        for slot in unique_slots:
            rid = str(slot.get("id") or "")
            pool = [
                memory_id
                for memory_id in contexts.get(rid, [])
                if memory_id in allowed_set and self._alignment_memory(memory_id)
            ]
            proof_ids = [memory_id for memory_id in proofs.get(rid, []) if memory_id in pool]
            if proof_ids:
                pool = proof_ids + [memory_id for memory_id in pool if memory_id not in proof_ids]
            pool.sort(
                key=lambda memory_id: self._rg_answerability_key(
                    slot,
                    self._alignment_memory(memory_id),
                    original_index.get(memory_id, 10**9),
                )
            )
            ranked[rid] = (slot, pool, proof_ids)
            first = next(
                (
                    memory_id
                    for memory_id in pool
                    if self._rg_projection_evidence(
                        slot, self._alignment_memory(memory_id)
                    ).get("capable")
                ),
                "",
            )
            if first and add(first):
                per_requirement.setdefault(rid, []).append(first)

        # Unresolved lanes retain one second, distinct answer surface. This is deliberately
        # bounded and is not a vote: it only prevents premature information loss.
        for rid, (slot, pool, proof_ids) in ranked.items():
            if len(reserved) >= bounded_limit:
                break
            if proof_ids:
                continue
            existing = per_requirement.get(rid, [])
            signatures = {
                self._rg_answer_signature(slot, self._alignment_memory(memory_id))
                for memory_id in existing
            }
            for memory_id in pool:
                if memory_id in existing:
                    continue
                memory = self._alignment_memory(memory_id)
                evidence = self._rg_projection_evidence(slot, memory)
                if not evidence.get("capable"):
                    continue
                signature = self._rg_answer_signature(slot, memory)
                if signature and signature in signatures:
                    continue
                if add(memory_id):
                    per_requirement.setdefault(rid, []).append(memory_id)
                break

        return reserved, per_requirement

    def _role_aware_support_ids(self, slots, slot_support, candidate_order, limit):
        baseline = list(
            super()._role_aware_support_ids(
                slots, slot_support, candidate_order, limit
            )
        )
        bounded_limit = max(0, int(limit))
        if not baseline or not bounded_limit:
            return baseline

        # If strict proof already closed every semantic requirement, the lower runtime has
        # already made the correct proof-complete compaction/structural-context decision.
        # Never re-expand a proven context here.
        semantic_slot_checker = getattr(self, "_semantic_context_slot", None)
        semantic_slots = [
            slot
            for slot in slots or []
            if not callable(semantic_slot_checker) or semantic_slot_checker(slot)
        ]
        statuses = (
            getattr(self, "_rg_runtime_requirement_status", {})
            or getattr(self, "_last_requirement_status", {})
            or {}
        )
        relation_status = getattr(self, "_rg_runtime_relation_status", {}) or {}
        if semantic_slots and all(
            statuses.get(str(slot.get("id") or "")) == "FOUND"
            for slot in semantic_slots
        ) and all(
            str(status or "").upper() == "PROVEN"
            for status in relation_status.values()
        ):
            return baseline

        reservations, reservation_map = self._rg_reserve_answerable_candidates(
            slots, candidate_order, bounded_limit
        )
        selected = []
        for memory_id in [*reservations, *baseline]:
            if memory_id and memory_id not in selected:
                selected.append(memory_id)
            if len(selected) >= bounded_limit:
                break

        telemetry = dict(getattr(self, "_last_query_memory_alignment", {}) or {})
        telemetry.update(
            {
                "answerability_reservations": deepcopy(reservation_map),
                "selected_ids": list(selected),
                "selection_semantics": "answerability_reservation_then_material_redundancy_fill",
                "proof_semantics": "projection_aware_proof_only_controls_stop",
            }
        )
        self._last_query_memory_alignment = telemetry
        return selected

    def prepare_batch_query(self, question, system_message=None, **kwargs):
        prepared = super().prepare_batch_query(
            question, system_message=system_message, **kwargs
        )
        extra = prepared.setdefault("extra", {})
        extra["requirement_graph_runtime_version"] = self.REQUIREMENT_GRAPH_RUNTIME_VERSION
        extra["requirement_graph_proof_contract"] = {
            "version": self.PROJECTION_PROOF_VERSION,
            "semantics": "question_owned_identity_plus_projection_plus_selector",
            "retrieval_hints_are_proof": False,
            "checks": deepcopy(
                getattr(self, "_last_requirement_graph_proof_checks", {}) or {}
            ),
        }
        stop_guard = dict(extra.get("requirement_graph_stop_guard") or {})
        stop_guard["proof_version"] = self.PROJECTION_PROOF_VERSION
        stop_guard["answerability_is_not_proof"] = True
        extra["requirement_graph_stop_guard"] = stop_guard
        return prepared
