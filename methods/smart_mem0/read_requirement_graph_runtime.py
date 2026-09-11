"""Deterministic runtime guards for RequirementGraph READ.

This layer contains no question taxonomy and no domain policy. It converts semantic
RequirementGraph intent into stable runtime invariants: derive canonical memory addresses
when the LLM omits them, keep CURRENT bound to an actual state head, keep raw lexical
and dense recall independently reachable, prevent retrieval materiality from becoming
proof/STOP authority, and compact proof-complete contexts without filling to a quota.
"""

from copy import deepcopy

import numpy as np


class ReadRequirementGraphRuntimeMixin:
    REQUIREMENT_GRAPH_RUNTIME_VERSION = "requirement-graph-runtime-v2"
    RAW_CHANNEL_ADMISSION_PER_RAIL = 2

    def _requirement_slot(self, requirement, ir, compiled_mode):
        slot = super()._requirement_slot(requirement, ir, compiled_mode)

        # Candidate alternatives are propositions. RequirementGraph nodes remain shared
        # participant evidence even when visible alternatives exist. Candidate-local probes
        # may broaden recall, but they must never redefine the material requirement role.
        if ir.get("visible_options") or ir.get("candidate_propositions"):
            slot["evidence_role"] = "REQUIREMENT"

        # LLM anchors are semantic hints, not a mandatory perfect schema. Resolve the
        # requirement `need` against the same durable concept surfaces written by memory
        # construction and merge those deterministic addresses with supplied anchors.
        #
        # IMPORTANT: these addresses are retrieval/materiality hints only. _retrieval_status
        # below deliberately re-runs strict slot proof and never upgrades FOUND from them.
        need = str(
            slot.get("need")
            or requirement.get("need")
            or requirement.get("evidence_family")
            or requirement.get("answer_obligation")
            or ""
        ).strip()
        subject_id = str(slot.get("subject_id") or slot.get("subject") or "")
        resolver = getattr(self, "_rc_resolve_target_keys", None)
        derived = []
        if need and callable(resolver):
            try:
                derived = list(resolver(need, subject_id) or [])
            except Exception:
                derived = []
        unique = getattr(self, "_rg_unique_text", None)
        values = [
            *(slot.get("material_anchors") or []),
            *derived,
            *(slot.get("resolved_keys") or []),
        ]
        slot["material_anchors"] = (
            unique(values, limit=8)
            if callable(unique)
            else list(dict.fromkeys(values))[:8]
        )
        slot["canonical_anchor_source"] = {
            "llm": list(requirement.get("material_anchors") or [])[:6],
            "runtime_resolved": derived[:4],
            "authority": "retrieval_and_materiality_only_not_proof",
        }
        return slot

    def _rg_selector_compatible(self, slot, memory):
        if not super()._rg_selector_compatible(slot, memory):
            return False
        relation = str(self._rg_selector_relation(slot) or "").upper()
        if relation != "CURRENT":
            return True

        # CURRENT is stronger than merely "not superseded". If the store has a state
        # identity/head for this memory, only an active head is materially compatible.
        head_checker = getattr(self, "_is_state_head", None)
        if callable(head_checker):
            try:
                return bool(head_checker(memory))
            except Exception:
                return False
        return False

    def _rg_raw_question_channel_candidates(self, query, eligible_ids):
        """Return bounded BM25 and dense heads before any fused top-k truncation."""
        query = str(query or "").strip()
        allowed = {str(memory_id) for memory_id in (eligible_ids or []) if memory_id}
        memories = list(getattr(self, "_memories", []) or [])
        if not query or not allowed or not memories:
            return []

        self._refresh_index()
        indices = [
            index
            for index, memory in enumerate(memories)
            if str(memory.get("id") or "") in allowed
        ]
        if not indices:
            return []

        bm25 = getattr(self, "_bm25", None)
        bm25_scores = (
            bm25.get_scores(self._tokenize(query))
            if bm25 is not None
            else np.zeros(len(memories), dtype=float)
        )

        embedding_matrix = getattr(self, "_embedding_matrix", None)
        embedder = getattr(self, "_embedder", None)
        if embedding_matrix is None or embedder is None:
            dense_scores = np.zeros(len(memories), dtype=float)
        else:
            query_vector = np.asarray(
                embedder.encode([query], show_progress_bar=False)
            )[0]
            matrix = np.asarray(embedding_matrix)
            query_norm = float(np.linalg.norm(query_vector))
            row_norms = np.linalg.norm(matrix, axis=1)
            denominator = row_norms * query_norm
            dense_scores = np.divide(
                matrix @ query_vector,
                denominator,
                out=np.zeros(len(matrix), dtype=float),
                where=denominator > 0,
            )

        per_rail = max(1, int(self.RAW_CHANNEL_ADMISSION_PER_RAIL))
        lexical = sorted(
            (index for index in indices if float(bm25_scores[index]) > 0.0),
            key=lambda index: (
                float(bm25_scores[index]),
                str(memories[index].get("id") or ""),
            ),
            reverse=True,
        )[:per_rail]
        dense = sorted(
            indices,
            key=lambda index: (
                float(dense_scores[index]),
                str(memories[index].get("id") or ""),
            ),
            reverse=True,
        )[:per_rail]
        lexical_rank = {index: rank + 1 for rank, index in enumerate(lexical)}
        dense_rank = {index: rank + 1 for rank, index in enumerate(dense)}
        query_terms = set(self._tokenize(query))

        output = []
        seen = set()
        for index in [*lexical, *dense]:
            memory_id = str(memories[index].get("id") or "")
            if not memory_id or memory_id in seen:
                continue
            item = self._snapshot(memories[index])
            bm25_position = lexical_rank.get(index)
            dense_position = dense_rank.get(index)
            score = 0.0
            if bm25_position:
                score += 1.0 / (float(getattr(self, "RRF_K", 60.0)) + bm25_position)
            if dense_position:
                score += 1.0 / (float(getattr(self, "RRF_K", 60.0)) + dense_position)
            sources = list(item.get("_alignment_recall_sources") or [])
            if bm25_position and "raw_lexical" not in sources:
                sources.append("raw_lexical")
            if dense_position and "raw_dense" not in sources:
                sources.append("raw_dense")
            item.update(
                {
                    "_score": score,
                    "_bm25_score": float(bm25_scores[index]),
                    "_dense_score": float(dense_scores[index]),
                    "_bm25_rank": bm25_position,
                    "_dense_rank": dense_position,
                    "_overlap": len(
                        query_terms
                        & set(self._tokenize(self._memory_text(item)))
                    ),
                    "_status": getattr(self, "_belief_status", {}).get(
                        memory_id, "active"
                    ),
                    "_alignment_recall_sources": sources,
                }
            )
            output.append(item)
            seen.add(memory_id)
        return output

    def _fusion_multiview(self, operation, outputs, seeds, frame):
        """Union semantic retrieval with raw BM25/dense heads before final truncation."""
        rows, relations, evidence_refs = super()._fusion_multiview(
            operation, outputs, seeds, frame
        )
        views = self._fusion_views(operation.get("retrieval_views") or [])
        raw_view = views.get("question")
        if not raw_view:
            return rows, relations, evidence_refs

        include_history = (
            str(operation.get("strategy") or "FOCAL").upper() == "TRAJECTORY"
        )
        eligible_ids = {
            str(memory.get("id") or "")
            for memory in getattr(self, "_memories", []) or []
            if memory.get("id")
            and self._memory_satisfies_frame(
                memory,
                frame,
                include_entities=bool(getattr(frame, "hard_entities", ())),
            )
            and self._query_visible_memory(
                memory, include_history=include_history
            )
        }
        channel_rows = self._rg_raw_question_channel_candidates(
            raw_view.get("query") or "", eligible_ids
        )
        if not channel_rows:
            return rows, relations, evidence_refs

        ordered = []
        seen = set()

        def add(memory):
            if not memory or not memory.get("id") or memory["id"] in seen:
                return
            ordered.append(deepcopy(memory))
            seen.add(memory["id"])

        # Keep one semantic head, then grant each raw channel bounded admission rights,
        # then preserve the existing deterministic fusion order.
        semantic_head = next(
            (
                memory
                for memory in (rows or [])
                if "semantic_ir"
                in set(memory.get("_alignment_recall_sources") or [])
            ),
            (rows or [None])[0],
        )
        add(semantic_head)
        for memory in channel_rows:
            add(memory)
        for memory in rows or []:
            add(memory)

        limit = max(1, int(operation.get("top_k", len(ordered)) or len(ordered)))
        return ordered[:limit], relations or [], evidence_refs or []

    def _retrieval_status(self, plan, slot_support, selected, relations):
        """Strict completion: materiality/ranking hints can never authorize FOUND/STOP."""
        requirement_status = {}
        for slot in (plan or {}).get("required_slots") or []:
            slot_id = str(slot.get("id") or "")
            support_ids = list((slot_support or {}).get(slot_id) or [])
            requirement_status[slot_id] = (
                "FOUND"
                if support_ids
                and self._slot_covered(slot, support_ids, selected, relations)
                else "EMPTY"
            )

        relation_mapper = getattr(self, "_relation_status_map", None)
        relation_status = (
            relation_mapper(plan, slot_support, selected, relations)
            if callable(relation_mapper)
            else {}
        )
        retrieval_complete = bool(requirement_status) and all(
            status == "FOUND" for status in requirement_status.values()
        )
        retrieval_complete = retrieval_complete and all(
            status == "PROVEN" for status in relation_status.values()
        )
        self._last_requirement_graph_stop_guard = {
            "materiality_promotes_found": False,
            "strict_requirement_status": dict(requirement_status),
            "strict_relation_status": dict(relation_status),
            "retrieval_complete": bool(retrieval_complete),
        }
        return requirement_status, relation_status, retrieval_complete

    def _execute_operation(self, operation, outputs, seeds, frame=None):
        # Normal query execution initializes these in _run_query_retrieval. Defensive
        # initialization keeps direct operation tests/restored callers deterministic too.
        if not isinstance(
            getattr(self, "_last_requirement_graph_discoveries", None), dict
        ):
            self._last_requirement_graph_discoveries = {}
        if not isinstance(
            getattr(self, "_requirement_graph_retrieval_stats", None), dict
        ):
            self._requirement_graph_retrieval_stats = {
                "version": getattr(
                    self, "REQUIREMENT_GRAPH_VERSION", "requirement-graph-v1"
                ),
                "canonical_address_additions": 0,
                "stored_relation_additions": 0,
                "candidate_world_peak": 0,
            }
        return super()._execute_operation(operation, outputs, seeds, frame)

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
        run = super()._run_query_retrieval(
            question,
            initial_seeds,
            frame,
            fast_supports,
            gate,
            planning_seeds=planning_seeds,
            planning_context=planning_context,
        )
        self._rg_runtime_requirement_status = dict(
            run.get("requirement_status") or {}
        )
        self._rg_runtime_relation_status = dict(run.get("relation_status") or {})
        return run

    def _role_aware_support_ids(self, slots, slot_support, candidate_order, limit):
        """Return a proof-complete minimal set when extra relation context is unnecessary."""
        selected = list(
            super()._role_aware_support_ids(
                slots, slot_support, candidate_order, limit
            )
        )
        if not selected:
            return selected

        links = list(getattr(self, "_active_query_links", []) or [])
        blocking_relations = {"CAUSES", "COMPARE", "TEMPORAL_ORDER", "VERIFY_SOURCE"}
        if any(
            str(link.get("type") or "").upper() in blocking_relations
            for link in links
        ):
            return selected
        if getattr(self, "_last_candidate_propositions", None):
            return selected

        semantic_slot_checker = getattr(self, "_semantic_context_slot", None)
        unique_slots = []
        seen_slots = set()
        for slot in slots or []:
            slot_id = str(slot.get("id") or "")
            if slot_id and slot_id not in seen_slots:
                unique_slots.append(slot)
                seen_slots.add(slot_id)
        semantic_slots = [
            slot
            for slot in unique_slots
            if not callable(semantic_slot_checker)
            or semantic_slot_checker(slot)
        ]
        if not semantic_slots:
            return selected

        statuses = (
            getattr(self, "_rg_runtime_requirement_status", {}) or
            getattr(self, "_last_requirement_status", {}) or {}
        )
        relation_status = getattr(self, "_rg_runtime_relation_status", {}) or {}
        if any(
            str(status or "").upper() != "PROVEN"
            for status in relation_status.values()
        ):
            return selected
        if any(
            statuses.get(str(slot.get("id") or "")) != "FOUND"
            for slot in unique_slots
        ):
            return selected

        proofs = getattr(self, "_last_requirement_proof_support", {}) or {}
        selected_set = set(selected)
        essential = []
        for slot in unique_slots:
            slot_id = str(slot.get("id") or "")
            is_semantic = (
                not callable(semantic_slot_checker)
                or semantic_slot_checker(slot)
            )
            source_ids = (
                proofs.get(slot_id) or []
                if is_semantic
                else (slot_support or {}).get(slot_id) or []
            )
            support_id = next(
                (
                    memory_id
                    for memory_id in source_ids
                    if memory_id in selected_set
                ),
                "",
            )
            if not support_id:
                return selected
            if support_id not in essential:
                essential.append(support_id)

        if not essential:
            return selected

        telemetry = dict(
            getattr(self, "_last_query_memory_alignment", {}) or {}
        )
        telemetry.update(
            {
                "pre_compaction_ids": list(selected),
                "selected_ids": list(essential),
                "selection_semantics": "smallest_proof_complete_evidence_set",
                "sufficient_context_compacted": len(essential) < len(selected),
            }
        )
        self._last_query_memory_alignment = telemetry
        return essential

    def prepare_batch_query(self, question, system_message=None, **kwargs):
        prepared = super().prepare_batch_query(
            question, system_message=system_message, **kwargs
        )
        extra = prepared.setdefault("extra", {})
        extra[
            "requirement_graph_runtime_version"
        ] = self.REQUIREMENT_GRAPH_RUNTIME_VERSION
        plan = extra.get("replan") or extra.get("plan") or {}
        slots = list(plan.get("required_slots") or [])
        extra["requirement_canonical_addresses"] = {
            str(slot.get("id") or ""): deepcopy(
                slot.get("canonical_anchor_source") or {}
            )
            for slot in slots
            if slot.get("id")
        }
        extra["requirement_graph_stop_guard"] = deepcopy(
            getattr(self, "_last_requirement_graph_stop_guard", {}) or {}
        )
        return prepared
