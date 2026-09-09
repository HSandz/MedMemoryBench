"""Proof-gated premise acquisition for the locked SmartMem0 READ path.

Rank relevance may admit a candidate, but direct semantic acquisition is not closed until
an answer-bearing target is grounded. Inference-sensitive requirements may also retain a
tiny same-source context envelope. No model call, benchmark route, or graph hop is added;
ProofContext remains the sole final-context owner.
"""

from copy import deepcopy


class ReadReasoningCompletionMixin:
    REASONING_COMPLETION_VERSION = "proof-gated-premise-recall-v1"
    ACQUISITION_AUX_MAX_VIEWS = 2
    ACQUISITION_REASONING_AUX_VIEWS = 1
    ACQUISITION_NOVEL_PER_VIEW = 2
    ACQUISITION_SOURCE_NOVEL = 2
    ACQUISITION_HARD_CAP = 16
    ACQUISITION_VIEW_ORDER = ("keys", "family", "question")
    ACQUISITION_RELATIONS = frozenset(
        {"COMPARE", "CAUSES", "POSSIBLE_CAUSE", "DEPENDS_ON", "TEMPORAL_ORDER", "INFER", "CURRENT"}
    )

    def _rcm_stats(self):
        stats = getattr(self, "_reasoning_completion_stats", None)
        if not isinstance(stats, dict):
            stats = {
                "version": self.REASONING_COMPLETION_VERSION,
                "proof_gate_checks": 0,
                "proof_gate_misses": 0,
                "proof_gated_requirements": 0,
                "auxiliary_views_opened": 0,
                "auxiliary_candidate_additions": 0,
                "proof_cold_expansions": 0,
                "proof_cold_additions": 0,
                "source_neighbor_additions": 0,
                "candidate_world_peak": 0,
                "reasoning_prompt_strengthened": False,
            }
            self._reasoning_completion_stats = stats
        return stats

    @staticmethod
    def _rcm_id(memory):
        return str((memory or {}).get("id") or "")

    @staticmethod
    def _rcm_unique(values):
        out, seen = [], set()
        for value in values or []:
            value = str(value or "")
            if value and value not in seen:
                out.append(value)
                seen.add(value)
        return out

    def _rcm_requires_proof(self, slot):
        if not isinstance(slot, dict) or slot.get("degraded"):
            return False
        if str(slot.get("type") or "DIRECT").upper() != "DIRECT":
            return False
        if not str(slot.get("proof_anchor") or slot.get("target_surface") or "").strip():
            return False
        semantic_slot = getattr(self, "_semantic_context_slot", None)
        if callable(semantic_slot):
            try:
                return bool(semantic_slot(slot))
            except Exception:
                return False
        return str(slot.get("evidence_role") or "").upper() in {"REQUIREMENT", "COMPARAND"}

    def _rcm_proof_ids(self, slot, rows):
        if not self._rcm_requires_proof(slot):
            return []
        checker = getattr(self, "_requirement_target_proof", None)
        if not callable(checker):
            return []
        out = []
        for memory in rows or []:
            memory_id = self._rcm_id(memory)
            if not memory_id:
                continue
            try:
                if checker(slot, memory):
                    out.append(memory_id)
            except Exception:
                pass
        return self._rcm_unique(out)

    def _rcm_reasoning_ids(self, plan):
        plan = plan or {}
        relations = list(plan.get("semantic_relations") or [])
        relation_types = {
            str((relation or {}).get("type") or "").upper()
            for relation in relations
            if isinstance(relation, dict)
        }
        reasoning = bool((plan.get("query_spec") or {}).get("requires_inference")) or bool(
            relation_types & self.ACQUISITION_RELATIONS
        )
        if not reasoning:
            return set()
        ids = set()
        for relation in relations:
            if str((relation or {}).get("type") or "").upper() not in self.ACQUISITION_RELATIONS:
                continue
            for key in ("from", "to"):
                value = str((relation or {}).get(key) or "")
                if value and value != "ANSWER":
                    ids.add(value)
        if not ids:
            ids = {
                str(slot.get("id") or "")
                for slot in plan.get("required_slots") or []
                if slot.get("id")
            }
        return ids

    def _compile_gap_operations(self, slots, question, budget_tier="MEDIUM", plan=None):
        operations = list(super()._compile_gap_operations(slots, question, budget_tier, plan=plan))
        slot_by_id = {
            str(slot.get("id") or ""): deepcopy(slot)
            for slot in slots or []
            if isinstance(slot, dict) and slot.get("id")
        }
        reasoning_ids = self._rcm_reasoning_ids(plan or {})
        for operation in operations:
            op = str(operation.get("_lean_op") or operation.get("op") or "").upper()
            produced = [str(value) for value in operation.get("produces") or [] if str(value)]
            if op != "SEARCH_FAMILY" or len(produced) != 1 or produced[0] not in slot_by_id:
                continue
            operation["_acquisition_requirement"] = slot_by_id[produced[0]]
            operation["_reasoning_acquisition"] = produced[0] in reasoning_ids
        return operations

    @staticmethod
    def _rcm_views(operation):
        out, seen = [], set()
        for raw in operation.get("retrieval_views") or []:
            if not isinstance(raw, dict):
                continue
            kind = str(raw.get("kind") or "").lower()
            query = " ".join(str(raw.get("query") or "").split()).strip()
            if kind and query and kind not in seen:
                out.append(dict(raw, kind=kind, query=query))
                seen.add(kind)
        return out

    def _rcm_aux_views(self, operation):
        views = self._rcm_views(operation)
        by_kind = {view["kind"]: view for view in views}
        primary = "obligation" if "obligation" in by_kind else (views[0]["kind"] if views else "")
        ordered = [by_kind[k] for k in self.ACQUISITION_VIEW_ORDER if k != primary and k in by_kind]
        ordered.extend(view for view in views if view["kind"] != primary and view not in ordered)
        return ordered

    @staticmethod
    def _rcm_child(operation, query):
        child = deepcopy(operation)
        for key in (
            "retrieval_views", "retrieval_view_semantics", "requirement_id",
            "_acquisition_requirement", "_reasoning_acquisition",
        ):
            child.pop(key, None)
        child["query"] = query
        return child

    def _rcm_merge(self, rows, additions, cap, source_neighbor=False):
        out = [deepcopy(row) for row in rows or []]
        seen = {self._rcm_id(row) for row in out if self._rcm_id(row)}
        added = []
        for raw in additions or []:
            memory_id = self._rcm_id(raw)
            if not memory_id or memory_id in seen or len(out) >= cap:
                continue
            memory = deepcopy(raw)
            if source_neighbor:
                memory["_source_neighbor_context"] = True
                memory["_supplementary_context"] = True
            out.append(memory)
            seen.add(memory_id)
            added.append(memory_id)
        return out, added

    def _rcm_visible(self, slot, memory):
        visible = getattr(self, "_query_visible_memory", None)
        if callable(visible):
            try:
                if not visible(memory):
                    return False
            except TypeError:
                try:
                    if not visible(memory, include_history=False):
                        return False
                except Exception:
                    pass
            except Exception:
                pass
        owner = getattr(self, "_rc_owner_match", None)
        if callable(owner):
            try:
                return bool(owner(slot, memory))
            except Exception:
                return False
        return True

    def _rcm_cold(self, slot, operation, frame, existing_ids, limit):
        tier_ids = getattr(self, "_retrieval_tier_ids", None)
        hybrid = getattr(self, "_hybrid_search", None)
        if limit <= 0 or not callable(tier_ids) or not callable(hybrid):
            return []
        try:
            ids = set(tier_ids(frame, "COLD", include_history=bool(slot.get("history"))) or [])
        except Exception:
            return []
        ids.difference_update(existing_ids)
        if not ids:
            return []
        views = self._rcm_views(operation)
        primary = next((view for view in views if view["kind"] == "obligation"), None)
        query = str((primary or {}).get("query") or operation.get("query") or "").strip()
        if not query:
            return []
        try:
            rows = list(hybrid(query, top_k=min(max(limit * 2, limit), len(ids)), candidate_ids=ids) or [])
        except Exception:
            return []
        return [deepcopy(row) for row in rows if self._rcm_visible(slot, row)][:limit]

    @staticmethod
    def _rcm_atom_position(memory):
        window = index = 10**6
        for part in str(memory.get("atom_id") or "").split(":"):
            if part.startswith("w") and part[1:].isdigit():
                window = int(part[1:])
            elif part.startswith("a") and part[1:].isdigit():
                index = int(part[1:])
        return window, index

    def _rcm_source_neighbors(self, slot, rows, existing_ids, limit):
        if limit <= 0:
            return []
        proof_ids = set(self._rcm_proof_ids(slot, rows))
        anchors = sorted(
            [row for row in rows or [] if self._rcm_id(row)],
            key=lambda row: (0 if self._rcm_id(row) in proof_ids else 1, 0 if row.get("capsule_id") else 1),
        )[:2]
        ranked = []
        for anchor_rank, anchor in enumerate(anchors):
            capsule, session = str(anchor.get("capsule_id") or ""), str(anchor.get("session_idx") or "")
            if not capsule and not session:
                continue
            anchor_pos = self._rcm_atom_position(anchor)
            for memory in getattr(self, "_memories", []) or []:
                memory_id = self._rcm_id(memory)
                if not memory_id or memory_id in existing_ids or not self._rcm_visible(slot, memory):
                    continue
                same_capsule = bool(capsule and str(memory.get("capsule_id") or "") == capsule)
                same_session = bool(session and str(memory.get("session_idx") or "") == session)
                if not same_capsule and not same_session:
                    continue
                pos = self._rcm_atom_position(memory)
                distance = abs(pos[0] - anchor_pos[0]) * 1000 + abs(pos[1] - anchor_pos[1])
                ranked.append((0 if same_capsule else 1, anchor_rank, distance, memory_id, memory))
        ranked.sort(key=lambda item: item[:-1])
        out, seen = [], set(existing_ids)
        for *_, memory in ranked:
            memory_id = self._rcm_id(memory)
            if memory_id in seen:
                continue
            out.append(deepcopy(memory))
            seen.add(memory_id)
            if len(out) >= limit:
                break
        return out

    def _execute_operation(self, operation, outputs, seeds, frame=None):
        result, relations, evidence_refs = super()._execute_operation(operation, outputs, seeds, frame)
        op = str(operation.get("_lean_op") or operation.get("op") or "").upper()
        slot = operation.get("_acquisition_requirement")
        if op != "SEARCH_FAMILY" or not isinstance(slot, dict):
            return result, relations, evidence_refs

        rows = [deepcopy(row) for row in result or []]
        cap = min(self.ACQUISITION_HARD_CAP, max(len(rows), 1) + 6)
        stats = self._rcm_stats()
        proof_required = self._rcm_requires_proof(slot)
        proof_ids = self._rcm_proof_ids(slot, rows) if proof_required else []
        if proof_required:
            stats["proof_gate_checks"] += 1
            if not proof_ids:
                stats["proof_gate_misses"] += 1

        reasoning = bool(operation.get("_reasoning_acquisition"))
        max_aux = self.ACQUISITION_AUX_MAX_VIEWS if proof_required and not proof_ids else (
            self.ACQUISITION_REASONING_AUX_VIEWS if reasoning else 0
        )
        opened = 0
        for view in self._rcm_aux_views(operation)[:max_aux]:
            child_rows, child_relations, child_evidence = super()._execute_operation(
                self._rcm_child(operation, view["query"]), outputs, seeds, frame
            )
            opened += 1
            relations = list(relations or [])
            for relation in child_relations or []:
                if relation not in relations:
                    relations.append(deepcopy(relation))
            evidence_refs = self._rcm_unique([*(evidence_refs or []), *(child_evidence or [])])
            existing = {self._rcm_id(row) for row in rows}
            novel = [row for row in child_rows or [] if self._rcm_id(row) not in existing][
                : self.ACQUISITION_NOVEL_PER_VIEW
            ]
            rows, added = self._rcm_merge(rows, novel, cap)
            stats["auxiliary_candidate_additions"] += len(added)
            if proof_required and self._rcm_proof_ids(slot, rows):
                break
        stats["auxiliary_views_opened"] += opened

        if proof_required and not self._rcm_proof_ids(slot, rows):
            cold = self._rcm_cold(
                slot, operation, frame, {self._rcm_id(row) for row in rows},
                min(self.ACQUISITION_NOVEL_PER_VIEW, max(0, cap - len(rows))),
            )
            if cold:
                stats["proof_cold_expansions"] += 1
                rows, added = self._rcm_merge(rows, cold, cap)
                stats["proof_cold_additions"] += len(added)

        if reasoning and rows:
            neighbors = self._rcm_source_neighbors(
                slot, rows, {self._rcm_id(row) for row in rows},
                min(self.ACQUISITION_SOURCE_NOVEL, max(0, cap - len(rows))),
            )
            rows, added = self._rcm_merge(rows, neighbors, cap, source_neighbor=True)
            stats["source_neighbor_additions"] += len(added)

        proof_set = set(self._rcm_proof_ids(slot, rows)) if proof_required else set()
        if proof_set:
            rows.sort(key=lambda memory: 0 if self._rcm_id(memory) in proof_set else 1)
        stats["candidate_world_peak"] = max(stats["candidate_world_peak"], len(rows))
        return rows[:cap], relations or [], evidence_refs or []

    def _retrieval_status(self, plan, slot_support, selected, relations):
        requirement_status, relation_status, complete = super()._retrieval_status(
            plan, slot_support, selected, relations
        )
        original = dict(requirement_status or {})
        selected_by_id = {self._rcm_id(memory): memory for memory in selected or [] if self._rcm_id(memory)}
        gated = 0
        for slot in (plan or {}).get("required_slots") or []:
            slot_id = str(slot.get("id") or "")
            if original.get(slot_id) != "FOUND" or not self._rcm_requires_proof(slot):
                continue
            supports = [
                selected_by_id[memory_id]
                for memory_id in (slot_support or {}).get(slot_id, [])
                if memory_id in selected_by_id
            ]
            if not self._rcm_proof_ids(slot, supports):
                requirement_status[slot_id] = "EMPTY"
                gated += 1
        if gated:
            complete = False
            self._rcm_stats()["proof_gated_requirements"] += gated
        self._last_retrieval_viability = {
            "requirements": dict(requirement_status or {}),
            "relations": dict(relation_status or {}),
            "complete": bool(complete),
            "candidate_viability_requirements": original,
            "semantics": "candidate_viability_plus_target_proof_acquisition_gate",
        }
        return requirement_status, relation_status, bool(complete)

    def _run_query_retrieval(
        self, question, initial_seeds, frame, fast_supports, gate,
        planning_seeds=None, planning_context=None,
    ):
        self._reasoning_completion_stats = None
        self._rcm_stats()
        run = super()._run_query_retrieval(
            question, initial_seeds, frame, fast_supports, gate,
            planning_seeds=planning_seeds, planning_context=planning_context,
        )
        run["reasoning_completion"] = deepcopy(self._rcm_stats())
        return run

    @staticmethod
    def _rcm_reasoning_plan(prepared):
        extra = prepared.get("extra") if isinstance(prepared, dict) else {}
        plan = (extra or {}).get("replan") or (extra or {}).get("plan") or {}
        return bool((plan.get("query_spec") or {}).get("requires_inference")) or bool(
            plan.get("semantic_relations") or []
        )

    def _rcm_strengthen_prompt(self, prepared):
        if not self._rcm_reasoning_plan(prepared):
            return False
        instruction = (
            " REASONING COMPLETION: distinguish participant-specific nodes from general-domain "
            "bridge nodes. Use every grounded participant premise that materially constrains the "
            "inference, preserve relevant chronology, and never replace a missing participant "
            "premise with generic knowledge. When a general-domain bridge is authorized, use it "
            "only between grounded participant nodes and make each intermediate link explicit."
        )
        for message in prepared.get("messages") or []:
            if not isinstance(message, dict) or str(message.get("role") or "").lower() != "system":
                continue
            if isinstance(message.get("content"), str):
                message["content"] += instruction
                return True
            for part in message.get("content") or []:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    part["text"] += instruction
                    return True
        return False

    def prepare_batch_query(self, question, system_message=None, **kwargs):
        prepared = super().prepare_batch_query(question, system_message=system_message, **kwargs)
        self._rcm_stats()["reasoning_prompt_strengthened"] = self._rcm_strengthen_prompt(prepared)
        extra = prepared.setdefault("extra", {})
        extra["reasoning_completion_version"] = self.REASONING_COMPLETION_VERSION
        extra["reasoning_completion"] = deepcopy(self._rcm_stats())
        return prepared
