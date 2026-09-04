"""Deterministic compiler and proof rules for minimal-IR SmartMem0 reads."""

from collections import defaultdict
from typing import Any, Dict, List

from .canonicalization import state_identity
from .contracts import QueryFrame, RETRIEVAL_BUDGETS, VALID_TEMPORAL_AXES


class ReadExecutionContractMixin:

    def _rc_memory_target_text(self, memory: Dict[str, Any]) -> str:
        return " ".join(
            str(value or "")
            for value in (
                memory.get("claim"),
                self._memory_value(memory),
                memory.get("verbatim_value"),
                memory.get("scope"),
                memory.get("state_key"),
                memory.get("object_anchor"),
                " ".join(memory.get("entities", [])),
                " ".join(memory.get("scope_entities", [])),
            )
        )

    def _rc_memory_matches_target(
        self, slot: Dict[str, Any], memory: Dict[str, Any]
    ) -> bool:
        """Prove against question-owned text; resolved keys are ranking hints only."""
        target = str(slot.get("target_surface") or "").strip()
        if not target:
            return False
        text = self._rc_memory_target_text(memory)
        if self._rc_token_sequence_present(target, text):
            return True
        target_terms = self._rc_content_terms(target)
        if not target_terms:
            return False
        text_terms = set(self._rc_content_terms(text))
        overlap = sum(term in text_terms for term in target_terms)
        # A long question span should not require verbatim identity, but one
        # generic token is too weak to prove a target.
        threshold = 1 if len(target_terms) <= 2 else 2
        return overlap >= threshold

    def _slot_contract_match(self, slot, memory, strict_targets=None):
        if not self._rc_owner_match(slot, memory):
            return False
        for field in slot.get("required_fields") or []:
            if field in VALID_TEMPORAL_AXES:
                if not self._date_for(memory, field):
                    return False
            elif field and not memory.get(field):
                return False
        role = str(slot.get("evidence_role") or "").upper()
        if strict_targets is None:
            strict_targets = role in {"ANSWER", "GENERIC_EVIDENCE", "COMPARAND"}
        if strict_targets and not self._rc_memory_matches_target(slot, memory):
            return False
        return True

    def _memory_matches_slot_role(self, slot, memory):
        if not self._rc_owner_match(slot, memory):
            return False
        role = str(slot.get("evidence_role") or "").upper()
        semantic_role = str(memory.get("semantic_role") or "").upper()
        tags = {str(tag).upper() for tag in memory.get("planning_tags", [])}
        kind = str(memory.get("kind") or "").upper()
        if role == "OPTION_CONTEXT":
            # Option-context evidence is authorized later by shared option
            # probes. Do not declare arbitrary same-owner memory relevant here.
            return bool(slot.get("target_surface")) and self._slot_contract_match(
                slot, memory, True
            )
        if role == "COMPARAND":
            return self._slot_contract_match(slot, memory, True)
        if role == "PRIOR_TRAJECTORY":
            return bool(
                self._date_for(memory, "event_time")
                or self._date_for(memory, "document_time")
                or state_identity(memory)
                or "TRAJECTORY" in tags
            ) and self._slot_contract_match(slot, memory, False)
        if role == "CONSTRAINT":
            return (
                semantic_role
                in {"SAFETY_CONSTRAINT", "ACCEPTED_POLICY", "PREFERENCE", "GUIDANCE"}
                or bool(tags.intersection({"CONSTRAINT", "RISK"}))
            ) and self._slot_contract_match(slot, memory, False)
        if role == "ACTION_RULE":
            return (
                semantic_role in {"SAFETY_CONSTRAINT", "ACCEPTED_POLICY", "GUIDANCE"}
                or "CONSTRAINT" in tags
            ) and self._slot_contract_match(slot, memory, False)
        if role == "FOCAL_STATE":
            return (
                kind == "STATE" or semantic_role in {"MEASUREMENT", "OBSERVATION"}
            ) and self._slot_contract_match(slot, memory, False)
        return self._slot_contract_match(
            slot,
            memory,
            role in {"ANSWER", "GENERIC_EVIDENCE", "FOCAL_TRIGGER", "OUTCOME"},
        )

    def _rc_search_query(self, slot: Dict[str, Any], question: str) -> str:
        parts = [
            str(slot.get("retrieval_hint") or "").strip(),
            str(slot.get("target_surface") or "").strip(),
            " ".join(str(key) for key in (slot.get("resolved_keys") or [])[:2]),
            str(question or "").strip(),
        ]
        unique = []
        for part in parts:
            if part and self._rc_text(part) not in {self._rc_text(x) for x in unique}:
                unique.append(part)
        return " | ".join(unique)

    def _rc_bundle_query(self, slots: List[Dict[str, Any]], question: str) -> str:
        parts = []
        for slot in slots:
            hint = str(slot.get("retrieval_hint") or "").strip()
            if hint and hint not in parts:
                parts.append(hint)
            target = str(slot.get("target_surface") or "").strip()
            if target and target not in parts:
                parts.append(target)
            for key in (slot.get("resolved_keys") or [])[:1]:
                if key and key not in parts:
                    parts.append(str(key))
        if question:
            parts.append(question)
        return " | ".join(part for part in parts if part)

    def _compile_gap_operations(self, slots, question, budget_tier="MEDIUM", plan=None):
        if not slots:
            return []
        plan = plan or {}
        max_ops = RETRIEVAL_BUDGETS.get(budget_tier, {}).get("max_operations", 4)
        options = plan.get("visible_options") or self._question_options(question) or {}
        if options:
            option_queries = [
                {"label": str(label), "query": str(text)}
                for label, text in options.items()
            ]
            return [
                {
                    "op": "SEMANTIC_SEARCH",
                    "query": self._rc_bundle_query(
                        slots, self._question_stem(question)
                    ),
                    "top_k": 8,
                    "strategy": "SHARED_OPTIONS",
                    "option_queries": option_queries,
                    "produces": [slot["id"] for slot in slots],
                }
            ]

        operations = []
        slot_by_id = {slot["id"]: slot for slot in slots}
        handled = set()
        for relation in plan.get("semantic_relations") or []:
            if relation.get("type") != "TEMPORAL_ORDER" or len(operations) >= max_ops:
                continue
            source_id, target_id = relation.get("from"), relation.get("to")
            source, target = slot_by_id.get(source_id), slot_by_id.get(target_id)
            order = str(relation.get("relation") or "").upper()
            if not source or not target or order not in {"BEFORE", "AFTER", "OVERLAPS"}:
                continue
            anchor_index = len(operations)
            operations.append(
                {
                    "op": "SEMANTIC_SEARCH",
                    "query": self._rc_search_query(target, question),
                    "top_k": 4,
                    "strategy": "FOCAL",
                    "produces": [target_id],
                }
            )
            if len(operations) < max_ops:
                operations.append(
                    {
                        "op": "TEMPORAL_FILTER",
                        "query": self._rc_search_query(source, question),
                        "relation": order,
                        "axis": "event_time",
                        "fallback_axis": "",
                        "anchor": f"${anchor_index}",
                        "produces": [source_id],
                    }
                )
                handled.update({source_id, target_id})

        for relation in plan.get("semantic_relations") or []:
            if relation.get("type") != "CAUSES" or len(operations) >= max_ops:
                continue
            source_id, target_id = relation.get("from"), relation.get("to")
            source, target = slot_by_id.get(source_id), slot_by_id.get(target_id)
            if not source or not target:
                continue
            anchor_index = len(operations)
            operations.append(
                {
                    "op": "LOCATE_ANCHOR",
                    "query": self._rc_search_query(source, question),
                    "produces": [source_id],
                }
            )
            if len(operations) < max_ops:
                operations.append(
                    {
                        "op": "FOLLOW_CAUSES",
                        "start": [f"${anchor_index}"],
                        "direction": "OUT",
                        "depth": 3,
                        "goal": self._rc_search_query(target, question),
                        "produces": [source_id, target_id],
                    }
                )
                handled.update({source_id, target_id})

        for slot in slots:
            if len(operations) >= max_ops:
                break
            slot_id, search_query = slot["id"], self._rc_search_query(slot, question)
            if slot_id in handled:
                continue
            slot_type = str(slot.get("type") or "DIRECT").upper()
            if slot_type == "CURRENT_STATE":
                operations.append(
                    {
                        "op": "RESOLVE_STATE",
                        "query": search_query,
                        "produces": [slot_id],
                    }
                )
                continue
            if slot_type == "TEMPORAL":
                relation = str(
                    slot.get("temporal_relation")
                    or slot.get("time_relation")
                    or "LOCATE"
                ).upper()
                axis = str(slot.get("time_axis") or "event_time").lower()
                anchor, end = str(slot.get("time_anchor") or ""), str(
                    slot.get("time_end") or ""
                )
                if relation == "EXACT" and not anchor:
                    relation = "LOCATE"
                if relation in {"BEFORE", "AFTER"} and not anchor:
                    relation = "LOCATE"
                if relation == "BETWEEN" and (not anchor or not end):
                    relation = "LOCATE"
                if relation in {"EARLIEST", "LATEST"}:
                    index = len(operations)
                    operations.append(
                        {
                            "op": "LOCATE_ANCHOR",
                            "query": search_query,
                            "produces": [slot_id],
                        }
                    )
                    if len(operations) < max_ops:
                        operations.append(
                            {
                                "op": "TEMPORAL_FILTER",
                                "query": search_query,
                                "relation": relation,
                                "axis": axis,
                                "fallback_axis": "",
                                "candidate_refs": [f"${index}"],
                                "produces": [slot_id],
                            }
                        )
                else:
                    operations.append(
                        {
                            "op": "TEMPORAL_FILTER",
                            "query": search_query,
                            "relation": relation,
                            "axis": axis,
                            "fallback_axis": "",
                            "anchor": anchor,
                            "end": end,
                            "produces": [slot_id],
                        }
                    )
                continue
            operations.append(
                {
                    "op": "SEMANTIC_SEARCH",
                    "query": search_query,
                    "top_k": 6,
                    "strategy": "FOCAL",
                    "produces": [slot_id],
                }
            )
        return operations[:max_ops]

    @staticmethod
    def _rc_operation_signature(operation):
        return (
            str(operation.get("op") or ""),
            str(operation.get("strategy") or ""),
            str(operation.get("query") or ""),
            str(operation.get("relation") or ""),
            str(operation.get("axis") or ""),
            str(operation.get("anchor") or ""),
            str(operation.get("end") or ""),
        )

    def _make_deterministic_recovery_plan(self, missing_slots, question, existing_plan):
        if not missing_slots:
            return None
        # Recovery must open a genuinely different evidence surface. Its first
        # deterministic broadening step is to remove optional resolver hints and
        # search from the question-owned target again.
        broadened = []
        changed = False
        for slot in missing_slots:
            copy = dict(slot)
            if copy.get("resolved_keys"):
                copy["resolved_keys"] = []
                changed = True
            elif copy.get("retrieval_hint"):
                copy["retrieval_hint"] = ""
                changed = True
            broadened.append(copy)
        if not changed:
            return None
        budget = existing_plan.get("budget_tier", "MEDIUM")
        shell = {
            "query_mode": existing_plan.get("query_mode", "DIRECT"),
            "visible_options": existing_plan.get("visible_options", {}),
            "semantic_relations": existing_plan.get("semantic_relations", []),
        }
        operations = self._compile_gap_operations(
            broadened, question, budget, plan=shell
        )
        previous = {
            self._rc_operation_signature(op)
            for op in existing_plan.get("operations", [])
        }
        if not operations or all(
            self._rc_operation_signature(op) in previous for op in operations
        ):
            return None
        return {
            "query_spec": existing_plan.get("query_spec", {}),
            "query_mode": shell["query_mode"],
            "required_slots": broadened,
            "semantic_relations": shell["semantic_relations"],
            "seed_coverage": [],
            "operations": operations,
            "option_coverage": [],
            "visible_options": shell["visible_options"],
            "need_evidence": False,
            "need_raw_evidence": False,
            "budget_tier": budget,
            "max_memories": RETRIEVAL_BUDGETS.get(budget, RETRIEVAL_BUDGETS["MEDIUM"])[
                "max_memories"
            ],
            "planner_fallback": True,
            "fallback_reason": "deterministic_broaden_without_resolved_keys",
            "valid": True,
        }

    def _rc_option_probe_labels_for_memory(self, memory_id: str) -> set:
        return {
            str(label)
            for label, ids in (
                getattr(self, "_last_option_probe_coverage", {}) or {}
            ).items()
            if memory_id in set(ids or [])
        }

    def _slot_covered(self, slot, support_ids, selected, relations):
        support_set = set(support_ids)
        memories = [memory for memory in selected if memory.get("id") in support_set]
        if not memories:
            return False
        slot_type = str(slot.get("type") or "DIRECT").upper()
        role = str(slot.get("evidence_role") or "").upper()
        if slot_type == "DIRECT":
            if role == "OPTION_CONTEXT":
                labels = set()
                for memory in memories:
                    labels.update(self._rc_option_probe_labels_for_memory(memory["id"]))
                # Exploration breadth proves only that the shared bundle looked
                # across choices; it does not imply any particular option is true.
                return len(labels) >= 2
            valid = [
                memory
                for memory in memories
                if self._memory_value(memory)
                and memory.get("assertion_mode", "DIRECT") in {"DIRECT", "RECAP"}
                and self._memory_matches_slot_role(slot, memory)
                and (
                    slot.get("history")
                    or memory.get(
                        "_status", self._belief_status.get(memory.get("id"), "active")
                    )
                    != "superseded"
                )
            ]
            return len({memory["id"] for memory in valid}) >= (
                2 if role == "PRIOR_TRAJECTORY" else 1
            )
        if slot_type == "CURRENT_STATE":
            heads = [
                memory
                for memory in memories
                if self._is_state_head(memory)
                and not self._has_competing_active_value(memory)
                and self._memory_value(memory)
                and self._slot_contract_match(slot, memory, role != "REQUIREMENT")
            ]
            identities = {
                state_identity(memory) for memory in heads if state_identity(memory)
            }
            return bool(heads and len(identities) == 1)
        if slot_type == "TEMPORAL":
            axis = str(slot.get("time_axis") or "").lower()
            relation = str(
                slot.get("temporal_relation") or slot.get("time_relation") or "LOCATE"
            ).upper()
            raw_anchor = str(
                slot.get("resolved_time_anchor") or slot.get("time_anchor") or ""
            )
            raw_end = str(slot.get("time_end") or "")
            anchor, end = self._parse_date(raw_anchor), self._parse_date(raw_end)
            if axis not in VALID_TEMPORAL_AXES:
                return False

            def good(memory):
                date = self._date_for(memory, axis)
                if not date or not self._slot_contract_match(
                    slot, memory, role != "REQUIREMENT"
                ):
                    return False
                if relation == "LOCATE":
                    return True
                if relation == "EXACT":
                    return bool(raw_anchor and self._date_matches(date, raw_anchor))
                if relation == "BEFORE":
                    return bool(anchor and date < anchor)
                if relation == "AFTER":
                    return bool(anchor and date > anchor)
                if relation == "BETWEEN":
                    return bool(anchor and end and anchor <= date <= end)
                if relation == "OVERLAPS":
                    return bool(anchor and date == anchor)
                return relation in {"EARLIEST", "LATEST"}

            return any(good(memory) for memory in memories)
        if slot_type == "CAUSE_PATH":
            by_id, adjacency = {
                memory["id"]: memory for memory in memories
            }, defaultdict(list)
            for relation in relations:
                if self._valid_causal_relation(relation, by_id):
                    adjacency[relation["source_id"]].append(relation["target_id"])
            if len(by_id) < 2 or not any(adjacency.values()):
                return False
            for root in by_id:
                seen, queue = {root}, [root]
                while queue:
                    for target in adjacency.get(queue.pop(0), []):
                        if target not in seen:
                            seen.add(target)
                            queue.append(target)
                if set(by_id).issubset(seen):
                    return True
            return False
        if slot_type == "TRANSITION":
            by_id = {memory["id"]: memory for memory in memories}
            return any(
                relation.get("type") in {"SUPERSEDE", "REFINE"}
                and relation.get("source_id") in by_id
                and relation.get("target_id") in by_id
                for relation in relations
            )
        return False

    def _coverage_map(self, plan, slot_support, selected, relations):
        coverage = {
            slot["id"]: self._slot_covered(
                slot, slot_support.get(slot["id"], []), selected, relations
            )
            for slot in plan.get("required_slots", [])
        }
        support_sets = {
            slot_id: set(memory_ids) for slot_id, memory_ids in slot_support.items()
        }
        selected_by_id = {memory["id"]: memory for memory in selected}

        def invalidate(*requirement_ids):
            for requirement_id in requirement_ids:
                if requirement_id in coverage:
                    coverage[requirement_id] = False

        def has_causal_path(source_ids, target_ids):
            adjacency = defaultdict(set)
            for relation in relations:
                if self._valid_causal_relation(relation, selected_by_id):
                    adjacency[relation["source_id"]].add(relation["target_id"])
            frontier = list(source_ids)
            seen = set(frontier)
            while frontier:
                node = frontier.pop(0)
                if node in target_ids:
                    return True
                for target in adjacency.get(node, set()):
                    if target not in seen:
                        seen.add(target)
                        frontier.append(target)
            return False

        for relation in plan.get("semantic_relations") or []:
            relation_type = str(relation.get("type") or "").upper()
            source, target = relation.get("from"), relation.get("to")
            source_ids = support_sets.get(source, set())
            target_ids = support_sets.get(target, set())
            if not source_ids:
                invalidate(source)
                continue
            if target and target != "ANSWER" and not target_ids:
                invalidate(target)
                continue
            if relation_type == "COMPARE" and source_ids == target_ids:
                invalidate(source, target)
            elif relation_type == "CAUSES" and not has_causal_path(
                source_ids, target_ids
            ):
                invalidate(source, target)
        return coverage

    def _operation_slot_support(self, slot, result, relations):
        if not result:
            return []
        role, slot_type = (
            str(slot.get("evidence_role") or "").upper(),
            str(slot.get("type") or "DIRECT").upper(),
        )
        query = self._rc_search_query(slot, "") or str(slot.get("id") or "")
        ranked = self._hybrid_search(
            query, top_k=len(result), candidate_ids={memory["id"] for memory in result}
        )
        by_id = {memory["id"]: memory for memory in result}
        ranked = [by_id[memory["id"]] for memory in ranked if memory["id"] in by_id]
        if role == "OPTION_CONTEXT":
            ranked = sorted(
                ranked,
                key=lambda memory: len(
                    self._rc_option_probe_labels_for_memory(memory["id"])
                ),
                reverse=True,
            )
            return [
                memory
                for memory in ranked
                if self._rc_owner_match(slot, memory)
                and self._rc_option_probe_labels_for_memory(memory["id"])
            ][:6]
        if slot_type == "DIRECT":
            strict = [
                memory
                for memory in ranked
                if self._memory_value(memory)
                and memory.get("assertion_mode", "DIRECT") in {"DIRECT", "RECAP"}
                and self._memory_matches_slot_role(slot, memory)
            ]
            if strict:
                # One retrieval requirement is one evidence obligation. Keep
                # only its best candidate by default; shared-option retrieval
                # deliberately preserves several propositions for final
                # evaluation without turning them into semantic requirements.
                limit = (
                    min(
                        6,
                        max(
                            2,
                            len(getattr(self, "_last_option_probe_coverage", {}) or {}),
                        ),
                    )
                    if role == "REQUIREMENT"
                    and getattr(self, "_last_option_probe_coverage", {})
                    else 1
                )
                return strict[:limit]
            # An atomic categorical question can name the class but not the
            # answer token (e.g. "which chronic metabolic disease"). Keep a
            # bounded same-owner semantic fallback as context, but coverage
            # remains false because _slot_covered still requires target proof.
            if role == "ANSWER":
                return [
                    memory
                    for memory in ranked
                    if self._memory_value(memory) and self._rc_owner_match(slot, memory)
                ][:3]
            return []
        if slot_type == "TEMPORAL":
            return [
                memory
                for memory in ranked
                if self._date_for(memory, str(slot.get("time_axis") or "event_time"))
                and self._slot_contract_match(slot, memory, role != "REQUIREMENT")
            ][:4]
        if slot_type == "CURRENT_STATE":
            return [
                memory
                for memory in ranked
                if self._is_state_head(memory)
                and self._slot_contract_match(slot, memory, role != "REQUIREMENT")
            ][:2]
        if slot_type == "CAUSE_PATH":
            endpoints = set()
            for relation in relations:
                if self._valid_causal_relation(relation, by_id):
                    endpoints.update((relation["source_id"], relation["target_id"]))
            return [memory for memory in ranked if memory["id"] in endpoints]
        return ranked[:4]

    @staticmethod
    def _multiple_choice_answer_instruction(option_labels):
        labels = ", ".join(sorted(str(label) for label in option_labels))
        return (
            f" This is multiple-choice with options {labels}. Identify the exact "
            "predicate in the question stem, then evaluate EVERY option against "
            "that same predicate and the shared participant-specific evidence. "
            "Do not invert the stem's polarity: evidence against an option "
            "disqualifies it for a supported/safe/recommended predicate, but may "
            "support it when the stem asks which action is contraindicated or "
            "unsafe. A "
            "missing personal-memory hit is neither support nor refutation. "
            "Output ONLY all matching uppercase labels separated by commas, or NONE."
        )

    def prepare_batch_query(self, question, system_message=None, **kwargs):
        prepared = super().prepare_batch_query(
            question, system_message=system_message, **kwargs
        )
        prepared.setdefault("extra", {})["read_contract_version"] = "minimal-ir-v1"
        prepared["extra"]["option_probe_coverage"] = dict(
            getattr(self, "_last_option_probe_coverage", {}) or {}
        )
        # A caller-supplied system contract still requires the final formatter;
        # without one, the direct controller answer remains a true one-call path.
        if system_message and prepared.get("precomputed_answer"):
            prepared["precomputed_answer"] = ""
        return prepared
