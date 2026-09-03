"""Deterministic execution/proof invariants for SmartMem0 P1B.2.2 reads."""

from collections import defaultdict
from typing import Any, Dict, List

from .canonicalization import state_identity
from .contracts import QueryFrame, RETRIEVAL_BUDGETS, VALID_TEMPORAL_AXES


class ReadExecutionContractMixin:
    def _rc_memory_target_text(self, memory: Dict[str, Any]) -> str:
        return " ".join(str(value or "") for value in (memory.get("claim"), self._memory_value(memory), memory.get("verbatim_value"), memory.get("scope"), memory.get("state_key"), memory.get("object_anchor"), " ".join(memory.get("entities", [])), " ".join(memory.get("scope_entities", []))))

    def _rc_memory_matches_target(self, slot: Dict[str, Any], memory: Dict[str, Any]) -> bool:
        target = str(slot.get("target_surface") or "").strip()
        resolved = {str(key) for key in (slot.get("resolved_keys") or []) if str(key).strip()}
        memory_keys = set(self._rc_memory_concept_keys(memory))
        text = self._rc_text(self._rc_memory_target_text(memory))
        key_hit = bool(resolved.intersection(memory_keys))
        if not target:
            return not resolved or key_hit
        phrase = self._rc_text(target)
        if phrase and phrase in text:
            return True
        target_terms = self._rc_terms(target)
        if not target_terms:
            return key_hit if resolved else True
        text_terms = set(self._rc_terms(text))
        overlap = sum(term in text_terms for term in target_terms)
        threshold = 1 if len(target_terms) <= 2 else 2
        if resolved and not key_hit:
            return False
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
            return True
        if role == "COMPARAND":
            return self._slot_contract_match(slot, memory, True)
        if role == "PRIOR_TRAJECTORY":
            return bool(self._date_for(memory, "event_time") or self._date_for(memory, "document_time") or state_identity(memory) or "TRAJECTORY" in tags) and self._slot_contract_match(slot, memory, False)
        if role == "CONSTRAINT":
            return (semantic_role in {"SAFETY_CONSTRAINT", "ACCEPTED_POLICY", "PREFERENCE", "GUIDANCE"} or bool(tags.intersection({"CONSTRAINT", "RISK"}))) and self._slot_contract_match(slot, memory, False)
        if role == "ACTION_RULE":
            return (semantic_role in {"SAFETY_CONSTRAINT", "ACCEPTED_POLICY", "GUIDANCE"} or "CONSTRAINT" in tags) and self._slot_contract_match(slot, memory, False)
        if role == "FOCAL_STATE":
            return (kind == "STATE" or semantic_role in {"MEASUREMENT", "OBSERVATION"}) and self._slot_contract_match(slot, memory, False)
        return self._slot_contract_match(slot, memory, role in {"ANSWER", "GENERIC_EVIDENCE", "FOCAL_TRIGGER", "OUTCOME"})

    def _rc_search_query(self, slot: Dict[str, Any], question: str) -> str:
        parts = [str(slot.get("target_surface") or "").strip(), " ".join(str(key) for key in (slot.get("resolved_keys") or [])[:4]), str(question or "").strip()]
        unique = []
        for part in parts:
            if part and self._rc_text(part) not in {self._rc_text(x) for x in unique}:
                unique.append(part)
        return " | ".join(unique)

    def _rc_bundle_query(self, slots: List[Dict[str, Any]], question: str) -> str:
        parts = []
        for slot in slots:
            target = str(slot.get("target_surface") or "").strip()
            if target and target not in parts:
                parts.append(target)
            for key in (slot.get("resolved_keys") or [])[:2]:
                if key and key not in parts:
                    parts.append(str(key))
        parts.append(question)
        return " | ".join(part for part in parts if part)

    def _semantic_operation_search(self, query, top_k, strategy, frame=None, option_queries=None):
        strategy = str(strategy or "FOCAL").upper()
        if strategy != "SHARED_OPTIONS":
            return super()._semantic_operation_search(query, top_k, strategy, frame=frame or QueryFrame(), option_queries=option_queries)
        frame = frame or QueryFrame()
        eligible_ids = {memory["id"] for memory in self._memories if self._memory_satisfies_frame(memory, frame, include_entities=bool(frame.hard_entities)) and self._query_visible_memory(memory)}
        if not eligible_ids:
            self._last_option_probe_coverage = {str(item.get("label") if isinstance(item, dict) else index): [] for index, item in enumerate(option_queries or [])}
            return []
        base = self._hybrid_search(query, top_k=min(max(int(top_k) * 3, 12), len(eligible_ids)), candidate_ids=eligible_ids)
        representatives, option_hits, seen = [], [], set()
        coverage: Dict[str, List[str]] = {}
        for index, item in enumerate(option_queries or []):
            if isinstance(item, dict):
                label = str(item.get("label") or index)
                option_text = str(item.get("query") or item.get("text") or "").strip()
            else:
                label, option_text = str(index), str(item or "").strip()
            if not option_text:
                coverage[label] = []
                continue
            probe = f"{query} | {option_text}" if query else option_text
            hits = self._hybrid_search(probe, top_k=min(4, len(eligible_ids)), candidate_ids=eligible_ids)
            coverage[label] = [memory["id"] for memory in hits[:3]]
            option_hits.extend(hits)
            representative = next((memory for memory in hits if memory["id"] not in seen), None)
            if representative:
                representatives.append(representative)
                seen.add(representative["id"])
        self._last_option_probe_coverage = coverage
        selected, selected_ids = [], set()
        for memory in (*representatives, *base, *option_hits):
            if memory["id"] in selected_ids:
                continue
            selected.append(self._snapshot(memory))
            selected_ids.add(memory["id"])
            if len(selected) >= int(top_k):
                break
        return selected

    def _compile_gap_operations(self, slots, question, budget_tier="MEDIUM", plan=None):
        if not slots:
            return []
        plan = plan or {}
        mode = str(plan.get("query_mode") or slots[0].get("qrf_operator") or "DIRECT").upper()
        max_ops = RETRIEVAL_BUDGETS.get(budget_tier, {}).get("max_operations", 4)
        if mode == "MULTI_OPTION":
            options = plan.get("visible_options") or self._question_options(question) or {}
            option_queries = [{"label": str(label), "query": str(text)} for label, text in options.items()]
            return [{"op": "SEMANTIC_SEARCH", "query": self._rc_bundle_query(slots, self._question_stem(question)), "top_k": 8, "strategy": "SHARED_OPTIONS", "option_queries": option_queries, "produces": [slot["id"] for slot in slots]}]
        if mode in {"DECISION", "CAUSAL", "MULTI_HOP"} and not any(slot.get("type") == "CAUSE_PATH" for slot in slots):
            trajectory = [slot for slot in slots if slot.get("evidence_role") == "PRIOR_TRAJECTORY"]
            focal = [slot for slot in slots if slot not in trajectory]
            operations = []
            if focal:
                operations.append({"op": "SEMANTIC_SEARCH", "query": self._rc_bundle_query(focal, question), "top_k": 6, "strategy": "DECISION_BUNDLE", "produces": [slot["id"] for slot in focal]})
            if trajectory and len(operations) < max_ops:
                operations.append({"op": "SEMANTIC_SEARCH", "query": self._rc_bundle_query(trajectory, question), "top_k": 6, "strategy": "TRAJECTORY", "produces": [slot["id"] for slot in trajectory]})
            return operations[:max_ops]
        operations = []
        for slot in slots:
            if len(operations) >= max_ops:
                break
            slot_id, search_query = slot["id"], self._rc_search_query(slot, question)
            slot_type = str(slot.get("type") or "DIRECT").upper()
            if slot_type == "CURRENT_STATE":
                operations.append({"op": "RESOLVE_STATE", "query": search_query, "produces": [slot_id]})
                continue
            if slot_type == "CAUSE_PATH":
                index = len(operations)
                operations.append({"op": "LOCATE_ANCHOR", "query": search_query, "produces": [slot_id]})
                if len(operations) < max_ops:
                    operations.append({"op": "FOLLOW_CAUSES", "start": [f"${index}"], "direction": "OUT", "depth": 3, "goal": search_query, "produces": [slot_id]})
                continue
            if slot_type == "TEMPORAL":
                relation = str(slot.get("temporal_relation") or slot.get("time_relation") or "LOCATE").upper()
                axis = str(slot.get("time_axis") or "event_time").lower()
                anchor, end = str(slot.get("time_anchor") or ""), str(slot.get("time_end") or "")
                if relation == "EXACT" and not anchor:
                    relation = "LOCATE"
                if relation in {"BEFORE", "AFTER"} and not anchor:
                    relation = "LOCATE"
                if relation == "BETWEEN" and (not anchor or not end):
                    relation = "LOCATE"
                if relation in {"EARLIEST", "LATEST"}:
                    index = len(operations)
                    operations.append({"op": "LOCATE_ANCHOR", "query": search_query, "produces": [slot_id]})
                    if len(operations) < max_ops:
                        operations.append({"op": "TEMPORAL_FILTER", "query": search_query, "relation": relation, "axis": axis, "fallback_axis": "", "candidate_refs": [f"${index}"], "produces": [slot_id]})
                else:
                    operations.append({"op": "TEMPORAL_FILTER", "query": search_query, "relation": relation, "axis": axis, "fallback_axis": "", "anchor": anchor, "end": end, "produces": [slot_id]})
                continue
            operations.append({"op": "SEMANTIC_SEARCH", "query": search_query, "top_k": 8, "strategy": "FOCAL", "produces": [slot_id]})
        return operations[:max_ops]

    def _make_deterministic_recovery_plan(self, missing_slots, question, existing_plan):
        if not missing_slots:
            return None
        budget = existing_plan.get("budget_tier", "MEDIUM")
        shell = {"query_mode": existing_plan.get("query_mode", "DIRECT"), "visible_options": existing_plan.get("visible_options", {})}
        operations = self._compile_gap_operations(missing_slots, question, budget, plan=shell)
        return {"query_spec": existing_plan.get("query_spec", {}), "query_mode": shell["query_mode"], "required_slots": missing_slots, "seed_coverage": [], "operations": operations, "option_coverage": [], "visible_options": shell["visible_options"], "need_evidence": False, "budget_tier": budget, "max_memories": RETRIEVAL_BUDGETS.get(budget, RETRIEVAL_BUDGETS["MEDIUM"])["max_memories"], "planner_fallback": True, "fallback_reason": "deterministic_missing_evidence_recovery", "valid": True}

    def _slot_covered(self, slot, support_ids, selected, relations):
        support_set = set(support_ids)
        memories = [memory for memory in selected if memory.get("id") in support_set]
        if not memories:
            return False
        slot_type = str(slot.get("type") or "DIRECT").upper()
        role = str(slot.get("evidence_role") or "").upper()
        if slot_type == "DIRECT":
            valid = [memory for memory in memories if self._memory_value(memory) and memory.get("assertion_mode", "DIRECT") in {"DIRECT", "RECAP"} and self._memory_matches_slot_role(slot, memory) and (slot.get("history") or memory.get("_status", self._belief_status.get(memory.get("id"), "active")) != "superseded")]
            return len({memory["id"] for memory in valid}) >= (2 if role == "PRIOR_TRAJECTORY" else 1)
        if slot_type == "CURRENT_STATE":
            heads = [memory for memory in memories if self._is_state_head(memory) and self._memory_value(memory) and self._slot_contract_match(slot, memory, True)]
            identities = {state_identity(memory) for memory in heads if state_identity(memory)}
            return bool(heads and len(identities) == 1)
        if slot_type == "TEMPORAL":
            axis = str(slot.get("time_axis") or "").lower()
            relation = str(slot.get("temporal_relation") or slot.get("time_relation") or "LOCATE").upper()
            anchor, end = self._parse_date(str(slot.get("time_anchor") or "")), self._parse_date(str(slot.get("time_end") or ""))
            if axis not in VALID_TEMPORAL_AXES:
                return False
            def good(memory):
                date = self._date_for(memory, axis)
                if not date or not self._slot_contract_match(slot, memory, True):
                    return False
                if relation == "LOCATE": return True
                if relation == "EXACT": return bool(anchor and (date == anchor or date.startswith(anchor)))
                if relation == "BEFORE": return bool(anchor and date < anchor)
                if relation == "AFTER": return bool(anchor and date > anchor)
                if relation == "BETWEEN": return bool(anchor and end and anchor <= date <= end)
                return relation in {"EARLIEST", "LATEST"}
            return any(good(memory) for memory in memories)
        if slot_type == "CAUSE_PATH":
            by_id, adjacency = {memory["id"]: memory for memory in memories}, defaultdict(list)
            for relation in relations:
                if self._valid_causal_relation(relation, by_id): adjacency[relation["source_id"]].append(relation["target_id"])
            if len(by_id) < 2 or not any(adjacency.values()): return False
            for root in by_id:
                seen, queue = {root}, [root]
                while queue:
                    for target in adjacency.get(queue.pop(0), []):
                        if target not in seen: seen.add(target); queue.append(target)
                if set(by_id).issubset(seen): return True
            return False
        if slot_type == "TRANSITION":
            by_id = {memory["id"]: memory for memory in memories}
            return any(relation.get("type") in {"SUPERSEDE", "REFINE"} and relation.get("source_id") in by_id and relation.get("target_id") in by_id for relation in relations)
        return False

    def _coverage_map(self, plan, slot_support, selected, relations):
        coverage = {slot["id"]: self._slot_covered(slot, slot_support.get(slot["id"], []), selected, relations) for slot in plan.get("required_slots", [])}
        mode = str(plan.get("query_mode") or "").upper()
        required = [slot for slot in plan.get("required_slots", []) if slot.get("required") is not False]
        if coverage and all(coverage.values()) and mode in {"COMPARISON", "MULTI_HOP", "CAUSAL"}:
            sets = [set(slot_support.get(slot["id"], [])) for slot in required]
            if len(set().union(*sets)) < 2:
                coverage = {key: False for key in coverage}
            if mode == "COMPARISON":
                comparands = [slot for slot in required if slot.get("evidence_role") == "COMPARAND"]
                sides = {slot.get("side_label") for slot in comparands}
                if len(comparands) != 2 or sides != {"LEFT", "RIGHT"}:
                    coverage = {key: False for key in coverage}
                elif len(sets) >= 2 and sets[0] == sets[1]:
                    coverage = {key: False for key in coverage}
        lattice = getattr(self, "_evidence_lattice", None)
        if lattice is not None:
            for slot in plan.get("required_slots", []):
                slot_id = slot["id"]
                lattice.update_gap(slot_id, "FILLED" if coverage.get(slot_id) else "MISSING", slot_support.get(slot_id, []))
        return coverage

    def _operation_slot_support(self, slot, result, relations):
        if not result:
            return []
        role, slot_type = str(slot.get("evidence_role") or "").upper(), str(slot.get("type") or "DIRECT").upper()
        query = self._rc_search_query(slot, "") or str(slot.get("id") or "")
        ranked = self._hybrid_search(query, top_k=len(result), candidate_ids={memory["id"] for memory in result})
        by_id = {memory["id"]: memory for memory in result}
        ranked = [by_id[memory["id"]] for memory in ranked if memory["id"] in by_id]
        if role == "OPTION_CONTEXT": return [memory for memory in ranked if self._rc_owner_match(slot, memory)][:6]
        if slot_type == "DIRECT": return [memory for memory in ranked if self._memory_value(memory) and memory.get("assertion_mode", "DIRECT") in {"DIRECT", "RECAP"} and self._memory_matches_slot_role(slot, memory)][:4]
        if slot_type == "TEMPORAL": return [memory for memory in ranked if self._date_for(memory, str(slot.get("time_axis") or "event_time")) and self._slot_contract_match(slot, memory, True)][:4]
        if slot_type == "CURRENT_STATE": return [memory for memory in ranked if self._is_state_head(memory) and self._slot_contract_match(slot, memory, True)][:2]
        if slot_type == "CAUSE_PATH":
            endpoints = set()
            for relation in relations:
                if self._valid_causal_relation(relation, by_id): endpoints.update((relation["source_id"], relation["target_id"]))
            return [memory for memory in ranked if memory["id"] in endpoints]
        return ranked[:4]

    def _role_aware_support_ids(self, slots, slot_support, candidate_order, limit):
        output, allowed = [], set(candidate_order)
        def add(memory_id):
            if memory_id in allowed and memory_id not in output and len(output) < limit: output.append(memory_id)
        for slot in slots:
            ids, role = slot_support.get(slot.get("id"), []), str(slot.get("evidence_role") or "").upper()
            reserve = 6 if role == "OPTION_CONTEXT" else (3 if role in {"PRIOR_TRAJECTORY", "FOCAL_TRIGGER", "OUTCOME"} else 1)
            for memory_id in ids[:reserve]: add(memory_id)
        supported = {memory_id for ids in slot_support.values() for memory_id in ids}
        for memory_id in candidate_order:
            if memory_id in supported: add(memory_id)
        return output

    @staticmethod
    def _multiple_choice_answer_instruction(option_labels):
        labels = ", ".join(sorted(str(label) for label in option_labels))
        return f" This is multiple-choice with options {labels}. Evaluate EVERY visible option against the same predicate using the shared participant-specific evidence. A missing personal-memory hit for an option is neither support nor refutation. Use general domain knowledge only to interpret grounded participant facts. Output ONLY all matching uppercase labels separated by commas, or NONE."

    def prepare_batch_query(self, question, system_message=None, **kwargs):
        prepared = super().prepare_batch_query(question, system_message=system_message, **kwargs)
        duplicate = " Before selecting labels, restate the stem's predicate internally: if it asks which option is safe, recommended, or okay, exclude options contradicted by an allergy or contraindication; if it asks which is unsafe or contraindicated, select those contradicted options. Never output the contraindicated labels for a safe/okay stem just because those labels appear in the memory."
        for message in prepared.get("messages", []):
            if isinstance(message, dict) and isinstance(message.get("content"), str): message["content"] = message["content"].replace(duplicate, "")
        prepared.setdefault("extra", {})["read_contract_version"] = "P1B.2.2"
        prepared["extra"]["option_probe_coverage"] = dict(getattr(self, "_last_option_probe_coverage", {}) or {})
        if system_message and prepared.get("precomputed_answer"):
            prepared["precomputed_answer"] = ""
        return prepared
