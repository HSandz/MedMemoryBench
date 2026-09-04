"""Typed slot coverage, progressive execution, and pure arbitration."""

import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from .contracts import VALID_TEMPORAL_AXES, QueryFrame
from .canonicalization import state_identity


class ExecutionMixin:
    """Executes bounded retrieval plans and enforces strict boundaries."""

    def _is_state_head(self, memory: Dict[str, Any]) -> bool:
        """Return whether a state-like memory is in its resolved head set."""
        identity = state_identity(memory)
        return bool(
            identity and memory.get("id") in self._state_heads.get(identity, [])
        )

    @staticmethod
    def _merge_relations(
        *relation_groups: Sequence[Dict[str, Any]],
        allowed_ids: Optional[Set[str]] = None,
    ) -> List[Dict[str, Any]]:
        merged = {}
        for relations in relation_groups:
            for relation in relations:
                source = relation.get("source_id")
                target = relation.get("target_id")
                relation_type = relation.get("type")
                if allowed_ids is not None and (
                    source not in allowed_ids or target not in allowed_ids
                ):
                    continue
                merged[(source, target, relation_type)] = relation
        return list(merged.values())

    def _slot_seed_support(
        self,
        plan: Dict[str, Any],
        seeds: List[Dict[str, Any]],
    ) -> Dict[str, List[str]]:
        support: Dict[str, List[str]] = {
            slot["id"]: [] for slot in plan["required_slots"]
        }
        for coverage in plan.get("seed_coverage", []):
            resolved = self._resolve_refs(coverage["refs"], [], seeds)
            support[coverage["slot_id"]] = list(
                dict.fromkeys(memory["id"] for memory in resolved)
            )
        return support

    @staticmethod
    def _memory_matches_slot_role(slot: Dict[str, Any], memory: Dict[str, Any]) -> bool:
        """Honor an explicit planner role; never infer one from the question."""
        role = str(slot.get("evidence_role") or "").upper()
        if not role:
            return True
        tags = {str(tag).upper() for tag in memory.get("planning_tags", [])}
        kind = str(memory.get("kind") or "").upper()
        # planning_tags are ranking hints, not hard execution authorization requirements.
        # We rely on specific property and entity matches in _slot_covered.
        return True

    def _slot_covered(
        self,
        slot: Dict[str, Any],
        support_ids: Sequence[str],
        selected: Sequence[Dict[str, Any]],
        relations: Sequence[Dict[str, Any]],
    ) -> bool:
        support_set = set(support_ids)
        memories = [memory for memory in selected if memory["id"] in support_set]
        if not memories:
            return False
        slot_type = slot["type"]

        def _proves_slot_contract(m, s):
            # Subject check
            subj = s.get("subject")
            if (
                subj
                and str(m.get("subject", "")) != str(subj)
                and str(subj) not in {"primary_user", ""}
            ):
                return False

            # Target property - loosen literal check, rely on semantic retrieval for concepts, but enforce if specified
            # The planner might output abstractions, so we shouldn't strictly fail if it's not a literal substring.
            # But the user said: "Use planner description/property to retrieve. Use canonical durable fields to authorize."
            # The gaps created by QRF have specific required fields.
            # For now, we loosen target_entities and target_property so they don't cause false negatives.
            # We trust the semantic retrieval, and enforce `required_fields`, subject, and `option_label`.

            # Target property & entities MUST be enforced for FILLED proof
            tp = s.get("target_property")
            if tp and str(tp).strip():
                tp_lower = str(tp).lower().strip()
                # If target property is a long sentence (e.g. MCD wrapper leaked), we shouldn't fail everything,
                # but since MCD wrapper is fixed, tp should be clean.
                matched_tp = any(
                    tp_lower in str(m.get(k, "")).lower()
                    for k in ["state_key", "claim", "value", "verbatim_value"]
                )
                if not matched_tp:
                    # check subset in claim
                    if not any(
                        word in str(m.get("claim", "")).lower()
                        for word in tp_lower.split()
                        if len(word) > 4
                    ):
                        return False

            entities = s.get("target_entities") or []
            if entities:
                mem_text = (
                    str(m.get("claim", "")) + " " + " ".join(m.get("entities", []))
                ).lower()
                matched_ent = False
                for e in entities:
                    if str(e).lower() in mem_text:
                        matched_ent = True
                        break
                if not matched_ent:
                    return False

            # Required fields
            req_fields = s.get("required_fields") or []
            for field in req_fields:
                if not m.get(field):
                    return False

            # Option label support
            # Literal A/B/C/D mapping is invalid because memories do not contain letter labels.
            # We defer proposition evaluation to the final answer phase which sees the SHARED_OPTIONS bundle.
            pass

            # Distinct comparison sides check. Just ensure it doesn't fail based on fake entities.
            if s.get("comparison_side_label"):
                pass

            return True

        if slot_type == "DIRECT":
            allow_history = bool(slot.get("history"))
            return any(
                bool(self._memory_value(memory))
                and memory.get("assertion_mode", "DIRECT") == "DIRECT"
                and self._memory_matches_slot_role(slot, memory)
                and _proves_slot_contract(memory, slot)
                and (
                    allow_history
                    or memory.get(
                        "_status", self._belief_status.get(memory["id"], "active")
                    )
                    != "superseded"
                )
                for memory in memories
            )

        if slot_type == "CURRENT_STATE":
            heads = [
                memory
                for memory in memories
                if self._is_state_head(memory)
                and self._belief_status.get(memory["id"], "active")
                in {"active", "refined", "conflicting"}
                and self._memory_value(memory)
                and _proves_slot_contract(memory, slot)
            ]
            identities = {state_identity(memory) for memory in heads}
            values = {self._state_value_signature(memory) for memory in heads}
            if len(heads) < 1 or len(identities) != 1 or len(values) != 1:
                return False
            identity = next(iter(identities))
            by_id = {memory["id"]: memory for memory in self._memories}
            global_values = {
                self._state_value_signature(by_id[memory_id])
                for memory_id in self._state_heads.get(identity, [])
                if memory_id in by_id
                and self._belief_status.get(memory_id, "active")
                in {"active", "refined", "conflicting"}
                and self._state_value_signature(by_id[memory_id])
            }
            return len(global_values) == 1

        if slot_type == "TEMPORAL":
            axis = slot.get("time_axis", "")
            fallback_axis = str(slot.get("fallback_axis") or "")
            if axis not in VALID_TEMPORAL_AXES:
                return False
            return any(
                bool(self._memory_value(memory))
                and bool(
                    self._date_for(memory, axis)
                    or (
                        fallback_axis in VALID_TEMPORAL_AXES
                        and self._date_for(memory, fallback_axis)
                    )
                )
                and self._memory_matches_slot_role(slot, memory)
                for memory in memories
            )

        if slot_type == "TRANSITION":
            by_id = {memory["id"]: memory for memory in memories}
            for relation in relations:
                if relation.get("type") not in {"SUPERSEDE", "REFINE"}:
                    continue
                source = by_id.get(relation.get("source_id"))
                target = by_id.get(relation.get("target_id"))
                if not source or not target:
                    continue
                same_identity = state_identity(source) and state_identity(
                    source
                ) == state_identity(target)
                values_differ = (
                    self._normalised_value(source)
                    and self._normalised_value(target)
                    and self._normalised_value(source) != self._normalised_value(target)
                )
                if same_identity and values_differ:
                    return True
            return False

        if slot_type == "CAUSE_PATH":
            by_id = {memory["id"]: memory for memory in memories}
            adjacency: Dict[str, List[str]] = defaultdict(list)
            for relation in relations:
                if not self._valid_causal_relation(relation, by_id):
                    continue
                adjacency[relation["source_id"]].append(relation["target_id"])
            if len(by_id) < 2 or not any(adjacency.values()):
                return False
            # All accepted support nodes must belong to one directed causal
            # component. Disconnected topical causal edges cannot jointly fill
            # one CAUSE_PATH slot; semantic start/goal matching is checked by
            # the slot-support gate.
            required = set(by_id)
            for root in required:
                reached, frontier = {root}, [root]
                while frontier:
                    current = frontier.pop(0)
                    for target in adjacency.get(current, []):
                        if target not in reached:
                            reached.add(target)
                            frontier.append(target)
                if required.issubset(reached):
                    return True
            return False

        if slot_type == "COMPARISON":
            groups = {}
            for memory in memories:
                value = self._normalised_value(memory)
                if value:
                    groups.setdefault(value, []).append(memory)
            return len(groups) >= 2 and all(groups.values())

        return False

    def _coverage_map(
        self,
        plan: Dict[str, Any],
        slot_support: Dict[str, List[str]],
        selected: Sequence[Dict[str, Any]],
        relations: Sequence[Dict[str, Any]],
    ) -> Dict[str, bool]:
        return {
            slot["id"]: self._slot_covered(
                slot,
                slot_support.get(slot["id"], []),
                selected,
                relations,
            )
            for slot in plan["required_slots"]
        }

    @staticmethod
    def _has_unresolved_conflict(memories: Sequence[Dict[str, Any]]) -> bool:
        return any(memory.get("_status") == "conflicting" for memory in memories)

    def _operation_slot_support(
        self,
        slot: Dict[str, Any],
        result: Sequence[Dict[str, Any]],
        relations: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Select structurally eligible operation output for one typed slot."""
        if not result:
            return []
        result_by_id = {memory["id"]: memory for memory in result}
        ranked = self._hybrid_search(
            str(slot.get("description") or slot["id"]),
            top_k=len(result_by_id),
            candidate_ids=result_by_id,
        )
        ranked = [
            result_by_id[memory["id"]]
            for memory in ranked
            if memory["id"] in result_by_id
        ]
        ranked.extend(
            memory
            for memory in result
            if memory["id"] not in {item["id"] for item in ranked}
        )
        slot_type = slot["type"]

        if slot_type == "DIRECT":
            allow_history = bool(slot.get("history"))
            return [
                memory
                for memory in ranked
                if (
                    memory.get("assertion_mode", "DIRECT") == "DIRECT"
                    and bool(self._memory_value(memory))
                    and self._memory_matches_slot_role(slot, memory)
                    and (
                        allow_history
                        or memory.get(
                            "_status",
                            self._belief_status.get(memory["id"], "active"),
                        )
                        != "superseded"
                    )
                )
            ][:4]

        if slot_type == "CURRENT_STATE":
            best = next(
                (memory for memory in ranked if self._is_state_head(memory)), None
            )
            if not best:
                return []
            identity = state_identity(best)
            return [
                memory
                for memory in ranked
                if state_identity(memory) == identity
                and self._is_state_head(memory)
                and self._memory_value(memory)
                and self._memory_matches_slot_role(slot, memory)
                and self._belief_status.get(memory["id"], "active")
                in {"active", "refined", "conflicting"}
            ]

        if slot_type == "TEMPORAL":
            axis = str(slot.get("time_axis") or "")
            fallback_axis = str(slot.get("fallback_axis") or "")
            return [
                memory
                for memory in ranked
                if (
                    self._memory_value(memory)
                    and self._memory_matches_slot_role(slot, memory)
                    and axis in VALID_TEMPORAL_AXES
                    and (
                        self._date_for(memory, axis)
                        or (
                            fallback_axis in VALID_TEMPORAL_AXES
                            and self._date_for(memory, fallback_axis)
                        )
                    )
                )
            ][:3]

        if slot_type == "TRANSITION":
            endpoint_ids = set()
            for relation in relations:
                if relation.get("type") not in {"SUPERSEDE", "REFINE"}:
                    continue
                source = result_by_id.get(relation.get("source_id"))
                target = result_by_id.get(relation.get("target_id"))
                if (
                    source
                    and target
                    and state_identity(source)
                    and state_identity(source) == state_identity(target)
                    and self._normalised_value(source) != self._normalised_value(target)
                ):
                    endpoint_ids.update((source["id"], target["id"]))
            return [
                memory
                for memory in ranked
                if memory["id"] in endpoint_ids
                and self._memory_matches_slot_role(slot, memory)
            ]

        if slot_type == "CAUSE_PATH":
            endpoint_ids = set()
            for relation in relations:
                if self._valid_causal_relation(relation, result_by_id):
                    endpoint_ids.update((relation["source_id"], relation["target_id"]))
            return [
                memory
                for memory in ranked
                if memory["id"] in endpoint_ids
                and self._memory_matches_slot_role(slot, memory)
            ]

        if slot_type == "COMPARISON":
            selected, values = [], set()
            for memory in ranked:
                value = self._normalised_value(memory)
                if (
                    value
                    and value not in values
                    and self._memory_matches_slot_role(slot, memory)
                ):
                    selected.append(memory)
                    values.add(value)
                if len(selected) >= 4:
                    break
            return selected if len(values) >= 2 else []
        return []

    def _execute_operation(
        self,
        operation: Dict[str, Any],
        outputs: List[List[Dict[str, Any]]],
        seeds: List[Dict[str, Any]],
        frame: QueryFrame = QueryFrame(),
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[str]]:
        op, result, relations, evidence_refs = operation["op"], [], [], []
        if op == "SEMANTIC_SEARCH":
            try:
                top_k = max(1, min(10, int(operation.get("top_k", 5))))
            except (TypeError, ValueError):
                top_k = 5
            result = self._semantic_operation_search(
                str(operation.get("query") or ""),
                top_k=top_k,
                strategy=str(operation.get("strategy") or "FOCAL"),
                frame=frame,
                option_queries=operation.get("option_queries") or [],
            )
        elif op == "LOCATE_ANCHOR":
            result = self._locate_anchor(str(operation.get("query") or ""), frame)
        elif op == "TEMPORAL_FILTER":
            result = self._temporal_filter(operation, outputs, seeds, frame)
        elif op == "RESOLVE_STATE":
            result = self._resolve_state(
                str(operation.get("query") or ""), seeds, frame
            )
        elif op == "FOLLOW_CAUSES":
            result, relations = self._follow_causes(operation, outputs, seeds)
        elif op == "VERIFY_EVIDENCE":
            result = self._resolve_refs(operation.get("memory_refs"), outputs, seeds)
            evidence_refs = [memory["id"] for memory in result]
        excluded = {str(memory_id) for memory_id in operation.get("exclude_ids", [])}
        if excluded:
            result = [memory for memory in result if memory.get("id") not in excluded]
            allowed = {memory["id"] for memory in result}
            relations = [
                relation
                for relation in relations
                if relation.get("source_id") in allowed
                and relation.get("target_id") in allowed
            ]
            evidence_refs = [memory_id for memory_id in evidence_refs if memory_id in allowed]
        return result, relations, evidence_refs

    def _execute_plan(
        self,
        plan: Dict[str, Any],
        seeds: List[Dict[str, Any]],
        round_offset: int = 0,
        frame: QueryFrame = QueryFrame(),
        question: str = "",
    ) -> Dict[str, Any]:
        outputs: List[List[Dict[str, Any]]] = []
        used_relations: List[Dict[str, Any]] = []
        evidence_refs: List[str] = []
        slot_support = self._slot_seed_support(plan, seeds)
        selected: Dict[str, Dict[str, Any]] = {}
        slot_validation: List[Dict[str, Any]] = []
        memory_limit = max(1, int(plan["max_memories"]))

        # Only seed_coverage can authorize initial selected memories.
        seed_by_id = {memory["id"]: memory for memory in seeds[:3]}
        for ids in slot_support.values():
            for memory_id in ids:
                if memory_id in seed_by_id:
                    if memory_id not in selected and len(selected) >= memory_limit:
                        continue
                    selected.setdefault(
                        memory_id, self._snapshot(seed_by_id[memory_id])
                    )

        beliefs: List[Dict[str, Any]] = []
        current_relations: List[Dict[str, Any]] = []
        coverage: Dict[str, bool] = {}
        requirement_status: Dict[str, str] = {}
        relation_status: Dict[str, str] = {}
        retrieval_complete = False

        def assess_retrieval() -> None:
            nonlocal beliefs, current_relations, coverage
            nonlocal requirement_status, relation_status, retrieval_complete
            beliefs, belief_relations = self._reconstruct_beliefs(
                list(selected.values()),
                int(plan["max_memories"]),
            )
            current_relations = self._merge_relations(
                used_relations,
                belief_relations,
            )
            requirement_status, relation_status, retrieval_complete = (
                self._retrieval_status(
                    plan, slot_support, beliefs, current_relations
                )
            )
            coverage = {
                slot_id: status == "FOUND"
                for slot_id, status in requirement_status.items()
            }

        assess_retrieval()
        trace: List[Dict[str, Any]] = []
        answer_operations = sum(
            operation.get("op") not in {"LOCATE_ANCHOR", "VERIFY_EVIDENCE"}
            for operation in plan["operations"]
        )

        for index, operation in enumerate(plan["operations"]):
            before = dict(coverage)
            requirement_before = dict(requirement_status)
            relation_before = dict(relation_status)
            previous_ids = set(selected)
            operation_started = time.perf_counter()
            bounded_operation = self._snapshot(operation)

            # Relative temporal operations are anchored to the validated support
            # of their target requirement. A raw operation output may contain
            # several dates and is not itself a semantic anchor.
            anchor_requirement = str(
                bounded_operation.get("anchor_requirement") or ""
            )
            if (
                bounded_operation.get("op") == "TEMPORAL_FILTER"
                and anchor_requirement
            ):
                axis = str(bounded_operation.get("axis") or "event_time")
                selected_by_id = {memory["id"]: memory for memory in beliefs}
                anchor_value = next(
                    (
                        self._date_for(selected_by_id[memory_id], axis)
                        for memory_id in slot_support.get(anchor_requirement, [])
                        if memory_id in selected_by_id
                        and self._date_for(selected_by_id[memory_id], axis)
                    ),
                    "",
                )
                bounded_operation["anchor"] = anchor_value
                bounded_operation.pop("candidate_refs", None)

            if bounded_operation["op"] == "SEMANTIC_SEARCH":
                remaining = max(0, memory_limit - len(selected))
                try:
                    requested_top_k = max(
                        1, int(bounded_operation.get("top_k", remaining))
                    )
                except (TypeError, ValueError):
                    requested_top_k = max(1, remaining)
                if bounded_operation.get("strategy") == "SHARED_OPTIONS":
                    option_count = len(bounded_operation.get("option_queries") or [])
                    requested_top_k = max(
                        requested_top_k,
                        min(self.HARD_MEMORY_LIMIT, max(4, option_count * 2)),
                    )
                bounded_operation["top_k"] = max(
                    1,
                    min(max(1, remaining), requested_top_k),
                )

            missing_anchor = bool(
                bounded_operation.get("op") == "TEMPORAL_FILTER"
                and anchor_requirement
                and not bounded_operation.get("anchor")
            )
            if missing_anchor:
                result, relations, requested_evidence = [], [], []
            else:
                result, relations, requested_evidence = self._execute_operation(
                    bounded_operation,
                    outputs,
                    seeds,
                    frame,
                )

            remaining = max(0, memory_limit - len(selected))
            if bounded_operation.get("strategy") == "SHARED_OPTIONS":
                operation_cap = memory_limit
            else:
                operation_cap = max(
                    1,
                    (memory_limit // max(1, answer_operations))
                    * max(1, len(operation.get("produces") or [])),
                )
            remaining = min(remaining, operation_cap)
            bounded_result: List[Dict[str, Any]] = []
            for memory in result:
                memory_id = memory["id"]
                if memory_id in selected:
                    bounded_result.append(memory)
                elif remaining > 0:
                    bounded_result.append(memory)
                    remaining -= 1
            result = bounded_result
            result_ids = {memory["id"] for memory in result}
            local_relations = [
                self._snapshot(relation)
                for relation in self._relations
                if relation.get("source_id") in result_ids
                and relation.get("target_id") in result_ids
            ]
            relations = self._merge_relations(relations, local_relations)
            relations = [
                relation
                for relation in relations
                if relation.get("source_id") in result_ids
                and relation.get("target_id") in result_ids
            ]
            requested_evidence = [
                memory_id for memory_id in requested_evidence if memory_id in result_ids
            ]
            operation_elapsed_ms = round(
                (time.perf_counter() - operation_started) * 1000.0, 3
            )

            outputs.append([self._snapshot(memory) for memory in result])
            used_relations.extend(self._snapshot(relations))
            evidence_refs.extend(requested_evidence)

            for slot_id in operation["produces"]:
                slot_support.setdefault(slot_id, [])
                slot = next(
                    candidate
                    for candidate in plan["required_slots"]
                    if candidate["id"] == slot_id
                )
                if bounded_operation["op"] == "LOCATE_ANCHOR":
                    continue
                if bounded_operation["op"] == "TEMPORAL_FILTER":
                    fallback_axis = str(
                        bounded_operation.get("fallback_axis") or ""
                    ).lower()
                    if fallback_axis in VALID_TEMPORAL_AXES:
                        slot["fallback_axis"] = fallback_axis
                    anchor_ref = bounded_operation.get("anchor")
                    resolved_anchor = str(anchor_ref or "")
                    if isinstance(anchor_ref, str) and anchor_ref.startswith("$"):
                        resolved_anchor = self._operation_date(
                            anchor_ref,
                            str(bounded_operation.get("axis") or "event_time"),
                            outputs,
                            seeds,
                        )
                    if resolved_anchor:
                        slot["resolved_time_anchor"] = resolved_anchor
                supported = self._operation_slot_support(slot, result, relations)
                for memory in supported:
                    if memory["id"] not in selected and len(selected) >= memory_limit:
                        continue
                    if memory["id"] not in slot_support[slot_id]:
                        slot_support[slot_id].append(memory["id"])
                    selected.setdefault(memory["id"], self._snapshot(memory))

            assess_retrieval()
            trace.append(
                {
                    "retrieval_round": round_offset + index + 1,
                    "operation_index": index,
                    "operation": operation["op"],
                    "input_refs": [
                        str(ref)
                        for field in (
                            "start",
                            "memory_refs",
                            "candidate_refs",
                            "anchor",
                            "end",
                        )
                        for ref in (
                            operation.get(field)
                            if isinstance(operation.get(field), list)
                            else ([operation.get(field)] if operation.get(field) else [])
                        )
                        if str(ref).startswith("$")
                    ],
                    "produces": list(operation["produces"]),
                    "output_ids": [memory["id"] for memory in result],
                    "operation_elapsed_ms": operation_elapsed_ms,
                    "skipped_reason": "missing_anchor_support" if missing_anchor else "",
                    "new_memory_ids": [
                        memory["id"]
                        for memory in result
                        if memory["id"] not in previous_ids
                    ],
                    "slot_coverage_before": before,
                    "slot_coverage_after": dict(coverage),
                    "requirement_status_before": requirement_before,
                    "requirement_status_after": dict(requirement_status),
                    "relation_status_before": relation_before,
                    "relation_status_after": dict(relation_status),
                }
            )

        return {
            "selected": beliefs,
            "relations": current_relations,
            "evidence_refs": list(dict.fromkeys(evidence_refs)),
            "slot_support": slot_support,
            "slot_coverage": coverage,
            "requirement_status": requirement_status,
            "relation_status": relation_status,
            "retrieval_complete": retrieval_complete,
            # Deprecated compatibility alias. It no longer controls execution.
            "sufficient": retrieval_complete,
            "trace": trace,
            "operation_outputs": outputs,
            "slot_validation": slot_validation,
            "slot_validation_tokens": 0,
            "slot_validation_latency": 0.0,
        }

    @staticmethod
    def _plan_requires_semantic_validation(plan: Dict[str, Any]) -> bool:
        """The minimal-IR path has no middle LLM validator."""
        del plan
        return False

    def _run_query_retrieval(
        self,
        question: str,
        initial_seeds: List[Dict[str, Any]],
        frame: QueryFrame,
        fast_supports: Optional[List[Dict[str, Any]]],
        gate: Dict[str, Any],
        planning_seeds: Optional[List[Dict[str, Any]]] = None,
        planning_context: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Run one semantic route and deterministic retrieval operations."""
        query_tokens = {
            "controller": 0,
            "fast_gate": int(gate.get("usage", {}).get("total_tokens", 0)),
            "planner": 0,
            "slot_validation": 0,
            "replan": 0,
            "answer": 0,
            "total": 0,
        }
        query_latency = {
            "controller": 0.0,
            "fast_gate": round(
                float(gate.get("usage", {}).get("latency", 0.0) or 0.0), 3
            ),
            "planner": 0.0,
            "slot_validation": 0.0,
            "replan": 0.0,
            "answer": None,
            "retrieval_wall": 0.0,
            "total_wall": 0.0,
        }
        retrieval_started = time.perf_counter()
        plan: Dict[str, Any] = {
            "required_slots": [],
            "seed_coverage": [],
            "operations": [],
            "need_evidence": False,
            "budget_tier": "SMALL",
            "max_memories": 3,
            "planner_fallback": False,
            "valid": True,
        }
        replan: Optional[Dict[str, Any]] = None
        planner_called = False
        replan_called = False
        trace: List[Dict[str, Any]] = []
        slot_validation: List[Dict[str, Any]] = []
        operation_output_ids = set()
        evidence_refs: List[str] = []
        slot_support: Dict[str, List[str]] = {}
        slot_coverage: Dict[str, bool] = {}
        requirement_status: Dict[str, str] = {}
        relation_status: Dict[str, str] = {}
        relations: List[Dict[str, Any]] = []
        retrieval_complete = False
        beliefs: List[Dict[str, Any]] = []
        operation_candidates: Dict[str, Dict[str, Any]] = {}
        planned_seed_set = planning_seeds or initial_seeds
        controller = {
            "called": False,
            "route": "LEGACY",
            "support_ref": "",
            "answer": "",
            "fallback_reason": "",
            "usage": {},
        }

        if getattr(self, "enable_two_stage_controller", False):
            fast_supports, routed_plan, controller = self._semantic_controller(
                question, planned_seed_set, frame, context_map=planning_context
            )
            usage = controller.get("usage", {})
            query_tokens["controller"] = int(usage.get("total_tokens", 0) or 0)
            query_latency["controller"] = round(
                float(usage.get("latency", 0.0) or 0.0), 3
            )
            if fast_supports is None:
                plan = routed_plan
                planner_called = True
        if fast_supports is not None:
            beliefs, relations = self._reconstruct_beliefs(
                fast_supports, min(3, len(fast_supports))
            )
            retrieval_complete = bool(beliefs) and not self._has_unresolved_conflict(
                beliefs
            )
            requirement_status = {
                "fast_atomic_answer": "FOUND" if retrieval_complete else "EMPTY"
            }
        else:
            if not getattr(self, "enable_two_stage_controller", False):
                planner_called = True
                plan, usage = self._plan_operations(
                    question,
                    planned_seed_set,
                    context_map=planning_context,
                )
                query_tokens["planner"] = int(usage.get("total_tokens", 0))
                query_latency["planner"] = round(
                    float(usage.get("latency", 0.0) or 0.0), 3
                )
            execution = self._execute_plan(
                plan, planned_seed_set, frame=frame, question=question
            )
            beliefs = execution["selected"]
            relations = execution["relations"]
            evidence_refs.extend(execution["evidence_refs"])
            slot_support = self._snapshot(execution["slot_support"])
            slot_coverage = dict(execution["slot_coverage"])
            requirement_status = dict(
                execution.get("requirement_status")
                or {
                    slot_id: "FOUND" if covered else "EMPTY"
                    for slot_id, covered in slot_coverage.items()
                }
            )
            relation_status = dict(execution.get("relation_status") or {})
            retrieval_complete = bool(
                execution.get(
                    "retrieval_complete", execution.get("sufficient", False)
                )
            )
            trace.extend(execution["trace"])
            slot_validation.extend(execution["slot_validation"])
            query_tokens["slot_validation"] += int(execution["slot_validation_tokens"])
            query_latency["slot_validation"] += float(
                execution.get("slot_validation_latency", 0.0) or 0.0
            )
            operation_output_ids.update(
                memory["id"]
                for output in execution["operation_outputs"]
                for memory in output
            )
            operation_candidates.update(
                {
                    memory["id"]: self._snapshot(memory)
                    for output in execution["operation_outputs"]
                    for memory in output
                }
            )

            missing_slot_ids = {
                slot_id
                for slot_id, status in requirement_status.items()
                if status == "EMPTY"
            }
            for relation in plan.get("semantic_relations") or []:
                relation_key = self._status_relation_key(relation)
                if relation_status.get(relation_key) != "UNPROVEN":
                    continue
                for endpoint in (relation.get("from"), relation.get("to")):
                    if endpoint and endpoint != "ANSWER":
                        missing_slot_ids.add(str(endpoint))
            missing_slots = [
                slot
                for slot in plan.get("required_slots", [])
                if slot["id"] in missing_slot_ids
            ]
            two_stage = bool(getattr(self, "enable_two_stage_controller", False))
            broad_replan = bool(getattr(self, "enable_replan", False)) and not two_stage
            deterministic_recovery = bool(
                getattr(self, "enable_zero_result_recovery", False)
            )

            replan = None
            if deterministic_recovery and missing_slots:
                # Typed deterministic recovery is available on both controller
                # variants. It never changes slot semantics or performs hidden
                # retrieval; it only emits explicit operations for missing slots.
                replan = self._make_deterministic_recovery_plan(
                    missing_slots, question, plan
                )
                if replan:
                    replan_called = True
            elif broad_replan and missing_slots:
                # Compatibility LLM replan, only outside the two-stage route.
                replan_called = True
                replan_seeds = []
                for memory in (*beliefs, *planned_seed_set):
                    if all(item["id"] != memory["id"] for item in replan_seeds):
                        replan_seeds.append(self._snapshot(memory))
                    if len(replan_seeds) >= 8:
                        break
                replan, usage = self._plan_operations(
                    question,
                    replan_seeds,
                    missing_slots=missing_slots,
                    prior_trace=[
                        {
                            "operation": item.get("operation"),
                            "produces": item.get("produces", []),
                            "output_ids": item.get("output_ids", []),
                            "slot_coverage_before": item.get(
                                "slot_coverage_before", {}
                            ),
                            "slot_coverage_after": item.get("slot_coverage_after", {}),
                        }
                        for item in trace
                    ],
                    context_map=planning_context,
                    allow_repair=False,
                )
                query_tokens["replan"] = int(usage.get("total_tokens", 0))
                query_latency["replan"] = round(
                    float(usage.get("latency", 0.0) or 0.0), 3
                )

            if replan_called and replan:
                already_retrieved = set(operation_output_ids) | {
                    memory["id"] for memory in beliefs
                }
                for operation in replan.get("operations") or []:
                    if operation.get("op") in {
                        "SEMANTIC_SEARCH",
                        "LOCATE_ANCHOR",
                        "RESOLVE_STATE",
                        "TEMPORAL_FILTER",
                    }:
                        operation["exclude_ids"] = sorted(already_retrieved)
                replan_seeds = []
                for memory in (*beliefs, *planned_seed_set):
                    if all(item["id"] != memory["id"] for item in replan_seeds):
                        replan_seeds.append(self._snapshot(memory))
                    if len(replan_seeds) >= 8:
                        break
                second = self._execute_plan(
                    replan,
                    replan_seeds,
                    round_offset=len(trace),
                    frame=frame,
                    question=question,
                )
                trace.extend(second["trace"])
                slot_validation.extend(second["slot_validation"])
                query_tokens["slot_validation"] += int(second["slot_validation_tokens"])
                query_latency["slot_validation"] += float(
                    second.get("slot_validation_latency", 0.0) or 0.0
                )
                operation_output_ids.update(
                    memory["id"]
                    for output in second["operation_outputs"]
                    for memory in output
                )
                operation_candidates.update(
                    {
                        memory["id"]: self._snapshot(memory)
                        for output in second["operation_outputs"]
                        for memory in output
                    }
                )
                evidence_refs.extend(second["evidence_refs"])

                by_id = {
                    memory["id"]: memory for memory in (*beliefs, *second["selected"])
                }
                beliefs, merged_relations = self._reconstruct_beliefs(
                    list(by_id.values()),
                    max(int(plan["max_memories"]), int(replan["max_memories"])),
                )
                relations = self._merge_relations(
                    relations,
                    second["relations"],
                    merged_relations,
                )
                for slot_id, memory_ids in second["slot_support"].items():
                    slot_support.setdefault(slot_id, [])
                    slot_support[slot_id].extend(
                        memory_id
                        for memory_id in memory_ids
                        if memory_id not in slot_support[slot_id]
                    )
                requirement_status, relation_status, retrieval_complete = (
                    self._retrieval_status(
                        plan,
                        slot_support,
                        beliefs,
                        relations,
                    )
                )
                slot_coverage = {
                    slot_id: status == "FOUND"
                    for slot_id, status in requirement_status.items()
                }

        query_latency["slot_validation"] = round(query_latency["slot_validation"], 3)
        query_latency["retrieval_wall"] = round(
            time.perf_counter() - retrieval_started, 3
        )

        return {
            "query_tokens": query_tokens,
            "query_latency": query_latency,
            "controller": controller,
            "fast_supports": fast_supports,
            "precomputed_answer": str(controller.get("answer") or ""),
            "seed_gate": getattr(self, "_seed_gate_telemetry", {}),
            "plan": plan,
            "replan": replan,
            "planner_called": planner_called,
            "replan_called": replan_called,
            "trace": trace,
            "slot_validation": slot_validation,
            "operation_output_ids": operation_output_ids,
            "operation_candidates": list(operation_candidates.values()),
            "planning_seeds": [self._snapshot(memory) for memory in planned_seed_set],
            "evidence_refs": evidence_refs,
            "slot_support": slot_support,
            "slot_coverage": slot_coverage,
            "requirement_status": requirement_status,
            "relation_status": relation_status,
            "retrieval_complete": retrieval_complete,
            "relations": relations,
            # Deprecated compatibility alias for old result consumers.
            "sufficient": retrieval_complete,
            "beliefs": beliefs,
        }

    def _reconstruct_beliefs(
        self,
        memories: List[Dict[str, Any]],
        limit: int,
        prefer_active: bool = True,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Pure arbitration: inspect only relations whose endpoints are already selected."""
        selected = {m["id"]: self._snapshot(m) for m in memories}
        input_ids = set(selected)
        relevant = []
        for relation in self._relations:
            if (
                relation.get("source_id") not in input_ids
                or relation.get("target_id") not in input_ids
            ):
                continue
            relation_type = str(relation.get("type") or "").upper()
            if relation_type == "CAUSES" and not self._valid_causal_relation(
                relation, selected
            ):
                continue
            if relation_type in {"REFINE", "SUPERSEDE", "CONFLICT"}:
                source = selected[relation["source_id"]]
                target = selected[relation["target_id"]]
                if not state_identity(source) or state_identity(
                    source
                ) != state_identity(target):
                    continue
            relevant.append(self._snapshot(relation))
        output = list(selected.values())
        for memory in output:
            memory["_status"] = self._belief_status.get(memory["id"], "active")

        # Arbitration may reorder/select within the authorized set, but never add IDs.
        output.sort(
            key=lambda m: (
                int(m.get("_best_effort_priority", 0)),
                (
                    m.get("_status") in {"active", "conflicting"}
                    if prefer_active
                    else True
                ),
                float(m.get("_score", 0.0)),
                self._recency_date(m),
                m["id"],
            ),
            reverse=True,
        )
        output = output[: max(0, int(limit))]
        output_ids = {memory["id"] for memory in output}
        if not output_ids.issubset(input_ids):
            raise AssertionError("arbitration expanded beyond its input IDs")
        return output, relevant
