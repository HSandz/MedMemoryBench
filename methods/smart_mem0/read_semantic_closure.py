"""Semantic closure for SmartMem0's locked two-stage READ path.

This layer repairs three representation/runtime gaps without adding an LLM call:
- deterministic selectors/source verification complete a retrieval program even when the
  semantic compute tier is SMALL;
- exact candidate-matching Top-3 seeds remain eligible CandidateSet evidence instead of
  disappearing merely because packet recall excludes seed duplication;
- final synthesis receives compact closure invariants for independent CandidateSet
  evaluation and genuinely multi-hop authorized reasoning.

It is dataset-, language-, and mode-neutral.  EFF and MIX share the same closure rules.
"""

from copy import deepcopy
from typing import Any, Dict, List


class ReadSemanticClosureMixin:
    SEMANTIC_CLOSURE_VERSION = "semantic-closure-v1"
    _WORK_OPS = frozenset({"SEARCH_FAMILY", "RESOLVE_STATE", "EXPAND_RELATION"})
    _TRANSFORM_OPS = frozenset({"SELECT", "VERIFY_SOURCE"})

    def _rc_normalize_ir(self, parsed: Dict[str, Any], question: str, frame: Any):
        """Align an explicit source-date constraint with the source/document axis.

        VERIFY_SOURCE is a semantic signal emitted by LLM #1.  Runtime does not parse
        English date words.  When that signal accompanies an EXACT anchored filter for a
        non-DATE answer, the date constrains the source occurrence rather than becoming the
        requested answer, so document_time is the conservative axis.
        """
        ir = super()._rc_normalize_ir(parsed, question, frame)
        verify_ids = {
            str(relation.get("from") or "")
            for relation in (ir.get("relations") or [])
            if str(relation.get("type") or "").upper() == "VERIFY_SOURCE"
        }
        if not verify_ids or str(ir.get("answer_type") or "").upper() == "DATE":
            return ir

        actions = list(ir.get("normalization_actions") or [])
        for requirement in ir.get("requirements") or []:
            rid = str(requirement.get("id") or "")
            if rid not in verify_ids:
                continue
            constraint = dict(
                requirement.get("time_constraint")
                or requirement.get("selector")
                or {}
            )
            relation = str(constraint.get("relation") or "").upper()
            axis = str(constraint.get("axis") or "").lower()
            anchor = str(constraint.get("anchor") or "").strip()
            if relation != "EXACT" or not anchor or axis != "event_time":
                continue
            constraint["axis"] = "document_time"
            requirement["time_constraint"] = dict(constraint)
            requirement["selector"] = dict(constraint)
            actions.append(
                {
                    "action": "SOURCE_FILTER_AXIS_NORMALIZED",
                    "requirement_id": rid,
                    "from_axis": "event_time",
                    "to_axis": "document_time",
                    "reason": "VERIFY_SOURCE_EXACT_FILTER_NON_DATE_PROJECTION",
                }
            )
        ir["normalization_actions"] = actions
        return ir

    @staticmethod
    def _closure_slot_relation(slot: Dict[str, Any]) -> str:
        return str(
            slot.get("time_relation")
            or slot.get("temporal_relation")
            or (slot.get("selector") or {}).get("relation")
            or ""
        ).upper()

    def _complete_deterministic_transforms(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Append selector/verification transforms without changing semantic work tier."""
        if not plan:
            return plan
        operations = [deepcopy(item) for item in (plan.get("operations") or [])]
        slots = {
            str(slot.get("id") or ""): slot
            for slot in (plan.get("required_slots") or [])
            if slot.get("id")
        }

        # EARLIEST/LATEST need family recall followed by a deterministic selector.
        # A one-obligation SMALL tier must not truncate that SELECT transform.
        for sid, slot in slots.items():
            relation = self._closure_slot_relation(slot)
            if relation not in {"EARLIEST", "LATEST"}:
                continue
            if any(
                str(operation.get("op") or "").upper() == "SELECT"
                and sid in (operation.get("produces") or [])
                for operation in operations
            ):
                continue
            producer_index = next(
                (
                    index
                    for index, operation in enumerate(operations)
                    if str(operation.get("op") or "").upper() == "SEARCH_FAMILY"
                    and sid in (operation.get("produces") or [])
                ),
                None,
            )
            if producer_index is None:
                continue
            producer = operations[producer_index]
            axis = str(
                slot.get("time_axis")
                or (slot.get("selector") or {}).get("axis")
                or "event_time"
            ).lower()
            # The family operation must be selector-neutral but history-aware.
            producer["family_mode"] = "temporal_extremum"
            producer["axis"] = axis
            operations.append(
                {
                    "op": "SELECT",
                    "query": str(producer.get("query") or ""),
                    "relation": relation,
                    "axis": axis,
                    "fallback_axis": "",
                    "candidate_refs": [f"${producer_index}"],
                    "produces": [sid],
                    "deterministic_transform": True,
                }
            )

        # VERIFY_SOURCE dereferences an already-authorized producer. It is not another
        # semantic retrieval decision and therefore does not consume the obligation tier.
        verify_ids = {
            str(relation.get("from") or "")
            for relation in (plan.get("semantic_relations") or [])
            if str(relation.get("type") or "").upper() == "VERIFY_SOURCE"
        }
        for sid in verify_ids:
            if not sid or any(
                str(operation.get("op") or "").upper() == "VERIFY_SOURCE"
                and sid in (operation.get("produces") or [])
                for operation in operations
            ):
                continue
            producer_index = next(
                (
                    index
                    for index in range(len(operations) - 1, -1, -1)
                    if sid in (operations[index].get("produces") or [])
                    and str(operations[index].get("op") or "").upper()
                    != "VERIFY_SOURCE"
                ),
                None,
            )
            if producer_index is None:
                continue
            operations.append(
                {
                    "op": "VERIFY_SOURCE",
                    "memory_refs": [f"${producer_index}"],
                    "produces": [sid],
                    "deterministic_transform": True,
                }
            )

        plan["operations"] = operations
        work_count = sum(
            str(operation.get("op") or "").upper() in self._WORK_OPS
            for operation in operations
        )
        transform_count = sum(
            str(operation.get("op") or "").upper() in self._TRANSFORM_OPS
            for operation in operations
        )
        basis = dict(plan.get("retrieval_budget_basis") or {})
        basis["logical_work_operations"] = work_count
        basis["deterministic_transforms"] = transform_count
        basis["physical_operations"] = len(operations)
        basis["transform_budget_semantics"] = "SELECT_VERIFY_SOURCE_DO_NOT_RAISE_SEMANTIC_TIER"
        plan["retrieval_budget_basis"] = basis
        plan["semantic_closure_version"] = self.SEMANTIC_CLOSURE_VERSION
        return plan

    def _controller_plan(self, ir, question, frame):
        return self._complete_deterministic_transforms(
            super()._controller_plan(ir, question, frame)
        )

    def _make_deterministic_recovery_plan(self, missing_slots, question, existing_plan):
        plan = super()._make_deterministic_recovery_plan(
            missing_slots, question, existing_plan
        )
        return self._complete_deterministic_transforms(plan) if plan else plan

    def _closure_memory_text(self, memory: Dict[str, Any]) -> str:
        return " ".join(
            str(value or "")
            for value in (
                memory.get("claim"),
                self._memory_value(memory),
                memory.get("verbatim_value"),
                memory.get("object_anchor"),
                " ".join(memory.get("entities") or []),
                " ".join(memory.get("scope_entities") or []),
            )
        )

    def _candidate_seed_match(self, proposition: str, memory: Dict[str, Any]) -> bool:
        """Exact/near-exact proposition identity is structural recall, never truth."""
        proposition_key = self._rc_text(proposition)
        if not proposition_key:
            return False
        memory_key = self._rc_text(self._closure_memory_text(memory))
        if proposition_key in memory_key:
            return True
        # For short named propositions, object/entity equality survives translation and
        # punctuation differences better than a long conditioned semantic query.
        if len(proposition_key.split()) <= 4:
            anchors = [
                memory.get("object_anchor"),
                *(memory.get("entities") or []),
                *(memory.get("scope_entities") or []),
            ]
            return proposition_key in {
                self._rc_text(value) for value in anchors if self._rc_text(value)
            }
        return False

    def _semantic_operation_search(
        self, query, top_k, strategy, frame=None, option_queries=None
    ):
        result = list(
            super()._semantic_operation_search(
                query,
                top_k,
                strategy,
                frame=frame,
                option_queries=option_queries,
            )
        )
        if str(strategy or "").upper() != "SHARED_OPTIONS" or not option_queries:
            return result

        propositions = {}
        for index, item in enumerate(option_queries or []):
            if isinstance(item, dict):
                label = str(item.get("label") or index)
                text = str(item.get("query") or item.get("text") or "").strip()
            else:
                label, text = str(index), str(item or "").strip()
            if text:
                propositions[label] = text
        if not propositions:
            return result

        coverage = deepcopy(
            getattr(self, "_last_option_probe_coverage", {})
            or getattr(self, "_last_proposition_probe_coverage", {})
            or {}
        )
        seed_matches: List[Dict[str, Any]] = []
        for label, proposition in propositions.items():
            current = list(coverage.get(label, []))
            for seed in (getattr(self, "_active_controller_seeds", []) or [])[:3]:
                memory_id = str(seed.get("id") or "")
                if not memory_id or not self._candidate_seed_match(proposition, seed):
                    continue
                if memory_id not in current:
                    current.insert(0, memory_id)
                if all(str(item.get("id") or "") != memory_id for item in seed_matches):
                    seed_matches.append(self._snapshot(seed))
            coverage[label] = current

        membership: Dict[str, set] = {}
        for label, ids in coverage.items():
            for memory_id in ids or []:
                membership.setdefault(str(memory_id), set()).add(str(label))
        local = {
            str(label): [
                str(memory_id)
                for memory_id in ids or []
                if membership.get(str(memory_id), set()) == {str(label)}
            ]
            for label, ids in coverage.items()
        }
        shared = [
            memory_id for memory_id, labels in membership.items() if len(labels) > 1
        ]
        self._last_option_probe_coverage = deepcopy(coverage)
        self._last_proposition_probe_coverage = deepcopy(coverage)
        self._last_candidate_local_coverage = deepcopy(local)
        self._last_candidate_shared_context_ids = list(dict.fromkeys(shared))

        ordered = []
        seen = set()
        for memory in (*seed_matches, *result):
            memory_id = str(memory.get("id") or "")
            if memory_id and memory_id not in seen:
                ordered.append(self._snapshot(memory))
                seen.add(memory_id)
            if len(ordered) >= max(1, int(top_k)):
                break
        return ordered

    def _semantic_closure_block(self, prepared: Dict[str, Any]) -> str:
        extra = prepared.get("extra") or {}
        plan = extra.get("replan") or extra.get("plan") or {}
        blocks = []
        if (extra.get("candidate_set") or {}).get("candidates"):
            blocks.append(
                "Candidate closure: evaluate every proposition independently. A multi-attribute "
                "current-condition question is not a global LATEST contest: apply recency within "
                "each proposition's evidence. Combine candidate-local and shared evidence, and "
                "return every supported candidate; UNKNOWN is not false."
            )
        bridges = list(plan.get("reasoning_bridges") or [])
        if any(
            str(bridge.get("type") or "").upper() in {"POSSIBLE_CAUSE", "INFER"}
            for bridge in bridges
        ):
            blocks.append(
                "Reasoning closure: when an authorized POSSIBLE_CAUSE/INFER bridge is not a "
                "stored direct relation, trace the necessary intermediate mechanism(s) explicitly "
                "before concluding. Prefer a real multi-hop chain over a generic label; use only "
                "supplied participant facts for personal history and only high-confidence general "
                "knowledge for the intermediate rule/mechanism."
            )
        return "\n".join(blocks)

    def prepare_batch_query(self, question, system_message=None, **kwargs):
        prepared = super().prepare_batch_query(
            question, system_message=system_message, **kwargs
        )
        extra = prepared.setdefault("extra", {})
        extra["semantic_closure_version"] = self.SEMANTIC_CLOSURE_VERSION
        block = self._semantic_closure_block(prepared)
        if block:
            for message in prepared.get("messages") or []:
                if str(message.get("role") or "").lower() == "system":
                    message["content"] = str(message.get("content") or "").rstrip() + "\n\n" + block
                    break
            extra["semantic_closure_materialized"] = True
        else:
            extra["semantic_closure_materialized"] = False
        return prepared
