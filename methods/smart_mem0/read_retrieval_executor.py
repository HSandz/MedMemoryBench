"""Lean retrieval program for SmartMem0 READ.

The controller emits evidence obligations; this facade is the active physical retrieval
compiler. Public plans use only SEARCH_FAMILY, SELECT, RESOLVE_STATE, EXPAND_RELATION,
and VERIFY_SOURCE. The older operation vocabulary survives only below this adapter.

RetrievalView cache entries are query-local. Semantic family search prefers HOT memories
and opens COLD only when the HOT view has no viable semantic hit. Temporal family
expansion remains selector-neutral: after a family root is found, SELECT owns the
EARLIEST/LATEST/EXACT/etc. selector.
"""

import json
from copy import deepcopy
from typing import Any, Dict, Sequence, Tuple

from .contracts import QueryFrame, RETRIEVAL_BUDGETS
from .read_family_retrieval_contract import ReadFamilyRetrievalContractMixin


class ReadRetrievalExecutorMixin(ReadFamilyRetrievalContractMixin):
    RETRIEVAL_EXECUTOR_VERSION = "lean-retrieval-v1"
    RETRIEVAL_VIEW_CACHE_VERSION = "retrieval-view-cache-v1"
    LEAN_OPERATIONS = frozenset(
        {"SEARCH_FAMILY", "SELECT", "RESOLVE_STATE", "EXPAND_RELATION", "VERIFY_SOURCE"}
    )
    SEMANTIC_HOT_DENSE_FLOOR = 0.36

    @classmethod
    def _retrieval_has_semantic_hit(cls, rows: Sequence[Dict[str, Any]]) -> bool:
        if not rows:
            return False
        scored = False
        for row in rows:
            scored |= "_overlap" in row or "_dense_score" in row
            if int(row.get("_overlap", 0) or 0) > 0:
                return True
            if float(row.get("_dense_score", 0.0) or 0.0) >= cls.SEMANTIC_HOT_DENSE_FLOOR:
                return True
        return not scored

    def _retrieval_tier_ids(
        self, frame: QueryFrame, tier: str, *, include_history: bool = False
    ) -> set:
        wanted = str(tier or "").upper()
        return {
            memory["id"]
            for memory in getattr(self, "_memories", []) or []
            if memory.get("id")
            and str(memory.get("memory_tier") or "HOT").upper() == wanted
            and self._memory_satisfies_frame(
                memory,
                frame,
                include_entities=bool(getattr(frame, "hard_entities", ()) or ()),
            )
            and self._query_visible_memory(memory, include_history=include_history)
        }

    def _search_family_hot_first(
        self,
        query: str,
        top_k: int,
        frame: QueryFrame,
        *,
        include_history: bool = False,
    ):
        hot_ids = self._retrieval_tier_ids(
            frame, "HOT", include_history=include_history
        )
        hot = (
            self._hybrid_search(
                query,
                top_k=min(max(1, int(top_k)), len(hot_ids)),
                candidate_ids=hot_ids,
            )
            if hot_ids
            else []
        )
        if self._retrieval_has_semantic_hit(hot):
            return [self._snapshot(memory) for memory in hot[:top_k]]

        cold_ids = self._retrieval_tier_ids(
            frame, "COLD", include_history=include_history
        )
        cold = (
            self._hybrid_search(
                query,
                top_k=min(max(1, int(top_k)), len(cold_ids)),
                candidate_ids=cold_ids,
            )
            if cold_ids
            else []
        )
        output, seen = [], set()
        for memory in (*cold, *hot):
            memory_id = str(memory.get("id") or "")
            if memory_id and memory_id not in seen:
                output.append(self._snapshot(memory))
                seen.add(memory_id)
            if len(output) >= max(1, int(top_k)):
                break
        return output

    @staticmethod
    def _canonical_operation(legacy, index, program):
        current = deepcopy(legacy)
        op = str(current.get("op") or "")
        if op == "SEMANTIC_SEARCH":
            current["op"] = "SEARCH_FAMILY"
            strategy = str(current.pop("strategy", "FOCAL") or "FOCAL").upper()
            current["family_mode"] = (
                "candidate_set" if strategy == "SHARED_OPTIONS" else "semantic"
            )
            current["candidate_queries"] = current.pop("option_queries", [])
        elif op == "LOCATE_ANCHOR":
            current["op"] = "SEARCH_FAMILY"
            current["family_mode"] = "anchor"
            ref = f"${index}"
            selector = next(
                (
                    candidate
                    for candidate in program
                    if candidate.get("op") == "TEMPORAL_FILTER"
                    and ref in (candidate.get("candidate_refs") or [])
                    and str(candidate.get("relation") or "").upper()
                    in {"EARLIEST", "LATEST"}
                ),
                None,
            )
            if selector:
                current["family_mode"] = "temporal_extremum"
                current["axis"] = str(selector.get("axis") or "event_time")
        elif op == "TEMPORAL_FILTER":
            current["op"] = "SELECT"
        elif op == "FOLLOW_CAUSES":
            current["op"] = "EXPAND_RELATION"
            current["relation"] = "CAUSES"
            current["max_hops"] = 1
            current.pop("depth", None)
        elif op == "VERIFY_EVIDENCE":
            current["op"] = "VERIFY_SOURCE"
        return current

    def _compile_gap_operations(
        self, slots, question, budget_tier="MEDIUM", plan=None
    ):
        legacy = super()._compile_gap_operations(
            slots, question, budget_tier, plan=plan
        )
        return [
            self._canonical_operation(operation, index, legacy)
            for index, operation in enumerate(legacy)
        ]

    @staticmethod
    def _retrieval_cache_key(operation):
        stable = {
            key: value
            for key, value in operation.items()
            if key not in {"produces", "exclude_ids"}
        }
        return json.dumps(stable, sort_keys=True, ensure_ascii=False, default=str)

    @staticmethod
    def _retrieval_cacheable(operation):
        return (
            str(operation.get("op") or "") in {"SEARCH_FAMILY", "RESOLVE_STATE"}
            and not operation.get("exclude_ids")
        )

    def _execute_operation(self, operation, outputs, seeds, frame=QueryFrame()):
        op = str(operation.get("op") or "")
        if op not in self.LEAN_OPERATIONS:
            return super()._execute_operation(operation, outputs, seeds, frame)

        cache = getattr(self, "_retrieval_view_cache", None)
        if cache is None:
            cache = self._retrieval_view_cache = {}
        key = self._retrieval_cache_key(operation)
        if self._retrieval_cacheable(operation) and key in cache:
            self._retrieval_view_cache_hits = (
                int(getattr(self, "_retrieval_view_cache_hits", 0) or 0) + 1
            )
            cached = cache[key]
            return (
                deepcopy(cached["result"]),
                deepcopy(cached["relations"]),
                list(cached["evidence_refs"]),
            )

        if op == "SEARCH_FAMILY":
            mode = str(operation.get("family_mode") or "semantic")
            if mode == "temporal_extremum":
                result = self._locate_temporal_family(
                    str(operation.get("query") or ""),
                    frame,
                    str(operation.get("axis") or "event_time"),
                )
                relations, evidence_refs = [], []
            elif mode in {"semantic", "anchor"}:
                result = self._search_family_hot_first(
                    str(operation.get("query") or ""),
                    max(1, int(operation.get("top_k", 8) or 8)),
                    frame,
                    include_history=mode == "anchor",
                )
                relations, evidence_refs = [], []
            else:
                legacy = deepcopy(operation)
                legacy["op"] = "SEMANTIC_SEARCH"
                legacy["strategy"] = "SHARED_OPTIONS"
                legacy["option_queries"] = legacy.pop("candidate_queries", [])
                legacy.pop("family_mode", None)
                result, relations, evidence_refs = super()._execute_operation(
                    legacy, outputs, seeds, frame
                )
        else:
            legacy = deepcopy(operation)
            if op == "SELECT":
                legacy["op"] = "TEMPORAL_FILTER"
            elif op == "EXPAND_RELATION":
                legacy["op"] = "FOLLOW_CAUSES"
                legacy["depth"] = 1
                legacy.pop("max_hops", None)
                legacy.pop("relation", None)
            elif op == "VERIFY_SOURCE":
                legacy["op"] = "VERIFY_EVIDENCE"
            result, relations, evidence_refs = super()._execute_operation(
                legacy, outputs, seeds, frame
            )

        if self._retrieval_cacheable(operation):
            cache[key] = {
                "result": deepcopy(result),
                "relations": deepcopy(relations),
                "evidence_refs": list(evidence_refs),
            }
            self._retrieval_view_cache_misses = (
                int(getattr(self, "_retrieval_view_cache_misses", 0) or 0) + 1
            )
        return result, relations, evidence_refs

    @staticmethod
    def _obligation_budget_tier(plan):
        count = len(plan.get("required_slots") or [])
        if count >= 3:
            return "LARGE"
        if count >= 2:
            return "MEDIUM"
        return "SMALL"

    def _controller_plan(self, ir, question, frame):
        plan = super()._controller_plan(ir, question, frame)
        tier = self._obligation_budget_tier(plan)
        operations = self._compile_gap_operations(
            plan.get("required_slots") or [], question, tier, plan=plan
        )
        if len(operations) > RETRIEVAL_BUDGETS[tier]["max_operations"]:
            tier = "MEDIUM" if tier == "SMALL" else "LARGE"
            operations = self._compile_gap_operations(
                plan.get("required_slots") or [], question, tier, plan=plan
            )
        plan["budget_tier"] = tier
        plan["max_memories"] = RETRIEVAL_BUDGETS[tier]["max_memories"]
        plan["operations"] = operations
        plan["retrieval_program_version"] = self.RETRIEVAL_EXECUTOR_VERSION
        plan["retrieval_budget_basis"] = {
            "evidence_obligations": len(plan.get("required_slots") or []),
            "candidate_count_not_budget_class": len(
                ir.get("candidate_propositions") or ir.get("visible_options") or {}
            ),
            "physical_operations": len(operations),
        }
        return plan

    @staticmethod
    def _lean_operation_signature(operation) -> Tuple[str, ...]:
        return (
            str(operation.get("op") or ""),
            str(operation.get("family_mode") or ""),
            str(operation.get("query") or ""),
            str(operation.get("relation") or ""),
            str(operation.get("axis") or ""),
            str(operation.get("anchor") or ""),
            str(operation.get("end") or ""),
        )

    def _make_deterministic_recovery_plan(
        self, missing_slots, question, existing_plan
    ):
        if not missing_slots:
            return None
        broadened = []
        for slot in missing_slots:
            copy = deepcopy(slot)
            copy["resolved_keys"] = []
            copy["retrieval_hint"] = ""
            copy["evidence_family"] = ""
            obligation = str(
                copy.get("answer_obligation")
                or copy.get("proof_anchor")
                or copy.get("target_surface")
                or ""
            ).strip()
            if obligation:
                copy["retrieval_target"] = obligation
                copy["target_surface"] = obligation
            broadened.append(copy)

        tier = str(existing_plan.get("budget_tier") or "MEDIUM")
        shell = {
            "query_mode": existing_plan.get("query_mode", "DIRECT"),
            "visible_options": existing_plan.get("visible_options", {}),
            "semantic_relations": existing_plan.get("semantic_relations", []),
        }
        operations = self._compile_gap_operations(
            broadened, "", tier, plan=shell
        )
        previous = {
            self._lean_operation_signature(operation)
            for operation in existing_plan.get("operations", [])
        }
        if not operations or all(
            self._lean_operation_signature(operation) in previous
            for operation in operations
        ):
            return None
        return {
            "query_spec": deepcopy(existing_plan.get("query_spec", {})),
            "query_mode": shell["query_mode"],
            "required_slots": broadened,
            "semantic_relations": deepcopy(shell["semantic_relations"]),
            "seed_coverage": [],
            "operations": operations,
            "option_coverage": [],
            "visible_options": deepcopy(shell["visible_options"]),
            "need_evidence": bool(existing_plan.get("need_evidence")),
            "need_raw_evidence": bool(existing_plan.get("need_raw_evidence")),
            "budget_tier": tier,
            "max_memories": RETRIEVAL_BUDGETS.get(
                tier, RETRIEVAL_BUDGETS["MEDIUM"]
            )["max_memories"],
            "planner_fallback": True,
            "fallback_reason": "deterministic_obligation_only_novelty_recovery",
            "valid": True,
            "retrieval_program_version": self.RETRIEVAL_EXECUTOR_VERSION,
        }

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
        self._retrieval_view_cache = {}
        self._retrieval_view_cache_hits = 0
        self._retrieval_view_cache_misses = 0
        run = super()._run_query_retrieval(
            question,
            initial_seeds,
            frame,
            fast_supports,
            gate,
            planning_seeds=planning_seeds,
            planning_context=planning_context,
        )
        state = {
            "version": self.RETRIEVAL_VIEW_CACHE_VERSION,
            "entries": len(self._retrieval_view_cache),
            "hits": int(self._retrieval_view_cache_hits),
            "misses": int(self._retrieval_view_cache_misses),
        }
        self._last_retrieval_view_cache = deepcopy(state)
        run["retrieval_view_cache"] = state
        return run

    def prepare_batch_query(self, question, system_message=None, **kwargs):
        prepared = super().prepare_batch_query(
            question, system_message=system_message, **kwargs
        )
        extra = prepared.setdefault("extra", {})
        extra["retrieval_executor_version"] = self.RETRIEVAL_EXECUTOR_VERSION
        extra["retrieval_view_cache"] = deepcopy(
            getattr(self, "_last_retrieval_view_cache", {}) or {}
        )
        source_plan = extra.get("plan") or {}
        extra["retrieval_program"] = [
            {
                "op": operation.get("op"),
                "produces": list(operation.get("produces") or []),
            }
            for operation in source_plan.get("operations") or []
        ]
        return prepared
