"""Primary-first retrieval fusion for SmartMem0's locked two-stage READ path.

Evidence obligations create the candidate universe. Family/key/question views may rerank
that bounded universe but cannot expand it unless the primary obligation has no viable hit.
CandidateSet probes remain retrieval-only and are reordered by proposition affinity. A
POSSIBLE_CAUSE bridge may consume at most one provenance-valid stored CAUSES edge from a
strongly bound endpoint. No model call, query-type route, or mode-specific rule is added.
"""

from copy import deepcopy
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from .contracts import RETRIEVAL_BUDGETS


class ReadRetrievalFusionMixin:
    RETRIEVAL_FUSION_VERSION = "primary-recall-aux-rank-v1"
    FUSION_RRF_K = 60.0
    FUSION_WEIGHTS = {"obligation": 1.0, "keys": 0.72, "family": 0.58, "question": 0.32}
    FUSION_RESCUE_ORDER = ("keys", "family", "question")

    @staticmethod
    def _fusion_unique(values: Iterable[Any]) -> List[str]:
        output, seen = [], set()
        for value in values:
            text = str(value or "")
            if text and text not in seen:
                output.append(text)
                seen.add(text)
        return output

    @staticmethod
    def _fusion_views(views: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        output = {}
        for raw in views or []:
            kind = str((raw or {}).get("kind") or "").lower()
            query = " ".join(str((raw or {}).get("query") or "").split()).strip()
            if kind and query and kind not in output:
                output[kind] = dict(raw, kind=kind, query=query)
        return output

    def _fusion_has_hit(self, rows) -> bool:
        checker = getattr(self, "_retrieval_has_semantic_hit", None)
        return bool(checker(rows)) if callable(checker) else bool(rows)

    def _fusion_rerank(self, query: str, ids: Sequence[str]):
        ids = self._fusion_unique(ids)
        if not query or not ids:
            return []
        return list(self._hybrid_search(query, top_k=len(ids), candidate_ids=set(ids)) or [])

    def _fusion_child(self, operation, query, outputs, seeds, frame):
        child = deepcopy(operation)
        for key in ("retrieval_views", "retrieval_view_semantics", "requirement_id"):
            child.pop(key, None)
        child["query"] = query
        return super()._execute_operation(child, outputs, seeds, frame)

    def _fusion_multiview(self, operation, outputs, seeds, frame):
        views = self._fusion_views(operation.get("retrieval_views") or [])
        if not views:
            return super()._execute_operation(operation, outputs, seeds, frame)

        rid = str(operation.get("requirement_id") or "")
        top_k = max(1, int(operation.get("top_k", 6) or 6))
        primary = views.get("obligation") or next(iter(views.values()))
        primary_kind, primary_query = primary["kind"], primary["query"]
        primary_rows, relations, evidence_refs = self._fusion_child(
            operation, primary_query, outputs, seeds, frame
        )
        primary_rows = list(primary_rows or [])
        primary_viable = self._fusion_has_hit(primary_rows)

        universe, producer_kind = list(primary_rows), primary_kind
        rescue = None
        if not primary_viable:
            rescue = next(
                (views[kind] for kind in self.FUSION_RESCUE_ORDER if kind != primary_kind and kind in views),
                None,
            )
        if rescue:
            rescued, rescue_relations, rescue_evidence = self._fusion_child(
                operation, rescue["query"], outputs, seeds, frame
            )
            self._fusion_stats["zero_hit_rescue_searches"] = int(self._fusion_stats.get("zero_hit_rescue_searches", 0)) + 1
            relations = self._merge_fusion(relations, rescue_relations)
            evidence_refs = self._fusion_unique([*(evidence_refs or []), *(rescue_evidence or [])])
            if rescued:
                universe, producer_kind = list(rescued), rescue["kind"]

        by_id = {str(m.get("id") or ""): deepcopy(m) for m in universe if m.get("id")}
        ids = list(by_id)
        score = {memory_id: 0.0 for memory_id in ids}
        bound = {memory_id: [] for memory_id in ids}
        coverage = {
            primary_kind: [
                str(m.get("id") or "") for m in primary_rows if str(m.get("id") or "") in by_id
            ]
        }
        if producer_kind != primary_kind:
            coverage[producer_kind] = list(ids)

        producer_weight = float(self.FUSION_WEIGHTS.get(producer_kind, 1.0))
        for rank, memory in enumerate(universe, 1):
            memory_id = str(memory.get("id") or "")
            if memory_id in score:
                score[memory_id] += producer_weight / (self.FUSION_RRF_K + rank)
                bound[memory_id].append(producer_kind)

        aux_reranks = 0
        for kind, view in views.items():
            if kind == producer_kind:
                continue
            ranked = self._fusion_rerank(view["query"], ids)
            if not ranked:
                continue
            aux_reranks += 1
            self._fusion_stats["auxiliary_searches_avoided"] = int(self._fusion_stats.get("auxiliary_searches_avoided", 0)) + 1
            coverage[kind] = [str(m.get("id") or "") for m in ranked if str(m.get("id") or "") in score]
            weight = float(self.FUSION_WEIGHTS.get(kind, 0.25))
            for rank, memory in enumerate(ranked, 1):
                memory_id = str(memory.get("id") or "")
                if memory_id in score and self._fusion_has_hit([memory]):
                    score[memory_id] += weight / (self.FUSION_RRF_K + rank)
                    if kind not in bound[memory_id]:
                        bound[memory_id].append(kind)

        original_rank = {str(m.get("id") or ""): i for i, m in enumerate(universe, 1) if m.get("id")}
        ordered = sorted(ids, key=lambda mid: (-score[mid], original_rank.get(mid, 10**9), mid))[:top_k]
        result, binding_scores, binding_views = [], {}, {}
        for memory_id in ordered:
            memory = deepcopy(by_id[memory_id])
            memory["_binding_score"] = round(score[memory_id], 8)
            memory["_binding_views"] = list(bound[memory_id])
            memory["_fusion_primary_candidate"] = True
            result.append(memory)
            binding_scores[memory_id] = score[memory_id]
            binding_views[memory_id] = list(bound[memory_id])

        if rid:
            self._last_requirement_view_coverage[rid] = deepcopy(coverage)
            self._last_requirement_binding_scores[rid] = binding_scores
            self._last_requirement_binding_views[rid] = binding_views
            self._fusion_stats.setdefault("requirements", {})[rid] = {
                "primary_view": primary_kind,
                "primary_viable": primary_viable,
                "rescue_view": producer_kind if producer_kind != primary_kind else "",
                "candidate_count": len(ids),
                "final_count": len(result),
                "auxiliary_reranks": aux_reranks,
            }
        return result, relations or [], evidence_refs or []

    @staticmethod
    def _merge_fusion(left, right):
        output = list(left or [])
        for item in right or []:
            if item not in output:
                output.append(item)
        return output

    def _fusion_bridge(self, operation):
        source_rid = str(operation.get("source_requirement") or "")
        scores = (getattr(self, "_last_requirement_binding_scores", {}) or {}).get(source_rid) or {}
        views = (getattr(self, "_last_requirement_binding_views", {}) or {}).get(source_rid) or {}
        ranked_sources = sorted(
            ((str(mid), float(score)) for mid, score in scores.items() if "obligation" in set(views.get(str(mid), []) or [])),
            key=lambda item: (-item[1], item[0]),
        )
        if not ranked_sources:
            return [], [], []

        by_id = {str(m.get("id") or ""): m for m in getattr(self, "_memories", []) or [] if m.get("id")}
        source_id = ranked_sources[0][0]
        source = by_id.get(source_id)
        if not source:
            return [], [], []
        validator = getattr(self, "_valid_causal_relation", None)
        edges = []
        for relation in getattr(self, "_relations", []) or []:
            target_id = str(relation.get("target_id") or "")
            if (
                str(relation.get("type") or "").upper() == "CAUSES"
                and str(relation.get("source_id") or "") == source_id
                and target_id in by_id
                and (not callable(validator) or validator(relation, by_id))
            ):
                edges.append((target_id, relation))
        if not edges:
            return [], [], []

        goal = str(operation.get("goal") or "").strip()
        target_ids = [target_id for target_id, _ in edges]
        ranked = self._fusion_rerank(goal, target_ids) if goal else []
        if ranked and not self._fusion_has_hit([ranked[0]]):
            return [], [], []
        target_id = str(ranked[0].get("id") or "") if ranked else target_ids[0]
        relation = next(rel for candidate_id, rel in edges if candidate_id == target_id)
        self._fusion_stats["bridge_relation_count"] = int(self._fusion_stats.get("bridge_relation_count", 0)) + 1
        self._fusion_stats.setdefault("bridge_relations", []).append(deepcopy(relation))
        return [self._snapshot(source), self._snapshot(by_id[target_id])], [deepcopy(relation)], []

    def _execute_operation(self, operation, outputs, seeds, frame=None):
        lean_op = str(operation.get("_lean_op") or operation.get("op") or "").upper()
        if operation.get("bridge_bound_expansion") and lean_op == "EXPAND_RELATION":
            return self._fusion_bridge(operation)
        if lean_op == "SEARCH_FAMILY" and operation.get("retrieval_views"):
            return self._fusion_multiview(operation, outputs, seeds, frame)
        return super()._execute_operation(operation, outputs, seeds, frame)

    def _compile_gap_operations(self, slots, question, budget_tier="MEDIUM", plan=None):
        operations = list(super()._compile_gap_operations(slots, question, budget_tier, plan=plan))
        plan = plan or {}
        max_ops = int(RETRIEVAL_BUDGETS.get(str(budget_tier or "MEDIUM").upper(), RETRIEVAL_BUDGETS["MEDIUM"])["max_operations"])
        if len(operations) >= max_ops:
            return operations
        slots_by_id = {str(slot.get("id") or ""): slot for slot in slots or [] if slot.get("id")}
        for relation in plan.get("semantic_relations") or []:
            if str(relation.get("type") or "").upper() != "POSSIBLE_CAUSE":
                continue
            source_id, target_id = str(relation.get("from") or ""), str(relation.get("to") or "")
            if source_id not in slots_by_id or target_id not in slots_by_id:
                continue
            if not any(
                source_id in (op.get("produces") or [])
                and str(op.get("op") or "").upper() == "SEARCH_FAMILY"
                and op.get("retrieval_views")
                for op in operations
            ):
                continue
            target = slots_by_id[target_id]
            operations.append(
                {
                    "op": "EXPAND_RELATION",
                    "relation": "CAUSES",
                    "direction": "OUT",
                    "max_hops": 1,
                    "bridge_bound_expansion": True,
                    "source_requirement": source_id,
                    "target_requirement": target_id,
                    "goal": str(target.get("retrieval_target") or target.get("target_surface") or target.get("description") or "").strip(),
                    "produces": [source_id, target_id],
                }
            )
            break
        return operations[:max_ops]

    @staticmethod
    def _fusion_options(option_queries) -> List[Tuple[str, str]]:
        output = []
        for index, item in enumerate(option_queries or []):
            if isinstance(item, dict):
                label = str(item.get("label") or index)
                text = str(item.get("text") or item.get("query") or "").strip()
            else:
                label, text = str(index), str(item or "").strip()
            if label and text:
                output.append((label, text))
        return output

    def _semantic_operation_search(self, query, top_k, strategy, frame=None, option_queries=None):
        result = list(super()._semantic_operation_search(query, top_k, strategy, frame=frame, option_queries=option_queries) or [])
        if str(strategy or "").upper() != "SHARED_OPTIONS" or not option_queries:
            return result

        coverage = deepcopy(getattr(self, "_last_option_probe_coverage", {}) or getattr(self, "_last_proposition_probe_coverage", {}) or {})
        propositions = self._fusion_options(option_queries)
        store = {str(m.get("id") or ""): m for m in getattr(self, "_memories", []) or [] if m.get("id")}
        for label, proposition in propositions:
            current = self._fusion_unique(coverage.get(label, []))
            ranked_ids = [str(m.get("id") or "") for m in self._fusion_rerank(proposition, current)]
            coverage[label] = self._fusion_unique([*ranked_ids, *current])

        membership: Dict[str, set] = {}
        for label, ids in coverage.items():
            for memory_id in ids or []:
                membership.setdefault(str(memory_id), set()).add(str(label))
        local = {
            str(label): [str(mid) for mid in ids or [] if membership.get(str(mid), set()) == {str(label)}]
            for label, ids in coverage.items()
        }
        shared = [mid for mid, labels in membership.items() if len(labels) > 1]
        self._last_option_probe_coverage = deepcopy(coverage)
        self._last_proposition_probe_coverage = deepcopy(coverage)
        self._last_candidate_local_coverage = deepcopy(local)
        self._last_candidate_shared_context_ids = list(dict.fromkeys(shared))

        result_by_id = {str(m.get("id") or ""): m for m in result if m.get("id")}
        ordered, seen = [], set()
        def add(memory_id):
            memory = result_by_id.get(memory_id) or store.get(memory_id)
            if memory_id and memory is not None and memory_id not in seen:
                ordered.append(self._snapshot(memory))
                seen.add(memory_id)
        for label, _ in propositions:
            if local.get(label):
                add(local[label][0])
        for memory in result:
            add(str(memory.get("id") or ""))
        return ordered[: max(1, int(top_k))]

    def _authorize_controller_answer(self, ir, seeds, frame):
        supports, reason = super()._authorize_controller_answer(ir, seeds, frame)
        if supports is None:
            return supports, reason
        requirements, candidate = list(ir.get("requirements") or []), ir.get("candidate") or {}
        if len(requirements) != 1 or not supports or not candidate:
            return supports, reason
        requirement = requirements[0]
        if str((requirement.get("proof_spec") or {}).get("status") or "").upper() == "VALID":
            return supports, reason
        target = " ".join(str(requirement.get("target") or requirement.get("answer_obligation") or requirement.get("focus_span") or "").split()).strip()
        if not target:
            return supports, reason
        memory_text = self._terminal_memory_text(supports[0])
        if self._rc_token_sequence_present(target, memory_text):
            return supports, reason
        if float(self._terminal_surface_similarity(target, memory_text)) < 0.34:
            return None, "DIRECT_OBLIGATION_NOT_CLOSED"
        return supports, reason

    def _run_query_retrieval(self, question, initial_seeds, frame, fast_supports, gate, planning_seeds=None, planning_context=None):
        self._fusion_stats = {
            "version": self.RETRIEVAL_FUSION_VERSION,
            "requirements": {},
            "auxiliary_searches_avoided": 0,
            "zero_hit_rescue_searches": 0,
            "bridge_relation_count": 0,
            "stored_bridge_relation_materialized": False,
            "bridge_relations": [],
        }
        run = super()._run_query_retrieval(
            question,
            initial_seeds,
            frame,
            fast_supports,
            gate,
            planning_seeds=planning_seeds,
            planning_context=planning_context,
        )
        run["retrieval_fusion"] = deepcopy(self._fusion_stats)
        return run

    def prepare_batch_query(self, question, system_message=None, **kwargs):
        prepared = super().prepare_batch_query(question, system_message=system_message, **kwargs)
        extra = prepared.setdefault("extra", {})
        fusion = deepcopy(getattr(self, "_fusion_stats", {}) or {})
        final = {str(m.get("id") or ""): m for m in prepared.get("retrieved_memories") or [] if m.get("id")}
        materialized = [
            relation
            for relation in fusion.get("bridge_relations") or []
            if str(relation.get("source_id") or "") in final and str(relation.get("target_id") or "") in final
        ]
        fusion["stored_bridge_relation_materialized"] = bool(materialized)
        fusion["materialized_bridge_relation_count"] = len(materialized)
        extra["retrieval_fusion"] = fusion
        extra["retrieval_fusion_version"] = self.RETRIEVAL_FUSION_VERSION
        if materialized:
            lines = [
                "=== STORED BRIDGE RELATION ===",
                "These directed relations are stored ledger relations whose endpoints both survived final ProofContext.",
            ]
            for relation in materialized[:2]:
                source, target = final[str(relation.get("source_id") or "")], final[str(relation.get("target_id") or "")]
                lines.append(
                    "- " + " ".join(str(source.get("claim") or "").split())[:180]
                    + " -[CAUSES]-> " + " ".join(str(target.get("claim") or "").split())[:180]
                )
            block = "\n".join(lines)
            for message in prepared.get("messages") or []:
                if str(message.get("role") or "").lower() == "system":
                    message["content"] = str(message.get("content") or "").rstrip() + "\n\n" + block
                    break
            extra["relations_used"] = sorted(set(extra.get("relations_used") or []) | {"CAUSES"})
        return prepared
