"""Adequacy-gated candidate acquisition for SmartMem0 READ.

The semantic controller still speaks once and owns the question meaning. This layer
changes only deterministic physical retrieval: a weak lexical/dense hit is allowed to
enter a candidate pool, but it is no longer sufficient to close candidate acquisition.
When the obligation view is weak, bounded auxiliary views may contribute a few novel
IDs before all views are fused over the same candidate world. HOT is likewise a storage
priority rather than a proof boundary: COLD opens when the HOT pool is not adequate.

No benchmark label, query type, language cue, extra model call, or graph hop is added.
ProofContext remains the sole final context owner.
"""

from copy import deepcopy
from typing import Any, Dict, Iterable, List, Sequence


class ReadProgressiveRetrievalMixin:
    """Turn one semantic intent into bounded, evidence-driven candidate acquisition."""

    PROGRESSIVE_RETRIEVAL_VERSION = "progressive-evidence-retrieval-v1"
    RETRIEVAL_FUSION_VERSION = "adequacy-gated-candidate-world-v2"

    # A weak hit is useful for candidate admission, but acquisition stops only on a
    # stronger generic signal. These thresholds are retrieval-scale heuristics only;
    # they never prove answer correctness or semantic truth.
    ADEQUACY_WEAK_DENSE_FLOOR = 0.36
    ADEQUACY_STRONG_DENSE_FLOOR = 0.50
    ADEQUACY_STRONG_OVERLAP = 2
    ADEQUACY_DENSE_MARGIN = 0.04
    ADEQUACY_WEAK_RANK_FACTOR = 0.25

    # Candidate acquisition may be wider than the final per-requirement result, but
    # remains tightly bounded. ProofContext applies its own, later context budget.
    FUSION_AUX_NOVEL_PER_VIEW = 2
    FUSION_CANDIDATE_MULTIPLIER = 2
    FUSION_CANDIDATE_HARD_CAP = 16

    @classmethod
    def _retrieval_pool_adequacy(
        cls, rows: Sequence[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Return deterministic retrieval adequacy without claiming answerability.

        Adequacy is deliberately stronger than the legacy ``has semantic hit`` gate.
        One weak overlap or borderline dense score can remain a candidate, but does
        not prevent another authorized retrieval surface from being consulted.
        """
        rows = list(rows or [])
        if not rows:
            return {
                "adequate": False,
                "reason": "empty",
                "best_overlap": 0,
                "best_dense": 0.0,
                "dense_margin": 0.0,
                "scored": False,
            }

        scored = any(
            "_overlap" in row or "_dense_score" in row
            for row in rows
        )
        if not scored:
            # Compatibility for legacy deterministic callers that do not expose
            # ranking telemetry. Real hybrid retrieval rows are scored.
            return {
                "adequate": True,
                "reason": "unscored_compatibility",
                "best_overlap": 0,
                "best_dense": 0.0,
                "dense_margin": 0.0,
                "scored": False,
            }

        best_overlap = 0
        best_dense = 0.0
        strong_lexical = False
        dense_scores = []
        lexical_dense_agreement = False
        for row in rows:
            overlap = int(row.get("_overlap", 0) or 0)
            dense = float(row.get("_dense_score", 0.0) or 0.0)
            best_overlap = max(best_overlap, overlap)
            best_dense = max(best_dense, dense)
            dense_scores.append(dense)
            strong_lexical |= overlap >= cls.ADEQUACY_STRONG_OVERLAP
            lexical_dense_agreement |= (
                overlap > 0 and dense >= cls.ADEQUACY_WEAK_DENSE_FLOOR
            )

        dense_scores.sort(reverse=True)
        dense_margin = (
            dense_scores[0] - dense_scores[1]
            if len(dense_scores) > 1
            else (dense_scores[0] if dense_scores else 0.0)
        )
        strong_dense = (
            best_dense >= cls.ADEQUACY_STRONG_DENSE_FLOOR
            and (len(dense_scores) == 1 or dense_margin >= cls.ADEQUACY_DENSE_MARGIN)
        )

        if strong_lexical:
            reason = "strong_lexical"
        elif strong_dense:
            reason = "strong_dense_margin"
        elif lexical_dense_agreement:
            reason = "lexical_dense_agreement"
        else:
            reason = "weak_only"
        return {
            "adequate": reason != "weak_only",
            "reason": reason,
            "best_overlap": best_overlap,
            "best_dense": round(best_dense, 6),
            "dense_margin": round(dense_margin, 6),
            "scored": True,
        }

    @classmethod
    def _retrieval_pool_is_adequate(
        cls, rows: Sequence[Dict[str, Any]]
    ) -> bool:
        return bool(cls._retrieval_pool_adequacy(rows).get("adequate"))

    def _progressive_stats(self) -> Dict[str, Any]:
        stats = getattr(self, "_progressive_retrieval_stats", None)
        if not isinstance(stats, dict):
            stats = {
                "version": self.PROGRESSIVE_RETRIEVAL_VERSION,
                "hot_adequacy_checks": 0,
                "cold_expansions": 0,
                "fusion_primary_checks": 0,
                "fusion_aux_views_opened": 0,
                "fusion_aux_candidate_additions": 0,
                "fusion_candidate_world_peak": 0,
            }
            self._progressive_retrieval_stats = stats
        return stats

    def _progressive_candidate_admissible(self, row: Dict[str, Any]) -> bool:
        """Use the old weak-hit semantics only as an admission floor, never STOP."""
        checker = getattr(self, "_fusion_has_hit", None)
        if callable(checker):
            return bool(checker([row]))
        return bool(row)

    def _progressive_rank_quality(self, row: Dict[str, Any]) -> float:
        """Downweight weak-only rows so strong auxiliary evidence can survive fusion."""
        adequacy = self._retrieval_pool_adequacy([row])
        if adequacy["adequate"]:
            return 1.0
        return (
            self.ADEQUACY_WEAK_RANK_FACTOR
            if self._progressive_candidate_admissible(row)
            else 0.0
        )

    @staticmethod
    def _progressive_ordered_aux_kinds(
        views: Dict[str, Dict[str, Any]],
        primary_kind: str,
        preferred: Iterable[str],
    ) -> List[str]:
        ordered: List[str] = []
        for kind in preferred:
            if kind != primary_kind and kind in views and kind not in ordered:
                ordered.append(kind)
        for kind in views:
            if kind != primary_kind and kind not in ordered:
                ordered.append(kind)
        return ordered

    def _search_family_hot_first(
        self,
        query: str,
        top_k: int,
        frame,
        *,
        include_history: bool = False,
    ):
        """Open COLD when HOT is merely related rather than retrieval-adequate."""
        limit = max(1, int(top_k))
        hot_ids = self._retrieval_tier_ids(
            frame, "HOT", include_history=include_history
        )
        hot = (
            self._hybrid_search(
                query,
                top_k=min(limit, len(hot_ids)),
                candidate_ids=hot_ids,
            )
            if hot_ids
            else []
        )
        adequacy = self._retrieval_pool_adequacy(hot)
        stats = self._progressive_stats()
        stats["hot_adequacy_checks"] = int(stats.get("hot_adequacy_checks", 0)) + 1
        stats["last_hot_adequacy"] = deepcopy(adequacy)
        if adequacy["adequate"]:
            return [self._snapshot(memory) for memory in hot[:limit]]

        cold_ids = self._retrieval_tier_ids(
            frame, "COLD", include_history=include_history
        )
        cold = (
            self._hybrid_search(
                query,
                top_k=min(limit, len(cold_ids)),
                candidate_ids=cold_ids,
            )
            if cold_ids
            else []
        )
        stats["cold_expansions"] = int(stats.get("cold_expansions", 0)) + 1
        if not cold:
            return [self._snapshot(memory) for memory in hot[:limit]]

        candidate_ids = {
            str(memory.get("id") or "")
            for memory in [*hot, *cold]
            if memory.get("id")
        }
        ranked = (
            list(
                self._hybrid_search(
                    query,
                    top_k=min(limit, len(candidate_ids)),
                    candidate_ids=candidate_ids,
                )
                or []
            )
            if candidate_ids
            else []
        )

        # Preserve deterministic fallbacks if a backend omits an already retrieved ID
        # during the restricted rerank.
        output, seen = [], set()
        for memory in [*ranked, *cold, *hot]:
            memory_id = str(memory.get("id") or "")
            if memory_id and memory_id not in seen:
                output.append(self._snapshot(memory))
                seen.add(memory_id)
            if len(output) >= limit:
                break
        return output

    def _fusion_multiview(self, operation, outputs, seeds, frame):
        """Acquire a bounded candidate world, then use every view only for ranking.

        The obligation view remains the semantic primary. If it is adequate, the old
        efficient behavior is preserved: other views cannot add IDs. If it is weak,
        authorized auxiliary views may add only a small number of novel IDs. Acquisition
        stops as soon as the candidate world becomes adequate; remaining views are
        restricted reranks over that world.
        """
        views = self._fusion_views(operation.get("retrieval_views") or [])
        if not views:
            return super()._fusion_multiview(operation, outputs, seeds, frame)

        rid = str(operation.get("requirement_id") or "")
        top_k = max(1, int(operation.get("top_k", 6) or 6))
        candidate_cap = min(
            self.FUSION_CANDIDATE_HARD_CAP,
            max(top_k, top_k * self.FUSION_CANDIDATE_MULTIPLIER),
        )
        primary = views.get("obligation") or next(iter(views.values()))
        primary_kind, primary_query = primary["kind"], primary["query"]
        primary_rows, relations, evidence_refs = self._fusion_child(
            operation, primary_query, outputs, seeds, frame
        )
        primary_rows = list(primary_rows or [])
        primary_ids = {
            str(memory.get("id") or "")
            for memory in primary_rows
            if memory.get("id")
        }
        primary_viable = self._fusion_has_hit(primary_rows)
        primary_adequacy = self._retrieval_pool_adequacy(primary_rows)

        stats = self._progressive_stats()
        stats["fusion_primary_checks"] = int(stats.get("fusion_primary_checks", 0)) + 1

        by_id: Dict[str, Dict[str, Any]] = {}
        admission_order: Dict[str, int] = {}
        for memory in primary_rows:
            memory_id = str(memory.get("id") or "")
            if memory_id and memory_id not in by_id:
                admission_order[memory_id] = len(admission_order) + 1
                by_id[memory_id] = deepcopy(memory)

        physical_rows: Dict[str, List[Dict[str, Any]]] = {
            primary_kind: primary_rows
        }
        opened_aux: List[str] = []
        additions_by_view: Dict[str, List[str]] = {}
        rescue_view = ""
        zero_hit_rescue_counted = False

        if not primary_adequacy["adequate"]:
            aux_kinds = self._progressive_ordered_aux_kinds(
                views,
                primary_kind,
                getattr(self, "FUSION_RESCUE_ORDER", ("keys", "family", "question")),
            )
            for kind in aux_kinds:
                if len(by_id) >= candidate_cap:
                    break
                view = views[kind]
                rows, child_relations, child_evidence = self._fusion_child(
                    operation, view["query"], outputs, seeds, frame
                )
                rows = list(rows or [])
                physical_rows[kind] = rows
                opened_aux.append(kind)
                relations = self._merge_fusion(relations, child_relations)
                evidence_refs = self._fusion_unique(
                    [*(evidence_refs or []), *(child_evidence or [])]
                )

                if not primary_viable and not zero_hit_rescue_counted:
                    self._fusion_stats["zero_hit_rescue_searches"] = int(
                        self._fusion_stats.get("zero_hit_rescue_searches", 0)
                    ) + 1
                    zero_hit_rescue_counted = True
                if not rescue_view and rows:
                    rescue_view = kind

                added: List[str] = []
                for memory in rows:
                    memory_id = str(memory.get("id") or "")
                    if (
                        not memory_id
                        or memory_id in by_id
                        or not self._progressive_candidate_admissible(memory)
                    ):
                        continue
                    admission_order[memory_id] = len(admission_order) + 1
                    by_id[memory_id] = deepcopy(memory)
                    added.append(memory_id)
                    if (
                        len(added) >= self.FUSION_AUX_NOVEL_PER_VIEW
                        or len(by_id) >= candidate_cap
                    ):
                        break
                additions_by_view[kind] = added
                if added and self._retrieval_pool_is_adequate(list(by_id.values())):
                    break

        ids = list(by_id)
        score = {memory_id: 0.0 for memory_id in ids}
        bound = {memory_id: [] for memory_id in ids}
        coverage: Dict[str, List[str]] = {}
        aux_reranks = 0

        for kind, view in views.items():
            if kind in physical_rows:
                ranked = [
                    deepcopy(memory)
                    for memory in physical_rows[kind]
                    if str(memory.get("id") or "") in score
                ]
            else:
                ranked = self._fusion_rerank(view["query"], ids)
                if kind != primary_kind:
                    aux_reranks += 1
                    self._fusion_stats["auxiliary_searches_avoided"] = int(
                        self._fusion_stats.get("auxiliary_searches_avoided", 0)
                    ) + 1
            coverage[kind] = [
                str(memory.get("id") or "")
                for memory in ranked
                if str(memory.get("id") or "") in score
            ]
            weight = float(getattr(self, "FUSION_WEIGHTS", {}).get(kind, 0.25))
            for rank, memory in enumerate(ranked, 1):
                memory_id = str(memory.get("id") or "")
                if (
                    memory_id in score
                    and self._progressive_candidate_admissible(memory)
                ):
                    score[memory_id] += (
                        weight
                        * self._progressive_rank_quality(memory)
                        / (float(getattr(self, "FUSION_RRF_K", 60.0)) + rank)
                    )
                    if kind not in bound[memory_id]:
                        bound[memory_id].append(kind)

        ordered = sorted(
            ids,
            key=lambda memory_id: (
                -score[memory_id],
                admission_order.get(memory_id, 10**9),
                memory_id,
            ),
        )[:top_k]
        result, binding_scores, binding_views = [], {}, {}
        for memory_id in ordered:
            memory = deepcopy(by_id[memory_id])
            memory["_binding_score"] = round(score[memory_id], 8)
            memory["_binding_views"] = list(bound[memory_id])
            memory["_fusion_primary_candidate"] = memory_id in primary_ids
            memory["_fusion_candidate_origin"] = (
                primary_kind
                if memory_id in primary_ids
                else next(
                    (
                        kind
                        for kind, added_ids in additions_by_view.items()
                        if memory_id in added_ids
                    ),
                    "auxiliary",
                )
            )
            result.append(memory)
            binding_scores[memory_id] = score[memory_id]
            binding_views[memory_id] = list(bound[memory_id])

        final_adequacy = self._retrieval_pool_adequacy(list(by_id.values()))
        stats["fusion_aux_views_opened"] = int(
            stats.get("fusion_aux_views_opened", 0)
        ) + len(opened_aux)
        stats["fusion_aux_candidate_additions"] = int(
            stats.get("fusion_aux_candidate_additions", 0)
        ) + sum(len(values) for values in additions_by_view.values())
        stats["fusion_candidate_world_peak"] = max(
            int(stats.get("fusion_candidate_world_peak", 0)), len(ids)
        )

        if rid:
            self._last_requirement_view_coverage[rid] = deepcopy(coverage)
            self._last_requirement_binding_scores[rid] = binding_scores
            self._last_requirement_binding_views[rid] = binding_views
            self._fusion_stats.setdefault("requirements", {})[rid] = {
                "primary_view": primary_kind,
                # Compatibility: viable is the old weak-hit signal; new acquisition
                # control is explicitly reported as primary_adequate.
                "primary_viable": primary_viable,
                "primary_adequate": bool(primary_adequacy["adequate"]),
                "primary_adequacy": deepcopy(primary_adequacy),
                "rescue_view": rescue_view if not primary_viable else "",
                "auxiliary_views_opened": list(opened_aux),
                "auxiliary_candidate_additions": deepcopy(additions_by_view),
                "candidate_world_cap": candidate_cap,
                "candidate_count": len(ids),
                "final_count": len(result),
                "adequacy_after_expansion": bool(final_adequacy["adequate"]),
                "final_adequacy": deepcopy(final_adequacy),
                "auxiliary_reranks": aux_reranks,
            }
        return result, relations or [], evidence_refs or []

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
        self._progressive_retrieval_stats = {
            "version": self.PROGRESSIVE_RETRIEVAL_VERSION,
            "hot_adequacy_checks": 0,
            "cold_expansions": 0,
            "fusion_primary_checks": 0,
            "fusion_aux_views_opened": 0,
            "fusion_aux_candidate_additions": 0,
            "fusion_candidate_world_peak": 0,
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
        run["progressive_retrieval"] = deepcopy(self._progressive_retrieval_stats)
        return run

    def prepare_batch_query(self, question, system_message=None, **kwargs):
        prepared = super().prepare_batch_query(
            question, system_message=system_message, **kwargs
        )
        extra = prepared.setdefault("extra", {})
        extra["progressive_retrieval_version"] = self.PROGRESSIVE_RETRIEVAL_VERSION
        extra["progressive_retrieval"] = deepcopy(
            getattr(self, "_progressive_retrieval_stats", {}) or {}
        )
        return prepared
