from methods.smart_mem0.read_progressive_retrieval import ReadProgressiveRetrievalMixin
from methods.smart_mem0.read_retrieval_fusion import ReadRetrievalFusionMixin


class _BaseHarness:
    def __init__(self):
        self._memories = []
        self._last_requirement_view_coverage = {}
        self._last_requirement_binding_scores = {}
        self._last_requirement_binding_views = {}
        self._fusion_stats = {
            "version": "test",
            "requirements": {},
            "auxiliary_searches_avoided": 0,
            "zero_hit_rescue_searches": 0,
            "bridge_relation_count": 0,
            "stored_bridge_relation_materialized": False,
            "bridge_relations": [],
        }
        self.calls = []

    @staticmethod
    def _snapshot(value):
        return dict(value)

    @staticmethod
    def _retrieval_has_semantic_hit(rows):
        if not rows:
            return False
        scored = False
        for row in rows:
            scored |= "_overlap" in row or "_dense_score" in row
            if int(row.get("_overlap", 0) or 0) > 0:
                return True
            if float(row.get("_dense_score", 0.0) or 0.0) >= 0.36:
                return True
        return not scored

    def _hybrid_search(self, query, top_k, candidate_ids=None):
        candidate_ids = set(candidate_ids) if candidate_ids is not None else None
        rows = [
            dict(memory)
            for memory in self._memories
            if candidate_ids is None or memory["id"] in candidate_ids
        ]
        return sorted(
            rows,
            key=lambda row: (
                int(row.get("_overlap", 0) or 0),
                float(row.get("_dense_score", 0.0) or 0.0),
                str(row.get("id") or ""),
            ),
            reverse=True,
        )[:top_k]

    def _retrieval_tier_ids(self, frame, tier, include_history=False):
        del frame, include_history
        return {
            memory["id"]
            for memory in self._memories
            if memory.get("tier") == tier
        }

    def _execute_operation(self, operation, outputs, seeds, frame=None):
        del outputs, seeds, frame
        query = operation.get("query")
        self.calls.append(query)
        rows = {
            "primary strong": [
                {"id": "p", "claim": "primary", "_overlap": 2, "_dense_score": 0.2}
            ],
            "primary weak": [
                {"id": "w", "claim": "weak", "_overlap": 1, "_dense_score": 0.2}
            ],
            "specific key": [
                {"id": "k", "claim": "key", "_overlap": 2, "_dense_score": 0.6}
            ],
            "family": [
                {"id": "f", "claim": "family", "_overlap": 2, "_dense_score": 0.6}
            ],
            "question": [
                {"id": "q", "claim": "question", "_overlap": 2, "_dense_score": 0.6}
            ],
        }
        return [dict(row) for row in rows.get(query, [])], [], []


class _Harness(
    ReadProgressiveRetrievalMixin,
    ReadRetrievalFusionMixin,
    _BaseHarness,
):
    pass


def _operation(primary):
    return {
        "op": "SEARCH_FAMILY",
        "top_k": 4,
        "requirement_id": "r1",
        "retrieval_views": [
            {"kind": "obligation", "query": primary},
            {"kind": "keys", "query": "specific key"},
            {"kind": "family", "query": "family"},
            {"kind": "question", "query": "question"},
        ],
    }


def test_pool_adequacy_does_not_treat_one_weak_overlap_as_sufficient():
    weak = [{"id": "m", "_overlap": 1, "_dense_score": 0.2}]
    assert _Harness._retrieval_has_semantic_hit(weak)
    adequacy = _Harness._retrieval_pool_adequacy(weak)
    assert not adequacy["adequate"]
    assert adequacy["reason"] == "weak_only"


def test_weak_hot_hit_opens_cold_and_reranks_tiers_together():
    harness = _Harness()
    harness._memories = [
        {"id": "hot", "tier": "HOT", "_overlap": 1, "_dense_score": 0.2},
        {"id": "cold", "tier": "COLD", "_overlap": 3, "_dense_score": 0.7},
    ]
    rows = harness._search_family_hot_first("q", 2, frame=None)
    assert [row["id"] for row in rows] == ["cold", "hot"]
    assert harness._progressive_stats()["cold_expansions"] == 1


def test_strong_hot_pool_stops_before_cold():
    harness = _Harness()
    harness._memories = [
        {"id": "hot", "tier": "HOT", "_overlap": 2, "_dense_score": 0.2},
        {"id": "cold", "tier": "COLD", "_overlap": 4, "_dense_score": 0.9},
    ]
    rows = harness._search_family_hot_first("q", 2, frame=None)
    assert [row["id"] for row in rows] == ["hot"]
    assert harness._progressive_stats()["cold_expansions"] == 0


def test_adequate_primary_keeps_auxiliary_views_as_restricted_reranks():
    harness = _Harness()
    rows, _, _ = harness._execute_operation(_operation("primary strong"), [], [], None)
    assert [row["id"] for row in rows] == ["p"]
    requirement = harness._fusion_stats["requirements"]["r1"]
    assert requirement["primary_adequate"]
    assert requirement["auxiliary_views_opened"] == []
    assert requirement["candidate_count"] == 1


def test_weak_primary_admits_bounded_novel_auxiliary_candidate_then_stops():
    harness = _Harness()
    rows, _, _ = harness._execute_operation(_operation("primary weak"), [], [], None)
    ids = [row["id"] for row in rows]
    assert set(ids) == {"w", "k"}
    assert ids[0] == "k"
    requirement = harness._fusion_stats["requirements"]["r1"]
    assert not requirement["primary_adequate"]
    assert requirement["auxiliary_views_opened"] == ["keys"]
    assert requirement["auxiliary_candidate_additions"] == {"keys": ["k"]}
    assert requirement["candidate_count"] == 2
    assert requirement["adequacy_after_expansion"]
    assert "family" not in harness.calls
    assert "question" not in harness.calls


def test_auxiliary_candidate_budget_is_separate_from_final_top_k():
    harness = _Harness()
    harness.FUSION_AUX_NOVEL_PER_VIEW = 2
    operation = _operation("primary weak")
    operation["top_k"] = 1
    rows, _, _ = harness._execute_operation(operation, [], [], None)
    requirement = harness._fusion_stats["requirements"]["r1"]
    assert requirement["candidate_count"] == 2
    assert len(rows) == 1
