from methods.smart_mem0.read_retrieval_fusion import ReadRetrievalFusionMixin


class _BaseHarness:
    def __init__(self):
        self._memories = []
        self._relations = []
        self._last_requirement_view_coverage = {}
        self._last_requirement_binding_scores = {}
        self._last_requirement_binding_views = {}
        self._last_option_probe_coverage = {}
        self._last_proposition_probe_coverage = {}
        self._last_candidate_local_coverage = {}
        self._last_candidate_shared_context_ids = []
        self._fusion_stats = {
            "version": "test",
            "requirements": {},
            "auxiliary_searches_avoided": 0,
            "zero_hit_rescue_searches": 0,
            "bridge_relation_count": 0,
            "stored_bridge_relation_materialized": False,
            "bridge_relations": [],
        }

    @staticmethod
    def _snapshot(value):
        return dict(value)

    @staticmethod
    def _retrieval_has_semantic_hit(rows):
        return any(
            int(row.get("_overlap", 0) or 0) > 0
            or float(row.get("_dense_score", 0.0) or 0.0) >= 0.36
            for row in rows
        )

    def _hybrid_search(self, query, top_k, candidate_ids=None):
        candidate_ids = set(candidate_ids) if candidate_ids is not None else None
        query_terms = set(str(query or "").lower().replace("_", " ").split())
        output = []
        for memory in self._memories:
            if candidate_ids is not None and memory["id"] not in candidate_ids:
                continue
            memory_terms = set(
                (str(memory.get("claim") or "") + " " + str(memory.get("value") or ""))
                .lower()
                .replace("_", " ")
                .split()
            )
            overlap = len(query_terms & memory_terms)
            output.append(
                dict(
                    memory,
                    _overlap=overlap,
                    _dense_score=0.5 if overlap else 0.1,
                )
            )
        return sorted(
            output,
            key=lambda row: (
                int(row.get("_overlap", 0)),
                float(row.get("_dense_score", 0.0)),
                str(row.get("id") or ""),
            ),
            reverse=True,
        )[:top_k]

    def _execute_operation(self, operation, outputs, seeds, frame=None):
        rows = {
            "missed insulin doses": [
                {
                    "id": "m_good",
                    "claim": "missed insulin doses during chaos",
                    "_overlap": 3,
                    "_dense_score": 0.7,
                },
                {
                    "id": "m_other",
                    "claim": "other insulin note",
                    "_overlap": 1,
                    "_dense_score": 0.4,
                },
            ],
            "insulin_regimen_adherence": [
                {
                    "id": "m_family",
                    "claim": "general medication history",
                    "_overlap": 1,
                    "_dense_score": 0.4,
                },
                {
                    "id": "m_good",
                    "claim": "missed insulin doses during chaos",
                    "_overlap": 1,
                    "_dense_score": 0.5,
                },
            ],
            "insulin adherence": [
                {
                    "id": "m_good",
                    "claim": "missed insulin doses during chaos",
                    "_overlap": 2,
                    "_dense_score": 0.6,
                }
            ],
            "full question symptoms and missed insulin": [
                {
                    "id": "m_symptom",
                    "claim": "morning nausea",
                    "_overlap": 2,
                    "_dense_score": 0.6,
                }
            ],
            "weak obligation": [],
            "specific key": [
                {
                    "id": "m_rescue",
                    "claim": "specific key fact",
                    "_overlap": 2,
                    "_dense_score": 0.6,
                }
            ],
        }
        return [dict(item) for item in rows.get(operation.get("query"), [])], [], []

    def _compile_gap_operations(self, slots, question, budget_tier="MEDIUM", plan=None):
        del question, budget_tier, plan
        return [
            {
                "op": "SEARCH_FAMILY",
                "query": "legacy",
                "top_k": 4,
                "family_mode": "semantic",
                "produces": [slot["id"]],
                "requirement_id": slot["id"],
                "retrieval_views": list(slot.get("retrieval_views") or []),
            }
            for slot in slots
        ]

    def _semantic_operation_search(
        self, query, top_k, strategy, frame=None, option_queries=None
    ):
        del query, strategy, frame, option_queries
        self._last_option_probe_coverage = {"A": ["m_a2", "m_a1"], "B": ["m_b1"]}
        self._last_proposition_probe_coverage = dict(self._last_option_probe_coverage)
        return [
            dict(memory)
            for memory in self._memories
            if memory["id"] in {"m_a1", "m_a2", "m_b1"}
        ][:top_k]

    def _authorize_controller_answer(self, ir, seeds, frame):
        del ir, frame
        return [seeds[0]], "AUTHORIZED"

    @staticmethod
    def _terminal_memory_text(memory):
        return str(memory.get("claim") or "")

    @staticmethod
    def _terminal_surface_similarity(left, right):
        left_terms = set(str(left or "").lower().split())
        right_terms = set(str(right or "").lower().split())
        return len(left_terms & right_terms) / max(1, len(left_terms))

    @staticmethod
    def _rc_token_sequence_present(left, right):
        return str(left or "").lower() in str(right or "").lower()

    @staticmethod
    def _valid_causal_relation(relation, by_id):
        del by_id
        return bool(relation.get("valid"))

    def _run_query_retrieval(self, *args, **kwargs):
        del args, kwargs
        return {}

    def prepare_batch_query(self, question, system_message=None, **kwargs):
        del question, system_message
        return {
            "retrieved_memories": kwargs.get("retrieved_memories", []),
            "messages": [{"role": "system", "content": "base"}],
            "extra": {},
        }


class _Harness(ReadRetrievalFusionMixin, _BaseHarness):
    pass


def _multiview_operation(obligation="missed insulin doses"):
    return {
        "op": "SEARCH_FAMILY",
        "top_k": 4,
        "requirement_id": "r1",
        "retrieval_views": [
            {"kind": "obligation", "query": obligation},
            {"kind": "family", "query": "insulin_regimen_adherence"},
            {"kind": "keys", "query": "insulin adherence"},
            {"kind": "question", "query": "full question symptoms and missed insulin"},
        ],
    }


def test_primary_recall_owns_candidate_universe_and_auxiliary_views_only_rerank():
    harness = _Harness()
    harness._memories = [
        {"id": "m_good", "claim": "missed insulin doses during chaos"},
        {"id": "m_other", "claim": "other insulin note"},
        {"id": "m_family", "claim": "general medication history"},
        {"id": "m_symptom", "claim": "morning nausea"},
    ]
    rows, _, _ = harness._execute_operation(_multiview_operation(), [], [], None)
    assert [row["id"] for row in rows] == ["m_good", "m_other"]
    assert "m_family" not in harness._last_requirement_binding_scores["r1"]
    assert harness._fusion_stats["auxiliary_searches_avoided"] >= 1


def test_weak_primary_opens_at_most_one_deterministic_rescue_view():
    harness = _Harness()
    harness._memories = [{"id": "m_rescue", "claim": "specific key fact"}]
    operation = {
        "op": "SEARCH_FAMILY",
        "top_k": 4,
        "requirement_id": "r1",
        "retrieval_views": [
            {"kind": "obligation", "query": "weak obligation"},
            {"kind": "keys", "query": "specific key"},
            {"kind": "family", "query": "generic family"},
        ],
    }
    rows, _, _ = harness._execute_operation(operation, [], [], None)
    assert [row["id"] for row in rows] == ["m_rescue"]
    assert harness._fusion_stats["zero_hit_rescue_searches"] == 1


def test_candidateset_local_order_is_reranked_by_candidate_proposition():
    harness = _Harness()
    harness._memories = [
        {"id": "m_a1", "claim": "alpha exact proposition"},
        {"id": "m_a2", "claim": "generic alpha context"},
        {"id": "m_b1", "claim": "beta exact"},
    ]
    rows = harness._semantic_operation_search(
        "shared",
        3,
        "SHARED_OPTIONS",
        option_queries=[
            {"label": "A", "query": "alpha exact proposition"},
            {"label": "B", "query": "beta exact"},
        ],
    )
    assert harness._last_candidate_local_coverage["A"][0] == "m_a1"
    assert rows[0]["id"] == "m_a1"


def test_possible_cause_compiles_only_one_bounded_stored_relation_expansion():
    harness = _Harness()
    slots = [
        {
            "id": "r1",
            "retrieval_views": [{"kind": "obligation", "query": "cause"}],
        },
        {
            "id": "r2",
            "retrieval_views": [{"kind": "obligation", "query": "effect"}],
            "target_surface": "effect",
        },
    ]
    operations = harness._compile_gap_operations(
        slots,
        "q",
        "MEDIUM",
        plan={
            "semantic_relations": [
                {"type": "POSSIBLE_CAUSE", "from": "r1", "to": "r2"}
            ]
        },
    )
    expansion = operations[-1]
    assert expansion["op"] == "EXPAND_RELATION"
    assert expansion["max_hops"] == 1
    assert sum(bool(op.get("bridge_bound_expansion")) for op in operations) == 1


def test_stored_bridge_uses_obligation_bound_source_and_validated_one_hop_edge():
    harness = _Harness()
    harness._memories = [
        {"id": "s", "claim": "cause"},
        {"id": "t", "claim": "effect target"},
        {"id": "bad", "claim": "other"},
    ]
    harness._relations = [
        {"type": "CAUSES", "source_id": "s", "target_id": "t", "valid": True},
        {"type": "CAUSES", "source_id": "s", "target_id": "bad", "valid": False},
    ]
    harness._last_requirement_binding_scores = {"r1": {"s": 0.9}}
    harness._last_requirement_binding_views = {"r1": {"s": ["obligation"]}}
    rows, relations, _ = harness._execute_operation(
        {
            "op": "FOLLOW_CAUSES",
            "_lean_op": "EXPAND_RELATION",
            "bridge_bound_expansion": True,
            "source_requirement": "r1",
            "goal": "effect target",
        },
        [],
        [],
        None,
    )
    assert [row["id"] for row in rows] == ["s", "t"]
    assert len(relations) == 1


def test_certified_direct_falls_back_when_seed_does_not_close_requested_predicate():
    harness = _Harness()
    seed = {"id": "m1", "claim": "blurred vision was reported"}
    supports, reason = harness._authorize_controller_answer(
        {
            "requirements": [{"id": "r1", "target": "fatigue after standing"}],
            "candidate": {"answer": "blurred vision", "support_ref": "$seed0"},
        },
        [seed],
        None,
    )
    assert supports is None
    assert reason == "DIRECT_OBLIGATION_NOT_CLOSED"


def test_stored_relation_is_materialized_only_when_both_endpoints_survive_proof_context():
    harness = _Harness()
    harness._fusion_stats = {
        "version": harness.RETRIEVAL_FUSION_VERSION,
        "requirements": {},
        "auxiliary_searches_avoided": 0,
        "zero_hit_rescue_searches": 0,
        "bridge_relation_count": 1,
        "stored_bridge_relation_materialized": False,
        "bridge_relations": [
            {"type": "CAUSES", "source_id": "s", "target_id": "t"}
        ],
    }
    prepared = harness.prepare_batch_query(
        "q",
        retrieved_memories=[
            {"id": "s", "claim": "cause"},
            {"id": "t", "claim": "effect"},
        ],
    )
    assert prepared["extra"]["retrieval_fusion"]["stored_bridge_relation_materialized"]
    assert "STORED BRIDGE RELATION" in prepared["messages"][0]["content"]
    assert "CAUSES" in prepared["extra"]["relations_used"]
