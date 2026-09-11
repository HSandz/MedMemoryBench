from methods.smart_mem0.agent import SmartMem0Agent
from methods.smart_mem0.read_reasoning_completion import ReadReasoningCompletionMixin
from methods.smart_mem0.read_stable_semantic_runtime import (
    ReadStableSemanticRuntimeMixin,
    STABLE_SEMANTIC_SCHEMA,
)


class _StableBase:
    def __init__(self):
        self.proof_calls = 0
        self.recovery_slots = []

    @staticmethod
    def _rc_text(value):
        return " ".join(str(value or "").replace("_", " ").casefold().split())

    @staticmethod
    def _rc_question_span(value, question):
        value = " ".join(str(value or "").split())
        return value if value and value.casefold() in str(question).casefold() else ""

    @staticmethod
    def _question_stem(question):
        return str(question or "").strip()

    @staticmethod
    def _rg_sparse_selector(value):
        if not isinstance(value, dict):
            return {}
        relation = str(value.get("relation") or "").upper()
        if relation == "CURRENT":
            return {"relation": "CURRENT"}
        if relation not in {
            "LOCATE", "EARLIEST", "LATEST", "EXACT", "BEFORE", "AFTER", "BETWEEN"
        }:
            return {}
        axis = str(value.get("axis") or "").lower()
        if axis not in {
            "event_time", "document_time", "origin_document_time", "effective_event_time"
        }:
            return {}
        result = {"relation": relation, "axis": axis}
        if value.get("anchor"):
            result["anchor"] = str(value["anchor"])
        if value.get("end"):
            result["end"] = str(value["end"])
        if relation in {"EXACT", "BEFORE", "AFTER"} and not result.get("anchor"):
            return {}
        if relation == "BETWEEN" and not (result.get("anchor") and result.get("end")):
            return {}
        return result

    @staticmethod
    def _slot_structure_covered(slot, support_ids, selected, relations):
        del slot, selected, relations
        return bool(support_ids)

    def _requirement_target_proof(self, slot, memory):
        del slot
        self.proof_calls += 1
        return memory.get("id") == "m1"

    def _prepare_requirement_context_state(self, run, initial_seeds):
        del initial_seeds
        # Emulate the historical destructive proof owner.
        run["requirement_context_candidates"] = {"r1": ["m1", "m2"]}
        run["requirement_proof_support"] = {"r1": ["m1"]}
        run["slot_support"]["r1"] = ["m1"]
        return run

    def _make_deterministic_recovery_plan(self, missing_slots, question, existing_plan):
        del question, existing_plan
        self.recovery_slots = [slot["id"] for slot in missing_slots]
        return {"required_slots": list(missing_slots)} if missing_slots else None


class _Harness(ReadStableSemanticRuntimeMixin, _StableBase):
    pass


def test_stable_runtime_is_active_and_reasoning_completion_is_not():
    mro = SmartMem0Agent.mro()
    assert ReadStableSemanticRuntimeMixin in mro
    assert ReadReasoningCompletionMixin not in mro
    assert mro.index(ReadStableSemanticRuntimeMixin) < mro.index(
        next(cls for cls in mro if cls.__name__ == "ReadRequirementGraphMixin")
    )


def test_controller_schema_has_no_memory_or_retrieval_authority():
    lowered = STABLE_SEMANTIC_SCHEMA.casefold()
    assert "top-3" not in lowered
    assert "seed" not in lowered
    assert "terminal_candidate" not in lowered
    assert "proof_spec" not in lowered
    assert "retrieval operation" not in lowered
    assert '"aliases"' not in lowered
    assert '"anchors"' not in lowered


def test_sanitizer_canonicalizes_ids_merges_exact_duplicates_and_drops_weak_infer():
    harness = _Harness()
    raw = {
        "projection": "VALUE",
        "requirements": [
            {
                "need": "account balance",
                "question_span": "account balance",
                "selector": {"relation": "LATEST", "axis": "document_time"},
                "aliases": ["ignored"],
            },
            {
                "need": "account balance",
                "question_span": "account balance",
                "selector": {"relation": "LATEST", "axis": "document_time"},
            },
        ],
        "relations": [
            {"type": "INFER", "from": 1, "to": "ANSWER"},
            {"type": "COMPARE", "from": 1, "to": 2},
        ],
        "terminal_candidate": {"answer": "ignored"},
    }
    sanitized = harness._stable_sanitize_controller_ir(
        raw, "What is the latest account balance?", {}
    )
    assert [item["id"] for item in sanitized["requirements"]] == ["r1"]
    assert len(sanitized["requirements"]) == 1
    assert sanitized["requirements"][0]["need"] == "account balance"
    assert sanitized["links"] == []
    assert "terminal_candidate" not in sanitized
    assert "aliases" not in sanitized["requirements"][0]


def test_infer_requires_explicit_bridge_goal():
    harness = _Harness()
    raw = {
        "projection": "TEXT",
        "requirements": [{"need": "stored observation"}],
        "relations": [
            {
                "type": "INFER",
                "from": 1,
                "to": "ANSWER",
                "bridge_goal": "apply the external rule to the grounded observation",
            }
        ],
    }
    sanitized = harness._stable_sanitize_controller_ir(
        raw, "What follows from the stored observation?", {}
    )
    assert sanitized["links"] == [
        {
            "type": "INFER",
            "from": "r1",
            "to": "ANSWER",
            "bridge_goal": "apply the external rule to the grounded observation",
        }
    ]


def test_retrieval_views_are_need_first_without_alias_or_full_question_view():
    harness = _Harness()
    harness._rg_unique_text = lambda values, limit=8: list(
        dict.fromkeys(str(value) for value in values if value)
    )[:limit]
    views = harness._er_requirement_views(
        {
            "need": "account balance",
            "resolved_keys": ["balance"],
            "question_span": "account balance",
        },
        "What is the latest account balance?",
    )
    assert [item["kind"] for item in views] == ["need", "keys"]
    assert views[0]["query"] == "account balance"
    assert all(item["kind"] not in {"aliases", "question"} for item in views)


def test_proof_is_memoized_per_requirement_memory_pair():
    harness = _Harness()
    slot = {"id": "r1"}
    memory = {"id": "m1"}
    assert harness._requirement_target_proof(slot, memory) is True
    assert harness._requirement_target_proof(slot, memory) is True
    assert harness.proof_calls == 1
    assert harness._stable_proof_cache_hits == 1


def test_proof_checkpoint_restores_structural_support_and_does_not_prune():
    harness = _Harness()
    run = {
        "plan": {"required_slots": [{"id": "r1"}]},
        "slot_support": {"r1": ["m1", "m2"]},
        "operation_candidates": [{"id": "m1"}, {"id": "m2"}],
        "beliefs": [],
        "planning_seeds": [],
        "relations": [],
    }
    prepared = harness._prepare_requirement_context_state(run, [])
    assert prepared["slot_support"]["r1"] == ["m1", "m2"]
    assert prepared["requirement_proof_support"]["r1"] == ["m1"]
    checkpoint = prepared["proof_checkpoint"]
    assert checkpoint["candidate_world_unchanged"] is True
    assert checkpoint["structural_support_restored"] is True
    assert checkpoint["retrieval_authority"] is False
    assert checkpoint["pruning_authority"] is False


def test_recovery_filters_out_relation_only_or_proof_only_gaps():
    harness = _Harness()
    harness._stable_last_requirement_status = {"r1": "FOUND", "r2": "EMPTY"}
    result = harness._make_deterministic_recovery_plan(
        [{"id": "r1"}, {"id": "r2"}], "question", {}
    )
    assert harness.recovery_slots == ["r2"]
    assert [slot["id"] for slot in result["required_slots"]] == ["r2"]

    harness._stable_last_requirement_status = {"r1": "FOUND"}
    assert harness._make_deterministic_recovery_plan(
        [{"id": "r1"}], "question", {}
    ) is None
