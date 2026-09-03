"""Method-local LLM accounting for SmartMem0 reads.

Benchmark-level trackers also count evaluator judges. This telemetry records
only calls implied by the SmartMem0 prepared query so resource claims can be
audited without subtracting benchmark infrastructure calls afterward.
"""


class ReadUsageContractMixin:
    def prepare_batch_query(self, question, system_message=None, **kwargs):
        prepared = super().prepare_batch_query(
            question, system_message=system_message, **kwargs
        )
        extra = prepared.setdefault("extra", {})
        controller = extra.get("semantic_controller") or {}
        controller_calls = int(bool(controller.get("called")))
        # At this point all output-contract handling has already run. A
        # surviving precomputed answer therefore really skips the final model;
        # otherwise one final answer generation is required.
        answer_calls = 0 if str(prepared.get("precomputed_answer") or "").strip() else 1
        extra["method_llm_calls"] = {
            "controller": controller_calls,
            "answer": answer_calls,
            "total": controller_calls + answer_calls,
            "excludes_evaluator_judges": True,
        }
        return prepared
