"""One accounting owner for synchronous and externally batched READ calls."""


def record_read_usage(extra):
    controller = int(bool((extra.get("semantic_controller") or {}).get("called")))
    answer = int(bool(extra.get("answer_llm_called")))
    middle = sum(
        1
        for event in extra.get("slot_validation", [])
        if event.get("called") and not event.get("cache_hit")
    )
    total = controller + answer + middle
    terminal = bool(extra.get("precomputed_answer_present"))
    extra["method_llm_calls"] = {
        "controller": controller,
        "answer": answer,
        "middle": middle,
        "total": total,
        "excludes_evaluator_judges": True,
        "two_stage_budget_violation": total > 2
        or bool(middle)
        or bool(terminal and answer),
        "count_semantics": "observed_method_invocations_excluding_transport_retries",
    }
    extra["answer_llm_planned"] = not terminal
    extra["two_stage_llm_budget"] = {
        "max_calls_per_query": 2,
        "violation": extra["method_llm_calls"]["two_stage_budget_violation"],
    }
    if "second_call" in extra:
        extra["second_call"]["called"] = bool(answer)
    if "two_stage_audit" in extra:
        extra["two_stage_audit"].update(
            controller_calls=controller, answer_calls=answer, total_calls=total
        )


class ReadUsageContractMixin:
    """Import compatibility for historical harnesses; not in the agent MRO."""

    def prepare_batch_query(self, question, system_message=None, **kwargs):
        prepared = super().prepare_batch_query(
            question, system_message=system_message, **kwargs
        )
        extra = prepared.setdefault("extra", {})
        extra["precomputed_answer_present"] = prepared.get(
            "precomputed_answer"
        ) not in (None, "")
        record_read_usage(extra)
        return prepared
