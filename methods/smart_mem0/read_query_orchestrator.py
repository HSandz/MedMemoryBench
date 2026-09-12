"""Strict two-stage READ orchestration and LLM-call budget enforcement."""

from copy import deepcopy
from .read_usage_contract import record_read_usage


class ReadQueryOrchestratorMixin:
    QUERY_ORCHESTRATOR_VERSION = "two-stage-orchestrator-v1"
    MAX_READ_LLM_CALLS = 2
    MIDDLE_TOKEN_STAGES = ("fast_gate", "planner", "slot_validation", "replan")

    @classmethod
    def _two_stage_audit(cls, extra, *, terminal=None, answer_called=None):
        tokens = dict((extra or {}).get("query_tokens") or {})
        controller = dict((extra or {}).get("semantic_controller") or {})
        middle_tokens = {
            stage: int(tokens.get(stage, 0) or 0) for stage in cls.MIDDLE_TOKEN_STAGES
        }
        validation_calls = sum(
            1
            for item in ((extra or {}).get("slot_validation") or [])
            if isinstance(item, dict)
            and item.get("called")
            and not item.get("cache_hit")
        )
        controller_calls = int(bool(controller.get("called")))
        if terminal is None:
            terminal = bool((extra or {}).get("precomputed_answer_present"))
        if answer_called is None:
            answer_called = bool((extra or {}).get("answer_llm_called"))
        answer_calls = int(bool(answer_called))
        total_calls = controller_calls + validation_calls + answer_calls
        violations = []
        if controller_calls != 1:
            violations.append("SEMANTIC_CONTROLLER_CALL_COUNT")
        if any(middle_tokens.values()) or validation_calls:
            violations.append("MIDDLE_LLM_ACTIVITY")
        if terminal and answer_calls:
            violations.append("TERMINAL_ANSWER_REGENERATED")
        if total_calls > cls.MAX_READ_LLM_CALLS:
            violations.append("READ_LLM_CALL_BUDGET")
        return {
            "version": cls.QUERY_ORCHESTRATOR_VERSION,
            "controller_calls": controller_calls,
            "middle_tokens": middle_tokens,
            "slot_validation_calls": validation_calls,
            "answer_calls": answer_calls,
            "terminal": bool(terminal),
            "total_calls": total_calls,
            "max_calls": cls.MAX_READ_LLM_CALLS,
            "violations": violations,
            "valid": not violations,
        }

    @classmethod
    def _assert_two_stage_audit(cls, extra, *, terminal=None, answer_called=None):
        audit = cls._two_stage_audit(
            extra, terminal=terminal, answer_called=answer_called
        )
        if not audit["valid"]:
            raise AssertionError(
                "SmartMem0 two-stage READ invariant violated: "
                + ",".join(audit["violations"])
            )
        return audit

    def prepare_batch_query(self, question, system_message=None, **kwargs):
        if not getattr(self, "enable_two_stage_controller", False):
            raise RuntimeError(
                "SmartMem0 READ supports only the two-stage controller architecture"
            )
        prepared = super().prepare_batch_query(
            question, system_message=system_message, **kwargs
        )
        extra = prepared.setdefault("extra", {})
        terminal = prepared.get("precomputed_answer") not in (None, "")
        extra["precomputed_answer_present"] = terminal
        audit = self._assert_two_stage_audit(
            extra, terminal=terminal, answer_called=False
        )
        extra["query_orchestrator_version"] = self.QUERY_ORCHESTRATOR_VERSION
        extra["two_stage_audit"] = deepcopy(audit)
        record_read_usage(extra)
        return prepared

    def finalize_batch_query(self, prepared, content):
        result = super().finalize_batch_query(prepared, content)
        record_read_usage(result.extra)
        return result

    def generate_prepared_batch_answer(self, prepared):
        extra = prepared.setdefault("extra", {})
        terminal = prepared.get("precomputed_answer") not in (None, "")
        preflight = self._assert_two_stage_audit(
            extra, terminal=terminal, answer_called=False
        )
        extra["two_stage_audit"] = deepcopy(preflight)
        result = super().generate_prepared_batch_answer(prepared)
        final_extra = result.extra
        final_audit = self._assert_two_stage_audit(
            final_extra,
            terminal=terminal,
            answer_called=bool(final_extra.get("answer_llm_called")),
        )
        final_extra["two_stage_audit"] = deepcopy(final_audit)
        final_extra["query_orchestrator_version"] = self.QUERY_ORCHESTRATOR_VERSION
        record_read_usage(final_extra)
        return result
