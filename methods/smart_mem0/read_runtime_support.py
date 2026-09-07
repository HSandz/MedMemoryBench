"""Deterministic runtime support for the two-stage SmartMem0 READ path.

These helpers used to live beside the legacy LLM gate/planner. Phase 6 separates them so
the active agent can drop PlanningMixin entirely while preserving the same authorization,
frame filtering, usage accounting, and reference-resolution invariants.
"""

import re
from typing import Any, Dict, List, Optional

from .canonicalization import state_identity
from .contracts import QueryFrame


class ReadRuntimeSupportMixin:
    RUNTIME_SUPPORT_VERSION = "two-stage-runtime-support-v1"

    def _response_usage(self, response: Any, prompt: str) -> Dict[str, Any]:
        input_tokens = int(getattr(response, "input_tokens", 0) or 0)
        output_tokens = int(getattr(response, "output_tokens", 0) or 0)
        if not input_tokens:
            input_tokens = len(self._tokenizer.encode(prompt))
        if not output_tokens:
            output_tokens = len(
                self._tokenizer.encode(str(getattr(response, "content", "")))
            )
        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "latency": float(getattr(response, "latency", 0.0) or 0.0),
        }

    def _planning_seed_set(
        self,
        question: str,
        recalled: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Return exactly the top-three RRF-authorized controller seeds."""
        del question
        selected: List[Dict[str, Any]] = []
        for memory in recalled:
            if memory and all(item["id"] != memory["id"] for item in selected):
                selected.append(self._snapshot(memory))
            if len(selected) >= 3:
                break
        return selected

    def _memory_satisfies_frame(
        self,
        memory: Dict[str, Any],
        frame: QueryFrame,
        include_dates: bool = True,
        include_entities: bool = True,
    ) -> bool:
        if (
            include_dates
            and frame.dates
            and not self._memory_matches_dates(memory, frame.dates)
        ):
            return False
        if frame.speaker_role and not self._speaker_role_match(
            memory, frame.speaker_role
        ):
            return False
        if include_entities and frame.hard_entities:
            values = {
                str(value).strip().lower()
                for value in (
                    *memory.get("entities", []),
                    *memory.get("scope_entities", []),
                    memory.get("subject", ""),
                    memory.get("subject_id", ""),
                    memory.get("object_anchor", ""),
                )
                if str(value).strip()
            }
            if not set(frame.hard_entities).intersection(values):
                return False
        return True

    def _has_competing_active_value(self, memory: Dict[str, Any]) -> bool:
        identity = state_identity(memory)
        if not identity:
            return False
        by_id = {item["id"]: item for item in self._memories}
        values = {
            self._state_value_signature(by_id[memory_id])
            for memory_id in self._state_heads.get(identity, [])
            if memory_id in by_id
            and self._belief_status.get(memory_id, "active")
            in {"active", "conflicting"}
            and self._state_value_signature(by_id[memory_id])
        }
        return len(values) > 1

    def _validate_fast_support(
        self,
        reference: Any,
        seeds: List[Dict[str, Any]],
        frame: QueryFrame,
    ) -> Optional[List[Dict[str, Any]]]:
        match = re.fullmatch(r"\$seed(\d+)", str(reference or ""))
        if not match or int(match.group(1)) >= min(3, len(seeds)):
            return None
        memory = seeds[int(match.group(1))]
        status = memory.get(
            "_status", self._belief_status.get(memory["id"], "active")
        )
        if memory.get("assertion_mode", "DIRECT") != "DIRECT":
            return None
        if not self._memory_value(memory) or status in {"superseded", "conflicting"}:
            return None
        if not self._memory_satisfies_frame(memory, frame):
            return None
        if self._has_competing_active_value(memory):
            return None
        support = self._snapshot(memory)
        if self._has_unresolved_conflict([support]):
            return None
        return [support]

    def _resolve_refs(
        self,
        refs: Any,
        outputs: List[List[Dict[str, Any]]],
        seeds: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Resolve only authorized seed/output refs; bare memory IDs never resolve."""
        values = refs if isinstance(refs, list) else [refs]
        resolved: List[Dict[str, Any]] = []
        for value in values:
            text = str(value or "")
            seed_match = re.fullmatch(r"\$seed(\d+)", text)
            output_match = re.fullmatch(r"\$(\d+)", text)
            if seed_match:
                index = int(seed_match.group(1))
                if index < len(seeds):
                    resolved.append(self._snapshot(seeds[index]))
            elif output_match:
                index = int(output_match.group(1))
                if index < len(outputs):
                    resolved.extend(
                        self._snapshot(memory) for memory in outputs[index]
                    )
        return list({memory["id"]: memory for memory in resolved}.values())

    def prepare_batch_query(self, question, system_message=None, **kwargs):
        prepared = super().prepare_batch_query(
            question, system_message=system_message, **kwargs
        )
        prepared.setdefault("extra", {})[
            "runtime_support_version"
        ] = self.RUNTIME_SUPPORT_VERSION
        return prepared
