"""Retired P1A natural-language fast path.

The original English-regex router is intentionally bypassed by the two-stage read
architecture. Natural-language semantics now has one owner: TwoStageControllerMixin.
This module keeps the public hook so QueryMixin can fail open without special cases.
"""

from typing import Any, Dict, Optional


def _prepare_p1a_query(
    agent,
    routing_question: str,
    answer_question: str,
    subject_aliases: dict = None,
    telemetry_out: dict = None,
) -> Optional[Dict[str, Any]]:
    if telemetry_out is None:
        telemetry_out = {}
    telemetry_out.update(
        {
            "attempted": False,
            "accepted": False,
            "route": "DISABLED",
            "routing_question": routing_question,
            "fallback_reason": "TWO_STAGE_CONTROLLER_OWNS_SEMANTICS",
        }
    )
    return None
