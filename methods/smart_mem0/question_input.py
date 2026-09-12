"""Caller-owned query structure; no inference from language-specific wrappers."""

from dataclasses import dataclass, field
from typing import Mapping, Optional


@dataclass(frozen=True)
class QuestionInput:
    text: str
    candidates: Mapping[str, str] = field(default_factory=dict)
    owner_id: Optional[str] = None
    selector_required: Optional[bool] = None

    def render(self):
        if not isinstance(self.text, str):
            raise TypeError("QuestionInput.text must be a string")
        if self.owner_id is not None and (
            not isinstance(self.owner_id, str) or not self.owner_id.strip()
        ):
            raise ValueError("owner_id must be a nonempty string or None")
        if (
            self.selector_required is not None
            and type(self.selector_required) is not bool
        ):
            raise TypeError("selector_required must be bool or None")
        candidates = dict(self.candidates)
        if any(
            not isinstance(k, str)
            or not k.strip()
            or not isinstance(v, str)
            or not v.strip()
            for k, v in candidates.items()
        ):
            raise ValueError(
                "Candidates require nonempty string labels and propositions"
            )
        return StructuredQuestion(
            self.text, candidates, self.owner_id, self.selector_required
        )


class StructuredQuestion(str):
    """String-compatible request carrying metadata without mutable agent state."""

    def __new__(cls, text, candidates, owner_id=None, selector_required=None):
        import json

        rendered = (
            text
            if not candidates
            else text + "\nCANDIDATES: " + json.dumps(candidates, ensure_ascii=False)
        )
        if owner_id is not None or selector_required is not None:
            rendered += "\nREQUEST METADATA: " + json.dumps(
                {"owner_id": owner_id, "selector_required": selector_required},
                ensure_ascii=False,
            )
        obj = super().__new__(cls, rendered)
        obj.stem = text
        obj.candidates = dict(candidates)
        obj.owner_id = owner_id
        obj.selector_required = selector_required
        return obj
