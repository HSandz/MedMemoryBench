"""Caller-owned query text plus optional hard metadata.

The core query contract intentionally has no benchmark/query-type fields. Visible
alternatives, numbered lists, or other presentation structures remain part of text.
"""

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class QuestionInput:
    text: str
    hard_metadata: Mapping[str, Any] = field(default_factory=dict)

    def render(self):
        if not isinstance(self.text, str):
            raise TypeError("QuestionInput.text must be a string")
        text = self.text.strip()
        if not text:
            raise ValueError("QuestionInput.text must be nonempty")
        metadata = dict(self.hard_metadata or {})
        if any(not isinstance(key, str) or not key.strip() for key in metadata):
            raise ValueError("hard_metadata keys must be nonempty strings")
        return StructuredQuestion(text, metadata)


class StructuredQuestion(str):
    """String-compatible request carrying hard metadata outside natural language."""

    def __new__(cls, text, hard_metadata=None):
        obj = super().__new__(cls, str(text))
        obj.hard_metadata = dict(hard_metadata or {})
        return obj
