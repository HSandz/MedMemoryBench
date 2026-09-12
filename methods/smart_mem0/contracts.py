"""Shared schemas and deterministic constants for SmartMem0."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

STOPWORDS = frozenset()

VALID_KINDS = {"FACT", "EVENT", "STATE"}

VALID_SUBJECT_CLASSES = {"PRIMARY_USER", "THIRD_PARTY", "GENERAL_KNOWLEDGE"}

VALID_SEMANTIC_ROLES = {
    "MEASUREMENT",
    "OBSERVATION",
    "SAFETY_CONSTRAINT",
    "ACCEPTED_POLICY",
    "PREFERENCE",
    "GUIDANCE",
    "IDENTITY",
}

VALID_MEMORY_TIERS = {"HOT", "COLD"}


# Bump this whenever capture, normalization, identity, or consolidation changes
# can alter the durable ledger. It is included in the snapshot fingerprint.
MEMORY_WRITE_SCHEMA_VERSION = 11

# Missing structural identifiers, not domain vocabulary.
GENERIC_STATE_KEYS = frozenset({"", "unknown"})

GENERIC_OBJECT_ANCHORS = frozenset({"", "unknown"})

# Only these kinds participate in versioned state-head resolution. FACT and
# EVENT nodes may mention a state-like topic without being state versions.
STATE_LIKE_KINDS = frozenset({"STATE"})

VALID_ASSERTION_MODES = {"DIRECT", "RECAP", "INFERRED"}

VALID_RELATIONS = {
    "SUPPORT",
    "REFINE",
    "SUPERSEDE",
    "CONFLICT",
    "RELATED",
    "CAUSES",
}

VALID_OPERATIONS = {
    "SEMANTIC_SEARCH",
    "LOCATE_ANCHOR",
    "TEMPORAL_FILTER",
    "RESOLVE_STATE",
    "FOLLOW_CAUSES",
    "VERIFY_EVIDENCE",
}

# SEMANTIC_SEARCH remains the only general recall primitive.  A strategy tells
# the deterministic executor how to diversify that operation's bounded output;
# it never authorizes retrieval outside the operation boundary.
VALID_SEARCH_STRATEGIES = {
    "FOCAL",
    "TRAJECTORY",
    "DECISION_BUNDLE",
    "SHARED_OPTIONS",
}

VALID_SLOT_TYPES = {
    "DIRECT",
    "CURRENT_STATE",
    "TEMPORAL",
    "TRANSITION",
    "CAUSE_PATH",
    "COMPARISON",
}

VALID_QUERY_MODES = {
    "DIRECT",
    "STATE",
    "TEMPORAL",
    "DECISION",
    "MULTI_OPTION",
    "CAUSAL",
    "COMPARISON",
    "MULTI_HOP",
}

VALID_REASONING_TYPES = {
    "NONE",
    "SYNTHESIS",
    "DECISION",
    "COMPARISON",
    "CAUSAL",
}

# Scope aliases are deliberately small and domain-agnostic. They normalize
# grammatical variants only; entity ownership remains in object_anchor.
SCOPE_ALIASES = {}

VALID_EVIDENCE_ROLES = {
    "ANSWER",
    "FOCAL_STATE",
    "LONGITUDINAL_CONTEXT",
    "ACTION_RULE",
    "CONSTRAINT",
    "ALTERNATIVE",
    "TEMPORAL",
    "CAUSE",
    "COMPARAND",
    "OPTION",
    "PRIOR_TRAJECTORY",
    "FOCAL_TRIGGER",
    "CAUSAL_BRIDGE",
    "OUTCOME",
    "OPTION_CONTEXT",
    "GENERIC_EVIDENCE",
}

VALID_PLANNING_TAGS = {
    "IDENTITY",
    "STATE",
    "EXPOSURE",
    "RESPONSE",
    "TRAJECTORY",
    "RISK",
    "CONSTRAINT",
    "RESOURCE",
}

VALID_TEMPORAL_AXES = {
    "event_time",
    "document_time",
    "origin_document_time",
    "effective_event_time",
}

VALID_TEMPORAL_RELATIONS = {
    "LOCATE",
    "EXACT",
    "EARLIEST",
    "LATEST",
    "BEFORE",
    "AFTER",
    "BETWEEN",
}

# Planner chooses a coarse workload class; the executor owns the exact limits.
# MEDIUM permits four one-step gap operations so a four-option query cannot be
# silently truncated after evaluating only two options. Multi-step temporal or
# causal paths are still bounded by their explicit operation sequences.
RETRIEVAL_BUDGETS = {
    "SMALL": {"max_memories": 3, "max_operations": 1},
    "MEDIUM": {"max_memories": 5, "max_operations": 4},
    "LARGE": {"max_memories": 8, "max_operations": 4},
}

SLOT_REQUIRED_FIELDS = {
    "DIRECT": {"value"},
    "CURRENT_STATE": {"state_identity", "value"},
    "TRANSITION": {"old_value", "new_value"},
    "CAUSE_PATH": {"path"},
    "COMPARISON": {"left", "right"},
}

MONTH_NUMBERS = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}

STATE_KEY_NOISE = frozenset()


@dataclass(frozen=True)
class QueryFrame:
    """Deterministic query constraints applied before semantic ranking."""

    dates: Tuple[str, ...] = ()
    speaker_role: str = ""
    entities: Tuple[str, ...] = ()
    hard_entities: Tuple[str, ...] = ()


@dataclass
class MemoryWriteContext:
    """Ephemeral interpretation context used only while ingesting one session."""

    local_turn_context: List[Dict[str, Any]] = field(default_factory=list)
    core_beliefs: List[Dict[str, Any]] = field(default_factory=list)
    relevant_prior_beliefs: List[Dict[str, Any]] = field(default_factory=list)
    provisional_memories: List[Dict[str, Any]] = field(default_factory=list)

    def clear(self) -> None:
        self.local_turn_context.clear()
        self.core_beliefs.clear()
        self.relevant_prior_beliefs.clear()
        self.provisional_memories.clear()
