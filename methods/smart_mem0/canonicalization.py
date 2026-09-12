"""Universal durable state addresses; missing ownership is never fabricated."""

from .lexical import normalized_text


def canonical_predicate_key(value):
    return normalized_text(value)


def canonical_predicate_family(value):
    # Families must be supplied by the schema, not inferred from suffixes.
    return canonical_predicate_key(value)


def canonical_object_key(value):
    return normalized_text(value)


def build_state_identity(subject_id, scope, state_key, object_anchor):
    owner, predicate = normalized_text(subject_id), canonical_predicate_key(state_key)
    if not owner or owner == "unknown" or not predicate or predicate == "unknown":
        return ""
    # Length prefixes make names containing separators collision-free.
    fields = (
        owner,
        normalized_text(scope),
        predicate,
        canonical_object_key(object_anchor),
    )
    return "|".join(f"{len(v)}:{v}" for v in fields)


def state_identity(memory):
    if memory.get("kind") != "STATE":
        return ""
    return build_state_identity(
        memory.get("owner_id") or memory.get("subject_id") or memory.get("subject"),
        memory.get("scope"),
        memory.get("state_key"),
        memory.get("object_anchor"),
    )


def measurement_identity(memory):
    """Compatibility accessor: measurement names are ordinary supplied predicates."""
    return canonical_predicate_key(memory.get("state_key"))


def has_exact_measurement_identity(memory):
    return bool(measurement_identity(memory))


def is_state_projection_eligible(memory):
    return bool(
        state_identity(memory)
        and memory.get("memory_tier", "COLD") == "HOT"
        and memory.get("assertion_mode", "DIRECT") == "DIRECT"
    )


def effective_time(memory):
    event = memory.get("event_time")
    return event if event and event != "UNKNOWN" else memory.get("document_time") or ""


class StateSpine:
    """Ordered view only; CURRENT certification uses explicit state heads."""

    def __init__(self, identity):
        self.identity, self.versions = identity, []

    def add_version(self, memory):
        self.versions.append(memory)
        self.versions.sort(key=lambda m: (effective_time(m), str(m.get("id") or "")))

    def latest(self):
        return self.versions[-1] if self.versions else None

    def earliest(self):
        return self.versions[0] if self.versions else None

    def as_of(self, target_date_str):
        eligible = [
            v
            for v in self.versions
            if effective_time(v) and effective_time(v) <= target_date_str
        ]
        return eligible[-1] if eligible else None


def canonicalize_state(memory):
    return memory
