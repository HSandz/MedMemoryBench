"""Persistent, unit-scoped memory snapshots for repeatable evaluations."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from benchmarks.base import EvaluationUnit


SNAPSHOT_ENVELOPE_VERSION = 1
SMART_MEM0_WRITE_PARAMS = {
    "enable_memory_write",
    "frozen_memory_path",
    "memory_snapshot_revision",
    "write_context_mode",
    "write_prior_belief_limit",
    "write_provisional_limit",
    "write_window_max_tokens",
    "write_window_max_turns",
}


def _safe_component(value: Any) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value)).strip("-.")
    return text or "unknown"


def _stable_hash(value: Any, length: int = 20) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:length]


def compute_memory_config_hash(method_config: Any, dataset_config: Any) -> str:
    """Hash write semantics without coupling snapshots to query-only settings."""
    raw_method = getattr(method_config, "raw_config", {}) or {}
    raw_params = raw_method.get("agent_params", {}) or {}
    write_params = {
        key: raw_params.get(key)
        for key in sorted(SMART_MEM0_WRITE_PARAMS)
        if key in raw_params
    }
    write_contract = {}
    if getattr(method_config, "method_name", "") == "smart_mem0":
        # A query-only edit must not invalidate a built ledger, but changes to
        # capture/consolidation semantics must never silently reuse old memory.
        from methods.smart_mem0.contracts import MEMORY_WRITE_SCHEMA_VERSION
        from methods.smart_mem0.prompts import (
            CONSOLIDATION_PROMPT,
            MEMORY_WRITE_PROMPT,
        )

        write_contract = {
            "schema_version": MEMORY_WRITE_SCHEMA_VERSION,
            "capture_prompt_hash": _stable_hash(MEMORY_WRITE_PROMPT, 16),
            "consolidation_prompt_hash": _stable_hash(CONSOLIDATION_PROMPT, 16),
        }
    payload = {
        "method_name": getattr(method_config, "method_name", ""),
        "model": raw_method.get("model", {}),
        "memorize_model": raw_method.get("memorize_model", {}),
        "embedding": raw_method.get("embedding", {}),
        "write_params": write_params,
        "write_contract": write_contract,
        "dataset_name": getattr(dataset_config, "dataset_name", ""),
        "language": getattr(dataset_config, "language", ""),
    }
    return _stable_hash(payload, 16)


@dataclass(frozen=True)
class MemorySnapshotKey:
    dataset: str
    method: str
    model: str
    config_hash: str
    context_id: str
    unit_id: str
    unit_fingerprint: str
    parent_fingerprint: str


class MemorySnapshotStore:
    """Stores complete agent state under an exact dataset/context/unit key."""

    def __init__(
        self,
        root: Path,
        *,
        dataset: str,
        method: str,
        model: str,
        config_hash: str,
    ) -> None:
        self.root = Path(root)
        self.dataset = str(dataset)
        self.method = str(method)
        self.model = str(model)
        self.config_hash = str(config_hash)

    def make_key(
        self,
        unit: EvaluationUnit,
        *,
        parent_fingerprint: str = "",
    ) -> MemorySnapshotKey:
        sessions = [
            {
                "session_id": str(session.session_id),
                "content_hash": _stable_hash(session.to_memory_text(), 32),
                "metadata": session.metadata,
            }
            for session in unit.sessions_to_inject
        ]
        fingerprint = _stable_hash(
            {
                "dataset": self.dataset,
                "context_id": str(unit.context_id),
                "unit_id": str(unit.unit_id),
                "parent_fingerprint": parent_fingerprint,
                "sessions": sessions,
            }
        )
        return MemorySnapshotKey(
            dataset=self.dataset,
            method=self.method,
            model=self.model,
            config_hash=self.config_hash,
            context_id=str(unit.context_id),
            unit_id=str(unit.unit_id),
            unit_fingerprint=fingerprint,
            parent_fingerprint=parent_fingerprint,
        )

    def path_for(self, key: MemorySnapshotKey) -> Path:
        namespace = f"{_safe_component(key.method)}_{_safe_component(key.model)}"
        filename = (
            f"unit_{_safe_component(key.unit_id)}_" f"{key.unit_fingerprint}.json"
        )
        return (
            self.root
            / _safe_component(key.dataset)
            / namespace
            / _safe_component(key.config_hash)
            / f"context_{_safe_component(key.context_id)}"
            / filename
        )

    def load(self, key: MemorySnapshotKey) -> Optional[Dict[str, Any]]:
        path = self.path_for(key)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(payload, dict):
            return None
        expected = {
            "dataset": key.dataset,
            "method": key.method,
            "model": key.model,
            "config_hash": key.config_hash,
            "context_id": key.context_id,
            "unit_id": key.unit_id,
            "unit_fingerprint": key.unit_fingerprint,
            "parent_fingerprint": key.parent_fingerprint,
        }
        if payload.get("envelope_version") != SNAPSHOT_ENVELOPE_VERSION:
            return None
        if any(
            str(payload.get(field, "")) != str(value)
            for field, value in expected.items()
        ):
            return None
        state = payload.get("agent_state")
        if not isinstance(state, dict):
            return None
        if payload.get("agent_state_hash") != _stable_hash(state, 64):
            return None
        return state

    def save(
        self,
        key: MemorySnapshotKey,
        agent_state: Dict[str, Any],
        *,
        session_ids: list[str],
    ) -> Path:
        path = self.path_for(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "envelope_version": SNAPSHOT_ENVELOPE_VERSION,
            "dataset": key.dataset,
            "method": key.method,
            "model": key.model,
            "config_hash": key.config_hash,
            "context_id": key.context_id,
            "unit_id": key.unit_id,
            "unit_fingerprint": key.unit_fingerprint,
            "parent_fingerprint": key.parent_fingerprint,
            "session_ids": [str(value) for value in session_ids],
            "agent_state_hash": _stable_hash(agent_state, 64),
            "agent_state": agent_state,
        }
        temporary_path = path.with_suffix(".tmp")
        temporary_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        temporary_path.replace(path)
        return path
