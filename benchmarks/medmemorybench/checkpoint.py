"""MedMemoryBench checkpoint module for resumable evaluation (independent mode only)."""

import json
import hashlib
import os
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict, replace
from typing import Dict, List, Any, Optional


@dataclass
class MedMemoryBenchCheckpoint:
    """Checkpoint data structure for MedMemoryBench (independent mode only)."""

    checkpoint_id: str
    created_at: str
    updated_at: str

    method_name: str
    model_name: str
    dataset_name: str = "medmemorybench"
    config_hash: str = ""
    checkpoint_version: int = 2
    integrity_hash: str = ""
    # Keeps each fresh batch run separate while allowing its resume to find it.
    batch_manifest_scope: str = ""

    status: str = "in_progress"
    evaluation_mode: str = "independent"

    completed_personas: List[int] = field(default_factory=list)
    current_persona_id: Optional[int] = None
    current_persona_completed_queries: List[str] = field(default_factory=list)
    current_persona_injected_sessions: List[str] = field(default_factory=list)
    active_unit_id: Optional[int] = None
    active_session_id: Optional[str] = None
    active_session_started_at: Optional[str] = None

    completed_results: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)

    total_personas: int = 0
    total_queries: int = 0
    completed_query_count: int = 0

    last_error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MedMemoryBenchCheckpoint":
        defaults = {
            "checkpoint_id": "",
            "created_at": "",
            "updated_at": "",
            "method_name": "",
            "model_name": "",
            "dataset_name": "medmemorybench",
            "config_hash": "",
            "checkpoint_version": 1,
            "integrity_hash": "",
            "batch_manifest_scope": "",
            "status": "in_progress",
            "evaluation_mode": "independent",
            "completed_personas": [],
            "current_persona_id": None,
            "current_persona_completed_queries": [],
            "current_persona_injected_sessions": [],
            "active_unit_id": None,
            "active_session_id": None,
            "active_session_started_at": None,
            "completed_results": {},
            "total_personas": 0,
            "total_queries": 0,
            "completed_query_count": 0,
            "last_error": None,
        }
        merged = {**defaults, **data}
        return cls(**merged)


class MedMemoryBenchCheckpointManager:
    """Checkpoint manager for MedMemoryBench evaluation."""

    def __init__(
        self,
        method_name: str,
        model_name: str,
        checkpoint_dir: Path,
        config_hash: str = "",
    ):
        self.method_name = method_name
        self.model_name = model_name
        self.checkpoint_dir = Path(checkpoint_dir)
        self.config_hash = config_hash
        self._checkpoint: Optional[MedMemoryBenchCheckpoint] = None
        self.recovered_from_backup = False

    @property
    def checkpoint_path(self) -> Path:
        safe_method = self.method_name.replace("/", "-").replace("\\", "-")
        safe_model = self.model_name.replace("/", "-").replace("\\", "-")
        subdir = f"{safe_method}_{safe_model}"
        return self.checkpoint_dir / "medmemorybench" / subdir / "checkpoint.json"

    @property
    def backup_path(self) -> Path:
        return self.checkpoint_path.with_name("checkpoint.prev.json")

    @property
    def temporary_path(self) -> Path:
        return self.checkpoint_path.with_name(f"{self.checkpoint_path.name}.tmp")

    def exists(self) -> bool:
        return self.checkpoint_path.exists() or self.backup_path.exists()

    def load(self) -> Optional[MedMemoryBenchCheckpoint]:
        if not self.exists():
            return None

        self.recovered_from_backup = False
        for path, is_backup in (
            (self.checkpoint_path, False),
            (self.backup_path, True),
        ):
            data = self._read_valid_payload(path)
            if data is None:
                continue
            try:
                self._checkpoint = MedMemoryBenchCheckpoint.from_dict(data)
            except (TypeError, ValueError):
                continue
            if is_backup:
                self.recovered_from_backup = True
                self._write_payload_atomic(self.checkpoint_path, data)
            return self._checkpoint
        return None

    def save(self) -> None:
        if self._checkpoint is None:
            return

        self._checkpoint.updated_at = datetime.now().isoformat()
        self._checkpoint.checkpoint_version = 2
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        payload = self._checkpoint.to_dict()
        payload["integrity_hash"] = self._compute_integrity_hash(payload)
        self._checkpoint.integrity_hash = payload["integrity_hash"]

        current_payload = self._read_valid_payload(self.checkpoint_path)
        if current_payload is not None:
            self._write_payload_atomic(self.backup_path, current_payload)
        self._write_payload_atomic(self.checkpoint_path, payload)

    @staticmethod
    def _compute_integrity_hash(payload: Dict[str, Any]) -> str:
        hash_payload = dict(payload)
        hash_payload.pop("integrity_hash", None)
        content = json.dumps(
            hash_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def _read_valid_payload(self, path: Path) -> Optional[Dict[str, Any]]:
        if not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(payload, dict):
            return None

        integrity_hash = payload.get("integrity_hash", "")
        if integrity_hash and integrity_hash != self._compute_integrity_hash(payload):
            return None
        if not self._has_valid_structure(payload):
            return None
        return payload

    @staticmethod
    def _has_valid_structure(payload: Dict[str, Any]) -> bool:
        return (
            isinstance(payload.get("completed_personas", []), list)
            and isinstance(payload.get("current_persona_completed_queries", []), list)
            and isinstance(payload.get("current_persona_injected_sessions", []), list)
            and isinstance(payload.get("completed_results", {}), dict)
        )

    def _write_payload_atomic(self, path: Path, payload: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_name(f"{path.name}.tmp")
        with open(temporary_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        try:
            directory_descriptor = os.open(path.parent, os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(directory_descriptor)
        except OSError:
            pass
        finally:
            os.close(directory_descriptor)

    def create(
        self,
        total_personas: int,
        total_queries: int,
        evaluation_mode: str,
    ) -> MedMemoryBenchCheckpoint:
        import uuid

        checkpoint_id = str(uuid.uuid4())
        self._checkpoint = MedMemoryBenchCheckpoint(
            checkpoint_id=checkpoint_id,
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
            method_name=self.method_name,
            model_name=self.model_name,
            config_hash=self.config_hash,
            batch_manifest_scope=checkpoint_id,
            evaluation_mode=evaluation_mode,
            total_personas=total_personas,
            total_queries=total_queries,
        )
        self.save()
        return self._checkpoint

    def validate_config(self) -> bool:
        if self._checkpoint is None:
            return False
        if not self._checkpoint.config_hash:
            return True
        return self._checkpoint.config_hash == self.config_hash

    def is_independent_mode(self) -> bool:
        if self._checkpoint is None:
            return False
        return self._checkpoint.evaluation_mode == "independent"

    def start_persona(self, persona_id: int, preserve_progress: bool = False) -> None:
        if self._checkpoint is None:
            return

        keep_progress = (
            preserve_progress
            and self._checkpoint.current_persona_id == persona_id
        )
        self._checkpoint.current_persona_id = persona_id
        if not keep_progress:
            self._checkpoint.current_persona_completed_queries = []
            self._checkpoint.current_persona_injected_sessions = []
        self.save()

    def mark_session_injected(self, session_id: str) -> None:
        if self._checkpoint is None:
            return

        if session_id not in self._checkpoint.current_persona_injected_sessions:
            self._checkpoint.current_persona_injected_sessions.append(session_id)
        if str(self._checkpoint.active_session_id) == str(session_id):
            self._checkpoint.active_unit_id = None
            self._checkpoint.active_session_id = None
            self._checkpoint.active_session_started_at = None
        self.save()

    def mark_session_started(self, session_id: str, unit_id: int) -> None:
        """Persist the intent to build a session before touching agent memory."""
        if self._checkpoint is None:
            return
        self._checkpoint.active_unit_id = unit_id
        self._checkpoint.active_session_id = session_id
        self._checkpoint.active_session_started_at = datetime.now().isoformat()
        self.save()

    def rollback_incomplete_session(self) -> Optional[Dict[str, Any]]:
        """Discard an unfinished session marker and return its recovery details."""
        if self._checkpoint is None or self._checkpoint.active_session_id is None:
            return None

        active_session_id = self._checkpoint.active_session_id
        rollback = {
            "persona_id": self._checkpoint.current_persona_id,
            "unit_id": self._checkpoint.active_unit_id,
            "session_id": active_session_id,
            "started_at": self._checkpoint.active_session_started_at,
        }
        self._checkpoint.current_persona_injected_sessions = [
            session_id
            for session_id in self._checkpoint.current_persona_injected_sessions
            if str(session_id) != str(active_session_id)
        ]
        self._checkpoint.active_unit_id = None
        self._checkpoint.active_session_id = None
        self._checkpoint.active_session_started_at = None
        self.save()
        return rollback

    def is_session_injected(self, session_id: str, persona_id: Optional[int] = None) -> bool:
        if self._checkpoint is None:
            return False
        if persona_id is not None and persona_id != self._checkpoint.current_persona_id:
            return False
        return any(
            str(injected_id) == str(session_id)
            for injected_id in self._checkpoint.current_persona_injected_sessions
        )

    def adopt_config_hash(self) -> None:
        """Make a forced resume use the current config on later resumes."""
        if self._checkpoint is None:
            return
        self._checkpoint.config_hash = self.config_hash
        self.save()

    def mark_query_completed(
        self,
        query_id: str,
        result_dict: Dict[str, Any],
        persona_id: Optional[int] = None,
    ) -> None:
        if self._checkpoint is None:
            return

        persona_id = persona_id if persona_id is not None else self._checkpoint.current_persona_id
        if persona_id is None:
            return

        persona_key = str(persona_id)
        existing_ids = [
            result.get("query_id")
            for result in self._checkpoint.completed_results.get(persona_key, [])
        ]

        if query_id not in existing_ids:
            self._checkpoint.completed_query_count += 1

        if (
            persona_id == self._checkpoint.current_persona_id
            and query_id not in self._checkpoint.current_persona_completed_queries
        ):
            self._checkpoint.current_persona_completed_queries.append(query_id)

        if persona_key not in self._checkpoint.completed_results:
            self._checkpoint.completed_results[persona_key] = []

        if query_id not in existing_ids:
            self._checkpoint.completed_results[persona_key].append(result_dict)

        self.save()

    def complete_persona(self, persona_id: int) -> None:
        if self._checkpoint is None:
            return

        if persona_id not in self._checkpoint.completed_personas:
            self._checkpoint.completed_personas.append(persona_id)

        self._checkpoint.current_persona_id = None
        self._checkpoint.current_persona_completed_queries = []
        self._checkpoint.current_persona_injected_sessions = []
        self._checkpoint.active_unit_id = None
        self._checkpoint.active_session_id = None
        self._checkpoint.active_session_started_at = None
        self.save()

    def mark_completed(self) -> None:
        if self._checkpoint is None:
            return

        self._checkpoint.status = "completed"
        self.save()

    def mark_failed(self, error: str) -> None:
        if self._checkpoint is None:
            return

        self._checkpoint.status = "failed"
        self._checkpoint.last_error = error
        self.save()

    def is_persona_completed(self, persona_id: int) -> bool:
        if self._checkpoint is None:
            return False
        return persona_id in self._checkpoint.completed_personas

    def is_query_completed(self, query_id: str, persona_id: Optional[int] = None) -> bool:
        if self._checkpoint is None:
            return False
        if persona_id is not None:
            return any(
                result.get("query_id") == query_id
                for result in self._checkpoint.completed_results.get(str(persona_id), [])
            )
        return query_id in self._checkpoint.current_persona_completed_queries

    def get_completed_results(self) -> Dict[int, List[Dict[str, Any]]]:
        if self._checkpoint is None:
            return {}
        return {
            int(k): v
            for k, v in self._checkpoint.completed_results.items()
        }

    def get_current_persona_id(self) -> Optional[int]:
        if self._checkpoint is None:
            return None
        return self._checkpoint.current_persona_id

    def get_batch_manifest_scope(self) -> str:
        """Return the persisted batch namespace for a fresh/resumed run."""
        if self._checkpoint is None:
            return ""
        return self._checkpoint.batch_manifest_scope

    def get_resume_info(self) -> Dict[str, Any]:
        if self._checkpoint is None:
            return {}

        return {
            "checkpoint_id": self._checkpoint.checkpoint_id,
            "completed_personas": len(self._checkpoint.completed_personas),
            "total_personas": self._checkpoint.total_personas,
            "completed_queries": self._checkpoint.completed_query_count,
            "total_queries": self._checkpoint.total_queries,
            "current_persona": self._checkpoint.current_persona_id,
            "current_persona_completed_queries": len(self._checkpoint.current_persona_completed_queries),
            "current_persona_injected_sessions": len(self._checkpoint.current_persona_injected_sessions),
            "active_unit_id": self._checkpoint.active_unit_id,
            "active_session_id": self._checkpoint.active_session_id,
        }

    def delete(self) -> None:
        for path in (
            self.checkpoint_path,
            self.backup_path,
            self.temporary_path,
            self.backup_path.with_name(f"{self.backup_path.name}.tmp"),
        ):
            path.unlink(missing_ok=True)
        self._checkpoint = None


def _snapshot_dataset_config(dataset_config) -> Dict[str, Any]:
    """Return only dataset settings that determine injected memory contents."""
    return {
        "dataset_name": getattr(dataset_config, "dataset_name", ""),
        "language": getattr(dataset_config, "language", ""),
        "data_root_dir": getattr(dataset_config, "data_root_dir", ""),
        "data_files": getattr(dataset_config, "data_files", {}),
        "evaluation_mode": getattr(dataset_config, "evaluation_mode", ""),
        "persona_ids": getattr(dataset_config, "persona_ids", None),
        "max_personas": getattr(dataset_config, "max_personas", None),
        "max_sessions_per_persona": getattr(
            dataset_config, "max_sessions_per_persona", None
        ),
        "evaluation_interval": getattr(dataset_config, "evaluation_interval", None),
        "inject_noise": getattr(dataset_config, "inject_noise", None),
    }


def _snapshot_model_config(model_config) -> Optional[Dict[str, Any]]:
    """Serialize model settings without changing legacy hashes for unset options."""
    if model_config is None:
        return None
    value = vars(model_config).copy()
    for key in (
        "openrouter_provider",
        "openrouter_service_tier",
        "nim_thinking_enabled",
        "nim_reasoning_budget",
    ):
        if value.get(key) is None:
            value.pop(key, None)
    return value


_QUERY_RELEVANT_AMEM_BUILD_KEYS = frozenset({
    # The query agent must use the same embedding/index and snapshot features.
    "amem_embedding_model",
    "amem_note_level",
    "amem_original_evolution",
    "amem_typed_relations",
    "amem_typed_retrieval",
    "amem_temporal_state",
    "amem_temporal_retrieval",
    "amem_provenance",
    "amem_provenance_retrieval",
})


def _snapshot_query_dataset_config(dataset_config) -> Dict[str, Any]:
    """Serialize dataset fields that determine query units and scoring."""
    return {
        "dataset_name": getattr(dataset_config, "dataset_name", ""),
        "language": getattr(dataset_config, "language", ""),
        "data_root_dir": getattr(dataset_config, "data_root_dir", ""),
        "data_files": getattr(dataset_config, "data_files", {}),
        "evaluation_mode": getattr(dataset_config, "evaluation_mode", ""),
        "persona_ids": getattr(dataset_config, "persona_ids", None),
        "max_personas": getattr(dataset_config, "max_personas", None),
        "max_sessions_per_persona": getattr(
            dataset_config, "max_sessions_per_persona", None
        ),
        "inject_noise": getattr(dataset_config, "inject_noise", None),
        "query_types": [
            vars(item) for item in getattr(dataset_config, "query_types", [])
        ],
    }


def _snapshot_embedding_config(embedding_config) -> Optional[Dict[str, Any]]:
    if embedding_config is None:
        return None
    value = vars(embedding_config).copy()
    # Credentials do not change retrieval semantics and must never affect hashes.
    value.pop("api_key", None)
    return value


def _snapshot_query_model_config(model_config) -> Optional[Dict[str, Any]]:
    """Serialize query-model settings without credentials."""
    value = _snapshot_model_config(model_config)
    if value is not None:
        value.pop("api_key", None)
    return value


def _query_relevant_build_config(method_config) -> Dict[str, Any]:
    build_config = (
        method_config.snapshot_build_config()
        if hasattr(method_config, "snapshot_build_config")
        else getattr(method_config, "agent_params", {})
    )
    return {
        key: build_config[key]
        for key in sorted(_QUERY_RELEVANT_AMEM_BUILD_KEYS)
        if key in build_config
    }


def compute_memory_query_compatibility_hash(method_config, dataset_config) -> str:
    """Hash only settings required to consume an existing memory snapshot."""
    try:
        content = json.dumps({
            "method_name": getattr(method_config, "method_name", ""),
            "method_type": getattr(method_config, "method_type", ""),
            "embedding": _snapshot_embedding_config(
                getattr(method_config, "embedding", None)
            ),
            "query_relevant_build_config": _query_relevant_build_config(
                method_config
            ),
            "dataset": _snapshot_query_dataset_config(dataset_config),
        }, sort_keys=True, default=str)
        return hashlib.md5(content.encode()).hexdigest()[:16]
    except Exception:
        return ""


def compute_build_config_hash(method_config, dataset_config) -> str:
    """Hash only settings that can change a serialized memory snapshot."""
    try:
        embedding = getattr(method_config, "embedding", None)
        memorize_model = getattr(method_config, "memorize_model", None)
        build_config = (
            method_config.snapshot_build_config()
            if hasattr(method_config, "snapshot_build_config")
            else getattr(method_config, "agent_params", {})
        )
        content = json.dumps({
            "method_name": getattr(method_config, "method_name", ""),
            "method_type": getattr(method_config, "method_type", ""),
            "embedding": vars(embedding) if embedding is not None else None,
            "memorize_model": _snapshot_model_config(memorize_model),
            "build_config": build_config,
            "dataset": _snapshot_dataset_config(dataset_config),
        }, sort_keys=True, default=str)
        return hashlib.md5(content.encode()).hexdigest()[:16]
    except Exception:
        return ""


def compute_query_config_hash(method_config, dataset_config) -> str:
    """Hash the effective method/query configuration for query artifacts."""
    try:
        content = json.dumps({
            "method_name": getattr(method_config, "method_name", ""),
            "method_type": getattr(method_config, "method_type", ""),
            "model": _snapshot_query_model_config(
                getattr(method_config, "model", None)
            ),
            "embedding": _snapshot_embedding_config(
                getattr(method_config, "embedding", None)
            ),
            "retrieval_config": (
                method_config.query_config()
                if hasattr(method_config, "query_config")
                else {}
            ),
            "query_relevant_build_config": _query_relevant_build_config(
                method_config
            ),
            "dataset": _snapshot_query_dataset_config(dataset_config),
        }, sort_keys=True, default=str)
        return hashlib.md5(content.encode()).hexdigest()[:16]
    except Exception:
        return ""


def compute_config_hash(method_config, dataset_config) -> str:
    """Backward-compatible alias for the memory build compatibility hash."""
    return compute_build_config_hash(method_config, dataset_config)


def derive_legacy_build_config_hash(
    manifest: Dict[str, Any],
    manifest_path: Optional[Path] = None,
) -> str:
    """Derive a build-only hash for a legacy manifest using its run snapshot."""
    explicit_hash = manifest.get("build_config_hash")
    if isinstance(explicit_hash, str) and explicit_hash:
        return explicit_hash
    if manifest_path is None:
        return str(manifest.get("config_hash") or "")

    run_config_path = Path(manifest_path).resolve().parent.parent / "run_config.json"
    if not run_config_path.exists():
        return str(manifest.get("config_hash") or "")
    try:
        run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
        from src.config import method_config_from_snapshot, dataset_config_from_snapshot

        method_config = method_config_from_snapshot(run_config["method_config"])
        dataset_config = dataset_config_from_snapshot(run_config["dataset_config"])
        dataset_overrides = method_config.raw_config.get("dataset_overrides", {})
        if dataset_overrides:
            dataset_config = replace(dataset_config, **dataset_overrides)
        return compute_build_config_hash(method_config, dataset_config)
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return str(manifest.get("config_hash") or "")


def is_manifest_build_compatible(
    manifest: Dict[str, Any],
    expected_hash: str,
    manifest_path: Optional[Path] = None,
) -> bool:
    """Accept new hashes and safely derived hashes from legacy run configs."""
    if not expected_hash:
        return False
    explicit_hash = str(manifest.get("build_config_hash") or "")
    if explicit_hash:
        return explicit_hash == expected_hash
    legacy_hash = str(manifest.get("config_hash") or "")
    return bool(
        legacy_hash == expected_hash
        or derive_legacy_build_config_hash(manifest, manifest_path) == expected_hash
    )


def is_manifest_query_compatible(
    manifest: Dict[str, Any],
    method_config,
    dataset_config,
    manifest_path: Optional[Path] = None,
) -> bool:
    """Validate settings that affect reading and querying a memory snapshot.

    Older manifests only contain hashes that also included build settings and
    the original query model. When a run snapshot is available, recompute the
    new memory-consumer fingerprint from both configurations so those runs can
    be queried with a different LLM or changed build-only options.
    """
    expected_hash = compute_memory_query_compatibility_hash(
        method_config, dataset_config
    )
    if not expected_hash:
        return False

    if manifest_path is not None:
        run_config_path = Path(manifest_path).resolve().parent.parent / "run_config.json"
        try:
            run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
            stored_method = run_config.get("method_config")
            stored_dataset = run_config.get("dataset_config")
            if (
                isinstance(stored_method, dict)
                and isinstance(stored_dataset, dict)
                and isinstance(stored_method.get("raw_config"), dict)
                and isinstance(stored_dataset.get("raw_config"), dict)
            ):
                from src.config import method_config_from_snapshot, dataset_config_from_snapshot

                stored_method_config = method_config_from_snapshot(stored_method)
                stored_dataset_config = dataset_config_from_snapshot(stored_dataset)
                query_match = (
                    compute_memory_query_compatibility_hash(
                        stored_method_config, stored_dataset_config
                    )
                    == expected_hash
                )
                if query_match:
                    return True
                # Test/legacy manifests without retrieval metadata predate the
                # stage-specific contract; retain their build-hash behavior.
                if manifest.get("retrieval_config_hash") or manifest.get(
                    "retrieval_compatibility_hash"
                ):
                    return False
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            pass

    # Minimal/legacy manifests may not have a complete run_config snapshot. In
    # that case retain the historical build-hash check as a conservative
    # fallback; complete snapshots use the stage-specific comparison above.
    return str(manifest.get("retrieval_compatibility_hash") or "") == expected_hash or (
        str(manifest.get("build_config_hash") or manifest.get("config_hash") or "")
        == compute_build_config_hash(method_config, dataset_config)
    )
