"""Atomic NumPy sidecars shared by snapshot-backed memory adapters."""

from __future__ import annotations

import hashlib
import os
import re
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


EMBEDDING_ARTIFACTS_KEY = "embedding_artifacts"
_ARTIFACT_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_]+$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def is_safe_basename(value: Any) -> bool:
    """Return whether a persisted artifact name cannot escape its snapshot dir."""
    return bool(
        isinstance(value, str)
        and value not in {"", ".", ".."}
        and "/" not in value
        and "\\" not in value
        and Path(value).name == value
    )


def file_sha256(path: Path) -> str:
    """Return the SHA-256 digest of one binary artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fsync_directory(path: Path) -> None:
    """Make an atomic replacement durable where directory fsync is supported."""
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _validate_artifact_name(name: str) -> None:
    if not isinstance(name, str):
        raise ValueError(f"Invalid binary artifact name: {name!r}")
    if not _ARTIFACT_NAME_PATTERN.fullmatch(name):
        raise ValueError(f"Invalid binary artifact name: {name!r}")


def _expected_dtype_and_shape(
    metadata: Dict[str, Any],
    *,
    label: str,
) -> tuple[np.dtype, List[int]]:
    dtype_value = metadata.get("dtype")
    if not isinstance(dtype_value, str):
        raise ValueError(f"{label} dtype metadata is invalid")
    try:
        dtype = np.dtype(dtype_value)
    except TypeError as exc:
        raise ValueError(f"{label} dtype metadata is invalid") from exc
    if str(dtype) != dtype_value or dtype.hasobject:
        raise ValueError(f"{label} dtype metadata is invalid")

    shape_value = metadata.get("shape")
    if not isinstance(shape_value, list) or any(
        isinstance(item, bool) or not isinstance(item, int) or item < 0
        for item in shape_value
    ):
        raise ValueError(f"{label} shape metadata is invalid")
    return dtype, list(shape_value)


def _artifact_path_from_metadata(
    snapshot_path: Path,
    metadata: Dict[str, Any],
    *,
    label: str,
) -> Path:
    if metadata.get("storage") != "npy":
        raise ValueError(f"{label} storage metadata is invalid")
    artifact_name = metadata.get("path")
    if not is_safe_basename(artifact_name) or not str(artifact_name).endswith(".npy"):
        raise ValueError(f"{label} path metadata is invalid")
    digest = metadata.get("sha256")
    if not isinstance(digest, str) or not _SHA256_PATTERN.fullmatch(digest):
        raise ValueError(f"{label} SHA-256 metadata is invalid")
    _expected_dtype_and_shape(metadata, label=label)
    return snapshot_path.parent / artifact_name


def _validate_declared_array_metadata(
    metadata: Dict[str, Any],
    values: np.ndarray,
    *,
    label: str,
) -> None:
    expected_dtype, expected_shape = _expected_dtype_and_shape(metadata, label=label)
    if values.dtype != expected_dtype or list(values.shape) != expected_shape:
        raise ValueError(f"{label} shape or dtype does not match its values")


def publish_npy_artifact(
    snapshot_path: Path,
    artifact_name: str,
    values: Any,
) -> Dict[str, Any]:
    """Atomically publish one content-addressed NumPy array beside a snapshot."""
    _validate_artifact_name(artifact_name)
    array = np.asarray(values)
    if array.dtype.hasobject:
        raise ValueError(f"Binary artifact {artifact_name!r} cannot use an object dtype")

    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = snapshot_path.parent / (
        f".{snapshot_path.stem}.{artifact_name}.{uuid.uuid4().hex}.tmp"
    )
    with temporary_path.open("wb") as handle:
        np.save(handle, array, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    digest = file_sha256(temporary_path)
    published_path = snapshot_path.parent / (
        f"{snapshot_path.stem}.{artifact_name}.{digest}.npy"
    )
    os.replace(temporary_path, published_path)
    fsync_directory(snapshot_path.parent)
    return {
        "storage": "npy",
        "path": published_path.name,
        "sha256": digest,
        "dtype": str(array.dtype),
        "shape": list(array.shape),
    }


def load_npy_artifact(
    snapshot_path: Path,
    metadata: Dict[str, Any],
    *,
    label: str,
) -> np.ndarray:
    """Validate and load one persisted NumPy artifact without pickle support."""
    artifact_path = _artifact_path_from_metadata(snapshot_path, metadata, label=label)
    expected_dtype, expected_shape = _expected_dtype_and_shape(metadata, label=label)
    if not artifact_path.is_file():
        raise ValueError(f"{label} sidecar is missing: {artifact_path}")
    if file_sha256(artifact_path) != metadata.get("sha256"):
        raise ValueError(f"{label} sidecar SHA-256 check failed: {artifact_path}")
    try:
        values = np.load(artifact_path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError(f"{label} sidecar cannot be loaded: {artifact_path}") from exc
    if not isinstance(values, np.ndarray) or values.dtype.hasobject:
        raise ValueError(f"{label} sidecar dtype is invalid: {artifact_path}")
    if values.dtype != expected_dtype:
        raise ValueError(f"{label} sidecar dtype is invalid: {artifact_path}")
    if list(values.shape) != expected_shape:
        raise ValueError(f"{label} sidecar shape is invalid: {artifact_path}")
    return values


def _named_artifact_ids(
    metadata: Dict[str, Any],
    values: np.ndarray,
    *,
    label: str,
) -> List[str]:
    ids = metadata.get("ids")
    if (
        not isinstance(ids, list)
        or any(not isinstance(identifier, str) for identifier in ids)
        or len(set(ids)) != len(ids)
        or values.ndim != 2
        or len(ids) != values.shape[0]
    ):
        raise ValueError(f"{label} ID mapping is invalid")
    return list(ids)


def publish_named_embedding_artifacts(
    memory_state: Dict[str, Any],
    snapshot_path: Path,
) -> Dict[str, Dict[str, Any]]:
    """Publish named dense matrices declared in an adapter memory state."""
    artifacts = memory_state.get(EMBEDDING_ARTIFACTS_KEY)
    if artifacts is None:
        return {}
    if not isinstance(artifacts, dict):
        raise ValueError("Named embedding artifact metadata is invalid")

    published: Dict[str, Dict[str, Any]] = {}
    for name in sorted(artifacts, key=lambda item: str(item)):
        _validate_artifact_name(name)
        descriptor = artifacts[name]
        label = f"embedding artifact {name!r}"
        if not isinstance(descriptor, dict) or "values" not in descriptor:
            raise ValueError(f"{label} values are unavailable for publication")
        values = np.asarray(descriptor["values"])
        if values.dtype.hasobject:
            raise ValueError(f"{label} dtype is invalid")
        _validate_declared_array_metadata(descriptor, values, label=label)
        ids = _named_artifact_ids(descriptor, values, label=label)
        published[name] = {
            **publish_npy_artifact(snapshot_path, name, values),
            "ids": ids,
        }
    memory_state[EMBEDDING_ARTIFACTS_KEY] = published
    return published


def load_named_embedding_artifacts(
    memory_state: Dict[str, Any],
    snapshot_path: Path,
) -> Dict[str, Dict[str, Any]]:
    """Validate named sidecars and attach their arrays for adapter restoration."""
    artifacts = memory_state.get(EMBEDDING_ARTIFACTS_KEY)
    if artifacts is None:
        return {}
    if not isinstance(artifacts, dict):
        raise ValueError("Named embedding artifact metadata is invalid")

    for name in sorted(artifacts, key=lambda item: str(item)):
        _validate_artifact_name(name)
        descriptor = artifacts[name]
        label = f"embedding artifact {name!r}"
        if not isinstance(descriptor, dict) or "values" in descriptor:
            raise ValueError(f"{label} metadata is invalid")
        values = load_npy_artifact(snapshot_path, descriptor, label=label)
        _named_artifact_ids(descriptor, values, label=label)
        descriptor["values"] = values
    return artifacts


def _legacy_amem_embedding_state(memory_state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    system_state = memory_state.get("system_state")
    if not isinstance(system_state, dict):
        return None
    retriever = system_state.get("retriever")
    if not isinstance(retriever, dict):
        return None
    embeddings = retriever.get("embeddings")
    return embeddings if isinstance(embeddings, dict) else None


def publish_memory_state_embedding_artifacts(
    memory_state: Dict[str, Any],
    snapshot_path: Path,
) -> Dict[str, Dict[str, Any]]:
    """Publish legacy A-MEM and generic named embedding sidecars together."""
    published: Dict[str, Dict[str, Any]] = {}
    legacy = _legacy_amem_embedding_state(memory_state)
    if legacy is not None:
        if "values" not in legacy:
            raise ValueError("A-MEM embedding values are unavailable for publication")
        values = np.asarray(legacy["values"])
        _validate_declared_array_metadata(legacy, values, label="A-MEM embedding")
        legacy.pop("values")
        legacy.update(publish_npy_artifact(snapshot_path, "embeddings", values))
        published["embeddings"] = dict(legacy)

    published.update(publish_named_embedding_artifacts(memory_state, snapshot_path))
    return published


def load_memory_state_embedding_artifacts(
    memory_state: Dict[str, Any],
    snapshot_path: Path,
) -> Dict[str, Dict[str, Any]]:
    """Validate all persisted embedding sidecars before an adapter sees state."""
    loaded: Dict[str, Dict[str, Any]] = {}
    legacy = _legacy_amem_embedding_state(memory_state)
    if legacy is not None:
        if "values" in legacy:
            raise ValueError("A-MEM embedding metadata is invalid")
        values = load_npy_artifact(snapshot_path, legacy, label="A-MEM embedding")
        legacy["values"] = values
        loaded["embeddings"] = legacy

    loaded.update(load_named_embedding_artifacts(memory_state, snapshot_path))
    return loaded


def memory_state_embedding_artifact_paths(
    memory_state: Dict[str, Any],
    snapshot_path: Path,
) -> List[Path]:
    """Return safe, declared sidecar paths for serialized-size accounting."""
    paths: List[Path] = []
    legacy = _legacy_amem_embedding_state(memory_state)
    if legacy is not None:
        paths.append(_artifact_path_from_metadata(snapshot_path, legacy, label="A-MEM embedding"))
    artifacts = memory_state.get(EMBEDDING_ARTIFACTS_KEY)
    if artifacts is not None:
        if not isinstance(artifacts, dict):
            raise ValueError("Named embedding artifact metadata is invalid")
        for name in sorted(artifacts, key=lambda item: str(item)):
            _validate_artifact_name(name)
            descriptor = artifacts[name]
            if not isinstance(descriptor, dict):
                raise ValueError(f"embedding artifact {name!r} metadata is invalid")
            paths.append(_artifact_path_from_metadata(
                snapshot_path,
                descriptor,
                label=f"embedding artifact {name!r}",
            ))
    return paths


def cleanup_memory_state_embedding_artifacts(
    memory_state: Dict[str, Any],
    snapshot_path: Path,
) -> None:
    """Remove obsolete sidecars only after the owning JSON snapshot is published."""
    referenced = {
        path.name
        for path in memory_state_embedding_artifact_paths(memory_state, snapshot_path)
    }
    candidates = list(snapshot_path.parent.glob(f"{snapshot_path.stem}.*.npy"))
    candidates.append(snapshot_path.with_suffix(".embeddings.npy"))
    for candidate in candidates:
        if candidate.name not in referenced and candidate.is_file():
            candidate.unlink()
