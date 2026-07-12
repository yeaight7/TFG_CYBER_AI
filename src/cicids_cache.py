"""Validated, provider-neutral cache for canonical unscaled CICIDS2017 shards."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

try:
    from src.canonical_schema import (
        CICIDS2017_TO_CANON,
        DEFAULT_IMPUTATION_VALUE,
        FEATURES_CANON,
        NUM_OBSERVATION_FEATURES,
        get_observation_feature_names,
    )
except ModuleNotFoundError:
    from canonical_schema import (
        CICIDS2017_TO_CANON,
        DEFAULT_IMPUTATION_VALUE,
        FEATURES_CANON,
        NUM_OBSERVATION_FEATURES,
        get_observation_feature_names,
    )


CACHE_SCHEMA_VERSION = "1"
PREPROCESSING_VERSION = "cicids2017_canonical_unscaled_v1"
MAX_CACHE_WORKERS = 8


class CacheValidationError(RuntimeError):
    """Raised when a cache is absent, stale, incompatible, or corrupt."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    """Hash an array using the Phase 1 dtype/shape/content contract."""
    digest = hashlib.sha256()
    digest.update(f"{array.dtype}|{array.shape}|".encode())
    digest.update(np.ascontiguousarray(array))
    return digest.hexdigest()


def canonical_schema_sha256() -> str:
    return _sha256_json(
        {
            "features": FEATURES_CANON,
            "mapping": CICIDS2017_TO_CANON,
            "imputation_value": DEFAULT_IMPUTATION_VALUE,
            "observation_dimensions": NUM_OBSERVATION_FEATURES,
        }
    )


def preprocessing_fingerprint() -> str:
    """Hash the cacheable subset of the maintained loader preprocessing contract."""
    return _sha256_json(
        {
            "version": PREPROCESSING_VERSION,
            "label_column": "Label",
            "benign_value": "BENIGN",
            "drop_identifier_columns": True,
            "canonical_schema_sha256": canonical_schema_sha256(),
            "observation_dtype": "float32",
            "label_dtype": "int64",
            "scaling": False,
        }
    )


def default_worker_count() -> int:
    return min(os.cpu_count() or 1, MAX_CACHE_WORKERS)


def resolve_worker_count(workers: int | None) -> int:
    resolved = default_worker_count() if workers is None else workers
    if resolved < 1 or resolved > MAX_CACHE_WORKERS:
        raise ValueError(f"workers must be between 1 and {MAX_CACHE_WORKERS}; got {resolved}")
    return resolved


def _official_names() -> tuple[str, ...]:
    try:
        from src.load_cicids2017 import _OFFICIAL_CICIDS2017_CSV_NAMES
    except ModuleNotFoundError:
        from load_cicids2017 import _OFFICIAL_CICIDS2017_CSV_NAMES

    return _OFFICIAL_CICIDS2017_CSV_NAMES


def _official_sources(dataset_root: Path) -> list[Path]:
    try:
        from src.load_cicids2017 import list_cicids2017_csv_files
    except ModuleNotFoundError:
        from load_cicids2017 import list_cicids2017_csv_files

    return list_cicids2017_csv_files(dataset_root)


def _git_metadata() -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parent.parent

    def run_git(*args: str) -> str:
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=repo_root,
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError):
            return "unknown"
        return result.stdout.strip()

    commit = run_git("rev-parse", "HEAD")
    status = run_git("status", "--short", "--untracked-files=normal")
    return {
        "commit": commit,
        "dirty": status not in ("", "unknown"),
        "dirty_summary": status.splitlines() if status not in ("", "unknown") else [],
    }


def _process_source_csv(source_path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Use the maintained uncached loader path to produce one canonical shard."""
    try:
        from src.load_cicids2017 import CICIDSLoadConfig, _load_and_process_csv_paths
    except ModuleNotFoundError:
        from load_cicids2017 import CICIDSLoadConfig, _load_and_process_csv_paths

    config = CICIDSLoadConfig(
        local_dir=source_path.parent,
        max_rows=None,
        sample_frac=None,
        label_col="Label",
        benign_value="BENIGN",
        drop_identifier_cols=True,
        scale=False,
        use_canonical=True,
        allow_non_official_csvs=False,
        cache_policy="off",
    )
    return _load_and_process_csv_paths([source_path], config)


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _replace_directory(temporary: Path, destination: Path) -> None:
    backup = destination.with_name(f".{destination.name}.backup-{uuid.uuid4().hex}")
    moved_existing = False
    try:
        if destination.exists():
            os.replace(destination, backup)
            moved_existing = True
        os.replace(temporary, destination)
    except Exception:
        if moved_existing and backup.exists() and not destination.exists():
            os.replace(backup, destination)
        raise
    finally:
        if backup.exists():
            shutil.rmtree(backup)


def _build_one_shard(source_path: Path, shards_root: Path) -> dict[str, Any]:
    destination = shards_root / source_path.name
    temporary = shards_root / f"{source_path.name}.tmp-{uuid.uuid4().hex}"
    temporary.mkdir(parents=True, exist_ok=False)
    try:
        source_size = source_path.stat().st_size
        source_hash = sha256_file(source_path)
        observations, labels, feature_names = _process_source_csv(source_path)
        if source_path.stat().st_size != source_size or sha256_file(source_path) != source_hash:
            raise CacheValidationError(f"source changed during cache build: {source_path.name}")

        observations = np.ascontiguousarray(observations, dtype=np.float32)
        labels = np.ascontiguousarray(labels, dtype=np.int64)
        if observations.ndim != 2 or observations.shape[1] != NUM_OBSERVATION_FEATURES:
            raise CacheValidationError(
                f"incompatible observations for {source_path.name}: {observations.shape}"
            )
        if labels.shape != (observations.shape[0],):
            raise CacheValidationError(
                f"incompatible labels for {source_path.name}: {labels.shape}"
            )
        if feature_names != get_observation_feature_names():
            raise CacheValidationError(f"incompatible feature names for {source_path.name}")

        np.save(temporary / "observations.npy", observations, allow_pickle=False)
        np.save(temporary / "labels.npy", labels, allow_pickle=False)
        metadata: dict[str, Any] = {
            "cache_schema_version": CACHE_SCHEMA_VERSION,
            "canonical_schema_sha256": canonical_schema_sha256(),
            "preprocessing_version": PREPROCESSING_VERSION,
            "preprocessing_fingerprint": preprocessing_fingerprint(),
            "source_filename": source_path.name,
            "source_size_bytes": source_size,
            "source_sha256": source_hash,
            "row_count": int(observations.shape[0]),
            "observation_dtype": str(observations.dtype),
            "observation_shape": list(observations.shape),
            "observations_sha256": sha256_array(observations),
            "label_dtype": str(labels.dtype),
            "label_shape": list(labels.shape),
            "labels_sha256": sha256_array(labels),
            "feature_names": feature_names,
            "producer_git": _git_metadata(),
        }
        (temporary / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        _replace_directory(temporary, destination)
        return metadata
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CacheValidationError(f"cache is corrupt or missing JSON: {path}") from exc
    if not isinstance(value, dict):
        raise CacheValidationError(f"cache JSON must contain an object: {path}")
    return value


def _validate_shard(source_path: Path, shard_dir: Path) -> dict[str, Any]:
    metadata = _read_json(shard_dir / "metadata.json")
    expected_scalars = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "canonical_schema_sha256": canonical_schema_sha256(),
        "preprocessing_version": PREPROCESSING_VERSION,
        "preprocessing_fingerprint": preprocessing_fingerprint(),
        "source_filename": source_path.name,
        "source_size_bytes": source_path.stat().st_size,
        "source_sha256": sha256_file(source_path),
        "observation_dtype": "float32",
        "label_dtype": "int64",
        "feature_names": get_observation_feature_names(),
    }
    for key, expected in expected_scalars.items():
        if metadata.get(key) != expected:
            raise CacheValidationError(
                f"cache shard is stale or incompatible: {source_path.name} ({key})"
            )

    try:
        observations = np.load(shard_dir / "observations.npy", allow_pickle=False)
        labels = np.load(shard_dir / "labels.npy", allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise CacheValidationError(f"cache shard is corrupt: {source_path.name}") from exc

    expected_shape = (int(metadata.get("row_count", -1)), NUM_OBSERVATION_FEATURES)
    if observations.dtype != np.float32 or observations.shape != expected_shape:
        raise CacheValidationError(f"cache shard is corrupt: {source_path.name} (observations)")
    if labels.dtype != np.int64 or labels.shape != (expected_shape[0],):
        raise CacheValidationError(f"cache shard is corrupt: {source_path.name} (labels)")
    if metadata.get("observation_shape") != list(observations.shape):
        raise CacheValidationError(f"cache shard is corrupt: {source_path.name} (shape metadata)")
    if metadata.get("label_shape") != list(labels.shape):
        raise CacheValidationError(f"cache shard is corrupt: {source_path.name} (label metadata)")
    if metadata.get("observations_sha256") != sha256_array(observations):
        raise CacheValidationError(f"cache shard is corrupt: {source_path.name} (observations hash)")
    if metadata.get("labels_sha256") != sha256_array(labels):
        raise CacheValidationError(f"cache shard is corrupt: {source_path.name} (labels hash)")
    return metadata


def _manifest_from_shards(shards: Iterable[dict[str, Any]]) -> dict[str, Any]:
    return {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "canonical_schema_sha256": canonical_schema_sha256(),
        "preprocessing_version": PREPROCESSING_VERSION,
        "preprocessing_fingerprint": preprocessing_fingerprint(),
        "feature_names": get_observation_feature_names(),
        "official_csv_order": list(_official_names()),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "validation_status": "valid",
        "shards": list(shards),
    }


def build_cache(
    dataset_root: Path | str,
    cache_root: Path | str,
    *,
    workers: int | None = None,
    rebuild_stale: bool = False,
) -> dict[str, Any]:
    """Build missing shards, refusing replacement of stale shards unless explicit."""
    dataset_root = Path(dataset_root)
    cache_root = Path(cache_root)
    resolved_workers = resolve_worker_count(workers)
    sources = _official_sources(dataset_root)
    shards_root = cache_root / "shards"
    shards_root.mkdir(parents=True, exist_ok=True)

    metadata_by_name: dict[str, dict[str, Any]] = {}
    sources_to_build: list[Path] = []
    for source in sources:
        shard_dir = shards_root / source.name
        if not shard_dir.exists():
            sources_to_build.append(source)
            continue
        try:
            metadata_by_name[source.name] = _validate_shard(source, shard_dir)
        except CacheValidationError as exc:
            if not rebuild_stale:
                raise CacheValidationError(
                    f"{exc}; pass --rebuild-stale to replace stale cache data"
                ) from exc
            sources_to_build.append(source)

    if sources_to_build:
        with ThreadPoolExecutor(max_workers=min(resolved_workers, len(sources_to_build))) as pool:
            built_metadata = pool.map(
                lambda source: _build_one_shard(source, shards_root),
                sources_to_build,
            )
            for source, metadata in zip(sources_to_build, built_metadata, strict=True):
                metadata_by_name[source.name] = metadata

    ordered_metadata = [metadata_by_name[name] for name in _official_names()]
    manifest = _manifest_from_shards(ordered_metadata)
    _write_json_atomic(cache_root / "cache_manifest.json", manifest)
    return validate_cache(dataset_root, cache_root)


def validate_cache(dataset_root: Path | str, cache_root: Path | str) -> dict[str, Any]:
    """Fully validate source identity, manifest compatibility, arrays, and hashes."""
    dataset_root = Path(dataset_root)
    cache_root = Path(cache_root)
    manifest = _read_json(cache_root / "cache_manifest.json")
    expected_manifest_fields = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "canonical_schema_sha256": canonical_schema_sha256(),
        "preprocessing_version": PREPROCESSING_VERSION,
        "preprocessing_fingerprint": preprocessing_fingerprint(),
        "feature_names": get_observation_feature_names(),
        "official_csv_order": list(_official_names()),
        "validation_status": "valid",
    }
    for key, expected in expected_manifest_fields.items():
        if manifest.get(key) != expected:
            raise CacheValidationError(f"cache manifest is stale or incompatible ({key})")

    sources = _official_sources(dataset_root)
    manifest_shards = manifest.get("shards")
    if not isinstance(manifest_shards, list) or len(manifest_shards) != len(sources):
        raise CacheValidationError("cache manifest is corrupt (shard count)")
    if [item.get("source_filename") for item in manifest_shards] != [p.name for p in sources]:
        raise CacheValidationError("cache manifest is corrupt (official shard order)")

    for source, declared_metadata in zip(sources, manifest_shards, strict=True):
        actual_metadata = _validate_shard(source, cache_root / "shards" / source.name)
        if actual_metadata != declared_metadata:
            raise CacheValidationError(
                f"cache manifest is stale or corrupt (metadata mismatch: {source.name})"
            )
    return manifest


def load_cached_csvs(
    dataset_root: Path | str,
    cache_root: Path | str,
    csv_paths: Iterable[Path],
    *,
    max_rows: int | None = None,
    max_rows_per_csv: int | None = None,
    sample_frac: float | None = None,
    sample_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, list[str], str]:
    """Validate and assemble selected cached shards without scaling or split state."""
    dataset_root = Path(dataset_root)
    cache_root = Path(cache_root)
    validate_cache(dataset_root, cache_root)

    observations_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    for csv_path in csv_paths:
        observations = np.load(
            cache_root / "shards" / csv_path.name / "observations.npy",
            allow_pickle=False,
        )
        labels = np.load(
            cache_root / "shards" / csv_path.name / "labels.npy",
            allow_pickle=False,
        )
        if max_rows_per_csv is not None:
            observations = observations[:max_rows_per_csv]
            labels = labels[:max_rows_per_csv]
        observations_parts.append(observations)
        label_parts.append(labels)

    observations = np.concatenate(observations_parts, axis=0)
    labels = np.concatenate(label_parts, axis=0)
    if max_rows is not None:
        observations = observations[:max_rows]
        labels = labels[:max_rows]
    if sample_frac is not None:
        if not 0.0 < sample_frac <= 1.0:
            raise ValueError("sample_frac debe estar en (0, 1].")
        indices = np.random.default_rng(sample_seed).choice(
            len(observations),
            size=int(len(observations) * sample_frac),
            replace=False,
        )
        observations = observations[indices]
        labels = labels[indices]

    manifest_hash = sha256_file(cache_root / "cache_manifest.json")
    return observations, labels, get_observation_feature_names(), manifest_hash
