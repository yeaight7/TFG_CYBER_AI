from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import src.cicids_cache as cache
from src.load_cicids2017 import (
    CICIDSLoadConfig,
    _OFFICIAL_CICIDS2017_CSV_NAMES,
    load_cicids2017_binary,
    load_cicids2017_csv_split,
    load_cicids2017_exact_csv_split,
    load_cicids2017_split,
)


def _write_official_csvs(dataset_root: Path) -> None:
    dataset_root.mkdir(parents=True)
    for file_index, filename in enumerate(_OFFICIAL_CICIDS2017_CSV_NAMES):
        rows = [
            "Flow Duration,Total Fwd Packets,Flow Bytes/s,Source IP,Label",
            f"{file_index + 1},{file_index + 2},{file_index + 3},10.0.0.1,BENIGN",
            f"{file_index + 4},{file_index + 5},{file_index + 6},10.0.0.2,ATTACK",
            f"{file_index + 7},{file_index + 8},Infinity,10.0.0.3,BENIGN",
        ]
        (dataset_root / filename).write_text("\n".join(rows) + "\n", encoding="utf-8")


@pytest.fixture
def synthetic_dataset(tmp_path: Path) -> Path:
    dataset_root = tmp_path / "dataset"
    _write_official_csvs(dataset_root)
    return dataset_root


def _uncached_config(dataset_root: Path) -> CICIDSLoadConfig:
    return CICIDSLoadConfig(
        local_dir=dataset_root,
        scale=False,
        use_canonical=True,
        cache_policy="off",
    )


def _cached_config(dataset_root: Path, cache_root: Path) -> CICIDSLoadConfig:
    return CICIDSLoadConfig(
        local_dir=dataset_root,
        scale=False,
        use_canonical=True,
        cache_root=cache_root,
        cache_policy="require",
    )


def _assert_loader_results_equal(uncached: tuple, cached: tuple) -> None:
    for uncached_array, cached_array in zip(uncached[:4], cached[:4], strict=True):
        np.testing.assert_array_equal(uncached_array, cached_array)
        assert uncached_array.tobytes() == cached_array.tobytes()
    assert uncached[4] is None and cached[4] is None
    assert uncached[5] == cached[5]


def _nested_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return set(value) | {key for item in value.values() for key in _nested_keys(item)}
    if isinstance(value, list):
        return {key for item in value for key in _nested_keys(item)}
    return set()


def test_build_validate_manifest_and_shard_contract(
    synthetic_dataset: Path,
    tmp_path: Path,
) -> None:
    cache_root = tmp_path / "configurable-cache-root"

    built = cache.build_cache(synthetic_dataset, cache_root, workers=2)
    validated = cache.validate_cache(synthetic_dataset, cache_root)

    assert built == validated
    assert built["cache_schema_version"] == cache.CACHE_SCHEMA_VERSION
    assert built["validation_status"] == "valid"
    assert built["official_csv_order"] == list(_OFFICIAL_CICIDS2017_CSV_NAMES)
    assert [shard["source_filename"] for shard in built["shards"]] == list(
        _OFFICIAL_CICIDS2017_CSV_NAMES
    )
    assert len(built["canonical_schema_sha256"]) == 64
    assert len(built["preprocessing_fingerprint"]) == 64

    forbidden = {"scaler", "scaled_arrays", "predictions", "model_state", "train_split", "test_split"}
    all_keys = _nested_keys(built)
    for filename in _OFFICIAL_CICIDS2017_CSV_NAMES:
        shard_dir = cache_root / "shards" / filename
        observations = np.load(shard_dir / "observations.npy", allow_pickle=False)
        labels = np.load(shard_dir / "labels.npy", allow_pickle=False)
        metadata = json.loads((shard_dir / "metadata.json").read_text(encoding="utf-8"))

        assert observations.dtype == np.float32
        assert observations.shape == (3, 152)
        assert labels.dtype == np.int64
        assert labels.shape == (3,)
        assert metadata["source_filename"] == filename
        assert metadata["observation_shape"] == [3, 152]
        assert metadata["label_shape"] == [3]
        assert metadata["producer_git"]["commit"]
        assert isinstance(metadata["producer_git"]["dirty"], bool)
        all_keys.update(_nested_keys(metadata))

    assert forbidden.isdisjoint(all_keys)


@pytest.mark.parametrize("loader_kind", ["random", "day", "exact"])
def test_cached_and_uncached_loaders_are_byte_identical(
    loader_kind: str,
    synthetic_dataset: Path,
    tmp_path: Path,
) -> None:
    cache_root = tmp_path / "cache"
    cache.build_cache(synthetic_dataset, cache_root, workers=1)
    uncached_cfg = _uncached_config(synthetic_dataset)
    cached_cfg = _cached_config(synthetic_dataset, cache_root)

    if loader_kind == "random":
        uncached = load_cicids2017_binary(uncached_cfg)
        cached = load_cicids2017_binary(cached_cfg)
    elif loader_kind == "day":
        uncached = load_cicids2017_csv_split(["Monday", "Tuesday"], ["Friday"], uncached_cfg)
        cached = load_cicids2017_csv_split(["Monday", "Tuesday"], ["Friday"], cached_cfg)
    else:
        train = list(_OFFICIAL_CICIDS2017_CSV_NAMES[:2])
        test = [_OFFICIAL_CICIDS2017_CSV_NAMES[-1]]
        uncached = load_cicids2017_exact_csv_split(train, test, uncached_cfg, max_rows_per_csv=2)
        cached = load_cicids2017_exact_csv_split(train, test, cached_cfg, max_rows_per_csv=2)

    _assert_loader_results_equal(uncached, cached)


@pytest.mark.parametrize(
    ("metadata_key", "replacement"),
    [
        ("canonical_schema_sha256", "0" * 64),
        ("preprocessing_fingerprint", "1" * 64),
        ("observations_sha256", "2" * 64),
    ],
)
def test_validate_rejects_schema_preprocessing_and_array_hash_corruption(
    metadata_key: str,
    replacement: str,
    synthetic_dataset: Path,
    tmp_path: Path,
) -> None:
    cache_root = tmp_path / "cache"
    cache.build_cache(synthetic_dataset, cache_root, workers=1)
    metadata_path = (
        cache_root / "shards" / _OFFICIAL_CICIDS2017_CSV_NAMES[0] / "metadata.json"
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata[metadata_key] = replacement
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(cache.CacheValidationError, match="stale|corrupt|incompatible"):
        cache.validate_cache(synthetic_dataset, cache_root)


def test_source_change_requires_explicit_rebuild(
    synthetic_dataset: Path,
    tmp_path: Path,
) -> None:
    cache_root = tmp_path / "cache"
    cache.build_cache(synthetic_dataset, cache_root, workers=1)
    changed_source = synthetic_dataset / _OFFICIAL_CICIDS2017_CSV_NAMES[0]
    changed_source.write_text(
        changed_source.read_text(encoding="utf-8").replace("1,2,3", "101,102,103"),
        encoding="utf-8",
    )

    with pytest.raises(cache.CacheValidationError, match="--rebuild-stale"):
        cache.build_cache(synthetic_dataset, cache_root, workers=1)

    rebuilt = cache.build_cache(
        synthetic_dataset,
        cache_root,
        workers=1,
        rebuild_stale=True,
    )
    assert rebuilt == cache.validate_cache(synthetic_dataset, cache_root)


def test_atomic_failure_leaves_no_partial_shard_or_temp_directory(
    synthetic_dataset: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache_root = tmp_path / "cache"

    def fail_processing(*args, **kwargs):
        raise RuntimeError("synthetic processing failure")

    monkeypatch.setattr(cache, "_process_source_csv", fail_processing)

    with pytest.raises(RuntimeError, match="synthetic processing failure"):
        cache.build_cache(synthetic_dataset, cache_root, workers=1)

    assert not (cache_root / "cache_manifest.json").exists()
    assert not any((cache_root / "shards").glob("*.tmp-*"))
    assert not any((cache_root / "shards").iterdir())


def test_prefer_falls_back_but_require_rejects_invalid_cache(
    synthetic_dataset: Path,
    tmp_path: Path,
) -> None:
    missing_cache = tmp_path / "missing-cache"
    prefer = CICIDSLoadConfig(
        local_dir=synthetic_dataset,
        scale=False,
        use_canonical=True,
        cache_root=missing_cache,
        cache_policy="prefer",
    )
    require = CICIDSLoadConfig(
        local_dir=synthetic_dataset,
        scale=False,
        use_canonical=True,
        cache_root=missing_cache,
        cache_policy="require",
    )

    preferred = load_cicids2017_binary(prefer)
    uncached = load_cicids2017_binary(_uncached_config(synthetic_dataset))
    _assert_loader_results_equal(uncached, preferred)
    with pytest.raises(cache.CacheValidationError):
        load_cicids2017_binary(require)


def test_worker_default_and_validation_are_safely_capped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cache.os, "cpu_count", lambda: 64)
    assert cache.default_worker_count() == 8
    assert cache.resolve_worker_count(None) == 8
    assert cache.resolve_worker_count(1) == 1
    with pytest.raises(ValueError, match="between 1 and 8"):
        cache.resolve_worker_count(9)


def test_unified_loader_records_validated_cache_identity(
    synthetic_dataset: Path,
    tmp_path: Path,
) -> None:
    cache_root = tmp_path / "cache"
    cache.build_cache(synthetic_dataset, cache_root, workers=1)

    uncached = load_cicids2017_split(
        preset="full",
        max_rows=20,
        scale=False,
        local_dir=synthetic_dataset,
        cache_policy="off",
    )
    cached = load_cicids2017_split(
        preset="full",
        max_rows=20,
        scale=False,
        local_dir=synthetic_dataset,
        cache_root=cache_root,
        cache_policy="require",
    )

    _assert_loader_results_equal(uncached, cached)
    assert uncached[6]["cache_manifest_sha256"] is None
    assert len(cached[6]["cache_manifest_sha256"]) == 64
    for key in ("train_set_sha256", "test_set_sha256", "y_train_sha256", "y_test_sha256"):
        assert uncached[6][key] == cached[6][key]


def test_build_and_validate_cli(
    synthetic_dataset: Path,
    tmp_path: Path,
) -> None:
    from scripts.build_cicids_cache import main

    cache_root = tmp_path / "cli-cache"
    assert main(
        [
            "build",
            "--dataset-root",
            str(synthetic_dataset),
            "--cache-root",
            str(cache_root),
            "--workers",
            "1",
        ]
    ) == 0
    assert main(
        [
            "validate",
            "--dataset-root",
            str(synthetic_dataset),
            "--cache-root",
            str(cache_root),
        ]
    ) == 0
