from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pytest

from src.artifact_integrity import verify_artifact_manifest
from src.experiment_profiles import MAIN_V1_PROFILE
from src.qrdqn_experiment import (
    PreparedSplit,
    QRDQNRunConfig,
    load_experiment_split,
    resolved_scientific_profile,
    run_qrdqn_experiment,
)
from tests.conftest import FakeQRDQN


@pytest.mark.parametrize("split_mode", ["random", "day", "exact-holdout"])
def test_all_split_modes_use_exact_main_v1_profile(tmp_path: Path, split_mode: str):
    config = QRDQNRunConfig(
        artifact_root=tmp_path,
        run_id=f"run-{split_mode}",
        dataset_root=tmp_path / "dataset",
        cache_root=tmp_path / "cache",
        split_mode=split_mode,
        holdout_csv="held-out.csv" if split_mode == "exact-holdout" else None,
        timesteps=1,
    )

    assert resolved_scientific_profile(config) == MAIN_V1_PROFILE.to_dict()


def test_random_loader_consumes_split_seed_not_model_seed(tmp_path: Path, monkeypatch):
    calls = []

    def fake_loader(**kwargs):
        calls.append(kwargs)
        arrays = np.zeros((2, 152), dtype=np.float32)
        labels = np.array([0, 1], dtype=np.int64)
        return arrays, labels, arrays.copy(), labels.copy(), None, ["f"] * 152, {}

    monkeypatch.setattr("src.qrdqn_experiment.load_cicids2017_split", fake_loader)
    base = dict(
        artifact_root=tmp_path,
        dataset_root=tmp_path / "dataset",
        cache_root=tmp_path / "cache",
        split_mode="random",
        split_seed=7,
        timesteps=1,
    )
    load_experiment_split(QRDQNRunConfig(run_id="m42", model_seed=42, **base))
    load_experiment_split(QRDQNRunConfig(run_id="m99", model_seed=99, **base))

    assert [call["split_seed"] for call in calls] == [7, 7]
    assert all("model_seed" not in call for call in calls)


def test_tiny_run_is_artifact_complete_and_scaler_is_train_only(
    tmp_path: Path,
    synthetic_split: PreparedSplit,
    fake_model_factory,
):
    config = QRDQNRunConfig(
        artifact_root=tmp_path,
        run_id="tiny-complete",
        dataset_root=tmp_path / "dataset",
        cache_root=tmp_path / "cache",
        split_seed=42,
        model_seed=43,
        timesteps=2,
        checkpoint_freq=1,
        checkpoint_keep=2,
        monitor_interval=0.01,
    )

    run_dir = run_qrdqn_experiment(
        config,
        split_loader=lambda _config: synthetic_split,
        model_factory=fake_model_factory,
    )

    assert verify_artifact_manifest(run_dir)["status"] == "completed"
    manifest = json.loads((run_dir / "artifact_manifest.json").read_text(encoding="utf-8"))
    resolved = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    scaler = joblib.load(run_dir / "scaler.joblib")
    predictions = np.load(run_dir / "predictions.npz")

    assert manifest["run"]["split_seed"] == 42
    assert manifest["run"]["model_seed"] == 43
    assert resolved["profile_id"] == "main-v1"
    assert resolved["profile_hash"] == MAIN_V1_PROFILE.content_hash
    np.testing.assert_allclose(scaler.mean_, synthetic_split.X_train.mean(axis=0))
    assert not np.isclose(scaler.mean_[0], np.vstack([synthetic_split.X_train, synthetic_split.X_test])[:, 0].mean())
    np.testing.assert_array_equal(predictions["y_true"], synthetic_split.y_test)
    assert len(list((run_dir / "checkpoints").glob("*.zip"))) == 2
    assert (run_dir / "tensorboard_scalars" / "tensorboard_scalar_export_manifest.json").is_file()


def test_failed_training_persists_failed_attempt_evidence(
    tmp_path: Path,
    synthetic_split: PreparedSplit,
):
    config = QRDQNRunConfig(
        artifact_root=tmp_path,
        run_id="tiny-failed",
        dataset_root=tmp_path / "dataset",
        cache_root=tmp_path / "cache",
        timesteps=1,
        monitor_interval=0.01,
    )

    with pytest.raises(RuntimeError, match="synthetic training failure"):
        run_qrdqn_experiment(
            config,
            split_loader=lambda _config: synthetic_split,
            model_factory=lambda _config, _env, tensorboard_dir, _device: FakeQRDQN(
                tensorboard_dir, fail=True
            ),
        )

    run_dir = tmp_path / "tiny-failed"
    assert verify_artifact_manifest(run_dir, require_completed=False)["status"] == "failed"
    assert json.loads((run_dir / "error.json").read_text(encoding="utf-8"))["type"] == "RuntimeError"
    assert (run_dir / "monitoring.json").is_file()
    assert (run_dir / "stdout.log").is_file()
    assert (run_dir / "stderr.log").is_file()
