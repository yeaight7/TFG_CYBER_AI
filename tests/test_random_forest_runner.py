from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pytest

from src.artifact_integrity import verify_artifact_manifest
from src.baseline_random_forest import (
    RF_SUPPORTED_RUNS,
    RandomForestRunConfig,
    load_random_forest_split,
    run_random_forest,
)
from src.cicids_cache import sha256_array
from src.qrdqn_experiment import PreparedSplit


EXPECTED_RF_RUNS = {
    "rf_random_full_s42_m42": {
        "split_mode": "random", "train_max_rows": None, "holdout_csv": None,
    },
    "rf_random_1m_s42_m42": {
        "split_mode": "random", "train_max_rows": 1_000_000, "holdout_csv": None,
    },
    "rf_day_full_s42_m42": {
        "split_mode": "day", "train_max_rows": None, "holdout_csv": None,
    },
    "rf_holdout_webattacks_m42": {
        "split_mode": "exact-holdout", "train_max_rows": None,
        "holdout_csv": "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    },
    "rf_holdout_infilteration_m42": {
        "split_mode": "exact-holdout", "train_max_rows": None,
        "holdout_csv": "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    },
    "rf_holdout_portscan_m42": {
        "split_mode": "exact-holdout", "train_max_rows": None,
        "holdout_csv": "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    },
    "rf_holdout_ddos_m42": {
        "split_mode": "exact-holdout", "train_max_rows": None,
        "holdout_csv": "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    },
}


class FakeRandomForest:
    def __init__(self, **params):
        self.params = params
        self.feature_importances_ = np.array([], dtype=np.float64)

    def fit(self, X_train: np.ndarray, _y_train: np.ndarray):
        self.feature_importances_ = np.linspace(0.0, 1.0, X_train.shape[1])
        return self

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        return (X_test[:, 0] > 0).astype(np.int64)


def _base_config(tmp_path: Path, **overrides) -> RandomForestRunConfig:
    values = {
        "artifact_root": tmp_path / "runs",
        "run_id": "rf-synthetic",
        "dataset_root": tmp_path / "dataset",
        "cache_root": tmp_path / "cache",
        "cache_policy": "require",
        "split_mode": "random",
        "split_seed": 42,
        "model_seed": 42,
    }
    values.update(overrides)
    return RandomForestRunConfig(**values)


def test_random_forest_supports_exactly_seven_locked_runs():
    assert RF_SUPPORTED_RUNS == EXPECTED_RF_RUNS


def test_random_forest_config_rejects_extra_ladder_and_non_targeted_holdouts(tmp_path: Path):
    with pytest.raises(ValueError, match="exactly 1,000,000"):
        _base_config(tmp_path, train_max_rows=250_000)
    with pytest.raises(ValueError, match="targeted holdout"):
        _base_config(
            tmp_path,
            split_mode="exact-holdout",
            holdout_csv="Wednesday-workingHours.pcap_ISCX.csv",
        )
    with pytest.raises(ValueError, match="full training partition"):
        _base_config(tmp_path, split_mode="day", train_max_rows=1_000_000)


def test_random_forest_and_qrdqn_share_matched_raw_test_hash(
    tmp_path: Path,
    synthetic_split: PreparedSplit,
):
    observed = []

    def shared_loader(qrdqn_config):
        observed.append(qrdqn_config)
        return synthetic_split

    rf_split = load_random_forest_split(
        _base_config(tmp_path, model_seed=99),
        qrdqn_split_loader=shared_loader,
    )

    assert sha256_array(rf_split.X_test) == sha256_array(synthetic_split.X_test)
    assert observed[0].split_seed == 42
    assert observed[0].model_seed == 99
    assert observed[0].split_mode == "random"


def test_synthetic_random_forest_run_is_independently_artifact_complete(
    tmp_path: Path,
    synthetic_split: PreparedSplit,
):
    config = _base_config(tmp_path, model_seed=43, monitor_interval=0.01)
    run_dir = run_random_forest(
        config,
        split_loader=lambda _config: synthetic_split,
        model_factory=lambda params: FakeRandomForest(**params),
    )

    assert verify_artifact_manifest(run_dir)["status"] == "completed"
    resolved = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    scaler = joblib.load(run_dir / "scaler.joblib")

    assert resolved["rf_params"]["random_state"] == 43
    assert resolved["split_metadata"]["test_set_sha256"] == sha256_array(
        synthetic_split.X_test
    )
    assert metrics["confusion_matrix"] == {"tn": 2, "fp": 0, "fn": 0, "tp": 2}
    np.testing.assert_allclose(scaler.mean_, synthetic_split.X_train.mean(axis=0))
    assert (run_dir / "model.joblib").is_file()
    assert (run_dir / "feature_importances.json").is_file()
    assert (run_dir / "feature_importances.csv").is_file()

