from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from src.artifact_integrity import verify_artifact_manifest
from src.experiment_profiles import MAIN_V1_PROFILE
from src.qrdqn_experiment import PreparedSplit
from src.validate_checks import (
    ShuffledLabelRunConfig,
    build_label_permutation,
    parse_args,
    run_shuffled_label_validation,
)


def test_shuffled_label_permutation_is_deterministic_and_hashes_indices():
    labels = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    first_indices, first_labels, first_hash = build_label_permutation(labels, seed=42)
    second_indices, second_labels, second_hash = build_label_permutation(labels, seed=42)
    third_indices, _third_labels, third_hash = build_label_permutation(labels, seed=43)

    np.testing.assert_array_equal(first_indices, second_indices)
    np.testing.assert_array_equal(first_labels, second_labels)
    np.testing.assert_array_equal(first_labels, labels[first_indices])
    assert first_hash == second_hash
    assert first_hash != third_hash
    assert not np.array_equal(first_indices, third_indices)


def test_synthetic_shuffled_label_job_is_complete_current_reward_and_auxiliary(
    tmp_path: Path,
    synthetic_split: PreparedSplit,
    fake_model_factory,
):
    config = ShuffledLabelRunConfig(
        artifact_root=tmp_path / "validation",
        run_id="shuffled-label-synthetic",
        dataset_root=tmp_path / "dataset",
        cache_root=tmp_path / "cache",
        monitor_interval=0.01,
    )
    run_dir = run_shuffled_label_validation(
        config,
        split_loader=lambda _config: synthetic_split,
        model_factory=fake_model_factory,
    )

    assert verify_artifact_manifest(run_dir)["status"] == "completed"
    resolved = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    manifest = json.loads((run_dir / "artifact_manifest.json").read_text(encoding="utf-8"))

    assert resolved["timesteps"] == 10_000
    assert resolved["split_seed"] == 42
    assert resolved["model_seed"] == 42
    assert resolved["reward_config"] == MAIN_V1_PROFILE.reward_config()
    assert resolved["job_classification"] == "auxiliary_validation"
    assert resolved["counts_toward_primary_model_training_executions"] is False
    assert len(resolved["label_permutation_sha256"]) == 64
    assert manifest["run"]["job_classification"] == "auxiliary_validation"
    assert manifest["run"]["counts_toward_primary_model_training_executions"] is False
    assert (run_dir / "model.zip").is_file()
    assert (run_dir / "scaler.joblib").is_file()
    assert (run_dir / "predictions.npz").is_file()
    assert (run_dir / "tensorboard").is_dir()


def test_check_b_cli_exposes_provider_neutral_artifact_inputs(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validate_checks.py",
            "--checks",
            "B",
            "--dataset-root",
            str(tmp_path / "dataset"),
            "--cache-root",
            str(tmp_path / "cache"),
            "--artifact-root",
            str(tmp_path / "artifacts"),
            "--run-id-b",
            "shuffled-control",
        ],
    )

    args = parse_args()

    assert args.split_seed == 42
    assert args.model_seed == 42
    assert args.shuffled_label_seed == 42
    assert args.timesteps_b == 10_000
    assert args.cache_policy == "require"
