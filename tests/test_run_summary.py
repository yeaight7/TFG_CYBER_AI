from __future__ import annotations

import json

import pytest

from src.run_summary import build_run_summary


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_run_summary_exposes_hyperparameters_hardware_and_scalar_last_values(
    fresh_main_run,
):
    summary = build_run_summary(fresh_main_run)
    environment = _read_json(fresh_main_run / "environment.json")

    assert summary["schema_version"] == "1.0"
    assert summary["run"]["run_id"] == "qrdqn_main_synthetic"
    assert summary["training_hyperparameters"]["learning_rate"] == 5e-5
    assert summary["training_execution"]["actual_timesteps"] == 10
    assert summary["hardware"] == environment["hardware"]
    assert summary["metrics"] == _read_json(fresh_main_run / "metrics.json")
    assert summary["monitoring"] == _read_json(fresh_main_run / "monitoring.json")

    learning_rate = summary["tensorboard_scalars"]["train/learning_rate"][0]
    assert learning_rate["last_value"] == pytest.approx(5e-5)
    assert learning_rate["last_step"] == 10
    assert learning_rate["samples"] == 1
    assert learning_rate["csv"].endswith("train__learning_rate.csv")


def test_run_summary_indexes_every_exported_scalar(fresh_main_run):
    summary = build_run_summary(fresh_main_run)

    assert set(summary["tensorboard_scalars"]) == {
        "train/learning_rate",
        "train/loss",
        "train/synthetic_reward",
    }
    assert summary["tensorboard_scalars"]["train/loss"][0]["last_value"] == 0.25
