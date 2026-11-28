"""Optional standalone synthetic QRDQN artifact smoke."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from src.artifact_integrity import sha256_file, verify_artifact_manifest
from src.experiment_profiles import MAIN_V1_PROFILE


_SMOKE_HYPERPARAMETERS = MAIN_V1_PROFILE.qrdqn_hyperparams()
DEFAULT_SMOKE_TIMESTEPS = int(
    _SMOKE_HYPERPARAMETERS["learning_starts"]
    + 2 * _SMOKE_HYPERPARAMETERS["train_freq"]
)
DEFAULT_SMOKE_MONITOR_INTERVAL_SECONDS = 1.0
REQUIRED_SMOKE_SCALARS = frozenset({"train/learning_rate", "train/loss"})


class PreflightError(RuntimeError):
    """The standalone smoke could not validate its artifacts."""


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PreflightError(f"Cannot read {label}: {path}") from error
    if not isinstance(value, dict):
        raise PreflightError(f"{label} must be a JSON object")
    return value


def run_synthetic_artifact_smoke(
    *,
    artifact_root: Path,
    model_factory: Callable[..., Any] | None = None,
    smoke_timesteps: int = DEFAULT_SMOKE_TIMESTEPS,
    monitor_interval: float = DEFAULT_SMOKE_MONITOR_INTERVAL_SECONDS,
) -> dict[str, Any]:
    if smoke_timesteps < DEFAULT_SMOKE_TIMESTEPS:
        raise ValueError(
            f"smoke_timesteps must be at least {DEFAULT_SMOKE_TIMESTEPS} "
            "to produce real training scalars"
        )
    if monitor_interval <= 0:
        raise ValueError("monitor_interval must be greater than zero")
    import numpy as np

    from src.qrdqn_experiment import PreparedSplit, QRDQNRunConfig, run_qrdqn_experiment

    smoke_id = f"preflight-{_utc_now().strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:8]}"
    campaign_dir = Path(artifact_root).resolve() / "preflight-smoke" / smoke_id
    attempt_root = campaign_dir / "attempts" / "preflight_qrdqn"
    X_train = np.zeros((8, 152), dtype=np.float32)
    X_test = np.zeros((4, 152), dtype=np.float32)
    X_train[:, 0] = np.array([-4, -3, -2, -1, 1, 2, 3, 4], dtype=np.float32)
    X_test[:, 0] = np.array([-5, -1, 1, 5], dtype=np.float32)
    split = PreparedSplit(
        X_train=X_train,
        y_train=np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64),
        X_test=X_test,
        y_test=np.array([0, 0, 1, 1], dtype=np.int64),
        feature_names=[f"feature_{index}" for index in range(152)],
        metadata={"split_mode": "random", "split_seed": 42, "synthetic": True},
    )
    config = QRDQNRunConfig(
        artifact_root=attempt_root,
        run_id="attempt-1",
        dataset_root=campaign_dir / "synthetic-dataset",
        cache_root=None,
        cache_policy="off",
        split_seed=42,
        model_seed=42,
        timesteps=smoke_timesteps,
        checkpoint_freq=0,
        monitor_interval=monitor_interval,
        torch_threads=1,
        torch_inter_op_threads=1,
        verbose=0,
        campaign_id="preflight-smoke",
        logical_run_id="preflight_qrdqn",
        attempt=1,
    )
    kwargs: dict[str, Any] = {"split_loader": lambda _config: split}
    if model_factory is not None:
        kwargs["model_factory"] = model_factory
    run_dir = run_qrdqn_experiment(config, **kwargs)
    verified = verify_artifact_manifest(run_dir)
    environment = _read_json(run_dir / "environment.json", label="smoke environment")
    run_summary = _read_json(run_dir / "run_summary.json", label="smoke run summary")
    scalar_tags = set(run_summary.get("tensorboard_scalars", {}))
    missing_scalars = sorted(REQUIRED_SMOKE_SCALARS - scalar_tags)
    if missing_scalars:
        raise PreflightError(
            f"Synthetic smoke missing required training scalars: {missing_scalars}"
        )
    return {
        "status": "passed",
        "workload": "synthetic-qrdqn-training-artifact-v2",
        "scientific_result": False,
        "requested_timesteps": smoke_timesteps,
        "monitor_interval_seconds": monitor_interval,
        "campaign_dir": str(campaign_dir),
        "run_dir": str(run_dir),
        "manifest_sha256": sha256_file(run_dir / "artifact_manifest.json"),
        "verified_schema_version": verified["schema_version"],
        "device_selected": environment.get("device_selected"),
        "run_summary": run_summary,
    }
