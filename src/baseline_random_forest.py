"""Artifact-complete single-run Random Forest comparison runner."""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

if __package__ in {None, ""}:  # pragma: no cover - direct script execution
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from src.cicids_cache import sha256_array
    from src.metrics_utils import confusion_to_metrics
    from src.qrdqn_experiment import (
        PreparedSplit,
        QRDQNRunConfig,
        load_experiment_split,
    )
    from src.resource_monitor import ResourceMonitor
    from src.run_artifacts import (
        ArtifactManifestWriter,
        ArtifactRequirement,
        TimingRecorder,
        atomic_write_json,
        atomic_write_text,
        collect_environment_metadata,
        tee_output,
    )
    from src.validate_leave_one_csv_out import TARGETED_QRDQN_HOLDOUTS
except ModuleNotFoundError:  # pragma: no cover - direct ``python src/...`` execution
    from cicids_cache import sha256_array
    from metrics_utils import confusion_to_metrics
    from qrdqn_experiment import PreparedSplit, QRDQNRunConfig, load_experiment_split
    from resource_monitor import ResourceMonitor
    from run_artifacts import (
        ArtifactManifestWriter,
        ArtifactRequirement,
        TimingRecorder,
        atomic_write_json,
        atomic_write_text,
        collect_environment_metadata,
        tee_output,
    )
    from validate_leave_one_csv_out import TARGETED_QRDQN_HOLDOUTS


_REPO_ROOT = Path(__file__).resolve().parent.parent
_TARGETED_HOLDOUTS = tuple(TARGETED_QRDQN_HOLDOUTS.values())
_RF_N_ESTIMATORS = 200

RF_SUPPORTED_RUNS = {
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


@dataclass(frozen=True)
class RandomForestRunConfig:
    artifact_root: Path
    run_id: str
    dataset_root: Path
    cache_root: Path | None
    cache_policy: str = "require"
    split_mode: str = "random"
    split_seed: int = 42
    model_seed: int = 42
    train_max_rows: int | None = None
    holdout_csv: str | None = None
    n_jobs: int = -1
    monitor_interval: float = 30.0
    attempt: int = 1
    campaign_id: str | None = None
    logical_run_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "artifact_root", Path(self.artifact_root))
        object.__setattr__(self, "dataset_root", Path(self.dataset_root))
        if self.cache_root is not None:
            object.__setattr__(self, "cache_root", Path(self.cache_root))
        if not self.run_id or Path(self.run_id).name != self.run_id:
            raise ValueError("run_id must be one non-empty path component")
        if self.cache_policy not in {"off", "prefer", "require"}:
            raise ValueError("cache_policy must be off, prefer, or require")
        if self.cache_policy == "require" and self.cache_root is None:
            raise ValueError("cache_root is required when cache_policy='require'")
        if self.split_mode not in {"random", "day", "exact-holdout"}:
            raise ValueError("split_mode must be random, day, or exact-holdout")
        if self.split_mode == "random":
            if self.holdout_csv is not None:
                raise ValueError("holdout_csv is valid only for exact-holdout")
            if self.train_max_rows not in {None, 1_000_000}:
                raise ValueError("Random Forest train_max_rows must be exactly 1,000,000 or omitted")
        elif self.split_mode == "day":
            if self.holdout_csv is not None:
                raise ValueError("holdout_csv is valid only for exact-holdout")
            if self.train_max_rows is not None:
                raise ValueError("Random Forest day mode requires the full training partition")
        else:
            if self.holdout_csv not in _TARGETED_HOLDOUTS:
                raise ValueError("Random Forest exact-holdout requires a targeted holdout filename")
            if self.train_max_rows is not None:
                raise ValueError("Random Forest holdouts require the full training partition")
        if self.n_jobs == 0:
            raise ValueError("n_jobs cannot be zero")
        if self.monitor_interval <= 0:
            raise ValueError("monitor_interval must be greater than zero")


SplitLoader = Callable[[RandomForestRunConfig], PreparedSplit]
ModelFactory = Callable[[dict[str, Any]], Any]


def load_random_forest_split(
    config: RandomForestRunConfig,
    *,
    qrdqn_split_loader: Callable[[QRDQNRunConfig], PreparedSplit] = load_experiment_split,
) -> PreparedSplit:
    """Reuse the Phase 4 canonical unscaled partition contract without model coupling."""
    shared_config = QRDQNRunConfig(
        artifact_root=config.artifact_root,
        run_id=config.run_id,
        dataset_root=config.dataset_root,
        cache_root=config.cache_root,
        cache_policy=config.cache_policy,
        split_mode=config.split_mode,
        split_seed=config.split_seed,
        model_seed=config.model_seed,
        timesteps=1,
        train_max_rows=config.train_max_rows,
        holdout_csv=config.holdout_csv,
        checkpoint_freq=0,
    )
    return qrdqn_split_loader(shared_config)


def _rf_params(config: RandomForestRunConfig) -> dict[str, Any]:
    return {
        "n_estimators": _RF_N_ESTIMATORS,
        "max_depth": None,
        "n_jobs": config.n_jobs,
        "class_weight": "balanced",
        "random_state": config.model_seed,
    }


def _requirements() -> dict[str, ArtifactRequirement]:
    return {
        "config": ArtifactRequirement("config.json"),
        "metrics": ArtifactRequirement("metrics.json"),
        "environment": ArtifactRequirement("environment.json"),
        "model": ArtifactRequirement("model.joblib"),
        "scaler": ArtifactRequirement("scaler.joblib"),
        "feature_names": ArtifactRequirement("feature_names.json"),
        "feature_importances_json": ArtifactRequirement("feature_importances.json"),
        "feature_importances_csv": ArtifactRequirement("feature_importances.csv"),
        "predictions": ArtifactRequirement("predictions.npz"),
        "timing": ArtifactRequirement("timing.json"),
        "system_metrics": ArtifactRequirement("system_metrics.csv"),
        "monitoring": ArtifactRequirement("monitoring.json"),
        "stdout": ArtifactRequirement("stdout.log"),
        "stderr": ArtifactRequirement("stderr.log"),
    }


def _json_config(config: RandomForestRunConfig) -> dict[str, Any]:
    payload = asdict(config)
    for key in ("artifact_root", "dataset_root", "cache_root"):
        value = payload[key]
        payload[key] = None if value is None else str(value)
    return payload


def _split_metadata(config: RandomForestRunConfig, split: PreparedSplit) -> dict[str, Any]:
    metadata = dict(split.metadata)
    metadata.update(
        {
            "split_mode": config.split_mode,
            "split_seed": config.split_seed,
            "split_seed_applies_to_partition": config.split_mode == "random",
            "model_seed_applies_to_partition": False,
            "train_max_rows": config.train_max_rows,
            "holdout_csv": config.holdout_csv,
            "train_set_sha256": sha256_array(split.X_train),
            "y_train_sha256": sha256_array(split.y_train),
            "test_set_sha256": sha256_array(split.X_test),
            "y_test_sha256": sha256_array(split.y_test),
            "array_hash_contract": "canonical_unscaled_v1",
            "n_train": int(len(split.y_train)),
            "n_test": int(len(split.y_test)),
        }
    )
    return metadata


def _write_feature_importances(
    run_dir: Path,
    feature_names: list[str],
    importances: np.ndarray,
) -> None:
    if len(feature_names) != len(importances):
        raise ValueError("Random Forest feature importances do not match feature names")
    rows = [
        {"feature": name, "importance": float(importance)}
        for name, importance in zip(feature_names, importances, strict=True)
    ]
    atomic_write_json(run_dir / "feature_importances.json", rows)
    csv_path = run_dir / "feature_importances.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["feature", "importance"])
        writer.writeheader()
        writer.writerows(rows)


def run_random_forest(
    config: RandomForestRunConfig,
    *,
    split_loader: SplitLoader = load_random_forest_split,
    model_factory: ModelFactory = lambda params: RandomForestClassifier(**params),
    monitor_factory: Callable[..., ResourceMonitor] = ResourceMonitor,
) -> Path:
    """Execute one allowed RF comparison run and seal schema-3 evidence."""
    run_dir = config.artifact_root / config.run_id
    run_metadata = {
        "campaign_id": config.campaign_id,
        "logical_run_id": config.logical_run_id or config.run_id,
        "physical_run_id": config.run_id,
        "attempt": config.attempt,
        "split_seed": config.split_seed,
        "model_seed": config.model_seed,
        "algorithm": "RandomForest",
    }
    manifest_writer = ArtifactManifestWriter(
        run_dir,
        run_metadata=run_metadata,
        requirements=_requirements(),
    )
    manifest_writer.start()
    atomic_write_text(run_dir / "stdout.log", "")
    atomic_write_text(run_dir / "stderr.log", "")
    requested = {
        "status": "running",
        "run_id": config.run_id,
        "algorithm": "RandomForest",
        "request": _json_config(config),
        "argv": list(sys.argv),
        "resolved_command": [sys.executable, *sys.argv],
        "split_seed": config.split_seed,
        "model_seed": config.model_seed,
        "rf_params": _rf_params(config),
    }
    atomic_write_json(run_dir / "config.json", requested)
    timing = TimingRecorder()
    monitor: ResourceMonitor | None = None
    monitor_stopped = False

    try:
        with tee_output(run_dir / "stdout.log", run_dir / "stderr.log"):
            atomic_write_json(
                run_dir / "environment.json",
                collect_environment_metadata(repo_root=_REPO_ROOT),
            )
            monitor = monitor_factory(run_dir, interval_seconds=config.monitor_interval)
            monitor.start()

            with timing.measure("preprocessing") as measurement:
                raw_split = split_loader(config)
                if raw_split.X_train.shape[1] != 152 or raw_split.X_test.shape[1] != 152:
                    raise ValueError("Canonical Random Forest observations must have 152 columns")
                split_metadata = _split_metadata(config, raw_split)
                scaler = StandardScaler()
                X_train = scaler.fit_transform(raw_split.X_train).astype(np.float32)
                X_test = scaler.transform(raw_split.X_test).astype(np.float32)
                measurement.set_units(len(X_train) + len(X_test), "rows")

            joblib.dump(scaler, run_dir / "scaler.joblib")
            atomic_write_json(run_dir / "feature_names.json", raw_split.feature_names)

            params = _rf_params(config)
            model = model_factory(params)
            with timing.measure("training") as measurement:
                model.fit(X_train, raw_split.y_train)
                measurement.set_units(len(X_train), "rows")
            joblib.dump(model, run_dir / "model.joblib")
            _write_feature_importances(
                run_dir,
                raw_split.feature_names,
                np.asarray(model.feature_importances_, dtype=np.float64),
            )

            with timing.measure("evaluation") as measurement:
                y_pred = np.asarray(model.predict(X_test), dtype=np.int64).reshape(-1)
                measurement.set_units(len(y_pred), "rows")
            y_true = raw_split.y_test.astype(np.int64)
            if len(y_true) != len(y_pred):
                raise ValueError("Prediction count does not match test-label count")
            matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
            tn, fp, fn, tp = (int(value) for value in matrix.ravel())
            metrics = confusion_to_metrics(
                tn, fp, fn, tp, undefined_metric_policy="null"
            )
            metrics.update(
                {
                    "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
                    "support": {
                        "n_test": int(len(y_true)),
                        "benign": int((y_true == 0).sum()),
                        "attack": int((y_true == 1).sum()),
                    },
                }
            )
            atomic_write_json(run_dir / "metrics.json", metrics)
            np.savez_compressed(run_dir / "predictions.npz", y_true=y_true, y_pred=y_pred)
            timing.write(run_dir / "timing.json")
            monitor.stop()
            monitor_stopped = True

            requested.update(
                {
                    "status": "completed",
                    "split_mode": config.split_mode,
                    "train_max_rows": config.train_max_rows,
                    "holdout_csv": config.holdout_csv,
                    "cache_root": None if config.cache_root is None else str(config.cache_root),
                    "cache_policy": config.cache_policy,
                    "split_metadata": split_metadata,
                    "scaler_fit": "final_training_partition_only",
                }
            )
            atomic_write_json(run_dir / "config.json", requested)

        manifest_writer.complete()
        return run_dir
    except BaseException as error:
        if monitor is not None and not monitor_stopped:
            try:
                monitor.stop()
            except Exception:
                pass
        try:
            timing.write(run_dir / "timing.json")
        except Exception:
            pass
        manifest_writer.fail(error)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single Random Forest campaign comparison run")
    parser.add_argument("--split-mode", choices=["random", "day", "exact-holdout"], required=True)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--model-seed", type=int, default=42)
    parser.add_argument("--train-max-rows", type=int, default=None)
    parser.add_argument("--holdout-csv", default=None)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/CICIDS2017"))
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--cache-policy", choices=["off", "prefer", "require"], default="require")
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--monitor-interval", type=float, default=30.0)
    parser.add_argument("--campaign-id", default=None)
    parser.add_argument("--logical-run-id", default=None)
    parser.add_argument("--attempt", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = RandomForestRunConfig(
        artifact_root=args.artifact_root,
        run_id=args.run_id,
        dataset_root=args.dataset_root,
        cache_root=args.cache_root,
        cache_policy=args.cache_policy,
        split_mode=args.split_mode,
        split_seed=args.split_seed,
        model_seed=args.model_seed,
        train_max_rows=args.train_max_rows,
        holdout_csv=args.holdout_csv,
        n_jobs=args.n_jobs,
        monitor_interval=args.monitor_interval,
        campaign_id=args.campaign_id,
        logical_run_id=args.logical_run_id,
        attempt=args.attempt,
    )
    run_dir = run_random_forest(config)
    print(f"Random Forest run complete: {run_dir}")


if __name__ == "__main__":
    main()
