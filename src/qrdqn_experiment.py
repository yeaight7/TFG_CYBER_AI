"""Reusable, artifact-complete single-run QRDQN execution API."""

from __future__ import annotations

import copy
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import torch
from sb3_contrib import QRDQN
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from src.canonical_schema import FEATURES_CANON
from src.cicids_cache import sha256_array, sha256_file as cache_sha256_file
from src.experiment_profiles import MAIN_V1_PROFILE, get_experiment_profile
from src.load_cicids2017 import (
    CICIDSLoadConfig,
    _cache_manifest_hash_if_used,
    _stratified_nested_prefix_indices,
    list_cicids2017_csv_files,
    load_cicids2017_exact_csv_split,
    load_cicids2017_split,
)
from src.metrics_utils import confusion_to_metrics
from src.resource_monitor import ResourceMonitor
from src.rl_defender_env import RLDatasetDefenderEnv
from src.run_artifacts import (
    ArtifactManifestWriter,
    ArtifactRequirement,
    TimingRecorder,
    atomic_write_json,
    atomic_write_text,
    collect_environment_metadata,
    retain_checkpoints,
    tee_output,
)
from src.tensorboard_export import export_tensorboard_scalars


_REPO_ROOT = Path(__file__).resolve().parent.parent
_N_CANON = len(FEATURES_CANON)
_SPLIT_MODES = {"random", "day", "exact-holdout"}
_CACHE_POLICIES = {"off", "prefer", "require"}


@dataclass(frozen=True)
class QRDQNRunConfig:
    artifact_root: Path
    run_id: str
    dataset_root: Path
    cache_root: Path | None
    cache_policy: str = "require"
    split_mode: str = "random"
    split_seed: int = 42
    model_seed: int = 42
    profile_id: str = "main-v1"
    timesteps: int = 3_000_000
    train_max_rows: int | None = None
    holdout_csv: str | None = None
    train_days: tuple[str, ...] | None = None
    test_days: tuple[str, ...] | None = None
    checkpoint_freq: int | None = None
    checkpoint_keep: int = 2
    monitor_interval: float = 30.0
    eval_batch_size: int = 8_192
    torch_threads: int | None = None
    torch_inter_op_threads: int | None = None
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
        if self.split_mode not in _SPLIT_MODES:
            raise ValueError(f"split_mode must be one of {sorted(_SPLIT_MODES)}")
        if self.cache_policy not in _CACHE_POLICIES:
            raise ValueError(f"cache_policy must be one of {sorted(_CACHE_POLICIES)}")
        if self.cache_policy == "require" and self.cache_root is None:
            raise ValueError("cache_root is required when cache_policy='require'")
        if self.profile_id != MAIN_V1_PROFILE.profile_id:
            raise ValueError("The Phase 4 single-run API supports only profile 'main-v1'")
        if self.timesteps <= 0:
            raise ValueError("timesteps must be greater than zero")
        if self.train_max_rows is not None and self.train_max_rows <= 0:
            raise ValueError("train_max_rows must be greater than zero")
        if self.split_mode == "exact-holdout" and not self.holdout_csv:
            raise ValueError("holdout_csv is required for split_mode='exact-holdout'")
        if self.split_mode != "exact-holdout" and self.holdout_csv is not None:
            raise ValueError("holdout_csv is valid only for split_mode='exact-holdout'")
        if self.checkpoint_freq is not None and self.checkpoint_freq < 0:
            raise ValueError("checkpoint_freq must be non-negative")
        if self.checkpoint_keep < 1:
            raise ValueError("checkpoint_keep must be at least 1")
        if self.monitor_interval <= 0:
            raise ValueError("monitor_interval must be greater than zero")
        if self.eval_batch_size <= 0:
            raise ValueError("eval_batch_size must be greater than zero")

    @property
    def effective_checkpoint_freq(self) -> int:
        if self.checkpoint_freq is not None:
            return self.checkpoint_freq
        return 500_000 if self.timesteps >= 1_000_000 else 0


@dataclass(frozen=True)
class PreparedSplit:
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    metadata: dict[str, Any]


SplitLoader = Callable[[QRDQNRunConfig], PreparedSplit]
ModelFactory = Callable[[QRDQNRunConfig, DummyVecEnv, Path, str], Any]


def resolved_scientific_profile(config: QRDQNRunConfig) -> dict[str, Any]:
    """Return the immutable scientific profile used for every split mode."""
    return get_experiment_profile(config.profile_id).to_dict()


def seed_model_rngs(model_seed: int) -> None:
    """Seed only RNGs owned by model and environment execution."""
    random.seed(model_seed)
    np.random.seed(model_seed)
    torch.manual_seed(model_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(model_seed)


def configure_torch_runtime(
    torch_threads: int | None,
    torch_inter_op_threads: int | None,
) -> dict[str, int | None]:
    if torch_threads is not None:
        if torch_threads <= 0:
            raise ValueError("torch_threads must be greater than zero")
        torch.set_num_threads(torch_threads)
    if torch_inter_op_threads is not None:
        if torch_inter_op_threads <= 0:
            raise ValueError("torch_inter_op_threads must be greater than zero")
        try:
            torch.set_num_interop_threads(torch_inter_op_threads)
        except RuntimeError:
            pass
    return {
        "requested_torch_threads": torch_threads,
        "requested_torch_inter_op_threads": torch_inter_op_threads,
        "torch_num_threads": int(torch.get_num_threads()),
        "torch_num_interop_threads": int(torch.get_num_interop_threads()),
    }


def _split_metadata(config: QRDQNRunConfig, split: PreparedSplit) -> dict[str, Any]:
    metadata = dict(split.metadata)
    metadata.update(
        {
            "split_mode": config.split_mode,
            "split_seed": config.split_seed,
            "split_seed_applies_to_partition": config.split_mode == "random",
            "model_seed_applies_to_partition": False,
            "train_set_sha256": sha256_array(split.X_train),
            "y_train_sha256": sha256_array(split.y_train),
            "test_set_sha256": sha256_array(split.X_test),
            "y_test_sha256": sha256_array(split.y_test),
            "n_train": int(len(split.y_train)),
            "n_test": int(len(split.y_test)),
            "train_benign": int((split.y_train == 0).sum()),
            "train_attack": int((split.y_train == 1).sum()),
            "test_benign": int((split.y_test == 0).sum()),
            "test_attack": int((split.y_test == 1).sum()),
            "train_benign_rate": float((split.y_train == 0).mean()),
            "train_attack_rate": float((split.y_train == 1).mean()),
            "test_benign_rate": float((split.y_test == 0).mean()),
            "test_attack_rate": float((split.y_test == 1).mean()),
            "array_hash_contract": "canonical_unscaled_v1",
            "cache_policy": config.cache_policy,
        }
    )
    return metadata


def _cache_source_provenance(config: QRDQNRunConfig) -> dict[str, Any]:
    if config.cache_root is None or config.cache_policy == "off":
        return {}
    manifest_path = config.cache_root / "cache_manifest.json"
    if not manifest_path.is_file():
        return {}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_hashes = {
        str(shard["source_filename"]): str(shard["source_sha256"])
        for shard in manifest.get("shards", [])
    }
    return {
        "cache_manifest_sha256": cache_sha256_file(manifest_path),
        "source_csv_sha256": source_hashes,
    }


def load_experiment_split(config: QRDQNRunConfig) -> PreparedSplit:
    """Load one canonical unscaled split without exposing model_seed to selection."""
    common = {
        "preset": "full",
        "split_seed": config.split_seed,
        "train_max_rows": config.train_max_rows,
        "scale": False,
        "use_canonical": True,
        "local_dir": config.dataset_root,
        "cache_root": config.cache_root,
        "cache_policy": config.cache_policy,
    }
    if config.split_mode in {"random", "day"}:
        X_train, y_train, X_test, y_test, _scaler, feature_names, metadata = (
            load_cicids2017_split(
                split_mode=config.split_mode,
                train_days=list(config.train_days) if config.train_days else None,
                test_days=list(config.test_days) if config.test_days else None,
                **common,
            )
        )
        metadata.update(_cache_source_provenance(config))
        return PreparedSplit(X_train, y_train, X_test, y_test, feature_names, metadata)

    official_paths = list_cicids2017_csv_files(config.dataset_root)
    official_names = [path.name for path in official_paths]
    assert config.holdout_csv is not None
    if config.holdout_csv not in official_names:
        raise ValueError(
            f"holdout_csv must exactly match an official CICIDS2017 CSV: {config.holdout_csv}"
        )
    train_names = [name for name in official_names if name != config.holdout_csv]
    loader_config = CICIDSLoadConfig(
        local_dir=config.dataset_root,
        use_canonical=True,
        scale=False,
        cache_root=config.cache_root,
        cache_policy=config.cache_policy,
    )
    X_train, y_train, X_test, y_test, _scaler, feature_names = (
        load_cicids2017_exact_csv_split(
            train_csv_names=train_names,
            test_csv_names=[config.holdout_csv],
            cfg=loader_config,
        )
    )
    n_train_full = int(len(y_train))
    subsample_method = None
    if config.train_max_rows is not None:
        selected = _stratified_nested_prefix_indices(
            y_train, config.train_max_rows, config.split_seed
        )
        X_train = X_train[selected]
        y_train = y_train[selected]
        subsample_method = "stratified_nested_prefix_v1"
    metadata = {
        "train_csv_files": train_names,
        "test_csv_files": [config.holdout_csv],
        "holdout_csv": config.holdout_csv,
        "n_train_full": n_train_full,
        "train_max_rows": config.train_max_rows,
        "subsample_method": subsample_method,
        "cache_manifest_sha256": _cache_manifest_hash_if_used(loader_config),
        "csv_selection_policy": "official_cicids2017_exact_holdout",
        **_cache_source_provenance(config),
    }
    return PreparedSplit(X_train, y_train, X_test, y_test, feature_names, metadata)


def _make_environment(
    X_train: np.ndarray,
    y_train: np.ndarray,
    reward_config: dict[str, float],
) -> DummyVecEnv:
    max_steps = min(10_000, len(X_train))

    def create() -> Monitor:
        return Monitor(
            RLDatasetDefenderEnv(
                X=X_train,
                y=y_train,
                benign_label=0,
                attack_label=1,
                reward_config=reward_config,
                max_steps_per_episode=max_steps,
                shuffle=True,
            )
        )

    return DummyVecEnv([create])


def create_qrdqn_model(
    config: QRDQNRunConfig,
    environment: DummyVecEnv,
    tensorboard_dir: Path,
    device: str,
) -> QRDQN:
    hyperparameters = MAIN_V1_PROFILE.qrdqn_hyperparams()
    return QRDQN(
        hyperparameters["policy"],
        environment,
        seed=config.model_seed,
        policy_kwargs=copy.deepcopy(hyperparameters["policy_kwargs"]),
        learning_rate=hyperparameters["learning_rate"],
        buffer_size=hyperparameters["buffer_size"],
        learning_starts=hyperparameters["learning_starts"],
        batch_size=hyperparameters["batch_size"],
        gamma=hyperparameters["gamma"],
        tau=hyperparameters["tau"],
        train_freq=hyperparameters["train_freq"],
        gradient_steps=hyperparameters["gradient_steps"],
        target_update_interval=hyperparameters["target_update_interval"],
        exploration_initial_eps=hyperparameters["exploration_initial_eps"],
        exploration_final_eps=hyperparameters["exploration_final_eps"],
        exploration_fraction=hyperparameters["exploration_fraction"],
        max_grad_norm=hyperparameters["max_grad_norm"],
        verbose=1,
        device=device,
        tensorboard_log=str(tensorboard_dir),
    )


def batched_predict(model: Any, observations: np.ndarray, batch_size: int) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for start in range(0, len(observations), batch_size):
        actions, _state = model.predict(
            observations[start : start + batch_size], deterministic=True
        )
        chunks.append(np.asarray(actions, dtype=np.int64).reshape(-1))
    if not chunks:
        return np.empty(0, dtype=np.int64)
    return np.concatenate(chunks)


def _requirements(checkpoints_enabled: bool) -> dict[str, ArtifactRequirement]:
    requirements = {
        "config": ArtifactRequirement("config.json"),
        "metrics": ArtifactRequirement("metrics.json"),
        "environment": ArtifactRequirement("environment.json"),
        "model": ArtifactRequirement("model.zip"),
        "scaler": ArtifactRequirement("scaler.joblib"),
        "train_percentiles": ArtifactRequirement("train_percentiles.npz"),
        "feature_names": ArtifactRequirement("feature_names.json"),
        "predictions": ArtifactRequirement("predictions.npz"),
        "timing": ArtifactRequirement("timing.json"),
        "system_metrics": ArtifactRequirement("system_metrics.csv"),
        "monitoring": ArtifactRequirement("monitoring.json"),
        "stdout": ArtifactRequirement("stdout.log"),
        "stderr": ArtifactRequirement("stderr.log"),
        "tensorboard": ArtifactRequirement("tensorboard", kind="directory"),
        "tensorboard_scalars": ArtifactRequirement(
            "tensorboard_scalars", kind="directory"
        ),
    }
    if checkpoints_enabled:
        requirements["checkpoints"] = ArtifactRequirement("checkpoints", kind="directory")
    return requirements


def _json_config(config: QRDQNRunConfig) -> dict[str, Any]:
    payload = asdict(config)
    for key in ("artifact_root", "dataset_root", "cache_root"):
        value = payload[key]
        payload[key] = None if value is None else str(value)
    return payload


def run_qrdqn_experiment(
    config: QRDQNRunConfig,
    *,
    split_loader: SplitLoader = load_experiment_split,
    model_factory: ModelFactory = create_qrdqn_model,
    monitor_factory: Callable[..., ResourceMonitor] = ResourceMonitor,
) -> Path:
    """Execute one QRDQN run and seal schema-3 evidence or failed-run evidence."""
    profile = get_experiment_profile(config.profile_id)
    run_dir = config.artifact_root / config.run_id
    checkpoint_freq = config.effective_checkpoint_freq
    checkpoints_enabled = checkpoint_freq > 0
    run_metadata = {
        "campaign_id": config.campaign_id,
        "logical_run_id": config.logical_run_id or config.run_id,
        "physical_run_id": config.run_id,
        "attempt": config.attempt,
        "split_seed": config.split_seed,
        "model_seed": config.model_seed,
        "profile_id": profile.profile_id,
        "profile_hash": profile.content_hash,
    }
    manifest_writer = ArtifactManifestWriter(
        run_dir,
        run_metadata=run_metadata,
        requirements=_requirements(checkpoints_enabled),
    )
    manifest_writer.start()
    atomic_write_text(run_dir / "stdout.log", "")
    atomic_write_text(run_dir / "stderr.log", "")
    requested_config = {
        "status": "running",
        "run_id": config.run_id,
        "algorithm": "QRDQN",
        "profile_id": profile.profile_id,
        "profile_hash": profile.content_hash,
        "profile": profile.to_dict(),
        "request": _json_config(config),
        "argv": list(sys.argv),
        "resolved_command": [sys.executable, *sys.argv],
        "split_seed": config.split_seed,
        "model_seed": config.model_seed,
    }
    atomic_write_json(run_dir / "config.json", requested_config)
    timing = TimingRecorder()
    monitor: ResourceMonitor | None = None
    monitor_stopped = False
    environment: DummyVecEnv | None = None

    try:
        with tee_output(run_dir / "stdout.log", run_dir / "stderr.log"):
            seed_model_rngs(config.model_seed)
            torch_runtime = configure_torch_runtime(
                config.torch_threads, config.torch_inter_op_threads
            )
            environment_metadata = collect_environment_metadata(
                repo_root=_REPO_ROOT,
                requested_torch_threads=config.torch_threads,
                requested_torch_interop_threads=config.torch_inter_op_threads,
            )
            environment_metadata["device_selected"] = (
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            atomic_write_json(run_dir / "environment.json", environment_metadata)

            monitor = monitor_factory(run_dir, interval_seconds=config.monitor_interval)
            monitor.start()

            with timing.measure("preprocessing") as measurement:
                raw_split = split_loader(config)
                if raw_split.X_train.shape[1] != 152 or raw_split.X_test.shape[1] != 152:
                    raise ValueError("Canonical QRDQN observations must have exactly 152 columns")
                split_metadata = _split_metadata(config, raw_split)
                p_low = np.percentile(raw_split.X_train[:, :_N_CANON], 0.5, axis=0)
                p_high = np.percentile(raw_split.X_train[:, :_N_CANON], 99.5, axis=0)
                scaler = StandardScaler()
                X_train = scaler.fit_transform(raw_split.X_train).astype(np.float32)
                X_test = scaler.transform(raw_split.X_test).astype(np.float32)
                measurement.set_units(len(X_train) + len(X_test), "rows")

            joblib.dump(scaler, run_dir / "scaler.joblib")
            np.savez_compressed(run_dir / "train_percentiles.npz", p_low=p_low, p_high=p_high)
            atomic_write_json(run_dir / "feature_names.json", raw_split.feature_names)

            resolved_config = {
                **requested_config,
                "status": "running",
                "profile": profile.to_dict(),
                "training_hyperparameters": profile.qrdqn_hyperparams(),
                "reward_config": profile.reward_config(),
                "split_mode": config.split_mode,
                "split_metadata": split_metadata,
                "timesteps": config.timesteps,
                "train_max_rows": config.train_max_rows,
                "holdout_csv": config.holdout_csv,
                "cache_root": None if config.cache_root is None else str(config.cache_root),
                "cache_policy": config.cache_policy,
                "checkpoint_freq": checkpoint_freq,
                "checkpoint_keep": config.checkpoint_keep,
                "replay_buffer_persistence": False,
                "eval_batch_size": config.eval_batch_size,
                "torch_runtime": torch_runtime,
            }
            atomic_write_json(run_dir / "config.json", resolved_config)

            tensorboard_dir = run_dir / "tensorboard"
            tensorboard_dir.mkdir(parents=True, exist_ok=True)
            atomic_write_json(
                tensorboard_dir / "run_metadata.json",
                {"run_id": config.run_id, "profile_hash": profile.content_hash},
            )
            environment = _make_environment(X_train, raw_split.y_train, profile.reward_config())
            environment.seed(config.model_seed)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = model_factory(config, environment, tensorboard_dir, device)

            checkpoint_callback = None
            if checkpoints_enabled:
                checkpoints_dir = run_dir / "checkpoints"
                checkpoints_dir.mkdir(parents=True, exist_ok=True)
                checkpoint_callback = CheckpointCallback(
                    save_freq=checkpoint_freq,
                    save_path=str(checkpoints_dir),
                    name_prefix=config.run_id,
                    save_replay_buffer=False,
                    save_vecnormalize=False,
                    verbose=1,
                )

            with timing.measure("training") as measurement:
                model.learn(
                    total_timesteps=config.timesteps,
                    callback=checkpoint_callback,
                    tb_log_name=config.run_id,
                    reset_num_timesteps=False,
                )
                measurement.set_units(config.timesteps, "timesteps")

            model.save(str(run_dir / "model"))
            if not (run_dir / "model.zip").is_file():
                raise FileNotFoundError("QRDQN model.save did not produce model.zip")
            if checkpoints_enabled:
                retain_checkpoints(run_dir / "checkpoints", keep=config.checkpoint_keep)

            with timing.measure("evaluation") as measurement:
                y_pred = batched_predict(model, X_test, config.eval_batch_size)
                measurement.set_units(len(y_pred), "rows")
            y_true = raw_split.y_test.astype(np.int64)
            if len(y_true) != len(y_pred):
                raise ValueError("Prediction count does not match test-label count")
            matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
            tn, fp, fn, tp = (int(value) for value in matrix.ravel())
            metrics = confusion_to_metrics(
                tn,
                fp,
                fn,
                tp,
                reward_config=profile.reward_config(),
                undefined_metric_policy="null",
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

            event_dirs = sorted(
                {path.parent for path in tensorboard_dir.rglob("events.out.tfevents.*")}
            )
            if not event_dirs:
                raise FileNotFoundError("QRDQN run produced no TensorBoard event data")
            export_tensorboard_scalars(
                event_dirs,
                run_dir / "tensorboard_scalars",
                run_id=config.run_id,
            )
            timing.write(run_dir / "timing.json")
            monitor.stop()
            monitor_stopped = True

            resolved_config["status"] = "completed"
            resolved_config["artifact_paths"] = {
                name: requirement.relative_path
                for name, requirement in _requirements(checkpoints_enabled).items()
            }
            atomic_write_json(run_dir / "config.json", resolved_config)

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
    finally:
        if environment is not None:
            environment.close()


def config_from_run_payload(
    payload: dict[str, Any],
    *,
    dataset_root: Path | None = None,
    cache_root: Path | None = None,
) -> QRDQNRunConfig:
    """Reconstruct a load-only split config from a completed Phase 4 run config."""
    request = payload.get("request", {})
    return QRDQNRunConfig(
        artifact_root=Path(request.get("artifact_root", ".")),
        run_id=str(payload["run_id"]),
        dataset_root=dataset_root or Path(request["dataset_root"]),
        cache_root=cache_root if cache_root is not None else (
            None if request.get("cache_root") is None else Path(request["cache_root"])
        ),
        cache_policy=str(payload.get("cache_policy", request.get("cache_policy", "require"))),
        split_mode=str(payload["split_mode"]),
        split_seed=int(payload["split_seed"]),
        model_seed=int(payload["model_seed"]),
        profile_id=str(payload["profile_id"]),
        timesteps=int(payload["timesteps"]),
        train_max_rows=payload.get("train_max_rows"),
        holdout_csv=payload.get("holdout_csv"),
        train_days=None if request.get("train_days") is None else tuple(request["train_days"]),
        test_days=None if request.get("test_days") is None else tuple(request["test_days"]),
        checkpoint_freq=0,
        monitor_interval=30.0,
    )
