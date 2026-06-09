"""
train_rl_defender.py — Entrenamiento de agente defensor RL sobre CICIDS2017.

Uso:
    python src/train_rl_defender.py                        # Fast preset (default), random split
    python src/train_rl_defender.py --smoke                # Alias for --preset fast
    python src/train_rl_defender.py --preset full          # Full training, all rows (~30-60 min)
    python src/train_rl_defender.py --split-mode day       # Day/CSV group split, fast preset
    python src/train_rl_defender.py --split-mode day --train-days Monday Tuesday --test-days Friday
    python src/train_rl_defender.py --preset full --split-mode day  # Full day split
    python src/train_rl_defender.py --timesteps 200000     # Custom timesteps
"""
from __future__ import annotations

import argparse
import copy
import json
import platform
import shutil
import sys
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, List
from datetime import datetime

import joblib
import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

from sb3_contrib import QRDQN
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from rl_defender_env import RLDatasetDefenderEnv
from load_cicids2017 import (
    load_cicids2017_split,
)

from canonical_schema import FEATURES_CANON
_N_CANON = len(FEATURES_CANON)


_REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = _REPO_ROOT / "models"
RUNS_DIR = _REPO_ROOT / "runs"


# --------------------------------------------------------------------------------------
# Configuración de recompensa para el agente defensor
# --------------------------------------------------------------------------------------
REWARD_CONFIG: Dict[str, float] = {
    "tp": 1.5,
    "fp": -2.0,
    "fn": -5.0,
    "omission": 0.0,
}


def resolve_total_timesteps(args: argparse.Namespace, is_fast: bool) -> int:
    if args.timesteps is not None:
        return args.timesteps
    if args.training_profile == "main-experiment":
        return 3_000_000
    return 25_000 if is_fast else 100_000


def resolve_training_hyperparams(
    training_profile: str,
    is_fast: bool,
    total_timesteps: int,
) -> Dict[str, Any]:
    if training_profile == "main-experiment":
        policy_kwargs = {
            "net_arch": [1024, 1024, 512],
            "n_quantiles": 200,
        }
        return {
            "policy": "MlpPolicy",
            "policy_kwargs": policy_kwargs,
            "learning_rate": 5e-5,
            "buffer_size": 1_000_000,
            "learning_starts": 50_000,
            "batch_size": 2048,
            "gamma": 0.0,
            "tau": 1.0,
            "train_freq": 100,
            "gradient_steps": 20,
            "target_update_interval": 10_000,
            "exploration_initial_eps": 1.0,
            "exploration_final_eps": 0.02,
            "exploration_fraction": 0.10,
            "max_grad_norm": 10.0,
        }

    policy_kwargs = {
        "net_arch": [512, 256],
        "n_quantiles": 200,
    }
    return {
        "policy": "MlpPolicy",
        "policy_kwargs": policy_kwargs,
        "learning_rate": 1e-4,
        "buffer_size": min(200_000, max(total_timesteps, 10_000)),
        "learning_starts": 100,
        "batch_size": 512 if is_fast else 2048,
        "gamma": 0.0,
        "tau": 1.0,
        "train_freq": 50 if is_fast else 100,
        "gradient_steps": 10 if is_fast else 20,
        "target_update_interval": 1_000 if is_fast else 10_000,
        "exploration_initial_eps": 1.0,
        "exploration_final_eps": 0.01,
        "exploration_fraction": 0.005,
        "max_grad_norm": None,
    }


def resolve_checkpoint_freq(
    training_profile: str,
    checkpoint_freq: int | None,
) -> int:
    if checkpoint_freq is not None:
        if checkpoint_freq < 0:
            raise ValueError("--checkpoint-freq must be >= 0")
        return checkpoint_freq
    if training_profile == "main-experiment":
        return 250_000
    return 0


def _package_version(package_name: str) -> str | None:
    try:
        return metadata.version(package_name)
    except metadata.PackageNotFoundError:
        return None


def collect_environment_metadata(device: str) -> Dict[str, Any]:
    cuda_device_name = None
    if torch.cuda.is_available():
        cuda_device_name = torch.cuda.get_device_name(0)

    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "device": device,
        "torch_version": torch.__version__,
        "torch_cuda_available": torch.cuda.is_available(),
        "torch_cuda_version": torch.version.cuda,
        "torch_cudnn_version": torch.backends.cudnn.version(),
        "cuda_device_name": cuda_device_name,
        "numpy_version": _package_version("numpy"),
        "pandas_version": _package_version("pandas"),
        "scikit_learn_version": _package_version("scikit-learn"),
        "gymnasium_version": _package_version("gymnasium"),
        "stable_baselines3_version": _package_version("stable-baselines3"),
        "sb3_contrib_version": _package_version("sb3-contrib"),
        "joblib_version": _package_version("joblib"),
    }


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def make_env_fn(
    X: np.ndarray, y: np.ndarray, reward_config: Dict[str, float], max_steps: int
):
    """Devuelve una función creadora de entornos para DummyVecEnv."""

    def _init():
        env = RLDatasetDefenderEnv(
            X=X,
            y=y,
            benign_label=0,
            attack_label=1,
            reward_config=reward_config,
            max_steps_per_episode=max_steps,
            shuffle=True,
        )
        return Monitor(env)

    return _init


def evaluate_model(
    model, X_test: np.ndarray, y_test: np.ndarray, reward_config: Dict[str, float],
) -> Dict[str, float]:
    """
    Evalúa el agente sobre test set.
    Devuelve dict con métricas clave.
    """
    env_test = RLDatasetDefenderEnv(
        X=X_test,
        y=y_test,
        benign_label=0,
        attack_label=1,
        reward_config=reward_config,
        max_steps_per_episode=len(X_test),
        shuffle=False,
    )

    obs, info = env_test.reset()
    done = False

    y_true: List[int] = []
    y_pred: List[int] = []

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env_test.step(int(action))
        done = terminated or truncated
        y_true.append(int(info["true_label"]))
        y_pred.append(int(action))

    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, digits=4, output_dict=True)

    print("\n=== Confusion matrix (0=PERMIT, 1=BLOCK) ===")
    print(cm)
    print("\n=== Classification report ===")
    print(classification_report(y_true, y_pred, digits=4))

    # Métricas clave para logging
    metrics: Dict[str, float] = {
        "accuracy": float(report["accuracy"]),
        "precision_attack": float(report.get("1", {}).get("precision", 0.0)),
        "recall_attack": float(report.get("1", {}).get("recall", 0.0)),
        "f1_attack": float(report.get("1", {}).get("f1-score", 0.0)),
        "precision_benign": float(report.get("0", {}).get("precision", 0.0)),
        "recall_benign": float(report.get("0", {}).get("recall", 0.0)),
        "f1_benign": float(report.get("0", {}).get("f1-score", 0.0)),
    }

    print("\n=== Key metrics ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RL Defender on CICIDS2017")
    parser.add_argument(
        "--smoke", action="store_true",
        help="Smoke test: alias for --preset fast (backward compat)",
    )
    parser.add_argument(
        "--preset", type=str, default="fast", choices=["fast", "full"],
        help="Preset: fast (lightweight, capped rows) or full (all rows). Default: fast",
    )
    parser.add_argument(
        "--split-mode", type=str, default="random", choices=["random", "day"],
        help="Split mode: random (stratified 80/20) or day (CSV/day group split). Default: random",
    )
    parser.add_argument(
        "--train-days", nargs="+", default=None,
        help="Day patterns for training (split-mode=day). Default: Monday Tuesday Wednesday",
    )
    parser.add_argument(
        "--test-days", nargs="+", default=None,
        help="Day patterns for testing (split-mode=day). Default: Thursday Friday",
    )
    parser.add_argument(
        "--timesteps", type=int, default=None,
        help=(
            "Total timesteps for training "
            "(default: 25k fast, 100k full, 3M main-experiment)"
        ),
    )
    parser.add_argument(
        "--max-rows", type=int, default=None,
        help="Max rows to load from dataset (overrides preset default)",
    )
    parser.add_argument(
        "--no-canonical", action="store_true",
        help="Disable canonical schema + missingness mask (use raw features)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--training-profile",
        type=str,
        default="default",
        choices=["default", "main-experiment"],
        help=(
            "Training hyperparameter profile. 'default' preserves current "
            "behavior; 'main-experiment' uses the fixed RunPod main run config."
        ),
    )
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=None,
        help=(
            "Save model checkpoints every N timesteps. 0 disables. "
            "Default: disabled for default profile, 250k for main-experiment."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ── Resolve preset (--smoke is an alias for --preset fast) ──
    preset = "fast" if args.smoke else args.preset
    is_fast = preset == "fast"
    training_profile = args.training_profile

    # ── Smoke / fast vs full defaults ──
    total_timesteps = resolve_total_timesteps(args, is_fast)

    use_canonical = not args.no_canonical
    seed = args.seed
    split_mode = args.split_mode
    hyperparams = resolve_training_hyperparams(
        training_profile=training_profile,
        is_fast=is_fast,
        total_timesteps=total_timesteps,
    )
    checkpoint_freq = resolve_checkpoint_freq(
        training_profile=training_profile,
        checkpoint_freq=args.checkpoint_freq,
    )

    # ── RUN_ID único ──
    run_started_at = datetime.now().isoformat(timespec="seconds")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    algo_tag = "qrdqn"
    canon_tag = "canonical" if use_canonical else "raw"
    exp_tag = f"{preset}_{split_mode}"
    run_prefix = "MAIN" if training_profile == "main-experiment" else "C03"
    RUN_ID = f"{run_prefix}_{algo_tag}_cicids2017_{canon_tag}_{exp_tag}_{timestamp}"

    # ── Directorios de salida ──
    run_dir = RUNS_DIR / "cicids2017" / RUN_ID
    checkpoints_dir = run_dir / "checkpoints"
    tb_log_dir = str(RUNS_DIR / "cicids2017")
    model_path = MODELS_DIR / RUN_ID
    model_zip_path = model_path.with_suffix(".zip")
    run_model_path = run_dir / "model.zip"
    scaler_path = run_dir / "scaler.joblib"
    percentiles_path = run_dir / "train_percentiles.npz"
    feature_names_path = run_dir / "feature_names.json"
    environment_path = run_dir / "environment.json"
    config_path = run_dir / "config.json"
    metrics_path = run_dir / "metrics.json"
    manifest_path = run_dir / "artifact_manifest.json"
    run_dir.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"{'='*60}")
    print(f"  Experimento: {RUN_ID}")
    print(f"  Preset: {preset.upper()}")
    print(f"  Training profile: {training_profile}")
    print(f"  Split mode: {split_mode}")
    print(f"  Max rows: {args.max_rows or '(preset default)'}")
    print(f"  Timesteps: {total_timesteps}")
    print(f"  Canonical: {use_canonical}")
    print(f"  Output: {run_dir}")
    if checkpoint_freq > 0:
        print(f"  Checkpoints: every {checkpoint_freq} steps -> {checkpoints_dir}")
    else:
        print("  Checkpoints: disabled")
    print(f"{'='*60}")
    print("\nResolved QRDQN hyperparameters:")
    print(json.dumps(hyperparams, indent=2))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        print(f"GPU detectada: {torch.cuda.get_device_name(0)}")
    else:
        print("GPU NO detectada. Se usara CPU.")
    environment_metadata = collect_environment_metadata(device)

    # ------------------------------------------------------------------
    # 1) Cargar dataset CICIDS2017 (sin escalar, para poder persistir scaler)
    # ------------------------------------------------------------------
    print("\nCargando CICIDS2017...")
    X_train, y_train, X_test, y_test, _scaler_unused, feature_names, split_meta = load_cicids2017_split(
        split_mode=split_mode,
        preset=preset,
        seed=seed,
        max_rows=args.max_rows,
        train_days=args.train_days,
        test_days=args.test_days,
        scale=False,
        use_canonical=use_canonical,
    )

    # Calcular percentiles p0.5 y p99.5 sobre X_train sin escalar
    p_low = np.percentile(X_train[:, :_N_CANON], 0.5, axis=0)   # shape (76,)
    p_high = np.percentile(X_train[:, :_N_CANON], 99.5, axis=0)

    # Ajustar y aplicar StandardScaler manualmente para persistirlo
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    print(f"Train: X={X_train.shape}, y={y_train.shape} "
          f"(benign={int((y_train==0).sum())}, attack={int((y_train==1).sum())})")
    print(f"Test:  X={X_test.shape}, y={y_test.shape} "
          f"(benign={int((y_test==0).sum())}, attack={int((y_test==1).sum())})")
    print(f"Features: {len(feature_names)}")

    joblib.dump(scaler, scaler_path)
    print(f"\nScaler guardado en: {scaler_path}")

    np.savez(percentiles_path, p_low=p_low, p_high=p_high)
    print(f"Percentiles guardados en: {percentiles_path}")

    write_json(feature_names_path, feature_names)
    print(f"Feature names guardados en: {feature_names_path}")

    write_json(environment_path, environment_metadata)
    print(f"Environment metadata guardado en: {environment_path}")

    config = {
        "run_id": RUN_ID,
        "status": "started",
        "started_at": run_started_at,
        "completed_at": None,
        "algorithm": "QRDQN",
        "policy": hyperparams["policy"],
        "training_profile": training_profile,
        "dataset": "CICIDS2017",
        "split_mode": split_mode,
        "preset": preset,
        "use_canonical": use_canonical,
        "max_rows": split_meta["max_rows"],
        "total_timesteps": total_timesteps,
        "seed": seed,
        "reward_config": REWARD_CONFIG,
        "train_shape": list(X_train.shape),
        "test_shape": list(X_test.shape),
        "n_features": len(feature_names),
        "device": device,
        "script": "src/train_rl_defender.py",
        "argv": sys.argv,
        "split_metadata": split_meta,
        "training_hyperparams": hyperparams,
        "policy_kwargs": hyperparams["policy_kwargs"],
        "learning_rate": hyperparams["learning_rate"],
        "buffer_size": hyperparams["buffer_size"],
        "learning_starts": hyperparams["learning_starts"],
        "batch_size": hyperparams["batch_size"],
        "gamma": hyperparams["gamma"],
        "tau": hyperparams["tau"],
        "train_freq": hyperparams["train_freq"],
        "gradient_steps": hyperparams["gradient_steps"],
        "target_update_interval": hyperparams["target_update_interval"],
        "exploration_initial_eps": hyperparams["exploration_initial_eps"],
        "exploration_final_eps": hyperparams["exploration_final_eps"],
        "exploration_fraction": hyperparams["exploration_fraction"],
        "max_grad_norm": hyperparams["max_grad_norm"],
        "checkpoint_freq": checkpoint_freq,
        "checkpoints_dir": str(checkpoints_dir) if checkpoint_freq > 0 else None,
        "save_replay_buffer_checkpoints": False,
        "tensorboard_log_dir": tb_log_dir,
        "model_path": str(model_zip_path),
        "run_model_path": str(run_model_path),
        "config_path": str(config_path),
        "metrics_path": str(metrics_path),
        "scaler_path": str(scaler_path),
        "percentiles_path": str(percentiles_path),
        "feature_names_path": str(feature_names_path),
        "environment_path": str(environment_path),
        "artifact_manifest_path": str(manifest_path),
    }
    write_json(config_path, config)
    print(f"Config inicial guardada en: {config_path}")

    artifact_manifest = {
        "run_id": RUN_ID,
        "status": "started",
        "download_all": [
            str(run_dir),
            str(model_zip_path),
        ],
        "artifacts": {
            "model": {
                "global_path": str(model_zip_path),
                "run_copy_path": str(run_model_path),
            },
            "config": str(config_path),
            "metrics": str(metrics_path),
            "scaler": str(scaler_path),
            "train_percentiles": str(percentiles_path),
            "feature_names": str(feature_names_path),
            "environment": str(environment_path),
            "tensorboard_dir": tb_log_dir,
            "checkpoints_dir": str(checkpoints_dir) if checkpoint_freq > 0 else None,
        },
    }
    write_json(manifest_path, artifact_manifest)
    print(f"Manifest inicial guardado en: {manifest_path}")

    # ------------------------------------------------------------------
    # 2) Crear entorno vectorizado
    # ------------------------------------------------------------------
    max_steps_ep = min(10_000, len(X_train))
    vec_env = DummyVecEnv([make_env_fn(X_train, y_train, REWARD_CONFIG, max_steps_ep)])

    # ------------------------------------------------------------------
    # 3) Definir modelo QRDQN
    # ------------------------------------------------------------------
    vec_env.seed(seed)
    model_policy_kwargs = copy.deepcopy(hyperparams["policy_kwargs"])

    model = QRDQN(
        hyperparams["policy"],
        vec_env,
        seed=seed,
        policy_kwargs=model_policy_kwargs,
        learning_rate=hyperparams["learning_rate"],
        buffer_size=hyperparams["buffer_size"],
        learning_starts=hyperparams["learning_starts"],
        batch_size=hyperparams["batch_size"],
        gamma=hyperparams["gamma"],
        tau=hyperparams["tau"],
        train_freq=hyperparams["train_freq"],
        gradient_steps=hyperparams["gradient_steps"],
        target_update_interval=hyperparams["target_update_interval"],
        exploration_initial_eps=hyperparams["exploration_initial_eps"],
        exploration_final_eps=hyperparams["exploration_final_eps"],
        exploration_fraction=hyperparams["exploration_fraction"],
        max_grad_norm=hyperparams["max_grad_norm"],
        verbose=1,
        device=device,
        tensorboard_log=tb_log_dir,
    )

    checkpoint_callback = None
    if checkpoint_freq > 0:
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nCheckpoints will be saved in: {checkpoints_dir}")
        checkpoint_callback = CheckpointCallback(
            save_freq=checkpoint_freq,
            save_path=str(checkpoints_dir),
            name_prefix=RUN_ID,
            save_replay_buffer=False,
            save_vecnormalize=False,
            verbose=1,
        )

    print(f"\nEntrenando QRDQN durante {total_timesteps} timesteps...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=checkpoint_callback,
        tb_log_name=RUN_ID,
        reset_num_timesteps=False,
    )

    # ------------------------------------------------------------------
    # 4) Guardar modelo
    # ------------------------------------------------------------------
    print(f"\nGuardando modelo en: {model_path}")
    model.save(str(model_path))
    shutil.copy2(model_zip_path, run_model_path)
    print(f"Copia del modelo guardada en: {run_model_path}")

    # ------------------------------------------------------------------
    # 5) Evaluación en test
    # ------------------------------------------------------------------
    print("\nEvaluando en conjunto de test...")
    metrics = evaluate_model(model, X_test, y_test, REWARD_CONFIG)

    # ------------------------------------------------------------------
    # 6) Guardar config + métricas
    # ------------------------------------------------------------------
    write_json(metrics_path, metrics)
    config["status"] = "completed"
    config["completed_at"] = datetime.now().isoformat(timespec="seconds")
    write_json(config_path, config)
    artifact_manifest["status"] = "completed"
    write_json(manifest_path, artifact_manifest)

    print(f"\nConfig guardada en: {config_path}")
    print(f"Metricas guardadas en: {metrics_path}")
    print(f"Manifest guardado en: {manifest_path}")
    print(f"Modelo guardado en: {model_path}.zip")
    print(f"Copia del modelo guardada en: {run_model_path}")
    print(f"TensorBoard: tensorboard --logdir {tb_log_dir}")
    print(f"\n{'='*60}")
    print(f"  Experimento completado: {RUN_ID}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
