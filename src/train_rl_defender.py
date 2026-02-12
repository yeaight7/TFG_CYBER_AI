"""
train_rl_defender.py — Entrenamiento de agente defensor RL sobre CICIDS2017.

Uso:
    python src/train_rl_defender.py --smoke        # Smoke test (~2-5 min)
    python src/train_rl_defender.py                # Entrenamiento completo (~30-60 min)
    python src/train_rl_defender.py --timesteps 200000  # Custom timesteps
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List
from datetime import datetime

import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from sb3_contrib import QRDQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from rl_defender_env import RLDatasetDefenderEnv
from load_cicids2017 import load_cicids2017_binary, CICIDSLoadConfig


_REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = _REPO_ROOT / "models"
RUNS_DIR = _REPO_ROOT / "runs"


# --------------------------------------------------------------------------------------
# Configuración de recompensa para el agente defensor
# --------------------------------------------------------------------------------------
REWARD_CONFIG: Dict[str, float] = {
    "tp": 1.5,
    "fp": -1.0,
    "fn": -5.0,
    "omission": 0.0,
}


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
        help="Smoke test: 50k rows, 5k timesteps (~2-5 min)",
    )
    parser.add_argument(
        "--timesteps", type=int, default=None,
        help="Total timesteps for training (default: 500k normal, 5k smoke)",
    )
    parser.add_argument(
        "--max-rows", type=int, default=None,
        help="Max rows to load from dataset (default: all normal, 10k smoke)",
    )
    parser.add_argument(
        "--no-canonical", action="store_true",
        help="Disable canonical schema + missingness mask (use raw features)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ── Smoke vs normal defaults ──
    if args.smoke:
        max_rows = args.max_rows or 50_000
        total_timesteps = args.timesteps or 5_000
        exp_tag = "smoke"
    else:
        max_rows = args.max_rows  # None = load all
        total_timesteps = args.timesteps or 500_000
        exp_tag = "full"

    use_canonical = not args.no_canonical
    seed = args.seed

    # ── RUN_ID único ──
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    algo_tag = "qrdqn"
    canon_tag = "canonical" if use_canonical else "raw"
    RUN_ID = f"C01_{algo_tag}_cicids2017_{canon_tag}_{exp_tag}_{timestamp}"

    # ── Directorios de salida ──
    run_dir = RUNS_DIR / "cicids2017" / RUN_ID
    run_dir.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"{'='*60}")
    print(f"  Experimento: {RUN_ID}")
    print(f"  Mode: {'SMOKE TEST' if args.smoke else 'FULL TRAINING'}")
    print(f"  Max rows: {max_rows or 'ALL'}")
    print(f"  Timesteps: {total_timesteps}")
    print(f"  Canonical: {use_canonical}")
    print(f"  Output: {run_dir}")
    print(f"{'='*60}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        print(f"GPU detectada: {torch.cuda.get_device_name(0)}")
    else:
        print("GPU NO detectada. Se usara CPU.")

    # ------------------------------------------------------------------
    # 1) Cargar dataset CICIDS2017
    # ------------------------------------------------------------------
    print("\nCargando CICIDS2017...")
    cfg = CICIDSLoadConfig(
        max_rows=max_rows,
        use_canonical=use_canonical,
        scale=True,
        random_state=seed,
    )
    X_train, y_train, X_test, y_test, scaler, feature_names = load_cicids2017_binary(cfg)

    print(f"Train: X={X_train.shape}, y={y_train.shape} "
          f"(benign={int((y_train==0).sum())}, attack={int((y_train==1).sum())})")
    print(f"Test:  X={X_test.shape}, y={y_test.shape} "
          f"(benign={int((y_test==0).sum())}, attack={int((y_test==1).sum())})")
    print(f"Features: {len(feature_names)}")

    # ------------------------------------------------------------------
    # 2) Crear entorno vectorizado
    # ------------------------------------------------------------------
    max_steps_ep = min(10_000, len(X_train))
    vec_env = DummyVecEnv([make_env_fn(X_train, y_train, REWARD_CONFIG, max_steps_ep)])

    # ------------------------------------------------------------------
    # 3) Definir modelo QRDQN
    # ------------------------------------------------------------------
    vec_env.seed(seed)
    policy_kwargs = dict(net_arch=[512, 256])
    tb_log_dir = str(RUNS_DIR / "cicids2017")

    # Hyperparámetros adaptados al modo (smoke vs full)
    batch_size = 256 if args.smoke else 1024
    gradient_steps = 10 if args.smoke else 20
    train_freq = 50 if args.smoke else 100
    target_update_interval = 1_000 if args.smoke else 10_000
    lr = 1e-4

    model = QRDQN(
        "MlpPolicy",
        vec_env,
        seed=seed,
        policy_kwargs=policy_kwargs,
        learning_rate=lr,
        buffer_size=min(200_000, max(total_timesteps, 10_000)),
        batch_size=batch_size,
        gradient_steps=gradient_steps,
        gamma=0.99,
        tau=1.0,
        train_freq=train_freq,
        target_update_interval=target_update_interval,
        verbose=1,
        device=device,
        tensorboard_log=tb_log_dir,
    )

    print(f"\nEntrenando QRDQN durante {total_timesteps} timesteps...")
    model.learn(
        total_timesteps=total_timesteps,
        tb_log_name=RUN_ID,
        reset_num_timesteps=False,
    )

    # ------------------------------------------------------------------
    # 4) Guardar modelo
    # ------------------------------------------------------------------
    model_path = MODELS_DIR / RUN_ID
    print(f"\nGuardando modelo en: {model_path}")
    model.save(str(model_path))

    # ------------------------------------------------------------------
    # 5) Evaluación en test
    # ------------------------------------------------------------------
    print("\nEvaluando en conjunto de test...")
    metrics = evaluate_model(model, X_test, y_test, REWARD_CONFIG)

    # ------------------------------------------------------------------
    # 6) Guardar config + métricas
    # ------------------------------------------------------------------
    config = {
        "run_id": RUN_ID,
        "algorithm": "QRDQN",
        "dataset": "CICIDS2017",
        "use_canonical": use_canonical,
        "max_rows": max_rows,
        "total_timesteps": total_timesteps,
        "seed": seed,
        "reward_config": REWARD_CONFIG,
        "train_shape": list(X_train.shape),
        "test_shape": list(X_test.shape),
        "n_features": len(feature_names),
        "device": device,
        "smoke": args.smoke,
        "policy_kwargs": {"net_arch": [512, 256]},
        "learning_rate": lr,
        "batch_size": batch_size,
        "gradient_steps": gradient_steps,
        "train_freq": train_freq,
    }

    config_path = run_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    metrics_path = run_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nConfig guardada en: {config_path}")
    print(f"Metricas guardadas en: {metrics_path}")
    print(f"Modelo guardado en: {model_path}.zip")
    print(f"TensorBoard: tensorboard --logdir {tb_log_dir}")
    print(f"\n{'='*60}")
    print(f"  Experimento completado: {RUN_ID}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()