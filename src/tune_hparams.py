"""
tune_hparams.py — Optimización de hiperparámetros con Optuna para QRDQN sobre CICIDS2017.

Uso:
    python src/tune_hparams.py                    # 10 trials, 50k rows
    python src/tune_hparams.py --n-trials 30      # 30 trials
    python src/tune_hparams.py --max-rows 200000  # Más datos por trial
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import optuna
import torch
from sb3_contrib import QRDQN
from sklearn.metrics import f1_score
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from load_cicids2017 import CICIDSLoadConfig, load_cicids2017_binary
from rl_defender_env import RLDatasetDefenderEnv

_REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = _REPO_ROOT / "runs"

REWARD_CONFIG: Dict[str, float] = {
    "tp": 1.5,
    "fp": -1.0,
    "fn": -5.0,
    "omission": 0.0,
}


def _evaluate_f1(model, X_test: np.ndarray, y_test: np.ndarray) -> float:
    """Evaluate model and return F1 score for attack class."""
    env = RLDatasetDefenderEnv(
        X=X_test, y=y_test,
        benign_label=0, attack_label=1,
        reward_config=REWARD_CONFIG,
        max_steps_per_episode=len(X_test),
        shuffle=False,
    )
    obs, _ = env.reset()
    done = False
    y_true, y_pred = [], []
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(int(action))
        done = terminated or truncated
        y_true.append(int(info["true_label"]))
        y_pred.append(int(action))
    return float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))


def objective(
    trial: optuna.Trial,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    timesteps: int,
    device: str,
) -> float:
    """Optuna objective: train QRDQN with suggested hparams, return F1 attack."""
    lr = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [256, 512, 1024, 2048])
    gradient_steps = trial.suggest_categorical("gradient_steps", [10, 50, 100])
    net_arch_choice = trial.suggest_categorical("net_arch", ["256_128", "512_256", "256_256"])
    gamma = trial.suggest_float("gamma", 0.95, 0.999)
    train_freq = trial.suggest_categorical("train_freq", [50, 100, 200])

    net_arch_map = {
        "256_128": [256, 128],
        "512_256": [512, 256],
        "256_256": [256, 256],
    }
    net_arch = net_arch_map[net_arch_choice]

    max_steps_ep = min(10_000, len(X_train))

    def _make_env():
        env = RLDatasetDefenderEnv(
            X=X_train, y=y_train,
            benign_label=0, attack_label=1,
            reward_config=REWARD_CONFIG,
            max_steps_per_episode=max_steps_ep,
            shuffle=True,
        )
        return Monitor(env)

    vec_env = DummyVecEnv([_make_env])

    model = QRDQN(
        "MlpPolicy",
        vec_env,
        seed=42,
        policy_kwargs=dict(net_arch=net_arch),
        learning_rate=lr,
        buffer_size=min(200_000, max(timesteps, 10_000)),
        batch_size=batch_size,
        gradient_steps=gradient_steps,
        gamma=gamma,
        tau=1.0,
        train_freq=train_freq,
        target_update_interval=max(1_000, timesteps // 50),
        verbose=0,
        device=device,
    )

    model.learn(total_timesteps=timesteps)
    f1_attack = _evaluate_f1(model, X_test, y_test)

    print(f"  Trial {trial.number}: F1_attack={f1_attack:.4f} "
          f"(lr={lr:.1e}, bs={batch_size}, arch={net_arch})")
    return f1_attack


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optuna hyperparameter tuning for QRDQN")
    parser.add_argument("--n-trials", type=int, default=10, help="Number of Optuna trials")
    parser.add_argument("--timesteps", type=int, default=10_000, help="Timesteps per trial")
    parser.add_argument("--max-rows", type=int, default=50_000, help="Max dataset rows")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"Loading CICIDS2017 (max_rows={args.max_rows})...")
    cfg = CICIDSLoadConfig(
        max_rows=args.max_rows,
        use_canonical=True,
        scale=True,
        random_state=args.seed,
    )
    X_train, y_train, X_test, y_test, _, _ = load_cicids2017_binary(cfg)

    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"Starting Optuna study ({args.n_trials} trials, {args.timesteps} timesteps each)...\n")

    study = optuna.create_study(direction="maximize", study_name=f"qrdqn_tune_{timestamp}")

    study.optimize(
        lambda trial: objective(
            trial, X_train, y_train, X_test, y_test, args.timesteps, device
        ),
        n_trials=args.n_trials,
    )

    print(f"\n{'='*60}")
    print(f"Best trial: #{study.best_trial.number}")
    print(f"Best F1 attack: {study.best_value:.4f}")
    print(f"Best params: {json.dumps(study.best_params, indent=2)}")
    print(f"{'='*60}")

    # Save results
    out_dir = RUNS_DIR / "optuna"
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / f"study_{timestamp}.json"
    results = {
        "best_value": study.best_value,
        "best_params": study.best_params,
        "n_trials": args.n_trials,
        "timesteps_per_trial": args.timesteps,
        "max_rows": args.max_rows,
    }
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
