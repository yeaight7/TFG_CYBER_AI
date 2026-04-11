"""
validate_leave_one_csv_out.py — Validación leave-one-exact-CSV-out para CICIDS2017.

Entrena QRDQN una vez por fold, dejando exactamente un CSV real como test y
usando el resto de CSVs para train. Los resultados se guardan agregados en
``runs/validation/VAL_leave_one_csv_out_<timestamp>/``.

Uso:
    python src/validate_leave_one_csv_out.py
    python src/validate_leave_one_csv_out.py --timesteps 30000
    python src/validate_leave_one_csv_out.py --holdout-csvs Friday-WorkingHours-Morning.pcap_ISCX.csv
    python src/validate_leave_one_csv_out.py --timesteps 5000 --max-rows-per-csv 10000
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix

from sb3_contrib import QRDQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from load_cicids2017 import (
    CICIDSLoadConfig,
    list_cicids2017_csv_files,
    load_cicids2017_exact_csv_split,
)
from rl_defender_env import RLDatasetDefenderEnv


_REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = _REPO_ROOT / "runs"

REWARD_CONFIG: Dict[str, float] = {
    "tp": 1.5,
    "fp": -1.5,
    "fn": -5.0,
    "omission": 0.0,
}

SUMMARY_METRICS: Tuple[str, ...] = (
    "accuracy",
    "balanced_accuracy",
    "precision_attack",
    "recall_attack",
    "f1_attack",
    "precision_benign",
    "recall_benign",
    "f1_benign",
    "specificity",
    "fpr",
    "fnr",
    "block_rate",
    "reward_total",
    "reward_per_sample",
    "training_time_sec",
    "evaluation_time_sec",
    "total_time_sec",
)


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def make_env_fn(
    X: np.ndarray,
    y: np.ndarray,
    reward_config: Dict[str, float],
    max_steps: int,
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


def _compute_reward_total(tp: int, fp: int, fn: int, tn: int, reward_config: Dict[str, float]) -> float:
    return float(
        tp * reward_config["tp"]
        + fp * reward_config["fp"]
        + fn * reward_config["fn"]
        + tn * reward_config.get("omission", 0.0)
    )


def _metrics_from_confusion(
    tn: int,
    fp: int,
    fn: int,
    tp: int,
    reward_config: Dict[str, float],
) -> Dict[str, float]:
    total = tn + fp + fn + tp

    precision_attack = _safe_div(tp, tp + fp)
    recall_attack = _safe_div(tp, tp + fn)
    f1_attack = _safe_div(2 * precision_attack * recall_attack, precision_attack + recall_attack)

    precision_benign = _safe_div(tn, tn + fn)
    recall_benign = _safe_div(tn, tn + fp)
    f1_benign = _safe_div(2 * precision_benign * recall_benign, precision_benign + recall_benign)

    accuracy = _safe_div(tp + tn, total)
    specificity = recall_benign
    fpr = _safe_div(fp, fp + tn)
    fnr = _safe_div(fn, fn + tp)
    block_rate = _safe_div(tp + fp, total)
    reward_total = _compute_reward_total(tp, fp, fn, tn, reward_config)
    reward_per_sample = _safe_div(reward_total, total)

    return {
        "accuracy": accuracy,
        "balanced_accuracy": (recall_attack + recall_benign) / 2.0,
        "precision_attack": precision_attack,
        "recall_attack": recall_attack,
        "f1_attack": f1_attack,
        "precision_benign": precision_benign,
        "recall_benign": recall_benign,
        "f1_benign": f1_benign,
        "specificity": specificity,
        "fpr": fpr,
        "fnr": fnr,
        "block_rate": block_rate,
        "reward_total": reward_total,
        "reward_per_sample": reward_per_sample,
    }


def evaluate_model_direct(
    model: QRDQN,
    X_test: np.ndarray,
    y_test: np.ndarray,
    reward_config: Dict[str, float],
) -> Dict[str, Any]:
    """Evalúa el modelo sobre `X_test` en batch frente a `y_test`."""
    eval_start = time.perf_counter()
    actions, _ = model.predict(X_test, deterministic=True)
    y_pred = np.asarray(actions, dtype=np.int64).reshape(-1)
    evaluation_time_sec = time.perf_counter() - eval_start

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    report = classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        digits=4,
        output_dict=True,
        zero_division=0,
    )
    tn, fp, fn, tp = (int(value) for value in cm.ravel())
    metrics = _metrics_from_confusion(tn, fp, fn, tp, reward_config)

    return {
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        **metrics,
        "evaluation_time_sec": float(evaluation_time_sec),
    }


def _summarize_metric(values: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def _build_aggregate_results(
    folds: List[Dict[str, Any]],
    reward_config: Dict[str, float],
) -> Dict[str, Any]:
    per_metric = {
        metric_name: _summarize_metric([float(fold[metric_name]) for fold in folds])
        for metric_name in SUMMARY_METRICS
    }

    total_tp = int(sum(fold["tp"] for fold in folds))
    total_fp = int(sum(fold["fp"] for fold in folds))
    total_fn = int(sum(fold["fn"] for fold in folds))
    total_tn = int(sum(fold["tn"] for fold in folds))

    global_metrics = _metrics_from_confusion(
        tn=total_tn,
        fp=total_fp,
        fn=total_fn,
        tp=total_tp,
        reward_config=reward_config,
    )

    return {
        "n_folds": len(folds),
        "per_metric": per_metric,
        "sum_counts": {
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
            "tn": total_tn,
        },
        "global_confusion_matrix": [
            [total_tn, total_fp],
            [total_fn, total_tp],
        ],
        "global_support": {
            "n_samples": total_tn + total_fp + total_fn + total_tp,
            "benign": total_tn + total_fp,
            "attack": total_fn + total_tp,
            "train_samples": int(sum(fold["n_train"] for fold in folds)),
            "test_samples": int(sum(fold["n_test"] for fold in folds)),
        },
        "global_metrics": global_metrics,
    }


def _resolve_holdout_csvs(requested_csvs: List[str] | None, available_csvs: List[str]) -> List[str]:
    if requested_csvs is None:
        return list(available_csvs)

    available_map = {csv_name.lower(): csv_name for csv_name in available_csvs}
    resolved: List[str] = []
    seen: set[str] = set()

    for requested in requested_csvs:
        key = requested.strip().lower()
        if key not in available_map:
            raise ValueError(
                f"CSV holdout no encontrado: '{requested}'. Disponibles: {available_csvs}"
            )
        canonical_name = available_map[key]
        if canonical_name in seen:
            raise ValueError(f"CSV holdout duplicado: '{canonical_name}'")
        resolved.append(canonical_name)
        seen.add(canonical_name)

    return resolved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Leave-one-exact-CSV-out validation for CICIDS2017",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=30_000,
        help="Timesteps de entrenamiento por fold (default: 30000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semilla aleatoria (default: 42)",
    )
    parser.add_argument(
        "--holdout-csvs",
        nargs="+",
        default=None,
        help="Nombres exactos de CSVs a usar como holdout. Por defecto, se ejecutan todos.",
    )
    parser.add_argument(
        "--max-rows-per-csv",
        type=int,
        default=None,
        help="Cap opcional de filas por CSV para smoke/dev runs",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    available_csv_paths = list_cicids2017_csv_files()
    available_csvs = [path.name for path in available_csv_paths]
    holdout_csvs = _resolve_holdout_csvs(args.holdout_csvs, available_csvs)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"VAL_leave_one_csv_out_{timestamp}"
    run_dir = RUNS_DIR / "validation" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'=' * 72}")
    print("  Leave-One-CSV-Out Validation")
    print(f"  RUN_ID:            {run_id}")
    print(f"  Device:            {device}")
    print(f"  Timesteps/fold:    {args.timesteps}")
    print(f"  Max rows per CSV:  {args.max_rows_per_csv if args.max_rows_per_csv is not None else 'ALL'}")
    print(f"  Folds a ejecutar:  {len(holdout_csvs)}")
    print(f"  Output:            {run_dir}")
    print(f"{'=' * 72}")

    cfg = CICIDSLoadConfig(
        max_rows=None,
        sample_frac=None,
        scale=True,
        use_canonical=True,
        random_state=args.seed,
    )

    folds: List[Dict[str, Any]] = []

    for fold_idx, holdout_csv in enumerate(holdout_csvs, start=1):
        train_csvs = [csv_name for csv_name in available_csvs if csv_name != holdout_csv]
        print(f"\n{'-' * 72}")
        print(f"Fold {fold_idx}/{len(holdout_csvs)}")
        print(f"  Test CSV:  {holdout_csv}")
        print(f"  Train CSVs: {len(train_csvs)}")
        print(f"{'-' * 72}")

        X_train, y_train, X_test, y_test, _, feature_names = load_cicids2017_exact_csv_split(
            train_csv_names=train_csvs,
            test_csv_names=[holdout_csv],
            cfg=cfg,
            max_rows_per_csv=args.max_rows_per_csv,
        )

        max_steps_ep = min(10_000, len(X_train))
        vec_env = DummyVecEnv([make_env_fn(X_train, y_train, REWARD_CONFIG, max_steps_ep)])
        vec_env.seed(args.seed)

        model = QRDQN(
            "MlpPolicy",
            vec_env,
            seed=args.seed,
            policy_kwargs=dict(net_arch=[512, 256]),
            learning_rate=1e-4,
            buffer_size=min(200_000, max(args.timesteps, 10_000)),
            batch_size=512,
            gradient_steps=20,
            gamma=0.99,
            tau=1.0,
            train_freq=100,
            target_update_interval=10_000,
            verbose=0,
            device=device,
        )

        train_start = time.perf_counter()
        model.learn(total_timesteps=args.timesteps)
        training_time_sec = time.perf_counter() - train_start

        eval_result = evaluate_model_direct(model, X_test, y_test, REWARD_CONFIG)
        total_time_sec = training_time_sec + float(eval_result["evaluation_time_sec"])

        fold_result: Dict[str, Any] = {
            "fold_index": fold_idx,
            "train_csvs": train_csvs,
            "test_csv": holdout_csv,
            "n_train": int(len(y_train)),
            "n_test": int(len(y_test)),
            "train_benign": int((y_train == 0).sum()),
            "train_attack": int((y_train == 1).sum()),
            "test_benign": int((y_test == 0).sum()),
            "test_attack": int((y_test == 1).sum()),
            "n_features": int(len(feature_names)),
            "training_time_sec": float(training_time_sec),
            "total_time_sec": float(total_time_sec),
            **eval_result,
        }
        folds.append(fold_result)

        print(
            "  Resultados fold: "
            f"acc={fold_result['accuracy']:.4f}, "
            f"bal_acc={fold_result['balanced_accuracy']:.4f}, "
            f"f1_atk={fold_result['f1_attack']:.4f}, "
            f"reward/sample={fold_result['reward_per_sample']:.4f}"
        )

        vec_env.close()

    results = {
        "csv_order": holdout_csvs,
        "folds": folds,
        "aggregate": _build_aggregate_results(folds, REWARD_CONFIG),
    }

    config = {
        "run_id": run_id,
        "dataset": "CICIDS2017",
        "validation_mode": "leave_one_exact_csv_out",
        "csv_order": holdout_csvs,
        "available_csvs": available_csvs,
        "timesteps": args.timesteps,
        "seed": args.seed,
        "max_rows_per_csv": args.max_rows_per_csv,
        "device": device,
        "reward_config": REWARD_CONFIG,
        "policy_kwargs": {"net_arch": [512, 256]},
        "learning_rate": 1e-4,
        "batch_size": 512,
        "gradient_steps": 20,
        "train_freq": 100,
        "target_update_interval": 10_000,
    }

    config_path = run_dir / "config.json"
    results_path = run_dir / "validation_results.json"

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 72}")
    print("  Resumen agregado")
    print(f"  Accuracy global:        {results['aggregate']['global_metrics']['accuracy']:.4f}")
    print(f"  Balanced accuracy:      {results['aggregate']['global_metrics']['balanced_accuracy']:.4f}")
    print(f"  F1 attack global:       {results['aggregate']['global_metrics']['f1_attack']:.4f}")
    print(f"  Reward/sample global:   {results['aggregate']['global_metrics']['reward_per_sample']:.4f}")
    print(f"  Resultados:             {results_path}")
    print(f"  Config:                 {config_path}")
    print(f"{'=' * 72}")


if __name__ == "__main__":
    main()
