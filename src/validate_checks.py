"""
validate_checks.py — Validación de resultados experimentales del agente RL.

Implementa tres checks para verificar que las métricas son genuinas:

  Check A: Evaluación directa usando model.predict(X_test[i]) y y_test,
           sin depender de info["true_label"] del entorno.
  Check B: Shuffled labels test (anti-leakage). Baraja y_train, entrena
           brevemente, y verifica que la accuracy baja a ~random.
  Check C: Split por CSV (más realista). Entrena en unos CSVs de CICIDS2017
           y testea en otros para evitar patrones duplicados.

Uso:
    # Ejecutar todos los checks con el modelo entrenado
    python src/validate_checks.py --model models/C01_qrdqn_cicids2017_canonical_full_20260212_200218.zip

    # Solo Check A (rápido, no re-entrena)
    python src/validate_checks.py --model models/<MODEL>.zip --checks A

    # Solo Check B (entrena brevemente con labels barajadas)
    python src/validate_checks.py --checks B

    # Solo Check C (entrena y testea con split por CSV)
    python src/validate_checks.py --checks C

    # Limitar filas para prueba rápida
    python src/validate_checks.py --model models/<MODEL>.zip --max-rows 50000 --checks A B C
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix

from sb3_contrib import QRDQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

from rl_defender_env import RLDatasetDefenderEnv
from load_cicids2017 import (
    CICIDSLoadConfig,
    load_cicids2017_binary,
    load_cicids2017_csv_split,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = _REPO_ROOT / "models"
RUNS_DIR = _REPO_ROOT / "runs"

SEED = 42

REWARD_CONFIG: Dict[str, float] = {
    "tp": 1.5,
    "fp": -1.0,
    "fn": -5.0,
    "omission": 0.0,
}

# CSV-split por defecto para Check C:
# Train: lunes a miércoles (tráfico más variado, incluye ataques)
# Test: jueves y viernes (diferentes tipos de ataque)
DEFAULT_TRAIN_CSVS = ["Monday", "Tuesday", "Wednesday"]
DEFAULT_TEST_CSVS = ["Thursday", "Friday"]


class ProgressCallback(BaseCallback):
    """Callback para mostrar progreso cada log_freq timesteps."""
    def __init__(self, log_freq: int = 10_000, verbose: int = 0):
        super().__init__(verbose)
        self.log_freq = log_freq
        
    def _on_step(self) -> bool:
        if self.num_timesteps % self.log_freq == 0:
            print(f"  → Timesteps: {self.num_timesteps}/{self.model._total_timesteps}")
        return True


# ──────────────────────────────────────────────────────────────
# Check A: Evaluación directa sin env
# ──────────────────────────────────────────────────────────────

def check_a_direct_eval(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict:
    """
    Evaluación directa: model.predict(X_test[i]) vs y_test[i].
    No pasa por el entorno RL, evitando posibles bugs en info["true_label"].
    """
    print("\n" + "=" * 60)
    print("  CHECK A — Evaluación directa (sin entorno RL)")
    print("=" * 60)

    n = len(X_test)
    y_pred = np.empty(n, dtype=np.int64)

    for i in range(n):
        obs = X_test[i]
        action, _ = model.predict(obs, deterministic=True)
        y_pred[i] = int(action)

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    report_dict = classification_report(y_test, y_pred, labels=[0, 1], digits=4, output_dict=True, zero_division=0)
    report_str = classification_report(y_test, y_pred, labels=[0, 1], digits=4, zero_division=0)

    print("\nConfusion matrix (0=PERMIT/benign, 1=BLOCK/attack):")
    print(cm)
    print("\nClassification report:")
    print(report_str)

    tn, fp, fn, tp = cm.ravel()
    print(f"  TP={tp}, FP={fp}, FN={fn}, TN={tn}")

    result = {
        "check": "A",
        "description": "Direct eval without env info['true_label']",
        "confusion_matrix": cm.tolist(),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "accuracy": float(report_dict["accuracy"]),
        "precision_attack": float(report_dict.get("1", {}).get("precision", 0.0)),
        "recall_attack": float(report_dict.get("1", {}).get("recall", 0.0)),
        "f1_attack": float(report_dict.get("1", {}).get("f1-score", 0.0)),
        "precision_benign": float(report_dict.get("0", {}).get("precision", 0.0)),
        "recall_benign": float(report_dict.get("0", {}).get("recall", 0.0)),
        "f1_benign": float(report_dict.get("0", {}).get("f1-score", 0.0)),
        "n_samples": n,
    }

    return result


# ──────────────────────────────────────────────────────────────
# Check B: Shuffled labels test (anti-leakage)
# ──────────────────────────────────────────────────────────────

def check_b_shuffled_labels(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    timesteps: int = 10_000,
    seed: int = SEED,
    device: str = "cuda",
) -> Dict:
    """
    Baraja y_train y entrena brevemente. Si el modelo aún obtiene accuracy
    alta → hay leakage o artefacto en los datos.

    Resultado esperado (sin leakage): accuracy ≈ proporción de la clase
    mayoritaria (baseline aleatorio).
    """
    print("\n" + "=" * 60)
    print("  CHECK B — Shuffled labels test (anti-leakage)")
    print("=" * 60)

    rng = np.random.default_rng(seed)
    y_shuffled = y_train.copy()
    rng.shuffle(y_shuffled)

    # Baseline esperado: accuracy = max(P(benign), P(attack))
    benign_rate = float((y_test == 0).mean())
    attack_rate = float((y_test == 1).mean())
    baseline_acc = max(benign_rate, attack_rate)
    print(f"\nBaseline esperado (clase mayoritaria): {baseline_acc:.4f}")
    print(f"  Test benign rate: {benign_rate:.4f}, attack rate: {attack_rate:.4f}")

    max_steps_ep = min(5_000, len(X_train))

    def _make_env():
        env = RLDatasetDefenderEnv(
            X=X_train,
            y=y_shuffled,
            benign_label=0,
            attack_label=1,
            reward_config=REWARD_CONFIG,
            max_steps_per_episode=max_steps_ep,
            shuffle=True,
        )
        return Monitor(env)

    vec_env = DummyVecEnv([_make_env])
    vec_env.seed(seed)

    model_shuffled = QRDQN(
        "MlpPolicy",
        vec_env,
        seed=seed,
        policy_kwargs=dict(net_arch=[256, 128]),
        learning_rate=1e-4,
        buffer_size=min(50_000, max(timesteps, 5_000)),
        batch_size=256,
        gradient_steps=10,
        gamma=0.99,
        tau=1.0,
        train_freq=50,
        target_update_interval=1_000,
        verbose=0,
        device=device,
    )

    print(f"\nEntrenando con labels BARAJADAS ({timesteps} timesteps)...")
    model_shuffled.learn(total_timesteps=timesteps)

    # Evaluación directa (como Check A)
    n = len(X_test)
    y_pred = np.empty(n, dtype=np.int64)
    for i in range(n):
        action, _ = model_shuffled.predict(X_test[i], deterministic=True)
        y_pred[i] = int(action)

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    report_dict = classification_report(y_test, y_pred, labels=[0, 1], digits=4, output_dict=True, zero_division=0)
    report_str = classification_report(y_test, y_pred, labels=[0, 1], digits=4, zero_division=0)

    shuffled_acc = float(report_dict["accuracy"])

    print("\nConfusion matrix (modelo entrenado con labels BARAJADAS):")
    print(cm)
    print("\nClassification report:")
    print(report_str)

    # Veredicto
    # Si accuracy con labels barajadas es similar al baseline → no hay leakage
    leakage_threshold = baseline_acc + 0.05  # margen del 5%
    leakage_detected = shuffled_acc > leakage_threshold

    print(f"\n  Accuracy con labels barajadas: {shuffled_acc:.4f}")
    print(f"  Baseline (clase mayoritaria):  {baseline_acc:.4f}")
    print(f"  Umbral de leakage:             {leakage_threshold:.4f}")
    if leakage_detected:
        print("  ⚠️  POSIBLE LEAKAGE: accuracy con labels barajadas supera el umbral")
    else:
        print("  ✅ SIN LEAKAGE: accuracy con labels barajadas es cercana al baseline")

    tn, fp, fn, tp = cm.ravel()

    result = {
        "check": "B",
        "description": "Shuffled labels test (anti-leakage)",
        "timesteps": timesteps,
        "shuffled_accuracy": shuffled_acc,
        "baseline_accuracy": baseline_acc,
        "leakage_threshold": leakage_threshold,
        "leakage_detected": leakage_detected,
        "confusion_matrix": cm.tolist(),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "precision_attack": float(report_dict.get("1", {}).get("precision", 0.0)),
        "recall_attack": float(report_dict.get("1", {}).get("recall", 0.0)),
        "f1_attack": float(report_dict.get("1", {}).get("f1-score", 0.0)),
        "n_train": len(X_train),
        "n_test": n,
        "test_benign_rate": benign_rate,
        "test_attack_rate": attack_rate,
    }

    return result


# ──────────────────────────────────────────────────────────────
# Check C: Split por CSV (más realista en CICIDS2017)
# ──────────────────────────────────────────────────────────────

def check_c_csv_split(
    train_csvs: List[str],
    test_csvs: List[str],
    timesteps: int = 50_000,
    max_rows: Optional[int] = None,
    seed: int = SEED,
    device: str = "cuda",
) -> Dict:
    """
    Entrena en unos CSVs de CICIDS2017 y testea en otros.
    Es el split más realista: el test contiene días/ataques que el modelo
    no ha visto durante entrenamiento.
    """
    print("\n" + "=" * 60)
    print("  CHECK C — Split por CSV (train y test en CSVs diferentes)")
    print("=" * 60)
    print(f"  Train CSVs: {train_csvs}")
    print(f"  Test  CSVs: {test_csvs}")
    print(f"  Timesteps:  {timesteps}")

    cfg = CICIDSLoadConfig(
        max_rows=max_rows,
        use_canonical=True,
        scale=True,
        random_state=seed,
    )

    X_train, y_train, X_test, y_test, scaler, feature_names = load_cicids2017_csv_split(
        train_csvs=train_csvs,
        test_csvs=test_csvs,
        cfg=cfg,
    )

    print(f"\nTrain: {X_train.shape} (benign={int((y_train==0).sum())}, attack={int((y_train==1).sum())})")
    print(f"Test:  {X_test.shape} (benign={int((y_test==0).sum())}, attack={int((y_test==1).sum())})")

    max_steps_ep = min(10_000, len(X_train))

    def _make_env():
        env = RLDatasetDefenderEnv(
            X=X_train,
            y=y_train,
            benign_label=0,
            attack_label=1,
            reward_config=REWARD_CONFIG,
            max_steps_per_episode=max_steps_ep,
            shuffle=True,
        )
        return Monitor(env)

    vec_env = DummyVecEnv([_make_env])
    vec_env.seed(seed)

    model = QRDQN(
        "MlpPolicy",
        vec_env,
        seed=seed,
        policy_kwargs=dict(net_arch=[512, 256]),
        learning_rate=1e-4,
        buffer_size=min(200_000, max(timesteps, 10_000)),
        batch_size=512,
        gradient_steps=20,
        gamma=0.99,
        tau=1.0,
        train_freq=100,
        target_update_interval=10_000,
        verbose=0,
        device=device,
    )

    print(f"\nEntrenando QRDQN con split por CSV ({timesteps} timesteps)...")
    progress_callback = ProgressCallback(log_freq=10_000)
    model.learn(total_timesteps=timesteps, callback=progress_callback)

    # Evaluación directa
    n = len(X_test)
    y_pred = np.empty(n, dtype=np.int64)
    for i in range(n):
        action, _ = model.predict(X_test[i], deterministic=True)
        y_pred[i] = int(action)

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    report_dict = classification_report(y_test, y_pred, labels=[0, 1], digits=4, output_dict=True, zero_division=0)
    report_str = classification_report(y_test, y_pred, labels=[0, 1], digits=4, zero_division=0)

    print("\nConfusion matrix (split por CSV):")
    print(cm)
    print("\nClassification report:")
    print(report_str)

    tn, fp, fn, tp = cm.ravel()
    print(f"  TP={tp}, FP={fp}, FN={fn}, TN={tn}")

    result = {
        "check": "C",
        "description": "CSV-split evaluation (train/test on different CSVs)",
        "train_csvs": train_csvs,
        "test_csvs": test_csvs,
        "timesteps": timesteps,
        "confusion_matrix": cm.tolist(),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "accuracy": float(report_dict["accuracy"]),
        "precision_attack": float(report_dict.get("1", {}).get("precision", 0.0)),
        "recall_attack": float(report_dict.get("1", {}).get("recall", 0.0)),
        "f1_attack": float(report_dict.get("1", {}).get("f1-score", 0.0)),
        "precision_benign": float(report_dict.get("0", {}).get("precision", 0.0)),
        "recall_benign": float(report_dict.get("0", {}).get("recall", 0.0)),
        "f1_benign": float(report_dict.get("0", {}).get("f1-score", 0.0)),
        "n_train": len(X_train),
        "n_test": n,
        "train_benign": int((y_train == 0).sum()),
        "train_attack": int((y_train == 1).sum()),
        "test_benign": int((y_test == 0).sum()),
        "test_attack": int((y_test == 1).sum()),
    }

    return result


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validation checks for RL defender results",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Path to trained model .zip (required for Check A)",
    )
    parser.add_argument(
        "--checks", nargs="+", default=["A", "B", "C"],
        choices=["A", "B", "C"],
        help="Which checks to run (default: all)",
    )
    parser.add_argument(
        "--max-rows", type=int, default=None,
        help="Max rows to load from dataset (default: all for C, 250k for A/B)",
    )
    parser.add_argument(
        "--timesteps-b", type=int, default=10_000,
        help="Timesteps for Check B shuffled training (default: 10000)",
    )
    parser.add_argument(
        "--timesteps-c", type=int, default=30_000,
        help="Timesteps for Check C CSV-split training (default: 30000)",
    )
    parser.add_argument(
        "--train-csvs", nargs="+", default=DEFAULT_TRAIN_CSVS,
        help="CSV name patterns for training in Check C (default: Monday Tuesday Wednesday)",
    )
    parser.add_argument(
        "--test-csvs", nargs="+", default=DEFAULT_TEST_CSVS,
        help="CSV name patterns for testing in Check C (default: Thursday Friday)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checks = [c.upper() for c in args.checks]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    RUN_ID = f"VAL_checks_{''.join(checks)}_{timestamp}"
    run_dir = RUNS_DIR / "validation" / RUN_ID
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'=' * 60}")
    print(f"  Validation Checks: {checks}")
    print(f"  RUN_ID: {RUN_ID}")
    print(f"  Output: {run_dir}")
    print(f"  Device: {device}")
    print(f"{'=' * 60}")

    results: Dict[str, Dict] = {}

    # ── Cargar datos para Check A y B ──
    need_random_split = "A" in checks or "B" in checks
    X_train = y_train = X_test = y_test = None

    if need_random_split:
        max_rows_ab = args.max_rows or 250_000
        print(f"\nCargando CICIDS2017 (random split, max_rows={max_rows_ab})...")
        cfg = CICIDSLoadConfig(
            max_rows=max_rows_ab,
            use_canonical=True,
            scale=True,
            random_state=args.seed,
        )
        X_train, y_train, X_test, y_test, _, _ = load_cicids2017_binary(cfg)
        print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    # ── Check A ──
    if "A" in checks:
        if args.model is None:
            print("\n⚠️  Check A requiere --model <path>. Saltando.")
        else:
            model_path = Path(args.model)
            if not model_path.exists():
                # Try relative to repo root
                model_path = _REPO_ROOT / args.model
            if not model_path.exists():
                print(f"\n⚠️  Modelo no encontrado: {args.model}. Saltando Check A.")
            else:
                print(f"\nCargando modelo: {model_path}")
                model = QRDQN.load(str(model_path), device=device)
                results["A"] = check_a_direct_eval(model, X_test, y_test)

    # ── Check B ──
    if "B" in checks:
        results["B"] = check_b_shuffled_labels(
            X_train, y_train, X_test, y_test,
            timesteps=args.timesteps_b,
            seed=args.seed,
            device=device,
        )

    # ── Check C ──
    if "C" in checks:
        results["C"] = check_c_csv_split(
            train_csvs=args.train_csvs,
            test_csvs=args.test_csvs,
            timesteps=args.timesteps_c,
            max_rows=args.max_rows,
            seed=args.seed,
            device=device,
        )

    # ── Guardar resultados ──
    results_path = run_dir / "validation_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    # Config del run
    config = {
        "run_id": RUN_ID,
        "checks": checks,
        "model_path": args.model,
        "max_rows": args.max_rows,
        "timesteps_b": args.timesteps_b,
        "timesteps_c": args.timesteps_c,
        "train_csvs_c": args.train_csvs,
        "test_csvs_c": args.test_csvs,
        "seed": args.seed,
        "device": device,
        "reward_config": REWARD_CONFIG,
    }
    config_path = run_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    # ── Resumen final ──
    print(f"\n{'=' * 60}")
    print(f"  RESUMEN DE VALIDACIÓN")
    print(f"{'=' * 60}")

    for check_id, res in results.items():
        print(f"\n  Check {check_id}: {res.get('description', '')}")
        if check_id == "A":
            print(f"    Accuracy:       {res['accuracy']:.4f}")
            print(f"    F1 attack:      {res['f1_attack']:.4f}")
            print(f"    TP={res['tp']}, FP={res['fp']}, FN={res['fn']}, TN={res['tn']}")
        elif check_id == "B":
            print(f"    Shuffled acc:   {res['shuffled_accuracy']:.4f}")
            print(f"    Baseline acc:   {res['baseline_accuracy']:.4f}")
            print(f"    Leakage:        {'⚠️ POSIBLE' if res['leakage_detected'] else '✅ NO'}")
        elif check_id == "C":
            print(f"    Accuracy:       {res['accuracy']:.4f}")
            print(f"    F1 attack:      {res['f1_attack']:.4f}")
            print(f"    TP={res['tp']}, FP={res['fp']}, FN={res['fn']}, TN={res['tn']}")

    print(f"\n  Resultados guardados en: {results_path}")
    print(f"  Config guardada en:      {config_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
