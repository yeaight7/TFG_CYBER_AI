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
    python src/validate_checks.py --run-dir runs/cicids2017/MAIN_qrdqn_cicids2017_canonical_full_random_20260609_193655

    # Solo Check A (rápido, no re-entrena)
    python src/validate_checks.py --run-dir runs/cicids2017/<RUN_ID> --checks A

    # Solo Check B (entrena brevemente con labels barajadas)
    python src/validate_checks.py --checks B

    # Solo Check C (entrena y testea con split por CSV)
    python src/validate_checks.py --checks C

    # Limitar filas para prueba rápida
    python src/validate_checks.py --run-dir runs/cicids2017/<RUN_ID> --max-rows 50000 --checks A B C
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import joblib
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

from sb3_contrib import QRDQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

if __package__ in {None, ""}:  # pragma: no cover - direct script execution
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from src.artifact_integrity import ArtifactTrustError, resolve_trusted_artifact
    from src.cicids_cache import sha256_array
    from src.experiment_profiles import MAIN_V1_PROFILE
    from src.load_cicids2017 import (
        DEFAULT_TEST_DAYS,
        DEFAULT_TRAIN_DAYS,
        load_cicids2017_split,
    )
    from src.metrics_utils import confusion_to_metrics
    from src.qrdqn_experiment import (
        PreparedSplit,
        QRDQNRunConfig,
        batched_predict,
        load_experiment_split,
        seed_model_rngs,
    )
    from src.resource_monitor import ResourceMonitor
    from src.rl_defender_env import RLDatasetDefenderEnv
    from src.run_artifacts import (
        ArtifactManifestWriter,
        ArtifactRequirement,
        TimingRecorder,
        atomic_write_json,
        atomic_write_text,
        collect_environment_metadata,
        tee_output,
    )
except ModuleNotFoundError:  # pragma: no cover - direct ``python src/...`` execution
    from artifact_integrity import ArtifactTrustError, resolve_trusted_artifact
    from cicids_cache import sha256_array
    from experiment_profiles import MAIN_V1_PROFILE
    from load_cicids2017 import DEFAULT_TEST_DAYS, DEFAULT_TRAIN_DAYS, load_cicids2017_split
    from metrics_utils import confusion_to_metrics
    from qrdqn_experiment import (
        PreparedSplit,
        QRDQNRunConfig,
        batched_predict,
        load_experiment_split,
        seed_model_rngs,
    )
    from resource_monitor import ResourceMonitor
    from rl_defender_env import RLDatasetDefenderEnv
    from run_artifacts import (
        ArtifactManifestWriter,
        ArtifactRequirement,
        TimingRecorder,
        atomic_write_json,
        atomic_write_text,
        collect_environment_metadata,
        tee_output,
    )

_REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = _REPO_ROOT / "models"
RUNS_DIR = _REPO_ROOT / "runs"

SEED = 42

REWARD_CONFIG: Dict[str, float] = MAIN_V1_PROFILE.reward_config()


@dataclass(frozen=True)
class ShuffledLabelRunConfig:
    artifact_root: Path
    run_id: str
    dataset_root: Path
    cache_root: Path | None
    cache_policy: str = "require"
    split_seed: int = 42
    model_seed: int = 42
    shuffled_label_seed: int = 42
    timesteps: int = 10_000
    monitor_interval: float = 30.0
    eval_batch_size: int = 8_192
    attempt: int = 1
    campaign_id: str | None = None
    logical_run_id: str = "shuffled_label_validation_s42_m42"

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
        if self.split_seed != 42 or self.model_seed != 42:
            raise ValueError("The shuffled-label campaign control requires split_seed=42 and model_seed=42")
        if self.timesteps <= 0:
            raise ValueError("timesteps must be greater than zero")
        if self.monitor_interval <= 0:
            raise ValueError("monitor_interval must be greater than zero")
        if self.eval_batch_size <= 0:
            raise ValueError("eval_batch_size must be greater than zero")


ShuffledSplitLoader = Callable[[QRDQNRunConfig], PreparedSplit]
ShuffledModelFactory = Callable[[ShuffledLabelRunConfig, DummyVecEnv, Path, str], Any]


def build_label_permutation(
    labels: np.ndarray,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Return deterministic permutation indices, labels, and the index-array hash."""
    indices = np.random.default_rng(seed).permutation(len(labels)).astype(np.int64)
    return indices, np.asarray(labels)[indices].copy(), sha256_array(indices)


def _shuffled_training_config(timesteps: int) -> dict[str, Any]:
    return {
        "policy": "MlpPolicy",
        "policy_kwargs": {"net_arch": [256, 128]},
        "learning_rate": 1e-4,
        "buffer_size": min(50_000, max(timesteps, 5_000)),
        "batch_size": 256,
        "gradient_steps": 10,
        "gamma": 0.0,
        "tau": 1.0,
        "train_freq": 50,
        "target_update_interval": 1_000,
    }


def _create_shuffled_model(
    config: ShuffledLabelRunConfig,
    environment: DummyVecEnv,
    tensorboard_dir: Path,
    device: str,
) -> QRDQN:
    training = _shuffled_training_config(config.timesteps)
    return QRDQN(
        training["policy"],
        environment,
        seed=config.model_seed,
        policy_kwargs=training["policy_kwargs"],
        learning_rate=training["learning_rate"],
        buffer_size=training["buffer_size"],
        batch_size=training["batch_size"],
        gradient_steps=training["gradient_steps"],
        gamma=training["gamma"],
        tau=training["tau"],
        train_freq=training["train_freq"],
        target_update_interval=training["target_update_interval"],
        verbose=1,
        device=device,
        tensorboard_log=str(tensorboard_dir),
    )


def _shuffled_requirements() -> dict[str, ArtifactRequirement]:
    return {
        "config": ArtifactRequirement("config.json"),
        "metrics": ArtifactRequirement("metrics.json"),
        "environment": ArtifactRequirement("environment.json"),
        "model": ArtifactRequirement("model.zip"),
        "scaler": ArtifactRequirement("scaler.joblib"),
        "feature_names": ArtifactRequirement("feature_names.json"),
        "predictions": ArtifactRequirement("predictions.npz"),
        "timing": ArtifactRequirement("timing.json"),
        "system_metrics": ArtifactRequirement("system_metrics.csv"),
        "monitoring": ArtifactRequirement("monitoring.json"),
        "stdout": ArtifactRequirement("stdout.log"),
        "stderr": ArtifactRequirement("stderr.log"),
        "tensorboard": ArtifactRequirement("tensorboard", kind="directory"),
    }


def _json_shuffled_config(config: ShuffledLabelRunConfig) -> dict[str, Any]:
    payload = asdict(config)
    for key in ("artifact_root", "dataset_root", "cache_root"):
        value = payload[key]
        payload[key] = None if value is None else str(value)
    return payload


def run_shuffled_label_validation(
    config: ShuffledLabelRunConfig,
    *,
    split_loader: ShuffledSplitLoader = load_experiment_split,
    model_factory: ShuffledModelFactory = _create_shuffled_model,
    monitor_factory: Callable[..., ResourceMonitor] = ResourceMonitor,
) -> Path:
    """Run the artifact-backed lightweight anti-leakage auxiliary control."""
    run_dir = config.artifact_root / config.run_id
    run_metadata = {
        "campaign_id": config.campaign_id,
        "logical_run_id": config.logical_run_id,
        "physical_run_id": config.run_id,
        "attempt": config.attempt,
        "split_seed": config.split_seed,
        "model_seed": config.model_seed,
        "job_classification": "auxiliary_validation",
        "counts_toward_primary_model_training_executions": False,
    }
    writer = ArtifactManifestWriter(
        run_dir,
        run_metadata=run_metadata,
        requirements=_shuffled_requirements(),
    )
    writer.start()
    atomic_write_text(run_dir / "stdout.log", "")
    atomic_write_text(run_dir / "stderr.log", "")
    requested = {
        "status": "running",
        "run_id": config.run_id,
        "request": _json_shuffled_config(config),
        "argv": list(sys.argv),
        "resolved_command": [sys.executable, *sys.argv],
        "split_seed": config.split_seed,
        "model_seed": config.model_seed,
        "shuffled_label_seed": config.shuffled_label_seed,
        "timesteps": config.timesteps,
        "job_classification": "auxiliary_validation",
        "counts_toward_primary_model_training_executions": False,
        "performance_comparison_eligible": False,
        "reward_config": REWARD_CONFIG,
        "lightweight_training_config": _shuffled_training_config(config.timesteps),
    }
    atomic_write_json(run_dir / "config.json", requested)
    timing = TimingRecorder()
    monitor: ResourceMonitor | None = None
    monitor_stopped = False
    environment: DummyVecEnv | None = None

    try:
        with tee_output(run_dir / "stdout.log", run_dir / "stderr.log"):
            seed_model_rngs(config.model_seed)
            atomic_write_json(
                run_dir / "environment.json",
                collect_environment_metadata(repo_root=_REPO_ROOT),
            )
            monitor = monitor_factory(run_dir, interval_seconds=config.monitor_interval)
            monitor.start()

            with timing.measure("preprocessing") as measurement:
                split_config = QRDQNRunConfig(
                    artifact_root=config.artifact_root,
                    run_id=config.run_id,
                    dataset_root=config.dataset_root,
                    cache_root=config.cache_root,
                    cache_policy=config.cache_policy,
                    split_mode="random",
                    split_seed=config.split_seed,
                    model_seed=config.model_seed,
                    timesteps=1,
                    checkpoint_freq=0,
                )
                raw_split = split_loader(split_config)
                if raw_split.X_train.shape[1] != 152 or raw_split.X_test.shape[1] != 152:
                    raise ValueError("Canonical shuffled-label observations must have 152 columns")
                permutation, shuffled_labels, permutation_hash = build_label_permutation(
                    raw_split.y_train,
                    seed=config.shuffled_label_seed,
                )
                scaler = StandardScaler()
                X_train = scaler.fit_transform(raw_split.X_train).astype(np.float32)
                X_test = scaler.transform(raw_split.X_test).astype(np.float32)
                split_metadata = {
                    **raw_split.metadata,
                    "split_mode": "random",
                    "split_seed": config.split_seed,
                    "model_seed_applies_to_partition": False,
                    "train_set_sha256": sha256_array(raw_split.X_train),
                    "y_train_sha256": sha256_array(raw_split.y_train),
                    "test_set_sha256": sha256_array(raw_split.X_test),
                    "y_test_sha256": sha256_array(raw_split.y_test),
                    "label_permutation_sha256": permutation_hash,
                    "shuffled_y_train_sha256": sha256_array(shuffled_labels),
                    "array_hash_contract": "canonical_unscaled_v1",
                    "n_train": int(len(raw_split.y_train)),
                    "n_test": int(len(raw_split.y_test)),
                }
                measurement.set_units(len(X_train) + len(X_test), "rows")

            joblib.dump(scaler, run_dir / "scaler.joblib")
            atomic_write_json(run_dir / "feature_names.json", raw_split.feature_names)
            tensorboard_dir = run_dir / "tensorboard"
            tensorboard_dir.mkdir(parents=True, exist_ok=True)

            max_steps = min(5_000, len(X_train))

            def create_environment() -> Monitor:
                return Monitor(
                    RLDatasetDefenderEnv(
                        X=X_train,
                        y=shuffled_labels,
                        benign_label=0,
                        attack_label=1,
                        reward_config=REWARD_CONFIG,
                        max_steps_per_episode=max_steps,
                        shuffle=True,
                    )
                )

            environment = DummyVecEnv([create_environment])
            environment.seed(config.model_seed)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = model_factory(config, environment, tensorboard_dir, device)
            with timing.measure("training") as measurement:
                model.learn(
                    total_timesteps=config.timesteps,
                    tb_log_name=config.run_id,
                    reset_num_timesteps=False,
                )
                measurement.set_units(config.timesteps, "timesteps")
            model.save(str(run_dir / "model"))
            if not (run_dir / "model.zip").is_file():
                raise FileNotFoundError("Shuffled-label model.save did not produce model.zip")

            with timing.measure("evaluation") as measurement:
                y_pred = batched_predict(model, X_test, config.eval_batch_size)
                measurement.set_units(len(y_pred), "rows")
            y_true = raw_split.y_test.astype(np.int64)
            matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
            tn, fp, fn, tp = (int(value) for value in matrix.ravel())
            metrics = confusion_to_metrics(
                tn,
                fp,
                fn,
                tp,
                reward_config=REWARD_CONFIG,
                undefined_metric_policy="null",
            )
            baseline_accuracy = max(float((y_true == 0).mean()), float((y_true == 1).mean()))
            metrics.update(
                {
                    "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
                    "shuffled_accuracy": metrics["accuracy"],
                    "baseline_accuracy": baseline_accuracy,
                    "leakage_threshold": baseline_accuracy + 0.05,
                    "leakage_detected": bool(
                        metrics["accuracy"] is not None
                        and metrics["accuracy"] > baseline_accuracy + 0.05
                    ),
                    "control_interpretation": "anti_leakage_only_not_model_performance",
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
                    "split_metadata": split_metadata,
                    "label_permutation_sha256": permutation_hash,
                    "permutation_length": int(len(permutation)),
                    "scaler_fit": "final_training_partition_only",
                }
            )
            atomic_write_json(run_dir / "config.json", requested)

        writer.complete()
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
        writer.fail(error)
        raise
    finally:
        if environment is not None:
            environment.close()


class ProgressCallback(BaseCallback):
    """Callback para mostrar progreso cada log_freq timesteps."""
    def __init__(self, log_freq: int = 10_000, verbose: int = 1):
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
        gamma=0.0,
        tau=1.0,
        train_freq=50,
        target_update_interval=1_000,
        verbose=1,
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
    preset: str = "fast",
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
    print(f"  Preset:     {preset}")
    print(f"  Timesteps:  {timesteps}")

    X_train, y_train, X_test, y_test, scaler, feature_names, split_meta = load_cicids2017_split(
        split_mode="day",
        preset=preset,
        seed=seed,
        max_rows=max_rows,
        train_days=train_csvs,
        test_days=test_csvs,
        scale=True,
        use_canonical=True,
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
        gamma=0.0,
        tau=1.0,
        train_freq=100,
        target_update_interval=10_000,
        verbose=1,
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
        "preset": preset,
        "timesteps": timesteps,
        "split_metadata": split_meta,
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
        help="Path to trained model .zip (must match --run-dir unless unsafe)",
    )
    parser.add_argument(
        "--run-dir", type=Path, default=None,
        help="Trusted training run dir containing artifact_manifest.json.",
    )
    parser.add_argument(
        "--checks", nargs="+", default=["A", "B", "C"],
        choices=["A", "B", "C"],
        help="Which checks to run (default: all)",
    )
    parser.add_argument(
        "--preset", type=str, default="fast", choices=["fast", "full"],
        help="Preset: fast (lightweight, capped rows) or full (all rows). Default: fast",
    )
    parser.add_argument(
        "--split-mode", type=str, default="random", choices=["random", "day"],
        help="Split mode for Check A/B data loading: random or day. Default: random",
    )
    parser.add_argument(
        "--max-rows", type=int, default=None,
        help="Max rows to load from dataset (overrides preset default)",
    )
    parser.add_argument(
        "--dataset-root", type=Path, default=_REPO_ROOT / "datasets" / "CICIDS2017",
        help="Provider-neutral CICIDS2017 dataset root",
    )
    parser.add_argument(
        "--cache-root", type=Path, default=None,
        help="Validated canonical unscaled cache root required by Check B",
    )
    parser.add_argument(
        "--cache-policy", choices=["off", "prefer", "require"], default="require",
        help="Canonical cache policy for the artifact-backed Check B control",
    )
    parser.add_argument(
        "--artifact-root", type=Path, default=RUNS_DIR / "validation",
        help="Artifact root for the independent Check B auxiliary run",
    )
    parser.add_argument(
        "--run-id-b", default=None,
        help="Physical run ID for the artifact-backed shuffled-label control",
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
        "--train-csvs", nargs="+", default=DEFAULT_TRAIN_DAYS,
        help="CSV name patterns for training in Check C (default: Monday Tuesday Wednesday)",
    )
    parser.add_argument(
        "--test-csvs", nargs="+", default=DEFAULT_TEST_DAYS,
        help="CSV name patterns for testing in Check C (default: Thursday Friday)",
    )
    parser.add_argument(
        "--split-seed", type=int, default=None,
        help="Partition seed (default: 42)",
    )
    parser.add_argument(
        "--model-seed", type=int, default=None,
        help="Model/environment seed (default: 42)",
    )
    parser.add_argument(
        "--shuffled-label-seed", type=int, default=None,
        help="Label permutation seed for Check B (default: 42)",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Legacy alias setting split, model, and shuffled-label seeds",
    )
    parser.add_argument(
        "--allow-unsafe-artifacts", action="store_true",
        help="Allow direct model paths without manifest hash verification.",
    )
    args = parser.parse_args()
    explicit_seeds = (args.split_seed, args.model_seed, args.shuffled_label_seed)
    if args.seed is not None and any(seed is not None for seed in explicit_seeds):
        parser.error("--seed cannot be combined with explicit seed flags")
    if args.seed is not None:
        args.split_seed = args.seed
        args.model_seed = args.seed
        args.shuffled_label_seed = args.seed
    else:
        args.split_seed = 42 if args.split_seed is None else args.split_seed
        args.model_seed = 42 if args.model_seed is None else args.model_seed
        args.shuffled_label_seed = (
            42 if args.shuffled_label_seed is None else args.shuffled_label_seed
        )
    return args


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
    print(f"  Preset: {args.preset}")
    print(f"  Split mode (A/B): {args.split_mode}")
    print(f"  Output: {run_dir}")
    print(f"  Device: {device}")
    print(f"{'=' * 60}")

    results: Dict[str, Dict] = {}

    # ── Cargar datos para Check A y B (uses unified split API) ──
    need_ab_data = "A" in checks
    X_train = y_train = X_test = y_test = None

    if need_ab_data:
        print(f"\nCargando CICIDS2017 (split_mode={args.split_mode}, preset={args.preset})...")
        X_train, y_train, X_test, y_test, _, _, ab_meta = load_cicids2017_split(
            split_mode=args.split_mode,
            preset=args.preset,
            split_seed=args.split_seed,
            max_rows=args.max_rows,
            scale=True,
            use_canonical=True,
            local_dir=args.dataset_root,
            cache_root=args.cache_root,
            cache_policy="off" if args.cache_root is None else args.cache_policy,
        )
        print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    # ── Check A ──
    if "A" in checks:
        try:
            requested_model = Path(args.model) if args.model is not None else None
            model_path = resolve_trusted_artifact(
                args.run_dir,
                "model",
                requested_model,
                repo_root=_REPO_ROOT,
                allow_unsafe=args.allow_unsafe_artifacts,
            )
        except ArtifactTrustError:
            print("\nCheck A requiere modelo confiable. Saltando carga de modelo no verificada.")
        else:
            print("\nCargando modelo confiable desde artefacto verificado.")
            model = QRDQN.load(str(model_path), device=device)
            results["A"] = check_a_direct_eval(model, X_test, y_test)

    # ── Check B ──
    if "B" in checks:
        shuffled_run_id = args.run_id_b or f"shuffled_label_validation_{timestamp}"
        shuffled_run_dir = run_shuffled_label_validation(
            ShuffledLabelRunConfig(
                artifact_root=args.artifact_root,
                run_id=shuffled_run_id,
                dataset_root=args.dataset_root,
                cache_root=args.cache_root,
                cache_policy=args.cache_policy,
                split_seed=args.split_seed,
                model_seed=args.model_seed,
                shuffled_label_seed=args.shuffled_label_seed,
                timesteps=args.timesteps_b,
            )
        )
        results["B"] = json.loads(
            (shuffled_run_dir / "metrics.json").read_text(encoding="utf-8")
        )

    # ── Check C ──
    if "C" in checks:
        results["C"] = check_c_csv_split(
            train_csvs=args.train_csvs,
            test_csvs=args.test_csvs,
            timesteps=args.timesteps_c,
            max_rows=args.max_rows,
            preset=args.preset,
            seed=args.split_seed,
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
        "preset": args.preset,
        "split_mode": args.split_mode,
        "model_path": args.model,
        "trusted_run_dir": str(args.run_dir) if args.run_dir else None,
        "allow_unsafe_artifacts": bool(args.allow_unsafe_artifacts),
        "max_rows": args.max_rows,
        "timesteps_b": args.timesteps_b,
        "timesteps_c": args.timesteps_c,
        "train_csvs_c": args.train_csvs,
        "test_csvs_c": args.test_csvs,
        "split_seed": args.split_seed,
        "model_seed": args.model_seed,
        "shuffled_label_seed": args.shuffled_label_seed,
        "device": device,
        "reward_config": REWARD_CONFIG,
    }
    config_path = run_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    # ── Resumen final ──
    print(f"\n{'=' * 60}")
    print("  RESUMEN DE VALIDACIÓN")
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
