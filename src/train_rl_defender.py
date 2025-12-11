from pathlib import Path
from typing import List

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv

from rl_defender_env import RLDatasetDefenderEnv
from load_nsl_kdd import load_nsl_kdd_binary


MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------------------
# Configuración de recompensa para el agente defensor
# --------------------------------------------------------------------------------------
REWARD_CONFIG = {
    "tp": 1.0,     # bloquear ataque
    "fp": -1.0,    # bloquear benigno
    "fn": -5.0,    # permitir ataque
    "omission": 0.5,  # no toma acción (pero acierta)
}


def make_env_fn(X: np.ndarray, y: np.ndarray):
    """
    Devuelve una función creadora de entornos para usar con DummyVecEnv.
    """

    def _init():
        return RLDatasetDefenderEnv(
            X=X,
            y=y,
            benign_label=0,
            attack_label=1,
            reward_config=REWARD_CONFIG,
            max_steps_per_episode=min(10_000, len(X)),
            shuffle=True,
        )

    return _init


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray) -> None:
    """
    Evalúa el agente entrenado sobre un conjunto de test y
    muestra métricas de clasificación (confusion matrix + classification report).
    """
    env_test = RLDatasetDefenderEnv(
        X=X_test,
        y=y_test,
        benign_label=0,
        attack_label=1,
        reward_config=REWARD_CONFIG,
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

    print("=== Confusion matrix (acciones: 0=PERMIT, 1=BLOCK) ===")
    print(confusion_matrix(y_true, y_pred))
    print()
    print("=== Classification report ===")
    print(classification_report(y_true, y_pred, digits=4))


def main():
    # ------------------------------------------------------------------
    # 1) Cargar NSL-KDD ya preprocesado (desde KaggleHub)
    # ------------------------------------------------------------------
    print("Descargando y cargando NSL-KDD vía kagglehub...")
    X_train, y_train, X_test, y_test = load_nsl_kdd_binary(
        use_20_percent=False  # pon False cuando quieras entrenar con el dataset completo
    )

    print(f"Train shape: X={X_train.shape}, y={y_train.shape}")
    print(f"Test  shape: X={X_test.shape}, y={y_test.shape}")

    # ------------------------------------------------------------------
    # 2) Crear entorno vectorizado para Stable-Baselines3
    # ------------------------------------------------------------------
    vec_env = DummyVecEnv([make_env_fn(X_train, y_train)])

    # ------------------------------------------------------------------
    # 3) Definir el modelo RL (DQN)
    # ------------------------------------------------------------------
    model = DQN(
        "MlpPolicy",
        vec_env,
        learning_rate=1e-3,
        buffer_size=100_000,
        batch_size=64,
        gamma=0.99,
        tau=1.0,
        train_freq=4,
        target_update_interval=10_000,
        verbose=1,
    )

    total_timesteps = 1_000_000
    print(f"Entrenando DQN durante {total_timesteps} timesteps...")
    model.learn(total_timesteps=total_timesteps)

    # ------------------------------------------------------------------
    # 4) Guardar modelo
    # ------------------------------------------------------------------
    model_path = MODELS_DIR / "rl_defender_dqn_nslkdd"
    print(f"Guardando modelo en: {model_path}")
    model.save(str(model_path))

    # ------------------------------------------------------------------
    # 5) Evaluación en test
    # ------------------------------------------------------------------
    print("Evaluando en conjunto de test...")
    evaluate_model(model, X_test, y_test)


if __name__ == "__main__":
    main()
