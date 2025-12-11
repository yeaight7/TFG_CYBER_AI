import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

from stable_baselines3 import DQN  # puedes cambiar a PPO si quieres
from stable_baselines3.common.vec_env import DummyVecEnv

from rl_defender_env import RLDatasetDefenderEnv


# Ruta a tu dataset (ajústala a tu caso)
# Se asume un CSV con una columna 'label' y el resto features numéricas.
DATA_PATH = Path("data/dataset.csv")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def load_dataset(path: Path):
    """
    Carga un dataset desde CSV.

    - Columna 'label' = etiqueta (0 benigno, 1 ataque).
    - El resto de columnas se usan como features numéricas.
    """
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el dataset en: {path}")

    df = pd.read_csv(path)

    if "label" not in df.columns:
        raise ValueError("El CSV debe contener una columna 'label'.")

    y = df["label"].values.astype(np.int64)
    X = df.drop(columns=["label"]).values.astype(np.float32)

    # Escalado de features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X).astype(np.float32)

    return X_scaled, y, scaler


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
            correct_reward=1.0,
            false_positive_penalty=-1.0,
            false_negative_penalty=-2.0,
            max_steps_per_episode=min(10_000, len(X)),  # por ejemplo
            shuffle=True,
        )

    return _init


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray):
    """
    Evalúa el agente entrenado sobre un conjunto de test y
    muestra métricas de clasificación.
    """
    env_test = RLDatasetDefenderEnv(
        X=X_test,
        y=y_test,
        benign_label=0,
        attack_label=1,
        correct_reward=1.0,
        false_positive_penalty=-1.0,
        false_negative_penalty=-2.0,
        max_steps_per_episode=len(X_test),
        shuffle=False,
    )

    obs, info = env_test.reset()
    done = False

    y_true: list[int] = []
    y_pred: list[int] = []

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
    print(f"Cargando dataset desde: {DATA_PATH}")
    X, y, scaler = load_dataset(DATA_PATH)

    # Split train/test estratificado
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # Entorno vectorizado (un solo entorno, pero DummyVecEnv lo envuelve para SB3)
    env = DummyVecEnv([make_env_fn(X_train, y_train)])

    # Definición del modelo RL (DQN). Puedes ajustar hiperparámetros.
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=1e-3,
        buffer_size=100_000,
        batch_size=64,
        gamma=0.99,
        tau=1.0,
        train_freq=4,
        target_update_interval=10_000,
        verbose=1,
    )

    total_timesteps = 200_000
    print(f"Entrenando DQN durante {total_timesteps} timesteps...")
    model.learn(total_timesteps=total_timesteps)

    model_path = MODELS_DIR / "rl_defender_dqn"
    print(f"Guardando modelo en: {model_path}")
    model.save(str(model_path))

    # Evaluación en test
    print("Evaluando en conjunto de test...")
    evaluate_model(model, X_test, y_test)


if __name__ == "__main__":
    main()
