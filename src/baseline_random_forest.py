from pathlib import Path
from typing import List, Optional, Tuple
from datetime import datetime

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

from load_nsl_kdd import load_nsl_kdd_binary
from load_cicids2017 import load_cicids2017_binary, CICIDSLoadConfig


MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Dataset: "nslkdd" o "cicids2017"
DATASET = "cicids2017"


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = 200,
    max_depth: int | None = None,
    n_jobs: int = -1,
) -> RandomForestClassifier:
    """
    Entrena un RandomForestClassifier sobre el dataset y devuelve el modelo.
    """
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        n_jobs=n_jobs,
        class_weight=None,  # note: try 'balanced'
        random_state=42,
    )
    rf.fit(X_train, y_train)
    return rf


def evaluate_random_forest(
    model: RandomForestClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> None:
    """
    Evalúa el Random Forest y muestra matriz de confusión + informe de clasificación.
    """
    y_pred = model.predict(X_test)

    print("=== Random Forest – Confusion matrix (clases: 0=normal, 1=ataque) ===")
    print(confusion_matrix(y_test, y_pred))
    print()
    print("=== Random Forest – Classification report ===")
    print(classification_report(y_test, y_pred, digits=4))


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    RUN_ID = f"rf_{DATASET}_canonical_{timestamp}"

    # 1) Cargar dataset con esquema canónico
    if DATASET == "cicids2017":
        print("Cargando CICIDS2017 con esquema canónico para baseline Random Forest...")
        cfg = CICIDSLoadConfig(
            max_rows=500_000,
            use_canonical=True,
            scale=False,  # RF no necesita scaling
        )
        X_train, y_train, X_test, y_test, scaler, feature_names = load_cicids2017_binary(cfg)
    elif DATASET == "nslkdd":
        print("Cargando NSL-KDD con esquema canónico para baseline Random Forest...")
        X_train, y_train, X_test, y_test, scaler, feature_names = load_nsl_kdd_binary(
            use_20_percent=False,
            use_canonical=True,
            scale=False,
        )
    else:
        raise ValueError(f"Dataset no soportado: {DATASET}. Usa 'nslkdd' o 'cicids2017'.")

    print(f"Train shape: X={X_train.shape}, y={y_train.shape}")
    print(f"Test  shape: X={X_test.shape}, y={y_test.shape}")
    print(f"Features: {len(feature_names)}")

    # 2) Entrenar Random Forest
    print("Entrenando Random Forest...")
    rf = train_random_forest(X_train, y_train)

    # 3) Guardar modelo
    model_path = MODELS_DIR / f"{RUN_ID}.joblib"
    try:
        import joblib
        joblib.dump(rf, model_path)
        print(f"Modelo Random Forest guardado en: {model_path}")
    except ImportError:
        print("joblib no está instalado; omitiendo guardado del modelo.")

    # 4) Evaluación en test
    print("Evaluando Random Forest en conjunto de test...")
    evaluate_random_forest(rf, X_test, y_test)


if __name__ == "__main__":
    main()
