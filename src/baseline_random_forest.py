from pathlib import Path
from typing import Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

from load_nsl_kdd import load_nsl_kdd_binary


MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = 200,
    max_depth: int | None = None,
    n_jobs: int = -1,
) -> RandomForestClassifier:
    """
    Entrena un RandomForestClassifier sobre NSL-KDD y devuelve el modelo.
    """
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        n_jobs=n_jobs,
        class_weight=None,  # si quieres, luego probamos 'balanced'
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
    # 1) Cargar mismo dataset que usa el RL
    print("Cargando NSL-KDD (20%) para baseline Random Forest...")
    X_train, y_train, X_test, y_test = load_nsl_kdd_binary(
        use_20_percent=False
    )

    print(f"Train shape: X={X_train.shape}, y={y_train.shape}")
    print(f"Test  shape: X={X_test.shape}, y={y_test.shape}")

    # 2) Entrenar Random Forest
    print("Entrenando Random Forest...")
    rf = train_random_forest(X_train, y_train)

    # 3) Guardar modelo (opcional, por si luego quieres cargarlo)
    model_path = MODELS_DIR / "rf_nslkdd.joblib"
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
