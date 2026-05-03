from pathlib import Path
from typing import List, Optional, Tuple
from datetime import datetime

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

from src.load_nsl_kdd import load_nsl_kdd_binary
from src.load_cicids2017 import load_cicids2017_binary, CICIDSLoadConfig


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
    
    # --- Sweep 1: Random Split (full) ---
    print("\n=== [SWEEP 1] Random Split (full) ===")
    cfg_random = CICIDSLoadConfig(max_rows=None, use_canonical=True, scale=False)
    X_train_r, y_train_r, X_test_r, y_test_r, _, feats = load_cicids2017_binary(cfg_random)
    rf_random = train_random_forest(X_train_r, y_train_r)
    evaluate_random_forest(rf_random, X_test_r, y_test_r)
    
    # --- Sweep 2: Day Split (Check C equivalent) ---
    print("\n=== [SWEEP 2] Day Split (Check C) ===")
    from load_cicids2017 import load_cicids2017_csv_split
    X_train_c, y_train_c, X_test_c, y_test_c, _, _ = load_cicids2017_csv_split(
        train_csvs=["Monday", "Tuesday", "Wednesday", "Thursday"],
        test_csvs=["Friday"],
        cfg=CICIDSLoadConfig(max_rows=None, use_canonical=True, scale=False)
    )
    rf_c = train_random_forest(X_train_c, y_train_c)
    evaluate_random_forest(rf_c, X_test_c, y_test_c)

    # --- Sweep 3: Leave-one-out (e.g. Wednesday out) ---
    print("\n=== [SWEEP 3] Leave-One-Out (Wednesday test) ===")
    from load_cicids2017 import load_cicids2017_exact_csv_split
    train_exact = [
        "Monday-WorkingHours.pcap_ISCX.csv",
        "Tuesday-WorkingHours.pcap_ISCX.csv",
        "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
        "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
        "Friday-WorkingHours-Morning.pcap_ISCX.csv",
        "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
        "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
    ]
    test_exact = ["Wednesday-workingHours.pcap_ISCX.csv"]
    X_train_l, y_train_l, X_test_l, y_test_l, _, _ = load_cicids2017_exact_csv_split(
        train_exact, test_exact, cfg=CICIDSLoadConfig(max_rows=None, use_canonical=True, scale=False)
    )
    rf_l = train_random_forest(X_train_l, y_train_l)
    evaluate_random_forest(rf_l, X_test_l, y_test_l)

    # 3) Guardar modelo (usamos el random split como el genérico)
    model_path = MODELS_DIR / f"{RUN_ID}.joblib"
    try:
        import joblib
        joblib.dump(rf_random, model_path)
        print(f"\nModelo Random Forest (Random split) guardado en: {model_path}")
    except ImportError:
        print("joblib no está instalado; omitiendo guardado del modelo.")


if __name__ == "__main__":
    main()
