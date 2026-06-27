from pathlib import Path
from datetime import datetime
import json

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

try:
    from src.load_cicids2017 import (
        CICIDSLoadConfig,
        load_cicids2017_binary,
        load_cicids2017_csv_split,
        load_cicids2017_exact_csv_split,
    )
    from src.metrics_utils import confusion_to_metrics
except ModuleNotFoundError:
    from load_cicids2017 import (
        CICIDSLoadConfig,
        load_cicids2017_binary,
        load_cicids2017_csv_split,
        load_cicids2017_exact_csv_split,
    )
    from metrics_utils import confusion_to_metrics


MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_DIR = Path("runs") / "cicids2017" / "baseline_random_forest_comparison"

# Dataset: "nslkdd" o "cicids2017"
DATASET = "cicids2017"

# Same-protocol comparison vs QRDQN: same canonical observation, scaled features
# (RF is scale-invariant, but this keeps the preprocessing identical), and
# class_weight="balanced" to match the cost-sensitive treatment the methodology
# describes (otherwise the baseline would be biased toward the benign majority).
RF_PARAMS = dict(
    n_estimators=200,
    max_depth=None,
    n_jobs=-1,
    class_weight="balanced",
    random_state=42,
)


def train_random_forest(X_train, y_train) -> RandomForestClassifier:
    """Entrena un RandomForestClassifier (balanced, mismo protocolo que QRDQN)."""
    rf = RandomForestClassifier(**RF_PARAMS)
    rf.fit(X_train, y_train)
    return rf


def score_and_save(name, sweep_cfg, model, X_test, y_test, run_id) -> dict:
    """Evalúa el RF, imprime el informe y persiste config.json + metrics.json."""
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(v) for v in cm.ravel())
    metrics = confusion_to_metrics(tn, fp, fn, tp)

    print(f"\n=== Random Forest [{name}] - Confusion matrix (0=normal, 1=ataque) ===")
    print(cm)
    print(classification_report(y_test, y_pred, digits=4))

    out_dir = RESULTS_DIR / f"{run_id}__{name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "run_id": run_id,
        "sweep": name,
        "model": "RandomForest",
        "rf_params": dict(RF_PARAMS),
        **sweep_cfg,
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"[artifact] {out_dir}")
    return metrics


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    RUN_ID = f"rf_{DATASET}_canonical_{timestamp}"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # --- Sweep 1: Random split (full) — same split/seed/preprocessing as MAIN QRDQN ---
    print("\n=== [SWEEP 1] Random Split (full) ===")
    X_tr, y_tr, X_te, y_te, _, _ = load_cicids2017_binary(
        CICIDSLoadConfig(max_rows=None, use_canonical=True, scale=True)
    )
    rf_random = train_random_forest(X_tr, y_tr)
    score_and_save(
        "random_split",
        {"split_mode": "random", "scale": True, "use_canonical": True, "seed": 42, "test_size": 0.2},
        rf_random, X_te, y_te, RUN_ID,
    )

    # --- Sweep 2: Day split (train Mon/Tue/Wed, test Thu/Fri) — matches QRDQN Check C / default ---
    print("\n=== [SWEEP 2] Day Split (train Mon/Tue/Wed, test Thu/Fri) ===")
    X_tr, y_tr, X_te, y_te, _, _ = load_cicids2017_csv_split(
        train_csvs=["Monday", "Tuesday", "Wednesday"],
        test_csvs=["Thursday", "Friday"],
        cfg=CICIDSLoadConfig(max_rows=None, use_canonical=True, scale=True),
    )
    rf_day = train_random_forest(X_tr, y_tr)
    score_and_save(
        "day_split",
        {"split_mode": "day", "train_days": ["Monday", "Tuesday", "Wednesday"],
         "test_days": ["Thursday", "Friday"], "scale": True, "use_canonical": True},
        rf_day, X_te, y_te, RUN_ID,
    )

    # --- Sweep 3: Leave-one-CSV-out (Wednesday held out) ---
    print("\n=== [SWEEP 3] Leave-One-Out (Wednesday test) ===")
    train_exact = [
        "Monday-WorkingHours.pcap_ISCX.csv",
        "Tuesday-WorkingHours.pcap_ISCX.csv",
        "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
        "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
        "Friday-WorkingHours-Morning.pcap_ISCX.csv",
        "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
        "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    ]
    test_exact = ["Wednesday-workingHours.pcap_ISCX.csv"]
    X_tr, y_tr, X_te, y_te, _, _ = load_cicids2017_exact_csv_split(
        train_exact, test_exact,
        cfg=CICIDSLoadConfig(max_rows=None, use_canonical=True, scale=True),
    )
    rf_loo = train_random_forest(X_tr, y_tr)
    score_and_save(
        "leave_one_out_wednesday",
        {"split_mode": "leave_one_csv_out", "test_csv": test_exact[0],
         "scale": True, "use_canonical": True},
        rf_loo, X_te, y_te, RUN_ID,
    )

    # Save the random-split model as the generic baseline artifact.
    model_path = MODELS_DIR / f"{RUN_ID}.joblib"
    try:
        import joblib
        joblib.dump(rf_random, model_path)
        print(f"\nModelo Random Forest (Random split) guardado en: {model_path}")
    except ImportError:
        print("joblib no esta instalado; omitiendo guardado del modelo.")


if __name__ == "__main__":
    main()
