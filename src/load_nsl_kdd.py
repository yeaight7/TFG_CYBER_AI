from pathlib import Path
from typing import Tuple
import shutil

import numpy as np
import pandas as pd

try:
    import kagglehub 
except ImportError as e:
    raise ImportError(
        "Instala kagglehub en tu venv: pip install kagglehub"
    ) from e


NSL_KDD_COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
    "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files",
    "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
    "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate",
    "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
    "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
    "attack_or_class",
    "level_or_difficulty",
]


def _download_nsl_kdd_via_kagglehub() -> Path:
    path = kagglehub.dataset_download("hassan06/nslkdd")
    return Path(path)

def _ensure_dataset_local_dir(target_dir: Path) -> Path:
    target_dir = target_dir.resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    kaggle_dir = _download_nsl_kdd_via_kagglehub()

    # Copiar todos los .txt (por si el dataset tiene variantes)
    for src_path in kaggle_dir.glob("*.txt"):
        if src_path.is_file():
            dest_path = target_dir / src_path.name
            if not dest_path.exists():
                shutil.copy2(src_path, dest_path)

    return target_dir

def load_nsl_kdd_binary(
    use_20_percent: bool = False,
    dataset_dir: Path | str | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Carga NSL-KDD desde Kaggle (hassan06/nslkdd), lo preprocesa y devuelve:

        X_train, y_train, X_test, y_test

    - Etiqueta binaria: 0 = normal, 1 = ataque.
    - Categóricas one-hot: protocol_type, service, flag.
    """
    
    if dataset_dir is None:
        repo_root = Path(__file__).resolve().parents[1] 
        dataset_dir = repo_root / "datasets" / "nsl_kdd"
    else:
        dataset_dir = Path(dataset_dir)

    base_dir = _ensure_dataset_local_dir(dataset_dir)

    # Elegir ficheros: versión completa o 20%
    if use_20_percent:
        train_file = base_dir / "KDDTrain+_20Percent.txt"
    else:
        train_file = base_dir / "KDDTrain+.txt"

    test_file = base_dir / "KDDTest+.txt"

    if not train_file.exists() or not test_file.exists():
        raise FileNotFoundError(
            f"No se han encontrado KDDTrain/KDDTest en {base_dir}. "
            f"Comprueba qué ficheros hay realmente en la carpeta."
        )

    # Cargar sin cabeceras, usando los nombres estándar
    train_df = pd.read_csv(
        train_file,
        header=None,
        names=NSL_KDD_COLUMNS,
    )
    test_df = pd.read_csv(
        test_file,
        header=None,
        names=NSL_KDD_COLUMNS,
    )

    # Unir para procesar de forma consistente
    full_df = pd.concat([train_df, test_df], axis=0, ignore_index=True)

    # Detectar columna de etiqueta (attack/class/label) y de dificultad (level/difficulty)
    label_col_candidates = ["attack", "class", "attack_or_class", "label"]
    label_col = None
    for c in label_col_candidates:
        if c in full_df.columns:
            label_col = c
            break
    if label_col is None:
        # caemos al nombre "attack_or_class" que le dimos, si existe
        if "attack_or_class" in full_df.columns:
            label_col = "attack_or_class"
        else:
            raise ValueError(
                f"No se ha encontrado columna de etiqueta entre {label_col_candidates} "
                f"en NSL-KDD."
            )

    difficulty_col_candidates = ["level", "difficulty", "level_or_difficulty"]
    diff_col = None
    for c in difficulty_col_candidates:
        if c in full_df.columns:
            diff_col = c
            break

    # Crear etiqueta binaria: 0 = normal, 1 = ataque
    label_str = full_df[label_col].astype(str)
    # robusto a 'normal' o 'normal.' y a mayúsculas
    full_df["label"] = (~label_str.str.contains("normal", case=False)).astype("int64")

    # Eliminar columnas de salida originales (string + dificultad)
    drop_cols = [label_col]
    if diff_col is not None:
        drop_cols.append(diff_col)
    full_df = full_df.drop(columns=drop_cols, errors="ignore")

    # One-hot de categóricas (si existen en esta versión)
    categorical_cols = [c for c in ["protocol_type", "service", "flag"] if c in full_df.columns]
    full_df = pd.get_dummies(full_df, columns=categorical_cols)

    # Volver a separar train/test respetando el tamaño original del train
    n_train = len(train_df)
    train_processed = full_df.iloc[:n_train].copy()
    test_processed = full_df.iloc[n_train:].copy()

    # Separar X/y
    y_train = train_processed["label"].to_numpy(dtype="int64")
    X_train = train_processed.drop(columns=["label"]).to_numpy(dtype="float32")

    y_test = test_processed["label"].to_numpy(dtype="int64")
    X_test = test_processed.drop(columns=["label"]).to_numpy(dtype="float32")

    return X_train, y_train, X_test, y_test
