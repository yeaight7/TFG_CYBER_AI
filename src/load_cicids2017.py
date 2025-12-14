from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple, List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import kagglehub


KAGGLE_HANDLE = "chethuhn/network-intrusion-dataset"


@dataclass(frozen=True)
class CICIDSLoadConfig:
    # Descarga/lectura
    handle: str = KAGGLE_HANDLE
    force_download: bool = False
    recursive: bool = True
    chunksize: int = 250_000            # para leer CSVs grandes por trozos
    max_rows: Optional[int] = None      # recorta el total cargado (útil para pruebas)
    sample_frac: Optional[float] = None # ej. 0.2 para quedarte con 20% tras cargar

    # Etiquetas
    label_col: str = "Label"
    benign_value: str = "BENIGN"        # CICIDS2017 suele usar "BENIGN"

    # Limpieza / features
    drop_identifier_cols: bool = True   # Flow ID / IPs / Timestamp, etc.
    scale: bool = True                 # StandardScaler (fit solo en train)

    # Split
    test_size: float = 0.2
    random_state: int = 42


def _download_dataset(cfg: CICIDSLoadConfig) -> Path:
    path = kagglehub.dataset_download(cfg.handle, force_download=cfg.force_download)
    return Path(path)


def _list_csv_files(root: Path, recursive: bool = True) -> List[Path]:
    pattern = "**/*.csv" if recursive else "*.csv"
    csvs = sorted(root.glob(pattern))
    if not csvs:
        raise FileNotFoundError(
            f"No se encontraron CSVs en: {root}. "
            "Comprueba que el dataset descargado contiene archivos .csv."
        )
    return csvs


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _find_label_column(df: pd.DataFrame, preferred: str) -> str:
    cols = {c.lower(): c for c in df.columns}
    if preferred.lower() in cols:
        return cols[preferred.lower()]
    # fallback: cualquier columna que se llame "label" ignorando espacios
    for c in df.columns:
        if str(c).strip().lower() == "label":
            return c
    raise ValueError(f"No se encontró columna de etiqueta. Columnas disponibles: {list(df.columns)[:20]} ...")


def _drop_identifier_like_columns(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    # Mantén label; elimina columnas típicas que generan leakage o no son útiles como features.
    drop_exact = {
        "Flow ID", "Timestamp",
        "Source IP", "Destination IP",
        "Src IP", "Dst IP",
        "External IP",
    }
    out = df.copy()
    for c in list(out.columns):
        if c == label_col:
            continue
        c_norm = str(c).strip()
        if c_norm in drop_exact:
            out.drop(columns=[c], inplace=True)
            continue
        # Cualquier columna con "ip" (pero no puertos), o "flow id", o "timestamp"
        low = c_norm.lower()
        if (" ip" in low) or (low.endswith("ip")) or ("flow id" in low) or ("timestamp" in low):
            out.drop(columns=[c], inplace=True)
    return out


def _coerce_numeric_features(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if c == label_col:
            continue
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def _clean_rows(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    out = df.copy()
    # Reemplaza inf por NaN y elimina filas incompletas
    out.replace([np.inf, -np.inf], np.nan, inplace=True)
    out.dropna(axis=0, how="any", inplace=True)
    # Elimina columnas constantes (opcional, pero suele ayudar)
    nunique = out.drop(columns=[label_col], errors="ignore").nunique()
    const_cols = nunique[nunique <= 1].index.tolist()
    if const_cols:
        out.drop(columns=const_cols, inplace=True, errors="ignore")
    return out


def _load_all_csvs(csv_paths: List[Path], cfg: CICIDSLoadConfig) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    loaded = 0

    for p in csv_paths:
        # Lectura por chunks para no reventar RAM
        for chunk in pd.read_csv(p, chunksize=cfg.chunksize, low_memory=True, encoding_errors="ignore"):
            chunk = _normalize_columns(chunk)

            # límite global de filas (si aplica)
            if cfg.max_rows is not None:
                remaining = cfg.max_rows - loaded
                if remaining <= 0:
                    break
                if len(chunk) > remaining:
                    chunk = chunk.iloc[:remaining].copy()

            frames.append(chunk)
            loaded += len(chunk)

            if cfg.max_rows is not None and loaded >= cfg.max_rows:
                break

        if cfg.max_rows is not None and loaded >= cfg.max_rows:
            break

    df = pd.concat(frames, ignore_index=True)
    return df


def load_cicids2017_binary(cfg: Optional[CICIDSLoadConfig] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[StandardScaler], List[str]]:
    """
    Devuelve:
      X_train, y_train, X_test, y_test, scaler, feature_names

    - y: 0 si Label == BENIGN, 1 en caso contrario
    - X: float32
    """
    cfg = cfg or CICIDSLoadConfig()

    root = _download_dataset(cfg)
    csvs = _list_csv_files(root, recursive=cfg.recursive)

    df = _load_all_csvs(csvs, cfg)

    label_col = _find_label_column(df, cfg.label_col)

    # Etiqueta binaria
    labels = df[label_col].astype(str).str.strip().str.upper()
    y = (labels != cfg.benign_value.upper()).astype(np.int64)

    # Features
    Xdf = df.drop(columns=[label_col]).copy()

    # Quita IDs / timestamps / IPs si procede
    if cfg.drop_identifier_cols:
        Xdf[label_col] = df[label_col]
        Xdf = _drop_identifier_like_columns(Xdf, label_col=label_col).drop(columns=[label_col])
    # Coerción numérica + limpieza
    tmp = Xdf.copy()
    tmp[label_col] = y  # para reutilizar limpieza
    tmp = _coerce_numeric_features(tmp, label_col=label_col)
    tmp = _clean_rows(tmp, label_col=label_col)

    y_clean = tmp[label_col].to_numpy(dtype=np.int64)
    X_clean_df = tmp.drop(columns=[label_col])

    # Si quedan columnas no numéricas por algún motivo, las eliminamos
    non_numeric = [c for c in X_clean_df.columns if not pd.api.types.is_numeric_dtype(X_clean_df[c])]
    if non_numeric:
        X_clean_df = X_clean_df.drop(columns=non_numeric)

    # Submuestreo opcional (después de limpiar, para no sesgar por NaNs)
    if cfg.sample_frac is not None:
        if not (0.0 < cfg.sample_frac <= 1.0):
            raise ValueError("sample_frac debe estar en (0, 1].")
        idx = np.random.default_rng(cfg.random_state).choice(
            len(X_clean_df),
            size=int(len(X_clean_df) * cfg.sample_frac),
            replace=False
        )
        X_clean_df = X_clean_df.iloc[idx].reset_index(drop=True)
        y_clean = y_clean[idx]

    feature_names = list(X_clean_df.columns)
    X_clean = X_clean_df.to_numpy(dtype=np.float32)

    # Split estratificado
    X_train, X_test, y_train, y_test = train_test_split(
        X_clean,
        y_clean,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=y_clean,
    )

    scaler: Optional[StandardScaler] = None
    if cfg.scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train).astype(np.float32)
        X_test = scaler.transform(X_test).astype(np.float32)

    return X_train, y_train, X_test, y_test, scaler, feature_names


if __name__ == "__main__":
    # Smoke test rápido
    cfg = CICIDSLoadConfig(max_rows=300_000, sample_frac=None)
    X_train, y_train, X_test, y_test, scaler, feats = load_cicids2017_binary(cfg)
    print(f"CICIDS2017: X_train={X_train.shape}, y_train={y_train.shape}")
    print(f"CICIDS2017: X_test ={X_test.shape}, y_test ={y_test.shape}")
    print(f"Features: {len(feats)}")
    benign_rate = (y_train == 0).mean()
    print(f"Train benign rate: {benign_rate:.4f}")
