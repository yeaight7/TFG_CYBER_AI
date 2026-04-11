from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from canonical_schema import (
    CICIDS2017_TO_CANON,
    NUM_OBSERVATION_FEATURES,
    map_to_canonical,
    get_observation_feature_names,
)


# Ruta por defecto: datasets/CICIDS2017/ relativa a la raíz del repo
_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_LOCAL_DIR = _REPO_ROOT / "datasets" / "CICIDS2017"


@dataclass(frozen=True)
class CICIDSLoadConfig:
    # Directorio local con CSVs de CICIDS2017
    local_dir: Path = _DEFAULT_LOCAL_DIR
    chunksize: int = 250_000            # para leer CSVs grandes por trozos
    max_rows: Optional[int] = None      # recorta el total cargado (útil para pruebas)
    sample_frac: Optional[float] = None # ej. 0.2 para quedarte con 20% tras cargar

    # Etiquetas
    label_col: str = "Label"
    benign_value: str = "BENIGN"        # CICIDS2017 suele usar "BENIGN"

    # Limpieza / features
    drop_identifier_cols: bool = True   # Destination Port, Flow ID, IPs, Timestamp, etc.
    scale: bool = True                  # StandardScaler (fit solo en train)

    # Canonical schema
    use_canonical: bool = True          # mapear al esquema canónico con missingness mask

    # Split
    test_size: float = 0.2
    random_state: int = 42


def _list_csv_files(root: Path) -> List[Path]:
    csvs = sorted(root.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(
            f"No se encontraron CSVs en: {root}. "
            "Comprueba que el directorio contiene archivos .csv de CICIDS2017."
        )
    return csvs


def list_cicids2017_csv_files(local_dir: Optional[Path] = None) -> List[Path]:
    """
    Lista los CSVs reales de CICIDS2017 en orden determinista.

    Parameters
    ----------
    local_dir : Path or None
        Directorio que contiene los CSVs. Si es ``None``, usa
        ``datasets/CICIDS2017/`` relativo a la raíz del repo.
    """
    return _list_csv_files(local_dir or _DEFAULT_LOCAL_DIR)


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
    # Destination Port se elimina porque puede actuar como proxy de la etiqueta
    # (ciertos ataques usan puertos específicos, causando data leakage).
    drop_exact = {
        "Flow ID", "Timestamp",
        "Source IP", "Destination IP",
        "Src IP", "Dst IP",
        "Source Port", "Destination Port",
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
    # Reemplaza inf por NaN
    out.replace([np.inf, -np.inf], np.nan, inplace=True)
    # Rellenar NaN con 0: apropiado para features de flujo (contadores, bytes, tasas)
    # donde ausencia de valor indica ausencia de actividad. La máscara de missingness
    # del esquema canónico complementa esto indicando qué features son confiables.
    feat_cols = [c for c in out.columns if c != label_col]
    out[feat_cols] = out[feat_cols].fillna(0)
    # Eliminar filas donde la etiqueta es NaN
    out.dropna(subset=[label_col], inplace=True)
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


def _load_csv_with_row_limit(path: Path, cfg: CICIDSLoadConfig, row_limit: Optional[int] = None) -> pd.DataFrame:
    """Carga un CSV individual con límite opcional de filas por archivo."""
    frames: List[pd.DataFrame] = []
    loaded = 0

    for chunk in pd.read_csv(path, chunksize=cfg.chunksize, low_memory=True, encoding_errors="ignore"):
        chunk = _normalize_columns(chunk)

        if row_limit is not None:
            remaining = row_limit - loaded
            if remaining <= 0:
                break
            if len(chunk) > remaining:
                chunk = chunk.iloc[:remaining].copy()

        frames.append(chunk)
        loaded += len(chunk)

        if row_limit is not None and loaded >= row_limit:
            break

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def _prepare_cicids_features(
    df: pd.DataFrame,
    cfg: CICIDSLoadConfig,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Limpia CICIDS2017 y devuelve X, y y nombres de features."""
    label_col = _find_label_column(df, cfg.label_col)

    labels = df[label_col].astype(str).str.strip().str.upper()
    y = (labels != cfg.benign_value.upper()).astype(np.int64)

    Xdf = df.drop(columns=[label_col]).copy()

    if cfg.drop_identifier_cols:
        Xdf[label_col] = df[label_col]
        Xdf = _drop_identifier_like_columns(Xdf, label_col=label_col).drop(columns=[label_col])

    tmp = Xdf.copy()
    tmp[label_col] = y
    tmp = _coerce_numeric_features(tmp, label_col=label_col)
    tmp = _clean_rows(tmp, label_col=label_col)

    y_clean = tmp[label_col].to_numpy(dtype=np.int64)
    X_clean_df = tmp.drop(columns=[label_col])

    non_numeric = [c for c in X_clean_df.columns if not pd.api.types.is_numeric_dtype(X_clean_df[c])]
    if non_numeric:
        X_clean_df = X_clean_df.drop(columns=non_numeric)

    if cfg.sample_frac is not None:
        if not (0.0 < cfg.sample_frac <= 1.0):
            raise ValueError("sample_frac debe estar en (0, 1].")
        idx = np.random.default_rng(cfg.random_state).choice(
            len(X_clean_df),
            size=int(len(X_clean_df) * cfg.sample_frac),
            replace=False,
        )
        X_clean_df = X_clean_df.iloc[idx].reset_index(drop=True)
        y_clean = y_clean[idx]

    if cfg.use_canonical:
        result = map_to_canonical(X_clean_df, CICIDS2017_TO_CANON)
        X_clean = result.combined
        feature_names = result.feature_names
        print(
            f"[CICIDS2017] Canonical mapping: "
            f"{result.n_present}/{result.n_present + result.n_missing} features present"
        )
    else:
        feature_names = list(X_clean_df.columns)
        X_clean = X_clean_df.to_numpy(dtype=np.float32)

    return X_clean, y_clean, feature_names


def _load_and_process_csv_paths(
    csv_paths: List[Path],
    cfg: CICIDSLoadConfig,
    max_rows_per_csv: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Carga una lista de CSVs, aplica preprocesado y devuelve X, y y features."""
    if max_rows_per_csv is not None and max_rows_per_csv <= 0:
        raise ValueError("max_rows_per_csv debe ser > 0.")

    if max_rows_per_csv is None:
        df = _load_all_csvs(csv_paths, cfg)
    else:
        frames = [_load_csv_with_row_limit(path, cfg, row_limit=max_rows_per_csv) for path in csv_paths]
        df = pd.concat(frames, ignore_index=True)

    return _prepare_cicids_features(df, cfg)


def _resolve_exact_csv_names(csv_names: List[str], all_csvs: List[Path]) -> List[Path]:
    """Resuelve nombres exactos de CSV a rutas reales, preservando el orden de entrada."""
    csv_map = {path.name.lower(): path for path in all_csvs}
    resolved: List[Path] = []
    seen: set[str] = set()

    for csv_name in csv_names:
        key = csv_name.strip().lower()
        if key not in csv_map:
            raise ValueError(
                f"CSV '{csv_name}' no encontrado. Disponibles: {[path.name for path in all_csvs]}"
            )
        if key in seen:
            raise ValueError(f"CSV duplicado en la selección exacta: '{csv_name}'")
        resolved.append(csv_map[key])
        seen.add(key)

    return resolved


def load_cicids2017_binary(
    cfg: Optional[CICIDSLoadConfig] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[StandardScaler], List[str]]:
    """
    Carga CICIDS2017 desde directorio local y lo adapta al esquema canónico.

    Devuelve:
      X_train, y_train, X_test, y_test, scaler, feature_names

    - X: float32, shape (n_samples, NUM_OBSERVATION_FEATURES) cuando use_canonical=True
    - y: int64, 0=BENIGN, 1=ATTACK
    - scaler: StandardScaler ajustado en train (o None si no se escaló)
    - feature_names: lista de nombres (features canónicas + máscara de missingness)
    """
    cfg = cfg or CICIDSLoadConfig()

    local_dir = Path(cfg.local_dir)
    if not local_dir.exists():
        raise FileNotFoundError(
            f"Directorio de CICIDS2017 no encontrado: {local_dir}. "
            "Descarga el dataset y colócalo en datasets/CICIDS2017/."
        )

    csvs = _list_csv_files(local_dir)
    print(f"[CICIDS2017] Cargando {len(csvs)} archivos CSV desde {local_dir}")

    df = _load_all_csvs(csvs, cfg)
    X_clean, y_clean, feature_names = _prepare_cicids_features(df, cfg)

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


def load_cicids2017_csv_split(
    train_csvs: List[str],
    test_csvs: List[str],
    cfg: Optional[CICIDSLoadConfig] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[StandardScaler], List[str]]:
    """
    Carga CICIDS2017 separando por archivos CSV (train vs test).

    En lugar de hacer train_test_split aleatorio, usa CSVs específicos para
    cada split. Esto es más realista porque los CSVs representan distintos
    días/tipos de tráfico.

    Parameters
    ----------
    train_csvs : list of str
        Nombres (o substrings) de los CSVs para entrenamiento.
        Ejemplo: ["Monday", "Tuesday", "Wednesday"]
    test_csvs : list of str
        Nombres (o substrings) de los CSVs para test.
        Ejemplo: ["Thursday", "Friday"]
    cfg : CICIDSLoadConfig, optional
        Configuración de carga (se ignoran test_size y random_state del split).

    Returns
    -------
    X_train, y_train, X_test, y_test, scaler, feature_names
    """
    cfg = cfg or CICIDSLoadConfig()
    local_dir = Path(cfg.local_dir)
    if not local_dir.exists():
        raise FileNotFoundError(
            f"Directorio de CICIDS2017 no encontrado: {local_dir}. "
            "Descarga el dataset y colócalo en datasets/CICIDS2017/."
        )

    all_csvs = _list_csv_files(local_dir)

    def _match_csvs(patterns: List[str]) -> List[Path]:
        matched: List[Path] = []
        for p in all_csvs:
            name = p.name.lower()
            if any(pat.lower() in name for pat in patterns):
                matched.append(p)
        return matched

    train_paths = _match_csvs(train_csvs)
    test_paths = _match_csvs(test_csvs)

    if not train_paths:
        raise ValueError(f"Ningun CSV coincide con train_csvs={train_csvs}. Disponibles: {[p.name for p in all_csvs]}")
    if not test_paths:
        raise ValueError(f"Ningun CSV coincide con test_csvs={test_csvs}. Disponibles: {[p.name for p in all_csvs]}")

    print(f"[CSV-split] Train CSVs ({len(train_paths)}): {[p.name for p in train_paths]}")
    print(f"[CSV-split] Test  CSVs ({len(test_paths)}): {[p.name for p in test_paths]}")
    X_train, y_train, feature_names = _load_and_process_csv_paths(train_paths, cfg)
    X_test, y_test, _ = _load_and_process_csv_paths(test_paths, cfg)

    scaler: Optional[StandardScaler] = None
    if cfg.scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train).astype(np.float32)
        X_test = scaler.transform(X_test).astype(np.float32)

    print(f"[CSV-split] Train: {X_train.shape} (benign={int((y_train==0).sum())}, attack={int((y_train==1).sum())})")
    print(f"[CSV-split] Test:  {X_test.shape} (benign={int((y_test==0).sum())}, attack={int((y_test==1).sum())})")

    return X_train, y_train, X_test, y_test, scaler, feature_names


def load_cicids2017_exact_csv_split(
    train_csv_names: List[str],
    test_csv_names: List[str],
    cfg: Optional[CICIDSLoadConfig] = None,
    max_rows_per_csv: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[StandardScaler], List[str]]:
    """
    Carga CICIDS2017 separando por nombres exactos de archivo CSV.

    A diferencia de ``load_cicids2017_csv_split()``, aquí los nombres deben
    corresponder exactamente a archivos reales del dataset. Este modo está
    pensado para validaciones leave-one-CSV-out.

    Parameters
    ----------
    train_csv_names : list of str
        Nombres exactos de los CSVs para entrenamiento.
    test_csv_names : list of str
        Nombres exactos de los CSVs para test.
    cfg : CICIDSLoadConfig, optional
        Configuración de carga y preprocesado.
    max_rows_per_csv : int or None
        Límite opcional de filas por CSV. Si se indica, se aplica de forma
        independiente a cada archivo para evitar sesgo por orden de lectura.
    """
    cfg = cfg or CICIDSLoadConfig()
    local_dir = Path(cfg.local_dir)
    if not local_dir.exists():
        raise FileNotFoundError(
            f"Directorio de CICIDS2017 no encontrado: {local_dir}. "
            "Descarga el dataset y colócalo en datasets/CICIDS2017/."
        )

    all_csvs = list_cicids2017_csv_files(local_dir)
    train_paths = _resolve_exact_csv_names(train_csv_names, all_csvs)
    test_paths = _resolve_exact_csv_names(test_csv_names, all_csvs)

    overlap = {path.name for path in train_paths} & {path.name for path in test_paths}
    if overlap:
        raise ValueError(f"Train y test no pueden compartir CSVs exactos: {sorted(overlap)}")

    effective_cfg = replace(cfg, max_rows=None)

    print(f"[Exact-CSV-split] Train CSVs ({len(train_paths)}): {[p.name for p in train_paths]}")
    print(f"[Exact-CSV-split] Test  CSVs ({len(test_paths)}): {[p.name for p in test_paths]}")
    if max_rows_per_csv is not None:
        print(f"[Exact-CSV-split] Max rows per CSV: {max_rows_per_csv}")

    X_train, y_train, feature_names = _load_and_process_csv_paths(
        train_paths,
        effective_cfg,
        max_rows_per_csv=max_rows_per_csv,
    )
    X_test, y_test, _ = _load_and_process_csv_paths(
        test_paths,
        effective_cfg,
        max_rows_per_csv=max_rows_per_csv,
    )

    scaler: Optional[StandardScaler] = None
    if cfg.scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train).astype(np.float32)
        X_test = scaler.transform(X_test).astype(np.float32)

    print(
        f"[Exact-CSV-split] Train: {X_train.shape} "
        f"(benign={int((y_train==0).sum())}, attack={int((y_train==1).sum())})"
    )
    print(
        f"[Exact-CSV-split] Test:  {X_test.shape} "
        f"(benign={int((y_test==0).sum())}, attack={int((y_test==1).sum())})"
    )

    return X_train, y_train, X_test, y_test, scaler, feature_names


# ──────────────────────────────────────────────────────────────
# Unified split API
# ──────────────────────────────────────────────────────────────

DEFAULT_TRAIN_DAYS: List[str] = ["Monday", "Tuesday", "Wednesday"]
DEFAULT_TEST_DAYS: List[str] = ["Thursday", "Friday"]

# Preset defaults for max_rows when the user does not provide --max-rows
_PRESET_MAX_ROWS: Dict[str, Dict[str, Optional[int]]] = {
    "fast": {"random": 100_000, "day": 100_000},
    "full": {"random": None, "day": None},
}


def load_cicids2017_split(
    split_mode: str = "random",
    preset: str = "fast",
    seed: int = 42,
    max_rows: Optional[int] = None,
    train_days: Optional[List[str]] = None,
    test_days: Optional[List[str]] = None,
    scale: bool = True,
    use_canonical: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[StandardScaler], List[str], Dict[str, Any]]:
    """
    Unified CICIDS2017 loader with split-mode and preset support.

    Parameters
    ----------
    split_mode : ``"random"`` | ``"day"``
        ``"random"`` — stratified 80/20 train-test split (current default behaviour).
        ``"day"`` — group split by CSV / day.
    preset : ``"fast"`` | ``"full"``
        ``"fast"`` — lightweight defaults (capped rows) for quick iteration.
        ``"full"`` — load all available rows for the chosen split.
    seed : int
        Random seed for reproducibility.
    max_rows : int or None
        Explicit cap on rows loaded.  When *None*, the preset default is used.
    train_days : list[str] or None
        Day patterns for training (only used when *split_mode="day"*).
        Defaults to ``["Monday", "Tuesday", "Wednesday"]``.
    test_days : list[str] or None
        Day patterns for testing (only used when *split_mode="day"*).
        Defaults to ``["Thursday", "Friday"]``.
    scale : bool
        Whether to fit a ``StandardScaler`` on the train split.
    use_canonical : bool
        Whether to map features to the canonical schema (76 + 76 mask = 152).

    Returns
    -------
    X_train, y_train, X_test, y_test, scaler, feature_names, metadata
        ``metadata`` is a plain dict with counts, rates, and split info that
        can be serialised straight to JSON.
    """
    if split_mode not in ("random", "day"):
        raise ValueError(f"split_mode must be 'random' or 'day', got '{split_mode}'")
    if preset not in ("fast", "full"):
        raise ValueError(f"preset must be 'fast' or 'full', got '{preset}'")

    # Resolve effective max_rows
    effective_max_rows = max_rows if max_rows is not None else _PRESET_MAX_ROWS[preset][split_mode]

    cfg = CICIDSLoadConfig(
        max_rows=effective_max_rows,
        use_canonical=use_canonical,
        scale=scale,
        random_state=seed,
    )

    if split_mode == "random":
        X_train, y_train, X_test, y_test, scaler, feature_names = load_cicids2017_binary(cfg)
        day_info: Dict[str, Any] = {}
    else:
        effective_train_days = train_days or DEFAULT_TRAIN_DAYS
        effective_test_days = test_days or DEFAULT_TEST_DAYS
        X_train, y_train, X_test, y_test, scaler, feature_names = load_cicids2017_csv_split(
            train_csvs=effective_train_days,
            test_csvs=effective_test_days,
            cfg=cfg,
        )
        day_info = {
            "train_days": effective_train_days,
            "test_days": effective_test_days,
        }

    # Build metadata dict (JSON-safe)
    metadata: Dict[str, Any] = {
        "split_mode": split_mode,
        "preset": preset,
        "seed": seed,
        "max_rows": effective_max_rows,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "train_benign": int((y_train == 0).sum()),
        "train_attack": int((y_train == 1).sum()),
        "test_benign": int((y_test == 0).sum()),
        "test_attack": int((y_test == 1).sum()),
        "train_benign_rate": float((y_train == 0).mean()),
        "train_attack_rate": float((y_train == 1).mean()),
        "test_benign_rate": float((y_test == 0).mean()),
        "test_attack_rate": float((y_test == 1).mean()),
        **day_info,
    }

    return X_train, y_train, X_test, y_test, scaler, feature_names, metadata


if __name__ == "__main__":
    # Smoke test rápido con datos locales
    cfg = CICIDSLoadConfig(max_rows=50_000, sample_frac=None)
    X_train, y_train, X_test, y_test, scaler, feats = load_cicids2017_binary(cfg)
    print(f"CICIDS2017: X_train={X_train.shape}, y_train={y_train.shape}")
    print(f"CICIDS2017: X_test ={X_test.shape}, y_test ={y_test.shape}")
    print(f"Features ({len(feats)}): {feats[:5]} ... {feats[-5:]}")
    benign_rate = (y_train == 0).mean()
    attack_rate = (y_train == 1).mean()
    print(f"Train benign rate: {benign_rate:.4f}, attack rate: {attack_rate:.4f}")
