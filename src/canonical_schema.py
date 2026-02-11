"""
canonical_schema.py — Definición formal del esquema canónico de features (FEATURES_CANON).

Todas las features aquí listadas:
  1. Existen en CICIDS2017 (dataset principal moderno)
  2. Son extraíbles de tráfico real/PCAP mediante flow extractors (CICFlowMeter, Zeek, etc.)
  3. NO incluyen IPs, timestamps, Flow IDs, ni puertos específicos (sin data leakage)
  4. Son numéricas y representan estadísticas de flujos de red
  5. Son estables y robustas (no dependen de peculiaridades del dataset)

El vector de observación final del agente será:
    obs = [x_1, x_2, ..., x_d, m_1, m_2, ..., m_d]

donde:
    x_i = valor de la feature i (imputado si no existe en el dataset de origen)
    m_i = 1 si la feature estaba presente en el dataset, 0 si fue imputada
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# FEATURES_CANON: lista ordenada de features canónicas
# ---------------------------------------------------------------------------
# Basada en las columnas de CICIDS2017 (CICFlowMeter output).
# Nombres normalizados a lower_snake_case.
# Criterio de inclusión:
#   - Estadísticas de flujo (duración, paquetes, bytes)
#   - Tasas y velocidades (pkt/s, bytes/s)
#   - Estadísticas de tamaños de paquetes (mean, std, min, max)
#   - Estadísticas de inter-arrival times (IAT)
#   - Flags TCP (SYN, ACK, FIN, RST, PSH, URG, ECE, CWE)
#   - Ventanas TCP
#   - Estadísticas derivadas (down/up ratio, avg pkt size, etc.)
# Criterio de exclusión:
#   - IPs, timestamps, Flow IDs, puertos específicos → data leakage
#   - Columnas con valores mayoritariamente constantes o redundantes

FEATURES_CANON: List[str] = [
    # ── Estadísticas generales del flujo ──
    "flow_duration",
    "total_fwd_packets",
    "total_bwd_packets",
    "total_length_of_fwd_packets",
    "total_length_of_bwd_packets",
    # ── Estadísticas de tamaño de paquetes (forward) ──
    "fwd_packet_length_max",
    "fwd_packet_length_min",
    "fwd_packet_length_mean",
    "fwd_packet_length_std",
    # ── Estadísticas de tamaño de paquetes (backward) ──
    "bwd_packet_length_max",
    "bwd_packet_length_min",
    "bwd_packet_length_mean",
    "bwd_packet_length_std",
    # ── Tasas de flujo ──
    "flow_bytes_per_s",
    "flow_packets_per_s",
    # ── Inter-arrival times del flujo ──
    "flow_iat_mean",
    "flow_iat_std",
    "flow_iat_max",
    "flow_iat_min",
    # ── Inter-arrival times forward ──
    "fwd_iat_total",
    "fwd_iat_mean",
    "fwd_iat_std",
    "fwd_iat_max",
    "fwd_iat_min",
    # ── Inter-arrival times backward ──
    "bwd_iat_total",
    "bwd_iat_mean",
    "bwd_iat_std",
    "bwd_iat_max",
    "bwd_iat_min",
    # ── Flags TCP ──
    "fwd_psh_flags",
    "bwd_psh_flags",
    "fwd_urg_flags",
    "bwd_urg_flags",
    "fin_flag_count",
    "syn_flag_count",
    "rst_flag_count",
    "psh_flag_count",
    "ack_flag_count",
    "urg_flag_count",
    "cwe_flag_count",
    "ece_flag_count",
    # ── Cabecera / header length ──
    "fwd_header_length",
    "bwd_header_length",
    # ── Paquetes por segundo (forward / backward) ──
    "fwd_packets_per_s",
    "bwd_packets_per_s",
    # ── Estadísticas de longitud de paquete (global) ──
    "min_packet_length",
    "max_packet_length",
    "packet_length_mean",
    "packet_length_std",
    "packet_length_variance",
    # ── Ratios y estadísticas derivadas ──
    "down_up_ratio",
    "average_packet_size",
    "avg_fwd_segment_size",
    "avg_bwd_segment_size",
    # ── Bulk statistics (forward) ──
    "fwd_avg_bytes_per_bulk",
    "fwd_avg_packets_per_bulk",
    "fwd_avg_bulk_rate",
    # ── Bulk statistics (backward) ──
    "bwd_avg_bytes_per_bulk",
    "bwd_avg_packets_per_bulk",
    "bwd_avg_bulk_rate",
    # ── Sub-flow statistics ──
    "subflow_fwd_packets",
    "subflow_fwd_bytes",
    "subflow_bwd_packets",
    "subflow_bwd_bytes",
    # ── Ventana TCP ──
    "init_win_bytes_forward",
    "init_win_bytes_backward",
    # ── Datos activos ──
    "act_data_pkt_fwd",
    "min_seg_size_forward",
    # ── Tiempos activos / idle ──
    "active_mean",
    "active_std",
    "active_max",
    "active_min",
    "idle_mean",
    "idle_std",
    "idle_max",
    "idle_min",
]

NUM_CANONICAL_FEATURES: int = len(FEATURES_CANON)
"""Número de features canónicas (sin máscara de missingness)."""

NUM_OBSERVATION_FEATURES: int = NUM_CANONICAL_FEATURES * 2
"""Dimensión del vector de observación completo: features + máscara de missingness."""


# ---------------------------------------------------------------------------
# Mapeo CICIDS2017 → esquema canónico
# ---------------------------------------------------------------------------
# Las columnas de CICIDS2017 usan espacios y Title Case.
# Aquí mapeamos cada columna del CSV al nombre canónico.

CICIDS2017_TO_CANON: Dict[str, str] = {
    "Flow Duration": "flow_duration",
    "Total Fwd Packets": "total_fwd_packets",
    "Total Backward Packets": "total_bwd_packets",
    "Total Length of Fwd Packets": "total_length_of_fwd_packets",
    "Total Length of Bwd Packets": "total_length_of_bwd_packets",
    "Fwd Packet Length Max": "fwd_packet_length_max",
    "Fwd Packet Length Min": "fwd_packet_length_min",
    "Fwd Packet Length Mean": "fwd_packet_length_mean",
    "Fwd Packet Length Std": "fwd_packet_length_std",
    "Bwd Packet Length Max": "bwd_packet_length_max",
    "Bwd Packet Length Min": "bwd_packet_length_min",
    "Bwd Packet Length Mean": "bwd_packet_length_mean",
    "Bwd Packet Length Std": "bwd_packet_length_std",
    "Flow Bytes/s": "flow_bytes_per_s",
    "Flow Packets/s": "flow_packets_per_s",
    "Flow IAT Mean": "flow_iat_mean",
    "Flow IAT Std": "flow_iat_std",
    "Flow IAT Max": "flow_iat_max",
    "Flow IAT Min": "flow_iat_min",
    "Fwd IAT Total": "fwd_iat_total",
    "Fwd IAT Mean": "fwd_iat_mean",
    "Fwd IAT Std": "fwd_iat_std",
    "Fwd IAT Max": "fwd_iat_max",
    "Fwd IAT Min": "fwd_iat_min",
    "Bwd IAT Total": "bwd_iat_total",
    "Bwd IAT Mean": "bwd_iat_mean",
    "Bwd IAT Std": "bwd_iat_std",
    "Bwd IAT Max": "bwd_iat_max",
    "Bwd IAT Min": "bwd_iat_min",
    "Fwd PSH Flags": "fwd_psh_flags",
    "Bwd PSH Flags": "bwd_psh_flags",
    "Fwd URG Flags": "fwd_urg_flags",
    "Bwd URG Flags": "bwd_urg_flags",
    "FIN Flag Count": "fin_flag_count",
    "SYN Flag Count": "syn_flag_count",
    "RST Flag Count": "rst_flag_count",
    "PSH Flag Count": "psh_flag_count",
    "ACK Flag Count": "ack_flag_count",
    "URG Flag Count": "urg_flag_count",
    "CWE Flag Count": "cwe_flag_count",
    "ECE Flag Count": "ece_flag_count",
    "Fwd Header Length": "fwd_header_length",
    "Bwd Header Length": "bwd_header_length",
    "Fwd Packets/s": "fwd_packets_per_s",
    "Bwd Packets/s": "bwd_packets_per_s",
    "Min Packet Length": "min_packet_length",
    "Max Packet Length": "max_packet_length",
    "Packet Length Mean": "packet_length_mean",
    "Packet Length Std": "packet_length_std",
    "Packet Length Variance": "packet_length_variance",
    "Down/Up Ratio": "down_up_ratio",
    "Average Packet Size": "average_packet_size",
    "Avg Fwd Segment Size": "avg_fwd_segment_size",
    "Avg Bwd Segment Size": "avg_bwd_segment_size",
    "Fwd Avg Bytes/Bulk": "fwd_avg_bytes_per_bulk",
    "Fwd Avg Packets/Bulk": "fwd_avg_packets_per_bulk",
    "Fwd Avg Bulk Rate": "fwd_avg_bulk_rate",
    "Bwd Avg Bytes/Bulk": "bwd_avg_bytes_per_bulk",
    "Bwd Avg Packets/Bulk": "bwd_avg_packets_per_bulk",
    "Bwd Avg Bulk Rate": "bwd_avg_bulk_rate",
    "Subflow Fwd Packets": "subflow_fwd_packets",
    "Subflow Fwd Bytes": "subflow_fwd_bytes",
    "Subflow Bwd Packets": "subflow_bwd_packets",
    "Subflow Bwd Bytes": "subflow_bwd_bytes",
    "Init_Win_bytes_forward": "init_win_bytes_forward",
    "Init_Win_bytes_backward": "init_win_bytes_backward",
    "act_data_pkt_fwd": "act_data_pkt_fwd",
    "min_seg_size_forward": "min_seg_size_forward",
    "Active Mean": "active_mean",
    "Active Std": "active_std",
    "Active Max": "active_max",
    "Active Min": "active_min",
    "Idle Mean": "idle_mean",
    "Idle Std": "idle_std",
    "Idle Max": "idle_max",
    "Idle Min": "idle_min",
}


# ---------------------------------------------------------------------------
# Mapeo NSL-KDD → esquema canónico (benchmark histórico, mapping parcial)
# ---------------------------------------------------------------------------
# NSL-KDD tiene features muy diferentes (basadas en conexiones antiguas, no flows).
# Solo algunas features son parcialmente mapeables al esquema canónico.
# La mayoría de features canónicas NO tienen equivalente en NSL-KDD,
# por lo que la máscara de missingness será mayoritariamente 0.

NSL_KDD_TO_CANON: Dict[str, str] = {
    "duration": "flow_duration",
    "src_bytes": "total_length_of_fwd_packets",
    "dst_bytes": "total_length_of_bwd_packets",
}


# ---------------------------------------------------------------------------
# Estrategia de imputación por defecto
# ---------------------------------------------------------------------------
# Para features ausentes (missingness), usamos estas estrategias:
#   - Contadores y sumatorios: imputar con 0
#   - Estadísticas (mean, std, min, max): imputar con 0
#   - Tasas/ratios: imputar con 0
# Se puede personalizar por feature si es necesario.

DEFAULT_IMPUTATION_VALUE: float = 0.0


# ---------------------------------------------------------------------------
# Funciones auxiliares
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CanonicalResult:
    """Resultado de mapear un DataFrame al esquema canónico."""
    X: np.ndarray                  # (n_samples, NUM_CANONICAL_FEATURES) — valores
    mask: np.ndarray               # (n_samples, NUM_CANONICAL_FEATURES) — 1=presente, 0=imputado
    combined: np.ndarray           # (n_samples, NUM_OBSERVATION_FEATURES) — [X | mask]
    feature_names: List[str]       # nombres de features + nombres de máscara
    n_present: int                 # número de features canónicas presentes en el dataset
    n_missing: int                 # número de features canónicas ausentes


def map_to_canonical(
    df: pd.DataFrame,
    column_mapping: Dict[str, str],
    imputation_value: float = DEFAULT_IMPUTATION_VALUE,
) -> CanonicalResult:
    """
    Mapea un DataFrame al esquema canónico de features.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con las columnas originales del dataset (sin etiqueta).
    column_mapping : dict
        Mapping de nombre de columna del dataset → nombre canónico.
        Ejemplo: ``{"Flow Duration": "flow_duration", ...}``
    imputation_value : float
        Valor de imputación para features ausentes. Default 0.0.

    Returns
    -------
    CanonicalResult
        Objeto con X, mask, combined, feature_names, n_present, n_missing.
    """
    n_samples = len(df)

    # Normalizar nombres de columnas en el DataFrame (strip espacios)
    df_cols = {str(c).strip(): c for c in df.columns}

    # Invertir mapping: canónico → lista de columnas originales que lo producen
    canon_to_source: Dict[str, Optional[str]] = {}
    for src_name, canon_name in column_mapping.items():
        src_stripped = src_name.strip()
        if src_stripped in df_cols:
            canon_to_source[canon_name] = df_cols[src_stripped]

    # Construir arrays
    X = np.full((n_samples, NUM_CANONICAL_FEATURES), imputation_value, dtype=np.float32)
    mask = np.zeros((n_samples, NUM_CANONICAL_FEATURES), dtype=np.float32)

    n_present = 0
    for i, canon_name in enumerate(FEATURES_CANON):
        source_col = canon_to_source.get(canon_name)
        if source_col is not None:
            values = pd.to_numeric(df[source_col], errors="coerce").to_numpy(dtype=np.float32).copy()
            # Reemplazar NaN/Inf por valor de imputación
            bad = ~np.isfinite(values)
            values[bad] = imputation_value
            X[:, i] = values
            # Máscara: 1 donde el valor era válido, 0 donde era NaN/Inf original
            mask[:, i] = (~bad).astype(np.float32)
            n_present += 1
        # else: X[:,i] ya tiene imputation_value, mask[:,i] ya es 0

    n_missing = NUM_CANONICAL_FEATURES - n_present

    combined = np.hstack([X, mask])

    feature_names = list(FEATURES_CANON) + [f"m_{f}" for f in FEATURES_CANON]

    return CanonicalResult(
        X=X,
        mask=mask,
        combined=combined,
        feature_names=feature_names,
        n_present=n_present,
        n_missing=n_missing,
    )


def get_canonical_feature_names() -> List[str]:
    """Devuelve la lista de nombres de features canónicas (sin máscara)."""
    return list(FEATURES_CANON)


def get_observation_feature_names() -> List[str]:
    """Devuelve la lista completa de nombres: features + máscara de missingness."""
    return list(FEATURES_CANON) + [f"m_{f}" for f in FEATURES_CANON]
