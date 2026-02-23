"""
scaling_utils.py — Clipping utilities for outlier handling in RL inference pipeline.

Provides two complementary clipping strategies:
  1. Percentile clipping (applied to raw/un-scaled features before scaling).
  2. Z-score clipping (applied to scaled features after StandardScaler.transform).

These are used in Phase 2 inference to prevent extreme values from real traffic
(e.g. TCP flag counts with |z| up to 89) from collapsing Q-value estimates.
"""

from __future__ import annotations

import numpy as np


def apply_percentile_clipping(
    X: np.ndarray,
    p_low: np.ndarray,
    p_high: np.ndarray,
) -> np.ndarray:
    """
    Clamp each feature to its training percentile range [p_low, p_high].

    Applied to raw (un-scaled) features **before** StandardScaler.transform().
    This prevents extreme outliers in real traffic from distorting the scaler
    output.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix, shape (n_samples, n_features).
    p_low : np.ndarray
        Per-feature lower clipping bound, shape (n_features,).
        Typically the 0.5th percentile computed on training data.
    p_high : np.ndarray
        Per-feature upper clipping bound, shape (n_features,).
        Typically the 99.5th percentile computed on training data.

    Returns
    -------
    np.ndarray
        Clipped feature matrix, same shape as X (new array, not in-place).
    """
    return np.clip(X, p_low, p_high).astype(X.dtype)


def apply_z_clipping(
    X_scaled: np.ndarray,
    max_z: float,
) -> np.ndarray:
    """
    Clamp scaled features to [-max_z, +max_z].

    Applied to scaled features **after** StandardScaler.transform() to prevent
    extreme z-scores (e.g. |z|=89 in TCP flag counts) from pushing Q-values
    into degenerate regimes.

    Parameters
    ----------
    X_scaled : np.ndarray
        Scaled feature matrix, shape (n_samples, n_features).
    max_z : float
        Maximum allowed absolute z-score. Typical value: 10.0.

    Returns
    -------
    np.ndarray
        Clipped scaled matrix, same shape as X_scaled (new array, not in-place).
    """
    return np.clip(X_scaled, -max_z, max_z).astype(X_scaled.dtype)
