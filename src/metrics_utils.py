"""metrics_utils.py — single source of truth for binary PERMIT/BLOCK metrics.

Training evaluation, Phase-2 inference, and the supervised baselines all compute
their metrics from a confusion matrix via :func:`confusion_to_metrics`, so the
numbers are defined identically everywhere.

Convention: ``0 = PERMIT / benign / negative``, ``1 = BLOCK / attack / positive``.
Confusion-matrix order matches ``sklearn.metrics.confusion_matrix(..., labels=[0, 1]).ravel()``
which yields ``(tn, fp, fn, tp)``.
"""
from __future__ import annotations

import math
from typing import Dict, Optional


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def confusion_to_metrics(
    tn: int,
    fp: int,
    fn: int,
    tp: int,
    reward_config: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """Compute the full binary metric set from confusion-matrix counts.

    Parameters
    ----------
    tn, fp, fn, tp : int
        Confusion-matrix counts (true-neg, false-pos, false-neg, true-pos), i.e.
        ``confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()``.
    reward_config : dict, optional
        If given (keys ``tp``/``fp``/``fn``/``omission``), also reports
        ``reward_total`` and ``reward_per_sample`` under the env's cost matrix.
    """
    tn, fp, fn, tp = int(tn), int(fp), int(fn), int(tp)
    total = tn + fp + fn + tp

    precision_attack = _safe_div(tp, tp + fp)
    recall_attack = _safe_div(tp, tp + fn)
    f1_attack = _safe_div(2 * precision_attack * recall_attack, precision_attack + recall_attack)

    precision_benign = _safe_div(tn, tn + fn)
    recall_benign = _safe_div(tn, tn + fp)
    f1_benign = _safe_div(2 * precision_benign * recall_benign, precision_benign + recall_benign)

    mcc_den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = float((tp * tn - fp * fn) / mcc_den) if mcc_den > 0 else 0.0

    metrics: Dict[str, float] = {
        "accuracy": _safe_div(tp + tn, total),
        "balanced_accuracy": (recall_attack + recall_benign) / 2.0,
        "mcc": mcc,
        "precision_attack": precision_attack,
        "recall_attack": recall_attack,
        "f1_attack": f1_attack,
        "precision_benign": precision_benign,
        "recall_benign": recall_benign,
        "f1_benign": f1_benign,
        "specificity": recall_benign,
        "fpr": _safe_div(fp, fp + tn),
        "fnr": _safe_div(fn, fn + tp),
        "block_rate": _safe_div(tp + fp, total),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }

    if reward_config is not None:
        reward_total = (
            tp * reward_config.get("tp", 0.0)
            + fp * reward_config.get("fp", 0.0)
            + fn * reward_config.get("fn", 0.0)
            + tn * reward_config.get("omission", 0.0)
        )
        metrics["reward_total"] = float(reward_total)
        metrics["reward_per_sample"] = _safe_div(reward_total, total)

    return metrics
