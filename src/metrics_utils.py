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
from typing import Dict, Literal, Optional


UndefinedMetricPolicy = Literal["legacy_zero", "null"]
MetricValue = float | int | None


def _safe_div(num: float, den: float, undefined_value: float | None) -> float | None:
    return float(num / den) if den else undefined_value


def confusion_to_metrics(
    tn: int,
    fp: int,
    fn: int,
    tp: int,
    reward_config: Optional[Dict[str, float]] = None,
    undefined_metric_policy: UndefinedMetricPolicy = "legacy_zero",
) -> Dict[str, MetricValue]:
    """Compute the full binary metric set from confusion-matrix counts.

    Parameters
    ----------
    tn, fp, fn, tp : int
        Confusion-matrix counts (true-neg, false-pos, false-neg, true-pos), i.e.
        ``confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()``.
    reward_config : dict, optional
        If given (keys ``tp``/``fp``/``fn``/``omission``), also reports
        ``reward_total`` and ``reward_per_sample`` under the env's cost matrix.
    undefined_metric_policy : {``"legacy_zero"``, ``"null"``}
        ``"legacy_zero"`` preserves existing consumers by returning ``0.0``
        for zero-denominator metrics. ``"null"`` returns ``None`` so new
        campaign JSON/CSV consumers can preserve undefined metrics.
    """
    if undefined_metric_policy not in ("legacy_zero", "null"):
        raise ValueError(
            "undefined_metric_policy must be 'legacy_zero' or 'null', "
            f"got '{undefined_metric_policy}'"
        )

    undefined_value = 0.0 if undefined_metric_policy == "legacy_zero" else None
    tn, fp, fn, tp = int(tn), int(fp), int(fn), int(tp)
    total = tn + fp + fn + tp

    precision_attack = _safe_div(tp, tp + fp, undefined_value)
    recall_attack = _safe_div(tp, tp + fn, undefined_value)
    f1_attack = _safe_div(2 * tp, 2 * tp + fp + fn, undefined_value)

    precision_benign = _safe_div(tn, tn + fn, undefined_value)
    recall_benign = _safe_div(tn, tn + fp, undefined_value)
    f1_benign = _safe_div(2 * tn, 2 * tn + fp + fn, undefined_value)

    if recall_attack is None or recall_benign is None:
        balanced_accuracy = None
    else:
        balanced_accuracy = (recall_attack + recall_benign) / 2.0

    mcc_den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = float((tp * tn - fp * fn) / mcc_den) if mcc_den > 0 else undefined_value

    metrics: Dict[str, MetricValue] = {
        "accuracy": _safe_div(tp + tn, total, undefined_value),
        "balanced_accuracy": balanced_accuracy,
        "mcc": mcc,
        "precision_attack": precision_attack,
        "recall_attack": recall_attack,
        "f1_attack": f1_attack,
        "precision_benign": precision_benign,
        "recall_benign": recall_benign,
        "f1_benign": f1_benign,
        "specificity": recall_benign,
        "fpr": _safe_div(fp, fp + tn, undefined_value),
        "fnr": _safe_div(fn, fn + tp, undefined_value),
        "block_rate": _safe_div(tp + fp, total, undefined_value),
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
        metrics["reward_per_sample"] = _safe_div(reward_total, total, undefined_value)

    return metrics
