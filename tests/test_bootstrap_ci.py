"""Tests for scripts/bootstrap_ci.py (Task A4).

Guards two things:
  1. The exact confusion counts are recovered from the committed MAIN artifacts
     and self-validate against every published metric.
  2. The multinomial-on-cells bootstrap is statistically equivalent to a true
     row-level (with-replacement) bootstrap — the methodological claim the
     script relies on to avoid re-running the model.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from scripts.bootstrap_ci import (  # noqa: E402
    bootstrap_metrics,
    recover_confusion_counts,
    self_validate,
)
from src.metrics_utils import confusion_to_metrics  # noqa: E402

_MAIN_RUN = _REPO / "runs" / "cicids2017" / "MAIN_qrdqn_cicids2017_canonical_full_random_20260609_193655"


def test_recover_counts_from_main_artifacts():
    """The committed MAIN metrics.json + config.json yield the exact cells."""
    if not (_MAIN_RUN / "metrics.json").exists():
        pytest.skip("MAIN run artifacts not present")
    metrics = json.loads((_MAIN_RUN / "metrics.json").read_text(encoding="utf-8"))
    config = json.loads((_MAIN_RUN / "config.json").read_text(encoding="utf-8"))
    tn, fp, fn, tp = recover_confusion_counts(metrics, config["split_metadata"])

    # Exact headline cells (cross-validated to 16 digits against the metrics).
    assert (tn, fp, fn, tp) == (451631, 2989, 518, 111011)
    assert tn + fp + fn + tp == int(config["split_metadata"]["n_test"]) == 566149

    # Self-validation must pass: re-derived metrics reproduce the published ones.
    self_validate(tn, fp, fn, tp, metrics, config.get("reward_config"))


def test_self_validate_rejects_wrong_counts():
    published = {"accuracy": 0.99, "recall_attack": 0.99, "recall_benign": 0.99}
    with pytest.raises(AssertionError):
        self_validate(10, 10, 10, 10, published, None)  # nowhere near 0.99


def test_point_estimate_matches_and_ci_brackets():
    tn, fp, fn, tp = 451631, 2989, 518, 111011
    rng = np.random.default_rng(7)
    ci = bootstrap_metrics(tn, fp, fn, tp, n_boot=3000, rng=rng, reward_config=None)
    point = confusion_to_metrics(tn, fp, fn, tp)
    for key, stats in ci.items():
        # Reported point equals the deterministic confusion-derived value.
        assert abs(stats["point"] - round(float(point[key]), 6)) < 1e-9
        # The 95% CI brackets the point estimate.
        assert stats["ci95_low"] <= stats["point"] <= stats["ci95_high"]
        # boot_mean is close to the point estimate for a well-behaved metric.
        assert abs(stats["boot_mean"] - stats["point"]) < 0.01


def test_reproducible_with_fixed_seed():
    counts = (451631, 2989, 518, 111011)
    a = bootstrap_metrics(*counts, n_boot=2000, rng=np.random.default_rng(123), reward_config=None)
    b = bootstrap_metrics(*counts, n_boot=2000, rng=np.random.default_rng(123), reward_config=None)
    assert a == b


def _row_level_recall_ci_stratified(fn, tp, n_boot, rng):
    """Reference: stratified row-level bootstrap of recall_attack.

    recall_attack depends only on the attack class (tp, fn); the stratified
    bootstrap holds N+ = fn+tp fixed and resamples attack rows with replacement.
    """
    attack = np.repeat([0, 1], [fn, tp])  # 0 = missed (fn), 1 = caught (tp)
    na = attack.shape[0]
    out = np.empty(n_boot)
    for b in range(n_boot):
        s = attack[rng.integers(0, na, na)]
        t_p = int(s.sum())
        f_n = na - t_p
        out[b] = t_p / (t_p + f_n) if (t_p + f_n) > 0 else 0.0
    return np.percentile(out, [2.5, 97.5])


def test_stratified_matches_row_level_bootstrap():
    """The default (stratified) bootstrap must agree with a stratified row-level bootstrap."""
    tn, fp, fn, tp = 800, 50, 100, 50  # small N+=150 so the recall CI is wide -> a real test
    n_boot = 20000
    mn = bootstrap_metrics(tn, fp, fn, tp, n_boot=n_boot,
                           rng=np.random.default_rng(2024), reward_config=None)  # stratified default
    lo_mn, hi_mn = mn["recall_attack"]["ci95_low"], mn["recall_attack"]["ci95_high"]
    lo_row, hi_row = _row_level_recall_ci_stratified(fn, tp, n_boot, np.random.default_rng(99))
    # Two Monte Carlo estimates of the same bootstrap distribution: agree closely.
    assert abs(lo_mn - lo_row) < 0.02
    assert abs(hi_mn - hi_row) < 0.02


def test_unstratified_branch_runs_and_brackets():
    """The optional unconditional multinomial bootstrap still produces valid CIs."""
    tn, fp, fn, tp = 451631, 2989, 518, 111011
    ci = bootstrap_metrics(tn, fp, fn, tp, n_boot=2000,
                           rng=np.random.default_rng(5), reward_config=None, stratified=False)
    for stats in ci.values():
        assert stats["ci95_low"] <= stats["point"] <= stats["ci95_high"]
