"""
analyze_duplicates.py — Phase-0 / Task A1 (read-only analysis).

Quantifies, WITHOUT any training, how much the headline random-split metric is
inflated by duplicate CICIDS2017 flows:

  1. Exact-duplicate rate over the full cleaned canonical dataset
     (feature-only, and feature+label).
  2. Cross-split leakage: how many seed-42 TEST rows are exact duplicates of a
     TRAIN row -- the channel that inflates the random-split test metrics.
  3. Per-class breakdown (benign vs attack).

It uses the SAME loader / canonical schema / seed-42 stratified split as the
MAIN run (``src/load_cicids2017.load_cicids2017_binary`` with ``scale=False``).
No model is loaded or trained. Results are printed and written to
``runs/validation/duplicate_analysis_seed42.json``.

Exact-duplicate detection uses a byte-level void-view of each row, so matches
are exact (no hashing / no collision risk).

Run:
    python scripts/analyze_duplicates.py
    (requires the CICIDS2017 LFS CSVs in datasets/CICIDS2017/; `git lfs pull`)
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from load_cicids2017 import CICIDSLoadConfig, load_cicids2017_binary  # noqa: E402


def _void_rows(X: np.ndarray) -> np.ndarray:
    """View a 2-D float32 array as a 1-D array of byte-exact per-row records."""
    Xc = np.ascontiguousarray(X, dtype=np.float32)
    return Xc.view(np.dtype((np.void, Xc.shape[1] * Xc.itemsize))).ravel()


def _dup_stats(void_rows: np.ndarray) -> dict:
    n_total = int(void_rows.shape[0])
    n_unique = int(np.unique(void_rows).shape[0])
    n_dup = n_total - n_unique
    return {
        "n_rows": n_total,
        "n_unique": n_unique,
        "n_duplicate_rows": n_dup,
        "duplicate_pct": round(100.0 * n_dup / n_total, 4) if n_total else 0.0,
    }


def _pct(n: int, d: int) -> float:
    return round(100.0 * n / d, 4) if d else 0.0


def main() -> None:
    print("=" * 72)
    print("  A1 — CICIDS2017 duplicate / cross-split leakage analysis (read-only)")
    print("=" * 72)

    # Same config as the MAIN run's loader path: seed 42, 80/20 stratified,
    # canonical schema, NO scaling (scaling is monotone per feature and does not
    # change exact-duplicate structure; raw values keep the comparison clean).
    cfg = CICIDSLoadConfig(max_rows=None, use_canonical=True, scale=False)
    X_train, y_train, X_test, y_test, _scaler, feature_names = load_cicids2017_binary(cfg)

    n_features = int(X_train.shape[1])
    y_all = np.concatenate([y_train, y_test])
    print(f"\nLoaded: train={X_train.shape}, test={X_test.shape}, features={n_features}")
    print(f"seed={cfg.random_state}, test_size={cfg.test_size}, canonical={cfg.use_canonical}")

    # ---- byte-exact row records -----------------------------------------
    tr_feat = _void_rows(X_train)
    te_feat = _void_rows(X_test)
    all_feat = np.concatenate([tr_feat, te_feat])

    Xtr_l = np.hstack([X_train, y_train.reshape(-1, 1).astype(np.float32)])
    Xte_l = np.hstack([X_test, y_test.reshape(-1, 1).astype(np.float32)])
    tr_fl = _void_rows(Xtr_l)
    te_fl = _void_rows(Xte_l)
    all_fl = np.concatenate([tr_fl, te_fl])

    overall_feat = _dup_stats(all_feat)
    overall_fl = _dup_stats(all_fl)

    # ---- cross-split leakage: TEST rows that exactly match a TRAIN row ----
    tr_feat_u = np.unique(tr_feat)
    test_in_train_feat = np.isin(te_feat, tr_feat_u)
    tr_fl_u = np.unique(tr_fl)
    test_in_train_fl = np.isin(te_fl, tr_fl_u)

    n_test = int(len(y_test))
    attack_mask = (y_test == 1)
    benign_mask = (y_test == 0)
    n_leak_feat = int(test_in_train_feat.sum())
    n_leak_fl = int(test_in_train_fl.sum())
    leak_feat_attack = int((test_in_train_feat & attack_mask).sum())
    leak_feat_benign = int((test_in_train_feat & benign_mask).sum())

    # ---- per-class duplicate rate within the full set --------------------
    dup_attack = _dup_stats(all_feat[(y_all == 1)])
    dup_benign = _dup_stats(all_feat[(y_all == 0)])

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "task": "A1 - duplicate / cross-split leakage",
        "loader": "src/load_cicids2017.load_cicids2017_binary",
        "config": {
            "seed": cfg.random_state,
            "test_size": cfg.test_size,
            "use_canonical": cfg.use_canonical,
            "scale": cfg.scale,
            "n_features": n_features,
        },
        "counts": {"n_train": int(len(y_train)), "n_test": n_test, "n_total": int(len(y_all))},
        "overall_duplicates_feature_only": overall_feat,
        "overall_duplicates_feature_plus_label": overall_fl,
        "cross_split_leakage_feature_only": {
            "n_test_rows_also_in_train": n_leak_feat,
            "pct_of_test": _pct(n_leak_feat, n_test),
            "attack_test_rows_in_train": leak_feat_attack,
            "attack_pct_of_test_attacks": _pct(leak_feat_attack, int(attack_mask.sum())),
            "benign_test_rows_in_train": leak_feat_benign,
            "benign_pct_of_test_benigns": _pct(leak_feat_benign, int(benign_mask.sum())),
        },
        "cross_split_leakage_feature_plus_label": {
            "n_test_rows_also_in_train": n_leak_fl,
            "pct_of_test": _pct(n_leak_fl, n_test),
        },
        "per_class_duplicate_rate_full_set_feature_only": {"attack": dup_attack, "benign": dup_benign},
    }

    csl = summary["cross_split_leakage_feature_only"]
    print("\n--- Overall exact-duplicate rate (full cleaned canonical set) ---")
    print(f"  feature-only    : {overall_feat['n_duplicate_rows']:>10,} / {overall_feat['n_rows']:,}  ({overall_feat['duplicate_pct']}%)")
    print(f"  feature + label : {overall_fl['n_duplicate_rows']:>10,} / {overall_fl['n_rows']:,}  ({overall_fl['duplicate_pct']}%)")

    print("\n--- Cross-split leakage (seed-42 random 80/20 split) ---")
    print("  TEST rows that are an exact duplicate of a TRAIN row (feature-only):")
    print(f"    {n_leak_feat:>10,} / {n_test:,}  ({csl['pct_of_test']}% of the test set)")
    print(f"    attack: {leak_feat_attack:,} ({csl['attack_pct_of_test_attacks']}% of test attacks)")
    print(f"    benign: {leak_feat_benign:,} ({csl['benign_pct_of_test_benigns']}% of test benigns)")
    print(f"  feature + label exact match: {n_leak_fl:,} / {n_test:,}  ({summary['cross_split_leakage_feature_plus_label']['pct_of_test']}%)")

    print("\n--- Per-class duplicate rate (full set, feature-only) ---")
    print(f"  attack rows: {dup_attack['n_duplicate_rows']:,} dup / {dup_attack['n_rows']:,}  ({dup_attack['duplicate_pct']}%)")
    print(f"  benign rows: {dup_benign['n_duplicate_rows']:,} dup / {dup_benign['n_rows']:,}  ({dup_benign['duplicate_pct']}%)")

    out_dir = _REPO / "runs" / "validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "duplicate_analysis_seed42.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n[output] {out_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
