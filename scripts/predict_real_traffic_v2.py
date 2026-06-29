"""
predict_real_traffic_v2.py — Robust Phase 2 offline inference pipeline.

Loads a trained RL model and a persisted scaler to classify real network flows
extracted from a PCAP/flow-extractor tool.

Key improvements over v1:
  - Loads scaler from --scaler artifact (no dataset reconstruction at import time).
  - Supports percentile clipping on raw features (--percentiles).
  - Supports z-score clipping on scaled features (--clip-z).
  - Provides z-score diagnostics to detect distribution shift.
  - Full CLI via argparse.

Usage:
    python scripts/predict_real_traffic_v2.py \\
        --flows pcaps/flows.csv \\
        --run-dir runs/cicids2017/MAIN_qrdqn_cicids2017_canonical_full_random_20260609_193655 \\
        --percentiles runs/cicids2017/MAIN_qrdqn_cicids2017_canonical_full_random_20260609_193655/train_percentiles.npz \\
        --clip-z 10.0 \\
        --export-diagnostics
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
sys.path.append(str(_REPO / "src"))

from canonical_schema import FEATURES_CANON, map_to_canonical  # noqa: E402
from scaling_utils import apply_percentile_clipping, apply_z_clipping  # noqa: E402
from metrics_utils import confusion_to_metrics  # noqa: E402
from artifact_integrity import resolve_trusted_artifact  # noqa: E402


# ---------------------------------------------------------------------------
# Column mapping: flowmeter-py / CICFlowMeter column names → canonical names
# (Kept identical to predict_real_traffic.py for compatibility.)
# ---------------------------------------------------------------------------

FLOWMETER_PY_TO_CANON: Dict[str, str] = {
    "flow_duration": "flow_duration",
    "tot_fwd_pkts": "total_fwd_packets",
    "tot_bwd_pkts": "total_bwd_packets",
    "totlen_fwd_pkts": "total_length_of_fwd_packets",
    "totlen_bwd_pkts": "total_length_of_bwd_packets",
    "fwd_pkt_len_max": "fwd_packet_length_max",
    "fwd_pkt_len_min": "fwd_packet_length_min",
    "fwd_pkt_len_mean": "fwd_packet_length_mean",
    "fwd_pkt_len_std": "fwd_packet_length_std",
    "bwd_pkt_len_max": "bwd_packet_length_max",
    "bwd_pkt_len_min": "bwd_packet_length_min",
    "bwd_pkt_len_mean": "bwd_packet_length_mean",
    "bwd_pkt_len_std": "bwd_packet_length_std",
    "flow_byts_s": "flow_bytes_per_s",
    "flow_pkts_s": "flow_packets_per_s",
    "flow_iat_mean": "flow_iat_mean",
    "flow_iat_std": "flow_iat_std",
    "flow_iat_max": "flow_iat_max",
    "flow_iat_min": "flow_iat_min",
    "fwd_iat_tot": "fwd_iat_total",
    "fwd_iat_mean": "fwd_iat_mean",
    "fwd_iat_std": "fwd_iat_std",
    "fwd_iat_max": "fwd_iat_max",
    "fwd_iat_min": "fwd_iat_min",
    "bwd_iat_tot": "bwd_iat_total",
    "bwd_iat_mean": "bwd_iat_mean",
    "bwd_iat_std": "bwd_iat_std",
    "bwd_iat_max": "bwd_iat_max",
    "bwd_iat_min": "bwd_iat_min",
    "fwd_psh_flags": "fwd_psh_flags",
    "bwd_psh_flags": "bwd_psh_flags",
    "fwd_urg_flags": "fwd_urg_flags",
    "bwd_urg_flags": "bwd_urg_flags",
    "fin_flag_cnt": "fin_flag_count",
    "syn_flag_cnt": "syn_flag_count",
    "rst_flag_cnt": "rst_flag_count",
    "psh_flag_cnt": "psh_flag_count",
    "ack_flag_cnt": "ack_flag_count",
    "urg_flag_cnt": "urg_flag_count",
    "cwr_flag_count": "cwe_flag_count",
    "ece_flag_cnt": "ece_flag_count",
    "fwd_header_len": "fwd_header_length",
    "bwd_header_len": "bwd_header_length",
    "fwd_pkts_s": "fwd_packets_per_s",
    "bwd_pkts_s": "bwd_packets_per_s",
    "pkt_len_min": "min_packet_length",
    "pkt_len_max": "max_packet_length",
    "pkt_len_mean": "packet_length_mean",
    "pkt_len_std": "packet_length_std",
    "pkt_len_var": "packet_length_variance",
    "down_up_ratio": "down_up_ratio",
    "pkt_size_avg": "average_packet_size",
    "fwd_seg_size_avg": "avg_fwd_segment_size",
    "bwd_seg_size_avg": "avg_bwd_segment_size",
    "fwd_byts_b_avg": "fwd_avg_bytes_per_bulk",
    "fwd_pkts_b_avg": "fwd_avg_packets_per_bulk",
    "fwd_blk_rate_avg": "fwd_avg_bulk_rate",
    "bwd_byts_b_avg": "bwd_avg_bytes_per_bulk",
    "bwd_pkts_b_avg": "bwd_avg_packets_per_bulk",
    "bwd_blk_rate_avg": "bwd_avg_bulk_rate",
    "subflow_fwd_pkts": "subflow_fwd_packets",
    "subflow_fwd_byts": "subflow_fwd_bytes",
    "subflow_bwd_pkts": "subflow_bwd_packets",
    "subflow_bwd_byts": "subflow_bwd_bytes",
    "init_fwd_win_byts": "init_win_bytes_forward",
    "init_bwd_win_byts": "init_win_bytes_backward",
    "fwd_act_data_pkts": "act_data_pkt_fwd",
    "fwd_seg_size_min": "min_seg_size_forward",
    "active_mean": "active_mean",
    "active_std": "active_std",
    "active_max": "active_max",
    "active_min": "active_min",
    "idle_mean": "idle_mean",
    "idle_std": "idle_std",
    "idle_max": "idle_max",
    "idle_min": "idle_min",
}

META_COLS: List[str] = ["src_ip", "dst_ip", "src_port", "dst_port", "protocol", "timestamp"]
TRUTH_COLS: List[str] = ["truth_label", "truth_y", "source_label"]

_TIME_COL_HINTS: Tuple[str, ...] = ("duration", "iat", "active", "idle")

# Number of raw canonical features (without the missingness mask)
_N_CANON = len(FEATURES_CANON)


# ---------------------------------------------------------------------------
# Helper: time-unit harmonisation
# ---------------------------------------------------------------------------

def maybe_convert_time_units(df: pd.DataFrame) -> pd.DataFrame:
    """
    If time columns look like seconds (median < 1), convert to microseconds.

    CICIDS2017 uses microseconds; real flow extractors may output seconds.
    """
    if "flow_duration" in df.columns:
        med = pd.to_numeric(df["flow_duration"], errors="coerce").median()
        if pd.notna(med) and med < 1.0:
            time_cols = [c for c in df.columns if any(h in c for h in _TIME_COL_HINTS)]
            df = df.copy()
            for c in time_cols:
                df[c] = pd.to_numeric(df[c], errors="coerce") * 1e6
            print(f"[units] Converted {len(time_cols)} time columns from seconds -> microseconds (×1e6).")
        else:
            print("[units] Time units look already large; no conversion applied.")
    return df


# ---------------------------------------------------------------------------
# Helper: model loading (QRDQN with DQN fallback)
# ---------------------------------------------------------------------------

def load_model(model_path: Path) -> Tuple[object, str]:
    """Load the trained model, returning ``(model, algo)``.

    Uses QRDQN, falling back to DQN **only** when ``sb3_contrib`` is not
    installed (``ImportError``). Any other failure — e.g. a corrupt or
    incompatible ``.zip`` — is allowed to propagate rather than being masked
    by a silent fallback. ``algo`` is the loaded class name, recorded in the
    run's ``config.json`` for provenance.
    """
    try:
        from sb3_contrib import QRDQN
    except ImportError:
        from stable_baselines3 import DQN
        model = DQN.load(str(model_path))
        return model, type(model).__name__
    model = QRDQN.load(str(model_path))
    return model, type(model).__name__


def build_predictions_df(
    y_pred: np.ndarray,
    meta: Optional[pd.DataFrame] = None,
    *,
    include_sensitive_metadata: bool = False,
) -> pd.DataFrame:
    out_df = pd.DataFrame({"pred_action": y_pred.astype(int)})
    if include_sensitive_metadata and meta is not None:
        out_df = pd.concat([meta.reset_index(drop=True), out_df], axis=1)
    return out_df


# ---------------------------------------------------------------------------
# Helper: batched prediction
# ---------------------------------------------------------------------------

def batched_predict(model, X: np.ndarray, batch_size: int = 4096) -> np.ndarray:
    """Run model.predict in batches to avoid OOM on large flow CSVs."""
    preds: List[np.ndarray] = []
    for i in range(0, len(X), batch_size):
        a, _ = model.predict(X[i : i + batch_size], deterministic=True)
        preds.append(np.asarray(a).reshape(-1))
    return np.concatenate(preds, axis=0)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def compute_diagnostics(
    X_scaled: np.ndarray,
    feature_names: List[str],
    top_n: int = 15,
) -> Dict:
    """
    Compute z-score diagnostics on scaled features (first _N_CANON dims only).

    Parameters
    ----------
    X_scaled : np.ndarray
        Scaled feature matrix, shape (n_samples, n_features).
        The missingness-mask columns (dims _N_CANON..) are excluded.
    feature_names : list[str]
        Names for all columns of X_scaled.
    top_n : int
        Number of worst features to include in the per-feature report.

    Returns
    -------
    dict
        Summary and per-feature diagnostics (JSON-serialisable).
    """
    X_feat = X_scaled[:, :_N_CANON]
    names_feat = feature_names[:_N_CANON]

    z_abs = np.abs(X_feat)
    z_abs_max = float(z_abs.max())
    z_abs_mean = float(z_abs.mean())
    z_gt10_count = int((z_abs > 10).sum())
    z_gt10_pct = float(z_gt10_count / z_abs.size * 100)

    print(
        f"\n[diagnostics] Z abs max: {z_abs_max:.2f} | "
        f"Z abs mean: {z_abs_mean:.4f} | "
        f"count(|z|>10): {z_gt10_count} ({z_gt10_pct:.2f}%)"
    )

    per_feature_max = z_abs.max(axis=0)
    per_feature_gt10 = (z_abs > 10).sum(axis=0)

    sorted_idx = np.argsort(per_feature_max)[::-1][:top_n]
    top_features = [
        {
            "feature": names_feat[i],
            "max_abs_z": float(per_feature_max[i]),
            "count_gt_10": int(per_feature_gt10[i]),
        }
        for i in sorted_idx
    ]

    print(f"[diagnostics] Top {top_n} worst features by max |z|:")
    for entry in top_features:
        print(
            f"  {entry['feature']:45s}  max|z|={entry['max_abs_z']:8.2f}  "
            f"count>10={entry['count_gt_10']}"
        )

    return {
        "z_abs_max": z_abs_max,
        "z_abs_mean": z_abs_mean,
        "z_gt10_count": z_gt10_count,
        "z_gt10_pct": z_gt10_pct,
        "top_features": top_features,
    }


def compute_truth_metrics(df: pd.DataFrame, y_pred: np.ndarray) -> Optional[Dict[str, float]]:
    """
    Compute evaluation metrics when ground-truth columns are present in the flows CSV.

    Predictions must align row-for-row with ``df`` because the ground-truth
    mask is derived from ``df`` and used to index ``y_pred``; a length mismatch
    raises ``ValueError`` rather than mis-indexing.
    """
    if len(y_pred) != len(df):
        raise ValueError(
            f"len(y_pred)={len(y_pred)} != len(df)={len(df)}; "
            "predictions are not aligned with the input flows."
        )

    truth_series: Optional[pd.Series] = None

    if "truth_y" in df.columns:
        truth_series = pd.to_numeric(df["truth_y"], errors="coerce")
    elif "truth_label" in df.columns:
        truth_series = (
            df["truth_label"]
            .astype(str)
            .str.strip()
            .str.upper()
            .map({"BENIGN": 0, "ATTACK": 1, "MALICIOUS": 1})
        )

    if truth_series is None:
        return None

    valid_mask = truth_series.isin([0, 1]).to_numpy()
    if not valid_mask.any():
        return None

    y_true = truth_series.to_numpy(dtype=np.float32)[valid_mask].astype(np.int64)
    y_pred_valid = y_pred[valid_mask].astype(np.int64)

    tp = int(((y_true == 1) & (y_pred_valid == 1)).sum())
    tn = int(((y_true == 0) & (y_pred_valid == 0)).sum())
    fp = int(((y_true == 0) & (y_pred_valid == 1)).sum())
    fn = int(((y_true == 1) & (y_pred_valid == 0)).sum())

    # Single source of truth for metric definitions (see src/metrics_utils.py).
    return {
        "n_labeled_flows": int(valid_mask.sum()),
        **confusion_to_metrics(tn, fp, fn, tp),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 2 robust offline inference pipeline for RL defender."
    )
    parser.add_argument(
        "--flows", required=True, type=Path,
        help="Path to flows.csv produced by a flow extractor.",
    )
    parser.add_argument(
        "--run-dir", type=Path, default=None,
        help=(
            "Trusted training run dir containing artifact_manifest.json. "
            "When provided, model/scaler/percentiles paths must match manifest hashes."
        ),
    )
    parser.add_argument(
        "--model", type=Path, default=None,
        help="Path to trained model .zip (QRDQN or DQN); must match --run-dir unless unsafe.",
    )
    parser.add_argument(
        "--scaler", type=Path, default=None,
        help="Path to scaler.joblib saved during training; must match --run-dir unless unsafe.",
    )
    parser.add_argument(
        "--percentiles", type=Path, default=None,
        help="Path to train_percentiles.npz for raw-feature percentile clipping.",
    )
    parser.add_argument(
        "--clip-z", type=float, default=None, metavar="MAX_Z",
        help="Z-score clipping threshold applied after scaling (e.g. 10.0).",
    )
    parser.add_argument(
        "--no-scale", action="store_true",
        help="Skip scaling entirely (debug/ablation mode).",
    )
    parser.add_argument(
        "--export-diagnostics", action="store_true",
        help="Save full z-score diagnostics to diagnostics.json in output folder.",
    )
    parser.add_argument(
        "--include-sensitive-metadata", action="store_true",
        help="Write local-only predictions_sensitive_local.csv with IPs/ports/timestamps/labels.",
    )
    parser.add_argument(
        "--allow-unsafe-artifacts", action="store_true",
        help="Allow direct model/scaler/percentile paths without manifest hash verification.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    run_id = "P2v2_pred_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = _REPO / "runs" / "phase2" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    run_dir = args.run_dir
    model_path = resolve_trusted_artifact(
        run_dir,
        "model",
        args.model,
        repo_root=_REPO,
        allow_unsafe=args.allow_unsafe_artifacts,
    )
    scaler_path = None
    if not args.no_scale:
        scaler_path = resolve_trusted_artifact(
            run_dir,
            "scaler",
            args.scaler,
            repo_root=_REPO,
            allow_unsafe=args.allow_unsafe_artifacts,
        )
    percentiles_path = None
    if args.percentiles is not None:
        percentiles_path = resolve_trusted_artifact(
            run_dir,
            "train_percentiles",
            args.percentiles,
            repo_root=_REPO,
            allow_unsafe=args.allow_unsafe_artifacts,
        )

    print(f"{'='*60}")
    print(f"  Phase 2 inference (v2): {run_id}")
    print("  flows   : loaded from configured input")
    print(f"  run dir : {'trusted manifest' if run_dir else 'unsafe override'}")
    print("  model   : trusted artifact")
    print(f"  scaler  : {'skipped' if scaler_path is None else 'trusted artifact'}")
    print(f"  percs   : {'not provided' if percentiles_path is None else 'trusted artifact'}")
    print(f"  clip-z  : {args.clip_z}")
    print(f"  no-scale: {args.no_scale}")
    print(f"  output  : runs/phase2/{run_id}")
    print(f"{'='*60}\n")

    # ── Step 1: Load flows CSV ──────────────────────────────────────────
    df = pd.read_csv(args.flows)
    print(f"[load] Loaded {len(df)} flows from {args.flows}")

    # ── Step 2: Separate metadata ───────────────────────────────────────
    present_meta_cols = [c for c in META_COLS + TRUTH_COLS if c in df.columns]
    meta = df[present_meta_cols].copy() if present_meta_cols else None

    # ── Step 3: Time-unit harmonisation ────────────────────────────────
    df = maybe_convert_time_units(df)

    # ── Step 4: Map to canonical schema ────────────────────────────────
    canon = map_to_canonical(df, FLOWMETER_PY_TO_CANON)
    print(f"[canonical] present={canon.n_present}  missing={canon.n_missing}")

    X = canon.combined.astype(np.float32)  # (n, 152): features + mask
    feature_names = canon.feature_names

    # ── Step 5: Percentile clipping (raw features only, first 76 dims) ──
    if percentiles_path is not None:
        percs = np.load(percentiles_path)
        p_low: np.ndarray = percs["p_low"]
        p_high: np.ndarray = percs["p_high"]
        X_feat = X[:, :_N_CANON]
        # FIX: slice percentiles to match feature-only dims (first _N_CANON)
        X_feat_clipped = apply_percentile_clipping(X_feat, p_low[:_N_CANON], p_high[:_N_CANON])
        X = np.concatenate([X_feat_clipped, X[:, _N_CANON:]], axis=1)
        print("[clip] Percentile clipping applied (p0.5 / p99.5).")

    # ── Step 6: Scaling ─────────────────────────────────────────────────
    if not args.no_scale:
        scaler = joblib.load(scaler_path)
        X = scaler.transform(X).astype(np.float32)
        print("[scale] StandardScaler applied.")
    else:
        print("[scale] Skipped (--no-scale).")

    # ── Step 7: Z-score clipping ────────────────────────────────────────
    if args.clip_z is not None:
        X = apply_z_clipping(X, args.clip_z)
        print(f"[clip] Z-score clipping applied (max_z={args.clip_z}).")

    # ── Step 8: Diagnostics ─────────────────────────────────────────────
    diag = compute_diagnostics(X, feature_names)
    if args.export_diagnostics:
        diag_path = out_dir / "diagnostics.json"
        diag_path.write_text(json.dumps(diag, indent=2), encoding="utf-8")
        print(f"[diagnostics] Saved to {diag_path}")

    # ── Step 9: Load model + predict ────────────────────────────────────
    model, model_class = load_model(model_path)
    y_pred = batched_predict(model, X, batch_size=4096)

    # Guard: predictions must align 1:1 with the input flows before y_pred is
    # sliced by a df-derived mask (truth metrics) or concatenated with df
    # metadata. Fail loud rather than silently misaligning labels.
    if len(y_pred) != len(df):
        raise ValueError(
            f"Prediction/row-count mismatch: {len(y_pred)} predictions vs "
            f"{len(df)} input flows; cannot align predictions to flows."
        )

    # ── Step 10: Save outputs ───────────────────────────────────────────
    out_df = build_predictions_df(y_pred, meta, include_sensitive_metadata=False)
    predictions_path = out_dir / "predictions.csv"
    out_df.to_csv(predictions_path, index=False)
    sensitive_predictions_path = None
    if args.include_sensitive_metadata:
        sensitive_predictions_path = out_dir / "predictions_sensitive_local.csv"
        build_predictions_df(
            y_pred,
            meta,
            include_sensitive_metadata=True,
        ).to_csv(sensitive_predictions_path, index=False)

    block_rate = float((y_pred == 1).mean())
    metrics = {
        "n_flows": int(len(y_pred)),
        "block_rate": block_rate,
        "allow_rate": float((y_pred == 0).mean()),
        "z_abs_max": diag["z_abs_max"],
        "z_abs_mean": diag["z_abs_mean"],
        "z_gt10_count": diag["z_gt10_count"],
        "z_gt10_pct": diag["z_gt10_pct"],
    }

    truth_metrics = compute_truth_metrics(df, y_pred)
    if truth_metrics is not None:
        metrics.update(truth_metrics)

    config = {
        "run_id": run_id,
        "flows_csv": str(args.flows),
        "trusted_run_dir": str(run_dir) if run_dir else None,
        "allow_unsafe_artifacts": bool(args.allow_unsafe_artifacts),
        "model_zip": str(model_path),
        "model_class": model_class,
        "scaler": str(scaler_path) if scaler_path else None,
        "percentiles": str(percentiles_path) if percentiles_path else None,
        "clip_z": args.clip_z,
        "no_scale": args.no_scale,
        "predictions_include_sensitive_metadata": False,
        "sensitive_predictions_path": str(sensitive_predictions_path) if sensitive_predictions_path else None,
        "mask_semantics": "1=present,0=missing",
        "note": "Robust Phase 2 offline inference (v2) on extracted flows.csv",
    }

    config_path = out_dir / "config.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"\n[output] predictions : {predictions_path}")
    if sensitive_predictions_path is not None:
        print(f"[output] sensitive   : {sensitive_predictions_path}")
    print(f"[output] config      : {config_path}")
    print(f"[output] metrics     : {metrics_path}")
    print(f"[result] Block rate  : {block_rate:.4f}")
    print(f"\n{'='*60}")
    print(f"  Inference completada: {run_id}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
