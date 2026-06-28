"""ARCHIVED (Phase 5 / G6) — legacy Phase-2 offline predictor.

Superseded by ``scripts/predict_real_traffic_v2.py``. Not referenced by any
code; retained for historical reference only. NOTE: importing this module
triggers a ~250k-row CICIDS2017 reload at module scope (see below), so it is
meant to be run as a script, not imported.
"""
from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

import sys
REPO = Path(__file__).resolve().parents[2]  # scripts/archive/<this> -> repo root
sys.path.append(str(REPO / "src"))

from canonical_schema import map_to_canonical  # noqa: E402

FLOWMETER_PY_TO_CANON = {
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
    # ojo: algunos extractores lo llaman cwr/cwe
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

META_COLS = ["src_ip", "dst_ip", "src_port", "dst_port", "protocol", "timestamp"]

TIME_COLS_HINTS = ("duration", "iat", "active", "idle")

from load_cicids2017 import load_cicids2017_split  # noqa: E402

# pon aquí los valores que usaste para entrenar el modelo C01
X_train, y_train, X_test, y_test, scaler, feat_names, meta = load_cicids2017_split(
    split_mode="random",
    preset="full",
    seed=42,
    max_rows=250_000,
    scale=True,
    use_canonical=True,
)

def maybe_convert_time_units(df: pd.DataFrame) -> pd.DataFrame:
    # If flow_duration median is < 1, it's probably seconds; CICIDS typically uses microseconds.
    if "flow_duration" in df.columns:
        med = pd.to_numeric(df["flow_duration"], errors="coerce").median()
        if pd.notna(med) and med < 1.0:
            time_cols = [c for c in df.columns if any(h in c for h in TIME_COLS_HINTS)]
            df = df.copy()
            for c in time_cols:
                df[c] = pd.to_numeric(df[c], errors="coerce") * 1e6
            print(f"[units] Converted {len(time_cols)} time columns from seconds -> microseconds (x1e6).")
        else:
            print("[units] Time units look already large; no conversion applied.")
    return df

def load_model(model_path: Path):
    try:
        from sb3_contrib import QRDQN
        return QRDQN.load(model_path)
    except Exception:
        from stable_baselines3 import DQN
        return DQN.load(model_path)

def batched_predict(model, X: np.ndarray, batch_size: int = 4096) -> np.ndarray:
    preds = []
    for i in range(0, len(X), batch_size):
        a, _ = model.predict(X[i:i+batch_size], deterministic=True)
        preds.append(np.asarray(a).reshape(-1))
    return np.concatenate(preds, axis=0)

def main():
    flows_csv = REPO / "pcaps" / "flows.csv"  # <- change
    model_zip = REPO / "models" / "archive" / "C01_qrdqn_cicids2017_canonical_full_20260212_200218.zip"  # <- change if needed

    run_id = "P2_pred_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = REPO / "runs" / "phase2" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(flows_csv)

    # Keep metadata for reporting
    meta = df[META_COLS].copy() if all(c in df.columns for c in META_COLS) else None

    # Optional unit harmonization
    df = maybe_convert_time_units(df)

    # Build canonical feature matrix (76)
    # missing = [f for f in FEATURES_CANON if f not in df.columns]
    # if missing:
    #     raise RuntimeError(f"Missing {len(missing)} canonical features in flows.csv. First missing: {missing[:10]}")

    # X_feat = df[FEATURES_CANON].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)

    # # Missingness mask (your README implies: 1=present, 0=absent)
    # mask = np.ones((X_feat.shape[0], len(FEATURES_CANON)), dtype=np.float32)

    # X = np.concatenate([X_feat, mask], axis=1)  # (n, 152)
    
    canon = map_to_canonical(df, FLOWMETER_PY_TO_CANON)
    print(f"[canonical] present={canon.n_present} missing={canon.n_missing}")

    # X = canon.combined  # shape (n, 152) ya incluye máscara correcta
    X = canon.combined.astype(np.float32)          # (n,152)
    X = scaler.transform(X).astype(np.float32) 

    model = load_model(model_zip)
    y_pred = batched_predict(model, X, batch_size=4096)

    # Save predictions
    out_csv = out_dir / "predictions.csv"
    out_json = out_dir / "config.json"
    out_metrics = out_dir / "metrics.json"

    out = pd.DataFrame({
        "pred_action": y_pred.astype(int),
    })
    if meta is not None:
        out = pd.concat([meta.reset_index(drop=True), out], axis=1)

    out.to_csv(out_csv, index=False)

    block_rate = float((y_pred == 1).mean())
    metrics = {
        "n_flows": int(len(y_pred)),
        "block_rate": block_rate,
        "allow_rate": float((y_pred == 0).mean()),
    }

    config = {
        "run_id": run_id,
        "flows_csv": str(flows_csv),
        "model_zip": str(model_zip),
        "mask_semantics": "1=present,0=missing",
        "note": "Offline Phase2 inference on extracted flows.csv",
    }

    out_json.write_text(json.dumps(config, indent=2), encoding="utf-8")
    out_metrics.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("Saved:", out_csv)
    print("Block rate:", block_rate)

if __name__ == "__main__":
    main()
