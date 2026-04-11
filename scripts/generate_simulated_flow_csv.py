"""
generate_simulated_flow_csv.py

Create a Phase 2-compatible simulated flow CSV from CICIDS2017 rows.

This script does not fabricate packet-level PCAPs. It generates flow-level
CSV exports that match the Phase 2 flow layout under `pcaps/`, with optional
ground-truth columns appended for later evaluation.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
sys.path.append(str(_REPO / "src"))

from canonical_schema import CICIDS2017_TO_CANON  # noqa: E402
from load_cicids2017 import list_cicids2017_csv_files  # noqa: E402
from predict_real_traffic_v2 import FLOWMETER_PY_TO_CANON  # noqa: E402


META_COLS: List[str] = ["src_ip", "dst_ip", "src_port", "dst_port", "protocol", "timestamp"]
TRUTH_COLS: List[str] = ["truth_label", "truth_y", "source_label"]
TIME_OUTPUT_COLS: Tuple[str, ...] = (
    "flow_duration",
    "flow_iat_mean",
    "flow_iat_std",
    "flow_iat_max",
    "flow_iat_min",
    "fwd_iat_tot",
    "fwd_iat_mean",
    "fwd_iat_std",
    "fwd_iat_max",
    "fwd_iat_min",
    "bwd_iat_tot",
    "bwd_iat_mean",
    "bwd_iat_std",
    "bwd_iat_max",
    "bwd_iat_min",
    "active_mean",
    "active_std",
    "active_max",
    "active_min",
    "idle_mean",
    "idle_std",
    "idle_max",
    "idle_min",
)
FILTER_METRICS: Dict[str, str] = {
    "Flow Duration": "flow_duration",
    "Total Fwd Packets": "tot_fwd_pkts",
    "Total Backward Packets": "tot_bwd_pkts",
    "Flow Packets/s": "flow_pkts_s",
    "Flow Bytes/s": "flow_byts_s",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a simulated flow CSV compatible with the Phase 2 pipeline."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Destination CSV path.",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=250_000,
        help="Number of rows to write. Sampling uses replacement when needed.",
    )
    parser.add_argument(
        "--reference-flows",
        type=Path,
        default=None,
        help="Optional reference flow CSV used to derive conservative filter windows.",
    )
    parser.add_argument(
        "--q-low",
        type=float,
        default=0.01,
        help="Low quantile for reference-based filtering.",
    )
    parser.add_argument(
        "--q-high",
        type=float,
        default=0.99,
        help="High quantile for reference-based filtering.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--base-timestamp",
        type=str,
        default="2026-04-11 12:00:00",
        help="Starting timestamp for generated metadata.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100_000,
        help="Chunk size used both for reading and writing.",
    )
    parser.add_argument(
        "--label-mode",
        choices=("benign", "attack"),
        default="benign",
        help="Whether to sample benign or attack CICIDS2017 rows.",
    )
    return parser.parse_args()


def get_output_columns(reference_flows: Path) -> List[str]:
    cols = pd.read_csv(reference_flows, nrows=0).columns.tolist()
    if len(cols) != 82:
        raise ValueError(f"Expected 82 columns in reference flow CSV, found {len(cols)}")
    return cols + TRUTH_COLS


def build_source_to_output_mapping() -> Dict[str, str]:
    canon_to_flow: Dict[str, str] = {}
    for flow_col, canon_col in FLOWMETER_PY_TO_CANON.items():
        canon_to_flow.setdefault(canon_col, flow_col)

    source_to_output: Dict[str, str] = {}
    for src_col, canon_col in CICIDS2017_TO_CANON.items():
        flow_col = canon_to_flow.get(canon_col)
        if flow_col is not None:
            source_to_output[src_col.strip()] = flow_col
    return source_to_output


def compute_reference_windows(reference_flows: Path, q_low: float, q_high: float) -> Dict[str, Tuple[float, float]]:
    if not (0.0 <= q_low < q_high <= 1.0):
        raise ValueError("Reference quantiles must satisfy 0 <= q_low < q_high <= 1.")

    ref = pd.read_csv(reference_flows, usecols=list(FILTER_METRICS.values()))
    windows: Dict[str, Tuple[float, float]] = {}
    for ref_col in FILTER_METRICS.values():
        series = pd.to_numeric(ref[ref_col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        windows[ref_col] = (float(series.quantile(q_low)), float(series.quantile(q_high)))
    return windows


def generate_timestamps(base_timestamp: datetime, n_rows: int, row_offset: int) -> List[str]:
    start = base_timestamp + timedelta(milliseconds=row_offset * 75)
    return [
        (start + timedelta(milliseconds=i * 75)).strftime("%Y-%m-%d %H:%M:%S")
        for i in range(n_rows)
    ]


def materialize_candidate_rows(
    source_dir: Path,
    output_columns: List[str],
    label_mode: str,
    reference_windows: Optional[Dict[str, Tuple[float, float]]],
    read_chunk_size: int,
) -> pd.DataFrame:
    label_key = "label"
    source_to_output = build_source_to_output_mapping()
    feature_cols = [c for c in output_columns if c not in META_COLS + TRUTH_COLS]
    frames: List[pd.DataFrame] = []
    truth_label = "BENIGN" if label_mode == "benign" else "ATTACK"
    truth_y = 0 if label_mode == "benign" else 1

    for csv_path in list_cicids2017_csv_files(source_dir):
        for chunk in pd.read_csv(
            csv_path,
            chunksize=read_chunk_size,
            low_memory=True,
            encoding_errors="ignore",
        ):
            chunk.columns = [str(c).strip() for c in chunk.columns]
            label_col = next((c for c in chunk.columns if c.strip().lower() == label_key), None)
            if label_col is None:
                raise ValueError(f"No label column found in {csv_path}")

            label_series = chunk[label_col].astype(str).str.strip().str.upper()
            if label_mode == "benign":
                selected_rows = chunk[label_series == "BENIGN"].copy()
            else:
                selected_rows = chunk[label_series != "BENIGN"].copy()

            if selected_rows.empty:
                continue

            selected_rows = selected_rows.rename(columns=source_to_output)

            filter_df = pd.DataFrame(index=selected_rows.index)
            for source_name, output_name in FILTER_METRICS.items():
                source_name = source_name.strip()
                if output_name not in selected_rows.columns and source_name in chunk.columns:
                    selected_rows[output_name] = pd.to_numeric(selected_rows[source_name], errors="coerce")
                filter_df[output_name] = pd.to_numeric(selected_rows[output_name], errors="coerce")

            filter_df["flow_duration"] = filter_df["flow_duration"] / 1e6
            valid_mask = np.ones(len(filter_df), dtype=bool)
            for col in FILTER_METRICS.values():
                series = filter_df[col].replace([np.inf, -np.inf], np.nan)
                if col == "flow_duration":
                    valid_mask &= (series > 0).fillna(False).to_numpy()
                else:
                    valid_mask &= (series >= 0).fillna(False).to_numpy()

            if reference_windows is not None:
                for col, (low, high) in reference_windows.items():
                    series = filter_df[col].replace([np.inf, -np.inf], np.nan)
                    valid_mask &= series.between(low, high).fillna(False).to_numpy()

            kept_rows = selected_rows.loc[valid_mask].copy()
            if kept_rows.empty:
                continue

            out = pd.DataFrame(index=kept_rows.index)
            for col in feature_cols:
                if col in kept_rows.columns:
                    out[col] = pd.to_numeric(kept_rows[col], errors="coerce")
                else:
                    out[col] = 0.0

            out["dst_port"] = pd.to_numeric(
                kept_rows.get("Destination Port", 80),
                errors="coerce",
            ).fillna(80).astype(int)

            for col in TIME_OUTPUT_COLS:
                if col in out.columns:
                    out[col] = pd.to_numeric(out[col], errors="coerce") / 1e6

            out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            out["truth_label"] = truth_label
            out["truth_y"] = truth_y
            out["source_label"] = label_series.loc[kept_rows.index].astype(str).str.strip().str.upper().to_numpy()
            frames.append(out.reset_index(drop=True))

    if not frames:
        raise RuntimeError(f"No candidate rows survived filtering for label mode '{label_mode}'.")

    candidates = pd.concat(frames, ignore_index=True)
    candidates["dst_port"] = candidates["dst_port"].astype(int)
    candidates["truth_y"] = candidates["truth_y"].astype(int)
    return candidates


def write_output(
    candidates: pd.DataFrame,
    output_columns: List[str],
    output_path: Path,
    total_rows: int,
    seed: int,
    base_timestamp: str,
    chunk_size: int,
) -> Dict[str, object]:
    rng = np.random.default_rng(seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    replace = total_rows > len(candidates)
    sampled_idx = rng.choice(len(candidates), size=total_rows, replace=replace)

    base_dt = datetime.strptime(base_timestamp, "%Y-%m-%d %H:%M:%S")
    src_ips = np.array(["172.18.0.3", "172.18.0.4", "172.18.0.5", "172.18.0.6"], dtype=object)
    dst_ips = np.array(["172.18.0.2", "172.18.0.7"], dtype=object)
    protocols = np.array([6, 17], dtype=np.int64)
    protocol_p = np.array([0.92, 0.08], dtype=np.float64)

    first = True
    written = 0
    for start in range(0, total_rows, chunk_size):
        stop = min(start + chunk_size, total_rows)
        batch = candidates.iloc[sampled_idx[start:stop]].copy().reset_index(drop=True)
        n_batch = len(batch)

        batch["src_ip"] = rng.choice(src_ips, size=n_batch)
        batch["dst_ip"] = rng.choice(dst_ips, size=n_batch)
        batch["src_port"] = rng.integers(32768, 60999, size=n_batch)
        batch["dst_port"] = batch["dst_port"].astype(int)
        batch["protocol"] = rng.choice(protocols, size=n_batch, p=protocol_p)
        batch["timestamp"] = generate_timestamps(base_dt, n_batch, row_offset=written)
        batch = batch[output_columns]

        batch.to_csv(output_path, mode="w" if first else "a", header=first, index=False)
        first = False
        written += n_batch

    return {
        "rows_written": written,
        "rows_requested": total_rows,
        "candidate_rows": int(len(candidates)),
        "sampling_with_replacement": replace,
        "output_path": str(output_path),
        "base_timestamp": base_timestamp,
    }


def write_provenance(
    output_path: Path,
    summary: Dict[str, object],
    label_mode: str,
    reference_windows: Optional[Dict[str, Tuple[float, float]]],
    q_low: float,
    q_high: float,
) -> Path:
    provenance = {
        "artifact_type": "dataset_derived_simulated_flow_csv",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "output_csv": str(output_path),
        "source_dataset": "CICIDS2017",
        "label_filter": label_mode.upper(),
        "time_unit_in_output": "seconds",
        **summary,
    }
    if reference_windows is not None:
        provenance["reference_filter_quantiles"] = {
            "q_low": q_low,
            "q_high": q_high,
        }
        provenance["reference_windows"] = {
            name: {"low": bounds[0], "high": bounds[1]}
            for name, bounds in reference_windows.items()
        }

    meta_path = output_path.with_suffix(output_path.suffix + ".meta.json")
    meta_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    return meta_path


def main() -> None:
    args = parse_args()
    default_output = _REPO / "pcaps" / f"flows_simulated_{args.label_mode}.csv"
    output_path = args.output or default_output
    reference_flows = args.reference_flows
    if reference_flows is None and args.label_mode == "benign":
        reference_flows = _REPO / "pcaps" / "flows_benign.csv"

    base_reference = reference_flows or (_REPO / "pcaps" / "flows_benign.csv")
    output_columns = get_output_columns(base_reference)
    reference_windows = (
        compute_reference_windows(reference_flows, args.q_low, args.q_high)
        if reference_flows is not None
        else None
    )

    candidates = materialize_candidate_rows(
        source_dir=_REPO / "datasets" / "CICIDS2017",
        output_columns=output_columns,
        label_mode=args.label_mode,
        reference_windows=reference_windows,
        read_chunk_size=args.chunk_size,
    )

    summary = write_output(
        candidates=candidates,
        output_columns=output_columns,
        output_path=output_path,
        total_rows=args.rows,
        seed=args.seed,
        base_timestamp=args.base_timestamp,
        chunk_size=args.chunk_size,
    )
    meta_path = write_provenance(
        output_path=output_path,
        summary=summary,
        label_mode=args.label_mode,
        reference_windows=reference_windows,
        q_low=args.q_low,
        q_high=args.q_high,
    )

    print(f"[simulated] label mode     : {args.label_mode}")
    print(f"[simulated] candidate rows : {len(candidates)}")
    print(f"[simulated] written rows   : {summary['rows_written']}")
    print(f"[simulated] output         : {output_path}")
    print(f"[simulated] provenance     : {meta_path}")


if __name__ == "__main__":
    main()
