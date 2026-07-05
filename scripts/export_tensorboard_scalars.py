#!/usr/bin/env python
"""
Export TensorBoard scalar logs to thesis-friendly CSV and PNG files.

Examples:
    python scripts/export_tensorboard_scalars.py --run-id MAIN_qrdqn_...
    python scripts/export_tensorboard_scalars.py --run-id MAIN_qrdqn_... --output-dir /tmp/plots
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_RUNS_DIR = _REPO_ROOT / "runs" / "cicids2017"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export TensorBoard scalar curves to CSV and PNG artifacts.",
    )
    parser.add_argument(
        "--run-id",
        required=True,
        help="Training RUN_ID, e.g. MAIN_qrdqn_cicids2017_canonical_full_random_YYYYMMDD_HHMMSS",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=_DEFAULT_RUNS_DIR,
        help="Directory containing run artifacts and TensorBoard event dirs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output dir. Default: runs/cicids2017/<RUN_ID>/plots/tensorboard_scalars",
    )
    parser.add_argument(
        "--no-update-manifest",
        action="store_true",
        help="Do not add plot artifact paths to artifact_manifest.json.",
    )
    return parser.parse_args()


def _safe_filename(tag: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "__", tag).strip("_")
    return safe or "scalar"


def _find_event_dirs(runs_dir: Path, run_id: str) -> list[Path]:
    candidates = []
    for path in sorted(runs_dir.glob(f"{run_id}*")):
        if not path.is_dir():
            continue
        if any(path.glob("events.out.tfevents.*")):
            candidates.append(path)
    return candidates


def _read_scalars(event_dir: Path) -> dict[str, pd.DataFrame]:
    accumulator = EventAccumulator(str(event_dir))
    accumulator.Reload()

    scalars: dict[str, pd.DataFrame] = {}
    for tag in accumulator.Tags().get("scalars", []):
        rows = [
            {"step": event.step, "wall_time": event.wall_time, "value": event.value}
            for event in accumulator.Scalars(tag)
        ]
        if rows:
            scalars[tag] = pd.DataFrame(rows)
    return scalars


def _plot_scalar(tag: str, df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(df["step"], df["value"], linewidth=1.8)
    ax.set_title(tag)
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Value")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _update_artifact_manifest(run_dir: Path, export_manifest_path: Path, output_dir: Path) -> None:
    manifest_path = run_dir / "artifact_manifest.json"
    if not manifest_path.exists():
        return

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifacts: dict[str, Any] = manifest.setdefault("artifacts", {})
    artifacts["tensorboard_scalar_plots"] = str(output_dir)
    artifacts["tensorboard_scalar_export_manifest"] = str(export_manifest_path)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    runs_dir = args.runs_dir.resolve()
    run_dir = runs_dir / args.run_id
    output_dir = args.output_dir or (run_dir / "plots" / "tensorboard_scalars")
    output_dir.mkdir(parents=True, exist_ok=True)

    event_dirs = _find_event_dirs(runs_dir, args.run_id)
    if not event_dirs:
        raise FileNotFoundError(
            f"No TensorBoard event dirs found for RUN_ID={args.run_id} under {runs_dir}"
        )

    exported: list[dict[str, Any]] = []
    for event_dir in event_dirs:
        scalars = _read_scalars(event_dir)
        for tag, df in scalars.items():
            event_prefix = _safe_filename(event_dir.name)
            tag_name = _safe_filename(tag)
            stem = f"{event_prefix}__{tag_name}"
            csv_path = output_dir / f"{stem}.csv"
            png_path = output_dir / f"{stem}.png"

            df.to_csv(csv_path, index=False)
            _plot_scalar(tag, df, png_path)

            exported.append(
                {
                    "event_dir": str(event_dir),
                    "tag": tag,
                    "rows": int(len(df)),
                    "csv": str(csv_path),
                    "png": str(png_path),
                }
            )

    export_manifest = {
        "run_id": args.run_id,
        "runs_dir": str(runs_dir),
        "output_dir": str(output_dir),
        "event_dirs": [str(path) for path in event_dirs],
        "exported_scalars": exported,
    }
    export_manifest_path = output_dir / "tensorboard_scalar_export_manifest.json"
    export_manifest_path.write_text(json.dumps(export_manifest, indent=2), encoding="utf-8")

    if not args.no_update_manifest:
        _update_artifact_manifest(run_dir, export_manifest_path, output_dir)

    print(f"Exported {len(exported)} scalar curve(s) to: {output_dir}")
    print(f"Export manifest: {export_manifest_path}")


if __name__ == "__main__":
    main()
