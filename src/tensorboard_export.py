"""Reusable TensorBoard scalar-to-CSV export for artifact-complete runs."""

from __future__ import annotations

import csv
import io
import re
from pathlib import Path
from typing import Any, Iterable

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from src.run_artifacts import atomic_write_json, atomic_write_text


def safe_filename(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "__", value).strip("_")
    return safe or "scalar"


def _plot_scalar(tag: str, rows: list[dict[str, Any]], output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot(
        [row["step"] for row in rows],
        [row["value"] for row in rows],
        linewidth=1.8,
    )
    axis.set_title(tag)
    axis.set_xlabel("Timesteps")
    axis.set_ylabel("Value")
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def export_tensorboard_scalars(
    event_dirs: Iterable[Path],
    output_dir: Path,
    *,
    run_id: str | None = None,
    include_plots: bool = False,
) -> dict[str, Any]:
    """Export every scalar series and return the written export manifest."""
    directories = [Path(path) for path in event_dirs]
    if not directories:
        raise FileNotFoundError("No TensorBoard event directories were provided")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    exported: list[dict[str, Any]] = []

    for event_dir in directories:
        if not event_dir.is_dir():
            raise FileNotFoundError(f"TensorBoard event directory does not exist: {event_dir}")
        accumulator = EventAccumulator(str(event_dir))
        accumulator.Reload()
        for tag in accumulator.Tags().get("scalars", []):
            rows = [
                {"step": event.step, "wall_time": event.wall_time, "value": event.value}
                for event in accumulator.Scalars(tag)
            ]
            if not rows:
                continue
            stem = f"{safe_filename(event_dir.name)}__{safe_filename(tag)}"
            csv_name = f"{stem}.csv"
            csv_buffer = io.StringIO(newline="")
            writer = csv.DictWriter(
                csv_buffer,
                fieldnames=("step", "wall_time", "value"),
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(rows)
            atomic_write_text(output_dir / csv_name, csv_buffer.getvalue())
            record = {
                "event_dir": str(event_dir),
                "tag": tag,
                "rows": len(rows),
                "csv": csv_name,
            }
            if include_plots:
                png_name = f"{stem}.png"
                _plot_scalar(tag, rows, output_dir / png_name)
                record["png"] = png_name
            exported.append(record)

    manifest = {
        "run_id": run_id,
        "output_dir": str(output_dir),
        "event_dirs": [str(path) for path in directories],
        "exported_scalars": exported,
    }
    atomic_write_json(output_dir / "tensorboard_scalar_export_manifest.json", manifest)
    return manifest
