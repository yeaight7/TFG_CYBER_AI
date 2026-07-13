"""Discoverable summaries derived only from persisted authoritative run artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping


RUN_SUMMARY_SCHEMA_VERSION = "1.0"


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _require_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _run_identity(config: Mapping[str, Any]) -> dict[str, Any]:
    request = _require_mapping(config.get("request", {}), label="config.request")
    return {
        "run_id": config.get("run_id"),
        "algorithm": config.get("algorithm"),
        "profile_id": config.get("profile_id"),
        "profile_hash": config.get("profile_hash"),
        "campaign_id": request.get("campaign_id"),
        "logical_run_id": request.get("logical_run_id"),
        "attempt": request.get("attempt"),
        "split_mode": config.get("split_mode"),
        "split_seed": config.get("split_seed"),
        "model_seed": config.get("model_seed"),
    }


def _scalar_last_values(
    run_dir: Path,
    scalar_manifest: Mapping[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    records = scalar_manifest.get("exported_scalars", [])
    if not isinstance(records, list):
        raise ValueError("TensorBoard scalar manifest exported_scalars must be a list")

    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("TensorBoard scalar record must be an object")
        tag = record.get("tag")
        csv_name = record.get("csv")
        if not isinstance(tag, str) or not isinstance(csv_name, str):
            raise ValueError("TensorBoard scalar record is missing tag or csv")
        csv_relative = Path("tensorboard_scalars") / csv_name
        csv_path = run_dir / csv_relative
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            raise ValueError(f"TensorBoard scalar CSV has no rows: {csv_relative}")
        last = rows[-1]
        grouped.setdefault(tag, []).append(
            {
                "event_dir": record.get("event_dir"),
                "csv": csv_relative.as_posix(),
                "samples": len(rows),
                "last_step": int(last["step"]),
                "last_wall_time": float(last["wall_time"]),
                "last_value": float(last["value"]),
            }
        )
    return grouped


def build_run_summary(run_dir: Path) -> dict[str, Any]:
    """Build a human-discoverable view from sealed-run source artifacts."""
    run_dir = Path(run_dir)
    config = _require_mapping(_read_json(run_dir / "config.json"), label="config.json")
    environment = _require_mapping(
        _read_json(run_dir / "environment.json"), label="environment.json"
    )
    metrics = _require_mapping(_read_json(run_dir / "metrics.json"), label="metrics.json")
    timing = _require_mapping(_read_json(run_dir / "timing.json"), label="timing.json")
    monitoring = _require_mapping(
        _read_json(run_dir / "monitoring.json"), label="monitoring.json"
    )
    scalar_manifest = _require_mapping(
        _read_json(
            run_dir
            / "tensorboard_scalars"
            / "tensorboard_scalar_export_manifest.json"
        ),
        label="tensorboard scalar export manifest",
    )

    return {
        "schema_version": RUN_SUMMARY_SCHEMA_VERSION,
        "run": _run_identity(config),
        "training_hyperparameters": config.get("training_hyperparameters"),
        "training_execution": config.get("training_execution"),
        "device_selected": environment.get("device_selected"),
        "hardware": environment.get("hardware"),
        "metrics": dict(metrics),
        "timing": dict(timing),
        "monitoring": dict(monitoring),
        "tensorboard_scalars": _scalar_last_values(run_dir, scalar_manifest),
        "artifact_paths": config.get("artifact_paths"),
    }


__all__ = ["RUN_SUMMARY_SCHEMA_VERSION", "build_run_summary"]
