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
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.run_artifacts import atomic_write_json  # noqa: E402
from src.tensorboard_export import export_tensorboard_scalars  # noqa: E402


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


def _find_event_dirs(runs_dir: Path, run_id: str) -> list[Path]:
    candidates = []
    for path in sorted(runs_dir.glob(f"{run_id}*")):
        if not path.is_dir():
            continue
        if any(path.glob("events.out.tfevents.*")):
            candidates.append(path)
    return candidates


def _update_legacy_running_manifest(
    run_dir: Path, export_manifest_path: Path, output_dir: Path
) -> bool:
    manifest_path = run_dir / "artifact_manifest.json"
    if not manifest_path.exists():
        return False

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "2.0" or manifest.get("status") == "completed":
        return False
    artifacts = manifest.setdefault("artifacts", {})
    artifacts["tensorboard_scalar_plots"] = str(output_dir)
    artifacts["tensorboard_scalar_export_manifest"] = str(export_manifest_path)
    atomic_write_json(manifest_path, manifest)
    return True


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

    export_manifest = export_tensorboard_scalars(
        event_dirs,
        output_dir,
        run_id=args.run_id,
        include_plots=True,
    )
    export_manifest_path = output_dir / "tensorboard_scalar_export_manifest.json"

    if not args.no_update_manifest:
        updated = _update_legacy_running_manifest(run_dir, export_manifest_path, output_dir)
        if not updated and (run_dir / "artifact_manifest.json").exists():
            print("Artifact manifest left unchanged because completed evidence is immutable.")

    print(f"Exported {len(export_manifest['exported_scalars'])} scalar curve(s) to: {output_dir}")
    print(f"Export manifest: {export_manifest_path}")


if __name__ == "__main__":
    main()
