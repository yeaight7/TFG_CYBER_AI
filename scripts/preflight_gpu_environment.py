"""Provider-neutral GPU experimental environment preflight CLI."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:  # pragma: no cover - direct script execution
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.gpu_preflight import (  # noqa: E402
    DEFAULT_MAX_AGE_HOURS,
    PreflightError,
    PreflightThresholds,
    run_preflight,
)


_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_SPEC = _REPO_ROOT / "experiments" / "final_experiment_campaign.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify hardware, software, data, cache, CUDA, artifact, and snapshot "
            "readiness for the provider-neutral GPU experimental environment."
        )
    )
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--cache-root", required=True, type=Path)
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument("--snapshot-root", required=True, type=Path)
    parser.add_argument("--campaign-spec", type=Path, default=_DEFAULT_SPEC)
    parser.add_argument("--phase2-input", type=Path)
    parser.add_argument("--expect-phase2-labels", action="store_true")
    parser.add_argument("--runtime-benchmark", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--repo-root", type=Path, default=_REPO_ROOT)
    parser.add_argument("--max-age-hours", type=float, default=DEFAULT_MAX_AGE_HOURS)
    parser.add_argument("--min-logical-cpus", type=int, default=16)
    parser.add_argument("--min-ram-gib", type=float, default=120)
    parser.add_argument("--min-gpu-count", type=int, default=1)
    parser.add_argument("--min-vram-gib", type=float, default=80)
    parser.add_argument("--min-free-gib", type=float, default=100)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output = args.output or (args.artifact_root / "preflight_report.json")
    try:
        thresholds = PreflightThresholds(
            min_logical_cpus=args.min_logical_cpus,
            min_ram_gib=args.min_ram_gib,
            min_gpu_count=args.min_gpu_count,
            min_vram_gib=args.min_vram_gib,
            min_free_gib=args.min_free_gib,
        )
        report = run_preflight(
            output_path=output,
            campaign_spec=args.campaign_spec,
            dataset_root=args.dataset_root,
            cache_root=args.cache_root,
            artifact_root=args.artifact_root,
            snapshot_root=args.snapshot_root,
            phase2_input=args.phase2_input,
            expect_phase2_labels=args.expect_phase2_labels,
            thresholds=thresholds,
            max_age_hours=args.max_age_hours,
            runtime_benchmark=args.runtime_benchmark,
            repo_root=args.repo_root,
        )
    except (OSError, PreflightError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
