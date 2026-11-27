"""Sequential final-experiment campaign CLI."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:  # pragma: no cover - direct script execution
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.campaign import (  # noqa: E402
    CampaignError,
    CampaignPaths,
    CampaignRunner,
    DEFAULT_CAMPAIGN_ARTIFACT_ROOT,
    load_campaign_spec,
)


_REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate, dry-run, execute, or resume the locked final experiment "
            "campaign sequentially."
        )
    )
    parser.add_argument("spec", type=Path)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=DEFAULT_CAMPAIGN_ARTIFACT_ROOT,
        help="Repository-relative official artifact root (default: runs/final_campaign).",
    )
    parser.add_argument("--cache-root", required=True, type=Path)
    parser.add_argument("--snapshot-root", required=True, type=Path)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Override the provider-neutral dataset path declared by the spec.",
    )
    parser.add_argument("--phase2-input", type=Path, default=None)
    parser.add_argument("--phase2-input-sha256", default=None)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--stage")
    selection.add_argument("--run", dest="logical_run_id")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        spec = load_campaign_spec(args.spec)
        dataset_root = args.dataset_root
        if dataset_root is None:
            dataset_root = _REPO_ROOT / str(spec.raw["dataset_root"])
        paths = CampaignPaths(
            artifact_root=args.artifact_root,
            cache_root=args.cache_root,
            dataset_root=dataset_root,
            snapshot_root=args.snapshot_root,
            phase2_input=args.phase2_input,
            phase2_input_sha256=args.phase2_input_sha256,
            repository_root=_REPO_ROOT,
        )
        runner = CampaignRunner(
            spec,
            campaign_id=args.campaign_id,
            paths=paths,
        )
        if args.dry_run:
            result = runner.dry_run(
                stage=args.stage,
                logical_run_id=args.logical_run_id,
            )
        else:
            result = runner.execute(
                resume=args.resume,
                stage=args.stage,
                logical_run_id=args.logical_run_id,
            )
    except (CampaignError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
