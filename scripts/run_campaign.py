"""Sequential final-experiment campaign CLI."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:  # pragma: no cover - direct script execution
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.campaign import (  # noqa: E402
    CampaignError,
    CampaignPaths,
    CampaignRunner,
    load_campaign_spec,
)
from src.gpu_preflight import PreflightError, verify_preflight_report  # noqa: E402


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
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument("--cache-root", required=True, type=Path)
    parser.add_argument("--snapshot-root", required=True, type=Path)
    parser.add_argument("--preflight-report", required=True, type=Path)
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


def _read_preflight(
    path: Path,
    *,
    campaign_spec_sha256: str,
    dataset_root: Path,
    cache_root: Path,
    artifact_root: Path,
    snapshot_root: Path,
) -> dict[str, Any]:
    return verify_preflight_report(
        path,
        expected_campaign_spec_sha256=campaign_spec_sha256,
        expected_dataset_root=dataset_root,
        expected_cache_root=cache_root,
        expected_artifact_root=artifact_root,
        expected_snapshot_root=snapshot_root,
    )


def _phase2_binding(
    args: argparse.Namespace,
    preflight: dict[str, Any] | None,
) -> tuple[Path | None, str | None]:
    path = args.phase2_input
    digest = args.phase2_input_sha256
    if preflight is None:
        return path, digest
    record = preflight.get("phase2_input")
    if isinstance(record, dict):
        reported_path = record.get("path")
        reported_digest = record.get("sha256")
        if path is None and isinstance(reported_path, str):
            path = Path(reported_path)
        if digest is None and isinstance(reported_digest, str):
            digest = reported_digest
        if path is not None and isinstance(reported_path, str):
            if path.resolve() != Path(reported_path).resolve():
                raise ValueError("Phase 2 input path conflicts with preflight report")
        if digest is not None and isinstance(reported_digest, str):
            if digest.lower() != reported_digest.lower():
                raise ValueError("Phase 2 input hash conflicts with preflight report")
    return path, digest


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        spec = load_campaign_spec(args.spec)
        dataset_root = args.dataset_root
        if dataset_root is None:
            dataset_root = _REPO_ROOT / str(spec.raw["dataset_root"])
        preflight = (
            None
            if args.dry_run
            else _read_preflight(
                args.preflight_report,
                campaign_spec_sha256=spec.content_hash,
                dataset_root=dataset_root,
                cache_root=args.cache_root,
                artifact_root=args.artifact_root,
                snapshot_root=args.snapshot_root,
            )
        )
        phase2_input, phase2_input_sha256 = _phase2_binding(args, preflight)
        paths = CampaignPaths(
            artifact_root=args.artifact_root,
            cache_root=args.cache_root,
            dataset_root=dataset_root,
            snapshot_root=args.snapshot_root,
            preflight_report=args.preflight_report,
            phase2_input=phase2_input,
            phase2_input_sha256=phase2_input_sha256,
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
    except (CampaignError, PreflightError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
