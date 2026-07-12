"""Provider-neutral snapshot and final-bundle CLI for campaign evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:  # pragma: no cover - direct script execution
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.campaign_export import (  # noqa: E402
    CampaignExportError,
    create_final_bundle,
    create_incremental_snapshot,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create and verify provider-neutral campaign evidence exports."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("snapshot", "bundle"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--campaign-dir", required=True, type=Path)
        subparser.add_argument("--destination", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.command == "snapshot":
            result = create_incremental_snapshot(args.campaign_dir, args.destination)
        else:
            result = create_final_bundle(args.campaign_dir, args.destination)
    except CampaignExportError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
