"""Validate and aggregate one completed final-experiment campaign."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:  # pragma: no cover - direct script execution
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.campaign_aggregation import (  # noqa: E402
    CampaignAggregationError,
    aggregate_campaign,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate a complete final-experiment campaign and write "
            "provenance-preserving JSON/CSV aggregates."
        )
    )
    parser.add_argument("--campaign-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = aggregate_campaign(args.campaign_dir, args.output_dir)
    except CampaignAggregationError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
