"""Build or validate canonical unscaled CICIDS2017 cache shards."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.cicids_cache import (  # noqa: E402
    CacheValidationError,
    build_cache,
    default_worker_count,
    validate_cache,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build or validate provider-neutral canonical unscaled CICIDS2017 cache shards."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="Build missing valid shards.")
    build.add_argument("--dataset-root", type=Path, required=True)
    build.add_argument("--cache-root", type=Path, required=True)
    build.add_argument("--workers", type=int, default=default_worker_count())
    build.add_argument(
        "--rebuild-stale",
        action="store_true",
        help="Explicitly replace stale, incompatible, or corrupt shards.",
    )

    validate = subparsers.add_parser("validate", help="Validate every shard and source hash.")
    validate.add_argument("--dataset-root", type=Path, required=True)
    validate.add_argument("--cache-root", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "build":
            manifest = build_cache(
                dataset_root=args.dataset_root,
                cache_root=args.cache_root,
                workers=args.workers,
                rebuild_stale=args.rebuild_stale,
            )
        else:
            manifest = validate_cache(
                dataset_root=args.dataset_root,
                cache_root=args.cache_root,
            )
    except (CacheValidationError, FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))

    print(
        json.dumps(
            {
                "status": manifest["validation_status"],
                "cache_root": str(args.cache_root),
                "shards": len(manifest["shards"]),
                "cache_schema_version": manifest["cache_schema_version"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
