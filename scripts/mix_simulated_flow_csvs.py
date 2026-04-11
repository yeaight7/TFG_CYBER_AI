"""
mix_simulated_flow_csvs.py

Chunk-mix two simulated flow CSVs into a single output file while preserving
all rows and shuffling within each mixed batch.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterator, Optional

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mix benign and attack simulated flow CSVs.")
    parser.add_argument("--benign", type=Path, required=True, help="Benign simulated flow CSV.")
    parser.add_argument("--attack", type=Path, required=True, help="Attack simulated flow CSV.")
    parser.add_argument("--output", type=Path, required=True, help="Mixed output CSV.")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=150_000,
        help="Approximate output rows per mixed batch.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed used for per-batch shuffling.")
    return parser.parse_args()


def count_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        return max(sum(1 for _ in handle) - 1, 0)


def next_or_none(reader: Iterator[pd.DataFrame]) -> Optional[pd.DataFrame]:
    try:
        return next(reader)
    except StopIteration:
        return None


def write_provenance(output_path: Path, info: Dict[str, object]) -> Path:
    meta_path = output_path.with_suffix(output_path.suffix + ".meta.json")
    meta_path.write_text(json.dumps(info, indent=2), encoding="utf-8")
    return meta_path


def main() -> None:
    args = parse_args()
    benign_rows = count_rows(args.benign)
    attack_rows = count_rows(args.attack)
    total_rows = benign_rows + attack_rows

    if benign_rows == 0 or attack_rows == 0:
        raise ValueError("Both input CSVs must contain at least one data row.")

    benign_share = benign_rows / total_rows
    benign_chunk_rows = max(1, int(round(args.chunk_size * benign_share)))
    attack_chunk_rows = max(1, args.chunk_size - benign_chunk_rows)

    benign_iter = pd.read_csv(args.benign, chunksize=benign_chunk_rows)
    attack_iter = pd.read_csv(args.attack, chunksize=attack_chunk_rows)

    benign_chunk = next_or_none(benign_iter)
    attack_chunk = next_or_none(attack_iter)
    first = True
    batch_idx = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)

    while benign_chunk is not None or attack_chunk is not None:
        frames = [frame for frame in (benign_chunk, attack_chunk) if frame is not None and not frame.empty]
        if not frames:
            break

        mixed = pd.concat(frames, ignore_index=True)
        mixed = mixed.sample(frac=1.0, random_state=args.seed + batch_idx).reset_index(drop=True)
        mixed.to_csv(args.output, mode="w" if first else "a", header=first, index=False)

        first = False
        batch_idx += 1
        benign_chunk = next_or_none(benign_iter)
        attack_chunk = next_or_none(attack_iter)

    meta_path = write_provenance(
        args.output,
        {
            "artifact_type": "mixed_simulated_flow_csv",
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "benign_input": str(args.benign),
            "attack_input": str(args.attack),
            "output_csv": str(args.output),
            "benign_rows": benign_rows,
            "attack_rows": attack_rows,
            "total_rows": total_rows,
            "chunk_size": args.chunk_size,
            "seed": args.seed,
        },
    )

    print(f"[mix] benign rows   : {benign_rows}")
    print(f"[mix] attack rows   : {attack_rows}")
    print(f"[mix] total rows    : {total_rows}")
    print(f"[mix] output        : {args.output}")
    print(f"[mix] provenance    : {meta_path}")


if __name__ == "__main__":
    main()
