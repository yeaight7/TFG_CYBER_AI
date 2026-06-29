#!/usr/bin/env python
"""
Verify the fixed CICIDS2017 test partition for training-size benchmark runs.

Load-only (NO training). Checks that:
  1. The full random split (preset=full, seed=42) reproduces the main-run
     test partition counts (n_test=566149, benign=454620, attack=111529).
  2. Train-only subsampling (train_max_rows) is nested, stratified, and
     leaves the test partition byte-identical (same content SHA-256).
  3. (--check-scaler) A StandardScaler fit on the reproduced full X_train
     matches the committed main-run scaler.joblib -> positive proof that the
     reproduced split equals the original run's split across environments.
  4. (--write-reference) Mints the committed reference manifest with counts,
     hashes, and library versions.

Examples:
    python scripts/verify_fixed_test_split.py --sizes 500000 1000000 --seed 42
    python scripts/verify_fixed_test_split.py --sizes 500000 1000000 --seed 42 \
        --check-scaler runs/cicids2017/MAIN_qrdqn_cicids2017_canonical_full_random_20260609_193655/scaler.joblib \
        --write-reference runs/cicids2017/test_partition_reference_seed42.json
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from src.load_cicids2017 import (  # noqa: E402
    SUBSAMPLE_METHOD_STRATIFIED_NESTED_PREFIX,
    _sha256_of_array,
    _stratified_nested_prefix_indices,
    load_cicids2017_split,
)
from src.artifact_integrity import resolve_trusted_artifact  # noqa: E402

# Committed main-run reference values (MAIN_qrdqn_cicids2017_canonical_full_random_20260609_193655)
MAIN_RUN_ID = "MAIN_qrdqn_cicids2017_canonical_full_random_20260609_193655"
EXPECTED = {
    "n_train": 2_264_594,
    "n_test": 566_149,
    "test_benign": 454_620,
    "test_attack": 111_529,
}

LOADER_ARGS = dict(
    split_mode="random",
    preset="full",
    seed=42,
    scale=False,
    use_canonical=True,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify the fixed test partition for the training-size benchmark (load-only).",
    )
    parser.add_argument(
        "--sizes", type=int, nargs="+", default=[500_000, 1_000_000],
        help="train_max_rows sizes to verify (default: 500000 1000000)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42, must match the main run)",
    )
    parser.add_argument(
        "--check-scaler", type=Path, default=None,
        help=(
            "Path to a committed main-run scaler.joblib. Fits a StandardScaler "
            "on the reproduced full X_train and compares mean_/scale_. "
            f"E.g. runs/cicids2017/{MAIN_RUN_ID}/scaler.joblib"
        ),
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("runs") / "cicids2017" / MAIN_RUN_ID,
        help="Trusted run dir containing artifact_manifest.json for --check-scaler.",
    )
    parser.add_argument(
        "--write-reference", type=Path, default=None,
        help="Write the reference manifest JSON to this path (e.g. runs/cicids2017/test_partition_reference_seed42.json)",
    )
    parser.add_argument(
        "--skip-count-check", action="store_true",
        help="Skip the main-run count assertions (for non-reference seeds/datasets)",
    )
    parser.add_argument(
        "--allow-unsafe-artifacts", action="store_true",
        help="Allow --check-scaler without artifact_manifest.json hash verification.",
    )
    return parser.parse_args()


def fail(msg: str) -> None:
    print(f"[FAIL] {msg}")
    sys.exit(1)


def ok(msg: str) -> None:
    print(f"[OK]   {msg}")


def main() -> None:
    args = parse_args()
    loader_args = dict(LOADER_ARGS, seed=args.seed)

    print(f"Loading full split (seed={args.seed}, preset=full, scale=False)...")
    X_train, y_train, X_test, y_test, _, _, meta = load_cicids2017_split(**loader_args)

    # 1) Counts vs committed main run
    if not args.skip_count_check:
        for key, expected in EXPECTED.items():
            actual = meta[key]
            if actual != expected:
                fail(f"{key}: expected {expected} (main run), got {actual}")
        ok(f"counts match main run {MAIN_RUN_ID}: "
           f"n_train={meta['n_train']}, n_test={meta['n_test']}, "
           f"test_benign={meta['test_benign']}, test_attack={meta['test_attack']}")

    test_hash = meta["test_set_sha256"]
    y_test_hash = meta["y_test_sha256"]
    ok(f"test_set_sha256 = {test_hash}")
    ok(f"y_test_sha256   = {y_test_hash}")

    # 2) Nested stratified subsamples (in-memory, no reload per size)
    sizes = sorted(args.sizes)
    attack_ratio = float((y_train == 1).mean())
    prev: set = set()
    train_hashes: dict[int, str] = {}
    for n in sizes:
        idx = _stratified_nested_prefix_indices(y_train, n, args.seed)
        if len(idx) != n or len(np.unique(idx)) != n:
            fail(f"size {n}: indices not unique/complete")
        n_attack_expected = int(round(n * attack_ratio))
        n_attack = int((y_train[idx] == 1).sum())
        if n_attack != n_attack_expected:
            fail(f"size {n}: attack count {n_attack} != expected {n_attack_expected}")
        if not prev.issubset(set(idx.tolist())):
            fail(f"size {n}: not a superset of the previous size (nesting broken)")
        prev = set(idx.tolist())
        train_hashes[n] = _sha256_of_array(X_train[idx])
        ok(f"size {n}: stratified (attack={n_attack}, ratio={n_attack / n:.6f}), "
           f"nested, train_set_sha256={train_hashes[n][:16]}...")

    # 3) End-to-end: one full loader call with train_max_rows
    n_e2e = sizes[-1]
    print(f"\nEnd-to-end check: load_cicids2017_split(train_max_rows={n_e2e})...")
    _, y_tr2, _, _, _, _, meta2 = load_cicids2017_split(**loader_args, train_max_rows=n_e2e)
    if meta2["test_set_sha256"] != test_hash:
        fail("end-to-end: test_set_sha256 changed under train_max_rows")
    if meta2["train_set_sha256"] != train_hashes[n_e2e]:
        fail("end-to-end: train subset hash != helper-derived hash")
    if meta2["subsample_method"] != SUBSAMPLE_METHOD_STRATIFIED_NESTED_PREFIX:
        fail(f"end-to-end: unexpected subsample_method {meta2['subsample_method']}")
    if int(len(y_tr2)) != n_e2e or meta2["n_train_full"] != int(len(y_train)):
        fail("end-to-end: train sizes inconsistent")
    ok(f"end-to-end: test partition byte-identical under train_max_rows={n_e2e}")

    # 4) Scaler audit vs committed main-run artifact
    if args.check_scaler is not None:
        import joblib
        from sklearn.preprocessing import StandardScaler

        scaler_path = resolve_trusted_artifact(
            args.run_dir,
            "scaler",
            args.check_scaler,
            repo_root=_REPO_ROOT,
            allow_unsafe=args.allow_unsafe_artifacts,
        )
        print("\nScaler audit vs trusted scaler artifact...")
        reference = joblib.load(scaler_path)
        reproduced = StandardScaler().fit(X_train)
        if not np.allclose(reproduced.mean_, reference.mean_, rtol=1e-6):
            fail("scaler mean_ mismatch: reproduced split != main-run split")
        if not np.allclose(reproduced.scale_, reference.scale_, rtol=1e-6):
            fail("scaler scale_ mismatch: reproduced split != main-run split")
        ok("scaler mean_/scale_ match the committed main-run scaler "
           "-> reproduced split equals the original run's split")

    # 5) Reference manifest
    if args.write_reference is not None:
        import pandas
        import sklearn

        manifest = {
            "description": (
                "Fixed test partition reference for the internal CICIDS2017 "
                "training-size benchmark. Any benchmark run's "
                "split_metadata.test_set_sha256 must equal test_set_sha256 below."
            ),
            "source_run_id": MAIN_RUN_ID,
            "loader_args": {**loader_args, "max_rows": None, "train_max_rows": None},
            "n_train_full": int(meta["n_train"]),
            "n_test": int(meta["n_test"]),
            "test_benign": int(meta["test_benign"]),
            "test_attack": int(meta["test_attack"]),
            "test_set_sha256": test_hash,
            "y_test_sha256": y_test_hash,
            "subsample_method": SUBSAMPLE_METHOD_STRATIFIED_NESTED_PREFIX,
            "train_subset_sha256": {str(n): h for n, h in train_hashes.items()},
            "environment": {
                "python": sys.version.split()[0],
                "numpy": np.__version__,
                "pandas": pandas.__version__,
                "scikit_learn": sklearn.__version__,
            },
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        }
        args.write_reference.parent.mkdir(parents=True, exist_ok=True)
        args.write_reference.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        ok(f"reference manifest written: {args.write_reference}")

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
