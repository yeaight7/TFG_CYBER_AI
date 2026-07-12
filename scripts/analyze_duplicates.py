"""Fresh-MAIN duplicate and cross-split analysis without training.

The script reproduces the canonical unscaled partition declared by a completed
schema-3 ``main-v1`` run, rejects any feature or label hash mismatch, then
measures exact duplicates using byte-level row views. Output is a separate
schema-3 job bound to the source manifest, dataset, cache, and split hashes.

Example:
    python scripts/analyze_duplicates.py --run-dir <FRESH_MAIN_RUN> \
        --output-dir <DUPLICATE_JOB_DIR>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from src.artifact_integrity import sha256_file, verify_artifact_manifest  # noqa: E402
from src.cicids_cache import sha256_array  # noqa: E402
from src.qrdqn_experiment import (  # noqa: E402
    PreparedSplit,
    config_from_run_payload,
    load_experiment_split,
)
from src.run_artifacts import (  # noqa: E402
    ArtifactManifestWriter,
    ArtifactRequirement,
    atomic_write_json,
)


def _void_rows(X: np.ndarray) -> np.ndarray:
    """View a 2-D float32 array as a 1-D array of byte-exact per-row records."""
    Xc = np.ascontiguousarray(X, dtype=np.float32)
    return Xc.view(np.dtype((np.void, Xc.shape[1] * Xc.itemsize))).ravel()


def _dup_stats(void_rows: np.ndarray) -> dict:
    n_total = int(void_rows.shape[0])
    n_unique = int(np.unique(void_rows).shape[0])
    n_dup = n_total - n_unique
    return {
        "n_rows": n_total,
        "n_unique": n_unique,
        "n_duplicate_rows": n_dup,
        "duplicate_pct": round(100.0 * n_dup / n_total, 4) if n_total else 0.0,
    }


def _pct(n: int, d: int) -> float:
    return round(100.0 * n / d, 4) if d else 0.0


SplitProvider = Callable[[dict[str, Any]], PreparedSplit]


def _default_split_provider(source_config: dict[str, Any]) -> PreparedSplit:
    return load_experiment_split(config_from_run_payload(source_config))


def run_duplicate_analysis(
    *,
    source_run_dir: Path,
    output_dir: Path,
    split_provider: SplitProvider = _default_split_provider,
) -> Path:
    """Verify the fresh MAIN raw split identity, then quantify exact duplicates."""
    source_run_dir = Path(source_run_dir)
    output_dir = Path(output_dir)
    verification = verify_artifact_manifest(source_run_dir)
    if verification["schema_version"] != "3.0":
        raise ValueError("Fresh MAIN duplicate analysis requires a schema-3 source run")
    source_config = json.loads(
        (source_run_dir / "config.json").read_text(encoding="utf-8")
    )
    if source_config.get("profile_id") != "main-v1":
        raise ValueError("Fresh MAIN duplicate analysis requires profile_id='main-v1'")
    source_manifest_sha256 = sha256_file(source_run_dir / "artifact_manifest.json")
    writer = ArtifactManifestWriter(
        output_dir,
        run_metadata={
            "logical_run_id": output_dir.name,
            "physical_run_id": output_dir.name,
            "attempt": 1,
            "split_seed": source_config["split_seed"],
            "model_seed": source_config["model_seed"],
            "source_run_id": source_config["run_id"],
            "source_manifest_sha256": source_manifest_sha256,
        },
        requirements={
            "config": ArtifactRequirement("config.json"),
            "duplicate_analysis": ArtifactRequirement("duplicate_analysis.json"),
        },
    )
    writer.start()
    try:
        split = split_provider(source_config)
        expected = source_config["split_metadata"]
        actual = {
            "train_set_sha256": sha256_array(split.X_train),
            "y_train_sha256": sha256_array(split.y_train),
            "test_set_sha256": sha256_array(split.X_test),
            "y_test_sha256": sha256_array(split.y_test),
        }
        for key, value in actual.items():
            if value != expected.get(key):
                raise ValueError(f"Reproduced {key} does not match fresh MAIN source metadata")

        X_train = split.X_train
        y_train = split.y_train
        X_test = split.X_test
        y_test = split.y_test
        n_features = int(X_train.shape[1])
        y_all = np.concatenate([y_train, y_test])
        tr_feat = _void_rows(X_train)
        te_feat = _void_rows(X_test)
        all_feat = np.concatenate([tr_feat, te_feat])

        Xtr_l = np.hstack([X_train, y_train.reshape(-1, 1).astype(np.float32)])
        Xte_l = np.hstack([X_test, y_test.reshape(-1, 1).astype(np.float32)])
        tr_fl = _void_rows(Xtr_l)
        te_fl = _void_rows(Xte_l)
        all_fl = np.concatenate([tr_fl, te_fl])

        overall_feat = _dup_stats(all_feat)
        overall_fl = _dup_stats(all_fl)
        tr_feat_u = np.unique(tr_feat)
        test_in_train_feat = np.isin(te_feat, tr_feat_u)
        tr_fl_u = np.unique(tr_fl)
        test_in_train_fl = np.isin(te_fl, tr_fl_u)

        n_test = int(len(y_test))
        attack_mask = y_test == 1
        benign_mask = y_test == 0
        n_leak_feat = int(test_in_train_feat.sum())
        n_leak_fl = int(test_in_train_fl.sum())
        leak_feat_attack = int((test_in_train_feat & attack_mask).sum())
        leak_feat_benign = int((test_in_train_feat & benign_mask).sum())
        dup_attack = _dup_stats(all_feat[y_all == 1])
        dup_benign = _dup_stats(all_feat[y_all == 0])

        summary = {
            "job_type": "fresh_main_duplicate_cross_split_analysis",
            "source_run_id": source_config["run_id"],
            "source_manifest_sha256": source_manifest_sha256,
            "source_cache_manifest_sha256": expected.get("cache_manifest_sha256"),
            "source_dataset_sha256": expected.get("source_csv_sha256", {}),
            "verified_split_hashes": actual,
            "config": {
                "split_seed": source_config["split_seed"],
                "n_features": n_features,
            },
            "counts": {
                "n_train": int(len(y_train)),
                "n_test": n_test,
                "n_total": int(len(y_all)),
            },
            "overall_duplicates_feature_only": overall_feat,
            "overall_duplicates_feature_plus_label": overall_fl,
            "cross_split_leakage_feature_only": {
                "n_test_rows_also_in_train": n_leak_feat,
                "pct_of_test": _pct(n_leak_feat, n_test),
                "attack_test_rows_in_train": leak_feat_attack,
                "attack_pct_of_test_attacks": _pct(
                    leak_feat_attack, int(attack_mask.sum())
                ),
                "benign_test_rows_in_train": leak_feat_benign,
                "benign_pct_of_test_benigns": _pct(
                    leak_feat_benign, int(benign_mask.sum())
                ),
            },
            "cross_split_leakage_feature_plus_label": {
                "n_test_rows_also_in_train": n_leak_fl,
                "pct_of_test": _pct(n_leak_fl, n_test),
            },
            "per_class_duplicate_rate_full_set_feature_only": {
                "attack": dup_attack,
                "benign": dup_benign,
            },
        }
        atomic_write_json(
            output_dir / "config.json",
            {
                "job_type": summary["job_type"],
                "source_run_id": source_config["run_id"],
                "source_manifest_sha256": source_manifest_sha256,
                "source_cache_manifest_sha256": expected.get("cache_manifest_sha256"),
                "source_dataset_sha256": expected.get("source_csv_sha256", {}),
                "split_seed": source_config["split_seed"],
            },
        )
        atomic_write_json(output_dir / "duplicate_analysis.json", summary)
        writer.complete()
        return output_dir
    except BaseException as error:
        writer.fail(error)
        raise


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Duplicate/cross-split analysis for a fresh schema-3 MAIN run"
    )
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    run_duplicate_analysis(source_run_dir=args.run_dir, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
