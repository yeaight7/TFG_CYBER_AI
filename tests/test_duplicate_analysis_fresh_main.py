from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from scripts.analyze_duplicates import run_duplicate_analysis
from src.artifact_integrity import verify_artifact_manifest
from src.qrdqn_experiment import PreparedSplit


def test_duplicate_analysis_accepts_matching_fresh_split(
    tmp_path: Path,
    fresh_main_run: Path,
    synthetic_split: PreparedSplit,
):
    output_dir = tmp_path / "duplicates"
    run_duplicate_analysis(
        source_run_dir=fresh_main_run,
        output_dir=output_dir,
        split_provider=lambda _config: synthetic_split,
    )

    assert verify_artifact_manifest(output_dir)["status"] == "completed"
    result = json.loads((output_dir / "duplicate_analysis.json").read_text(encoding="utf-8"))
    assert result["source_run_id"] == fresh_main_run.name
    assert result["verified_split_hashes"]["test_set_sha256"]


def test_duplicate_analysis_rejects_mismatched_fresh_split_hashes(
    tmp_path: Path,
    fresh_main_run: Path,
    synthetic_split: PreparedSplit,
):
    changed = synthetic_split.X_test.copy()
    changed[0, 0] += 1
    mismatched = replace(synthetic_split, X_test=changed)
    output_dir = tmp_path / "duplicates-mismatch"

    with pytest.raises(ValueError, match="test_set_sha256"):
        run_duplicate_analysis(
            source_run_dir=fresh_main_run,
            output_dir=output_dir,
            split_provider=lambda _config: mismatched,
        )

    assert verify_artifact_manifest(output_dir, require_completed=False)["status"] == "failed"
