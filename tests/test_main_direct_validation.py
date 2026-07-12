from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.validate_main_direct import run_main_direct_validation
from src.artifact_integrity import verify_artifact_manifest
from src.qrdqn_experiment import PreparedSplit
from tests.conftest import FakeQRDQN


def test_direct_validation_uses_independent_truth_and_writes_valid_artifacts(
    tmp_path: Path,
    fresh_main_run: Path,
    synthetic_split: PreparedSplit,
):
    output_dir = tmp_path / "direct-validation"

    run_main_direct_validation(
        source_run_dir=fresh_main_run,
        output_dir=output_dir,
        split_provider=lambda _source_config: synthetic_split,
        model_loader=lambda _path: FakeQRDQN(output_dir / "unused"),
        eval_batch_size=2,
    )

    assert verify_artifact_manifest(output_dir)["status"] == "completed"
    result = json.loads((output_dir / "validation_results.json").read_text(encoding="utf-8"))
    predictions = np.load(output_dir / "predictions.npz")
    np.testing.assert_array_equal(predictions["y_true"], synthetic_split.y_test)
    assert result["evaluation_basis"] == "direct_predictions_against_reproduced_test_labels"
    assert result["source_run_id"] == fresh_main_run.name
    assert len(result["source_manifest_sha256"]) == 64
    assert result["confusion_matrix"] == {"tn": 2, "fp": 0, "fn": 0, "tp": 2}
