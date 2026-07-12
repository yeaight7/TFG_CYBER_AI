from __future__ import annotations

import json
from pathlib import Path

from scripts.bootstrap_ci import run_bootstrap_from_predictions
from src.artifact_integrity import sha256_file, verify_artifact_manifest


def test_bootstrap_is_bound_to_fresh_main_prediction_hash(
    tmp_path: Path,
    fresh_main_run: Path,
):
    output_dir = tmp_path / "bootstrap"

    run_bootstrap_from_predictions(
        source_run_dir=fresh_main_run,
        output_dir=output_dir,
        n_boot=100,
        boot_seed=12345,
    )

    assert verify_artifact_manifest(output_dir)["status"] == "completed"
    payload = json.loads((output_dir / "bootstrap_ci.json").read_text(encoding="utf-8"))
    assert payload["source_run_id"] == fresh_main_run.name
    assert payload["source_predictions_sha256"] == sha256_file(
        fresh_main_run / "predictions.npz"
    )
    assert payload["n_boot"] == 100
