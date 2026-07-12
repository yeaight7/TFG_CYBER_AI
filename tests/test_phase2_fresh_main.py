from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts.predict_real_traffic_v2 import run_phase2_inference
from src.artifact_integrity import sha256_file, verify_artifact_manifest


def test_phase2_inference_records_fresh_main_and_input_provenance(
    tmp_path: Path,
    fresh_main_run: Path,
    fake_model_factory,
):
    flows_path = tmp_path / "synthetic_flows.csv"
    pd.DataFrame(
        {
            "flow_duration": [1_000.0, 2_000.0],
            "truth_y": [0, 1],
            "src_ip": ["10.0.0.1", "10.0.0.2"],
        }
    ).to_csv(flows_path, index=False)
    output_dir = tmp_path / "phase2"

    run_phase2_inference(
        source_run_dir=fresh_main_run,
        flows_path=flows_path,
        output_dir=output_dir,
        clip_z=10.0,
        monitor_interval=0.01,
        model_loader=lambda _path: (
            fake_model_factory(None, None, output_dir / "unused", "cpu"),
            "FakeQRDQN",
        ),
    )

    assert verify_artifact_manifest(output_dir)["status"] == "completed"
    config = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))
    source_manifest_sha256 = sha256_file(fresh_main_run / "artifact_manifest.json")
    assert config["source_run_id"] == fresh_main_run.name
    assert config["source_manifest_sha256"] == source_manifest_sha256
    assert config["source_artifact_sha256"]["manifest"] == source_manifest_sha256
    assert config["input"]["sha256"] == sha256_file(flows_path)
    assert config["input"]["filename"] == flows_path.name
    assert config["input"]["size_bytes"] == flows_path.stat().st_size
    for key in ("model", "scaler", "train_percentiles", "feature_names", "manifest"):
        assert len(config["source_artifact_sha256"][key]) == 64
    assert config["sensitive_metadata_exported"] is False
    assert (output_dir / "diagnostics.json").is_file()
    assert (output_dir / "predictions.npz").is_file()
