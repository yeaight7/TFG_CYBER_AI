from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.campaign_export import (
    CampaignExportError,
    create_final_bundle,
    create_incremental_snapshot,
    verify_final_bundle,
    verify_incremental_snapshot,
)
from src.run_artifacts import (
    ArtifactManifestWriter,
    ArtifactRequirement,
    atomic_write_json,
)


def _campaign_dir(tmp_path: Path, *, complete: bool) -> Path:
    campaign_dir = tmp_path / "artifacts" / "campaign-test"
    attempt_dir = campaign_dir / "attempts" / "run-a" / "attempt-1"
    writer = ArtifactManifestWriter(
        attempt_dir,
        run_metadata={
            "campaign_id": "campaign-test",
            "logical_run_id": "run-a",
            "physical_run_id": "attempt-1",
            "attempt": 1,
            "split_seed": 42,
            "model_seed": 42,
        },
        requirements={"config": ArtifactRequirement("config.json")},
    )
    writer.start()
    atomic_write_json(attempt_dir / "config.json", {"value": 1})
    writer.complete()

    status = "completed" if complete else "pending"
    attempts = (
        [
            {
                "attempt": 1,
                "artifact_dir": "attempts/run-a/attempt-1",
                "status": "completed",
            }
        ]
        if complete
        else []
    )
    atomic_write_json(
        campaign_dir / "campaign_state.json",
        {
            "schema_version": "1.0",
            "campaign_id": "campaign-test",
            "campaign_spec_sha256": "a" * 64,
            "cache_manifest_sha256": "b" * 64,
            "preflight_report_sha256": "c" * 64,
            "entries": {
                "run-a": {
                    "classification": "primary_model_training",
                    "status": status,
                    "attempts": attempts,
                },
                "run-a-alias": {
                    "classification": "alias",
                    "status": "reused" if complete else "pending",
                    "attempts": [],
                    "reuse_of": "run-a",
                },
            },
        },
    )
    atomic_write_json(campaign_dir / "campaign_spec_original.json", {"source": "locked"})
    atomic_write_json(campaign_dir / "campaign_spec_resolved.json", {"resolved": True})
    atomic_write_json(campaign_dir / "preflight_report.json", {"status": "passed"})
    atomic_write_json(campaign_dir / "cache_manifest.json", {"validation_status": "valid"})
    return campaign_dir


def test_incremental_snapshot_is_hash_based_noop_and_detects_corruption(
    tmp_path: Path,
) -> None:
    campaign_dir = _campaign_dir(tmp_path, complete=False)
    destination = tmp_path / "durable" / "campaign-test"

    first = create_incremental_snapshot(campaign_dir, destination)
    assert first["status"] == "verified"
    assert "campaign_state.json" in first["copied_files"]
    assert "cache_manifest.json" in first["copied_files"]
    assert not any("observations.npy" in path for path in first["files"])
    verify_incremental_snapshot(destination)

    second = create_incremental_snapshot(campaign_dir, destination)
    assert second["status"] == "verified"
    assert second["copied_files"] == []
    assert second["unchanged_files"] == second["files"]

    state_path = campaign_dir / "campaign_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["updated_at"] = "2026-07-12T12:00:00+00:00"
    atomic_write_json(state_path, state)
    third = create_incremental_snapshot(campaign_dir, destination)
    assert third["copied_files"] == ["campaign_state.json"]

    (destination / "campaign_state.json").write_text("corrupt", encoding="utf-8")
    with pytest.raises(CampaignExportError, match="checksum"):
        verify_incremental_snapshot(destination)


def test_final_bundle_is_deterministic_and_verified(tmp_path: Path) -> None:
    campaign_dir = _campaign_dir(tmp_path, complete=True)
    first_path = tmp_path / "first.tar.gz"
    second_path = tmp_path / "second.tar.gz"

    first = create_final_bundle(campaign_dir, first_path)
    second = create_final_bundle(campaign_dir, second_path)

    assert first["status"] == "verified"
    assert second["status"] == "verified"
    assert first["archive_sha256"] == second["archive_sha256"]
    assert first_path.read_bytes() == second_path.read_bytes()
    verified = verify_final_bundle(first_path)
    assert verified["archive_sha256"] == first["archive_sha256"]

    data = bytearray(first_path.read_bytes())
    data[len(data) // 2] ^= 0x01
    first_path.write_bytes(bytes(data))
    with pytest.raises(CampaignExportError):
        verify_final_bundle(first_path)


def test_final_bundle_refuses_incomplete_campaign(tmp_path: Path) -> None:
    campaign_dir = _campaign_dir(tmp_path, complete=False)

    with pytest.raises(CampaignExportError, match="incomplete"):
        create_final_bundle(campaign_dir, tmp_path / "campaign.tar.gz")


def test_export_cli_runs_snapshot_and_verified_bundle(tmp_path: Path, capsys) -> None:
    from scripts.export_campaign import main

    incomplete = _campaign_dir(tmp_path / "snapshot", complete=False)
    snapshot_destination = tmp_path / "snapshot-destination"
    assert (
        main(
            [
                "snapshot",
                "--campaign-dir",
                str(incomplete),
                "--destination",
                str(snapshot_destination),
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["status"] == "verified"

    complete = _campaign_dir(tmp_path / "bundle", complete=True)
    bundle_destination = tmp_path / "campaign.tar.gz"
    assert (
        main(
            [
                "bundle",
                "--campaign-dir",
                str(complete),
                "--destination",
                str(bundle_destination),
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["status"] == "verified"
