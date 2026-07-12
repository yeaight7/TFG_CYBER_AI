from __future__ import annotations

import importlib
import csv
import json
import os
import sys
from pathlib import Path

import pytest


def _module(name: str):
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError:
        pytest.fail(f"Phase 3 module {name!r} is missing", pytrace=False)


def _api(module_name: str, name: str):
    module = _module(module_name)
    assert hasattr(module, name), f"{module_name}.{name} is missing"
    return getattr(module, name)


def _requirements():
    requirement = _api("src.run_artifacts", "ArtifactRequirement")
    return {
        "config": requirement("config.json"),
        "metrics": requirement("metrics.json"),
        "tensorboard": requirement("tensorboard", kind="directory"),
    }


def _minimal_run_metadata():
    return {
        "physical_run_id": "attempt-1",
        "split_seed": None,
        "model_seed": None,
    }


def _completed_run(tmp_path: Path) -> tuple[Path, object]:
    writer_type = _api("src.run_artifacts", "ArtifactManifestWriter")
    run_dir = tmp_path / "campaign" / "attempt-1"
    writer = writer_type(
        run_dir,
        run_metadata={
            "campaign_id": "campaign-test",
            "logical_run_id": "logical-test",
            "physical_run_id": "physical-test-attempt-1",
            "attempt": 1,
            "split_seed": 42,
            "model_seed": 43,
            "cache_manifest_sha256": "a" * 64,
        },
        requirements=_requirements(),
    )
    writer.start()
    (run_dir / "config.json").write_text('{"status": "completed"}\n', encoding="utf-8")
    (run_dir / "metrics.json").write_text('{"accuracy": null}\n', encoding="utf-8")
    tensorboard_dir = run_dir / "tensorboard"
    tensorboard_dir.mkdir()
    (tensorboard_dir / "events.out.tfevents.test").write_bytes(b"event-data")
    writer.complete()
    return run_dir, writer


def test_schema3_manifest_declares_and_verifies_complete_artifacts(tmp_path):
    verify = _api("src.artifact_integrity", "verify_artifact_manifest")
    run_dir, _writer = _completed_run(tmp_path)

    result = verify(run_dir)
    manifest = json.loads((run_dir / "artifact_manifest.json").read_text(encoding="utf-8"))

    assert result["schema_version"] == "3.0"
    assert result["status"] == "completed"
    assert manifest["run"]["split_seed"] == 42
    assert manifest["run"]["model_seed"] == 43
    assert manifest["path_base"] == "run_dir"
    assert manifest["file_artifacts"]["config"]["relative_path"] == "config.json"
    assert manifest["inventory"]["tensorboard/events.out.tfevents.test"]["sha256"]

    sums = (run_dir / "SHA256SUMS").read_text(encoding="utf-8")
    assert "  config.json\n" in sums
    assert "  tensorboard/events.out.tfevents.test\n" in sums


def test_schema3_completion_rejects_missing_required_artifact(tmp_path):
    writer_type = _api("src.run_artifacts", "ArtifactManifestWriter")
    validation_error = _api("src.run_artifacts", "ArtifactValidationError")
    requirement = _api("src.run_artifacts", "ArtifactRequirement")
    writer = writer_type(
        tmp_path / "run",
        run_metadata=_minimal_run_metadata(),
        requirements={"config": requirement("config.json")},
    )
    writer.start()

    with pytest.raises(validation_error, match="config"):
        writer.complete()


def test_schema3_completion_rejects_empty_required_directory(tmp_path):
    writer_type = _api("src.run_artifacts", "ArtifactManifestWriter")
    validation_error = _api("src.run_artifacts", "ArtifactValidationError")
    requirement = _api("src.run_artifacts", "ArtifactRequirement")
    run_dir = tmp_path / "run"
    writer = writer_type(
        run_dir,
        run_metadata=_minimal_run_metadata(),
        requirements={"tensorboard": requirement("tensorboard", kind="directory")},
    )
    writer.start()
    (run_dir / "tensorboard").mkdir()

    with pytest.raises(validation_error, match="tensorboard"):
        writer.complete()


def test_schema3_verifier_detects_checksum_corruption(tmp_path):
    verify = _api("src.artifact_integrity", "verify_artifact_manifest")
    trust_error = _api("src.artifact_integrity", "ArtifactTrustError")
    run_dir, _writer = _completed_run(tmp_path)
    (run_dir / "metrics.json").write_text('{"accuracy": 1.00}\n', encoding="utf-8")

    with pytest.raises(trust_error, match="SHA-256 mismatch"):
        verify(run_dir)


def test_completed_schema3_manifest_is_immutable(tmp_path):
    state_error = _api("src.run_artifacts", "ArtifactStateError")
    _run_dir, writer = _completed_run(tmp_path)

    with pytest.raises(state_error, match="immutable"):
        writer.set_status("failed")
    with pytest.raises(state_error, match="immutable"):
        writer.complete()


def test_failed_schema3_attempt_writes_error_and_is_immutable(tmp_path):
    writer_type = _api("src.run_artifacts", "ArtifactManifestWriter")
    state_error = _api("src.run_artifacts", "ArtifactStateError")
    requirement = _api("src.run_artifacts", "ArtifactRequirement")
    verify = _api("src.artifact_integrity", "verify_artifact_manifest")
    run_dir = tmp_path / "run"
    writer = writer_type(
        run_dir,
        run_metadata=_minimal_run_metadata(),
        requirements={"config": requirement("config.json")},
    )
    writer.start()
    (run_dir / "config.json").write_text("{}\n", encoding="utf-8")

    writer.fail(RuntimeError("training failed"))

    error = json.loads((run_dir / "error.json").read_text(encoding="utf-8"))
    manifest = json.loads((run_dir / "artifact_manifest.json").read_text(encoding="utf-8"))
    assert error["type"] == "RuntimeError"
    assert error["message"] == "training failed"
    assert manifest["status"] == "failed"
    assert verify(run_dir, require_completed=False) == {
        "schema_version": "3.0",
        "status": "failed",
        "verified_files": 2,
    }
    with pytest.raises(state_error, match="immutable"):
        writer.complete()
    with pytest.raises(state_error, match="immutable"):
        writer.start()


def test_schema3_verifier_rejects_file_artifact_inventory_disagreement(tmp_path):
    verify = _api("src.artifact_integrity", "verify_artifact_manifest")
    trust_error = _api("src.artifact_integrity", "ArtifactTrustError")
    run_dir, _writer = _completed_run(tmp_path)
    manifest_path = run_dir / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["file_artifacts"]["config"]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(trust_error, match="file_artifacts"):
        verify(run_dir)


def test_schema3_writer_requires_explicit_seed_fields(tmp_path):
    writer_type = _api("src.run_artifacts", "ArtifactManifestWriter")
    requirement = _api("src.run_artifacts", "ArtifactRequirement")

    with pytest.raises(ValueError, match="model_seed.*split_seed|split_seed.*model_seed"):
        writer_type(
            tmp_path / "run",
            run_metadata={"physical_run_id": "attempt-1"},
            requirements={"config": requirement("config.json")},
        )


def test_schema3_running_manifest_identity_cannot_change_before_completion(tmp_path):
    writer_type = _api("src.run_artifacts", "ArtifactManifestWriter")
    state_error = _api("src.run_artifacts", "ArtifactStateError")
    requirement = _api("src.run_artifacts", "ArtifactRequirement")
    run_dir = tmp_path / "run"
    writer = writer_type(
        run_dir,
        run_metadata=_minimal_run_metadata(),
        requirements={"config": requirement("config.json")},
    )
    writer.start()
    (run_dir / "config.json").write_text("{}\n", encoding="utf-8")
    manifest_path = run_dir / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["run"]["model_seed"] = 999
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(state_error, match="identity"):
        writer.complete()


def test_atomic_json_failure_preserves_previous_file(tmp_path, monkeypatch):
    atomic_write_json = _api("src.run_artifacts", "atomic_write_json")
    module = _module("src.run_artifacts")
    destination = tmp_path / "state.json"
    destination.write_text('{"status": "running"}\n', encoding="utf-8")

    def fail_replace(_source, _destination):
        raise OSError("simulated replace failure")

    monkeypatch.setattr(module.os, "replace", fail_replace)

    with pytest.raises(OSError, match="replace failure"):
        atomic_write_json(destination, {"status": "completed"})

    assert json.loads(destination.read_text(encoding="utf-8")) == {"status": "running"}
    assert not list(tmp_path.glob(".state.json.tmp-*"))


def test_schema2_manifest_remains_verifiable(tmp_path):
    build_file_artifacts = _api("src.artifact_integrity", "build_file_artifacts")
    verify = _api("src.artifact_integrity", "verify_artifact_manifest")
    run_dir = tmp_path / "runs" / "historical"
    run_dir.mkdir(parents=True)
    artifact = run_dir / "config.json"
    artifact.write_text('{"legacy": true}\n', encoding="utf-8")
    manifest = {
        "schema_version": "2.0",
        "run_id": "historical",
        "status": "completed",
        "file_artifacts": build_file_artifacts({"config": artifact}, repo_root=tmp_path),
    }
    (run_dir / "artifact_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    result = verify(run_dir, repo_root=tmp_path)

    assert result == {
        "schema_version": "2.0",
        "status": "completed",
        "verified_files": 1,
    }


def test_schema3_trusted_artifacts_resolve_relative_to_run_dir(tmp_path):
    resolve = _api("src.artifact_integrity", "resolve_trusted_artifact")
    run_dir, _writer = _completed_run(tmp_path)

    resolved = resolve(run_dir, "config", repo_root=tmp_path / "different-repo")

    assert resolved == (run_dir / "config.json").resolve()


def test_schema3_requested_relative_path_is_run_relative(tmp_path):
    resolve = _api("src.artifact_integrity", "resolve_trusted_artifact")
    run_dir, _writer = _completed_run(tmp_path)

    resolved = resolve(
        run_dir,
        "config",
        requested_path=Path("config.json"),
        repo_root=tmp_path / "different-repo",
    )

    assert resolved == (run_dir / "config.json").resolve()


def test_tee_output_preserves_console_and_writes_logs(tmp_path, capsys):
    tee_output = _api("src.run_artifacts", "tee_output")
    stdout_path = tmp_path / "stdout.log"
    stderr_path = tmp_path / "stderr.log"

    with tee_output(stdout_path, stderr_path):
        print("standard output")
        print("standard error", file=sys.stderr)

    captured = capsys.readouterr()
    assert "standard output" in captured.out
    assert "standard error" in captured.err
    assert stdout_path.read_text(encoding="utf-8") == "standard output\n"
    assert stderr_path.read_text(encoding="utf-8") == "standard error\n"


def test_timing_recorder_reports_throughput_and_writes_atomically(tmp_path):
    recorder_type = _api("src.run_artifacts", "TimingRecorder")
    ticks = iter([10.0, 12.0])
    recorder = recorder_type(clock=lambda: next(ticks))

    with recorder.measure("preprocessing") as measurement:
        measurement.set_units(1_000, "rows")

    output = tmp_path / "timing.json"
    recorder.write(output)
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload["phases"]["preprocessing"]["duration_seconds"] == 2.0
    assert payload["phases"]["preprocessing"]["units"] == 1_000
    assert payload["phases"]["preprocessing"]["throughput_per_second"] == 500.0
    assert payload["phases"]["preprocessing"]["unit"] == "rows"


def test_environment_metadata_records_git_packages_cuda_and_thread_settings(tmp_path):
    collect = _api("src.run_artifacts", "collect_environment_metadata")

    metadata = collect(
        repo_root=tmp_path,
        requested_torch_threads=8,
        requested_torch_interop_threads=1,
        package_names=("numpy", "package-that-does-not-exist"),
    )

    assert metadata["git"]["commit"] is None
    assert metadata["git"]["dirty"] is None
    assert metadata["packages"]["numpy"]
    assert metadata["packages"]["package-that-does-not-exist"] is None
    assert metadata["python"]["version"]
    assert "cuda_available" in metadata["torch"]
    assert metadata["threads"]["requested"]["torch_intra_op"] == 8
    assert metadata["threads"]["requested"]["torch_inter_op"] == 1
    assert set(metadata["threads"]["environment"]) == {
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
    }


def test_checkpoint_retention_is_bounded_and_checksum_backed(tmp_path):
    retain = _api("src.run_artifacts", "retain_checkpoints")
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    paths = []
    for index, name in enumerate(("step-100.zip", "step-200.zip", "step-300.zip"), start=1):
        path = checkpoints / name
        path.write_text(name, encoding="utf-8")
        os.utime(path, (index, index))
        paths.append(path)

    result = retain(checkpoints, keep=2)
    inventory = json.loads((checkpoints / "checkpoint_inventory.json").read_text(encoding="utf-8"))

    assert sorted(path.name for path in checkpoints.glob("*.zip")) == [
        "step-200.zip",
        "step-300.zip",
    ]
    assert [item["filename"] for item in result] == ["step-200.zip", "step-300.zip"]
    assert inventory["status"] == "completed"
    assert [item["filename"] for item in inventory["retained"]] == [
        "step-200.zip",
        "step-300.zip",
    ]
    assert inventory["removed"][0]["filename"] == "step-100.zip"
    assert all(len(item["sha256"]) == 64 for item in inventory["retained"])


def test_tensorboard_helper_exports_scalar_csvs(tmp_path):
    export = _api("src.tensorboard_export", "export_tensorboard_scalars")
    from torch.utils.tensorboard import SummaryWriter

    event_dir = tmp_path / "tensorboard" / "run"
    writer = SummaryWriter(log_dir=event_dir)
    writer.add_scalar("train/loss", 0.75, 10)
    writer.flush()
    writer.close()

    output_dir = tmp_path / "tensorboard_scalars"
    manifest = export([event_dir], output_dir)

    assert manifest["exported_scalars"][0]["tag"] == "train/loss"
    csv_path = output_dir / manifest["exported_scalars"][0]["csv"]
    assert csv_path.is_file()
    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["step"] == "10"
    assert float(rows[0]["value"]) == pytest.approx(0.75)
