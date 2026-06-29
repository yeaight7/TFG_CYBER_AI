import json

import pytest

from src.artifact_integrity import (
    ArtifactTrustError,
    build_file_artifacts,
    resolve_trusted_artifact,
)


def _write_manifest(run_dir, repo_root, artifacts):
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": "2.0",
        "run_id": run_dir.name,
        "status": "completed",
        "file_artifacts": build_file_artifacts(artifacts, repo_root=repo_root),
    }
    (run_dir / "artifact_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    return manifest


def test_resolve_trusted_artifact_accepts_manifest_hash(tmp_path):
    model = tmp_path / "models" / "run.zip"
    model.parent.mkdir()
    model.write_text("model-bytes", encoding="utf-8")
    run_dir = tmp_path / "runs" / "run"
    _write_manifest(run_dir, tmp_path, {"model": model})

    resolved = resolve_trusted_artifact(run_dir, "model", model, repo_root=tmp_path)

    assert resolved == model.resolve()


def test_resolve_trusted_artifact_rejects_arbitrary_path(tmp_path):
    model = tmp_path / "models" / "run.zip"
    other = tmp_path / "models" / "other.zip"
    model.parent.mkdir()
    model.write_text("model-bytes", encoding="utf-8")
    other.write_text("other-bytes", encoding="utf-8")
    run_dir = tmp_path / "runs" / "run"
    _write_manifest(run_dir, tmp_path, {"model": model})

    with pytest.raises(ArtifactTrustError, match="does not match manifest"):
        resolve_trusted_artifact(run_dir, "model", other, repo_root=tmp_path)


def test_resolve_trusted_artifact_rejects_hash_mismatch(tmp_path):
    scaler = tmp_path / "runs" / "run" / "scaler.joblib"
    scaler.parent.mkdir(parents=True)
    scaler.write_text("trusted", encoding="utf-8")
    run_dir = scaler.parent
    _write_manifest(run_dir, tmp_path, {"scaler": scaler})
    scaler.write_text("tampered", encoding="utf-8")

    with pytest.raises(ArtifactTrustError, match="SHA-256 mismatch"):
        resolve_trusted_artifact(run_dir, "scaler", scaler, repo_root=tmp_path)


def test_allow_unsafe_artifacts_requires_explicit_existing_path(tmp_path):
    arbitrary = tmp_path / "outside-manifest.joblib"
    arbitrary.write_text("local experiment", encoding="utf-8")

    assert resolve_trusted_artifact(
        None,
        "scaler",
        arbitrary,
        repo_root=tmp_path,
        allow_unsafe=True,
    ) == arbitrary.resolve()

    with pytest.raises(ArtifactTrustError, match="requires an explicit path"):
        resolve_trusted_artifact(None, "scaler", None, repo_root=tmp_path, allow_unsafe=True)
