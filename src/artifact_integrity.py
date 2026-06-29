from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parent.parent


class ArtifactTrustError(ValueError):
    """Raised when an artifact cannot be proven trusted before deserialization."""


def sha256_file(path: Path) -> str:
    p = Path(path)
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def repo_relative(path: Path, repo_root: Path = REPO_ROOT) -> str:
    try:
        return Path(path).resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return str(path)


def _resolve_existing_path(path: Path, repo_root: Path) -> Path:
    candidate = path if path.is_absolute() else repo_root / path
    resolved = candidate.resolve()
    if not resolved.is_file():
        raise ArtifactTrustError(f"Trusted artifact does not exist: {candidate}")
    try:
        resolved.relative_to(repo_root.resolve())
    except ValueError as exc:
        raise ArtifactTrustError(f"Trusted artifact is outside repository: {resolved}") from exc
    return resolved


def build_file_artifacts(
    artifact_paths: Mapping[str, Path | None],
    repo_root: Path = REPO_ROOT,
) -> dict[str, dict[str, str | None]]:
    out: dict[str, dict[str, str | None]] = {}
    for name, path in artifact_paths.items():
        if path is None:
            out[name] = {"relative_path": None, "sha256": None}
            continue
        p = Path(path)
        out[name] = {
            "relative_path": repo_relative(p, repo_root),
            "sha256": sha256_file(p) if p.is_file() else None,
        }
    return out


def load_artifact_manifest(run_dir: Path) -> dict[str, Any]:
    manifest_path = Path(run_dir) / "artifact_manifest.json"
    if not manifest_path.is_file():
        raise ArtifactTrustError(f"artifact_manifest.json not found under run dir: {run_dir}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _legacy_artifact_entry(
    manifest: Mapping[str, Any],
    artifact_name: str,
) -> dict[str, str | None] | None:
    checksums = manifest.get("checksums_sha256")
    rel_paths = manifest.get("relative_paths")
    if not isinstance(checksums, Mapping) or not isinstance(rel_paths, Mapping):
        return None
    rel_path = rel_paths.get(artifact_name)
    digest = checksums.get(artifact_name)
    if not isinstance(rel_path, str) or not isinstance(digest, str):
        return None
    return {"relative_path": rel_path, "sha256": digest}


def artifact_manifest_entry(
    manifest: Mapping[str, Any],
    artifact_name: str,
) -> dict[str, str | None]:
    file_artifacts = manifest.get("file_artifacts")
    entry = file_artifacts.get(artifact_name) if isinstance(file_artifacts, Mapping) else None
    if isinstance(entry, Mapping):
        rel_path = entry.get("relative_path")
        digest = entry.get("sha256")
        if isinstance(rel_path, str) and isinstance(digest, str):
            return {"relative_path": rel_path, "sha256": digest}

    legacy_entry = _legacy_artifact_entry(manifest, artifact_name)
    if legacy_entry is not None:
        return legacy_entry

    raise ArtifactTrustError(
        f"artifact_manifest.json does not contain trusted file artifact '{artifact_name}'"
    )


def resolve_trusted_artifact(
    run_dir: Path | None,
    artifact_name: str,
    requested_path: Path | None = None,
    *,
    repo_root: Path = REPO_ROOT,
    allow_unsafe: bool = False,
) -> Path:
    if allow_unsafe:
        if requested_path is None:
            raise ArtifactTrustError(
                f"--allow-unsafe-artifacts requires an explicit path for {artifact_name}"
            )
        p = requested_path if requested_path.is_absolute() else repo_root / requested_path
        resolved = p.resolve()
        if not resolved.is_file():
            raise ArtifactTrustError(f"Unsafe artifact path does not exist: {p}")
        return resolved

    if run_dir is None:
        raise ArtifactTrustError(
            f"{artifact_name} must be resolved through --run-dir artifact_manifest.json "
            "or explicitly allowed with --allow-unsafe-artifacts"
        )

    manifest = load_artifact_manifest(run_dir)
    entry = artifact_manifest_entry(manifest, artifact_name)
    rel_path = entry["relative_path"]
    expected_sha256 = entry["sha256"]
    if rel_path is None or expected_sha256 is None:
        raise ArtifactTrustError(f"Artifact '{artifact_name}' is not hash-covered in manifest")

    trusted_path = _resolve_existing_path(Path(rel_path), repo_root)
    if requested_path is not None:
        requested_resolved = (
            requested_path if requested_path.is_absolute() else repo_root / requested_path
        ).resolve()
        if requested_resolved != trusted_path:
            raise ArtifactTrustError(
                f"Requested {artifact_name} path does not match manifest: "
                f"{requested_resolved} != {trusted_path}"
            )

    actual_sha256 = sha256_file(trusted_path)
    if actual_sha256 != expected_sha256:
        raise ArtifactTrustError(
            f"SHA-256 mismatch for {artifact_name}: "
            f"expected {expected_sha256}, got {actual_sha256}"
        )
    return trusted_path
