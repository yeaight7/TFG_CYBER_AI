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


def _schema_major(manifest: Mapping[str, Any]) -> int:
    version = manifest.get("schema_version")
    if not isinstance(version, str):
        raise ArtifactTrustError("artifact_manifest.json has no string schema_version")
    try:
        return int(version.split(".", maxsplit=1)[0])
    except ValueError as exc:
        raise ArtifactTrustError(f"Unsupported artifact schema version: {version}") from exc


def _resolve_run_relative(path: Path, run_dir: Path) -> Path:
    if path.is_absolute() or ".." in path.parts:
        raise ArtifactTrustError(f"Artifact path escapes run directory: {path}")
    resolved = (run_dir / path).resolve()
    try:
        resolved.relative_to(run_dir.resolve())
    except ValueError as exc:
        raise ArtifactTrustError(f"Artifact path escapes run directory: {path}") from exc
    if not resolved.is_file():
        raise ArtifactTrustError(f"Trusted artifact does not exist: {resolved}")
    return resolved


def _verify_digest(path: Path, expected_sha256: str) -> None:
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise ArtifactTrustError(
            f"SHA-256 mismatch for {path}: expected {expected_sha256}, got {actual_sha256}"
        )


def verify_artifact_manifest(
    run_dir: Path,
    *,
    repo_root: Path = REPO_ROOT,
    require_completed: bool = True,
) -> dict[str, Any]:
    """Verify schema-3 evidence or read historical schema-2 evidence compatibly."""
    run_dir = Path(run_dir)
    manifest = load_artifact_manifest(run_dir)
    schema_major = _schema_major(manifest)
    status = manifest.get("status")
    if require_completed and status != "completed":
        raise ArtifactTrustError(f"Artifact manifest is not completed: status={status!r}")

    if schema_major == 2:
        file_artifacts = manifest.get("file_artifacts")
        if not isinstance(file_artifacts, Mapping):
            raise ArtifactTrustError("Schema-2 manifest has no file_artifacts mapping")
        verified = 0
        for name in file_artifacts:
            entry = file_artifacts[name]
            if isinstance(entry, Mapping) and entry.get("relative_path") is None:
                continue
            trusted = artifact_manifest_entry(manifest, name)
            relative_path = trusted["relative_path"]
            expected_sha256 = trusted["sha256"]
            if relative_path is None or expected_sha256 is None:
                continue
            path = _resolve_existing_path(Path(relative_path), repo_root)
            _verify_digest(path, expected_sha256)
            verified += 1
        return {
            "schema_version": str(manifest["schema_version"]),
            "status": str(status),
            "verified_files": verified,
        }

    if schema_major != 3 or manifest.get("path_base") != "run_dir":
        raise ArtifactTrustError(
            f"Unsupported artifact manifest contract: schema={manifest.get('schema_version')!r}"
        )

    requirements = manifest.get("required_artifacts")
    inventory = manifest.get("inventory")
    if not isinstance(requirements, Mapping) or not isinstance(inventory, Mapping):
        raise ArtifactTrustError("Schema-3 manifest lacks requirements or inventory")

    enforce_complete_contract = status == "completed"
    expected_inventory_paths: set[str] = set()
    expected_file_artifact_names: set[str] = set()
    for name, requirement in requirements.items():
        if not isinstance(requirement, Mapping):
            raise ArtifactTrustError(f"Invalid requirement entry: {name}")
        relative_value = requirement.get("relative_path")
        kind = requirement.get("kind")
        required = requirement.get("required", True)
        if not isinstance(relative_value, str) or kind not in {"file", "directory"}:
            raise ArtifactTrustError(f"Invalid requirement entry: {name}")
        relative_path = Path(relative_value)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ArtifactTrustError(f"Artifact path escapes run directory: {relative_value}")
        resolved = (run_dir / relative_path).resolve()
        try:
            resolved.relative_to(run_dir.resolve())
        except ValueError as exc:
            raise ArtifactTrustError(
                f"Artifact path escapes run directory: {relative_value}"
            ) from exc
        exists = resolved.is_file() if kind == "file" else resolved.is_dir()
        if required and enforce_complete_contract and not exists:
            raise ArtifactTrustError(f"Required artifact is missing: {name} ({relative_value})")
        if not exists:
            continue
        if kind == "file":
            expected_inventory_paths.add(relative_path.as_posix())
            expected_file_artifact_names.add(str(name))
        else:
            directory_files = {
                path.relative_to(run_dir).as_posix()
                for path in resolved.rglob("*")
                if path.is_file()
            }
            if required and enforce_complete_contract and not directory_files:
                raise ArtifactTrustError(
                    f"Required artifact directory is empty: {name} ({relative_value})"
                )
            expected_inventory_paths.update(directory_files)

    if enforce_complete_contract and set(inventory) != expected_inventory_paths:
        missing = sorted(expected_inventory_paths - set(inventory))
        unexpected = sorted(set(inventory) - expected_inventory_paths)
        raise ArtifactTrustError(
            f"Artifact inventory mismatch: missing={missing}, unexpected={unexpected}"
        )

    for relative_value, record in inventory.items():
        if not isinstance(relative_value, str) or not isinstance(record, Mapping):
            raise ArtifactTrustError("Invalid schema-3 inventory entry")
        digest = record.get("sha256")
        size_bytes = record.get("size_bytes")
        if not isinstance(digest, str) or not isinstance(size_bytes, int):
            raise ArtifactTrustError(f"Invalid inventory metadata for {relative_value}")
        path = _resolve_run_relative(Path(relative_value), run_dir)
        if path.stat().st_size != size_bytes:
            raise ArtifactTrustError(
                f"Size mismatch for {path}: expected {size_bytes}, got {path.stat().st_size}"
            )
        _verify_digest(path, digest)

    file_artifacts = manifest.get("file_artifacts")
    if not isinstance(file_artifacts, Mapping):
        raise ArtifactTrustError("Schema-3 manifest has no file_artifacts mapping")
    if enforce_complete_contract and set(file_artifacts) != expected_file_artifact_names:
        raise ArtifactTrustError(
            "Schema-3 file_artifacts do not match declared file requirements"
        )
    for name, entry in file_artifacts.items():
        if not isinstance(entry, Mapping):
            raise ArtifactTrustError(f"Invalid file_artifacts entry: {name}")
        relative_value = entry.get("relative_path")
        digest = entry.get("sha256")
        size_bytes = entry.get("size_bytes")
        inventory_record = inventory.get(relative_value) if isinstance(relative_value, str) else None
        if (
            not isinstance(inventory_record, Mapping)
            or inventory_record.get("sha256") != digest
            or inventory_record.get("size_bytes") != size_bytes
        ):
            raise ArtifactTrustError(
                f"Schema-3 file_artifacts entry disagrees with inventory: {name}"
            )

    checksum_record = manifest.get("checksum_file")
    if not isinstance(checksum_record, Mapping):
        raise ArtifactTrustError("Schema-3 manifest has no checksum_file record")
    checksum_relative = checksum_record.get("relative_path")
    checksum_digest = checksum_record.get("sha256")
    if not isinstance(checksum_relative, str) or not isinstance(checksum_digest, str):
        raise ArtifactTrustError("Invalid schema-3 checksum_file record")
    checksum_path = _resolve_run_relative(Path(checksum_relative), run_dir)
    checksum_size = checksum_record.get("size_bytes")
    if not isinstance(checksum_size, int) or checksum_path.stat().st_size != checksum_size:
        raise ArtifactTrustError(f"Size mismatch for checksum file: {checksum_path}")
    _verify_digest(checksum_path, checksum_digest)
    expected_sums = "".join(
        f"{record['sha256']}  {relative_path}\n"
        for relative_path, record in sorted(inventory.items())
    )
    if checksum_path.read_text(encoding="utf-8") != expected_sums:
        raise ArtifactTrustError("SHA256SUMS content does not match artifact inventory")

    return {
        "schema_version": str(manifest["schema_version"]),
        "status": str(status),
        "verified_files": len(inventory),
    }


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

    schema3_run_relative = _schema_major(manifest) == 3 and manifest.get("path_base") == "run_dir"
    if schema3_run_relative:
        trusted_path = _resolve_run_relative(Path(rel_path), Path(run_dir))
    else:
        trusted_path = _resolve_existing_path(Path(rel_path), repo_root)
    if requested_path is not None:
        requested_base = Path(run_dir) if schema3_run_relative else repo_root
        requested_resolved = (
            requested_path if requested_path.is_absolute() else requested_base / requested_path
        ).resolve()
        if requested_resolved != trusted_path:
            raise ArtifactTrustError(
                f"Requested {artifact_name} path does not match manifest: "
                f"{requested_resolved} != {trusted_path}"
            )

    _verify_digest(trusted_path, expected_sha256)
    return trusted_path
