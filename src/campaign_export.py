"""Verified incremental snapshots and deterministic final campaign bundles."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import shutil
import tarfile
import uuid
from pathlib import Path
from typing import Any, Mapping

from src.artifact_integrity import ArtifactTrustError, verify_artifact_manifest
from src.run_artifacts import atomic_write_json


SNAPSHOT_SCHEMA_VERSION = "1.0"
BUNDLE_SCHEMA_VERSION = "1.0"
SNAPSHOT_MANIFEST = "snapshot_manifest.json"
_REQUIRED_CAMPAIGN_FILES = (
    "campaign_state.json",
    "campaign_spec_original.json",
    "campaign_spec_resolved.json",
    "preflight_report.json",
    "cache_manifest.json",
)


class CampaignExportError(RuntimeError):
    """Campaign evidence cannot be exported or verified safely."""


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CampaignExportError(f"Cannot read JSON evidence: {path}") from error
    if not isinstance(value, dict):
        raise CampaignExportError(f"Expected a JSON object: {path}")
    return value


def _resolve_relative(root: Path, value: Any, *, field: str) -> Path:
    relative = Path(str(value))
    if relative.is_absolute() or ".." in relative.parts:
        raise CampaignExportError(f"Unsafe {field}: {value!r}")
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as error:
        raise CampaignExportError(f"Unsafe {field}: {value!r}") from error
    return resolved


def _validate_campaign(campaign_dir: Path, *, require_complete: bool) -> dict[str, Any]:
    campaign_dir = campaign_dir.resolve()
    if not campaign_dir.is_dir():
        raise CampaignExportError(f"Campaign directory does not exist: {campaign_dir}")
    for relative in _REQUIRED_CAMPAIGN_FILES:
        if not (campaign_dir / relative).is_file():
            raise CampaignExportError(f"Campaign export requires {relative}")

    state = _read_json(campaign_dir / "campaign_state.json")
    entries = state.get("entries")
    if not isinstance(entries, Mapping) or not entries:
        raise CampaignExportError("Campaign state has no entries")
    incomplete: list[str] = []
    for logical_id, raw_record in entries.items():
        if not isinstance(raw_record, Mapping):
            raise CampaignExportError(f"Invalid campaign state entry: {logical_id}")
        status = raw_record.get("status")
        if status == "completed":
            attempts = raw_record.get("attempts")
            if not isinstance(attempts, list) or not attempts:
                raise CampaignExportError(f"Completed entry has no attempt: {logical_id}")
            attempt = attempts[-1]
            if not isinstance(attempt, Mapping):
                raise CampaignExportError(f"Invalid completed attempt: {logical_id}")
            attempt_dir = _resolve_relative(
                campaign_dir,
                attempt.get("artifact_dir"),
                field=f"artifact_dir for {logical_id}",
            )
            try:
                verify_artifact_manifest(attempt_dir)
            except (ArtifactTrustError, OSError, ValueError) as error:
                raise CampaignExportError(
                    f"Completed campaign artifact is invalid: {logical_id}: {error}"
                ) from error
        elif status == "reused":
            source = raw_record.get("reuse_of")
            source_record = entries.get(source) if isinstance(source, str) else None
            if not isinstance(source_record, Mapping) or source_record.get("status") != "completed":
                raise CampaignExportError(f"Alias source is not completed: {logical_id}")
        elif require_complete:
            incomplete.append(str(logical_id))
    if incomplete:
        raise CampaignExportError(
            "Campaign is incomplete; final bundle blocked for: " + ", ".join(sorted(incomplete))
        )
    return state


def _campaign_inventory(campaign_dir: Path) -> dict[str, dict[str, Any]]:
    inventory: dict[str, dict[str, Any]] = {}
    for path in sorted(campaign_dir.rglob("*")):
        if path.is_symlink():
            raise CampaignExportError(f"Campaign export refuses symbolic links: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(campaign_dir).as_posix()
        if relative == SNAPSHOT_MANIFEST or path.name.endswith(".tmp"):
            continue
        inventory[relative] = {
            "sha256": _sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
    return inventory


def _assert_destination_outside_source(source: Path, destination: Path) -> None:
    source = source.resolve()
    destination = destination.resolve()
    if destination == source:
        raise CampaignExportError("Export destination must differ from campaign directory")
    try:
        destination.relative_to(source)
    except ValueError:
        return
    raise CampaignExportError("Export destination must be outside campaign directory")


def _copy_atomic(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        shutil.copyfile(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def verify_incremental_snapshot(destination: Path | str) -> dict[str, Any]:
    destination = Path(destination).resolve()
    manifest = _read_json(destination / SNAPSHOT_MANIFEST)
    if manifest.get("schema_version") != SNAPSHOT_SCHEMA_VERSION:
        raise CampaignExportError("Unsupported snapshot manifest schema")
    files = manifest.get("files")
    if not isinstance(files, Mapping):
        raise CampaignExportError("Snapshot manifest files must be an object")
    for relative, raw_record in files.items():
        if not isinstance(relative, str) or not isinstance(raw_record, Mapping):
            raise CampaignExportError("Snapshot manifest contains an invalid file record")
        path = _resolve_relative(destination, relative, field="snapshot path")
        if not path.is_file():
            raise CampaignExportError(f"Snapshot file is missing: {relative}")
        expected_size = raw_record.get("size_bytes")
        if path.stat().st_size != expected_size:
            raise CampaignExportError(f"Snapshot checksum/size mismatch: {relative}")
        if _sha256_file(path) != raw_record.get("sha256"):
            raise CampaignExportError(f"Snapshot checksum mismatch: {relative}")
    return {
        "status": "verified",
        "campaign_id": manifest.get("campaign_id"),
        "files": sorted(files),
        "snapshot_manifest_sha256": _sha256_file(destination / SNAPSHOT_MANIFEST),
    }


def create_incremental_snapshot(
    campaign_dir: Path | str,
    destination: Path | str,
) -> dict[str, Any]:
    campaign_dir = Path(campaign_dir).resolve()
    destination = Path(destination).resolve()
    _assert_destination_outside_source(campaign_dir, destination)
    state = _validate_campaign(campaign_dir, require_complete=False)
    inventory = _campaign_inventory(campaign_dir)
    destination.mkdir(parents=True, exist_ok=True)

    previous: dict[str, Any] | None = None
    manifest_path = destination / SNAPSHOT_MANIFEST
    if manifest_path.is_file():
        previous = _read_json(manifest_path)
        if previous.get("schema_version") != SNAPSHOT_SCHEMA_VERSION:
            raise CampaignExportError("Existing snapshot uses an unsupported schema")
        previous_files = previous.get("files")
        if not isinstance(previous_files, Mapping):
            raise CampaignExportError("Existing snapshot manifest is invalid")
        removed = sorted(set(previous_files).difference(inventory))
        if removed:
            raise CampaignExportError(
                "Snapshot source lost previously exported evidence: " + ", ".join(removed)
            )

    copied: list[str] = []
    unchanged: list[str] = []
    previous_files = previous.get("files", {}) if previous is not None else {}
    for relative, record in inventory.items():
        target = destination / Path(relative)
        expected = record["sha256"]
        already_valid = (
            previous_files.get(relative) == record
            and target.is_file()
            and target.stat().st_size == record["size_bytes"]
            and _sha256_file(target) == expected
        )
        if already_valid:
            unchanged.append(relative)
            continue
        _copy_atomic(campaign_dir / Path(relative), target)
        copied.append(relative)

    manifest = {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "campaign_id": state.get("campaign_id"),
        "campaign_spec_sha256": state.get("campaign_spec_sha256"),
        "cache_manifest_sha256": state.get("cache_manifest_sha256"),
        "preflight_report_sha256": state.get("preflight_report_sha256"),
        "files": inventory,
    }
    if previous != manifest:
        atomic_write_json(manifest_path, manifest)
    verified = verify_incremental_snapshot(destination)
    return {
        **verified,
        "copied_files": sorted(copied),
        "unchanged_files": sorted(unchanged),
    }


def _tar_info(name: str, size: int) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.size = size
    info.mode = 0o644
    info.mtime = 0
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    return info


def _canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _bundle_manifest(state: Mapping[str, Any], files: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "campaign_id": state.get("campaign_id"),
        "campaign_spec_sha256": state.get("campaign_spec_sha256"),
        "cache_manifest_sha256": state.get("cache_manifest_sha256"),
        "preflight_report_sha256": state.get("preflight_report_sha256"),
        "files": dict(files),
    }


def create_final_bundle(
    campaign_dir: Path | str,
    destination: Path | str,
) -> dict[str, Any]:
    campaign_dir = Path(campaign_dir).resolve()
    destination = Path(destination).resolve()
    _assert_destination_outside_source(campaign_dir, destination)
    state = _validate_campaign(campaign_dir, require_complete=True)
    inventory = _campaign_inventory(campaign_dir)
    manifest_bytes = _canonical_json_bytes(_bundle_manifest(state, inventory))
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
                with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as tar:
                    tar.addfile(
                        _tar_info("bundle_manifest.json", len(manifest_bytes)),
                        io.BytesIO(manifest_bytes),
                    )
                    for relative, record in inventory.items():
                        source = campaign_dir / Path(relative)
                        with source.open("rb") as handle:
                            tar.addfile(
                                _tar_info(f"campaign/{relative}", int(record["size_bytes"])),
                                handle,
                            )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return verify_final_bundle(destination)


def verify_final_bundle(archive_path: Path | str) -> dict[str, Any]:
    archive_path = Path(archive_path).resolve()
    try:
        with tarfile.open(archive_path, mode="r:gz") as tar:
            members = tar.getmembers()
            names = [member.name for member in members]
            if len(names) != len(set(names)) or "bundle_manifest.json" not in names:
                raise CampaignExportError("Final bundle member inventory is invalid")
            manifest_member = tar.getmember("bundle_manifest.json")
            manifest_handle = tar.extractfile(manifest_member)
            if manifest_handle is None:
                raise CampaignExportError("Final bundle manifest is unreadable")
            manifest = json.loads(manifest_handle.read().decode("utf-8"))
            if not isinstance(manifest, dict) or manifest.get("schema_version") != BUNDLE_SCHEMA_VERSION:
                raise CampaignExportError("Final bundle manifest schema is invalid")
            files = manifest.get("files")
            if not isinstance(files, Mapping):
                raise CampaignExportError("Final bundle file inventory is invalid")
            expected_names = {"bundle_manifest.json", *(f"campaign/{path}" for path in files)}
            if set(names) != expected_names:
                raise CampaignExportError("Final bundle contains undeclared or missing files")
            for relative, raw_record in files.items():
                if not isinstance(relative, str) or not isinstance(raw_record, Mapping):
                    raise CampaignExportError("Final bundle contains an invalid file record")
                member = tar.getmember(f"campaign/{relative}")
                handle = tar.extractfile(member)
                if handle is None:
                    raise CampaignExportError(f"Final bundle file is unreadable: {relative}")
                data = handle.read()
                if len(data) != raw_record.get("size_bytes"):
                    raise CampaignExportError(f"Final bundle size mismatch: {relative}")
                if _sha256_bytes(data) != raw_record.get("sha256"):
                    raise CampaignExportError(f"Final bundle checksum mismatch: {relative}")
    except CampaignExportError:
        raise
    except (OSError, tarfile.TarError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CampaignExportError(f"Cannot verify final bundle: {archive_path}") from error
    return {
        "status": "verified",
        "campaign_id": manifest.get("campaign_id"),
        "files": sorted(files),
        "archive_path": str(archive_path),
        "archive_sha256": _sha256_file(archive_path),
    }


__all__ = [
    "BUNDLE_SCHEMA_VERSION",
    "CampaignExportError",
    "SNAPSHOT_SCHEMA_VERSION",
    "create_final_bundle",
    "create_incremental_snapshot",
    "verify_final_bundle",
    "verify_incremental_snapshot",
]
