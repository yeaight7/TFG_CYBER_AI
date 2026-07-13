"""Provider-neutral run artifact, metadata, logging, timing, and checkpoint primitives."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
import traceback
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, TextIO

from src.artifact_integrity import sha256_file
from src.system_telemetry import collect_host_inventory


ARTIFACT_SCHEMA_VERSION = "3.0"
ARTIFACT_MANIFEST_FILENAME = "artifact_manifest.json"
CHECKSUM_FILENAME = "SHA256SUMS"


class ArtifactValidationError(RuntimeError):
    """Raised when declared artifact evidence is absent or malformed."""


class ArtifactStateError(RuntimeError):
    """Raised when an artifact state transition would mutate completed evidence."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_text(path: Path, content: str) -> None:
    """Write text through a sibling temporary file and atomically replace the target."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{uuid.uuid4().hex}")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_json(path: Path, payload: Any) -> None:
    content = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    atomic_write_text(path, content)


def _safe_relative_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ArtifactValidationError(f"Artifact path must remain under the run directory: {value}")
    if path.as_posix() in {ARTIFACT_MANIFEST_FILENAME, CHECKSUM_FILENAME}:
        raise ArtifactValidationError(f"Artifact path is reserved by schema 3: {value}")
    return path


@dataclass(frozen=True)
class ArtifactRequirement:
    """A file or directory that a run declares before model-specific execution."""

    relative_path: str
    kind: str = "file"
    required: bool = True

    def __post_init__(self) -> None:
        _safe_relative_path(self.relative_path)
        if self.kind not in {"file", "directory"}:
            raise ValueError("Artifact requirement kind must be 'file' or 'directory'")

    def to_dict(self) -> dict[str, Any]:
        return {
            "relative_path": Path(self.relative_path).as_posix(),
            "kind": self.kind,
            "required": self.required,
        }


def _manifest_status(manifest_path: Path) -> str | None:
    if not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactStateError(f"Cannot read existing artifact manifest: {manifest_path}") from exc
    status = manifest.get("status")
    return status if isinstance(status, str) else None


class ArtifactManifestWriter:
    """Declare, transition, inventory, and complete one schema-3 run directory."""

    def __init__(
        self,
        run_dir: Path,
        *,
        run_metadata: Mapping[str, Any],
        requirements: Mapping[str, ArtifactRequirement],
    ) -> None:
        self.run_dir = Path(run_dir)
        self.manifest_path = self.run_dir / ARTIFACT_MANIFEST_FILENAME
        self.checksum_path = self.run_dir / CHECKSUM_FILENAME
        self.run_metadata = dict(run_metadata)
        missing_seed_fields = sorted(
            {"split_seed", "model_seed"}.difference(self.run_metadata)
        )
        if missing_seed_fields:
            raise ValueError(
                "Schema-3 run metadata requires explicit seed fields: "
                + ", ".join(missing_seed_fields)
            )
        for field in ("split_seed", "model_seed"):
            value = self.run_metadata[field]
            if value is not None and (not isinstance(value, int) or isinstance(value, bool)):
                raise TypeError(f"{field} must be an integer or None")
        self.requirements = dict(requirements)
        if not self.requirements:
            raise ValueError("At least one artifact requirement must be declared")
        if len(self.requirements) != len(set(self.requirements)):
            raise ValueError("Artifact requirement names must be unique")
        for name, requirement in self.requirements.items():
            if not name or not isinstance(requirement, ArtifactRequirement):
                raise TypeError("requirements must map non-empty names to ArtifactRequirement")
        self._started_at: str | None = None

    def _assert_mutable(self) -> None:
        status = _manifest_status(self.manifest_path)
        if status in {"completed", "failed", "interrupted", "invalid"}:
            raise ArtifactStateError(
                f"Terminal artifact evidence is immutable: {self.manifest_path} ({status})"
            )

    def _assert_started(self) -> None:
        if _manifest_status(self.manifest_path) != "running":
            raise ArtifactStateError("Artifact attempt must be started and running")

    def _base_manifest(self, status: str) -> dict[str, Any]:
        now = _utc_now()
        return {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "path_base": "run_dir",
            "status": status,
            "started_at": self._started_at or now,
            "updated_at": now,
            "completed_at": None,
            "run": self.run_metadata,
            "required_artifacts": self._requirements_payload(),
            "file_artifacts": {},
            "inventory": {},
            "checksum_file": None,
        }

    def _requirements_payload(self) -> dict[str, dict[str, Any]]:
        return {
            name: requirement.to_dict()
            for name, requirement in sorted(self.requirements.items())
        }

    def start(self) -> dict[str, Any]:
        if self.manifest_path.exists():
            status = _manifest_status(self.manifest_path)
            if status in {"completed", "failed", "interrupted", "invalid"}:
                raise ArtifactStateError(
                    f"Terminal artifact evidence is immutable: {self.manifest_path} ({status})"
                )
            raise ArtifactStateError(f"Artifact attempt has already been started: {self.run_dir}")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._started_at = _utc_now()
        manifest = self._base_manifest("running")
        atomic_write_json(self.manifest_path, manifest)
        return manifest

    def set_status(self, status: str, *, details: Mapping[str, Any] | None = None) -> dict[str, Any]:
        self._assert_mutable()
        if status == "completed":
            raise ArtifactStateError("Use complete() to validate and seal completed evidence")
        if status == "failed":
            raise ArtifactStateError("Use fail() to persist error evidence")
        if status not in {"running", "interrupted", "invalid"}:
            raise ValueError(f"Unsupported artifact status: {status}")
        manifest = self._load_or_base(status)
        manifest["status"] = status
        manifest["updated_at"] = _utc_now()
        if details is not None:
            manifest["status_details"] = dict(details)
        atomic_write_json(self.manifest_path, manifest)
        return manifest

    def _load_or_base(self, status: str) -> dict[str, Any]:
        if not self.manifest_path.is_file():
            return self._base_manifest(status)
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("schema_version") != ARTIFACT_SCHEMA_VERSION
            or manifest.get("path_base") != "run_dir"
            or manifest.get("run") != self.run_metadata
            or manifest.get("required_artifacts") != self._requirements_payload()
        ):
            raise ArtifactStateError(
                "Running artifact identity or declared requirements changed on disk"
            )
        started_at = manifest.get("started_at")
        if isinstance(started_at, str):
            self._started_at = started_at
        return manifest

    def _collect_inventory(
        self,
        *,
        require_all: bool = True,
    ) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
        inventory: dict[str, dict[str, Any]] = {}
        file_artifacts: dict[str, dict[str, Any]] = {}
        missing: list[str] = []

        for name, requirement in sorted(self.requirements.items()):
            relative_path = _safe_relative_path(requirement.relative_path)
            absolute_path = self.run_dir / relative_path
            exists = absolute_path.is_file() if requirement.kind == "file" else absolute_path.is_dir()
            if not exists:
                if requirement.required and require_all:
                    missing.append(f"{name} ({relative_path.as_posix()})")
                continue

            if requirement.kind == "file":
                files = [absolute_path]
            else:
                files = sorted(path for path in absolute_path.rglob("*") if path.is_file())
                if requirement.required and require_all and not files:
                    missing.append(f"{name} ({relative_path.as_posix()}: empty directory)")
                    continue

            for file_path in files:
                relative_file = file_path.relative_to(self.run_dir).as_posix()
                inventory[relative_file] = {
                    "sha256": sha256_file(file_path),
                    "size_bytes": file_path.stat().st_size,
                }

            if requirement.kind == "file":
                record = inventory[relative_path.as_posix()]
                file_artifacts[name] = {
                    "relative_path": relative_path.as_posix(),
                    "sha256": record["sha256"],
                    "size_bytes": record["size_bytes"],
                }

        if missing:
            raise ArtifactValidationError(
                "Missing required artifacts: " + ", ".join(missing)
            )
        return inventory, file_artifacts

    def complete(self) -> dict[str, Any]:
        self._assert_mutable()
        self._assert_started()
        manifest = self._load_or_base("completed")
        inventory, file_artifacts = self._collect_inventory()
        checksum_content = "".join(
            f"{record['sha256']}  {relative_path}\n"
            for relative_path, record in sorted(inventory.items())
        )
        atomic_write_text(self.checksum_path, checksum_content)

        now = _utc_now()
        manifest.update(
            {
                "status": "completed",
                "updated_at": now,
                "completed_at": now,
                "file_artifacts": file_artifacts,
                "inventory": inventory,
                "checksum_file": {
                    "relative_path": CHECKSUM_FILENAME,
                    "sha256": sha256_file(self.checksum_path),
                    "size_bytes": self.checksum_path.stat().st_size,
                },
            }
        )
        atomic_write_json(self.manifest_path, manifest)
        return manifest

    def fail(self, error: BaseException) -> dict[str, Any]:
        """Seal a failed attempt with error evidence without requiring complete artifacts."""
        self._assert_mutable()
        self._assert_started()
        manifest = self._load_or_base("failed")
        error_path = self.run_dir / "error.json"
        error_payload = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": "".join(traceback.format_exception(error)),
            "recorded_at": _utc_now(),
        }
        atomic_write_json(error_path, error_payload)
        inventory, file_artifacts = self._collect_inventory(require_all=False)
        error_record = {
            "sha256": sha256_file(error_path),
            "size_bytes": error_path.stat().st_size,
        }
        inventory["error.json"] = error_record
        file_artifacts["error"] = {"relative_path": "error.json", **error_record}
        checksum_content = "".join(
            f"{record['sha256']}  {relative_path}\n"
            for relative_path, record in sorted(inventory.items())
        )
        atomic_write_text(self.checksum_path, checksum_content)

        now = _utc_now()
        manifest.update(
            {
                "status": "failed",
                "updated_at": now,
                "failed_at": now,
                "file_artifacts": file_artifacts,
                "inventory": inventory,
                "checksum_file": {
                    "relative_path": CHECKSUM_FILENAME,
                    "sha256": sha256_file(self.checksum_path),
                    "size_bytes": self.checksum_path.stat().st_size,
                },
                "error": "error.json",
            }
        )
        atomic_write_json(self.manifest_path, manifest)
        return manifest


class _TeeStream:
    def __init__(self, original: TextIO, log: TextIO) -> None:
        self._original = original
        self._log = log

    def write(self, value: str) -> int:
        written = self._original.write(value)
        self._log.write(value)
        return written

    def flush(self) -> None:
        self._original.flush()
        self._log.flush()

    def isatty(self) -> bool:
        return self._original.isatty()

    @property
    def encoding(self) -> str | None:
        return self._original.encoding

    def fileno(self) -> int:
        return self._original.fileno()


@contextmanager
def tee_output(stdout_path: Path, stderr_path: Path) -> Iterator[None]:
    """Mirror Python stdout/stderr to durable UTF-8 log files."""
    stdout_path = Path(stdout_path)
    stderr_path = Path(stderr_path)
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    with (
        stdout_path.open("a", encoding="utf-8", buffering=1) as stdout_log,
        stderr_path.open("a", encoding="utf-8", buffering=1) as stderr_log,
    ):
        sys.stdout = _TeeStream(original_stdout, stdout_log)
        sys.stderr = _TeeStream(original_stderr, stderr_log)
        try:
            yield
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            sys.stdout = original_stdout
            sys.stderr = original_stderr


@dataclass
class _ActiveMeasurement:
    units: int | float | None = None
    unit: str | None = None

    def set_units(self, units: int | float, unit: str) -> None:
        if units < 0:
            raise ValueError("Measured units must be non-negative")
        if not unit:
            raise ValueError("Measured unit name must be non-empty")
        self.units = units
        self.unit = unit


class TimingRecorder:
    """Record named phase durations and optional throughput without model coupling."""

    def __init__(self, *, clock: Callable[[], float] = time.perf_counter) -> None:
        self._clock = clock
        self._phases: dict[str, dict[str, Any]] = {}

    @contextmanager
    def measure(self, name: str) -> Iterator[_ActiveMeasurement]:
        if not name or name in self._phases:
            raise ValueError(f"Timing phase must be unique and non-empty: {name!r}")
        measurement = _ActiveMeasurement()
        started = self._clock()
        try:
            yield measurement
        finally:
            duration = max(0.0, self._clock() - started)
            throughput = None
            if measurement.units is not None and duration > 0:
                throughput = measurement.units / duration
            self._phases[name] = {
                "duration_seconds": duration,
                "units": measurement.units,
                "unit": measurement.unit,
                "throughput_per_second": throughput,
            }

    def to_dict(self) -> dict[str, Any]:
        return {"phases": dict(self._phases)}

    def write(self, path: Path) -> None:
        atomic_write_json(path, self.to_dict())


def _run_git(repo_root: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _package_version(package_name: str) -> str | None:
    try:
        return metadata.version(package_name)
    except metadata.PackageNotFoundError:
        return None


def collect_environment_metadata(
    *,
    repo_root: Path,
    requested_torch_threads: int | None = None,
    requested_torch_interop_threads: int | None = None,
    storage_paths: Mapping[str, Path] | None = None,
    hardware_collector: Callable[..., dict[str, Any]] = collect_host_inventory,
    package_names: tuple[str, ...] = (
        "numpy",
        "pandas",
        "scikit-learn",
        "gymnasium",
        "stable-baselines3",
        "sb3-contrib",
        "torch",
        "joblib",
        "psutil",
        "tensorboard",
    ),
) -> dict[str, Any]:
    """Capture Git, platform, Python, package, CUDA, GPU, and thread metadata."""
    repo_root = Path(repo_root)
    commit = _run_git(repo_root, "rev-parse", "HEAD")
    dirty_summary = _run_git(repo_root, "status", "--short", "--untracked-files=normal")

    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        torch_metadata = {
            "version": torch.__version__,
            "cuda_available": cuda_available,
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
        }
        effective_threads = {
            "torch_intra_op": int(torch.get_num_threads()),
            "torch_inter_op": int(torch.get_num_interop_threads()),
        }
    except (ImportError, RuntimeError) as exc:
        torch_metadata = {
            "version": None,
            "cuda_available": False,
            "cuda_version": None,
            "cudnn_version": None,
            "error": f"{type(exc).__name__}: {exc}",
        }
        effective_threads = {"torch_intra_op": None, "torch_inter_op": None}

    hardware = hardware_collector(storage_paths=storage_paths)
    return {
        "captured_at": _utc_now(),
        "git": {
            "commit": commit,
            "dirty": None if dirty_summary is None else bool(dirty_summary),
            "dirty_summary": [] if not dirty_summary else dirty_summary.splitlines(),
        },
        "python": {
            "version": sys.version,
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "packages": {name: _package_version(name) for name in package_names},
        "torch": torch_metadata,
        "nvidia_smi": hardware.get(
            "nvidia_smi", {"available": False, "gpus": []}
        ),
        "hardware": hardware,
        "threads": {
            "requested": {
                "torch_intra_op": requested_torch_threads,
                "torch_inter_op": requested_torch_interop_threads,
            },
            "effective": effective_threads,
            "environment": {
                name: os.environ.get(name)
                for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
            },
        },
    }


def retain_checkpoints(
    checkpoints_dir: Path,
    *,
    keep: int,
    pattern: str = "*.zip",
    inventory_filename: str = "checkpoint_inventory.json",
) -> list[dict[str, Any]]:
    """Checksum a replacement set before deleting older model-only checkpoints."""
    if keep < 1:
        raise ValueError("keep must be at least 1")
    checkpoints_dir = Path(checkpoints_dir)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    inventory_path = checkpoints_dir / inventory_filename
    checkpoints = sorted(
        (path for path in checkpoints_dir.glob(pattern) if path.is_file()),
        key=lambda path: (path.stat().st_mtime_ns, path.name),
    )
    records = [
        {
            "filename": path.name,
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
        }
        for path in checkpoints
    ]
    remove_count = max(0, len(records) - keep)
    removed = records[:remove_count]
    retained = records[remove_count:]

    atomic_write_json(
        inventory_path,
        {
            "status": "checksums_recorded",
            "updated_at": _utc_now(),
            "candidates": records,
            "keep": keep,
        },
    )
    for record in removed:
        (checkpoints_dir / record["filename"]).unlink()

    atomic_write_json(
        inventory_path,
        {
            "status": "completed",
            "updated_at": _utc_now(),
            "keep": keep,
            "retained": retained,
            "removed": removed,
        },
    )
    return retained
