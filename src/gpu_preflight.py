"""Provider-neutral GPU-host preflight report contract and cheap smoke probes."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from src.artifact_integrity import verify_artifact_manifest
from src.campaign_export import create_incremental_snapshot, verify_incremental_snapshot
from src.cicids_cache import validate_cache
from src.experiment_profiles import MAIN_V1_PROFILE
from src.load_cicids2017 import list_cicids2017_csv_files
from src.run_artifacts import atomic_write_json
from src.system_telemetry import collect_host_inventory


PREFLIGHT_SCHEMA_VERSION = "1.0"
DEFAULT_MAX_AGE_HOURS = 24.0
_SMOKE_HYPERPARAMETERS = MAIN_V1_PROFILE.qrdqn_hyperparams()
DEFAULT_SMOKE_TIMESTEPS = int(
    _SMOKE_HYPERPARAMETERS["learning_starts"]
    + 2 * _SMOKE_HYPERPARAMETERS["train_freq"]
)
DEFAULT_SMOKE_MONITOR_INTERVAL_SECONDS = 1.0
REQUIRED_SMOKE_SCALARS = frozenset({"train/learning_rate", "train/loss"})
_GIB = 1024**3
_REQUIRED_PACKAGES = (
    "numpy",
    "pandas",
    "scikit-learn",
    "gymnasium",
    "torch",
    "stable-baselines3",
    "sb3-contrib",
    "joblib",
    "psutil",
    "tensorboard",
)


class PreflightError(RuntimeError):
    """Preflight evidence is failed, stale, corrupt, or mismatched."""


@dataclass(frozen=True)
class PreflightThresholds:
    min_logical_cpus: int = 16
    min_ram_gib: float = 120
    min_gpu_count: int = 1
    min_vram_gib: float = 80
    min_free_gib: float = 100

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if value <= 0:
                raise ValueError(f"{name} must be greater than zero")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PreflightError(f"Cannot read {label}: {path}") from error
    if not isinstance(value, dict):
        raise PreflightError(f"{label} must be a JSON object")
    return value


def _artifact_root_reference(path: Path | str, *, repository_root: Path) -> str:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = repository_root / resolved
    resolved = resolved.resolve()
    try:
        return resolved.relative_to(repository_root.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def collect_hardware(
    *,
    command_runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    inventory_collector: Callable[..., Mapping[str, Any]] = collect_host_inventory,
) -> dict[str, Any]:
    inventory = dict(inventory_collector(command_runner=command_runner))
    gpus = [
        {**dict(gpu), "vram_bytes": gpu.get("memory_total_bytes")}
        for gpu in inventory.get("gpus", [])
    ]
    nvidia_smi = dict(inventory.get("nvidia_smi", {}))
    return {
        "status": (
            "passed" if gpus and nvidia_smi.get("status") == "available" else "failed"
        ),
        "cpu": inventory.get("cpu", {}),
        "ram": inventory.get("memory", {}),
        "gpus": gpus,
        "nvidia_smi": nvidia_smi,
        "runtime": inventory.get("runtime", {}),
        "swap": inventory.get("swap", {}),
        "cgroup": inventory.get("cgroup", {}),
        "errors": inventory.get("errors", []),
    }


def collect_software(*, repo_root: Path) -> dict[str, Any]:
    del repo_root
    packages: dict[str, str | None] = {}
    for name in _REQUIRED_PACKAGES:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    try:
        import torch

        torch_runtime = {
            "version": str(torch.__version__),
            "cuda_build": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
        }
    except Exception as error:
        torch_runtime = {"error": str(error)}
    return {
        "status": "passed" if all(packages.values()) and "error" not in torch_runtime else "failed",
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "torch": torch_runtime,
    }


def _nearest_existing(path: Path) -> Path:
    candidate = path.resolve()
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    if not candidate.exists():
        raise FileNotFoundError(f"No existing filesystem ancestor for {path}")
    return candidate


def collect_filesystems(*, paths: Mapping[str, Path]) -> dict[str, Any]:
    locations: dict[str, Any] = {}
    for name, path in paths.items():
        anchor = _nearest_existing(Path(path))
        usage = shutil.disk_usage(anchor)
        locations[name] = {
            "path": str(Path(path).resolve()),
            "filesystem_anchor": str(anchor),
            "total_bytes": int(usage.total),
            "free_bytes": int(usage.free),
        }
    return {"status": "passed", "locations": locations}


def _run_git(
    repo_root: Path,
    command_runner: Callable[..., subprocess.CompletedProcess[str]],
    *args: str,
) -> subprocess.CompletedProcess[str]:
    return command_runner(
        ["git", *args],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )


def collect_git(
    *,
    repo_root: Path,
    command_runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, Any]:
    try:
        revision = _run_git(repo_root, command_runner, "rev-parse", "HEAD")
        status = _run_git(repo_root, command_runner, "status", "--short")
        lfs = _run_git(repo_root, command_runner, "lfs", "version")
    except (OSError, subprocess.SubprocessError) as error:
        return {"status": "failed", "error": str(error)}
    revision_value = revision.stdout.strip() if revision.returncode == 0 else None
    dirty_summary = [line for line in status.stdout.splitlines() if line.strip()]
    lfs_available = lfs.returncode == 0
    return {
        "status": "passed" if revision_value and lfs_available else "failed",
        "commit_sha": revision_value,
        "dirty": bool(dirty_summary),
        "dirty_summary": dirty_summary,
        "git_lfs": {
            "available": lfs_available,
            "version": lfs.stdout.strip() if lfs_available else None,
            "error": lfs.stderr.strip() if not lfs_available else None,
        },
    }


def collect_dataset(*, dataset_root: Path) -> dict[str, Any]:
    files = list_cicids2017_csv_files(dataset_root)
    records: list[dict[str, Any]] = []
    all_materialized = True
    for path in files:
        with path.open("rb") as handle:
            prefix = handle.read(200)
        is_pointer = prefix.startswith(b"version https://git-lfs.github.com/spec/v1")
        all_materialized = all_materialized and not is_pointer
        records.append(
            {
                "filename": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": None if is_pointer else _sha256_file(path),
                "git_lfs_materialized": not is_pointer,
            }
        )
    return {
        "status": "passed" if len(records) == 8 and all_materialized else "failed",
        "official_csv_count": len(records),
        "all_materialized": all_materialized,
        "files": records,
    }


def _validate_truth_column(path: Path, column: str) -> bool:
    import pandas as pd

    for chunk in pd.read_csv(path, usecols=[column], chunksize=100_000, low_memory=False):
        if column == "truth_y":
            values = pd.to_numeric(chunk[column], errors="coerce")
        else:
            values = (
                chunk[column]
                .astype(str)
                .str.strip()
                .str.upper()
                .map({"BENIGN": 0, "ATTACK": 1, "MALICIOUS": 1})
            )
        if not values.isin([0, 1]).all():
            return False
    return True


def collect_phase2_input(
    *,
    phase2_input: Path | None,
    expect_labels: bool,
) -> dict[str, Any]:
    if phase2_input is None:
        return {
            "status": "failed",
            "error": "A validated Phase 2 laboratory-flow input is required",
            "truth_labels_expected": expect_labels,
            "sensitive_metadata_export_enabled": False,
        }
    path = Path(phase2_input).resolve()
    if not path.is_file():
        return {
            "status": "failed",
            "path": str(path),
            "error": "Phase 2 input is missing or unreadable",
            "truth_labels_expected": expect_labels,
            "sensitive_metadata_export_enabled": False,
        }
    import pandas as pd

    columns = list(pd.read_csv(path, nrows=0).columns)
    truth_column = "truth_y" if "truth_y" in columns else (
        "truth_label" if "truth_label" in columns else None
    )
    truth_valid = None
    if truth_column is not None and expect_labels:
        truth_valid = _validate_truth_column(path, truth_column)
    passed = not expect_labels or (truth_column is not None and truth_valid is True)
    return {
        "status": "passed" if passed else "failed",
        "path": str(path),
        "filename": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
        "columns": columns,
        "truth_labels_expected": expect_labels,
        "truth_labels_available": truth_column is not None,
        "truth_label_column": truth_column,
        "truth_labels_valid": truth_valid,
        "sensitive_metadata_export_enabled": False,
    }


def collect_cache(*, dataset_root: Path, cache_root: Path) -> dict[str, Any]:
    result = dict(validate_cache(dataset_root, cache_root))
    manifest_path = Path(cache_root) / "cache_manifest.json"
    digest = result.get("manifest_sha256")
    if digest is None and manifest_path.is_file():
        digest = _sha256_file(manifest_path)
    return {
        "status": "passed" if result.get("validation_status") == "valid" else "failed",
        **result,
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": digest,
    }


def probe_cuda() -> dict[str, Any]:
    try:
        import torch

        available = bool(torch.cuda.is_available())
        record: dict[str, Any] = {
            "status": "failed",
            "available": available,
            "device_count": int(torch.cuda.device_count()),
            "cuda_build": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
        }
        if not available:
            record["tensor_operation"] = "not_run"
            return record
        device = torch.device("cuda:0")
        value = (torch.tensor([1.0, 2.0], device=device) * 2).sum().item()
        torch.cuda.synchronize(device)
        if value != 6.0:
            raise RuntimeError("Unexpected CUDA tensor result")
        record.update(
            {
                "status": "passed",
                "selected_device": str(device),
                "device_name": torch.cuda.get_device_name(device),
                "tensor_operation": "passed",
            }
        )
        return record
    except Exception as error:
        return {"status": "failed", "available": False, "error": str(error)}


def run_synthetic_artifact_smoke(
    *,
    artifact_root: Path,
    model_factory: Callable[..., Any] | None = None,
    smoke_timesteps: int = DEFAULT_SMOKE_TIMESTEPS,
    monitor_interval: float = DEFAULT_SMOKE_MONITOR_INTERVAL_SECONDS,
) -> dict[str, Any]:
    if smoke_timesteps < DEFAULT_SMOKE_TIMESTEPS:
        raise ValueError(
            f"smoke_timesteps must be at least {DEFAULT_SMOKE_TIMESTEPS} "
            "to produce real training scalars"
        )
    if monitor_interval <= 0:
        raise ValueError("monitor_interval must be greater than zero")
    import numpy as np

    from src.qrdqn_experiment import PreparedSplit, QRDQNRunConfig, run_qrdqn_experiment

    smoke_id = f"preflight-{_utc_now().strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:8]}"
    campaign_dir = Path(artifact_root).resolve() / "preflight-smoke" / smoke_id
    attempt_root = campaign_dir / "attempts" / "preflight_qrdqn"
    X_train = np.zeros((8, 152), dtype=np.float32)
    X_test = np.zeros((4, 152), dtype=np.float32)
    X_train[:, 0] = np.array([-4, -3, -2, -1, 1, 2, 3, 4], dtype=np.float32)
    X_test[:, 0] = np.array([-5, -1, 1, 5], dtype=np.float32)
    split = PreparedSplit(
        X_train=X_train,
        y_train=np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64),
        X_test=X_test,
        y_test=np.array([0, 0, 1, 1], dtype=np.int64),
        feature_names=[f"feature_{index}" for index in range(152)],
        metadata={"split_mode": "random", "split_seed": 42, "synthetic": True},
    )
    config = QRDQNRunConfig(
        artifact_root=attempt_root,
        run_id="attempt-1",
        dataset_root=campaign_dir / "synthetic-dataset",
        cache_root=None,
        cache_policy="off",
        split_seed=42,
        model_seed=42,
        timesteps=smoke_timesteps,
        checkpoint_freq=0,
        monitor_interval=monitor_interval,
        torch_threads=1,
        torch_inter_op_threads=1,
        verbose=0,
        campaign_id="preflight-smoke",
        logical_run_id="preflight_qrdqn",
        attempt=1,
    )
    kwargs: dict[str, Any] = {"split_loader": lambda _config: split}
    if model_factory is not None:
        kwargs["model_factory"] = model_factory
    run_dir = run_qrdqn_experiment(config, **kwargs)
    verified = verify_artifact_manifest(run_dir)
    environment = _read_json(run_dir / "environment.json", label="smoke environment")
    run_summary = _read_json(run_dir / "run_summary.json", label="smoke run summary")
    scalar_tags = set(run_summary.get("tensorboard_scalars", {}))
    missing_scalars = sorted(REQUIRED_SMOKE_SCALARS - scalar_tags)
    if missing_scalars:
        raise PreflightError(
            f"Synthetic smoke missing required training scalars: {missing_scalars}"
        )
    return {
        "status": "passed",
        "workload": "synthetic-qrdqn-training-artifact-v2",
        "scientific_result": False,
        "requested_timesteps": smoke_timesteps,
        "monitor_interval_seconds": monitor_interval,
        "campaign_dir": str(campaign_dir),
        "run_dir": str(run_dir),
        "manifest_sha256": _sha256_file(run_dir / "artifact_manifest.json"),
        "verified_schema_version": verified["schema_version"],
        "device_selected": environment.get("device_selected"),
        "run_summary": run_summary,
    }


def run_snapshot_smoke(
    *,
    artifact_record: Mapping[str, Any],
    snapshot_root: Path,
    campaign_spec: Path,
    cache_manifest: Path,
) -> dict[str, Any]:
    campaign_dir = Path(str(artifact_record["campaign_dir"])).resolve()
    run_dir = Path(str(artifact_record["run_dir"])).resolve()
    try:
        artifact_relative = run_dir.relative_to(campaign_dir).as_posix()
    except ValueError as error:
        raise PreflightError("Synthetic artifact is outside its smoke campaign") from error
    spec_payload = _read_json(Path(campaign_spec), label="campaign specification")
    spec_hash = _sha256_bytes(_canonical_json(spec_payload).encode("utf-8"))
    cache_manifest = Path(cache_manifest).resolve()
    if not cache_manifest.is_file():
        raise PreflightError("Snapshot smoke requires a cache manifest")
    preflight_placeholder = {"status": "smoke", "scientific_result": False}
    atomic_write_json(campaign_dir / "campaign_spec_original.json", spec_payload)
    atomic_write_json(campaign_dir / "campaign_spec_resolved.json", spec_payload)
    atomic_write_json(campaign_dir / "preflight_report.json", preflight_placeholder)
    shutil.copyfile(cache_manifest, campaign_dir / "cache_manifest.json")
    atomic_write_json(
        campaign_dir / "campaign_state.json",
        {
            "schema_version": "1.0",
            "campaign_id": "preflight-smoke",
            "campaign_spec_sha256": spec_hash,
            "cache_manifest_sha256": _sha256_file(cache_manifest),
            "preflight_report_sha256": _sha256_file(campaign_dir / "preflight_report.json"),
            "entries": {
                "preflight_qrdqn": {
                    "classification": "primary_model_training",
                    "status": "completed",
                    "attempts": [
                        {
                            "attempt": 1,
                            "artifact_dir": artifact_relative,
                            "status": "completed",
                        }
                    ],
                }
            },
        },
    )
    destination = Path(snapshot_root).resolve() / "preflight-smoke" / campaign_dir.name
    first = create_incremental_snapshot(campaign_dir, destination)
    second = create_incremental_snapshot(campaign_dir, destination)
    verified = verify_incremental_snapshot(destination)
    if second["copied_files"]:
        raise PreflightError("Incremental snapshot no-op copied unchanged files")
    return {
        "status": "passed",
        "destination": str(destination),
        "files": verified["files"],
        "snapshot_manifest_sha256": verified["snapshot_manifest_sha256"],
        "initial_copied_files": first["copied_files"],
        "incremental_noop_verified": True,
    }


def collect_runtime_benchmark(*, runtime_benchmark: Path | None) -> dict[str, Any]:
    if runtime_benchmark is None:
        return {"status": "not_requested"}
    path = Path(runtime_benchmark).resolve()
    try:
        payload = _read_json(path, label="runtime benchmark")
    except PreflightError as error:
        return {"status": "failed", "path": str(path), "error": str(error)}
    return {
        "status": "passed" if payload.get("status") == "completed" else "failed",
        "path": str(path),
        "sha256": _sha256_file(path),
        "result_count": len(payload.get("results", [])),
    }


def _default_collectors() -> dict[str, Callable[..., Mapping[str, Any]]]:
    return {
        "hardware": collect_hardware,
        "software": collect_software,
        "filesystems": collect_filesystems,
        "git": collect_git,
        "dataset": collect_dataset,
        "phase2": collect_phase2_input,
        "cache": collect_cache,
        "cuda": probe_cuda,
        "artifact_smoke": run_synthetic_artifact_smoke,
        "snapshot_smoke": run_snapshot_smoke,
        "runtime_benchmark": collect_runtime_benchmark,
    }


def _call_probe(
    name: str,
    probe: Callable[..., Mapping[str, Any]],
    **kwargs: Any,
) -> dict[str, Any]:
    try:
        result = probe(**kwargs)
    except Exception as error:
        return {"status": "failed", "error": f"{type(error).__name__}: {error}"}
    if not isinstance(result, Mapping):
        return {"status": "failed", "error": f"{name} probe returned a non-object"}
    return dict(result)


def _failure(check: str, message: str, **details: Any) -> dict[str, Any]:
    return {"check": check, "message": message, **details}


def _threshold_failures(
    hardware: Mapping[str, Any],
    filesystems: Mapping[str, Any],
    thresholds: PreflightThresholds,
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    cpu = hardware.get("cpu") if isinstance(hardware.get("cpu"), Mapping) else {}
    logical_cpus = cpu.get("logical_cpus")
    if not isinstance(logical_cpus, int) or logical_cpus < thresholds.min_logical_cpus:
        failures.append(
            _failure(
                "logical_cpus",
                "Logical CPU threshold not met",
                actual=logical_cpus,
                required=thresholds.min_logical_cpus,
            )
        )
    ram = hardware.get("ram") if isinstance(hardware.get("ram"), Mapping) else {}
    total_ram = ram.get("total_bytes")
    required_ram = int(thresholds.min_ram_gib * _GIB)
    if not isinstance(total_ram, int) or total_ram < required_ram:
        failures.append(
            _failure("ram", "RAM threshold not met", actual_bytes=total_ram, required_bytes=required_ram)
        )
    gpus = hardware.get("gpus") if isinstance(hardware.get("gpus"), list) else []
    if len(gpus) < thresholds.min_gpu_count:
        failures.append(
            _failure(
                "gpu_count",
                "GPU-count threshold not met",
                actual=len(gpus),
                required=thresholds.min_gpu_count,
            )
        )
    max_vram = max(
        (
            int(gpu.get("vram_bytes", 0))
            for gpu in gpus
            if isinstance(gpu, Mapping)
        ),
        default=0,
    )
    required_vram = int(thresholds.min_vram_gib * _GIB)
    if max_vram < required_vram:
        failures.append(
            _failure(
                "gpu_vram",
                "Selected-GPU VRAM threshold not met",
                actual_bytes=max_vram,
                required_bytes=required_vram,
            )
        )
    locations = (
        filesystems.get("locations")
        if isinstance(filesystems.get("locations"), Mapping)
        else {}
    )
    required_free = int(thresholds.min_free_gib * _GIB)
    for name in ("artifacts", "snapshots", "final_archive"):
        record = locations.get(name) if isinstance(locations, Mapping) else None
        free = record.get("free_bytes") if isinstance(record, Mapping) else None
        if not isinstance(free, int) or free < required_free:
            failures.append(
                _failure(
                    f"filesystem_{name}",
                    "Filesystem free-space threshold not met",
                    actual_bytes=free,
                    required_bytes=required_free,
                )
            )
    return failures


def _report_hash(report_without_hash: Mapping[str, Any]) -> str:
    return _sha256_bytes(_canonical_json(report_without_hash).encode("utf-8"))


def run_preflight(
    *,
    output_path: Path | str,
    campaign_spec: Path | str,
    dataset_root: Path | str,
    cache_root: Path | str,
    artifact_root: Path | str,
    snapshot_root: Path | str,
    phase2_input: Path | str | None = None,
    expect_phase2_labels: bool = False,
    thresholds: PreflightThresholds | None = None,
    max_age_hours: float = DEFAULT_MAX_AGE_HOURS,
    runtime_benchmark: Path | str | None = None,
    smoke_timesteps: int = DEFAULT_SMOKE_TIMESTEPS,
    smoke_monitor_interval: float = DEFAULT_SMOKE_MONITOR_INTERVAL_SECONDS,
    repo_root: Path | str | None = None,
    now: datetime | None = None,
    collectors: Mapping[str, Callable[..., Mapping[str, Any]]] | None = None,
) -> dict[str, Any]:
    if max_age_hours <= 0:
        raise ValueError("max_age_hours must be greater than zero")
    thresholds = thresholds or PreflightThresholds()
    now = now or _utc_now()
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    repo_root = Path(repo_root or Path(__file__).resolve().parent.parent).resolve()
    campaign_spec = Path(campaign_spec).resolve()
    dataset_root = Path(dataset_root).resolve()
    cache_root = Path(cache_root).resolve()
    artifact_root = Path(artifact_root).resolve()
    snapshot_root = Path(snapshot_root).resolve()
    phase2_path = None if phase2_input is None else Path(phase2_input).resolve()
    runtime_path = None if runtime_benchmark is None else Path(runtime_benchmark).resolve()
    spec_payload = _read_json(campaign_spec, label="campaign specification")
    spec_content_sha256 = _sha256_bytes(_canonical_json(spec_payload).encode("utf-8"))

    probes = _default_collectors()
    if collectors is not None:
        probes.update(collectors)
    checks: dict[str, dict[str, Any]] = {}
    checks["hardware"] = _call_probe("hardware", probes["hardware"])
    checks["software"] = _call_probe("software", probes["software"], repo_root=repo_root)
    checks["filesystems"] = _call_probe(
        "filesystems",
        probes["filesystems"],
        paths={
            "dataset": dataset_root,
            "cache": cache_root,
            "artifacts": artifact_root,
            "snapshots": snapshot_root,
            "final_archive": snapshot_root,
        },
    )
    checks["git"] = _call_probe("git", probes["git"], repo_root=repo_root)
    checks["dataset"] = _call_probe(
        "dataset", probes["dataset"], dataset_root=dataset_root
    )
    checks["phase2_input"] = _call_probe(
        "phase2",
        probes["phase2"],
        phase2_input=phase2_path,
        expect_labels=expect_phase2_labels,
    )
    checks["cache"] = _call_probe(
        "cache", probes["cache"], dataset_root=dataset_root, cache_root=cache_root
    )
    checks["cuda"] = _call_probe("cuda", probes["cuda"])
    checks["artifact_smoke"] = _call_probe(
        "artifact_smoke",
        probes["artifact_smoke"],
        artifact_root=artifact_root,
        smoke_timesteps=smoke_timesteps,
        monitor_interval=smoke_monitor_interval,
    )
    checks["snapshot_smoke"] = _call_probe(
        "snapshot_smoke",
        probes["snapshot_smoke"],
        artifact_record=checks["artifact_smoke"],
        snapshot_root=snapshot_root,
        campaign_spec=campaign_spec,
        cache_manifest=cache_root / "cache_manifest.json",
    )
    checks["runtime_benchmark"] = _call_probe(
        "runtime_benchmark",
        probes["runtime_benchmark"],
        runtime_benchmark=runtime_path,
    )

    failures: list[dict[str, Any]] = []
    for name, record in checks.items():
        if name == "runtime_benchmark" and record.get("status") == "not_requested":
            continue
        if record.get("status") not in {"passed", "verified"}:
            failures.append(
                _failure(name, "Preflight probe did not pass", error=record.get("error"))
            )
    failures.extend(_threshold_failures(checks["hardware"], checks["filesystems"], thresholds))

    cache_sha = checks["cache"].get("manifest_sha256")
    phase2_sha = checks["phase2_input"].get("sha256")
    bindings = {
        "campaign_spec_path": str(campaign_spec),
        "campaign_spec_sha256": spec_content_sha256,
        "campaign_spec_file_sha256": _sha256_file(campaign_spec),
        "dataset_root": str(dataset_root),
        "cache_root": str(cache_root),
        "artifact_root": _artifact_root_reference(
            artifact_root,
            repository_root=repo_root,
        ),
        "snapshot_root": str(snapshot_root),
        "cache_manifest_sha256": cache_sha,
        "phase2_input_path": None if phase2_path is None else str(phase2_path),
        "phase2_input_sha256": phase2_sha,
    }
    report_without_hash: dict[str, Any] = {
        "schema_version": PREFLIGHT_SCHEMA_VERSION,
        "status": "passed" if not failures else "failed",
        "created_at": now.astimezone(timezone.utc).isoformat(),
        "valid_until": (now + timedelta(hours=max_age_hours)).astimezone(timezone.utc).isoformat(),
        "bindings": bindings,
        "thresholds": asdict(thresholds),
        "threshold_overrides_recorded": asdict(thresholds) != asdict(PreflightThresholds()),
        "checks": checks,
        "phase2_input": checks["phase2_input"],
        "failures": failures,
        "warnings": (
            ["Optional runtime benchmark was not supplied"]
            if checks["runtime_benchmark"].get("status") == "not_requested"
            else []
        ),
    }
    report = {**report_without_hash, "report_sha256": _report_hash(report_without_hash)}
    atomic_write_json(Path(output_path), report)
    return report


def _parse_timestamp(value: Any, *, field: str) -> datetime:
    if not isinstance(value, str):
        raise PreflightError(f"Preflight {field} is missing")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise PreflightError(f"Preflight {field} is invalid") from error
    if parsed.tzinfo is None:
        raise PreflightError(f"Preflight {field} must include a timezone")
    return parsed


def verify_preflight_report(
    report_path: Path | str,
    *,
    expected_campaign_spec_sha256: str | None = None,
    expected_dataset_root: Path | str | None = None,
    expected_cache_root: Path | str | None = None,
    expected_artifact_root: Path | str | None = None,
    expected_snapshot_root: Path | str | None = None,
    expected_cache_manifest_sha256: str | None = None,
    expected_phase2_input_sha256: str | None = None,
    repository_root: Path | str | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    report = _read_json(Path(report_path), label="preflight report")
    if report.get("schema_version") != PREFLIGHT_SCHEMA_VERSION:
        raise PreflightError("Unsupported preflight report schema")
    expected_hash = report.get("report_sha256")
    without_hash = {key: value for key, value in report.items() if key != "report_sha256"}
    if not isinstance(expected_hash, str) or _report_hash(without_hash) != expected_hash:
        raise PreflightError("Preflight report hash does not match its contents")
    if report.get("status") != "passed":
        raise PreflightError("Real campaign execution requires a successful preflight report")
    now = now or _utc_now()
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    if now > _parse_timestamp(report.get("valid_until"), field="valid_until"):
        raise PreflightError("Preflight report is stale")
    bindings = report.get("bindings")
    if not isinstance(bindings, Mapping):
        raise PreflightError("Preflight bindings are missing")
    repository_root = Path(
        repository_root or Path(__file__).resolve().parent.parent
    ).resolve()
    expected = {
        "campaign_spec_sha256": expected_campaign_spec_sha256,
        "dataset_root": (
            None if expected_dataset_root is None else str(Path(expected_dataset_root).resolve())
        ),
        "cache_root": None if expected_cache_root is None else str(Path(expected_cache_root).resolve()),
        "artifact_root": (
            None
            if expected_artifact_root is None
            else _artifact_root_reference(
                expected_artifact_root,
                repository_root=repository_root,
            )
        ),
        "snapshot_root": (
            None if expected_snapshot_root is None else str(Path(expected_snapshot_root).resolve())
        ),
        "cache_manifest_sha256": expected_cache_manifest_sha256,
        "phase2_input_sha256": expected_phase2_input_sha256,
    }
    mismatches = {
        key: value
        for key, value in expected.items()
        if value is not None and bindings.get(key) != value
    }
    if mismatches:
        raise PreflightError(f"Preflight report bindings do not match: {mismatches}")
    return report


__all__ = [
    "DEFAULT_MAX_AGE_HOURS",
    "DEFAULT_SMOKE_MONITOR_INTERVAL_SECONDS",
    "DEFAULT_SMOKE_TIMESTEPS",
    "PREFLIGHT_SCHEMA_VERSION",
    "PreflightError",
    "PreflightThresholds",
    "collect_hardware",
    "run_preflight",
    "run_snapshot_smoke",
    "run_synthetic_artifact_smoke",
    "verify_preflight_report",
]
