"""Provider-neutral, runtime-visible static system inventory collection."""

from __future__ import annotations

import csv
import os
import platform
import shutil
import socket
import subprocess
from pathlib import Path
from typing import Any, Callable, Mapping

import psutil


MIB = 1024 * 1024
_AUTO_TORCH = object()
_CGROUP_ROOT = Path("/sys/fs/cgroup")


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return None


def _read_limit(path: Path) -> int | None:
    value = _read_text(path)
    if value in {None, "", "max"}:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _parse_cpu_max(value: str | None) -> tuple[int | None, int | None]:
    if not value:
        return None, None
    fields = value.split()
    if len(fields) != 2:
        return None, None
    quota_text, period_text = fields
    try:
        period = int(period_text)
    except ValueError:
        period = None
    if quota_text == "max":
        return None, period
    try:
        quota = int(quota_text)
    except ValueError:
        quota = None
    return quota, period


def collect_cgroup_limits(root: Path = _CGROUP_ROOT) -> dict[str, Any]:
    """Collect cgroup v2 CPU and memory limits without inventing unlimited values."""
    root = Path(root)
    cpu_max = _read_text(root / "cpu.max")
    memory_current_text = _read_text(root / "memory.current")
    memory_max_text = _read_text(root / "memory.max")
    quota, period = _parse_cpu_max(cpu_max)
    version = (
        2
        if cpu_max is not None
        or memory_current_text is not None
        or memory_max_text is not None
        else None
    )
    return {
        "version": version,
        "cpu": {
            "quota_us": quota,
            "period_us": period,
            "limit_cpus": (
                None if quota is None or period in {None, 0} else quota / period
            ),
        },
        "memory": {
            "current_bytes": _read_limit(root / "memory.current"),
            "limit_bytes": _read_limit(root / "memory.max"),
        },
    }


def parse_cpuinfo(text: str) -> dict[str, str | None]:
    """Extract stable CPU identity fields from Linux-style cpuinfo text."""
    fields: dict[str, str] = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, value = (part.strip() for part in line.split(":", 1))
        fields.setdefault(key.lower(), value)
    return {
        "model": fields.get("model name") or fields.get("hardware"),
        "vendor": fields.get("vendor_id") or fields.get("cpu implementer"),
    }


def _cpu_socket_count(cpuinfo_text: str) -> int | None:
    identifiers = {
        line.split(":", 1)[1].strip()
        for line in cpuinfo_text.splitlines()
        if line.lower().startswith("physical id") and ":" in line
    }
    return len(identifiers) or None


def _cpu_frequency() -> dict[str, float | None]:
    try:
        frequency = psutil.cpu_freq()
    except (AttributeError, OSError, RuntimeError):
        frequency = None
    return {
        "current_mhz": None if frequency is None else float(frequency.current),
        "min_mhz": None if frequency is None else float(frequency.min),
        "max_mhz": None if frequency is None else float(frequency.max),
    }


def _process_available_cpus() -> int | None:
    get_affinity = getattr(os, "sched_getaffinity", None)
    if get_affinity is None:
        return None
    try:
        return len(get_affinity(0))
    except OSError:
        return None


def _cpu_inventory(cpuinfo_text: str) -> dict[str, Any]:
    identity = parse_cpuinfo(cpuinfo_text)
    portable_processor = platform.processor() or None
    return {
        "model": identity["model"] or portable_processor or platform.machine() or None,
        "vendor": identity["vendor"],
        "architecture": platform.machine() or None,
        "sockets": _cpu_socket_count(cpuinfo_text),
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cpus": psutil.cpu_count(logical=True),
        "process_available_cpus": _process_available_cpus(),
        "frequency": _cpu_frequency(),
    }


def _memory_inventory(memory: Any) -> dict[str, int | float | None]:
    return {
        "total_bytes": int(memory.total),
        "available_bytes": int(memory.available),
        "used_bytes": int(memory.used),
        "free_bytes": int(memory.free),
        "cached_bytes": _optional_int(memory, "cached"),
        "buffers_bytes": _optional_int(memory, "buffers"),
        "percent": float(memory.percent),
    }


def _swap_inventory(swap: Any) -> dict[str, int | float]:
    return {
        "total_bytes": int(swap.total),
        "used_bytes": int(swap.used),
        "free_bytes": int(swap.free),
        "percent": float(swap.percent),
    }


def _optional_int(record: Any, name: str) -> int | None:
    value = getattr(record, name, None)
    return None if value is None else int(value)


def _runtime_identity() -> dict[str, Any]:
    release = platform.release()
    version = platform.version()
    cgroup_text = _read_text(Path("/proc/1/cgroup")) or ""
    container_markers = ("docker", "containerd", "kubepods", "podman")
    return {
        "hostname": socket.gethostname(),
        "system": platform.system(),
        "release": release,
        "version": version,
        "machine": platform.machine(),
        "wsl": "microsoft" in f"{release} {version}".lower(),
        "container": Path("/.dockerenv").exists()
        or any(marker in cgroup_text.lower() for marker in container_markers),
    }


def _float_or_none(value: str) -> float | None:
    normalized = value.strip()
    if normalized.lower() in {"", "n/a", "na", "none", "[not supported]"}:
        return None
    try:
        return float(normalized)
    except ValueError:
        return None


def _query_static_gpus(
    command_runner: Callable[..., Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, str]]]:
    command = [
        "nvidia-smi",
        (
            "--query-gpu=index,name,uuid,pci.bus_id,memory.total,"
            "driver_version,power.limit"
        ),
        "--format=csv,noheader,nounits",
    ]
    try:
        result = command_runner(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return (
            [],
            {
                "available": False,
                "gpus": [],
                "status": "unavailable",
                "gpu_count": 0,
                "error": str(error),
            },
            [_error_record("nvidia-smi", error)],
        )
    if result.returncode != 0:
        message = result.stderr.strip() or f"nvidia-smi exited with {result.returncode}"
        return (
            [],
            {
                "available": False,
                "gpus": [],
                "status": "failed",
                "gpu_count": 0,
                "returncode": int(result.returncode),
                "error": message,
            },
            [
                {
                    "source": "nvidia-smi",
                    "type": "CommandError",
                    "message": message,
                }
            ],
        )

    gpus: list[dict[str, Any]] = []
    for row in csv.reader(line for line in result.stdout.splitlines() if line.strip()):
        if len(row) != 7:
            continue
        index, name, uuid, pci_bus_id, memory_mib, driver, power_limit = (
            value.strip() for value in row
        )
        memory_value = _float_or_none(memory_mib)
        gpus.append(
            {
                "index": int(index) if index.isdigit() else index,
                "name": name,
                "uuid": uuid or None,
                "pci_bus_id": pci_bus_id or None,
                "memory_total_bytes": (
                    None if memory_value is None else int(memory_value * MIB)
                ),
                "driver_version": driver or None,
                "power_limit_watts": _float_or_none(power_limit),
            }
        )
    status = {
        "available": bool(gpus),
        "gpus": gpus,
        "status": "available" if gpus else "failed",
        "gpu_count": len(gpus),
    }
    if not gpus:
        status["error"] = "no valid GPU rows"
    return gpus, status, []


def _torch_devices(torch_module: Any) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    if torch_module is None:
        return [], []
    try:
        if not torch_module.cuda.is_available():
            return [], []
        devices = []
        for index in range(int(torch_module.cuda.device_count())):
            properties = torch_module.cuda.get_device_properties(index)
            devices.append(
                {
                    "index": index,
                    "name": properties.name,
                    "compute_capability": [int(properties.major), int(properties.minor)],
                    "total_memory_bytes": int(properties.total_memory),
                    "multiprocessor_count": _optional_int(
                        properties, "multi_processor_count"
                    ),
                }
            )
        return devices, []
    except (AttributeError, RuntimeError) as error:
        return [], [_error_record("torch.cuda", error)]


def _load_torch() -> tuple[Any, list[dict[str, str]]]:
    try:
        import torch
    except (ImportError, RuntimeError) as error:
        return None, [_error_record("torch", error)]
    return torch, []


def _nearest_existing_path(path: Path) -> Path | None:
    candidate = path.expanduser().resolve()
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return candidate if candidate.exists() else None


def _storage_inventory(
    storage_paths: Mapping[str, Path] | None,
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    inventory: dict[str, Any] = {}
    errors: list[dict[str, str]] = []
    for label, requested in (storage_paths or {}).items():
        requested_path = Path(requested).expanduser().resolve()
        measured_path = _nearest_existing_path(requested_path)
        if measured_path is None:
            error = FileNotFoundError(f"no existing parent for {requested_path}")
            errors.append(_error_record(f"storage:{label}", error))
            inventory[label] = {
                "requested_path": str(requested_path),
                "measured_path": None,
                "total_bytes": None,
                "used_bytes": None,
                "free_bytes": None,
            }
            continue
        try:
            usage = shutil.disk_usage(measured_path)
        except OSError as error:
            errors.append(_error_record(f"storage:{label}", error))
            inventory[label] = {
                "requested_path": str(requested_path),
                "measured_path": str(measured_path),
                "total_bytes": None,
                "used_bytes": None,
                "free_bytes": None,
            }
            continue
        inventory[label] = {
            "requested_path": str(requested_path),
            "measured_path": str(measured_path),
            "total_bytes": int(usage.total),
            "used_bytes": int(usage.used),
            "free_bytes": int(usage.free),
        }
    return inventory, errors


def _error_record(source: str, error: BaseException) -> dict[str, str]:
    return {
        "source": source,
        "type": type(error).__name__,
        "message": str(error),
    }


def collect_host_inventory(
    *,
    storage_paths: Mapping[str, Path] | None = None,
    command_runner: Callable[..., Any] = subprocess.run,
    cpuinfo_text: str | None = None,
    cgroup_root: Path = _CGROUP_ROOT,
    torch_module: Any = _AUTO_TORCH,
) -> dict[str, Any]:
    """Collect hardware visible to the current Linux/WSL/container runtime."""
    if cpuinfo_text is None:
        cpuinfo_text = _read_text(Path("/proc/cpuinfo")) or ""

    errors: list[dict[str, str]] = []
    gpus, nvidia_status, nvidia_errors = _query_static_gpus(command_runner)
    errors.extend(nvidia_errors)

    if torch_module is _AUTO_TORCH:
        torch_module, torch_import_errors = _load_torch()
        errors.extend(torch_import_errors)
    torch_devices, torch_errors = _torch_devices(torch_module)
    errors.extend(torch_errors)

    storage, storage_errors = _storage_inventory(storage_paths)
    errors.extend(storage_errors)

    memory = psutil.virtual_memory()
    swap = psutil.swap_memory()
    return {
        "schema_version": "1.0",
        "status": "completed_with_warnings" if errors else "completed",
        "runtime": _runtime_identity(),
        "cpu": _cpu_inventory(cpuinfo_text),
        "memory": _memory_inventory(memory),
        "swap": _swap_inventory(swap),
        "cgroup": collect_cgroup_limits(cgroup_root),
        "gpus": gpus,
        "torch_cuda_devices": torch_devices,
        "storage": storage,
        "nvidia_smi": nvidia_status,
        "errors": errors,
    }


__all__ = [
    "collect_cgroup_limits",
    "collect_host_inventory",
    "parse_cpuinfo",
]
