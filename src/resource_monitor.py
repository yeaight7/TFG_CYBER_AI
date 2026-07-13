"""Non-fatal process, system, and optional NVIDIA GPU resource monitoring."""

from __future__ import annotations

import csv
import io
import os
import subprocess
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import psutil

from src.run_artifacts import atomic_write_json, atomic_write_text


DEFAULT_MONITOR_INTERVAL_SECONDS = 30.0
MIB = 1024 * 1024
LEGACY_SYSTEM_METRIC_FIELDS = (
    "timestamp",
    "process_cpu_percent",
    "system_cpu_percent",
    "process_rss_bytes",
    "system_ram_used_bytes",
    "system_ram_total_bytes",
    "gpu_utilization_percent",
    "gpu_memory_used_bytes",
    "gpu_memory_total_bytes",
    "gpu_power_watts",
    "gpu_temperature_c",
)
SYSTEM_METRIC_FIELDS = LEGACY_SYSTEM_METRIC_FIELDS + (
    "process_vms_bytes",
    "process_threads",
    "process_read_bytes",
    "process_write_bytes",
    "system_load_1m",
    "system_load_5m",
    "system_load_15m",
    "system_cpu_frequency_mhz",
    "system_ram_available_bytes",
    "system_ram_free_bytes",
    "system_ram_percent",
    "system_swap_used_bytes",
    "system_swap_free_bytes",
    "system_swap_total_bytes",
    "system_swap_percent",
    "system_disk_read_bytes",
    "system_disk_write_bytes",
    "gpu_memory_utilization_percent",
    "gpu_memory_free_bytes",
    "gpu_power_limit_watts",
    "gpu_sm_clock_mhz",
    "gpu_memory_clock_mhz",
    "gpu_fan_speed_percent",
    "gpu_performance_state",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _float_or_none(value: str) -> float | None:
    value = value.strip()
    if not value or value.lower() in {"n/a", "na", "none", "[not supported]"}:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _text_or_none(value: str) -> str | None:
    normalized = value.strip()
    if not normalized or normalized.lower() in {
        "n/a",
        "na",
        "none",
        "[not supported]",
    }:
        return None
    return normalized


def query_nvidia_metrics(
    *,
    command_runner: Callable[..., Any] = subprocess.run,
    gpu_index: int = 0,
) -> dict[str, float | int | str | None]:
    command = [
        "nvidia-smi",
        f"--id={gpu_index}",
        (
            "--query-gpu=utilization.gpu,utilization.memory,memory.used,"
            "memory.free,memory.total,power.draw,power.limit,temperature.gpu,"
            "clocks.current.sm,clocks.current.memory,fan.speed,pstate"
        ),
        "--format=csv,noheader,nounits",
    ]
    result = command_runner(command, check=True, capture_output=True, text=True)
    line = next((line for line in result.stdout.splitlines() if line.strip()), "")
    values = [value.strip() for value in line.split(",")]
    if len(values) != 12:
        raise RuntimeError("nvidia-smi returned an unexpected resource metric row")
    (
        utilization,
        memory_utilization,
        memory_used_mib,
        memory_free_mib,
        memory_total_mib,
        power,
        power_limit,
        temperature,
        sm_clock,
        memory_clock,
        fan_speed,
    ) = map(_float_or_none, values[:11])
    return {
        "gpu_utilization_percent": utilization,
        "gpu_memory_utilization_percent": memory_utilization,
        "gpu_memory_used_bytes": (
            None if memory_used_mib is None else int(memory_used_mib * MIB)
        ),
        "gpu_memory_free_bytes": (
            None if memory_free_mib is None else int(memory_free_mib * MIB)
        ),
        "gpu_memory_total_bytes": (
            None if memory_total_mib is None else int(memory_total_mib * MIB)
        ),
        "gpu_power_watts": power,
        "gpu_power_limit_watts": power_limit,
        "gpu_temperature_c": temperature,
        "gpu_sm_clock_mhz": sm_clock,
        "gpu_memory_clock_mhz": memory_clock,
        "gpu_fan_speed_percent": fan_speed,
        "gpu_performance_state": _text_or_none(values[11]),
    }


def _monitoring_error(tool: str, error: BaseException) -> dict[str, str]:
    return {
        "tool": tool,
        "type": type(error).__name__,
        "message": str(error),
    }


def collect_resource_sample(
    *,
    process: psutil.Process | None = None,
    command_runner: Callable[..., Any] = subprocess.run,
    gpu_index: int = 0,
) -> dict[str, Any]:
    process = process or psutil.Process()
    virtual_memory = psutil.virtual_memory()
    swap_memory = psutil.swap_memory()
    memory_info = process.memory_info()
    sample: dict[str, Any] = {field: None for field in SYSTEM_METRIC_FIELDS}
    sample.update(
        {
        "timestamp": _utc_now(),
        "process_cpu_percent": float(process.cpu_percent(interval=None)),
        "system_cpu_percent": float(psutil.cpu_percent(interval=None)),
        "process_rss_bytes": int(memory_info.rss),
        "process_vms_bytes": int(memory_info.vms),
        "process_threads": int(process.num_threads()),
        "system_ram_used_bytes": int(virtual_memory.used),
        "system_ram_total_bytes": int(virtual_memory.total),
        "system_ram_available_bytes": int(virtual_memory.available),
        "system_ram_free_bytes": int(virtual_memory.free),
        "system_ram_percent": float(virtual_memory.percent),
        "system_swap_used_bytes": int(swap_memory.used),
        "system_swap_free_bytes": int(swap_memory.free),
        "system_swap_total_bytes": int(swap_memory.total),
        "system_swap_percent": float(swap_memory.percent),
        }
    )
    errors: list[dict[str, str]] = []

    try:
        io_counters = process.io_counters()
        sample["process_read_bytes"] = int(io_counters.read_bytes)
        sample["process_write_bytes"] = int(io_counters.write_bytes)
    except (AttributeError, OSError, psutil.Error) as exc:
        errors.append(_monitoring_error("psutil.process_io", exc))

    try:
        load_1m, load_5m, load_15m = os.getloadavg()
        sample["system_load_1m"] = float(load_1m)
        sample["system_load_5m"] = float(load_5m)
        sample["system_load_15m"] = float(load_15m)
    except (AttributeError, OSError) as exc:
        errors.append(_monitoring_error("os.getloadavg", exc))

    try:
        frequency = psutil.cpu_freq()
        if frequency is not None:
            sample["system_cpu_frequency_mhz"] = float(frequency.current)
    except (AttributeError, OSError, RuntimeError) as exc:
        errors.append(_monitoring_error("psutil.cpu_freq", exc))

    try:
        disk_io = psutil.disk_io_counters()
        if disk_io is not None:
            sample["system_disk_read_bytes"] = int(disk_io.read_bytes)
            sample["system_disk_write_bytes"] = int(disk_io.write_bytes)
    except (AttributeError, OSError, RuntimeError) as exc:
        errors.append(_monitoring_error("psutil.disk_io", exc))

    try:
        sample.update(
            query_nvidia_metrics(command_runner=command_runner, gpu_index=gpu_index)
        )
    except (OSError, subprocess.SubprocessError, RuntimeError) as exc:
        errors.append(_monitoring_error("nvidia-smi", exc))
    if errors:
        sample["_monitoring_errors"] = errors
    return sample


def summarize_samples(
    samples: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, int], dict[str, dict[str, float | int]]]:
    """Return non-null coverage and numeric aggregates for each metric field."""
    coverage: dict[str, int] = {}
    statistics: dict[str, dict[str, float | int]] = {}
    for field in SYSTEM_METRIC_FIELDS:
        values = [sample.get(field) for sample in samples if sample.get(field) is not None]
        coverage[field] = len(values)
        numeric = [
            float(value)
            for value in values
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        ]
        if numeric:
            statistics[field] = {
                "samples": len(numeric),
                "min": min(numeric),
                "max": max(numeric),
                "mean": sum(numeric) / len(numeric),
                "last": numeric[-1],
            }
    return coverage, statistics


class ResourceMonitor:
    """Collect telemetry on a background interval and always retain warning evidence."""

    def __init__(
        self,
        run_dir: Path,
        *,
        interval_seconds: float = DEFAULT_MONITOR_INTERVAL_SECONDS,
        sample_provider: Callable[[], dict[str, Any]] | None = None,
        gpu_index: int = 0,
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be greater than zero")
        self.run_dir = Path(run_dir)
        self.interval_seconds = float(interval_seconds)
        self.gpu_index = int(gpu_index)
        self.sample_provider = sample_provider or (
            lambda: collect_resource_sample(gpu_index=self.gpu_index)
        )
        self._samples: list[dict[str, Any]] = []
        self._errors: list[dict[str, str]] = []
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._started_at = _utc_now()

    def sample_once(self) -> bool:
        try:
            sample = self.sample_provider()
            normalized = {field: sample.get(field) for field in SYSTEM_METRIC_FIELDS}
            if normalized["timestamp"] is None:
                normalized["timestamp"] = _utc_now()
            with self._lock:
                self._samples.append(normalized)
                for error in sample.get("_monitoring_errors", []):
                    self._errors.append({"timestamp": _utc_now(), **error})
            return True
        except Exception as exc:  # Monitoring must never fail model execution.
            with self._lock:
                self._errors.append(
                    {
                        "timestamp": _utc_now(),
                        "type": type(exc).__name__,
                        "message": str(exc),
                    }
                )
            return False

    def _run(self) -> None:
        while not self._stop_event.is_set():
            self.sample_once()
            self._stop_event.wait(self.interval_seconds)

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("Resource monitor has already been started")
        self._thread = threading.Thread(
            target=self._run,
            name="resource-monitor",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> dict[str, Any]:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.interval_seconds + 1.0))
        with self._lock:
            samples = list(self._samples)
            errors = list(self._errors)

        self.run_dir.mkdir(parents=True, exist_ok=True)
        csv_buffer = io.StringIO(newline="")
        writer = csv.DictWriter(csv_buffer, fieldnames=SYSTEM_METRIC_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(samples)
        atomic_write_text(self.run_dir / "system_metrics.csv", csv_buffer.getvalue())

        field_coverage, statistics = summarize_samples(samples)
        summary = {
            "schema_version": "2.0",
            "status": "completed_with_warnings" if errors else "completed",
            "started_at": self._started_at,
            "stopped_at": _utc_now(),
            "interval_seconds": self.interval_seconds,
            "sample_count": len(samples),
            "errors": errors,
            "fields": list(SYSTEM_METRIC_FIELDS),
            "sources": {
                "process_system": "psutil",
                "gpu": "nvidia-smi",
            },
            "selected_gpu_index": self.gpu_index,
            "field_coverage": field_coverage,
            "statistics": statistics,
            "gpu_monitoring": {
                "tool": "nvidia-smi",
                "samples_with_data": sum(
                    sample["gpu_utilization_percent"] is not None for sample in samples
                ),
            },
        }
        atomic_write_json(self.run_dir / "monitoring.json", summary)
        return summary
