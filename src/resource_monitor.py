"""Non-fatal process, system, and optional NVIDIA GPU resource monitoring."""

from __future__ import annotations

import csv
import io
import subprocess
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import psutil

from src.run_artifacts import atomic_write_json, atomic_write_text


DEFAULT_MONITOR_INTERVAL_SECONDS = 30.0
MIB = 1024 * 1024
SYSTEM_METRIC_FIELDS = (
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


def query_nvidia_metrics(
    *,
    command_runner: Callable[..., Any] = subprocess.run,
    gpu_index: int = 0,
) -> dict[str, float | int | None]:
    command = [
        "nvidia-smi",
        f"--id={gpu_index}",
        "--query-gpu=utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    result = command_runner(command, check=True, capture_output=True, text=True)
    line = next((line for line in result.stdout.splitlines() if line.strip()), "")
    values = [value.strip() for value in line.split(",")]
    if len(values) != 5:
        raise RuntimeError("nvidia-smi returned an unexpected resource metric row")
    utilization, memory_used_mib, memory_total_mib, power, temperature = map(
        _float_or_none, values
    )
    return {
        "gpu_utilization_percent": utilization,
        "gpu_memory_used_bytes": (
            None if memory_used_mib is None else int(memory_used_mib * MIB)
        ),
        "gpu_memory_total_bytes": (
            None if memory_total_mib is None else int(memory_total_mib * MIB)
        ),
        "gpu_power_watts": power,
        "gpu_temperature_c": temperature,
    }


def collect_resource_sample(
    *,
    process: psutil.Process | None = None,
    command_runner: Callable[..., Any] = subprocess.run,
    gpu_index: int = 0,
) -> dict[str, Any]:
    process = process or psutil.Process()
    virtual_memory = psutil.virtual_memory()
    sample = {
        "timestamp": _utc_now(),
        "process_cpu_percent": float(process.cpu_percent(interval=None)),
        "system_cpu_percent": float(psutil.cpu_percent(interval=None)),
        "process_rss_bytes": int(process.memory_info().rss),
        "system_ram_used_bytes": int(virtual_memory.used),
        "system_ram_total_bytes": int(virtual_memory.total),
        "gpu_utilization_percent": None,
        "gpu_memory_used_bytes": None,
        "gpu_memory_total_bytes": None,
        "gpu_power_watts": None,
        "gpu_temperature_c": None,
    }
    try:
        sample.update(
            query_nvidia_metrics(command_runner=command_runner, gpu_index=gpu_index)
        )
    except (OSError, subprocess.SubprocessError, RuntimeError) as exc:
        sample["_monitoring_errors"] = [
            {
                "tool": "nvidia-smi",
                "type": type(exc).__name__,
                "message": str(exc),
            }
        ]
    return sample


class ResourceMonitor:
    """Collect telemetry on a background interval and always retain warning evidence."""

    def __init__(
        self,
        run_dir: Path,
        *,
        interval_seconds: float = DEFAULT_MONITOR_INTERVAL_SECONDS,
        sample_provider: Callable[[], dict[str, Any]] = collect_resource_sample,
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be greater than zero")
        self.run_dir = Path(run_dir)
        self.interval_seconds = float(interval_seconds)
        self.sample_provider = sample_provider
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

        summary = {
            "status": "completed_with_warnings" if errors else "completed",
            "started_at": self._started_at,
            "stopped_at": _utc_now(),
            "interval_seconds": self.interval_seconds,
            "sample_count": len(samples),
            "errors": errors,
            "gpu_monitoring": {
                "tool": "nvidia-smi",
                "samples_with_data": sum(
                    sample["gpu_utilization_percent"] is not None for sample in samples
                ),
            },
        }
        atomic_write_json(self.run_dir / "monitoring.json", summary)
        return summary
