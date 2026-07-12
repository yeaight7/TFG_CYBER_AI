from __future__ import annotations

import importlib
import json

import pytest


def _module():
    try:
        return importlib.import_module("src.resource_monitor")
    except ModuleNotFoundError:
        pytest.fail("Phase 3 module 'src.resource_monitor' is missing", pytrace=False)


def _api(name: str):
    module = _module()
    assert hasattr(module, name), f"src.resource_monitor.{name} is missing"
    return getattr(module, name)


def _sample(timestamp: str = "2026-07-12T10:00:00+00:00"):
    return {
        "timestamp": timestamp,
        "process_cpu_percent": 12.5,
        "system_cpu_percent": 37.5,
        "process_rss_bytes": 1_024,
        "system_ram_used_bytes": 2_048,
        "system_ram_total_bytes": 4_096,
        "gpu_utilization_percent": None,
        "gpu_memory_used_bytes": None,
        "gpu_memory_total_bytes": None,
        "gpu_power_watts": None,
        "gpu_temperature_c": None,
    }


def test_resource_monitor_writes_csv_and_summary(tmp_path):
    monitor_type = _api("ResourceMonitor")
    monitor = monitor_type(tmp_path, interval_seconds=30, sample_provider=_sample)

    assert monitor.sample_once() is True
    summary = monitor.stop()

    assert summary["status"] == "completed"
    assert summary["sample_count"] == 1
    assert (tmp_path / "system_metrics.csv").read_text(encoding="utf-8").splitlines()[1].startswith(
        "2026-07-12T10:00:00+00:00,12.5,37.5"
    )
    assert json.loads((tmp_path / "monitoring.json").read_text(encoding="utf-8")) == summary


def test_resource_monitor_failure_is_non_fatal_and_recorded(tmp_path):
    monitor_type = _api("ResourceMonitor")

    def fail_sample():
        raise RuntimeError("telemetry unavailable")

    monitor = monitor_type(tmp_path, interval_seconds=30, sample_provider=fail_sample)

    assert monitor.sample_once() is False
    summary = monitor.stop()

    assert summary["status"] == "completed_with_warnings"
    assert summary["sample_count"] == 0
    assert summary["errors"][0]["type"] == "RuntimeError"
    assert summary["errors"][0]["message"] == "telemetry unavailable"
    assert (tmp_path / "system_metrics.csv").is_file()


def test_absent_nvidia_smi_returns_null_gpu_fields():
    collect = _api("collect_resource_sample")

    def missing_tool(*_args, **_kwargs):
        raise FileNotFoundError("nvidia-smi")

    sample = collect(command_runner=missing_tool)

    assert sample["gpu_utilization_percent"] is None
    assert sample["gpu_memory_used_bytes"] is None
    assert sample["gpu_memory_total_bytes"] is None
    assert sample["gpu_power_watts"] is None
    assert sample["gpu_temperature_c"] is None
    assert sample["_monitoring_errors"][0]["tool"] == "nvidia-smi"
    assert sample["_monitoring_errors"][0]["type"] == "FileNotFoundError"


def test_absent_nvidia_smi_warning_is_persisted_by_monitor(tmp_path):
    monitor_type = _api("ResourceMonitor")
    sample = _sample()
    sample["_monitoring_errors"] = [
        {"tool": "nvidia-smi", "type": "FileNotFoundError", "message": "not installed"}
    ]
    monitor = monitor_type(tmp_path, sample_provider=lambda: sample)

    assert monitor.sample_once() is True
    summary = monitor.stop()

    assert summary["status"] == "completed_with_warnings"
    assert summary["errors"][0]["tool"] == "nvidia-smi"
    assert summary["errors"][0]["type"] == "FileNotFoundError"


def test_nvidia_smi_metrics_are_parsed_without_units():
    query = _api("query_nvidia_metrics")

    class Result:
        stdout = "85, 1024, 81920, 250.5, 72\n"

    metrics = query(command_runner=lambda *_args, **_kwargs: Result())

    assert metrics == {
        "gpu_utilization_percent": 85.0,
        "gpu_memory_used_bytes": 1_073_741_824,
        "gpu_memory_total_bytes": 85_899_345_920,
        "gpu_power_watts": 250.5,
        "gpu_temperature_c": 72.0,
    }
