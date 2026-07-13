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


def _sample(timestamp: str = "2026-07-12T10:00:00+00:00", **overrides):
    sample = {
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
    sample.update(overrides)
    return sample


def test_metric_contract_preserves_old_prefix_and_adds_complete_telemetry():
    fields = _api("SYSTEM_METRIC_FIELDS")

    assert fields[:11] == (
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
    assert {
        "process_vms_bytes",
        "process_threads",
        "process_read_bytes",
        "process_write_bytes",
        "system_load_1m",
        "system_cpu_frequency_mhz",
        "system_ram_available_bytes",
        "system_ram_free_bytes",
        "system_ram_percent",
        "system_swap_total_bytes",
        "system_disk_read_bytes",
        "gpu_memory_utilization_percent",
        "gpu_memory_free_bytes",
        "gpu_power_limit_watts",
        "gpu_sm_clock_mhz",
        "gpu_memory_clock_mhz",
        "gpu_fan_speed_percent",
        "gpu_performance_state",
    }.issubset(fields)


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


def test_monitoring_summary_records_schema_coverage_and_aggregates(tmp_path):
    monitor_type = _api("ResourceMonitor")
    samples = iter(
        [
            _sample(system_cpu_percent=25.0),
            _sample(
                timestamp="2026-07-12T10:00:01+00:00",
                system_cpu_percent=75.0,
            ),
        ]
    )
    monitor = monitor_type(
        tmp_path,
        interval_seconds=1,
        sample_provider=lambda: next(samples),
    )

    assert monitor.sample_once() is True
    assert monitor.sample_once() is True
    summary = monitor.stop()

    assert summary["schema_version"] == "2.0"
    assert summary["fields"] == list(_api("SYSTEM_METRIC_FIELDS"))
    assert summary["sources"] == {
        "process_system": "psutil",
        "gpu": "nvidia-smi",
    }
    assert summary["selected_gpu_index"] == 0
    assert summary["field_coverage"]["system_cpu_percent"] == 2
    assert summary["statistics"]["system_cpu_percent"] == {
        "samples": 2,
        "min": 25.0,
        "max": 75.0,
        "mean": 50.0,
        "last": 75.0,
    }


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
    assert sample["gpu_memory_utilization_percent"] is None
    assert sample["gpu_memory_free_bytes"] is None
    assert sample["gpu_power_limit_watts"] is None
    assert sample["gpu_performance_state"] is None
    nvidia_errors = [
        error
        for error in sample["_monitoring_errors"]
        if error["tool"] == "nvidia-smi"
    ]
    assert nvidia_errors[0]["type"] == "FileNotFoundError"


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
        stdout = "85, 10, 1024, 80896, 81920, 250.5, 300, 72, 1800, 7000, N/A, P2\n"

    metrics = query(command_runner=lambda *_args, **_kwargs: Result())

    assert metrics == {
        "gpu_utilization_percent": 85.0,
        "gpu_memory_utilization_percent": 10.0,
        "gpu_memory_used_bytes": 1_073_741_824,
        "gpu_memory_free_bytes": 84_825_604_096,
        "gpu_memory_total_bytes": 85_899_345_920,
        "gpu_power_watts": 250.5,
        "gpu_power_limit_watts": 300.0,
        "gpu_temperature_c": 72.0,
        "gpu_sm_clock_mhz": 1800.0,
        "gpu_memory_clock_mhz": 7000.0,
        "gpu_fan_speed_percent": None,
        "gpu_performance_state": "P2",
    }


def test_resource_sample_records_extended_process_and_system_fields():
    collect = _api("collect_resource_sample")

    def missing_tool(*_args, **_kwargs):
        raise FileNotFoundError("nvidia-smi")

    sample = collect(command_runner=missing_tool)

    assert sample["process_vms_bytes"] > 0
    assert sample["process_threads"] > 0
    assert sample["process_read_bytes"] is None or sample["process_read_bytes"] >= 0
    assert sample["system_ram_available_bytes"] > 0
    assert sample["system_ram_free_bytes"] >= 0
    assert sample["system_ram_percent"] >= 0
    assert sample["system_swap_total_bytes"] >= 0
