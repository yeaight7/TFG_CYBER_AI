from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

from src.system_telemetry import (
    collect_cgroup_limits,
    collect_host_inventory,
    parse_cpuinfo,
)


def _nvidia_runner(stdout: str, *, returncode: int = 0, stderr: str = ""):
    def run(command, **_kwargs):
        return subprocess.CompletedProcess(command, returncode, stdout=stdout, stderr=stderr)

    return run


def test_parse_cpuinfo_prefers_linux_model_and_vendor() -> None:
    payload = parse_cpuinfo(
        "processor : 0\n"
        "vendor_id : GenuineIntel\n"
        "model name : Intel(R) Core(TM) i7-12700H\n"
    )

    assert payload == {
        "model": "Intel(R) Core(TM) i7-12700H",
        "vendor": "GenuineIntel",
    }


def test_parse_cpuinfo_supports_arm_fallback_fields() -> None:
    payload = parse_cpuinfo("CPU implementer : 0x41\nHardware : Neoverse Test\n")

    assert payload == {"model": "Neoverse Test", "vendor": "0x41"}


def test_collect_cgroup_v2_limits(tmp_path: Path) -> None:
    (tmp_path / "cpu.max").write_text("400000 100000\n", encoding="utf-8")
    (tmp_path / "memory.current").write_text("1024\n", encoding="utf-8")
    (tmp_path / "memory.max").write_text("4096\n", encoding="utf-8")

    limits = collect_cgroup_limits(tmp_path)

    assert limits == {
        "version": 2,
        "cpu": {
            "quota_us": 400000,
            "period_us": 100000,
            "limit_cpus": 4.0,
        },
        "memory": {"current_bytes": 1024, "limit_bytes": 4096},
    }


def test_collect_cgroup_unlimited_values_remain_null(tmp_path: Path) -> None:
    (tmp_path / "cpu.max").write_text("max 100000\n", encoding="utf-8")
    (tmp_path / "memory.max").write_text("max\n", encoding="utf-8")

    limits = collect_cgroup_limits(tmp_path)

    assert limits["cpu"] == {
        "quota_us": None,
        "period_us": 100000,
        "limit_cpus": None,
    }
    assert limits["memory"]["limit_bytes"] is None


def test_host_inventory_records_runtime_visible_resources(tmp_path: Path) -> None:
    inventory = collect_host_inventory(
        storage_paths={"artifacts": tmp_path},
        command_runner=_nvidia_runner(
            "0, NVIDIA RTX, GPU-abc, 00000000:01:00.0, 6144, 595.79, 115.0\n"
        ),
        cpuinfo_text=(
            "vendor_id : AuthenticAMD\n"
            "model name : AMD Ryzen Test CPU\n"
            "physical id : 0\n"
        ),
        cgroup_root=tmp_path / "missing-cgroup",
        torch_module=None,
    )

    assert inventory["schema_version"] == "1.0"
    assert inventory["cpu"]["model"] == "AMD Ryzen Test CPU"
    assert inventory["cpu"]["vendor"] == "AuthenticAMD"
    assert inventory["cpu"]["sockets"] == 1
    assert inventory["cpu"]["logical_cpus"]
    assert inventory["memory"]["total_bytes"] > 0
    assert inventory["memory"]["available_bytes"] > 0
    assert inventory["swap"]["total_bytes"] >= 0
    assert inventory["gpus"] == [
        {
            "index": 0,
            "name": "NVIDIA RTX",
            "uuid": "GPU-abc",
            "pci_bus_id": "00000000:01:00.0",
            "memory_total_bytes": 6 * 1024**3,
            "driver_version": "595.79",
            "power_limit_watts": 115.0,
        }
    ]
    assert inventory["nvidia_smi"]["available"] is True
    assert inventory["nvidia_smi"]["status"] == "available"
    assert inventory["nvidia_smi"]["gpu_count"] == 1
    assert inventory["nvidia_smi"]["gpus"] == inventory["gpus"]
    assert inventory["storage"]["artifacts"]["free_bytes"] > 0
    assert inventory["torch_cuda_devices"] == []


def test_host_inventory_records_optional_nvidia_failure(tmp_path: Path) -> None:
    def missing_nvidia(*_args, **_kwargs):
        raise FileNotFoundError("nvidia-smi")

    inventory = collect_host_inventory(
        storage_paths={"artifacts": tmp_path},
        command_runner=missing_nvidia,
        cpuinfo_text="model name : Synthetic CPU\n",
        cgroup_root=tmp_path / "missing-cgroup",
        torch_module=SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)),
    )

    assert inventory["status"] == "completed_with_warnings"
    assert inventory["gpus"] == []
    assert inventory["nvidia_smi"]["status"] == "unavailable"
    assert inventory["errors"][0]["source"] == "nvidia-smi"
    assert inventory["errors"][0]["type"] == "FileNotFoundError"
