from __future__ import annotations

import json
import subprocess
import tomllib
from pathlib import Path

import pytest

from scripts.benchmark_experimental_runtime import (
    parse_thread_config,
    run_runtime_benchmark,
)
from src.artifact_integrity import verify_artifact_manifest
from src.gpu_preflight import (
    DEFAULT_SMOKE_MONITOR_INTERVAL_SECONDS,
    DEFAULT_SMOKE_TIMESTEPS,
    run_synthetic_artifact_smoke,
)


def test_tiny_qrdqn_artifact_smoke(
    tmp_path: Path,
    fake_model_factory,
) -> None:
    artifact_root = tmp_path / "artifacts"
    artifact = run_synthetic_artifact_smoke(
        artifact_root=artifact_root,
        model_factory=fake_model_factory,
    )
    assert artifact["status"] == "passed"
    assert artifact["requested_timesteps"] == 50_200
    assert artifact["run_summary"]["training_hyperparameters"]["learning_rate"] == 5e-5
    assert "train/learning_rate" in artifact["run_summary"]["tensorboard_scalars"]
    assert "train/loss" in artifact["run_summary"]["tensorboard_scalars"]
    assert verify_artifact_manifest(Path(artifact["run_dir"]))["status"] == "completed"



def test_smoke_defaults_cross_learning_start_and_use_one_second_monitoring() -> None:
    assert DEFAULT_SMOKE_TIMESTEPS == 50_200
    assert DEFAULT_SMOKE_MONITOR_INTERVAL_SECONDS == 1.0


def test_smoke_rejects_workload_below_training_scalar_floor(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least 50200"):
        run_synthetic_artifact_smoke(
            artifact_root=tmp_path,
            smoke_timesteps=50_199,
        )


def test_runtime_benchmark_uses_one_subprocess_per_thread_config(tmp_path: Path) -> None:
    calls: list[list[str]] = []

    def fake_runner(command, **_kwargs):
        calls.append(list(command))
        config = command[command.index("--worker-config") + 1]
        threads, interop = parse_thread_config(config)
        payload = {
            "status": "completed",
            "requested": {"torch_threads": threads, "torch_inter_op_threads": interop},
            "effective": {"torch_threads": threads, "torch_inter_op_threads": interop},
            "measurements": {"preprocessing_rows_per_second": 1.0},
        }
        return subprocess.CompletedProcess(command, 0, stdout=json.dumps(payload), stderr="")

    output = tmp_path / "benchmark.json"
    report = run_runtime_benchmark(
        output_path=output,
        thread_configs=["1:1", "4:1", "8:1"],
        command_runner=fake_runner,
    )

    assert report["status"] == "completed"
    assert len(report["results"]) == 3
    assert len(calls) == 3
    assert all("--worker-config" in command for command in calls)
    assert "selected_config" not in report
    assert json.loads(output.read_text(encoding="utf-8")) == report


@pytest.mark.parametrize("value", ["0:1", "1:0", "4", "a:b", "1:2:3"])
def test_runtime_benchmark_rejects_invalid_thread_config(value: str) -> None:
    with pytest.raises(ValueError):
        parse_thread_config(value)


def test_project_lock_uses_cuda_on_linux_and_keeps_cpu_wheels() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    project = tomllib.loads((repo_root / "pyproject.toml").read_text(encoding="utf-8"))
    lock = tomllib.loads((repo_root / "uv.lock").read_text(encoding="utf-8"))

    assert "torch==2.15.0" in project["project"]["dependencies"]
    sources = project["tool"]["uv"]["sources"]["torch"]
    assert {"index": "pytorch-cu132", "marker": "sys_platform == 'linux'"} in sources
    assert {"index": "pytorch-cpu", "marker": "sys_platform != 'linux'"} in sources
    torch_wheels = {pkg["version"] for pkg in lock["package"] if pkg["name"] == "torch"}
    assert {"2.15.0+cu132", "2.15.0+cpu"} <= torch_wheels
