from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timedelta, timezone
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
    PreflightError,
    PreflightThresholds,
    collect_hardware,
    run_preflight,
    run_snapshot_smoke,
    run_synthetic_artifact_smoke,
    verify_preflight_report,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _passing_collectors(cache_manifest: Path, phase2_input: Path):
    high_free = 200 * 1024**3
    return {
        "hardware": lambda **_kwargs: {
            "status": "passed",
            "cpu": {
                "model": "Synthetic CPU",
                "sockets": 1,
                "physical_cores": 16,
                "logical_cpus": 32,
            },
            "ram": {"total_bytes": 140 * 1024**3},
            "gpus": [
                {
                    "index": 0,
                    "name": "Synthetic GPU",
                    "vram_bytes": 96 * 1024**3,
                    "driver_version": "999.0",
                }
            ],
            "nvidia_smi": {"status": "available"},
        },
        "software": lambda **_kwargs: {
            "status": "passed",
            "python": "3.12.11",
            "packages": {"torch": "2.12.1+cu130"},
        },
        "filesystems": lambda **_kwargs: {
            "status": "passed",
            "locations": {
                name: {"path": name, "free_bytes": high_free}
                for name in ("dataset", "cache", "artifacts", "snapshots", "final_archive")
            },
        },
        "git": lambda **_kwargs: {
            "status": "passed",
            "commit_sha": "d" * 40,
            "dirty": False,
            "dirty_summary": [],
            "git_lfs": {"available": True, "version": "git-lfs/3.7.0"},
        },
        "dataset": lambda **_kwargs: {
            "status": "passed",
            "official_csv_count": 8,
            "all_materialized": True,
            "files": [{"filename": f"file-{index}.csv", "sha256": str(index) * 64} for index in range(8)],
        },
        "phase2": lambda **_kwargs: {
            "status": "passed",
            "path": str(phase2_input.resolve()),
            "size_bytes": phase2_input.stat().st_size,
            "sha256": _sha256(phase2_input),
            "truth_labels_expected": True,
            "truth_labels_available": True,
            "truth_labels_valid": True,
            "sensitive_metadata_export_enabled": False,
        },
        "cache": lambda **_kwargs: {
            "status": "passed",
            "validation_status": "valid",
            "manifest_path": str(cache_manifest.resolve()),
            "manifest_sha256": _sha256(cache_manifest),
        },
        "cuda": lambda **_kwargs: {
            "status": "passed",
            "available": True,
            "device_count": 1,
            "tensor_operation": "passed",
        },
        "artifact_smoke": lambda **_kwargs: {
            "status": "passed",
            "run_dir": "synthetic-artifact",
            "manifest_sha256": "e" * 64,
        },
        "snapshot_smoke": lambda **_kwargs: {
            "status": "passed",
            "snapshot_manifest_sha256": "f" * 64,
            "incremental_noop_verified": True,
        },
    }


def _preflight_paths(tmp_path: Path) -> dict[str, Path]:
    paths = {
        "dataset_root": tmp_path / "dataset",
        "cache_root": tmp_path / "cache",
        "artifact_root": tmp_path / "artifacts",
        "snapshot_root": tmp_path / "snapshots",
    }
    for path in paths.values():
        path.mkdir(parents=True)
    return paths


def test_preflight_report_is_hash_covered_fresh_and_bound(tmp_path: Path) -> None:
    paths = _preflight_paths(tmp_path)
    spec = tmp_path / "campaign.json"
    spec.write_text(json.dumps({"campaign_spec_id": "synthetic", "entries": []}), encoding="utf-8")
    cache_manifest = paths["cache_root"] / "cache_manifest.json"
    cache_manifest.write_text('{"validation_status":"valid"}\n', encoding="utf-8")
    phase2_input = tmp_path / "phase2.csv"
    phase2_input.write_text("truth_y,value\n0,1\n1,2\n", encoding="utf-8")
    output = tmp_path / "preflight.json"
    now = datetime(2026, 7, 12, 12, 0, tzinfo=timezone.utc)

    report = run_preflight(
        output_path=output,
        campaign_spec=spec,
        phase2_input=phase2_input,
        expect_phase2_labels=True,
        now=now,
        collectors=_passing_collectors(cache_manifest, phase2_input),
        **paths,
    )

    assert report["status"] == "passed"
    assert report["bindings"]["cache_manifest_sha256"] == _sha256(cache_manifest)
    assert report["bindings"]["phase2_input_sha256"] == _sha256(phase2_input)
    assert report["report_sha256"]
    verified = verify_preflight_report(
        output,
        expected_campaign_spec_sha256=report["bindings"]["campaign_spec_sha256"],
        expected_dataset_root=paths["dataset_root"],
        expected_cache_root=paths["cache_root"],
        expected_artifact_root=paths["artifact_root"],
        expected_snapshot_root=paths["snapshot_root"],
        expected_cache_manifest_sha256=_sha256(cache_manifest),
        now=now + timedelta(hours=1),
    )
    assert verified["report_sha256"] == report["report_sha256"]

    tampered = json.loads(output.read_text(encoding="utf-8"))
    tampered["status"] = "failed"
    output.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(PreflightError, match="hash"):
        verify_preflight_report(output, now=now)


def test_preflight_persists_repository_relative_official_artifact_root(
    tmp_path: Path,
) -> None:
    repository_root = tmp_path / "repository"
    paths = {
        "dataset_root": repository_root / "datasets" / "CICIDS2017",
        "cache_root": repository_root / "cache" / "cicids2017",
        "artifact_root": repository_root / "runs" / "final_campaign",
        "snapshot_root": tmp_path / "external-exports",
    }
    for path in paths.values():
        path.mkdir(parents=True)
    spec = repository_root / "experiments" / "campaign.json"
    spec.parent.mkdir(parents=True)
    spec.write_text("{}", encoding="utf-8")
    cache_manifest = paths["cache_root"] / "cache_manifest.json"
    cache_manifest.write_text("{}", encoding="utf-8")
    phase2_input = repository_root / "pcaps" / "phase2.csv"
    phase2_input.parent.mkdir(parents=True)
    phase2_input.write_text("truth_y\n0\n", encoding="utf-8")
    output = paths["artifact_root"] / "preflight.json"
    now = datetime(2026, 7, 12, 12, 0, tzinfo=timezone.utc)

    report = run_preflight(
        output_path=output,
        campaign_spec=spec,
        phase2_input=phase2_input,
        now=now,
        repo_root=repository_root,
        collectors=_passing_collectors(cache_manifest, phase2_input),
        **paths,
    )

    assert report["bindings"]["artifact_root"] == "runs/final_campaign"
    verified = verify_preflight_report(
        output,
        expected_artifact_root=paths["artifact_root"],
        repository_root=repository_root,
        now=now,
    )
    assert verified["status"] == "passed"


def test_preflight_threshold_failure_and_staleness_block_verification(tmp_path: Path) -> None:
    paths = _preflight_paths(tmp_path)
    spec = tmp_path / "campaign.json"
    spec.write_text("{}", encoding="utf-8")
    cache_manifest = paths["cache_root"] / "cache_manifest.json"
    cache_manifest.write_text("{}", encoding="utf-8")
    phase2_input = tmp_path / "phase2.csv"
    phase2_input.write_text("truth_y\n0\n", encoding="utf-8")
    collectors = _passing_collectors(cache_manifest, phase2_input)
    collectors["hardware"] = lambda **_kwargs: {
        "status": "passed",
        "cpu": {"logical_cpus": 8},
        "ram": {"total_bytes": 64 * 1024**3},
        "gpus": [{"vram_bytes": 24 * 1024**3}],
        "nvidia_smi": {"status": "available"},
    }
    now = datetime(2026, 7, 12, 12, 0, tzinfo=timezone.utc)
    output = tmp_path / "failed-preflight.json"

    report = run_preflight(
        output_path=output,
        campaign_spec=spec,
        phase2_input=phase2_input,
        now=now,
        max_age_hours=2,
        collectors=collectors,
        **paths,
    )
    assert report["status"] == "failed"
    assert {failure["check"] for failure in report["failures"]} >= {
        "logical_cpus",
        "ram",
        "gpu_vram",
    }
    with pytest.raises(PreflightError, match="successful"):
        verify_preflight_report(output, now=now)

    passing_output = tmp_path / "passing-preflight.json"
    passing = run_preflight(
        output_path=passing_output,
        campaign_spec=spec,
        phase2_input=phase2_input,
        now=now,
        max_age_hours=2,
        collectors=_passing_collectors(cache_manifest, phase2_input),
        **paths,
    )
    assert passing["status"] == "passed"
    with pytest.raises(PreflightError, match="stale"):
        verify_preflight_report(passing_output, now=now + timedelta(hours=3))


def test_nvidia_smi_absence_is_recorded_without_crashing() -> None:
    def missing_runner(*_args, **_kwargs):
        raise FileNotFoundError("nvidia-smi")

    hardware = collect_hardware(command_runner=missing_runner)

    assert hardware["nvidia_smi"]["status"] == "unavailable"
    assert hardware["status"] == "failed"
    assert hardware["gpus"] == []


def test_preflight_hardware_preserves_threshold_keys_from_shared_inventory() -> None:
    inventory = {
        "cpu": {
            "model": "Synthetic CPU",
            "sockets": 1,
            "physical_cores": 16,
            "logical_cpus": 32,
        },
        "memory": {"total_bytes": 140 * 1024**3},
        "swap": {"total_bytes": 8 * 1024**3},
        "gpus": [
            {
                "index": 0,
                "name": "Synthetic GPU",
                "memory_total_bytes": 96 * 1024**3,
            }
        ],
        "runtime": {"wsl": False, "container": True},
        "cgroup": {"version": 2},
        "nvidia_smi": {"status": "available", "gpu_count": 1},
        "errors": [],
    }

    hardware = collect_hardware(inventory_collector=lambda **_kwargs: inventory)

    assert hardware["status"] == "passed"
    assert hardware["cpu"]["logical_cpus"] == 32
    assert hardware["ram"]["total_bytes"] == 140 * 1024**3
    assert hardware["gpus"][0]["vram_bytes"] == 96 * 1024**3
    assert hardware["runtime"]["container"] is True


def test_tiny_qrdqn_artifact_and_incremental_snapshot_smoke(
    tmp_path: Path,
    fake_model_factory,
) -> None:
    artifact_root = tmp_path / "artifacts"
    snapshot_root = tmp_path / "snapshots"
    spec = tmp_path / "campaign.json"
    spec.write_text("{}\n", encoding="utf-8")
    cache_manifest = tmp_path / "cache_manifest.json"
    cache_manifest.write_text("{}\n", encoding="utf-8")

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

    snapshot = run_snapshot_smoke(
        artifact_record=artifact,
        snapshot_root=snapshot_root,
        campaign_spec=spec,
        cache_manifest=cache_manifest,
    )
    assert snapshot["status"] == "passed"
    assert snapshot["incremental_noop_verified"] is True


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


def test_default_thresholds_match_approved_platform_contract() -> None:
    thresholds = PreflightThresholds()

    assert thresholds.min_logical_cpus == 16
    assert thresholds.min_ram_gib == 120
    assert thresholds.min_gpu_count == 1
    assert thresholds.min_vram_gib == 80
    assert thresholds.min_free_gib == 100


def test_preflight_cli_exposes_required_provider_neutral_paths(tmp_path: Path) -> None:
    from scripts.preflight_gpu_environment import parse_args

    args = parse_args(
        [
            "--dataset-root",
            str(tmp_path / "dataset"),
            "--cache-root",
            str(tmp_path / "cache"),
            "--artifact-root",
            str(tmp_path / "artifacts"),
            "--snapshot-root",
            str(tmp_path / "snapshots"),
        ]
    )

    assert args.dataset_root == tmp_path / "dataset"
    assert args.cache_root == tmp_path / "cache"
    assert args.artifact_root == tmp_path / "artifacts"
    assert args.snapshot_root == tmp_path / "snapshots"
    assert args.smoke_timesteps == DEFAULT_SMOKE_TIMESTEPS
    assert args.smoke_monitor_interval == DEFAULT_SMOKE_MONITOR_INTERVAL_SECONDS


def test_gpu_requirements_are_neutral_and_legacy_file_is_only_a_pointer() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    neutral = (repo_root / "requirements-gpu-cu130.txt").read_text(encoding="utf-8")
    legacy = (repo_root / "requirements-runpod-cu130.txt").read_text(encoding="utf-8")

    assert "torch==2.12.1+cu130" in neutral
    assert "psutil==7.2.2" in neutral
    assert "RunPod" not in neutral
    assert legacy.strip().endswith("-r requirements-gpu-cu130.txt")


def test_preflight_cli_reports_invalid_threshold_without_traceback(
    tmp_path: Path,
    capsys,
) -> None:
    from scripts.preflight_gpu_environment import main

    result = main(
        [
            "--dataset-root",
            str(tmp_path / "dataset"),
            "--cache-root",
            str(tmp_path / "cache"),
            "--artifact-root",
            str(tmp_path / "artifacts"),
            "--snapshot-root",
            str(tmp_path / "snapshots"),
            "--min-logical-cpus",
            "0",
        ]
    )

    assert result == 2
    assert "min_logical_cpus must be greater than zero" in capsys.readouterr().err
