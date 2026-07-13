# Complete Smoke Telemetry Implementation Plan

**Goal:** Record provider-neutral static hardware, dynamic resource use, resolved QRDQN configuration, actual execution, and discoverable TensorBoard values while making the real synthetic smoke large enough to emit training scalars.

**Architecture:** Add one shared system-inventory collector used by run artifacts and preflight, extend `ResourceMonitor` additively, generate `run_summary.json` from persisted authoritative artifacts, and raise the preflight smoke to the profile-derived 50,200-timestep floor. Preserve historical artifact verification and all existing field meanings.

**Tech Stack:** Python 3.12, psutil, PyTorch, `nvidia-smi`, Stable-Baselines3/QRDQN, TensorBoard, pytest, Ruff.

---

### Task 1: Shared static system inventory

**Files:**
- Create: `src/system_telemetry.py`
- Create: `tests/test_system_telemetry.py`

- [ ] **Step 1: Write failing CPU and cgroup parsing tests.**

```python
def test_parse_cpuinfo_prefers_linux_model_and_vendor() -> None:
    payload = parse_cpuinfo(
        "processor : 0\nvendor_id : GenuineIntel\n"
        "model name : Intel(R) Core(TM) i7-12700H\n"
    )
    assert payload == {
        "model": "Intel(R) Core(TM) i7-12700H",
        "vendor": "GenuineIntel",
    }


def test_collect_cgroup_v2_limits(tmp_path: Path) -> None:
    (tmp_path / "cpu.max").write_text("400000 100000\n", encoding="utf-8")
    (tmp_path / "memory.current").write_text("1024\n", encoding="utf-8")
    (tmp_path / "memory.max").write_text("4096\n", encoding="utf-8")
    limits = collect_cgroup_limits(tmp_path)
    assert limits["cpu"]["limit_cpus"] == 4.0
    assert limits["memory"] == {"current_bytes": 1024, "limit_bytes": 4096}
```

- [ ] **Step 2: Run `uv run pytest -q tests/test_system_telemetry.py`.**

Expected: collection fails because `src.system_telemetry` does not exist.

- [ ] **Step 3: Implement deterministic parsers.**

```python
def parse_cpuinfo(text: str) -> dict[str, str | None]:
    fields: dict[str, str] = {}
    for line in text.splitlines():
        if ":" in line:
            key, value = (part.strip() for part in line.split(":", 1))
            fields.setdefault(key.lower(), value)
    return {
        "model": fields.get("model name") or fields.get("hardware"),
        "vendor": fields.get("vendor_id") or fields.get("cpu implementer"),
    }
```

Implement `_read_text`, `_read_limit`, `_parse_cpu_max`, and `collect_cgroup_limits`. Interpret `max` as `None` so unlimited cgroups never become fabricated numeric limits.

- [ ] **Step 4: Add failing host-inventory shape tests.**

```python
inventory = collect_host_inventory(
    storage_paths={"artifacts": tmp_path},
    command_runner=_nvidia_runner(
        "0, NVIDIA RTX, GPU-abc, 00000000:01:00.0, 6144, 595.79, 115.0\n"
    ),
    cpuinfo_text="vendor_id : AuthenticAMD\nmodel name : AMD Ryzen Test CPU\n",
    cgroup_root=tmp_path / "missing-cgroup",
    torch_module=None,
)
assert inventory["cpu"]["model"] == "AMD Ryzen Test CPU"
assert inventory["memory"]["available_bytes"] > 0
assert inventory["gpus"][0]["uuid"] == "GPU-abc"
assert inventory["storage"]["artifacts"]["free_bytes"] > 0
```

- [ ] **Step 5: Implement `collect_host_inventory`.**

Return `schema_version`, `status`, `runtime`, `cpu`, `memory`, `swap`, `cgroup`, `gpus`, `torch_cuda_devices`, `storage`, `nvidia_smi`, and `errors`. Use Linux CPU information, platform/socket data, psutil, process affinity, CPU frequency, disk usage, cgroup v2, the NVIDIA query `index,name,uuid,pci.bus_id,memory.total,driver_version,power.limit`, and PyTorch CUDA properties. Each optional failure appends `{source, type, message}` and leaves unavailable values null.

- [ ] **Step 6: Run `uv run pytest -q tests/test_system_telemetry.py`.**

Expected: all system-inventory tests pass.

### Task 2: Environment and preflight reuse

**Files:**
- Modify: `src/run_artifacts.py`
- Modify: `src/gpu_preflight.py`
- Modify: `tests/test_run_artifacts.py`
- Modify: `tests/test_preflight.py`

- [ ] **Step 1: Write failing reuse tests.**

```python
metadata = collect_environment_metadata(
    repo_root=tmp_path,
    storage_paths={"artifacts": tmp_path},
    hardware_collector=lambda **_kwargs: expected_inventory,
    package_names=(),
)
assert metadata["hardware"] == expected_inventory
assert metadata["nvidia_smi"] == expected_inventory["nvidia_smi"]
```

```python
hardware = collect_hardware(inventory_collector=lambda **_kwargs: inventory)
assert hardware["cpu"]["logical_cpus"] == 32
assert hardware["ram"]["total_bytes"] == 140 * 1024**3
assert hardware["gpus"][0]["vram_bytes"] == 96 * 1024**3
```

- [ ] **Step 2: Run the selected tests.**

Run: `uv run pytest -q tests/test_run_artifacts.py tests/test_preflight.py -k "environment_metadata or hardware"`

Expected: failures show that the injected collector parameters are absent.

- [ ] **Step 3: Extend `collect_environment_metadata` additively.**

Add `storage_paths: Mapping[str, Path] | None` and a `hardware_collector` keyword. Store the inventory at `hardware` and retain top-level `platform`, `torch`, `nvidia_smi`, `packages`, `git`, and `threads`.

- [ ] **Step 4: Adapt the shared inventory to the existing preflight keys.**

```python
def collect_hardware(*, command_runner=subprocess.run, inventory_collector=collect_host_inventory):
    inventory = inventory_collector(command_runner=command_runner)
    gpus = [
        {**gpu, "vram_bytes": gpu.get("memory_total_bytes")}
        for gpu in inventory["gpus"]
    ]
    return {
        "status": "passed" if gpus and inventory["nvidia_smi"]["status"] == "available" else "failed",
        "cpu": inventory["cpu"],
        "ram": inventory["memory"],
        "gpus": gpus,
        "nvidia_smi": inventory["nvidia_smi"],
        "runtime": inventory["runtime"],
        "swap": inventory["swap"],
        "cgroup": inventory["cgroup"],
        "errors": inventory["errors"],
    }
```

- [ ] **Step 5: Run focused environment/preflight tests.**

Run: `uv run pytest -q tests/test_system_telemetry.py tests/test_run_artifacts.py tests/test_preflight.py -k "hardware or environment or nvidia"`

Expected: selected tests pass and threshold-consumed keys are unchanged.

### Task 3: Expanded dynamic resource samples

**Files:**
- Modify: `src/resource_monitor.py`
- Modify: `tests/test_resource_monitor.py`

- [ ] **Step 1: Write a failing ordered-field contract test.**

```python
assert SYSTEM_METRIC_FIELDS[:11] == EXISTING_SYSTEM_METRIC_FIELDS
for field in (
    "process_vms_bytes",
    "process_read_bytes",
    "system_ram_available_bytes",
    "system_swap_total_bytes",
    "system_disk_read_bytes",
    "gpu_memory_free_bytes",
    "gpu_performance_state",
):
    assert field in SYSTEM_METRIC_FIELDS
```

- [ ] **Step 2: Run `uv run pytest -q tests/test_resource_monitor.py -k metric_contract`.**

Expected: assertion fails on the first absent appended field.

- [ ] **Step 3: Append all approved fields without reordering the original eleven.**

Append process VMS/threads/read/write, load averages, CPU frequency, RAM available/free/percent, swap values, disk I/O, GPU memory utilization/free, power limit, clocks, fan, and performance state.

- [ ] **Step 4: Expand NVIDIA parsing with a failing exact-value test.**

```python
result.stdout = "85, 10, 1024, 5120, 6144, 75.5, 115, 72, 1800, 7000, N/A, P2\n"
metrics = query_nvidia_metrics(command_runner=lambda *_args, **_kwargs: result)
assert metrics["gpu_memory_free_bytes"] == 5120 * 1024**2
assert metrics["gpu_power_limit_watts"] == 115.0
assert metrics["gpu_sm_clock_mhz"] == 1800.0
assert metrics["gpu_fan_speed_percent"] is None
assert metrics["gpu_performance_state"] == "P2"
```

- [ ] **Step 5: Implement best-effort process/system sampling.**

Use `process.memory_info`, `process.num_threads`, `process.io_counters`, `os.getloadavg`, `psutil.cpu_freq`, `psutil.virtual_memory`, `psutil.swap_memory`, and `psutil.disk_io_counters`. A failed optional probe leaves its fields null and appends a structured monitoring warning.

- [ ] **Step 6: Run selected sample tests.**

Run: `uv run pytest -q tests/test_resource_monitor.py -k "metric_contract or nvidia_smi or resource_sample or absent_nvidia"`

Expected: selected tests pass.

### Task 4: Monitoring coverage and numeric summaries

**Files:**
- Modify: `src/resource_monitor.py`
- Modify: `tests/test_resource_monitor.py`

- [ ] **Step 1: Write a failing aggregation test.**

```python
summary = monitor.stop()
assert summary["schema_version"] == "2.0"
assert summary["field_coverage"]["system_cpu_percent"] == 2
assert summary["statistics"]["system_cpu_percent"] == {
    "samples": 2,
    "min": 25.0,
    "max": 75.0,
    "mean": 50.0,
    "last": 75.0,
}
```

- [ ] **Step 2: Run `uv run pytest -q tests/test_resource_monitor.py -k monitoring_summary`.**

Expected: fails because schema, coverage, and statistics are absent.

- [ ] **Step 3: Implement `summarize_samples`.**

```python
def summarize_samples(samples):
    coverage = {}
    statistics = {}
    for field in SYSTEM_METRIC_FIELDS:
        values = [row.get(field) for row in samples if row.get(field) is not None]
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
```

Add `schema_version="2.0"`, ordered fields, collection sources, GPU index, coverage, and statistics to `monitoring.json` while retaining every current key.

- [ ] **Step 4: Run `uv run pytest -q tests/test_resource_monitor.py`.**

Expected: all resource-monitor tests pass.

### Task 5: Actual execution and discoverable run summary

**Files:**
- Create: `src/run_summary.py`
- Create: `tests/test_run_summary.py`
- Modify: `src/qrdqn_experiment.py`
- Modify: `tests/conftest.py`
- Modify: `tests/test_qrdqn_experiment.py`

- [ ] **Step 1: Make the lightweight model expose actual timesteps and training tags.**

```python
self.num_timesteps = total_timesteps + 8
writer.add_scalar("train/learning_rate", 5e-5, self.num_timesteps)
writer.add_scalar("train/loss", 0.25, self.num_timesteps)
```

- [ ] **Step 2: Write and run a failing execution-accounting test.**

```python
assert config["training_execution"] == {
    "requested_timesteps": 2,
    "actual_timesteps": 10,
    "overshoot_timesteps": 8,
    "completion_reason": "requested_timesteps_reached",
}
assert timing["phases"]["training"]["units"] == 10
```

Run: `uv run pytest -q tests/test_qrdqn_experiment.py -k actual_timesteps`

Expected: fails because `training_execution` is absent.

- [ ] **Step 3: Record `model.num_timesteps` after learning.**

Set training throughput units from the actual integer count. If the model does not expose an integer, record null plus `model_returned_early_or_unreported`; never substitute the request.

- [ ] **Step 4: Write a failing run-summary test.**

```python
summary = build_run_summary(fresh_main_run)
assert summary["training_hyperparameters"]["learning_rate"] == 5e-5
assert summary["hardware"] == environment["hardware"]
learning_rate = summary["tensorboard_scalars"]["train/learning_rate"][0]
assert learning_rate["last_value"] == 5e-5
assert learning_rate["last_step"] == 10
```

- [ ] **Step 5: Implement `build_run_summary` using persisted artifacts only.**

```python
return {
    "schema_version": "1.0",
    "run": run_identity(config),
    "training_hyperparameters": config["training_hyperparameters"],
    "training_execution": config["training_execution"],
    "device_selected": environment.get("device_selected"),
    "hardware": environment["hardware"],
    "metrics": metrics,
    "timing": timing,
    "monitoring": monitoring,
    "tensorboard_scalars": scalar_last_values(run_dir, scalar_manifest),
    "artifact_paths": config["artifact_paths"],
}
```

Read every scalar CSV named by `tensorboard_scalar_export_manifest.json` and group records by tag with event directory, CSV path, sample count, last step, and last value.

- [ ] **Step 6: Require and write `run_summary.json` before manifest completion.**

Complete and persist `config.json`, write `run_summary.json`, and then seal the manifest. Historical manifests stay valid because they declare their own requirements.

- [ ] **Step 7: Run summary and runner tests.**

Run: `uv run pytest -q tests/test_run_summary.py tests/test_qrdqn_experiment.py`

Expected: all tests pass and new manifests inventory `run_summary.json`.

### Task 6: Fixed real smoke and scalar gate

**Files:**
- Modify: `src/gpu_preflight.py`
- Modify: `scripts/preflight_gpu_environment.py`
- Modify: `tests/test_preflight.py`

- [ ] **Step 1: Write failing defaults and visibility tests.**

```python
assert DEFAULT_SMOKE_TIMESTEPS == 50_200
assert DEFAULT_SMOKE_MONITOR_INTERVAL_SECONDS == 1.0
result = run_synthetic_artifact_smoke(
    artifact_root=tmp_path,
    model_factory=fake_model_factory,
)
assert result["scientific_result"] is False
assert result["run_summary"]["training_hyperparameters"]["learning_rate"] == 5e-5
assert "train/learning_rate" in result["run_summary"]["tensorboard_scalars"]
assert "train/loss" in result["run_summary"]["tensorboard_scalars"]
```

- [ ] **Step 2: Run selected smoke tests.**

Run: `uv run pytest -q tests/test_preflight.py -k "smoke_defaults or smoke_result"`

Expected: fails because the current smoke requests two timesteps and returns no summary.

- [ ] **Step 3: Implement the profile-derived floor.**

```python
_SMOKE_HYPERPARAMETERS = MAIN_V1_PROFILE.qrdqn_hyperparams()
DEFAULT_SMOKE_TIMESTEPS = int(
    _SMOKE_HYPERPARAMETERS["learning_starts"]
    + 2 * _SMOKE_HYPERPARAMETERS["train_freq"]
)
DEFAULT_SMOKE_MONITOR_INTERVAL_SECONDS = 1.0
REQUIRED_SMOKE_SCALARS = {"train/learning_rate", "train/loss"}
```

Accept `smoke_timesteps` and `monitor_interval` arguments. Reject fewer than 50,200 timesteps. Verify both scalar tags after the manifest, then return the complete `run_summary` with `scientific_result=false`.

Set the smoke's runtime-only QRDQN verbosity to zero while preserving the
existing verbosity default for campaign runs. This keeps the terminal focused
on the returned complete summary instead of thousands of rollout tables.

- [ ] **Step 4: Add CLI propagation.**

```python
parser.add_argument("--smoke-timesteps", type=int, default=DEFAULT_SMOKE_TIMESTEPS)
parser.add_argument(
    "--smoke-monitor-interval",
    type=float,
    default=DEFAULT_SMOKE_MONITOR_INTERVAL_SECONDS,
)
```

Pass both values through `run_preflight` to the artifact-smoke collector.

- [ ] **Step 5: Run `uv run pytest -q tests/test_preflight.py`.**

Expected: all tests pass through the existing lightweight model injection; tests use no fake clock and do not execute 50,200 real timesteps.

### Task 7: Maintained operator documentation

**Files:**
- Modify: `docs/gpu_experimental_environment.md`
- Modify: `docs/reproducibility.md`
- Test: `tests/test_provider_neutrality.py`

- [ ] **Step 1: Document the artifact map.**

```text
environment.json      static runtime-visible hardware/software inventory
system_metrics.csv    timestamped process/system/GPU samples
monitoring.json       coverage, warnings, and numeric summaries
run_summary.json      resolved hyperparameters, actual execution, metrics, and scalar index
```

- [ ] **Step 2: Document smoke semantics and inspection commands.**

State that the non-scientific smoke defaults to 50,200 timesteps and one-second monitoring, crosses `learning_starts`, requires learning-rate/loss exports, and has hardware-dependent duration expected under a few minutes on the WSL2 development host.

```bash
RUN_DIR="$(find "$HOME/tfg-qrdqn-test-artifacts/preflight-smoke" -type f -name run_summary.json -printf '%T@ %h\n' | sort -nr | head -n1 | cut -d' ' -f2-)"
python -m json.tool "$RUN_DIR/environment.json"
python -m json.tool "$RUN_DIR/run_summary.json"
column -s, -t < <(head -n 6 "$RUN_DIR/system_metrics.csv")
```

- [ ] **Step 3: Run `uv run pytest -q tests/test_provider_neutrality.py`.**

Expected: documentation-path tests pass.

### Task 8: Final validation

**Files:**
- Validate all files changed by Tasks 1–7.

- [ ] **Step 1: Run focused tests.**

```powershell
uv run pytest -q tests/test_system_telemetry.py tests/test_resource_monitor.py tests/test_run_artifacts.py tests/test_run_summary.py tests/test_qrdqn_experiment.py tests/test_preflight.py tests/test_provider_neutrality.py
```

Expected: all focused tests pass.

- [ ] **Step 2: Run `uv run pytest -q`.**

Expected: the complete unit-test suite passes.

- [ ] **Step 3: Run static, lock, and whitespace checks.**

```powershell
uv run ruff check .
uv lock --check
git diff --check
git status --short
```

Expected: every check passes and status contains only intended files.

- [ ] **Step 4: Confirm protected paths are unchanged.**

```powershell
git status --short -- memoria report runs models datasets
git diff --name-only main...HEAD
git diff --name-only
```

Expected: no modifications under `memoria/`, `report/`, historical `runs/`, models, or datasets; no real CICIDS2017 experiment has run.

- [ ] **Step 5: Leave implementation changes uncommitted for review.**

Do not push or open a pull request. Report the existing design commit separately and provide the WSL2 real-smoke command as the remaining host-only validation.
