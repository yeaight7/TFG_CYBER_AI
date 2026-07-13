# Complete Smoke Telemetry Design

**Date:** 2026-07-13
**Status:** Approved for implementation
**Scope:** Provider-neutral runtime-visible hardware inventory, resource monitoring, and a discoverable real QRDQN preflight smoke

## Objective

Make every future QRDQN run, including the preflight smoke, self-describing enough to answer:

- what CPU, RAM, GPU, storage, operating environment, and runtime limits were available;
- how those resources behaved throughout the run;
- which resolved scientific profile and hyperparameters were used;
- how many timesteps actually executed and how long each phase took;
- which TensorBoard scalars were produced and what their latest values were;
- which telemetry fields were unavailable and why.

The design records only resources visible to the Linux process. On WSL2 this means the WSL VM allocation; on RunPod it means the container-visible allocation. It does not attempt to inspect the Windows host or provider control plane.

## Chosen Approach

Use a comprehensive, shared telemetry contract rather than a minimal field patch or raw host-command dumps.

- A single provider-neutral hardware collector feeds both per-run `environment.json` and the full `preflight_report.json`.
- `ResourceMonitor` samples the process, system, storage, and selected NVIDIA GPU during execution.
- A generated `run_summary.json` makes the important resolved configuration, hardware, monitoring aggregates, metrics, timings, and TensorBoard scalars discoverable without replacing their authoritative source artifacts.
- The synthetic smoke executes a fixed real QRDQN workload large enough to cross `learning_starts` and flush training scalars. It is not governed by a fake clock or wall-clock callback.

Raw `lshw`, complete `/proc` dumps, `nvidia-smi -q`, Windows-host queries, provider APIs, and network-interface inventories are deliberately excluded. They are noisy, platform-specific, potentially sensitive, and unnecessary for scientific reconstruction.

## Static Hardware and Runtime Inventory

`environment.json` keeps every existing field and gains an additive `hardware` object. The inventory is captured before training and contains best-effort values plus explicit collection errors.

### CPU

- model and vendor from `/proc/cpuinfo` on Linux/WSL, with portable fallbacks;
- architecture;
- socket count;
- physical core count;
- logical CPU count visible to the runtime;
- process-available CPU count from affinity/cpuset restrictions when supported;
- current, minimum, and maximum reported frequency;
- cgroup CPU quota/period and derived CPU limit when available.

Generic values such as `x86_64` must not replace a more specific `/proc/cpuinfo` model name.

### Memory and swap

- total, available, used, free, cached, buffers, and percentage at capture time;
- swap total, used, free, and percentage;
- cgroup memory current and limit values when available;
- explicit distinction between runtime-visible memory and any external physical host capacity.

### GPU

For every NVIDIA GPU visible through `nvidia-smi`:

- index, name, UUID, and PCI bus identifier;
- total VRAM;
- driver version;
- power limit when supported.

For CUDA devices visible through PyTorch:

- device name;
- compute capability;
- total memory;
- multiprocessor count when exposed by the installed PyTorch version.

The existing `nvidia_smi` and `torch` keys remain present for compatibility.

### Runtime, virtualization, and storage

- hostname, kernel/platform fields, WSL detection, and best-effort container/cgroup detection;
- repository, run-artifact, dataset, and cache filesystem total/used/free capacity for paths supplied to the runner;
- Python executable/version, package versions, CUDA/cuDNN versions, Git commit/dirty state, and thread settings already captured by the existing contract.

## Dynamic Resource Samples

`system_metrics.csv` preserves its existing columns and appends the following best-effort fields:

### Process and CPU

- process CPU percentage;
- system CPU percentage;
- one-, five-, and fifteen-minute system load;
- process RSS and VMS bytes;
- process thread count;
- process read/write bytes;
- current CPU frequency when available.

### RAM, swap, and storage I/O

- system RAM total, used, available, free, and percentage;
- swap total, used, free, and percentage;
- system disk read/write byte counters.

### Selected GPU

- GPU utilization percentage;
- memory-controller utilization percentage;
- VRAM used, free, and total bytes;
- power draw and power limit;
- temperature;
- SM and memory clocks;
- fan percentage and performance state when supported.

Unsupported values are stored as empty CSV cells. They are never guessed or replaced with fabricated zeros.

`monitoring.json` keeps its current status/error fields and gains:

- telemetry schema version and ordered column list;
- selected GPU index and collection sources;
- sample count and coverage per field;
- minimum, maximum, arithmetic mean, and last value for numeric fields with data;
- structured warnings for unavailable fields or tools.

## Discoverable Run Summary

Every new QRDQN run writes `run_summary.json` before sealing its schema-3 manifest. Historical manifests remain valid because requirements are declared per manifest.

The summary is generated from the authoritative run artifacts and contains:

- algorithm, profile ID/hash, seeds, split mode, device, and run identifiers;
- the complete resolved training hyperparameter object, including learning rate;
- requested timesteps, actual `model.num_timesteps`, overshoot if any, and completion reason;
- phase timings and throughput;
- final evaluation metrics;
- concise hardware inventory and monitoring aggregates;
- every exported TensorBoard scalar tag with CSV path, sample count, last step, and last value;
- paths to the authoritative config, environment, monitoring, metrics, timing, and scalar artifacts.

The summary improves visibility but does not become an independent scientific source of truth. Its values must be derived from the already-written artifacts in the same run directory.

## Real Synthetic Smoke Workload

`run_synthetic_artifact_smoke` remains explicitly non-scientific and continues using synthetic 152-dimensional observations and the frozen `main-v1` profile.

- Default requested workload: **50,200 timesteps**.
- Rationale: `main-v1` has `learning_starts=50,000` and `train_freq=100`; two additional collection blocks permit at least one real training update and a subsequent logger flush.
- Default monitoring interval: **1 second**.
- The timestep count is configurable for controlled host diagnosis but may not be set below the profile-derived scalar-production floor.
- There is no fake clock, sleeping workload, duration callback, or guessed timestep-to-seconds conversion.
- Runtime is hardware-dependent. The target is under a few minutes on the WSL2/RTX 3060 development host and materially shorter on the final GPU host, but no false wall-clock guarantee is recorded.

The smoke must fail its verification step if the real run does not produce both `train/learning_rate` and `train/loss` scalar exports. Its returned JSON includes the complete `run_summary.json` payload so the existing `json.dumps(result, indent=2)` command displays hyperparameters, hardware, monitoring summaries, timings, metrics, and scalar values directly in the terminal.

Unit tests continue to use injected lightweight model factories where appropriate; they do not use fake clocks and do not execute the 50,200-timestep workload.

## Data Flow

1. Resolve the run config and scientific profile.
2. Capture shared static hardware/runtime inventory into `environment.json`.
3. Start `ResourceMonitor` and sample at the configured interval.
4. Execute QRDQN and obtain the actual model timestep count.
5. Stop monitoring and persist `system_metrics.csv` plus `monitoring.json` summaries.
6. Persist model, preprocessing, predictions, evaluation metrics, timing, and TensorBoard exports.
7. Build `run_summary.json` only from those persisted artifacts.
8. Seal the artifact manifest and checksums, then verify them.
9. For preflight smoke, require training scalar presence and return the complete summary with `scientific_result=false`.

## Failure Handling

- Missing optional telemetry remains non-fatal to a scientific run but produces null/empty fields and structured warnings.
- Failure to write required artifact files remains fatal and produces failed-run evidence through the existing runner path.
- Full preflight keeps its existing readiness behavior: missing required CUDA/NVIDIA capability or failed thresholds makes the preflight fail.
- Malformed command output is recorded with tool, error type, and message; it is not silently coerced.
- The summary generator rejects missing or malformed authoritative artifacts rather than inventing values.

## Compatibility

- Existing `environment.json`, `system_metrics.csv`, and `monitoring.json` keys/columns retain their meanings.
- New environment keys and CSV columns are additive.
- `collect_hardware` retains the preflight keys currently consumed by threshold validation.
- Historical schema-2 and schema-3 manifests remain verifiable.
- `run_summary.json` is required only by manifests produced after this change.
- The frozen `main-v1` scientific profile, reward matrix, seeds, split semantics, dataset behavior, and campaign matrix are unchanged.

## Testing and Validation

Implementation follows red-green-refactor tests for:

- Linux/WSL CPU model and vendor parsing with portable fallbacks;
- affinity, cgroup CPU, and cgroup memory limits;
- RAM/swap and storage inventory;
- NVIDIA and PyTorch device metadata with partial-tool failure;
- expanded CSV field ordering, nullable samples, and numeric monitoring summaries;
- actual-versus-requested timestep accounting;
- generated `run_summary.json` consistency with authoritative artifacts;
- complete hyperparameter visibility, including learning rate;
- fixed smoke timestep floor and one-second monitoring default;
- required `train/learning_rate` and `train/loss` scalar verification;
- old manifest compatibility and unchanged preflight thresholds.

Required final checks are focused telemetry/preflight/runner tests, the complete unit-test suite, Ruff, `uv lock --check`, `git diff --check`, and a host-only real smoke command for the user. The implementation environment will not pretend to have validated WSL2/RunPod hardware when that host-only run has not been executed there.

## Non-goals

- full CICIDS2017 training or campaign execution;
- changing scientific hyperparameters or selecting them from telemetry;
- measuring Windows-host resources outside WSL2;
- provider APIs, credentials, uploads, dashboards, or cloud monitoring agents;
- raw full-system dumps or collection of network addresses and secrets;
- modifying historical runs, models, datasets, `memoria/`, `report/`, or thesis placeholders.
