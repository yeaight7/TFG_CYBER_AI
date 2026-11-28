# GPU Experimental Environment

This is the maintained setup and operating guide for the final experiment campaign. It is provider-neutral: host rental, remote access, account, hostname, and mount details are operational choices and are not scientific-method fields.

The campaign has not been executed yet. Commands in this guide prepare, validate, dry-run, execute, snapshot, and aggregate future evidence; they do not describe measured campaign results.

## Scientific contract

- Use the frozen `main-v1` profile and its recorded content hash.
- Preserve 76 canonical feature values plus 76 missingness-mask values: 152 observation dimensions.
- Preserve `0 = BENIGN/PERMIT`, `1 = ATTACK/BLOCK`, and rewards `tp=1.5`, `fp=-2.0`, `fn=-5.0`, `omission=0.0`.
- `split_seed` controls random partitioning and nested train subsets; `model_seed` controls model, environment, and runtime RNG.
- Fit a new scaler on each run's final training partition only. Transform its test partition with that scaler.
- Cache canonical unscaled arrays only. Never cache a fitted global scaler.
- Execute one experiment at a time. QRDQN and Random Forest campaign jobs remain sequential.
- Treat model-only checkpoints as recovery diagnostics. Interrupted official training starts a new physical attempt; it is not exact mid-training continuation.

## Fresh campaign MAIN versus historical MAIN

`qrdqn_main_random_full_s42_m42` will be the fresh campaign MAIN: a new 3,000,000-timestep physical execution using the final GPU experimental environment, `split_seed=42`, and `model_seed=42`.

The committed `MAIN_qrdqn_cicids2017_canonical_full_random_20260609_193655` is the historical MAIN. Its artifacts and metrics remain historical evidence. It is not the fresh campaign MAIN, not an alias target, and not one of the campaign's 22 new primary model-training executions. See [results.md](results.md) for measured historical results.

## Exact campaign matrix

The locked specification is [final_experiment_campaign.json](../experiments/final_experiment_campaign.json). It contains 22 new primary model-training executions, five auxiliary validation, analysis, and inference jobs, and two aliases. These produce 24 logical primary-training result points without duplicating physical evidence.

### QRDQN primary-training points

| Logical ID | Partition / subset | Split seed | Model seed | Timesteps | Execution |
|---|---|---:|---:|---:|---|
| `qrdqn_main_random_full_s42_m42` | Full random train / fixed random test | 42 | 42 | 3,000,000 | New; fresh campaign MAIN |
| `qrdqn_day_full_s42_m42` | Monday–Wednesday / Thursday–Friday | N/A to file selection | 42 | 3,000,000 | New |
| `qrdqn_ladder_100k_s42_m42` | Nested 100,000-row train / fixed random test | 42 | 42 | 132,474 | New |
| `qrdqn_ladder_250k_s42_m42` | Nested 250,000-row train / fixed random test | 42 | 42 | 331,185 | New |
| `qrdqn_ladder_500k_s42_m42` | Nested 500,000-row train / fixed random test | 42 | 42 | 662,370 | New |
| `qrdqn_ladder_1m_s42_m42` | Nested 1,000,000-row train / fixed random test | 42 | 42 | 1,324,741 | New |
| `qrdqn_ladder_2m_s42_m42` | Nested 2,000,000-row train / fixed random test | 42 | 42 | 2,649,482 | New |
| `qrdqn_ladder_full_s42_m42` | Full random train / fixed random test | 42 | 42 | 3,000,000 | Alias of fresh campaign MAIN |
| `qrdqn_seed_1m_s42_m42` | Same nested 1,000,000 rows / fixed random test | 42 | 42 | 1,324,741 | Alias of 1M ladder run |
| `qrdqn_seed_1m_s42_m43` | Same nested 1,000,000 rows / fixed random test | 42 | 43 | 1,324,741 | New |
| `qrdqn_seed_1m_s42_m44` | Same nested 1,000,000 rows / fixed random test | 42 | 44 | 1,324,741 | New |
| `qrdqn_seed_1m_s42_m45` | Same nested 1,000,000 rows / fixed random test | 42 | 45 | 1,324,741 | New |
| `qrdqn_seed_1m_s42_m46` | Same nested 1,000,000 rows / fixed random test | 42 | 46 | 1,324,741 | New |
| `qrdqn_holdout_webattacks_m42` | Hold out WebAttacks CSV; train on other seven | N/A to file selection | 42 | 1,000,000 | New |
| `qrdqn_holdout_infilteration_m42` | Hold out Infilteration CSV; train on other seven | N/A to file selection | 42 | 1,000,000 | New |
| `qrdqn_holdout_portscan_m42` | Hold out PortScan CSV; train on other seven | N/A to file selection | 42 | 1,000,000 | New |
| `qrdqn_holdout_ddos_m42` | Hold out DDoS CSV; train on other seven | N/A to file selection | 42 | 1,000,000 | New |

### Random Forest primary-training executions

| Logical ID | Partition / subset | Split seed | Model seed | Execution |
|---|---|---:|---:|---|
| `rf_random_full_s42_m42` | Full random train / fixed random test | 42 | 42 | New |
| `rf_random_1m_s42_m42` | Nested 1,000,000-row train / fixed random test | 42 | 42 | New |
| `rf_day_full_s42_m42` | Monday–Wednesday / Thursday–Friday | N/A to file selection | 42 | New |
| `rf_holdout_webattacks_m42` | Hold out WebAttacks CSV; train on other seven | N/A to file selection | 42 | New |
| `rf_holdout_infilteration_m42` | Hold out Infilteration CSV; train on other seven | N/A to file selection | 42 | New |
| `rf_holdout_portscan_m42` | Hold out PortScan CSV; train on other seven | N/A to file selection | 42 | New |
| `rf_holdout_ddos_m42` | Hold out DDoS CSV; train on other seven | N/A to file selection | 42 | New |

There is no Random Forest six-point ladder and no Random Forest multi-seed study.

### Auxiliary jobs

| Logical ID | Purpose | Training budget |
|---|---|---:|
| `main_direct_validation` | Directly compare fresh MAIN predictions with persisted fixed-test labels | None |
| `main_bootstrap_ci` | Fixed-test sampling confidence intervals from fresh MAIN predictions | None |
| `main_duplicate_analysis` | Exact duplicate and cross-split analysis for the fresh MAIN partition | None |
| `shuffled_label_validation_s42_m42` | Lightweight anti-leakage control | 10,000 timesteps |
| `phase2_fresh_main` | Offline laboratory-flow inference using fresh MAIN artifacts | None |

## Mandatory interpretation caveats

The four selected CSV holdouts are a targeted four-holdout generalisation study, not exhaustive eight-fold leave-one-CSV-out. They are not claimed to be the only CICIDS2017 CSVs containing attacks.

The model-seed block measures **seed sensitivity under a fixed 1M-row / 1,324,741-timestep budget** for model seeds 42–46. It does not estimate variance of the 3M MAIN execution; seed 42 reuses the 1M ladder physical execution.

Bootstrap intervals measure fixed-test sampling precision for one trained model, not training-seed variability. Random-split results measure in-distribution performance and do not replace day or targeted-holdout generalisation evidence. Phase 2 remains a separate offline laboratory-domain evaluation and must not be merged with CICIDS2017 internal-test metrics.

## Environment setup

The locked Linux campaign environment uses Python 3.12, PyTorch 2.15.0+cu132, and SB3/sb3-contrib 2.9.2. Use a CUDA 13.2-compatible host and install the project environment from the lock.

```bash
git clone https://github.com/yeaight7/TFG_CYBER_AI.git
cd TFG_CYBER_AI
uv sync --locked --python 3.12
source .venv/bin/activate
```

Each run records the environment it actually uses. See [reproducibility.md](reproducibility.md) for dev extras and the earlier CUDA 13.0 comparison stack.

Materialise the eight curated official CSVs under `datasets/CICIDS2017/` and verify their hashes against [the repository dataset table](../README.md#provenance-and-integrity). Do not commit local raw datasets, generated caches, or provider credentials.

Keep the official campaign artifacts beneath the clone. Dataset/cache inputs and the external export destination remain configurable filesystem paths:

```bash
export DATASET_ROOT=/path/to/datasets/CICIDS2017
export CACHE_ROOT=/path/to/cache/cicids2017
export ARTIFACT_ROOT=runs/final_campaign
export SNAPSHOT_ROOT=../final-campaign-exports
```

## Cache

Build one canonical unscaled shard per official CSV, then validate all source and array hashes:

```bash
python scripts/build_cicids_cache.py build \
  --dataset-root "$DATASET_ROOT" \
  --cache-root "$CACHE_ROOT" \
  --workers 8

python scripts/build_cicids_cache.py validate \
  --dataset-root "$DATASET_ROOT" \
  --cache-root "$CACHE_ROOT"
```

Building creates missing valid shards. Replacing stale data requires the explicit `--rebuild-stale` flag. Official campaign runs use `cache_policy=require`.

## Optional runtime benchmark

Thread benchmarks run in isolated subprocesses and do not select scientific settings automatically:

```bash
python scripts/benchmark_experimental_runtime.py \
  --output /path/to/runtime-benchmark.json \
  --thread-config 1:1 4:1 8:1 16:1
```

Run the probe with the locked campaign environment to measure the real training path. To compare stacks, repeat it in the CUDA 13.0 reference environment on the same allocation:

```bash
python scripts/benchmark_experimental_runtime.py --campaign-qrdqn --device auto \
  --thread-config 1:1 4:1 8:1 16:1 \
  --batch-sizes 2048 4096 8192 16384 32768
```

This uses a seeded synthetic dataset, the frozen MAIN QRDQN profile, and 50,400 timesteps. It measures warm-up rollout, learned-policy rollout, and optimizer updates separately. Inference includes NumPy-to-device input and device-to-NumPy action transfers. Each batch size gets a warm-up and three timed predictions with exact action comparisons against the 4,096 and 8,192 defaults. Each configuration runs in its own process. The probe writes `config.json` and `metrics.json` under a unique `runs/runtime-probes/` ID. These files are runtime evidence.

For a fixed trained model, append `--inference-run-dir <verified-model-run-dir>`. Add `--inference-observations <prepared-float32.npy>` to reuse an evaluation or Phase 2 input after its normal preprocessing. Use the same model and observation hashes for both stacks. Consider only parity-passing batch sizes and keep the default when the timing spread obscures the difference.

The 15 primary QRDQN runs request 20,399,216 timesteps. SB3 rounds these to 20,399,600 steps across the runs, including 750,000 steps before learning begins, and makes 3,929,920 optimizer updates. Estimate the QRDQN loop time from the sum of warm-up steps, learned-policy steps, and updates divided by their respective measured rates. Then add preprocessing, evaluation, checkpointing, exports, the seven Random Forest runs, and five auxiliary jobs. Use native host timings for disk- and CPU-heavy work. The 200-quantile loss alone has 81,920,000 pairwise elements per update, so VRAM size and advertised FLOPS give limited information about campaign wall time.

## Optional QRDQN smoke and run telemetry

The standalone synthetic smoke exercises 50,200 timesteps with one-second resource sampling and exports the learning rate and loss. Run it when diagnosing a training environment.

To run the smoke and print its summary:

```bash
uv run python - <<'PY'
import json
from pathlib import Path
from src.gpu_preflight import run_synthetic_artifact_smoke

result = run_synthetic_artifact_smoke(
    artifact_root=Path.home() / "tfg-qrdqn-test-artifacts",
)
print(json.dumps(result, indent=2))
PY
```

Each new QRDQN attempt contains these complementary evidence files:

| Artifact | Purpose |
|---|---|
| `environment.json` | Static runtime-visible CPU, RAM, swap, cgroup, storage, GPU, CUDA, package, Git, OS, and thread inventory |
| `system_metrics.csv` | Timestamped process, CPU, RAM, swap, disk-I/O, GPU-utilisation, VRAM, power, clock, fan, and temperature samples |
| `monitoring.json` | Collection status, warnings, field coverage, and min/max/mean/last numeric summaries |
| `run_summary.json` | Resolved hyperparameters, requested/actual timesteps, device, hardware, metrics, timings, monitoring summary, and latest value of every exported TensorBoard scalar |
| `config.json` | Authoritative resolved run request, frozen profile, seeds, split metadata, and artifact paths |
| `artifact_manifest.json` / `SHA256SUMS` | Required-file contract, inventory, sizes, and integrity hashes |

The returned smoke JSON embeds `run_summary.json`, so learning rate and other resolved hyperparameters and scalar values appear directly in the terminal. The smoke suppresses per-rollout SB3 tables; normal campaign runs retain their existing verbosity. To inspect the newest saved smoke without copying its generated identifier:

```bash
RUN_DIR="$(find "$HOME/tfg-qrdqn-test-artifacts/preflight-smoke" -type f -name run_summary.json -printf '%T@ %h\n' | sort -nr | head -n1 | cut -d' ' -f2-)"
python -m json.tool "$RUN_DIR/environment.json"
python -m json.tool "$RUN_DIR/run_summary.json"
column -s, -t < <(head -n 6 "$RUN_DIR/system_metrics.csv")
```

On WSL2, CPU counts, RAM, cgroup limits, and filesystem capacity describe the Linux runtime allocation available to the experiment, not unallocated Windows host resources. The same rule applies to container-visible resources on a GPU provider.

## Campaign dry-run and execution

Dry-run first; it creates no campaign artifacts:

```bash
python scripts/run_campaign.py experiments/final_experiment_campaign.json \
  --campaign-id final-experiment-v1-<timestamp> \
  --dataset-root "$DATASET_ROOT" \
  --cache-root "$CACHE_ROOT" \
  --artifact-root "$ARTIFACT_ROOT" \
  --snapshot-root "$SNAPSHOT_ROOT" \
  --phase2-input pcaps/lab_capture_traffic.csv \
  --phase2-input-sha256 <SHA256> \
  --dry-run
```

Use the same inputs for execution and resume. Supply the Phase 2 path and hash when starting a campaign that will include Phase 2. The runner binds the cache and Phase 2 hashes to that campaign. Optional `--stage <STAGE_ID>` or `--run <LOGICAL_RUN_ID>` selection never executes missing dependencies automatically.

```bash
python scripts/run_campaign.py experiments/final_experiment_campaign.json \
  --campaign-id final-experiment-v1-<timestamp> \
  --dataset-root "$DATASET_ROOT" \
  --cache-root "$CACHE_ROOT" \
  --artifact-root "$ARTIFACT_ROOT" \
  --snapshot-root "$SNAPSHOT_ROOT" \
  --phase2-input pcaps/lab_capture_traffic.csv \
  --phase2-input-sha256 <SHA256> \
  --resume
```

After comparing parity and throughput, set `--eval-batch-size` for QRDQN evaluation, shuffled-label validation, and direct MAIN validation. Set `--phase2-batch-size` for Phase 2. The defaults are 8,192 and 4,096 respectively. `--torch-threads` and `--torch-inter-op-threads` configure the PyTorch subprocesses when supplied. Repeat the same options on resume. The dry-run lists the resolved settings and the affected commands.

The runner dispatches sequential subprocesses. It skips only checksum-validated completed runs, retains failed/interrupted attempts, and writes retries to new attempt directories.

Canonical physical runs use this on-repository layout:

```text
runs/final_campaign/<CAMPAIGN_ID>/attempts/<LOGICAL_RUN_ID>/attempt-<N>/...
```

## Per-run export, optional snapshot, and final bundle

After every completed and validated physical run, the runner exports the complete run directory and creates a verified per-run archive. Logical aliases do not create duplicate exports. With the variables above, the external layout is:

```text
$SNAPSHOT_ROOT/
├── runs/final_campaign/<CAMPAIGN_ID>/attempts/<LOGICAL_RUN_ID>/attempt-<N>/...
└── tarballs/runs/final_campaign/<CAMPAIGN_ID>/attempts/<LOGICAL_RUN_ID>/
    ├── attempt-<N>.tar.gz
    ├── attempt-<N>.tar.gz.sha256
    └── attempt-<N>.export.json
```

The directory copy and tarball contain every generated run-local file, including ignored and untracked outputs. Extracting a per-run tarball at repository root restores its canonical `runs/final_campaign/.../attempt-<N>/` directory:

```bash
tar -xzf "$SNAPSHOT_ROOT/tarballs/runs/final_campaign/<CAMPAIGN_ID>/attempts/<LOGICAL_RUN_ID>/attempt-<N>.tar.gz" -C .
```

The runner retries a failed export on `--resume` without rerunning the validated physical job. Manual per-run retry and optional campaign-wide snapshot/bundle interfaces are:

```bash
python scripts/export_campaign.py run \
  --run-dir "$ARTIFACT_ROOT/<CAMPAIGN_ID>/attempts/<LOGICAL_RUN_ID>/attempt-<N>" \
  --destination "$SNAPSHOT_ROOT" \
  --repository-root .

python scripts/export_campaign.py snapshot \
  --campaign-dir "$ARTIFACT_ROOT/<CAMPAIGN_ID>" \
  --destination "$SNAPSHOT_ROOT/campaign-snapshots/<CAMPAIGN_ID>"

python scripts/export_campaign.py bundle \
  --campaign-dir "$ARTIFACT_ROOT/<CAMPAIGN_ID>" \
  --destination "$SNAPSHOT_ROOT/campaign-bundles/<CAMPAIGN_ID>.tar.gz"
```

Exports copy evidence; they never move, delete, or mutate canonical campaign data. Per-run archives are reopened and verified against a complete file inventory and SHA-256 sidecar. The destination may be a sibling directory on the same filesystem or any mounted filesystem path. Its existence alone does not prove off-host or independent durability; the operator remains responsible for manual download/recovery handling and Git/LFS publication.

## Aggregation and future figures

Run aggregation only after all required primary and auxiliary evidence validates:

```bash
python scripts/aggregate_campaign.py \
  --campaign-dir "$ARTIFACT_ROOT/final-experiment-v1-<timestamp>" \
  --output-dir /path/to/validated-aggregates

python scripts/generate_campaign_figures.py \
  --aggregate-dir /path/to/validated-aggregates \
  --output-dir /path/to/future-figures
```

Aggregation rejects incomplete, corrupt, failed, historical-substitute, or provenance-incompatible evidence. Figure generation also refuses incomplete aggregates. Output must remain outside `memoria/` and `report/`; no thesis placeholder becomes complete until separately authorised work uses real validated campaign artifacts.

## Compatibility and historical records

- [runpod_main_experiment.md](runpod_main_experiment.md) is retained only as a historical compatibility pointer for the committed historical MAIN.
- Historical artifacts keep their original environment and hardware metadata unchanged.
