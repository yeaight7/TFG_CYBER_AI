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

Use a provider-neutral Linux GPU host. The default preflight thresholds are at least 16 logical CPUs, 120 GiB RAM, one CUDA GPU with 80 GiB VRAM, and 100 GiB free at the configured artifact/snapshot capacity. These are launch gates, not claims about an unverified host.

```bash
git clone https://github.com/yeaight7/TFG_CYBER_AI.git
cd TFG_CYBER_AI
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-gpu-cu130.txt
```

For development and unit tests, `uv sync` uses `pyproject.toml` and `uv.lock`. The compatibility requirements filename is documented in [reproducibility.md](reproducibility.md).

Materialise the eight curated official CSVs under `datasets/CICIDS2017/` and verify their hashes against [the repository dataset table](../README.md#provenance-and-integrity). Do not commit local raw datasets, generated caches, or provider credentials.

Choose durable, absolute host paths for these roots:

```bash
export DATASET_ROOT=/path/to/datasets/CICIDS2017
export CACHE_ROOT=/path/to/cache/cicids2017
export ARTIFACT_ROOT=/path/to/artifacts
export SNAPSHOT_ROOT=/path/to/durable-snapshots
export PREFLIGHT_REPORT=/path/to/preflight.json
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

## Preflight

Run preflight in the same isolated environment and against the exact campaign inputs:

```bash
python scripts/preflight_gpu_environment.py \
  --dataset-root "$DATASET_ROOT" \
  --cache-root "$CACHE_ROOT" \
  --artifact-root "$ARTIFACT_ROOT" \
  --snapshot-root "$SNAPSHOT_ROOT" \
  --campaign-spec experiments/final_experiment_campaign.json \
  --phase2-input pcaps/lab_capture_traffic.csv \
  --expect-phase2-labels \
  --runtime-benchmark /path/to/runtime-benchmark.json \
  --output "$PREFLIGHT_REPORT"
```

Real campaign execution rejects a missing, failed, stale, or cache-mismatched preflight report. A changed Phase 2 input requires a new or explicitly amended report. Actual hardware, driver, CUDA, storage, dataset, cache, monitoring, and snapshot checks remain host-only until this command succeeds on the final host.

## Campaign dry-run and execution

Dry-run first; it creates no campaign artifacts:

```bash
python scripts/run_campaign.py experiments/final_experiment_campaign.json \
  --campaign-id final-experiment-v1-<timestamp> \
  --dataset-root "$DATASET_ROOT" \
  --cache-root "$CACHE_ROOT" \
  --artifact-root "$ARTIFACT_ROOT" \
  --snapshot-root "$SNAPSHOT_ROOT" \
  --preflight-report "$PREFLIGHT_REPORT" \
  --phase2-input pcaps/lab_capture_traffic.csv \
  --phase2-input-sha256 <SHA256> \
  --dry-run
```

After successful preflight and reviewed dry-run, use the same resolved inputs with `--resume`. Optional `--stage <STAGE_ID>` or `--run <LOGICAL_RUN_ID>` selection never executes missing dependencies automatically.

```bash
python scripts/run_campaign.py experiments/final_experiment_campaign.json \
  --campaign-id final-experiment-v1-<timestamp> \
  --dataset-root "$DATASET_ROOT" \
  --cache-root "$CACHE_ROOT" \
  --artifact-root "$ARTIFACT_ROOT" \
  --snapshot-root "$SNAPSHOT_ROOT" \
  --preflight-report "$PREFLIGHT_REPORT" \
  --phase2-input pcaps/lab_capture_traffic.csv \
  --phase2-input-sha256 <SHA256> \
  --resume
```

The runner dispatches sequential subprocesses. It skips only checksum-validated completed runs, retains failed/interrupted attempts, and writes retries to new attempt directories.

## Snapshot and final bundle

Campaign progression verifies an incremental snapshot after each completed physical execution. Manual export interfaces are:

```bash
python scripts/export_campaign.py snapshot \
  --campaign-dir "$ARTIFACT_ROOT/final-experiment-v1-<timestamp>" \
  --destination "$SNAPSHOT_ROOT"

python scripts/export_campaign.py bundle \
  --campaign-dir "$ARTIFACT_ROOT/final-experiment-v1-<timestamp>" \
  --destination /path/to/final-bundles
```

Exports copy evidence; they never move or delete source campaign data. The final archive is reopened and checksum-verified.

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
- `requirements-runpod-cu130.txt` is a historical compatibility filename that includes `requirements-gpu-cu130.txt`.
- Historical artifacts keep their original environment and hardware metadata unchanged.

