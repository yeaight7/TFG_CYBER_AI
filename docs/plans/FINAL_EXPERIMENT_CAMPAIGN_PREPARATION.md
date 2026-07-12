# Final Experimental Campaign Preparation Plan
**Version 1.0.0** · Approved implementation plan · 2026-07-11

---

## AI READING INSTRUCTION

**[SPEC]**
- Read every `[SPEC]` block before proposing or implementing a phase.
- This plan authorises no implementation by itself. A later user request must name the phase to execute.
- Execute only the requested phase, verify its acceptance criteria, report its stopping point, and stop.
- Never continue automatically into a later phase, even when the next dependency is obvious.
- Historical artifacts, `memoria/`, `report/`, and pending thesis placeholders are outside the implementation scope.
- Create a new branch for each phase. Resume at the first open task, on that phase's branch (`chore/yeaight7/final-experiment-phase-<N>-<slug>`)

---

## 1. Campaign Objective and Methodological Boundaries

**[SPEC]**
Prepare the repository for a final, artifact-backed experimental campaign on a provider-neutral Linux GPU experimental platform with approximately:

- 16 vCPUs;
- 140 GB RAM;
- 96 GB VRAM;
- one experiment at a time;
- an ephemeral host requiring verified incremental snapshots.

The preparation work must deliver deterministic data contracts, explicit seed ownership, reusable unscaled canonical data, complete run evidence, safe run-level recovery, host preflight, future aggregation, and provider-neutral operating instructions. Implementation tasks may run unit tests, static checks, synthetic-data tests, and very small smoke runs only. They must not execute a full CICIDS2017 training campaign.

**[SPEC]**
Scientific invariants:

- Preserve the canonical observation contract: 76 canonical values plus 76 missingness-mask values, for 152 dimensions.
- Preserve `0 = BENIGN/PERMIT` and `1 = ATTACK/BLOCK`.
- Preserve the reward matrix exactly: `tp=1.5`, `fp=-2.0`, `fn=-5.0`, `omission=0.0`.
- Preserve the current official MAIN QRDQN hyperparameters exactly.
- Fit every scaler only on that run's final training partition; transform test data with the same fitted scaler.
- Cache canonical observations before scaling. Never cache a globally fitted scaler.
- Keep official QRDQN training on one environment. Any future multi-environment performance experiment must be optional, separately tested, scientifically evaluated, and disabled by default.
- Run QRDQN and Random Forest sequentially. No simultaneous official experiments.
- Keep historical artifacts and Git history unchanged.
- Do not modify `memoria/` or `report/` during campaign preparation.
- Do not fill, remove, or reinterpret pending thesis figure/table placeholders before real campaign artifacts exist.
- Do not fabricate metrics or convert missing experimental evidence into claims.

### 1.1 Authoritative MAIN profile

**[SPEC]**
Create one shared, versioned profile, provisionally identified as `main-v1`, containing:

| Field | Value |
|---|---:|
| policy | `MlpPolicy` |
| net architecture | `[1024, 1024, 512]` |
| quantiles | `200` |
| learning rate | `5e-5` |
| replay buffer | `1,000,000` |
| learning starts | `50,000` |
| batch size | `2,048` |
| gamma | `0.0` |
| tau | `1.0` |
| train frequency | `100` |
| gradient steps | `20` |
| target update interval | `10,000` |
| exploration initial epsilon | `1.0` |
| exploration final epsilon | `0.02` |
| exploration fraction | `0.10` |
| max gradient norm | `10.0` |
| reward configuration | `tp=1.5`, `fp=-2.0`, `fn=-5.0`, `omission=0.0` |

The profile must have a stable identifier and deterministic content hash. Timesteps, split definition, seeds, checkpoint policy, thread settings, paths, and monitoring interval are run/runtime settings rather than scientific profile fields.

---

## 2. Exact Campaign Matrix and Reuse Relationships

### 2.1 QRDQN

**[SPEC]**

| Logical run | Train partition | Test partition | Split seed | Model seed | Timesteps | Physical execution |
|---|---|---|---:|---:|---:|---|
| `qrdqn_main_random_full_s42_m42` | Full random train | Fixed random test | 42 | 42 | 3,000,000 | New campaign MAIN |
| `qrdqn_day_full_s42_m42` | Monday–Wednesday | Thursday–Friday | Fixed definition | 42 | 3,000,000 | New |
| `qrdqn_ladder_100k_s42_m42` | Nested 100,000-row subset | Fixed random test | 42 | 42 | 132,474 | New |
| `qrdqn_ladder_250k_s42_m42` | Nested 250,000-row subset | Fixed random test | 42 | 42 | 331,185 | New |
| `qrdqn_ladder_500k_s42_m42` | Nested 500,000-row subset | Fixed random test | 42 | 42 | 662,370 | New |
| `qrdqn_ladder_1m_s42_m42` | Nested 1,000,000-row subset | Fixed random test | 42 | 42 | 1,324,741 | New |
| `qrdqn_ladder_2m_s42_m42` | Nested 2,000,000-row subset | Fixed random test | 42 | 42 | 2,649,482 | New |
| `qrdqn_ladder_full_s42_m42` | Full random train | Fixed random test | 42 | 42 | 3,000,000 | Alias of fresh campaign MAIN |
| `qrdqn_seed_1m_s42_m42` | Nested 1,000,000-row subset | Fixed random test | 42 | 42 | 1,324,741 | Alias of 1M ladder run |
| `qrdqn_seed_1m_s42_m43` | Same 1M rows | Fixed random test | 42 | 43 | 1,324,741 | New |
| `qrdqn_seed_1m_s42_m44` | Same 1M rows | Fixed random test | 42 | 44 | 1,324,741 | New |
| `qrdqn_seed_1m_s42_m45` | Same 1M rows | Fixed random test | 42 | 45 | 1,324,741 | New |
| `qrdqn_seed_1m_s42_m46` | Same 1M rows | Fixed random test | 42 | 46 | 1,324,741 | New |

The multi-seed block must be described as **“seed sensitivity under a fixed 1M-row / 1,324,741-timestep budget.”** It is not variance of the 3M MAIN execution.

### 2.2 Targeted four-holdout generalisation study

**[SPEC]**
Each run uses the shared MAIN profile, `model_seed=42`, and 1,000,000 timesteps. Each test partition is exactly one selected CSV; training uses the other seven official CSVs.

| Logical run | Exact held-out CSV | Physical execution |
|---|---|---|
| `qrdqn_holdout_webattacks_m42` | `Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv` | New |
| `qrdqn_holdout_infilteration_m42` | `Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv` | New |
| `qrdqn_holdout_portscan_m42` | `Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv` | New |
| `qrdqn_holdout_ddos_m42` | `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv` | New |

This is a targeted four-holdout generalisation study. It is not exhaustive eight-fold leave-one-CSV-out, and the selected files must not be described as the only CICIDS2017 CSVs containing attacks.

### 2.3 Random Forest

**[SPEC]**

| Logical run | Train partition | Test partition | Split seed | Model seed/random state | Physical execution |
|---|---|---|---:|---:|---|
| `rf_random_full_s42_m42` | Full random train | Fixed random test | 42 | 42 | New |
| `rf_random_1m_s42_m42` | Nested 1,000,000-row subset | Fixed random test | 42 | 42 | New |
| `rf_day_full_s42_m42` | Monday–Wednesday | Thursday–Friday | Fixed definition | 42 | New |
| `rf_holdout_webattacks_m42` | Other seven CSVs | WebAttacks CSV | N/A | 42 | New |
| `rf_holdout_infilteration_m42` | Other seven CSVs | Infilteration CSV | N/A | 42 | New |
| `rf_holdout_portscan_m42` | Other seven CSVs | PortScan CSV | N/A | 42 | New |
| `rf_holdout_ddos_m42` | Other seven CSVs | DDoS CSV | N/A | 42 | New |

There is no Random Forest six-point ladder and no Random Forest multi-seed study.

### 2.4 Primary model-training execution count

**[SPEC]**

| Group | New primary model-training executions |
|---|---:|
| Fresh QRDQN MAIN | 1 |
| Full day-split QRDQN | 1 |
| QRDQN ladder excluding aliased full point | 5 |
| Additional QRDQN model seeds 43–46 | 4 |
| Targeted QRDQN holdouts | 4 |
| Random Forest | 7 |
| **Total** | **22** |

There are 24 logical primary-training result points and two aliases:

1. ladder/full aliases the fresh campaign MAIN;
2. seed-sensitivity model seed 42 aliases the 1M ladder execution.

The historical RTX 3090 Ti MAIN is historical evidence only. It is not the campaign MAIN, not an alias target, and not one of the 22 new primary model-training executions.

The campaign also contains five auxiliary validation, analysis, and inference jobs. These jobs are not included in the count of 22 primary model-training executions.

### 2.5 Auxiliary validation, analysis, and inference jobs

**[SPEC]**

| Logical job | Purpose | Dependency | Training budget |
|---|---|---|---:|
| `main_direct_validation` | Independently compare fresh MAIN model predictions against the fixed test labels, without relying on environment-reported truth metadata | Fresh campaign MAIN | No training |
| `shuffled_label_validation_s42_m42` | Lightweight anti-leakage validation using shuffled training labels, the current reward matrix, and the maintained code path | Shared profile, fixed random split, and validated cache | 10,000 timesteps |
| `main_bootstrap_ci` | Compute confidence intervals from the fresh MAIN `y_true` and `y_pred` artifacts | Fresh campaign MAIN and direct prediction artifacts | No training |
| `main_duplicate_analysis` | Recompute or verify exact duplicates and cross-split leakage for the fresh MAIN fixed random partition | Fresh campaign MAIN split hashes and canonical unscaled data | No training |
| `phase2_fresh_main` | Run offline Phase 2 inference with the fresh MAIN model, scaler, and training percentiles | Fresh campaign MAIN and validated laboratory-flow input | No training |

Rules:

- These are auxiliary jobs, not primary campaign model-training executions.
- The direct validation, bootstrap, duplicate analysis, and Phase 2 inference must reference the fresh campaign MAIN physical run and its artifact hashes.
- Historical Check A, Check B, bootstrap, duplicate-analysis, and Phase 2 artifacts remain historical evidence only.
- The fresh 3M day-split run supersedes the historical reduced Check C as the official day-generalisation evidence.
- No additional reduced Check C is required except, optionally, as a cheap smoke test with no scientific result claim.
- Bootstrap confidence intervals quantify fixed-test sampling precision for one trained model. They do not quantify training-seed variance.
- The shuffled-label validation is an anti-leakage control and must not be interpreted as a performance comparison with MAIN.

---

## 3. Seed Ownership Rules

**[SPEC]**
- `split_seed` exclusively controls the random train/test partition and deterministic nested train subsets.
- `model_seed` exclusively controls Python `random`, NumPy global/model RNG, PyTorch CPU/CUDA RNG, SB3 model initialisation, exploration, replay sampling, and environment shuffle/RNG.
- Changing `model_seed` must not change train/test row selection, raw array hashes, labels, scaler input, or source CSV selection.
- Changing `split_seed` in random mode must change the partition hashes.
- Day and exact-holdout partitions are filename-defined; their metadata still records `split_seed` as not applicable to partition selection rather than implying that it selected files.
- Random Forest uses `split_seed` for partition/subset construction and `model_seed` as `RandomForestClassifier.random_state`.
- New campaign configs, manifests, state, aggregates, and reuse records always contain both fields explicitly, using `null` only when a seed is genuinely not applicable.
- Legacy `--seed N` may remain as a compatibility alias that sets both seeds only when neither explicit seed flag is supplied. Mixing legacy and explicit seed flags must fail.

---

## 4. Canonical Unscaled Cache Design

**[SPEC]**
Use one provider-neutral, configurable cache root. Store one shard per official CSV:

```text
<cache-root>/
├── cache_manifest.json
└── shards/
    └── <official-csv-name>/
        ├── observations.npy
        ├── labels.npy
        └── metadata.json
```

Cache contract:

- `observations.npy`: canonical unscaled `float32`, shape `(rows, 152)`;
- `labels.npy`: `int64`, same row order;
- shard identity: exact official source filename and source SHA-256;
- metadata: row count, dtype, shape, feature names, array hashes, source hash, cache schema version, canonical schema hash, preprocessing fingerprint/version, producer Git SHA and dirty summary;
- global manifest: all eight shards, official order, shared feature/schema metadata, creation timestamp, and validation status;
- no fitted scaler, scaled array, train/test assignment, prediction, or model state in cache;
- stale/incompatible/corrupt shards fail validation by default;
- rebuilding stale data requires an explicit flag;
- shard writes use temporary locations and atomic replacement;
- safe parallel construction uses at most one worker per official CSV and is configurable; default is `min(os.cpu_count(), 8)`;
- campaign runs use cache policy `require`; development compatibility may offer `off` and `prefer`;
- cached and uncached paths must produce byte-identical canonical arrays and labels.

Every experiment assembles raw cached shards, selects its partition/subset, records raw hashes, fits a fresh `StandardScaler` on final training rows only, then transforms train and test.

---

## 5. Interfaces and Expected CLIs

**[SPEC]**
Exact option names may only change through an explicit plan revision. Expected interfaces:

### 5.1 Cache

```bash
python scripts/build_cicids_cache.py build \
  --dataset-root datasets/CICIDS2017 \
  --cache-root <CACHE_ROOT> \
  --workers 8

python scripts/build_cicids_cache.py validate \
  --dataset-root datasets/CICIDS2017 \
  --cache-root <CACHE_ROOT>
```

`build` creates missing valid shards; `--rebuild-stale` is required to replace stale shards.

### 5.2 Single QRDQN run

```bash
python src/train_rl_defender.py \
  --split-mode random|day|exact-holdout \
  --split-seed 42 \
  --model-seed 42 \
  --profile main-v1 \
  --timesteps <N> \
  --train-max-rows <N-or-omit> \
  --holdout-csv <EXACT_NAME-or-omit> \
  --dataset-root <DATASET_ROOT> \
  --cache-root <CACHE_ROOT> \
  --cache-policy require \
  --artifact-root <ARTIFACT_ROOT> \
  --run-id <RUN_ID> \
  --checkpoint-freq <N> \
  --checkpoint-keep <N> \
  --monitor-interval 30
```

### 5.3 Targeted QRDQN holdout wrapper

```bash
python src/validate_leave_one_csv_out.py \
  --holdout-csvs <EXACT_NAME> <EXACT_NAME> <EXACT_NAME> <EXACT_NAME> \
  --profile main-v1 \
  --timesteps 1000000 \
  --model-seed 42 \
  --cache-root <CACHE_ROOT> \
  --artifact-root <ARTIFACT_ROOT> \
  --resume
```

The legacy filename may remain for compatibility, but help text and outputs must say “targeted holdout”, not imply exhaustive execution. No omitted-list default may launch all eight files.

### 5.4 Single Random Forest run

```bash
python src/baseline_random_forest.py \
  --split-mode random|day|exact-holdout \
  --split-seed 42 \
  --model-seed 42 \
  --train-max-rows <1000000-or-omit> \
  --holdout-csv <EXACT_NAME-or-omit> \
  --n-jobs -1 \
  --cache-root <CACHE_ROOT> \
  --artifact-root <ARTIFACT_ROOT> \
  --run-id <RUN_ID>
```

### 5.5 Campaign runner

```bash
python scripts/run_campaign.py experiments/final_experiment_campaign.json \
  --campaign-id <CAMPAIGN_ID> \
  --artifact-root <ARTIFACT_ROOT> \
  --cache-root <CACHE_ROOT> \
  --snapshot-root <SNAPSHOT_ROOT> \
  --preflight-report <PREFLIGHT_JSON> \
  --dry-run

python scripts/run_campaign.py experiments/final_experiment_campaign.json \
  --campaign-id <CAMPAIGN_ID> \
  --artifact-root <ARTIFACT_ROOT> \
  --cache-root <CACHE_ROOT> \
  --snapshot-root <SNAPSHOT_ROOT> \
  --preflight-report <PREFLIGHT_JSON> \
  --resume
```

Selection flags: `--stage <STAGE_ID>` and `--run <LOGICAL_RUN_ID>`. Real execution requires a valid preflight report; dry-run does not.

### 5.6 Runtime benchmark, preflight, export, aggregation, figures

```bash
python scripts/benchmark_experimental_runtime.py --output <JSON> --thread-config 1:1 4:1 8:1 16:1
python scripts/preflight_gpu_environment.py --dataset-root <PATH> --cache-root <PATH> --artifact-root <PATH> --snapshot-root <PATH>
python scripts/export_campaign.py snapshot --campaign-dir <PATH> --destination <PATH>
python scripts/export_campaign.py bundle --campaign-dir <PATH> --destination <PATH>
python scripts/aggregate_campaign.py --campaign-dir <PATH> --output-dir <PATH>
python scripts/generate_campaign_figures.py --aggregate-dir <PATH> --output-dir <PATH>
```

---

## 6. Run Artifact Contract

### 6.1 QRDQN completed run

**[SPEC]**
Every new meaningful QRDQN run must contain:

```text
<run-dir>/
├── config.json
├── metrics.json
├── environment.json
├── artifact_manifest.json
├── SHA256SUMS
├── model.zip
├── scaler.joblib
├── train_percentiles.npz
├── feature_names.json
├── predictions.npz
├── timing.json
├── system_metrics.csv
├── monitoring.json
├── stdout.log
├── stderr.log
├── tensorboard/
├── tensorboard_scalars/
└── checkpoints/              # when configured
```

Required recorded facts:

- campaign ID, logical run ID, physical run ID, attempt number, status, start/end timestamps;
- exact argv, resolved command, resolved config, profile ID/hash;
- `split_seed`, `model_seed`, split/subset method, train/test hashes, label hashes, source CSV hashes, cache manifest hash;
- train/test support, class prevalence, confusion matrix, defined/null metrics;
- Git commit SHA and dirty-state summary;
- Python/package/platform/CUDA/cuDNN/driver/GPU metadata;
- requested/effective PyTorch intra-op/inter-op threads and OMP/MKL/OpenBLAS settings;
- preprocessing rows/sec, training FPS, evaluation rows/sec, phase timings;
- model, scaler, percentiles, features, compressed `y_true`/`y_pred` predictions;
- TensorBoard event files and exported scalar CSVs;
- resource monitoring or a non-fatal monitoring error record;
- SHA-256 for every required file.

Do not persist per-sample 200-quantile tensors. Replay-buffer persistence is optional and disabled by default.

### 6.2 Checkpoint policy

**[SPEC]**
- Model-only checkpoints are configurable for long QRDQN runs.
- Official default: enable checkpoints for runs of at least 1,000,000 timesteps, save every 500,000 timesteps, retain the newest two verified checkpoints, and always retain the separate final model.
- Runs shorter than 1,000,000 timesteps may disable checkpoints.
- Retention deletes an older checkpoint only after the replacement exists and its checksum is recorded.
- Replay buffers remain disabled by default.
- Model-only checkpoints are diagnostic/recovery evidence, not proof of exact continuation.
- An interrupted official run starts a new physical attempt from the beginning unless a future, separately approved implementation persists and restores model, optimiser, replay buffer, RNG, environment, exploration, and timestep state correctly.

### 6.3 Random Forest completed run

**[SPEC]**
Use the equivalent applicable artifact contract, replacing `model.zip` with `model.joblib`, omitting TensorBoard/checkpoints, and adding:

- `feature_importances.json`;
- `feature_importances.csv`.

### 6.4 Failed run

**[SPEC]**
A failed attempt retains its directory with config, environment, logs, monitoring gathered so far, `error.json`, timestamps, and a manifest with `status=failed`. It is never overwritten or marked complete.

### 6.5 Auxiliary-job artifact contracts

#### 6.5.1 Fresh MAIN direct validation

**[SPEC]**
The direct-validation job must persist:

```text
<job-dir>/
├── config.json
├── validation_results.json
├── predictions.npz
├── environment.json
├── artifact_manifest.json
├── SHA256SUMS
├── timing.json
├── stdout.log
└── stderr.log

```

It must record:

* fresh MAIN physical run ID and manifest hash;
* fresh model hash;
* scaler hash;
* fixed test feature and label hashes;
* direct `y_true` and `y_pred`;
* confusion matrix and derived metrics;
* proof that predictions were evaluated directly against persisted test labels rather than environment-reported truth metadata.

#### 6.5.2 Shuffled-label validation

**[SPEC]**
The shuffled-label auxiliary job must persist:

* resolved lightweight training configuration;
* `split_seed=42` and `model_seed=42`;
* shuffled-label seed and label-permutation hash;
* model and scaler artifacts;
* metrics and compressed predictions;
* environment metadata;
* TensorBoard data where available;
* logs, timings, monitoring, manifest, and checksums.

Its default scientific-control budget is 10,000 timesteps. Any change to this budget must be explicit in the campaign specification and must not be presented as equivalent to MAIN training.

#### 6.5.3 Bootstrap confidence intervals

**[SPEC]**
The bootstrap job must persist:

* `config.json`;
* `bootstrap_ci.json`;
* source fresh MAIN run ID;
* source `predictions.npz` hash;
* bootstrap seed;
* number of resamples;
* point estimates and interval bounds;
* manifest and checksums.

The default protocol remains 10,000 stratified resamples with bootstrap seed `12345`, unless a later explicit protocol revision changes it.

#### 6.5.4 Duplicate and cross-split analysis

**[SPEC]**
The duplicate-analysis job must persist:

* `config.json`;
* `duplicate_analysis.json`;
* source dataset and cache-manifest hashes;
* fresh MAIN train/test feature and label hashes;
* exact duplicate counts;
* test rows present in train;
* benign and attack-specific cross-split rates;
* manifest and checksums.

It must analyse the same canonical unscaled random partition used by the fresh campaign MAIN.

#### 6.5.5 Phase 2 inference with fresh MAIN

**[SPEC]**
The Phase 2 job must persist:

```text
<job-dir>/
├── config.json
├── metrics.json
├── diagnostics.json
├── predictions.npz
├── environment.json
├── artifact_manifest.json
├── SHA256SUMS
├── timing.json
├── system_metrics.csv
├── monitoring.json
├── stdout.log
└── stderr.log
```

It must record:

* fresh MAIN run, model, scaler, percentile, feature-name, and manifest hashes;
* input laboratory-flow filename, size, and SHA-256;
* exact clipping and preprocessing options;
* predictions and truth labels when available;
* metrics only when valid truth labels are present;
* distribution-shift diagnostics;
* no sensitive network metadata by default.

Historical Phase 2 results must not be reused as results of the fresh campaign MAIN.

---

## 7. Resumability and Overwrite Safety

**[SPEC]**
- Campaign resumability is run-level.
- Campaign state is written atomically after transition to running, completed, failed, reused, invalid, or snapshot-failed.
- A run is completed only after its required artifact set and checksums validate.
- Resume skips only validated completed runs.
- Completed evidence is immutable.
- Failed or invalid retries use a new `attempt-N` directory and retain previous attempts.
- A logical alias never copies or mutates its source run; campaign state records `reuse_of`, source manifest hash, and validation time.
- The two permitted aliases are fixed by Section 2.4. No other automatic reuse is allowed.
- A selected stage/run fails before execution if required dependencies are absent or invalid; it does not silently execute unselected dependencies.
- An orphaned `running` state after interruption becomes an interrupted attempt, not completed evidence.
- Snapshot failure stops the campaign before the next experiment while preserving the already completed run.
- Existing historical directories are never adopted as writable campaign attempts.
- Exact mid-training resume must not be advertised or inferred from model-only checkpoints.

Campaign stages execute in this order:

1. `qrdqn_main`;
2. `main_direct_validation`;
3. `main_bootstrap_and_duplicate_analysis`;
4. `shuffled_label_validation`;
5. `phase2_fresh_main`;
6. `qrdqn_day`;
7. `qrdqn_ladder`;
8. `qrdqn_seed_sensitivity`;
9. `qrdqn_targeted_holdouts`;
10. `random_forest`;
11. aggregation and final bundle after all required primary and auxiliary jobs validate.

Dependency rules:

- direct validation depends on the validated fresh campaign MAIN;
- bootstrap depends on validated fresh MAIN predictions;
- duplicate analysis depends on the exact fresh MAIN fixed random partition;
- shuffled-label validation depends on the validated fixed split, cache, profile, and auxiliary configuration;
- Phase 2 depends on the fresh MAIN model, scaler, percentiles, and a validated laboratory-flow input;
- no result aggregation or final bundle may complete while a required auxiliary job is missing, failed, or invalid.

---

## 8. Monitoring and Resource-Use Plan

**[SPEC]**
- Record process and system CPU utilisation, process/system RAM, GPU utilisation, VRAM, power, and temperature every 30 seconds by default.
- Use `psutil` for CPU/RAM and `nvidia-smi` when available for GPU data.
- Monitoring failure is non-fatal and recorded in `monitoring.json`.
- Record effective PyTorch intra-op/inter-op counts plus `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, and `OPENBLAS_NUM_THREADS`.
- Support explicit thread configuration without changing the official learning algorithm.
- Benchmark thread configurations in isolated subprocesses because PyTorch inter-op settings may only be set once per process.
- Report preprocessing throughput, QRDQN training FPS, evaluation throughput, RF fit time, and prediction throughput.
- RF defaults to `n_jobs=-1`, using available logical CPUs.
- Cache construction uses the maximum safe per-CSV parallelism, capped at eight shard workers.
- Official campaign processes remain sequential regardless of available CPU/GPU capacity.
- Do not add `SubprocVecEnv`, multiple official training environments, concurrent experiments, or algorithmic batching that changes experiment semantics.

---

## 9. GPU Environment Preflight

**[SPEC]**
Preflight must use the repository's isolated Python environment and produce a hash-covered report containing:

- CPU model, sockets, physical cores, logical CPUs;
- total RAM;
- GPU count, name, VRAM;
- NVIDIA driver;
- CUDA visible to PyTorch;
- cuDNN version;
- Python and installed package versions;
- filesystem free space for dataset, cache, artifacts, snapshots, and final archive;
- Git commit/dirty state;
- Git LFS availability and materialisation of all eight dataset CSVs;
- source dataset SHA-256 values;
- availability, readability, size, and SHA-256 of the required Phase 2 laboratory-flow input; presence and validity of truth-label columns when Phase 2 metrics are expected; confirmation that sensitive metadata export is disabled by default;
- cache validation and cache-manifest hash;
- minimal CUDA tensor operation;
- tiny synthetic end-to-end QRDQN artifact run;
- artifact-manifest verification;
- incremental snapshot creation and verification;
- optional lightweight runtime benchmark results.

Default host acceptance thresholds for the expected platform:

- at least 16 logical CPUs;
- at least 120 GiB RAM;
- at least one CUDA GPU;
- at least 80 GiB VRAM on the selected GPU;
- at least 100 GiB free in the configured artifact/snapshot capacity, or a larger value derived from measured checkpoint/model sizes.

Real campaign execution refuses a missing, failed, stale, or cache-mismatched preflight report. Threshold overrides must be explicit and recorded.

The preflight report must bind the Phase 2 input hash to the resolved campaign specification. A different laboratory-flow file requires a new or explicitly amended preflight report before Phase 2 execution.

---

## 10. Snapshot and Export Strategy

**[SPEC]**
- After every validated physical execution, create or update an incremental snapshot under a configurable destination.
- Snapshot only changed/new files by comparing manifest hashes.
- Include campaign state, original and resolved campaign specifications, preflight report, every completed run, aliases/reuse records, manifests, and `SHA256SUMS`.
- Include the cache manifest but not rebuildable cache arrays by default.
- Do not require cloud credentials, provider SDKs, fixed hostnames, or account paths.
- Verify the incremental snapshot before starting the next experiment.
- When all required runs and aliases validate, create a compressed `.tar.gz` campaign bundle.
- Reopen the final archive and verify all declared files and hashes.
- Keep the source campaign intact when exporting; export never moves or deletes evidence.
- Standard later transfer mechanisms such as filesystem copy, `rsync`, or removable/mounted storage remain operational choices outside the scientific description.

---

## 11. Aggregation and Future Figure/Table Strategy

**[SPEC]**
Prepare aggregation code that validates campaign state and manifests before producing:

```text
<aggregate-output>/
├── campaign_summary.json
├── main.json
├── main_direct_validation.json
├── main_bootstrap_ci.json
├── main_duplicate_analysis.json
├── shuffled_label_validation.json
├── phase2_fresh_main.json
├── day_split.json
├── size_ladder.json
├── size_ladder.csv
├── seed_sensitivity.json
├── seed_sensitivity.csv
├── targeted_holdouts.json
├── targeted_holdouts.csv
├── random_forest.json
├── random_forest.csv
└── qrdqn_vs_rf.csv
```

Aggregation rules:

- preserve campaign/run IDs, seeds, hashes, profile hash, support, prevalence, timings, and reuse provenance;
- represent undefined metrics as JSON `null` and empty CSV cells;
- compute macro holdout summaries only over folds where each metric is defined and report `n_defined`;
- compute pooled confusion-matrix metrics where meaningful;
- include aliased logical points once in their intended analysis group without duplicating physical evidence;
- reject missing, corrupt, failed, or incompatible runs;
- never invent values for pending runs.
- require the direct-validation, bootstrap, duplicate-analysis, and Phase 2 artifacts to reference the same fresh campaign MAIN and expected artifact hashes;
- keep shuffled-label results in a separate auxiliary-control group and never mix them into MAIN performance aggregates;
- preserve the Phase 2 input hash, preprocessing options, clipping configuration, and domain-shift diagnostics;
- reject historical MAIN-derived auxiliary artifacts as substitutes for fresh campaign auxiliary jobs;

Figure/table generators may later produce size-ladder curves, seed-sensitivity distributions, day/generalisation comparisons, targeted-holdout summaries, and matched QRDQN/RF comparisons. They must require an explicit output directory outside `memoria/` and `report/`, refuse final output from an incomplete campaign, and remain unexecuted until real campaign artifacts exist.

---

## 12. Provider-Neutral Documentation Strategy

**[SPEC]**
- Use “GPU experimental environment”, “GPU host”, or “experimental platform” in maintained scientific documentation, filenames, CLI help, new artifacts, and comments.
- Create a neutral GPU environment/setup guide and a neutral requirements filename.
- Keep the old provider-specific requirements/setup filenames only as compatibility aliases or explicitly historical pointers when necessary.
- Provider, rental, remote-access, hostname, account, and mount details are operational and must not become scientific-method fields.
- Historical artifacts may retain their original provider/hardware metadata unchanged.
- Update maintained documentation to distinguish historical MAIN from the fresh campaign MAIN.
- State the exact targeted four-holdout scope and seed-sensitivity budget.
- Keep archived audits and historical run evidence unchanged.
- Do not edit `memoria/` or `report/`; thesis wording changes occur only after real artifacts exist in a separately authorised task.

---

## 13. Phased Implementation Roadmap

### Phase 1 — Experimental contracts, shared profile, and seed separation

**[SPEC]**
**Objective:** establish authoritative scientific configuration, seed ownership, raw split hashing, and undefined-metric semantics without changing campaign execution.

**Scope:**

- create the shared `main-v1` profile and profile hash;
- separate `split_seed` and `model_seed` through loaders and current QRDQN entry point;
- preserve controlled legacy aliases;
- hash unscaled train/test arrays and labels;
- add nullable metric policy for new campaign workflows.

**Files likely affected:**

- create `src/experiment_profiles.py`;
- modify `src/load_cicids2017.py`;
- modify `src/train_rl_defender.py`;
- modify `src/metrics_utils.py`;
- modify `tests/test_train_rl_defender_config.py`;
- modify `tests/test_load_cicids2017.py`;
- modify `tests/test_metrics_utils.py`.

**Dependencies:** current maintained loader, environment, reward, and MAIN-profile tests.

**Non-goals:** cache, complete artifacts, holdout/RF refactors, campaign specification, GPU execution, documentation sweep.

**Tests and validation:** profile equality/hash; legacy seed compatibility; model-seed independence of split hashes; split-seed partition changes; deterministic/nested subsets; null versus legacy zero metric policy; existing unit suite and Ruff.

**Acceptance criteria:** no scientific value changes; both seeds appear explicitly in new resolved configs; data selection consumes only `split_seed`; model/environment RNG consumes only `model_seed`; all tests pass.

**Expected deliverables:** shared profile module, seed-resolution contract, upgraded split metadata, focused tests.

**Stopping point:** report Phase 1 files, commands, results, and risks. Do not begin cache work.

### Phase 2 — Canonical unscaled cache

**[SPEC]**
**Objective:** eliminate repeated CSV parsing while preserving canonical preprocessing and anti-leakage behavior exactly.

**Scope:** implement per-CSV unscaled shards, cache metadata/invalidation, parallel atomic build, loader integration, and build/validate CLI.

**Files likely affected:**

- create `src/cicids_cache.py`;
- create `scripts/build_cicids_cache.py`;
- create `tests/conftest.py` if shared synthetic official-CSV fixtures are needed;
- create `tests/test_cicids_cache.py`;
- modify `src/load_cicids2017.py`;
- modify `tests/test_load_cicids2017.py`;
- modify `.gitignore` for generated cache roots.

**Dependencies:** Phase 1 split/hash contract.

**Non-goals:** training-run refactor, full real-dataset cache construction during implementation, global scaler storage, campaign orchestration.

**Tests and validation:** synthetic cached/uncached exact equality; official order/provenance; invalidation on source/schema/preprocessing/hash corruption; explicit rebuild behavior; atomic failure cleanup; unit suite and Ruff.

**Acceptance criteria:** cached and uncached outputs are byte-identical; no scaler enters cache; stale cache cannot be consumed silently; paths are configurable.

**Expected deliverables:** cache library, CLI, manifest schema, synthetic equivalence/invalidation tests.

**Stopping point:** report Phase 2 only. Do not implement artifact or training changes.

### Phase 3 — Artifact, monitoring, and environment metadata foundation

**[SPEC]**
**Objective:** provide a reusable manifest writer/verifier, complete metadata capture, logging, throughput timing, monitoring, and bounded checkpoint support.

**Scope:** artifact schema 3, atomic writes, legacy schema-2 reads, SHA-256 inventory, Git/environment capture, stdout/stderr teeing, system monitor, TensorBoard scalar export helper, model-only checkpoint retention primitives.

**Files likely affected:**

- create `src/run_artifacts.py`;
- create `src/resource_monitor.py`;
- create `src/tensorboard_export.py`;
- create `tests/test_run_artifacts.py`;
- create `tests/test_resource_monitor.py`;
- modify `src/artifact_integrity.py`;
- modify `scripts/export_tensorboard_scalars.py`;
- modify `pyproject.toml`, `requirements.txt`, and `uv.lock` if `psutil` must become a direct dependency.

**Dependencies:** Phase 1 config/seed fields; Phase 2 cache-manifest identity.

**Non-goals:** conversion of QRDQN/RF runners, campaign state, real GPU telemetry validation, exact mid-run resume.

**Tests and validation:** complete/incomplete manifest fixtures; checksum corruption; schema-2 historical compatibility; monitoring tool absence; bounded checkpoint retention; atomic status writes; unit suite, lock check, Ruff.

**Acceptance criteria:** required artifacts can be declared and verified independently of model code; monitoring failure is non-fatal; completed artifacts are immutable; checkpoint retention is bounded.

**Expected deliverables:** common artifact/monitoring APIs and tests.

**Stopping point:** report Phase 3 only. Do not refactor QRDQN execution.

### Phase 4 — Single-run QRDQN execution API

**[SPEC]**
**Objective:** make one QRDQN run reproducible, configurable, cache-backed, and artifact-complete across random, day, and exact-holdout partitions.

**Scope:**
- reusable single-run API, CLI integration, shared MAIN profile consumption, separate seeds, fresh scaler/percentiles, batched prediction, complete artifacts, throughput, TensorBoard, model-only checkpoints.
- implement an independent fresh-MAIN direct-validation job;
- make bootstrap confidence intervals consume the fresh MAIN persisted `y_true` and `y_pred` artifacts rather than historical counts;
- make duplicate/cross-split analysis consume and verify the fresh MAIN canonical unscaled split hashes;
- adapt the maintained Phase 2 inference path to the common artifact schema and fresh-MAIN artifact resolution;

**Files likely affected:**

- create `src/qrdqn_experiment.py`;
- create `tests/test_qrdqn_experiment.py`;
- modify `src/train_rl_defender.py`;
- modify `scripts/run_main_experiment.sh`;
- modify `scripts/verify_fixed_test_split.py`;
- create `scripts/validate_main_direct.py`;
- create `tests/test_main_direct_validation.py`;
- create `tests/test_bootstrap_fresh_main.py`;
- create `tests/test_duplicate_analysis_fresh_main.py`;
- create `tests/test_phase2_fresh_main.py`;
- modify `scripts/bootstrap_ci.py`;
- modify `scripts/analyze_duplicates.py`;
- modify `scripts/predict_real_traffic_v2.py`.

**Dependencies:** Phases 1–3.

**Non-goals:**
- multi-run holdout orchestration, RF refactor, campaign state, full training, multiple environments, replay persistence, exact interrupted-training continuation.
- real Phase 2 execution;
- publication of fresh metrics;
- reuse of historical MAIN-derived validation or Phase 2 results as fresh campaign evidence;

**Tests and validation:**
- tiny synthetic end-to-end run;
- profile equality for all split modes;
- seed ownership; scaler train-only fit;
- prediction/artifact completeness;
- checkpoint retention;
- failed-run evidence;
- direct-script CLI help;
- unit suite and Ruff
- synthetic direct validation matches independently generated truth labels;
- bootstrap output is bound to the source prediction hash;
- duplicate analysis rejects mismatched split hashes;
- Phase 2 synthetic inference persists input/model/scaler/percentile hashes and complete artifacts;

**Acceptance criteria:** a tiny run produces a valid complete manifest; day/holdout override only split, timesteps, and seeds; no reduced holdout profile remains in the single-run path. The direct-validation, bootstrap, duplicate-analysis, and Phase 2 utilities can consume a synthetic fresh MAIN run and produce independently valid artifact manifests without using any historical result as evidence.

**Expected deliverables:** reusable QRDQN runner and compatible CLI.

**Stopping point:** report Phase 4 only. Do not build multi-run holdout or RF workflows.

### Phase 5 — Targeted holdout and Random Forest runners

**[SPEC]**
**Objective:** implement only the locked targeted holdout study and seven-run RF comparison surface.

**Scope:**
- explicit QRDQN holdout list, per-holdout immediate artifacts/resume, defined-only macro and pooled metrics, configurable one-run RF modes, shared cache/splits/scalers/hashes, RF evidence.
- implement the lightweight shuffled-label anti-leakage auxiliary workflow using the current reward matrix, fixed split contract, and separate artifact directory;
- ensure the shuffled-label job uses the explicit 10,000-timestep auxiliary budget by default and is never treated as a primary campaign training execution;

**Files likely affected:**

- create `tests/test_holdout_workflow.py`;
- create `tests/test_random_forest_runner.py`;
- modify `src/validate_leave_one_csv_out.py`;
- modify `src/baseline_random_forest.py`;
- modify `src/metrics_utils.py` only if aggregation helpers cannot remain external;
- modify `src/validate_checks.py` or replace its maintained Check-B path with a dedicated provider-neutral auxiliary runner;
- create `tests/test_shuffled_label_validation.py`.

**Dependencies:** Phases 1–4.

**Non-goals:**
- eight-fold automatic execution, claims of exhaustive coverage, RF ladder beyond full/1M, RF multi-seed study, campaign orchestration, real training;
- another reduced Check C scientific execution;
- treating shuffled-label metrics as comparable model-performance evidence.

**Tests and validation:**
- exact selected filenames;
- unknown/duplicate rejection;
- no implicit all-eight default;
- per-holdout resume/skip;
- null metrics; pooled confusion metrics;
- RF/QRDQN test-hash equality for matched partitions;
- RF artifact completeness;
- suite and Ruff;
- shuffled labels are deterministic under the configured seed and have a persisted permutation hash;
- the auxiliary job uses the current reward configuration;
- its complete artifacts validate independently;
- it is classified as an auxiliary job rather than one of the 22 primary model-training executions.

**Acceptance criteria:** only four targeted QRDQN and four targeted RF holdouts are represented; RF supports exactly random/full, random/1M, day/full, and those holdouts; every synthetic run validates independently. A lightweight, artifact-backed shuffled-label validation is available for the final campaign, while the full 3M day split remains the only official replacement for the historical proxy Check C.

**Expected deliverables:** targeted holdout workflow, configurable RF runner, focused tests.

**Stopping point:** report Phase 5 only. Do not create the campaign specification or runner.

### Phase 6 — Sequential campaign orchestration and resumability

**[SPEC]**
**Objective:** encode the exact 22-execution campaign and execute/resume it safely one physical run at a time.

**Scope:**
- committed JSON specification, validation, logical aliases, sequential subprocess dispatch, campaign IDs/state, attempts, dry-run, stage/run selection, artifact gates, overwrite protection;
- encode and dispatch the five auxiliary jobs defined in Section 2.5;
- enforce their dependencies on the fresh MAIN, fixed split, validated cache, and Phase 2 input;
- distinguish primary model-training executions, aliases, and auxiliary jobs in campaign state and dry-run output.

**Files likely affected:**

- create `experiments/final_experiment_campaign.json`;
- create `src/campaign.py`;
- create `scripts/run_campaign.py`;
- create `tests/test_campaign_runner.py`.

**Dependencies:** Phases 1–5.

**Non-goals:** real campaign execution, host preflight implementation, snapshots, aggregation, automatic dependency execution outside selection, mid-training resume.

**Tests and validation:**
- schema rejection;
- exact matrix/count;
- two aliases;
- fresh MAIN enforcement;
- sequential ordering;
- dry-run no artifacts;
- completed skip;
- failed/interrupted retry;
- invalid artifact halt;
- stage/run dependency gates;
- provider-neutral paths;
- suite and Ruff;
- dry-run reports exactly 22 new primary model-training executions;
- dry-run reports exactly five auxiliary jobs;
- dry-run reports exactly two approved logical aliases;
- fresh-MAIN auxiliary jobs cannot resolve to historical MAIN artifacts;
- Phase 2 cannot start without its validated input hash.

**Acceptance criteria:** dry-run shows exactly 22 new primary model-training executions, five auxiliary validation/analysis/inference jobs, 24 primary-training logical result points, and two approved aliases. No historical MAIN reuse, no concurrent dispatch, and no auxiliary job may consume incompatible or historical substitute evidence.

**Expected deliverables:** committed campaign spec, runner, state schema, resume tests.

**Stopping point:** report Phase 6 only. Do not implement preflight, benchmarks, or export.

### Phase 7 — Preflight, runtime benchmarking, and snapshots

**[SPEC]**
**Objective:** prove host readiness cheaply and protect every completed run on an ephemeral platform.

**Scope:** neutral GPU requirements/setup inputs, thread benchmark, hardware/software/data/cache/CUDA preflight, synthetic smoke, incremental verified snapshot, final verified bundle, campaign integration gates.

**Files likely affected:**

- create `requirements-gpu-cu130.txt`;
- create `scripts/benchmark_experimental_runtime.py`;
- create `scripts/preflight_gpu_environment.py`;
- create `src/campaign_export.py`;
- create `scripts/export_campaign.py`;
- create `tests/test_preflight.py`;
- create `tests/test_campaign_export.py`;
- modify `src/campaign.py` and `scripts/run_campaign.py` to require preflight and post-run snapshot validation;
- modify `requirements-runpod-cu130.txt` only as a compatibility include/pointer;
- modify `pyproject.toml`, `requirements.txt`, and `uv.lock` if dependency alignment remains necessary.

**Dependencies:** Phases 1–6.

**Non-goals:** paid/full experiments, automatic scientific config selection from benchmark results, cloud upload, provider credentials, multiple QRDQN environments.

**Tests and validation:** mocked hardware/tool outputs; failed thresholds; cache/preflight binding; tiny synthetic artifact/export; incremental no-op on unchanged files; corruption detection; final archive verification; monitor absence; suite, lock check, Ruff.

**Acceptance criteria:** real campaign cannot start without matching successful preflight; every completed run must have a verified snapshot before progression; final bundle verification is deterministic.

**Expected deliverables:** preflight report contract, benchmark, snapshot/bundle tools, campaign safety integration.

**Stopping point:** report Phase 7 only. Do not aggregate or generate figures.

### Phase 8 — Aggregation and future figure/table preparation

**[SPEC]**
**Objective:** prepare artifact-driven result aggregation and rendering without producing or inserting pending results.

**Scope:**
- manifest/state validation, JSON/CSV groups, alias provenance, nullable macro/pooled metrics, future chart/table generators with completeness gates;
- aggregate fresh MAIN direct validation, bootstrap intervals, duplicate analysis, shuffled-label validation, and fresh-MAIN Phase 2 outputs as separate provenance-preserving groups.

**Files likely affected:**

- create `src/campaign_aggregation.py`;
- create `scripts/aggregate_campaign.py`;
- create `scripts/generate_campaign_figures.py`;
- create `tests/test_campaign_aggregation.py`.

**Dependencies:** Phases 1–7 and synthetic campaign fixtures.

**Non-goals:** real aggregates, final figures/tables, thesis edits, placeholder removal, result claims.

**Tests and validation:**
- synthetic expected rows;
- alias deduplication;
- defined-only macros;
- pooled metrics;
- incompatible run rejection;
- incomplete-campaign refusal;
- output path guard against `memoria/`/`report/`;
- suite and Ruff;
- fresh MAIN and all derived auxiliary outputs share the expected source run and hashes;
- historical auxiliary artifacts are rejected as fresh substitutes;
- shuffled-label controls remain excluded from performance aggregates;
- Phase 2 remains separate from CICIDS2017 internal-test metrics.

**Acceptance criteria:** generators consume only validated aggregates and refuse incomplete evidence; no final output is produced during implementation.

**Expected deliverables:** aggregation and future-rendering code plus synthetic tests.

**Stopping point:** report Phase 8 only. Do not modify maintained documentation or thesis files.

### Phase 9 — Provider-neutral maintained documentation

**[SPEC]**
**Objective:** document the prepared workflow and scientific caveats without rewriting historical evidence or thesis results.

**Scope:** neutral setup/run/cache/preflight/export/aggregation guidance, CLI examples, campaign matrix, historical-versus-fresh MAIN distinction, compatibility pointers, terminology checks.

**Files likely affected:**

- create `docs/gpu_experimental_environment.md`;
- create `tests/test_provider_neutrality.py`;
- modify `README.md`;
- modify `.github/AGENT_CONTEXT.md`;
- modify `docs/README.md`;
- modify `docs/reproducibility.md`;
- modify `docs/results.md` only to clarify pending future campaign scope without adding results;
- modify `docs/runpod_main_experiment.md` into an explicitly historical/compatibility pointer if retained;
- modify `experiments/cicids2017_qrdqn_experiments.md` to describe the approved future protocol;
- modify `scripts/run_main_experiment.sh` comments/help if provider-specific language remains.

**Dependencies:** Phases 1–8 so documentation matches implemented interfaces.

**Non-goals:** `memoria/`, `report/`, historical artifacts, archived audits, result numbers, completed placeholders, provider-specific operational instructions.

**Tests and validation:** CLI examples against `--help`; link/path checks; terminology scan over maintained/new surfaces; exclusion of historical artifacts and labelled historical paragraphs; suite and Ruff.

**Acceptance criteria:** active scientific/setup language is provider-neutral; fresh campaign MAIN and exact matrix are clear; both mandatory methodological caveats are explicit; no historical evidence changed.

**Expected deliverables:** maintained neutral documentation and terminology regression test.

**Stopping point:** report Phase 9 only. Do not perform or fix findings from the final review.

### Phase 10 — Final read-only integration review

**[SPEC]**
**Objective:** decide whether the repository is ready to transfer to the GPU host without making changes.

**Scope:** read-only diff/status review, campaign-spec count, CLI consistency, dependency lock, unit/lint suite, synthetic dry-run, artifact/cache/preflight/export contract review, scope-exclusion check.

**Files likely affected:** none. This phase is strictly read-only.

**Dependencies:** completed and individually accepted Phases 1–9.

**Non-goals:** code/docs fixes, commits, full cache build, GPU assumptions presented as verified, any experiment execution, thesis updates.

**Tests and validation:**

```bash
uv lock --check
uv run pytest
uv run ruff check .
python scripts/build_cicids_cache.py --help
python src/train_rl_defender.py --help
python src/baseline_random_forest.py --help
python scripts/run_campaign.py experiments/final_experiment_campaign.json --dry-run <required-temp-path-options>
git diff --check
git status --short
```

Review must also confirm:

- exactly 22 new physical executions;
- fresh MAIN, full-ladder alias, and 1M seed-42 alias;
- four exact targeted holdouts for both models;
- no RF ladder/multi-seed expansion;
- no source path tied to a provider;
- no changes under historical `runs/`, `memoria/`, or `report/`;
- no unsupported exact mid-run resume claim.

**Acceptance criteria:** all cheap checks pass, exclusions hold, and remaining checks are exclusively host-dependent. Findings are reported; they are not fixed in this phase.

**Expected deliverables:** read-only readiness report with pass/fail evidence and host-only checklist.

**Stopping point:** stop. GPU-host setup or campaign execution requires a new explicit user request.

---

## 14. Host-Only Checks

**[SPEC]**
The following cannot be truthfully completed before the final GPU host exists:

- actual CPU topology, RAM, GPU model/count/VRAM, driver, CUDA, and cuDNN;
- PyTorch CUDA visibility and minimal CUDA tensor operation;
- effective thread settings and comparative benchmark throughput on that host;
- real `nvidia-smi` monitoring fields and sampling stability;
- filesystem capacity at configured artifact and snapshot destinations;
- Git LFS materialisation after repository transfer;
- full source CSV hashes on the host;
- full cache build/validation performance and cache-manifest binding;
- tiny CUDA QRDQN smoke run and complete artifact verification;
- verified incremental snapshot to the chosen durable destination;
- final command dry-run using actual host paths;
- exact wall-time/storage estimates for model-only checkpoints and RF models.

These checks must pass before the first paid/full campaign execution. Failure blocks campaign launch but does not authorise scientific configuration changes.

---

## 15. Methodological Caveats for the Future Thesis Update

**[SPEC]**
After real artifacts exist, any separately authorised thesis update must state:

1. The final campaign MAIN is a fresh 3M-timestep execution on the final experimental platform. The historical RTX 3090 Ti MAIN remains separate historical evidence.
2. The four selected CSV holdouts form a targeted four-holdout generalisation study, not exhaustive eight-fold leave-one-CSV-out.
3. The four selected files are not claimed to be the only CICIDS2017 CSVs containing attacks.
4. The multi-seed block measures **seed sensitivity under a fixed 1M-row / 1,324,741-timestep budget** for model seeds 42–46.
5. The multi-seed block does not estimate variance of the 3M MAIN execution.
6. Seed-42 sensitivity reuses the 1M ladder physical execution; full ladder reuses the fresh campaign MAIN.
7. Random-split results measure in-distribution performance and do not replace day/holdout generalisation evidence.
8. Day and targeted-holdout runs use the same shared MAIN profile; differences arise from partition, budget, and seeds as declared.
9. Undefined per-holdout metrics remain missing and macro summaries include only defined folds; pooled metrics answer a different, explicitly labelled question.
10. Model-only checkpoints and run-level retry do not constitute exact mid-training continuation.
11. Hardware utilisation observations describe the measured platform and do not justify an algorithmic-semantic change by themselves.
12. No figure, table, aggregate, or conclusion becomes final until backed by validated campaign artifacts and hashes.
13. The fresh MAIN direct validation, bootstrap confidence intervals, duplicate analysis, and Phase 2 inference are all regenerated from or explicitly verified against the fresh campaign MAIN; historical derived results are not silently inherited.
14. The shuffled-label validation uses a lightweight 10,000-timestep auxiliary budget and is an anti-leakage control, not a performance-equivalent MAIN repetition.
15. The fresh 3M day-split result supersedes the historical reduced Check C as official day-generalisation evidence.
16. Phase 2 results remain a separate offline laboratory-domain evaluation and must not be merged with CICIDS2017 internal-test metrics.

---

## 16. Global Completion Criteria

**[SPEC]**
Campaign preparation is complete only when all ten phases have been requested and accepted separately, Phase 10 reports no blocking integration defect, and the repository contains:

- shared frozen MAIN profile and seed ownership contract;
- validated unscaled canonical cache implementation;
- complete artifact/monitoring/checkpoint foundation;
- single-run QRDQN, targeted holdout, and scoped RF runners;
- exact sequential campaign specification covering 22 primary model-training executions, five auxiliary jobs, two aliases, and a safe state machine;
- provider-neutral preflight, benchmark, snapshot, and bundle tooling;
- artifact-driven aggregation/future-rendering code;
- provider-neutral maintained documentation;
- full passing unit/lint suite using only cheap/synthetic validation;
- no modifications to historical evidence, `memoria/`, `report/`, or pending thesis placeholders;
- implemented and synthetic-tested workflow for fresh-MAIN direct validation;
- implemented and synthetic-tested workflow for fresh-MAIN bootstrap confidence intervals;
- implemented and synthetic-tested workflow for fresh-MAIN duplicate/cross-split analysis;
- implemented and synthetic-tested lightweight shuffled-label anti-leakage workflow;
- implemented and synthetic-tested fresh-MAIN Phase 2 inference workflow, including validation of the laboratory input contract;

Campaign execution itself is a separate activity and requires explicit authorisation after host preflight succeeds.
