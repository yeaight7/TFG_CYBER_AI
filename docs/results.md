# Consolidated Results Snapshot

This document summarises **artifact-backed** results currently present under `runs/` and `models/`.

## Rules for Reading This Page

- If a metric comes from a committed artifact, it is presented as a measured result.
- If a behaviour changed across runs, the result is tied to the exact `RUN_ID`.
- Historical run metadata must not be confused with the **current code defaults**.

## Current Code Defaults vs Historical Run Settings

The **current codebase defaults** for training and validation are:

```python
REWARD_CONFIG = {
    "tp": 1.5,
    "fp": -2.0,
    "fn": -5.0,
    "omission": 0.0,
}
```

## CICIDS2017 Training Runs

### Summary

| Run | Rows | Timesteps | Split | Accuracy | Recall attack | F1 attack | Reward config |
|-----|------|-----------|-------|----------|---------------|-----------|---------------|
| C01 smoke | 50,000 | 5,000 | random | 0.9697 | 0.9996 | 0.9692 | `tp=1.5, fp=-1.0, fn=-5.0, om=0.0` |
| C01 full | 250,000 | 100,000 | random | 0.9962 | 0.9998 | 0.9963 | `tp=1.5, fp=-1.0, fn=-5.0, om=0.0` |
| C02 fast | 100,000 | 10,000 | random | 0.9766 | 0.9996 | 0.9812 | `tp=1.5, fp=-1.0, fn=-5.0, om=0.0` |
| **C03 full** | 500,000 | 100,000 | random | **0.99859** | **0.99945** | **0.99876** | `tp=1.5, fp=-2.0, fn=-5.0, om=0.0` |

### Best Committed Historical Run

| Field | Value |
|-------|-------|
| RUN_ID | `C03_qrdqn_cicids2017_canonical_full_random_20260223_232439` |
| Algorithm | QRDQN |
| Dataset | CICIDS2017 |
| Observation size | 152 |
| Train / test | 400,000 / 100,000 |
| Device | `cuda` |
| Learning rate | `1e-4` |
| Batch size | `2048` |
| Gradient steps | `20` |
| Reward config | `tp=1.5, fp=-2.0, fn=-5.0, omission=0.0` |

Metrics from `runs/cicids2017/C03_qrdqn_cicids2017_canonical_full_random_20260223_232439/metrics.json`:

| Metric | Value |
|--------|-------|
| Accuracy | `0.99859` |
| Precision attack | `0.99806` |
| Recall attack | `0.99945` |
| F1 attack | `0.99876` |
| Precision benign | `0.99928` |
| Recall benign | `0.99746` |
| F1 benign | `0.99837` |

### Training-Size Benchmark (Fixed Test Partition)

Internal benchmark to justify the full-data main experiment. Smaller runs use `--train-max-rows` (train-only subsampling AFTER the split): the test partition is byte-identical to the main run's 566,149-row test set (seed 42; benign 454,620 / attack 111,529), verified per run via `split_metadata.test_set_sha256` against `runs/cicids2017/test_partition_reference_seed42.json`. Timesteps scale proportionally with train size. Protocol details: [../experiments/cicids2017_qrdqn_experiments.md](../experiments/cicids2017_qrdqn_experiments.md).

These results are an **internal CICIDS2017 benchmark only** — random stratified split with a fixed held-out test partition. They are not comparable to, and must not be mixed with, the Phase 2 offline-inference results, which measure distribution shift on real lab traffic.

No benchmark training artifacts are committed yet; the table below will be filled only from committed `runs/cicids2017/` artifacts.

| Run | Train rows | Timesteps | test_set_sha256 match | Accuracy | Recall attack | F1 attack |
|-----|------------|-----------|----------------------|----------|---------------|-----------|
| _(pending)_ | | | | | | |

## Validation Artifacts

### Check A — Direct Evaluation

Artifact:

- `runs/validation/VAL_checks_A_20260212_235443/validation_results.json`

| Metric | Value |
|--------|-------|
| Accuracy | `0.9939` |
| Precision attack | `0.98758` |
| Recall attack | `0.99979` |
| F1 attack | `0.99365` |
| TP / FP / FN / TN | `4772 / 60 / 1 / 5167` |

### Check B — Shuffled Labels

Artifact:

- `runs/validation/VAL_checks_B_20260212_235736/validation_results.json`

| Metric | Value |
|--------|-------|
| Shuffled accuracy | `0.4773` |
| Majority-class baseline | `0.5227` |
| Leakage detected | `false` |

Interpretation:

- this historical artifact supports the no-leakage claim
- the model collapsed to one class under shuffled labels, which is acceptable for this anti-leakage test because performance did not stay artificially high

### Check C — Hard CSV/Day Split

Artifact:

- `runs/validation/VAL_checks_C_20260213_004847/validation_results.json`

| Metric | Value |
|--------|-------|
| Accuracy | `0.8413509399739979` |
| Precision attack | `0.764791673901073` |
| Recall attack | `0.5295374374439701` |
| F1 attack | `0.6257849253737402` |
| TP / FP / FN / TN | `154169 / 47414 / 136970 / 823660` |
| Train rows | `1,668,530` |
| Test rows | `1,162,213` |

This remains the hardest committed generalisation artifact in the repository.

### Leave-One-Exact-CSV-Out

Status:

- implemented in `src/validate_leave_one_csv_out.py`
- no committed full artifact currently exists under `runs/validation/`

Because there is no committed run folder for this validation yet, no measured metrics are reported here.

## Phase 2 Offline Inference

Phase 2 results must always be read per artifact because behaviour changed over time.

### Early v2 Benign-Only Artifact

Artifact:

- `runs/phase2/P2v2_pred_20260224_004121/`

| Field | Value |
|-------|-------|
| Flows CSV | `pcaps/flows_benign.csv` |
| Block rate | `1.0` |
| Allow rate | `0.0` |
| z abs max | `10.0` |
| z abs mean | `1.1148942708969116` |

This artifact documents a strong domain-shift problem on benign real traffic.

### Later v2 Benign-Only Artifact

Artifact:

- `runs/phase2/P2v2_pred_20260408_230318/`

| Field | Value |
|-------|-------|
| Flows CSV | `pcaps/flows_benign.csv` |
| Block rate | `0.0` |
| Allow rate | `1.0` |
| z abs max | `10.0` |
| z abs mean | `0.6870886087417603` |

This later artifact shows that Phase 2 behaviour is sensitive to configuration and run conditions, which is exactly why documentation must cite the specific run artifact.

## NSL-KDD Historical Benchmark

The maintained historical summary lives in [../experiments/nslkdd_experiments.md](../experiments/nslkdd_experiments.md).

Short version:

| Experiment | Model | Accuracy | Recall attack | FP rate |
|------------|-------|----------|---------------|---------|
| E01 | DQN | `0.7602` | `0.600` | `0.028` |
| E02 | Random Forest | `0.7693` | `0.615` | `0.0267` |
| E05 | DQN | `0.7563` | `0.5955` | `0.0313` |

NSL-KDD remains historical benchmarking material only.

## Artifact Locations

### Training

- `runs/cicids2017/C01_qrdqn_cicids2017_canonical_smoke_20260212_195959/`
- `runs/cicids2017/C01_qrdqn_cicids2017_canonical_full_20260212_200218/`
- `runs/cicids2017/C02_qrdqn_cicids2017_canonical_fast_random_20260223_181122/`
- `runs/cicids2017/C03_qrdqn_cicids2017_canonical_full_random_20260223_232439/`

### Validation

- `runs/validation/VAL_checks_A_20260212_235443/`
- `runs/validation/VAL_checks_B_20260212_235736/`
- `runs/validation/VAL_checks_C_20260213_004847/`

### Phase 2

- `runs/phase2/P2v2_pred_20260224_004121/`
- `runs/phase2/P2v2_pred_20260408_230318/`

### Models

- `models/C03_qrdqn_cicids2017_canonical_full_random_20260223_232439.zip`

## Random Forest Baseline

The supervised Random Forest baseline has been updated to match the QRDQN evaluation splits strictly. 

**Execution Protocol:**
Run `uv run python src/baseline_random_forest.py` to generate the latest metrics. This will execute three sweeps across the canonical schema:
1. **Random Split (full)**
2. **Day Split (Check C)**
3. **Leave-One-Out (Wednesday test)**

*Metrics to be populated below after the first manual full-sweep execution (requires ~10-15 minutes on full dataset).*

### Baseline Metrics Placeholder

| Split | F1 Attack | Precision | Recall | Notes |
|-------|-----------|-----------|--------|-------|
| Random | pending | pending | pending | Compare against QRDQN C03 |
| Check C | pending | pending | pending | Compare against QRDQN Check C |
| Leave-One-Out | pending | pending | pending | Evaluate domain shift tolerance |

## Open Documentation Gap

The repository now includes code for leave-one-exact-CSV-out validation, but the documentation cannot yet report aggregate metrics for it until a full committed run is added under `runs/validation/`.
