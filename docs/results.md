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

The **official trunk** is the MAIN run (full data, fixed test partition). Secondary runs (same design, fewer training rows) form the training-size benchmark (pending). The earlier `C0x` runs were **pre-design probes** (exploration before the experimental design was fixed); they are kept below as a historical appendix only and are **not** part of the official results.

### Official run (MAIN)

| Run | Train/test rows | Timesteps | Split | Accuracy | Recall attack | F1 attack | Reward config |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **MAIN full** | 2,264,594/566,149 | 3,000,000 | random (fixed test) | **0.99381** | **0.99536** | **0.98445** | `tp=1.5, fp=-2.0, fn=-5.0, om=0.0` |

Secondary runs (training-size benchmark, fewer `--train-max-rows` over the same fixed test partition) are tracked under [Training-Size Benchmark](#training-size-benchmark-fixed-test-partition) (pending).

### Main Committed Run (full data)

| Field | Value |
|---|---|
| RUN_ID | `MAIN_qrdqn_cicids2017_canonical_full_random_20260609_193655` |
| Algorithm | QRDQN |
| Dataset | CICIDS2017 |
| Observation size | 152 |
| Train / test | 2,264,594 / 566,149 |
| Device | `cuda` |
| Total timesteps | `3,000,000` |
| Reward config | `tp=1.5, fp=-2.0, fn=-5.0, omission=0.0` |

> **Observation layout (152-dimensional = 76 canonical feature values + 76 missingness-mask values).**
> The observation is `[x_1..x_76, m_1..m_76]`: the first 76 entries are the canonical
> flow-feature values and the second 76 are a presence/absence mask
> (`1 = present/valid`, `0 = missing/imputed`). The mask is computed *after* the upstream
> `fillna(0)` cleaning step in `src/load_cicids2017.py` (where `±Inf → NaN → 0`); inside
> `src/canonical_schema.py` it is set to 1 (`mask[:, i] = (~bad).astype(np.float32)`) for every canonical
> feature that has a mapped source column. Because the CICIDS2017 → canonical mapping
> covers all 76 features, **on native CICIDS2017 the mask is constant = 1 for every row**:
> it encodes *source-column presence*, not per-value missingness. The mask only becomes
> informative for cross-domain / Phase 2 lab inference, where some source columns are
> absent and their mask stays 0 — e.g. a legacy NSL-KDD mapping covers only 3 of the 76
> canonical features.

Metrics from `runs/cicids2017/MAIN_qrdqn_cicids2017_canonical_full_random_20260609_193655/metrics.json`:

| Metric | Value |
|---|---|
| Accuracy | `0.99381` |
| Precision attack | `0.97378` |
| Recall attack | `0.99536` |
| F1 attack | `0.98445` |
| Precision benign | `0.99885` |
| Recall benign | `0.99343` |
| F1 benign | `0.99613` |

#### Operational metrics + bootstrap confidence intervals (A4 / A5)

Operational metrics (derived from the confirmed test confusion matrix `tn=451,631 · fp=2,989 · fn=518 · tp=111,011`, total 566,149) and **95% bootstrap confidence intervals** over the fixed seed-42 test set (`runs/validation/bootstrap_ci_seed42.json`, 10,000 stratified resamples, `scripts/bootstrap_ci.py`):

| Metric | Point | 95% CI |
|---|---|---|
| Recall attack (detection) | `0.99536` | `[0.99495, 0.99575]` |
| FPR (benign blocked) | `0.00658` | `[0.00634, 0.00681]` |
| FNR (attacks missed) | `0.00465` | `[0.00425, 0.00505]` |
| Balanced accuracy | `0.99439` | `[0.99416, 0.99462]` |
| MCC | `0.98068` | `[0.98004, 0.98130]` |
| Precision attack | `0.97378` | `[0.97286, 0.97468]` |
| F1 attack | `0.98445` | `[0.98394, 0.98496]` |
| Accuracy | `0.99381` | `[0.99360, 0.99401]` |

The point estimates reproduce `metrics.json` exactly and were **independently re-verified end-to-end** by re-running the saved MAIN model over the reproduced seed-42 test split (`--from-model`): the regenerated confusion matrix matched `(451631, 2989, 518, 111011)` exactly. The intervals are tight (±0.0002–0.001), so the single-seed point estimates are **stable under test-set resampling** — this says nothing about training-seed variability (a different training seed could land elsewhere; multiple training runs were **not** performed). *Scope:* the CIs quantify test-set **sampling** precision for this one fixed trained model. Because the seed-42 split is *stratified*, the bootstrap resamples within each class with the per-class totals (N⁻=454,620 benign, N⁺=111,529 attack) held fixed, faithful to the sampling design.

### Historical pre-design probes (C0x) — not part of the official trunk

These runs were exploratory probes launched **before** the experimental design was fixed (different `max_rows`, timesteps, and — for C01/C02 — a different false-positive reward). They are retained for traceability only and are slated for archival (`runs/archive/`); they must not be presented as official results.

| Run | Rows | Timesteps | Split | Accuracy | Recall attack | F1 attack | Reward config |
|---|---|---|---|---|---|---|---|
| C01 smoke | 50,000 | 5,000 | random | 0.9697 | 0.9996 | 0.9692 | `tp=1.5, fp=-1.0, fn=-5.0, om=0.0` |
| C01 full | 250,000 | 100,000 | random | 0.9962 | 0.9998 | 0.9963 | `tp=1.5, fp=-1.0, fn=-5.0, om=0.0` |
| C02 fast | 100,000 | 10,000 | random | 0.9766 | 0.9996 | 0.9812 | `tp=1.5, fp=-1.0, fn=-5.0, om=0.0` |
| C03 full | 500,000 | 100,000 | random | 0.99859 | 0.99945 | 0.99876 | `tp=1.5, fp=-2.0, fn=-5.0, om=0.0` |

The `Rows` column lists total rows loaded before the split (the historical `max_rows` semantics). C03 is the strongest probe; its detail follows.

#### C03 (best historical probe)

C03 used `max_rows=500000`; its 100,000-row test set has a distorted class mix (benign rate ≈0.434) and is NOT comparable with the main run's 566,149-row test set (benign rate ≈0.803).

| Field | Value |
|---|---|
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

Metrics from `runs/archive/cicids2017/C03_qrdqn_cicids2017_canonical_full_random_20260223_232439/metrics.json`:

| Metric | Value |
|---|---|
| Accuracy | `0.99859` |
| Precision attack | `0.99806` |
| Recall attack | `0.99945` |
| F1 attack | `0.99876` |
| Precision benign | `0.99928` |
| Recall benign | `0.99746` |
| F1 benign | `0.99837` |

### Training-Size Benchmark (Fixed Test Partition)

Internal benchmark to justify the full-data main experiment. Smaller runs use `--train-max-rows` (train-only subsampling AFTER the split): the test partition is byte-identical to the main run's 566,149-row test set (seed 42; benign 454,620 / attack 111,529), verified per run via `split_metadata.test_set_sha256` against the reference manifest `runs/cicids2017/test_partition_reference_seed42.json` (minted 2026-06-27; the reproduced split's scaler `mean_`/`scale_` match the committed MAIN scaler, confirming the committed artifacts correspond to the seed-42 split). Timesteps scale proportionally with train size. Protocol details: [../experiments/cicids2017_qrdqn_experiments.md](../experiments/cicids2017_qrdqn_experiments.md).

These results are an **internal CICIDS2017 benchmark only** — random stratified split with a fixed held-out test partition. They are not comparable to, and must not be mixed with, the Phase 2 offline-inference results, which measure distribution shift on operator-generated lab traffic (real captured packets, closed home lab; limited external validity).

No benchmark training artifacts are committed yet; the table below will be filled only from committed `runs/cicids2017/` artifacts.

| Run | Train rows | Timesteps | test_set_sha256 match | Accuracy | Recall attack | F1 attack |
|---|---|---|---|---|---|---|
| *(pending)* | — | — | — | — | — | — |

## Validation Artifacts

### Check A — Direct Evaluation

Artifact:

- `runs/validation/VAL_checks_A_20260212_235443/validation_results.json`

| Metric | Value |
|---|---|
| Accuracy | `0.9939` |
| Precision attack | `0.98758` |
| Recall attack | `0.99979` |
| F1 attack | `0.99365` |
| TP / FP / FN / TN | `4772 / 60 / 1 / 5167` |

### Check B — Shuffled Labels

Artifact:

- `runs/validation/VAL_checks_B_20260212_235736/validation_results.json`

| Metric | Value |
|---|---|
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
|---|---|
| Accuracy | `0.8413509399739979` |
| Precision attack | `0.764791673901073` |
| Recall attack | `0.5295374374439701` |
| F1 attack | `0.6257849253737402` |
| TP / FP / FN / TN | `154169 / 47414 / 136970 / 823660` |
| Train rows | `1,668,530` |
| Test rows | `1,162,213` |

This remains the hardest committed generalisation artifact in the repository.

### leave-one-CSV-out

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
|---|---|
| Flows CSV | `pcaps/flows_benign.csv` |
| Block rate | `1.0` |
| Allow rate | `0.0` |
| z abs max | `10.0` |
| z abs mean | `1.1148942708969116` |

This artifact documents a strong domain-shift problem on benign operator-generated lab-capture traffic (real captured packets, closed home lab; limited external validity).

### Later v2 Benign-Only Artifact

Artifact:

- `runs/phase2/P2v2_pred_20260408_230318/`

| Field | Value |
|---|---|
| Flows CSV | `pcaps/flows_benign.csv` |
| Block rate | `0.0` |
| Allow rate | `1.0` |
| z abs max | `10.0` |
| z abs mean | `0.6870886087417603` |

This later artifact shows that Phase 2 behaviour is sensitive to configuration and run conditions, which is exactly why documentation must cite the specific run artifact.

### Labeled Lab-Capture Artifact (main model)

Artifact:

- `runs/phase2/P2v2_pred_20260610_161231_MAIN/`

| Field | Value |
|---|---|
| Flows CSV | `pcaps/lab_capture_traffic.csv` |
| Model | `models/MAIN_qrdqn_cicids2017_canonical_full_random_20260609_193655.zip` |
| Block rate | `0.252364` |
| Accuracy | `0.991862` |
| Precision attack | `0.97919` |
| Recall attack | `0.988452` |
| F1 attack | `0.983801` |

This Phase 2 metric is an operator-generated lab-capture benchmark (real captured packets, closed home lab; limited external validity) and must not be mixed with the internal CICIDS2017 test results.

## NSL-KDD Historical Benchmark

The maintained historical summary lives in [../experiments/nslkdd_experiments.md](../experiments/nslkdd_experiments.md).

Short version:

| Experiment | Model | Accuracy | Recall attack | FP rate |
|---|---|---|---|---|
| E01 | DQN | `0.7602` | `0.600` | `0.028` |
| E02 | Random Forest | `0.7693` | `0.615` | `0.0267` |
| E05 | DQN | `0.7563` | `0.5955` | `0.0313` |

NSL-KDD remains historical benchmarking material only.

## Artifact Locations

### Training

- `runs/cicids2017/MAIN_qrdqn_cicids2017_canonical_full_random_20260609_193655/` (official run)
- Historical pre-design probes (C0x):
  - `runs/archive/cicids2017/C01_qrdqn_cicids2017_canonical_smoke_20260212_195959/`
  - `runs/archive/cicids2017/C01_qrdqn_cicids2017_canonical_full_20260212_200218/`
  - `runs/archive/cicids2017/C02_qrdqn_cicids2017_canonical_fast_random_20260223_181122/`
  - `runs/archive/cicids2017/C03_qrdqn_cicids2017_canonical_full_random_20260223_232439/`

### Validation

- `runs/validation/VAL_checks_A_20260212_235443/`
- `runs/validation/VAL_checks_B_20260212_235736/`
- `runs/validation/VAL_checks_C_20260213_004847/`

### Phase 2

- `runs/phase2/P2v2_pred_20260224_004121/`
- `runs/phase2/P2v2_pred_20260408_230318/`
- `runs/phase2/P2v2_pred_20260610_161231_MAIN/`

### Models

- `models/archive/C03_qrdqn_cicids2017_canonical_full_random_20260223_232439.zip`
- `models/MAIN_qrdqn_cicids2017_canonical_full_random_20260609_193655.zip`

## Random Forest Baseline

The supervised Random Forest baseline is run over the canonical schema under the **same protocol** as QRDQN: canonical observation, scaling, and `class_weight="balanced"`. The **Random Split** sweep shares the identical 566,149-row seed-42 test set with MAIN, and the **Day Split** sweep now uses the same Mon/Tue/Wed → Thu/Fri partition as QRDQN Check C, so both are strict same-split comparisons. Committed run: `rf_cicids2017_canonical_20260628_024735` (balanced/scaled; `config.json` + `metrics.json` per sweep).

**Execution Protocol:**
Run `uv run python src/baseline_random_forest.py` to regenerate. This executes three balanced/scaled sweeps across the canonical schema:

1. **Random Split (full)** — test = **566,149 rows (benign 454,620 / attack 111,529), byte-identical to MAIN's fixed test partition** (seed 42).
2. **Day Split** — train on Mon/Tue/Wed CSVs (1,668,530 rows), test on Thu/Fri CSVs (1,162,213 rows) — **identical to QRDQN Check C's partition**.
3. **Leave-One-Out (Wednesday test)** — train on 7 CSVs (2,138,040 rows), test on Wednesday (692,703 rows). No QRDQN counterpart artifact committed → RF-only.

Per-sweep artifacts under `runs/cicids2017/baseline_random_forest_comparison/rf_cicids2017_canonical_20260628_024735__{random_split,day_split,leave_one_out_wednesday}/`. (The legacy `results_rf.txt` is a superseded unbalanced prototype.)

### Baseline Metrics (balanced/scaled, `rf_cicids2017_canonical_20260628_024735`)

| Split | Test rows | Accuracy | F1 Attack | Precision Attack | Recall Attack | Notes |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Random | 566,149 | 0.99872 | 0.99676 | 0.99501 | 0.99853 | Same fixed test partition as MAIN → directly comparable. RF marginally edges QRDQN (MAIN F1 attack 0.98445). |
| Day Split | 1,162,213 (Thu+Fri) | 0.76913 | 0.15005 | 0.96473 | 0.08135 | **Same partition as QRDQN Check C** (train Mon–Wed). Directly comparable: QRDQN Check C recall attack 0.52954 / F1 0.62578 — RF attack recall collapses far more under day-shift. |
| Leave-One-Out | 692,703 (Wed) | 0.63782 | 0.01427 | 0.98482 | 0.00719 | Wednesday held-out; domain-shift stress. RF-only (no committed QRDQN LOO artifact). |

## Open Documentation Gap

The repository now includes code for leave-one-CSV-out validation, but the documentation cannot yet report aggregate metrics for it until a full committed run is added under `runs/validation/`.
