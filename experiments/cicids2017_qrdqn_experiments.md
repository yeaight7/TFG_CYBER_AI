# CICIDS2017 + QRDQN Experiment History

This document archives the committed CICIDS2017 training, validation, and Phase 2-facing runs built around the current QRDQN pipeline.

> **Official run:** `MAIN_qrdqn_cicids2017_canonical_full_random_20260609_193655` is the project's official result (full data, fixed 566,149-row test partition, 3,000,000 timesteps). Runs C01–C03 are **pre-design probes** — exploration runs from before the experimental design was fixed — and have been archived under `runs/archive/cicids2017/`. They are not official results and their metrics are not comparable to MAIN's fixed test partition.

## Status

- **Maintained archive**
- tied to committed artifacts under `runs/` and `models/`
- complements, but does not replace:
  - [../.github/AGENT_CONTEXT.md](../.github/AGENT_CONTEXT.md)
  - [../docs/results.md](../docs/results.md)

## How To Read This Page

- A result is only reported here when the repository contains the corresponding artifact.
- Historical run metadata must not be confused with the **current code defaults**.
- The current codebase default reward configuration is:

```python
REWARD_CONFIG = {
    "tp": 1.5,
    "fp": -2.0,
    "fn": -5.0,
    "omission": 0.0,
}
```

- The earliest historical runs (C01, C02) used a softer false-positive penalty, `fp = -1.0`, which differs from the current default `fp = -2.0`.

## Why This Branch Matters

The CICIDS2017 + QRDQN branch is the maintained Phase 1 baseline because it introduced:

- the fixed canonical schema of 76 flow features
- the 76-value missingness mask, producing 152-dimensional observations
- explicit artifact persistence for scaler and training percentiles
- a validation suite that goes beyond random-split accuracy
- the model and preprocessing chain used by the Phase 2 offline inference pipeline

## Training Progression

| ID | RUN_ID | Rows | Timesteps | Split | Reward config | Accuracy | Recall attack | F1 attack | Notes |
|----|--------|------|-----------|-------|---------------|----------|---------------|-----------|-------|
| C01 smoke | `C01_qrdqn_cicids2017_canonical_smoke_20260212_195959` | 50,000 | 5,000 | random | `tp=1.5, fp=-1.0, fn=-5.0, om=0.0` | `0.9697` | `0.99958` | `0.96922` | Fast smoke run used to confirm the canonical QRDQN path worked end to end. |
| C01 full | `C01_qrdqn_cicids2017_canonical_full_20260212_200218` | 250,000 | 100,000 | random | `tp=1.5, fp=-1.0, fn=-5.0, om=0.0` | `0.99618` | `0.99980` | `0.99628` | First larger random-split run showing the approach scaled cleanly. |
| C02 fast | `C02_qrdqn_cicids2017_canonical_fast_random_20260223_181122` | 100,000 | 10,000 | random | `tp=1.5, fp=-1.0, fn=-5.0, om=0.0` | `0.9766` | `0.99959` | `0.98123` | Faster preset retained strong attack recall while trading off some benign recall. |
| C03 full | `C03_qrdqn_cicids2017_canonical_full_random_20260223_232439` | 500,000 | 100,000 | random | `tp=1.5, fp=-2.0, fn=-5.0, om=0.0` | `0.99859` | `0.99945` | `0.99876` | Strongest pre-design probe (not official); measured on a 100,000-row capped test set with a distorted class mix — not comparable to MAIN's fixed test partition. |
| MAIN full | `MAIN_qrdqn_cicids2017_canonical_full_random_20260609_193655` | 2,830,743 | 3,000,000 | random | `tp=1.5, fp=-2.0, fn=-5.0, om=0.0` | `0.99381` | `0.99536` | `0.98445` | Completed full-data canonical main experiment (max_rows=null, seed 42, RunPod RTX 3090). Its 566,149-row test set is the reference benchmark; not comparable to the smaller capped C01-C03 test sets. |

## Strongest Pre-Design Probe (C03, not official)

C03 remains the strongest of the early capped-data runs, but its accuracy was measured on a 100,000-row test set with a distorted class mix (test_benign_rate 0.434), so it is **not** comparable to the completed full-data main run (`MAIN_qrdqn_cicids2017_canonical_full_random_20260609_193655`), whose 566,149-row test set (test_benign_rate 0.803) yields accuracy 0.99381. See the Training-Size Benchmark section.

The strongest of the pre-design probe artifacts is:

- `runs/archive/cicids2017/C03_qrdqn_cicids2017_canonical_full_random_20260223_232439/`

Key metadata from `config.json`:

| Field | Value |
|-------|-------|
| Algorithm | `QRDQN` |
| Dataset | `CICIDS2017` |
| Observation size | `152` |
| Train / test | `400000 / 100000` |
| Split mode | `random` |
| Device | `cuda` |
| Learning rate | `1e-4` |
| Batch size | `2048` |
| Gradient steps | `20` |
| Reward config | `tp=1.5, fp=-2.0, fn=-5.0, omission=0.0` |

Measured metrics from `metrics.json`:

| Metric | Value |
|--------|-------|
| Accuracy | `0.99859` |
| Precision attack | `0.99806` |
| Recall attack | `0.99945` |
| F1 attack | `0.99876` |
| Precision benign | `0.99928` |
| Recall benign | `0.99746` |
| F1 benign | `0.99837` |

## Partial Or Unreportable Runs

The folder:

- `runs/archive/cicids2017/C03_qrdqn_cicids2017_canonical_fast_day_20260223_231440_0/`

currently contains only a TensorBoard event file and does not include a committed `config.json` or `metrics.json`. It should therefore be treated as an exploratory or incomplete retained run, not as a measured result.

## Validation History

The CICIDS2017 branch became much more credible once validation moved beyond random-split accuracy.

| Validation | Artifact | Key result | Interpretation |
|------------|----------|------------|----------------|
| Check A | `runs/validation/VAL_checks_A_20260212_235443/` | accuracy `0.9939` | Direct prediction on `X_test` remained strong without relying on environment internals. |
| Check B | `runs/validation/VAL_checks_B_20260212_235736/` | shuffled accuracy `0.4773`, leakage detected `false` | Historical anti-leakage check indicates performance does not remain artificially high under label destruction. |
| Check C | `runs/validation/VAL_checks_C_20260213_004847/` | accuracy `0.84135`, attack recall `0.52954` | Harder generalisation regime across different CICIDS2017 CSV/day groups exposed a much more realistic difficulty level. |

### Check C Detail

`VAL_checks_C_20260213_004847` used:

- training CSV groups: `Monday`, `Tuesday`, `Wednesday`
- test CSV groups: `Thursday`, `Friday`
- timesteps: `30000`
- train rows: `1,668,530`
- test rows: `1,162,213`

This artifact is important because it demonstrates that excellent random-split metrics do not automatically imply robust day-to-day generalisation.

### Leave-One-Exact-CSV-Out

The repository now includes the implementation:

- `src/validate_leave_one_csv_out.py`

but no committed full aggregate artifact exists yet under `runs/validation/`. That means the workflow is implemented, documented, and ready to run, but it is not yet reportable here as a measured result.

## Training-Size Benchmark (Fixed Test Partition)

Internal benchmark protocol used to justify the full-data main experiment. **Not** part of the Phase 2 / offline-inference comparison.

### Motivation

The historical `--max-rows` knob caps rows at CSV-read time, *before* the stratified split, so it changes both the test-set size and its class mix. Evidence: the committed C03 artifact (`max_rows=500000`) has a test set of only 100,000 rows with `test_benign_rate ≈ 0.434`, versus 566,149 rows with `test_benign_rate ≈ 0.803` in the full main run — the sequential read starts with the all-benign Monday CSV. Results across `max_rows` values are therefore not comparable.

### Method

- `--train-max-rows N` subsamples **only the train partition after the split**; the test partition is byte-identical to the full run (566,149 rows; benign 454,620 / attack 111,529; seed 42), verified by `split_metadata.test_set_sha256`.
- Subsampling: `stratified_nested_prefix_v1` — deterministic, stratified to the full-train class ratio (±1 row), and nested (500k ⊂ 1M ⊂ full for the same seed).
- Timestep budget **scales proportionally with train size**: `timesteps = round(3,000,000 × N / 2,264,594)` (1M → ≈1,324,741; 500k → ≈662,370). The benchmark therefore measures proportional data+compute, not data size alone.
- Reference manifest: will live at `runs/cicids2017/test_partition_reference_seed42.json` once minted by `scripts/verify_fixed_test_split.py` in the same environment as the main run (pending). Once minted, every benchmark run's `split_metadata.test_set_sha256` must match it.
- Cross-environment proof: a local run of the verification script (scikit-learn 1.8.0) reproduced the exact split of the RunPod main run (scikit-learn 1.9.0) — `StandardScaler` `mean_`/`scale_` fit on the reproduced full train partition match the committed `scaler.joblib` of `MAIN_qrdqn_cicids2017_canonical_full_random_20260609_193655`.

### Commands

```bash
# 1M train rows (same fixed test partition as the main run):
python src/train_rl_defender.py --preset full --split-mode random --train-max-rows 1000000 --timesteps 1324741 --seed 42 --training-profile main-experiment

# 500k train rows:
python src/train_rl_defender.py --preset full --split-mode random --train-max-rows 500000 --timesteps 662370 --seed 42 --training-profile main-experiment

# Verification (load-only, no training):
python scripts/verify_fixed_test_split.py --sizes 500000 1000000 --seed 42 \
    --check-scaler runs/cicids2017/MAIN_qrdqn_cicids2017_canonical_full_random_20260609_193655/scaler.joblib
```

No benchmark training runs are committed yet; results will be reported here and in `docs/results.md` once their artifacts exist under `runs/cicids2017/`.

## Link To Phase 2

The CICIDS2017 QRDQN branch is also the foundation for the maintained Phase 2 offline inference path.

The Phase 2 entry point `scripts/predict_real_traffic_v2.py` takes the model/scaler/percentiles as required arguments (no hardcoded default). The earliest committed Phase 2 v2 runs used the C03 assets, but the newest committed run (`P2v2_pred_20260610_161231_MAIN`, 2026-06-10) used the full-data main model:

- model: `models/MAIN_qrdqn_cicids2017_canonical_full_random_20260609_193655.zip`
- scaler: `runs/cicids2017/MAIN_qrdqn_cicids2017_canonical_full_random_20260609_193655/scaler.joblib`
- percentiles: `runs/cicids2017/MAIN_qrdqn_cicids2017_canonical_full_random_20260609_193655/train_percentiles.npz`

Two committed Phase 2 v2 artifacts show why run-level traceability matters:

| Run | Flows CSV | Block rate | Allow rate | z abs mean | Interpretation |
|-----|-----------|------------|------------|------------|----------------|
| `P2v2_pred_20260224_004121` | `pcaps/flows_benign.csv` | `1.0` | `0.0` | `1.11489` | Early benign-only run blocked everything, highlighting domain-shift risk. |
| `P2v2_pred_20260408_230318` | `pcaps/flows_benign.csv` | `0.0` | `1.0` | `0.68709` | Later benign-only run behaved very differently, so Phase 2 claims must cite the exact artifact. |

## Historical Interpretation

The repository history is now best understood as three steps:

1. **NSL-KDD + DQN** established the initial RL framing and reward-shaping intuition.
2. **CICIDS2017 + canonical schema + QRDQN** produced the maintained baseline and most credible offline results.
3. **Phase 2 v2 inference** reused the CICIDS2017-trained assets and exposed the remaining domain-shift problem on real lab traffic.

That progression is why the current thesis-facing story should not stop at the early NSL-KDD experiments, but also should not oversell the Phase 2 branch as solved.
