# Consolidated Experiment Results

This document is auto-generated from run artifacts stored under `runs/`.
All metrics are extracted directly from JSON files — no hardcoded values.

---

## CICIDS2017 — QRDQN Training Runs

### C01 Smoke Test

| Field | Value |
|-------|-------|
| **RUN_ID** | `C01_qrdqn_cicids2017_canonical_smoke_20260212_195959` |
| **Algorithm** | QRDQN |
| **Dataset** | CICIDS2017 (canonical, 50 000 rows) |
| **Observation dim** | 152 (76 features + 76 missingness mask) |
| **Timesteps** | 5 000 |
| **Device** | cuda |
| **Net arch** | [512, 256] |
| **Learning rate** | 1 × 10⁻⁴ |
| **Batch size** | 256 |
| **Train / Test** | 40 000 / 10 000 |

**Metrics (random split)**

| Metric | Value |
|--------|-------|
| Accuracy | 0.9697 |
| Precision (attack) | 0.9407 |
| Recall (attack) | 0.9996 |
| F1 (attack) | 0.9692 |
| Precision (benign) | 0.9996 |
| Recall (benign) | 0.9424 |
| F1 (benign) | 0.9702 |

> Source: `runs/cicids2017/C01_qrdqn_cicids2017_canonical_smoke_20260212_195959/metrics.json`

---

### C01 Full Training

| Field | Value |
|-------|-------|
| **RUN_ID** | `C01_qrdqn_cicids2017_canonical_full_20260212_200218` |
| **Algorithm** | QRDQN |
| **Dataset** | CICIDS2017 (canonical, 250 000 rows) |
| **Observation dim** | 152 |
| **Timesteps** | 100 000 |
| **Device** | cuda |
| **Net arch** | [512, 256] |
| **Learning rate** | 1 × 10⁻⁴ |
| **Batch size** | 2 048 |
| **Train / Test** | 200 000 / 50 000 |

**Metrics (random split)**

| Metric | Value |
|--------|-------|
| Accuracy | 0.9962 |
| Precision (attack) | 0.9928 |
| Recall (attack) | 0.9998 |
| F1 (attack) | 0.9963 |
| Precision (benign) | 0.9998 |
| Recall (benign) | 0.9924 |
| F1 (benign) | 0.9961 |

> Source: `runs/cicids2017/C01_qrdqn_cicids2017_canonical_full_20260212_200218/metrics.json`

---

## Validation Checks

Three validation checks verify that the reported metrics are genuine:

| Check | Purpose |
|-------|---------|
| **A** | Direct `model.predict(X_test[i])` vs `y_test[i]` without relying on the env's `info["true_label"]` |
| **B** | Shuffled-labels anti-leakage test — train with randomly permuted labels and confirm accuracy drops to chance |
| **C** | CSV-split evaluation — train on Monday–Wednesday CSVs, test on Thursday–Friday (unseen days/attacks) |

### Check A — Direct Evaluation

| Field | Value |
|-------|-------|
| **RUN_ID** | `VAL_checks_A_20260212_235443` |
| **Model** | `C01_qrdqn_cicids2017_canonical_full_20260212_200218.zip` |
| **Samples** | 10 000 (50 000 rows, random split) |

| Metric | Value |
|--------|-------|
| Accuracy | 0.9939 |
| Precision (attack) | 0.9876 |
| Recall (attack) | 0.9998 |
| F1 (attack) | 0.9936 |
| TP | 4 772 |
| FP | 60 |
| FN | 1 |
| TN | 5 167 |

> Source: `runs/validation/VAL_checks_A_20260212_235443/validation_results.json`

---

### Check B — Shuffled Labels (Anti-Leakage)

| Field | Value |
|-------|-------|
| **RUN_ID** | `VAL_checks_B_20260212_235736` |
| **Timesteps** | 2 000 |
| **Train / Test** | 40 000 / 10 000 |

| Metric | Value |
|--------|-------|
| Shuffled accuracy | 0.4773 |
| Baseline (majority class) | 0.5227 |
| Leakage threshold | 0.5727 |
| **Leakage detected** | ✅ **NO** |

Interpretation: accuracy with shuffled labels is *below* the majority-class baseline, confirming no data leakage.

> Source: `runs/validation/VAL_checks_B_20260212_235736/validation_results.json`

---

### Check C — CSV-Split Evaluation (Day-Level Generalization)

| Field | Value |
|-------|-------|
| **RUN_ID** | `VAL_checks_C_20260213_004847` |
| **Train CSVs** | Monday, Tuesday, Wednesday |
| **Test CSVs** | Thursday, Friday |
| **Timesteps** | 30 000 |
| **Device** | cuda |
| **Train** | 1 668 530 rows (1 402 023 benign, 266 507 attack) |
| **Test** | 1 162 213 rows (871 074 benign, 291 139 attack) |

| Metric | Value |
|--------|-------|
| Accuracy | 0.8414 |
| Precision (attack) | 0.7648 |
| Recall (attack) | 0.5295 |
| F1 (attack) | 0.6258 |
| Precision (benign) | 0.8574 |
| Recall (benign) | 0.9456 |
| F1 (benign) | 0.8993 |
| TP | 154 169 |
| FP | 47 414 |
| FN | 136 970 |
| TN | 823 660 |

Note: Check C uses a much harder split (unseen days with different attack types) and only 30 000 training timesteps, so lower metrics are expected. A longer training run would improve generalization.

> Source: `runs/validation/VAL_checks_C_20260213_004847/validation_results.json`

---

## NSL-KDD — Phase 1 Benchmark (Historical)

Phase 1 experiments on NSL-KDD are documented in [`experiments/nslkdd_experiments.md`](../experiments/nslkdd_experiments.md). Summary of key runs:

| ID | Model | Dataset | Reward (tp, fp, fn, om) | Steps | Acc | Recall atk | FP rate |
|----|-------|---------|-------------------------|-------|-----|------------|---------|
| E01 | DQN | NSL-KDD 20% | 1.0, −1.0, −2.0, 0.0 | 200k | 0.7602 | 0.600 | 0.028 |
| E02 | RF | NSL-KDD 20% | — | — | 0.7693 | 0.615 | 0.027 |
| E05 | DQN | NSL-KDD 20% | 2.0, −1.0, −6.0, 0.2 | 500k | 0.7563 | 0.596 | 0.031 |

NSL-KDD is used solely as a Phase 1 benchmark and is **not** part of the final simulation model.

---

## Run Artifact Locations

| Run | Path |
|-----|------|
| C01 smoke | `runs/cicids2017/C01_qrdqn_cicids2017_canonical_smoke_20260212_195959/` |
| C01 full | `runs/cicids2017/C01_qrdqn_cicids2017_canonical_full_20260212_200218/` |
| Val A | `runs/validation/VAL_checks_A_20260212_235443/` |
| Val B | `runs/validation/VAL_checks_B_20260212_235736/` |
| Val C | `runs/validation/VAL_checks_C_20260213_004847/` |
| Optuna | `runs/optuna/study_20260212_222134.json` |
| NSL-KDD A02 | `runs/nslkdd/A02_dqn_arch512x256_lr1e-4_bs2048_t500k_20251214_184003_0/` |
