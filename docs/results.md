# Consolidated Experiment Results

This document is auto-generated from run artifacts stored under `runs/`.
All metrics are extracted directly from JSON files — no hardcoded values.

---

## CICIDS2017 — QRDQN Training Runs

### Summary Table

| Run | Preset | Rows | Timesteps | Split | Accuracy | Recall atk | F1 atk |
|-----|--------|------|-----------|-------|----------|------------|--------|
| C01 smoke | fast | 50k | 5k | random | 0.9697 | 0.9996 | 0.9692 |
| C01 full | full | 250k | 100k | random | 0.9962 | 0.9998 | 0.9963 |
| C02 fast | fast | 100k | 10k | random | 0.9766 | 0.9996 | 0.9812 |
| **C03 full** | **full** | **500k** | **100k** | **random** | **0.9986** | **0.9995** | **0.9988** |

> **Best model**: C03 full (accuracy 0.9986, F1 attack 0.9988)

---

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

### C02 Fast Training (Random Split)

| Field | Value |
|-------|-------|
| **RUN_ID** | `C02_qrdqn_cicids2017_canonical_fast_random_20260223_181122` |
| **Algorithm** | QRDQN |
| **Dataset** | CICIDS2017 (canonical, 100 000 rows) |
| **Observation dim** | 152 |
| **Timesteps** | 10 000 |
| **Device** | cuda |
| **Net arch** | [512, 256] |
| **Learning rate** | 1 × 10⁻⁴ |
| **Batch size** | 256 |
| **Gradient steps** | 10 |
| **Train freq** | 50 |
| **Train / Test** | 80 000 / 20 000 |
| **Reward** | tp=1.5, fp=−1.0, fn=−5.0, om=0.0 |

**Metrics (random split)**

| Metric | Value |
|--------|-------|
| Accuracy | 0.9766 |
| Precision (attack) | 0.9635 |
| Recall (attack) | 0.9996 |
| F1 (attack) | 0.9812 |
| Precision (benign) | 0.9993 |
| Recall (benign) | 0.9403 |
| F1 (benign) | 0.9689 |

> Source: `runs/cicids2017/C02_qrdqn_cicids2017_canonical_fast_random_20260223_181122/metrics.json`

---

### C03 Full Training (Random Split) — ⭐ Best Model

| Field | Value |
|-------|-------|
| **RUN_ID** | `C03_qrdqn_cicids2017_canonical_full_random_20260223_232439` |
| **Algorithm** | QRDQN |
| **Dataset** | CICIDS2017 (canonical, 500 000 rows) |
| **Observation dim** | 152 |
| **Timesteps** | 100 000 |
| **Device** | cuda |
| **Net arch** | [512, 256] |
| **Learning rate** | 1 × 10⁻⁴ |
| **Batch size** | 2 048 |
| **Gradient steps** | 20 |
| **Train freq** | 100 |
| **Train / Test** | 400 000 / 100 000 |
| **Reward** | tp=1.5, fp=−2.0, fn=−5.0, om=0.0 |

**Metrics (random split)**

| Metric | Value |
|--------|-------|
| Accuracy | **0.9986** |
| Precision (attack) | 0.9981 |
| Recall (attack) | **0.9995** |
| F1 (attack) | **0.9988** |
| Precision (benign) | 0.9993 |
| Recall (benign) | 0.9975 |
| F1 (benign) | 0.9984 |

Key differences from C01 full: 500k rows (vs 250k), gradient_steps=20 (vs default), fp penalty=−2.0 (vs −1.0).

> Source: `runs/cicids2017/C03_qrdqn_cicids2017_canonical_full_random_20260223_232439/metrics.json`

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

## Optuna Hyperparameter Study

| Field | Value |
|-------|-------|
| **Study** | `study_20260212_222134.json` |
| **Trials** | 10 |
| **Timesteps per trial** | 10 000 |
| **Max rows** | 50 000 |

**Best trial result**: Accuracy 0.9939

| Hyperparameter | Best Value |
|----------------|------------|
| Learning rate | 5.2 × 10⁻⁴ |
| Batch size | 256 |
| Gradient steps | 10 |
| Gamma | 0.956 |
| Train freq | 100 |

> Source: `runs/optuna/study_20260212_222134.json`

---

## Phase 2 — Offline Inference on Lab-Captured Traffic

Phase 2 runs evaluate the trained QRDQN model on flow features extracted from real PCAPs captured in a private lab environment. The v2 inference script (`scripts/predict_real_traffic_v2.py`) applies z-score clipping to handle out-of-distribution features.

### v2 Prediction Runs (Robust Inference)

All v2 runs use the **C03 full model** with the matching scaler and z-score clipping at 10.0.

| Run ID | Flows CSV | Flows | Block Rate | Allow Rate | z-abs max | z-abs mean |
|--------|-----------|-------|------------|------------|-----------|------------|
| P2v2\_…235033 | flows.csv | 1 261 | 20.7 % | 79.3 % | 10.0 | 0.714 |
| P2v2\_…004121 | flows\_benign.csv | 5 327 | 100 % | 0 % | 10.0 | 1.115 |
| P2v2\_…004242 | flows\_scan.csv | 8 721 | 59.7 % | 40.3 % | 10.0 | 0.991 |
| P2v2\_…004306 | flows\_mix.csv | 5 511 | 95.5 % | 4.5 % | 10.0 | 1.107 |

**Observations**:
- `flows_benign.csv` (pure benign traffic): 100 % block rate indicates distribution shift between lab-captured benign traffic and CICIDS2017 training data. Further calibration or fine-tuning is needed.
- `flows_mix.csv` (mixed traffic): 95.5 % block rate — agent is aggressive, consistent with the strong FN penalty in the reward config.
- `flows_scan.csv` (port scans): 59.7 % block rate — partial detection of scan traffic.
- `flows.csv` (general lab traffic): 20.7 % block rate — more conservative predictions.

### v1 Prediction Runs (Legacy)

| Run ID | Flows CSV | Model | Block Rate | Allow Rate |
|--------|-----------|-------|------------|------------|
| P2\_pred\_20260223\_155850 | flows.csv | C01 full | 0 % | 100 % |
| P2\_pred\_20260223\_163318 | flows.csv | C01 full | 100 % | 0 % |

v1 runs showed extreme predictions (all-block or all-allow), which motivated the development of the v2 robust inference pipeline with proper scaling and z-score clipping.

> Source: `runs/phase2/P2v2_pred_*/metrics.json`

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
| C02 fast | `runs/cicids2017/C02_qrdqn_cicids2017_canonical_fast_random_20260223_181122/` |
| **C03 full** | `runs/cicids2017/C03_qrdqn_cicids2017_canonical_full_random_20260223_232439/` |
| Val A | `runs/validation/VAL_checks_A_20260212_235443/` |
| Val B | `runs/validation/VAL_checks_B_20260212_235736/` |
| Val C | `runs/validation/VAL_checks_C_20260213_004847/` |
| Optuna | `runs/optuna/study_20260212_222134.json` |
| P2v2 (flows.csv) | `runs/phase2/P2v2_pred_20260223_235033/` |
| P2v2 (benign) | `runs/phase2/P2v2_pred_20260224_004121/` |
| P2v2 (scan) | `runs/phase2/P2v2_pred_20260224_004242/` |
| P2v2 (mix) | `runs/phase2/P2v2_pred_20260224_004306/` |
| NSL-KDD A02 | `runs/nslkdd/A02_dqn_arch512x256_lr1e-4_bs2048_t500k_20251214_184003_0/` |
