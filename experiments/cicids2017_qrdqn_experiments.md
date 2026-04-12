# CICIDS2017 + QRDQN Experiment History

This document archives the committed CICIDS2017 training, validation, and Phase 2-facing runs built around the current QRDQN pipeline.

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
    "fp": -1.5,
    "fn": -5.0,
    "omission": 0.0,
}
```

- Some of the strongest historical runs used different reward values, especially `fp = -2.0`.

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
| C03 full | `C03_qrdqn_cicids2017_canonical_full_random_20260223_232439` | 500,000 | 100,000 | random | `tp=1.5, fp=-2.0, fn=-5.0, om=0.0` | `0.99859` | `0.99945` | `0.99876` | Best committed historical CICIDS2017 result in the repository. |

## Best Committed Historical Run

The strongest committed CICIDS2017 artifact is:

- `runs/cicids2017/C03_qrdqn_cicids2017_canonical_full_random_20260223_232439/`

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

- `runs/cicids2017/C03_qrdqn_cicids2017_canonical_fast_day_20260223_231440_0/`

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

## Link To Phase 2

The CICIDS2017 QRDQN branch is also the foundation for the maintained Phase 2 offline inference path:

- model: `models/C03_qrdqn_cicids2017_canonical_full_random_20260223_232439.zip`
- scaler: `runs/cicids2017/C03_qrdqn_cicids2017_canonical_full_random_20260223_232439/scaler.joblib`
- percentiles: `runs/cicids2017/C03_qrdqn_cicids2017_canonical_full_random_20260223_232439/train_percentiles.npz`

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
