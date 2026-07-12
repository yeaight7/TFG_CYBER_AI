# AGENT_CONTEXT — TFG_CYBER_AI (could be obsolete or misaligned with new project direction)

This file is the project-wide technical source of truth for contributors and coding agents.

## Project Goal

Build and evaluate an RL-based cybersecurity defender that observes flow-based network features and decides:

- `0 = PERMIT`
- `1 = BLOCK`

The project is organised in two major phases:

- **Phase 1**: offline training and validation on datasets
- **Phase 2**: offline inference on flow features extracted from traffic captured in a private lab

## Current Implementation Snapshot

### Implemented

- Fixed canonical schema with 76 flow-based features
- Missingness-mask augmentation, resulting in 152-dimensional observations
- CICIDS2017 adapter with canonical mapping
- NSL-KDD adapter for historical benchmarking
- Custom Gymnasium environment for binary defender actions
- QRDQN training pipeline
- Validation checks A, B, and C
- leave-one-CSV-out validation script for CICIDS2017
- Robust Phase 2 offline inference pipeline (`predict_real_traffic_v2.py`)

### Not Implemented

- Real-time active blocking with `iptables` / `nftables`
- Multi-agent or adversarial RL
- Multi-dataset merged training beyond the current adapter architecture
- Fully completed Phase 2 calibration for real benign traffic

## Core Invariants

### Canonical Observation Space

- `FEATURES_CANON` contains **76** canonical flow features.
- Final observation size is **152**:
  - 76 canonical feature values
  - 76 missingness-mask values
- Mask semantics:
  - `1 = present / valid`
  - `0 = missing / imputed`

### Labels and Adapter Contract

All dataset adapters should expose:

`(X_train, y_train, X_test, y_test, scaler, feature_names)`

Expected semantics:

- `X`: `float32`
- `y`: `int64`
- labels:
  - `0 = BENIGN`
  - `1 = ATTACK`

### Anti-Leakage Policy

The following must not be used as model features:

- source or destination IPs
- absolute timestamps
- Flow IDs or unique identifiers
- direct port fields when they act as label proxies

## Datasets

### CICIDS2017

This is the primary dataset and the basis for the canonical schema.

- flow-based features exported by CICFlowMeter
- current adapter supports:
  - random stratified split
  - CSV/day pattern split
  - exact-file split for leave-one-CSV-out validation
  - train-only subsampling for the internal training-size benchmark (`train_max_rows`)

#### Train-only subsampling (`train_max_rows`)

`load_cicids2017_split(..., train_max_rows=N)` subsamples only the **train** partition AFTER the split; the test partition stays byte-identical to the `train_max_rows=None` run for the same seed/preset (checkable via the `test_set_sha256` content hash in `split_metadata`; the cross-run reference manifest is still pending — see below). This exists because `max_rows` caps rows at CSV-read time, *before* the split, which shrinks the test set and distorts its class mix (sequential read starts with all-benign Monday) — unusable for comparing training-set sizes.

- requires the effective `max_rows` to be `None` (`preset="full"`, no explicit `--max-rows`); otherwise the loader raises
- subsampling is deterministic, stratified, and **nested** (`stratified_nested_prefix_v1`: 500k ⊂ 1M ⊂ full for the same seed)
- every load **via the current code** (`src/load_cicids2017.py`) records `test_set_sha256`, `y_test_sha256`, `train_set_sha256`, `y_train_sha256`, `n_train_full`, `subsample_method`, and `scale` in `split_metadata`
- **the committed MAIN run predates this hashing**: `runs/cicids2017/MAIN_qrdqn_cicids2017_canonical_full_random_20260609_193655/config.json → split_metadata` contains only counts/ratios (no `*_sha256` keys). The byte-identity guarantee therefore applies to runs produced after the hashing was added, not to MAIN's committed artifact.
- the fixed test-partition reference manifest will live at `runs/cicids2017/test_partition_reference_seed42.json` once minted on RunPod by `scripts/verify_fixed_test_split.py` (**pending — the hash-based cross-run verification has not yet been exercised against a committed reference**)
- this is an **internal benchmark** mechanism only; do not mix its claims with Phase 2 / offline-inference results

Impact of this change: current defaults unchanged (`train_max_rows=None` reproduces prior behavior exactly), historical comparisons unaffected, prior run artifacts remain reproducible, and the new `split_metadata` keys are additive.

#### Dataset versions

Two versions of the CICIDS2017 data exist locally:

| Version | Path | Tracked in git | Description |
|---|---|---|---|
| Curated | `datasets/CICIDS2017/*.csv` | Yes | Leakage-prone and redundant columns removed as a pre-ingestion step. This is what the adapter loads. |
| Raw | `datasets/CICIDS2017/Raw_dataset/` | No (gitignored) | Original, unmodified CICFlowMeter CSV exports. All original columns preserved. Kept locally for reference and reproducibility. |

The adapter in `src/load_cicids2017.py` performs additional cleaning at load time (numeric coercion, inf/NaN handling, canonical mapping). The curated CSVs reduce the ingestion surface, but the code-level anti-leakage policy in the adapter is the authoritative gate. If someone starts from the raw exports, the adapter still works — the anti-leakage drops in code cover what curation already removed.

### NSL-KDD

This dataset is kept for historical Phase 1 benchmarking.

- **Removed from the repo (2026-06-27, decision D-8):** `datasets/nsl_kdd/` and `models/rf_nslkdd.joblib` are no longer tracked (gitignored). `src/load_nsl_kdd.py` is retained for reference only and needs NSL-KDD files supplied locally.
- not part of the final Phase 2 model path
- only partially mappable to the canonical schema
- useful for historical comparisons, not for the final simulation-facing pipeline

## Main Code Entry Points

| Purpose | File |
|---|---|
| Canonical schema | `src/canonical_schema.py` |
| CICIDS2017 adapter | `src/load_cicids2017.py` |
| NSL-KDD adapter | `src/load_nsl_kdd.py` |
| RL environment | `src/rl_defender_env.py` |
| Training | `src/train_rl_defender.py` |
| Validation checks A/B/C | `src/validate_checks.py` |
| leave-one-CSV-out | `src/validate_leave_one_csv_out.py` |
| Phase 2 robust inference | `scripts/predict_real_traffic_v2.py` |
| Random Forest baseline | `src/baseline_random_forest.py` |

## Training and Validation

### Current Code Defaults

The current codebase is standardised around:

```python
REWARD_CONFIG = {
    "tp": 1.5,
    "fp": -2.0,
    "fn": -5.0,
    "omission": 0.0,
}
```

This matches:

- `src/train_rl_defender.py`
- `src/validate_checks.py`
- `src/validate_leave_one_csv_out.py`
- default environment reward values in `src/rl_defender_env.py`

### Important Historical Note

Some older committed historical run artifacts were produced with different reward settings. Example:

- early C01 artifacts used:
  - `tp = 1.5`
  - `fp = -1.0`
  - `fn = -5.0`
  - `omission = 0.0`

Documentation must distinguish between:

- **current code defaults**
- **historical run metadata**

### Validation Workflows

| Validation | Status | Notes |
|---|---|---|
| Check A | Implemented | direct prediction vs `y_test` |
| Check B | Implemented | shuffled-label anti-leakage |
| Check C | Implemented | hard CSV/day split |
| leave-one-CSV-out | Implemented in code | no committed full run artifact yet |

## Phase 2 Status

### Current State

Phase 2 currently supports **offline inference** on extracted flow CSVs.

The maintained inference entry point is:

- `scripts/predict_real_traffic_v2.py`

Key features:

- persisted scaler loading
- optional percentile clipping on raw features
- optional z-score clipping on scaled features
- diagnostics export
- batched prediction

### What Phase 2 Does Not Yet Guarantee

- stable performance on all real benign traffic
- active inline blocking
- retraining or calibration loop in the lab

## Results Snapshot

Artifact-backed highlights currently available in the repository:

- Official run (MAIN) — completed, fixed test partition:
  - `MAIN_qrdqn_cicids2017_canonical_full_random_20260609_193655`
  - training_profile: `main-experiment`, preset: `full`, timesteps: 3,000,000
  - train shape: [2,264,594 × 152], test shape: [566,149 × 152]
  - accuracy `0.9938055`, attack recall `0.9953555`, attack F1 `0.9844499`
- Best historical pre-design probe (not official):
  - `C03_qrdqn_cicids2017_canonical_full_random_20260223_232439` (max_rows=500,000; 100,000-row test set — not the fixed benchmark partition)
  - accuracy `0.99859`, attack recall `0.99945`, attack F1 `0.99876`
  - **Not directly comparable with MAIN**: C03 used `max_rows=500,000` (distorted 100,000-row test set); MAIN uses the fixed 566,149-row test partition
- Historical Check C artifact:
  - accuracy `0.84135`
- Phase 2:
  - early v2 benign-only artifact shows aggressive blocking
  - later v2 benign-only artifact shows different behaviour, so claims must always cite the specific run

See [../docs/results.md](../docs/results.md) for the maintained results snapshot.

## Immediate Next Steps

- Run and review a full leave-one-CSV-out validation artifact
- Reassess Phase 2 behaviour on benign and mixed traffic with the current robust inference settings
- Decide whether additional calibration or fine-tuning on lab-derived data is required
- Keep documentation aligned with run artifacts instead of stale narrative

## Documentation Map

- [../README.md](../README.md): public overview
- [../docs/README.md](../docs/README.md): documentation index
- [../docs/results.md](../docs/results.md): artifact-backed results
- [../docs/AGENT_CONTEXT.md](../docs/AGENT_CONTEXT.md): Phase 2 scope only

## Rule for Contributors

If you change:

- reward configuration
- split logic
- preprocessing
- canonical mapping
- evaluation methodology

then you must update the relevant documentation and clearly state whether the change affects:

- current defaults
- historical comparisons
- reproducibility of prior run artifacts
