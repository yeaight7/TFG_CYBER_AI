# Methodology Handoff

This file is for later methodology/design writing and implementation tasks. It is not final thesis prose.

## Internal Benchmark Design

| Item | Actionable plan | Current status / caveat |
|---|---|---|
| Main dataset | Use CICIDS2017 as internal public benchmark. | Implemented as primary dataset. Verify exact local curated CSV counts before writing. |
| Schema | Describe 76 canonical flow features plus 76 missingness-mask values. | Strong repo fact. Verify against `src/canonical_schema.py` before final thesis. |
| Labels | Binary labels: `0 = BENIGN`, `1 = ATTACK`; actions: `0 = PERMIT`, `1 = BLOCK`. | Strong repo fact. |
| Splits | Report random split, hard CSV/day split, and leave-one-CSV-out if artifact exists. | Check C artifact exists; leave-one-CSV-out code exists but full committed artifact is missing in current docs. |
| Artifacts | Every reported run needs `RUN_ID`, config, metrics, and path. | Required by repo docs. |

## Data-Efficiency Curve

Purpose: show how performance changes as training data increases.

Recommended design:

- Train/evaluate with several row budgets or fractions, e.g. smoke, medium, full.
- Keep split protocol constant per curve.
- Run multiple seeds if compute allows.
- Plot attack recall, attack F1, false positives, false negatives, and training time.
- Compare QRDQN curve against Random Forest baseline where possible.

Status:

- Historical C01/C02/C03 runs provide some size differences, but they may differ in other settings.
- Do not call this a clean data-efficiency curve unless the protocol is controlled.

## Supervised Baseline

| Baseline | Required treatment | Current status |
|---|---|---|
| Random Forest | Primary supervised tabular baseline on same canonical schema and same splits. | Protocol exists; `docs/results.md` says metrics pending. |
| Other ML/DL baselines | Optional only if time permits. | Avoid expanding scope unless requested. |
| Comparison rule | Same data, same split, same metrics, same preprocessing. | Needed before making algorithmic claims. |

Forbidden:

- Do not claim QRDQN outperforms supervised learning until same-split baseline results exist.

## Multiple Seeds

Recommended:

- Use at least 3 seeds for any final comparative claim.
- Report mean and standard deviation for key metrics.
- Keep run configs and seed values in artifacts.

If compute is limited:

- State single-seed limitation explicitly.
- Avoid strong superiority claims.

## Metrics

Required metrics:

- Accuracy.
- Precision, recall, F1 for attack class.
- Precision, recall, F1 for benign class.
- TP, FP, FN, TN.
- False positive rate and false negative rate if not already explicit.

Optional only if supported by outputs:

- AUROC.
- AUPRC.
- Calibration metrics.

Do not add probability-score metrics unless the model/export path provides valid scores.

## False Positive / False Negative Analysis

Required discussion:

- False negatives correspond to missed attacks.
- False positives correspond to unnecessary blocking / availability or analyst-load cost.
- Reward design is an asymmetric scenario assumption, not a measured operational cost.

Recommended table:

| Error type | Security meaning | Cost assumption | Metric to inspect |
|---|---|---|---|
| FP | Benign flow blocked | Availability / usability cost | Benign recall, FPR, FP count |
| FN | Attack flow permitted | Missed intrusion cost | Attack recall, FNR, FN count |

## Attack-Family Error Analysis

Recommended if labels allow:

- Report per-attack-family recall/F1.
- Identify families with high FN.
- Separate high-volume easy classes from low-volume difficult classes.
- Avoid averaging away minority attack failure.

Current risk:

- Project uses binary labels for main RL path. If family labels are not preserved in artifacts, mark family analysis as planned or unavailable.

## Leakage Controls

Must state:

- Drop or do not use IP addresses.
- Drop or do not use absolute timestamps.
- Drop Flow IDs and unique identifiers.
- Avoid ports as direct label proxies.
- Run shuffled-label anti-leakage check when applicable.

Current repo support:

- Anti-leakage policy exists in docs and loader code.
- Check B shuffled-label artifact exists historically.

Caveat:

- Check B reduces leakage concern; it does not prove all leakage impossible.

## Strict Split / Leave-One-CSV-Out

Recommended evidence ladder:

1. Random split: internal sanity benchmark.
2. Hard CSV/day split: stronger internal generalization check.
3. leave-one-CSV-out: strongest internal CICIDS2017 file/domain shift check.

Current status:

- Check C artifact exists and should be cited with exact run ID.
- leave-one-CSV-out implementation exists.
- No committed full leave-one-CSV-out artifact is currently reported in `docs/results.md`.

Writing rule:

- Describe leave-one-CSV-out as implemented/planned unless the artifact exists.

## External Lab Validation

Preferred design:

- Capture private lab traffic.
- Extract flow CSVs with a documented toolchain.
- Map flows through the same canonical schema.
- Run offline inference with `scripts/predict_real_traffic_v2.py`.
- Store config, metrics, predictions, diagnostics, and `RUN_ID`.
- Report results per artifact.

Current status:

- Phase 2 robust offline inference exists.
- Existing benign-only artifacts show different behavior across runs.
- Do not claim external validation is complete unless final artifact and protocol support it.

## Plan A / B / C if External Validation Is Not Viable

| Plan | Use when | Methodology text |
|---|---|---|
| Plan A: Full external lab validation | Lab traffic is captured, processed, and artifact-backed. | Present as controlled offline external-distribution validation, not deployment proof. |
| Plan B: Benign-only lab sanity check | Only benign lab flows are available. | Present as a false-positive / domain-shift sanity check, not attack detection validation. |
| Plan C: No usable lab artifact | Lab capture or mapping fails. | Present CICIDS2017 strict-split validation as the main evidence and list lab validation as future work. |

## Methodology Writing Checklist

- Every result includes run ID and artifact path.
- Every algorithm comparison uses same split and preprocessing.
- Reward weights are described as assumptions.
- Missing experiments are labeled pending or future work.
- External validation is called planned/preferred unless backed by current artifacts.
- No active real-time blocking is claimed.
