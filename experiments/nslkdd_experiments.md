# NSL-KDD Historical Experiments

This document preserves the historical Phase 1 benchmark notes for NSL-KDD.

## Status

- **Historical benchmark only**
- useful for explaining project evolution
- not part of the current Phase 2-facing model path

## Why NSL-KDD Was Used

NSL-KDD was used early in the project to validate that the RL framework could learn a non-trivial binary decision policy before the pipeline was migrated to CICIDS2017 and the canonical schema.

## Why NSL-KDD Is Not the Final Model Basis

- its features are connection-era and not aligned with modern flow extractors
- it does not define the canonical schema
- it is not the dataset used for the current Phase 2 inference path

## Historical Summary Table

| ID | Model | Dataset | Reward config | Steps | Accuracy | Recall attack | FP rate | Status |
|----|-------|---------|---------------|-------|----------|---------------|---------|--------|
| E01 | DQN | NSL-KDD 20% train | `1.0, -1.0, -2.0, 0.0` | 200k | 0.7602 | 0.6000 | 0.0280 | Historical |
| E02 | Random Forest | NSL-KDD 20% train | — | — | 0.7693 | 0.6150 | 0.0267 | Historical |
| E03 | DQN | NSL-KDD 20% train | `1.0, -1.0, -5.0, 0.5` | 1000k | 0.7208 | 0.5280 | 0.0249 | Historical |
| E04 | DQN | NSL-KDD full train | `1.0, -1.0, -5.0, 0.5` | 1000k | 0.7155 | 0.5180 | 0.0254 | Historical |
| E05 | DQN | NSL-KDD 20% train | `2.0, -1.0, -6.0, 0.2` | 500k | 0.7563 | 0.5955 | 0.0313 | Historical |
| E06 | DQN | NSL-KDD 20% train | `1.5, -1.0, -5.0, 0.0` | 500k | 0.7555 | 0.5928 | 0.0296 | Historical |

## Main Takeaways

- Random Forest slightly outperformed the early DQN baseline on NSL-KDD.
- The RL reward configuration clearly changed behaviour, especially the balance between attack recall and false positives.
- These experiments were useful as a proof of concept, but they do not define the maintained pipeline for the thesis deliverable.

## Historical Interpretation

### DQN vs Random Forest

| Metric | E02 RF | E01 DQN | Better historical result |
|--------|--------|---------|--------------------------|
| Accuracy | 0.7693 | 0.7602 | RF |
| Recall attack | 0.6150 | 0.6000 | RF |
| FP rate | 0.0267 | 0.0280 | RF |

The main value of the RL branch at this stage was not raw performance leadership, but the ability to express asymmetric operational costs through a configurable reward function.

### Reward-System Sensitivity

The historical NSL-KDD experiments helped establish that:

- stronger false-negative penalties shift the policy toward more aggressive blocking
- omission rewards matter less than the main TP/FP/FN trade-offs
- reward shaping changes behaviour in ways that are harder to express cleanly with a static supervised baseline

## Legacy Architecture Notes

The NSL-KDD branch predates the fully matured CICIDS2017 canonical pipeline. Some of the implementation and preprocessing choices documented in these historical experiments should not be presented as the current project baseline.

For the current baseline, see:

- [../README.md](../README.md)
- [../.github/AGENT_CONTEXT.md](../.github/AGENT_CONTEXT.md)
- [../docs/results.md](../docs/results.md)

## Archived Follow-Up Ideas

The following ideas remain archived and should be treated as exploratory, not committed roadmap items:

- PPO / A2C / SAC comparisons
- larger-scale hyperparameter sweeps on NSL-KDD
- transfer-learning variants starting from NSL-KDD
- adversarial robustness work on the legacy branch
