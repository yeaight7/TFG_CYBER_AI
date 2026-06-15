---
type: community
cohesion: 0.09
members: 28
---

# Inference Diagnostics

**Cohesion:** 0.09 - loosely connected
**Members:** 28 nodes

## Members
- [[Clamp each feature to its training percentile range p_low, p_high.      Appl]] - rationale - src/scaling_utils.py
- [[Clamp scaled features to -max_z, +max_z.      Applied to scaled features a]] - rationale - src/scaling_utils.py
- [[Compute evaluation metrics when ground-truth columns are present in the flows CS]] - rationale - scripts/predict_real_traffic_v2.py
- [[Compute z-score diagnostics on scaled features (first _N_CANON dims only).]] - rationale - scripts/predict_real_traffic_v2.py
- [[Distribution Shift Defense via Dual Clipping]] - rationale - src/scaling_utils.py
- [[If time columns look like seconds (median  1), convert to microseconds.]] - rationale - scripts/predict_real_traffic_v2.py
- [[Load QRDQN model, falling back to DQN if sb3_contrib is unavailable.]] - rationale - scripts/predict_real_traffic_v2.py
- [[NSL-KDD Dataset (historical benchmark)]] - document - docs/Personal Research/deep-defense-research/02-datos-esquema-canonico-y-preprocesado.md
- [[Optuna Hyperparameter Search]] - rationale - docs/Personal Research/project-components-defense-research-report.md
- [[Project Components Defense Research Report]] - document - docs/Personal Research/project-components-defense-research-report.md
- [[Run model.predict in batches to avoid OOM on large flow CSVs.]] - rationale - scripts/predict_real_traffic_v2.py
- [[Scaling Utilities (apply_percentile_clipping, apply_z_clipping)]] - code - src/scaling_utils.py
- [[apply_percentile_clipping()]] - code - src/scaling_utils.py
- [[apply_z_clipping()]] - code - src/scaling_utils.py
- [[batched_predict()_1]] - code - scripts/predict_real_traffic_v2.py
- [[compute_diagnostics()]] - code - scripts/predict_real_traffic_v2.py
- [[compute_truth_metrics()]] - code - scripts/predict_real_traffic_v2.py
- [[load_model()_1]] - code - scripts/predict_real_traffic_v2.py
- [[main()_3]] - code - scripts/predict_real_traffic_v2.py
- [[maybe_convert_time_units()_1]] - code - scripts/predict_real_traffic_v2.py
- [[parse_args()_1]] - code - scripts/predict_real_traffic_v2.py
- [[predict_real_traffic_v2.py]] - code - scripts/predict_real_traffic_v2.py
- [[predict_real_traffic_v2.py — Robust Phase 2 offline inference pipeline.  Loads]] - rationale - scripts/predict_real_traffic_v2.py
- [[scaling_utils.py]] - code - src/scaling_utils.py
- [[scaling_utils.py — Clipping utilities for outlier handling in RL inference pipel]] - rationale - src/scaling_utils.py
- [[test_compute_diagnostics()]] - code - tests/test_predict_real_traffic_v2.py
- [[test_maybe_convert_time_units()]] - code - tests/test_predict_real_traffic_v2.py
- [[test_predict_real_traffic_v2.py]] - code - tests/test_predict_real_traffic_v2.py

## Live Query (requires Dataview plugin)

```dataview
TABLE source_file, type FROM #community/Inference_Diagnostics
SORT file.name ASC
```

## Connections to other communities
- 2 edges to [[_COMMUNITY_Data Load Tools]]
- 1 edge to [[_COMMUNITY_Canonical Schema Setup]]
- 1 edge to [[_COMMUNITY_Project Overview]]

## Top bridge nodes
- [[main()_3]] - degree 10, connects to 1 community
- [[apply_percentile_clipping()]] - degree 5, connects to 1 community
- [[apply_z_clipping()]] - degree 5, connects to 1 community
- [[NSL-KDD Dataset (historical benchmark)]] - degree 2, connects to 1 community