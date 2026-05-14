---
type: community
members: 32
---

# Community 1

**Members:** 32 nodes

## Members
- [[Active Blocking Future Work]] - document - .github/AGENT_CONTEXT.md
- [[Clamp each feature to its training percentile range p_low, p_high.      Appl]] - rationale - src/scaling_utils.py
- [[Clamp scaled features to -max_z, +max_z.      Applied to scaled features a]] - rationale - src/scaling_utils.py
- [[Compute evaluation metrics when ground-truth columns are present in the flows CS]] - rationale - scripts/predict_real_traffic_v2.py
- [[Compute z-score diagnostics on scaled features (first _N_CANON dims only).]] - rationale - scripts/predict_real_traffic_v2.py
- [[Domain Shift Risk]] - document - docs/AGENT_CONTEXT.md
- [[Honest Defense Positioning]] - document - docs/DEFENSA_TFG_PROGRESO.md
- [[If time columns look like seconds (median  1), convert to microseconds.]] - rationale - scripts/predict_real_traffic_v2.py
- [[Load QRDQN model, falling back to DQN if sb3_contrib is unavailable.]] - rationale - scripts/predict_real_traffic_v2.py
- [[Phase 2 Execution Plan]] - document - docs/phase2_plan.md
- [[Phase 2 Offline Inference]] - document - docs/AGENT_CONTEXT.md
- [[Private Lab Setup Guide]] - document - docs/gcp_lab.md
- [[Private Lab Workflow]] - document - docs/phase2_plan.md
- [[Reproducible Run Artifacts]] - document - docs/AGENT_CONTEXT.md
- [[Robust v2 Inference Pipeline]] - document - docs/AGENT_CONTEXT.md
- [[Run model.predict in batches to avoid OOM on large flow CSVs.]] - rationale - scripts/predict_real_traffic_v2.py
- [[Spanish Defense Progress Notes]] - document - docs/DEFENSA_TFG_PROGRESO.md
- [[Spanish Oral Defense Script]] - document - docs/DEFENSA_TFG_SCRIPT.md
- [[Two-VM Private Lab Topology]] - document - docs/gcp_lab.md
- [[apply_percentile_clipping()]] - code - src/scaling_utils.py
- [[apply_z_clipping()]] - code - src/scaling_utils.py
- [[batched_predict()_1]] - code - scripts/predict_real_traffic_v2.py
- [[compute_diagnostics()]] - code - scripts/predict_real_traffic_v2.py
- [[compute_truth_metrics()]] - code - scripts/predict_real_traffic_v2.py
- [[load_model()_1]] - code - scripts/predict_real_traffic_v2.py
- [[main()_2]] - code - scripts/predict_real_traffic_v2.py
- [[maybe_convert_time_units()_1]] - code - scripts/predict_real_traffic_v2.py
- [[parse_args()]] - code - scripts/predict_real_traffic_v2.py
- [[predict_real_traffic_v2.py]] - code - scripts/predict_real_traffic_v2.py
- [[predict_real_traffic_v2.py — Robust Phase 2 offline inference pipeline.  Loads]] - rationale - scripts/predict_real_traffic_v2.py
- [[scaling_utils.py]] - code - src/scaling_utils.py
- [[scaling_utils.py — Clipping utilities for outlier handling in RL inference pipel]] - rationale - src/scaling_utils.py

## Live Query (requires Dataview plugin)

```dataview
TABLE source_file, type FROM #community/Community_1
SORT file.name ASC
```

## Connections to other communities
- 2 edges to [[_COMMUNITY_Community 6]]
- 1 edge to [[_COMMUNITY_Community 3]]

## Top bridge nodes
- [[main()_2]] - degree 11, connects to 1 community
- [[Phase 2 Offline Inference]] - degree 7, connects to 1 community