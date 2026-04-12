---
type: community
members: 19
---

# Phase 2 Inference Scripts

**Members:** 19 nodes

## Members
- [[Compute evaluation metrics when ground-truth columns are present in the flows CS]] - rationale - scripts\predict_real_traffic_v2.py
- [[Compute z-score diagnostics on scaled features (first _N_CANON dims only).]] - rationale - scripts\predict_real_traffic_v2.py
- [[If time columns look like seconds (median  1), convert to microseconds.]] - rationale - scripts\predict_real_traffic_v2.py
- [[Load QRDQN model, falling back to DQN if sb3_contrib is unavailable.]] - rationale - scripts\predict_real_traffic_v2.py
- [[Run model.predict in batches to avoid OOM on large flow CSVs.]] - rationale - scripts\predict_real_traffic_v2.py
- [[batched_predict()]] - code - scripts\predict_real_traffic.py
- [[batched_predict()_1]] - code - scripts\predict_real_traffic_v2.py
- [[compute_diagnostics()]] - code - scripts\predict_real_traffic_v2.py
- [[compute_truth_metrics()]] - code - scripts\predict_real_traffic_v2.py
- [[load_model()]] - code - scripts\predict_real_traffic.py
- [[load_model()_1]] - code - scripts\predict_real_traffic_v2.py
- [[main()_1]] - code - scripts\predict_real_traffic.py
- [[main()_2]] - code - scripts\predict_real_traffic_v2.py
- [[maybe_convert_time_units()]] - code - scripts\predict_real_traffic.py
- [[maybe_convert_time_units()_1]] - code - scripts\predict_real_traffic_v2.py
- [[parse_args()]] - code - scripts\predict_real_traffic_v2.py
- [[predict_real_traffic.py]] - code - scripts\predict_real_traffic.py
- [[predict_real_traffic_v2.py]] - code - scripts\predict_real_traffic_v2.py
- [[predict_real_traffic_v2.py — Robust Phase 2 offline inference pipeline.  Loads]] - rationale - scripts\predict_real_traffic_v2.py

## Live Query (requires Dataview plugin)

```dataview
TABLE source_file, type FROM #community/Phase_2_Inference_Scripts
SORT file.name ASC
```

## Connections to other communities
- 3 edges to [[_COMMUNITY_Project Docs and Phase Plan]]

## Top bridge nodes
- [[predict_real_traffic_v2.py]] - degree 12, connects to 1 community