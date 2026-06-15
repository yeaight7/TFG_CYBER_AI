---
type: community
cohesion: 0.15
members: 17
---

# Data Load Tools

**Cohesion:** 0.15 - loosely connected
**Members:** 17 nodes

## Members
- [[CanonicalResult dataclass]] - code - src/canonical_schema.py
- [[FEATURES_CANON (76 canonical network flow features)]] - code - src/canonical_schema.py
- [[Missingness Mask Design (observation = features + mask)]] - rationale - src/canonical_schema.py
- [[NSL_KDD_TO_CANON mapping dict (partial)]] - code - src/canonical_schema.py
- [[batched_predict in v1]] - code - scripts/deprecated_predict_real_traffic.py
- [[batched_predict in v2]] - code - scripts/predict_real_traffic_v2.py
- [[compute_diagnostics z-score analysis]] - code - scripts/predict_real_traffic_v2.py
- [[compute_truth_metrics optional ground-truth eval]] - code - scripts/predict_real_traffic_v2.py
- [[deprecated_predict_real_traffic main pipeline]] - code - scripts/deprecated_predict_real_traffic.py
- [[load_model (QRDQNDQN fallback) in v1]] - code - scripts/deprecated_predict_real_traffic.py
- [[load_model (QRDQNDQN fallback) in v2]] - code - scripts/predict_real_traffic_v2.py
- [[load_nsl_kdd_binary function]] - code - src/load_nsl_kdd.py
- [[map_to_canonical function]] - code - src/canonical_schema.py
- [[maybe_convert_time_units seconds-to-microseconds]] - code - scripts/predict_real_traffic_v2.py
- [[predict_real_traffic_v2 main pipeline]] - code - scripts/predict_real_traffic_v2.py
- [[test_canonical_schema test module]] - code - tests/test_canonical_schema.py
- [[test_predict_real_traffic_v2 test module]] - code - tests/test_predict_real_traffic_v2.py

## Live Query (requires Dataview plugin)

```dataview
TABLE source_file, type FROM #community/Data_Load_Tools
SORT file.name ASC
```

## Connections to other communities
- 2 edges to [[_COMMUNITY_Inference Diagnostics]]

## Top bridge nodes
- [[predict_real_traffic_v2 main pipeline]] - degree 9, connects to 1 community