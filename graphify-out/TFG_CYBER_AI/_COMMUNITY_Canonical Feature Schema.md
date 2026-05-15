---
type: community
members: 22
---

# Canonical Feature Schema

**Members:** 22 nodes

## Members
- [[CICIDS2017_TO_CANON mapping dict]] - code - src/canonical_schema.py
- [[CanonicalResult dataclass]] - code - src/canonical_schema.py
- [[FEATURES_CANON (76 canonical network flow features)]] - code - src/canonical_schema.py
- [[FLOWMETER_PY_TO_CANON mapping in v1]] - code - scripts/deprecated_predict_real_traffic.py
- [[FLOWMETER_PY_TO_CANON mapping in v2]] - code - scripts/predict_real_traffic_v2.py
- [[Missingness Mask Design (observation = features + mask)]] - rationale - src/canonical_schema.py
- [[NSL_KDD_TO_CANON mapping dict (partial)]] - code - src/canonical_schema.py
- [[_prepare_cicids_features internal function]] - code - src/load_cicids2017.py
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
- [[test_load_cicids2017 test module]] - code - tests/test_load_cicids2017.py
- [[test_predict_real_traffic_v2 test module]] - code - tests/test_predict_real_traffic_v2.py

## Live Query (requires Dataview plugin)

```dataview
TABLE source_file, type FROM #community/Canonical_Feature_Schema
SORT file.name ASC
```

## Connections to other communities
- 2 edges to [[_COMMUNITY_Scaling & Clipping Utilities]]
- 2 edges to [[_COMMUNITY_RL Training & Validation Patterns]]

## Top bridge nodes
- [[predict_real_traffic_v2 main pipeline]] - degree 9, connects to 1 community
- [[deprecated_predict_real_traffic main pipeline]] - degree 5, connects to 1 community
- [[_prepare_cicids_features internal function]] - degree 4, connects to 1 community