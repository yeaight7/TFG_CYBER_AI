---
type: "query"
date: "2026-05-15T10:46:20.807955+00:00"
question: "Why do CICIDS2017_TO_CANON and FLOWMETER_PY_TO_CANON exist as two separate feature mapping dictionaries?"
contributor: "graphify"
source_nodes: ["CICIDS2017_TO_CANON mapping dict", "FLOWMETER_PY_TO_CANON mapping in v2", "map_to_canonical function", "canonical_schema_missingness_mask"]
---

# Q: Why do CICIDS2017_TO_CANON and FLOWMETER_PY_TO_CANON exist as two separate feature mapping dictionaries?

## Answer

They solve the same problem but for different input sources. CICIDS2017_TO_CANON (src/canonical_schema.py:155) maps raw CICIDS2017 CSV column names to the 76-feature canonical schema during training data loading via _prepare_cicids_features(). FLOWMETER_PY_TO_CANON (scripts/predict_real_traffic_v2.py:49, also in deprecated v1) maps CICFlowMeter Python API output column names to the same canonical schema during Phase 2 real-traffic inference. The split exists because CICIDS2017 CSV headers use different column names than CICFlowMeter's Python library output. Both dicts converge into map_to_canonical(), which produces the CanonicalResult with 76 features + 76-dim missingness mask. The INFERRED semantically_similar_to edge (confidence 0.85) correctly detected this structural redundancy - it is intentional, not a bug. A future improvement could unify them under a single registry keyed by source name.

## Source Nodes

- CICIDS2017_TO_CANON mapping dict
- FLOWMETER_PY_TO_CANON mapping in v2
- map_to_canonical function
- canonical_schema_missingness_mask