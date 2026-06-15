---
type: community
cohesion: 0.67
members: 3
---

# Features Map

**Cohesion:** 0.67 - moderately connected
**Members:** 3 nodes

## Members
- [[CICIDS2017_TO_CANON mapping dict]] - code - src/canonical_schema.py
- [[FLOWMETER_PY_TO_CANON mapping in v1]] - code - scripts/deprecated_predict_real_traffic.py
- [[FLOWMETER_PY_TO_CANON mapping in v2]] - code - scripts/predict_real_traffic_v2.py

## Live Query (requires Dataview plugin)

```dataview
TABLE source_file, type FROM #community/Features_Map
SORT file.name ASC
```
