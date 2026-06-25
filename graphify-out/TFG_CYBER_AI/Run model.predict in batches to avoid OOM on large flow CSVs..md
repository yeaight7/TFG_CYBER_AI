---
source_file: "scripts/predict_real_traffic_v2.py"
type: "rationale"
community: "Scaling & Clipping Utils"
location: "L179"
tags:
  - graphify/rationale
  - graphify/EXTRACTED
  - community/Scaling__Clipping_Utils
---

# Run model.predict in batches to avoid OOM on large flow CSVs.

## Connections
- [[batched_predict()_1]] - `rationale_for` [EXTRACTED]

#graphify/rationale #graphify/EXTRACTED #community/Scaling__Clipping_Utils