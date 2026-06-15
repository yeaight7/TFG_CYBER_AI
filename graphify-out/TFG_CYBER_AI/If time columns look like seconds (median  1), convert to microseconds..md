---
source_file: "scripts/predict_real_traffic_v2.py"
type: "rationale"
community: "Inference Diagnostics"
location: "L142"
tags:
  - graphify/rationale
  - graphify/EXTRACTED
  - community/Inference_Diagnostics
---

# If time columns look like seconds (median < 1), convert to microseconds.

## Connections
- [[maybe_convert_time_units()_1]] - `rationale_for` [EXTRACTED]

#graphify/rationale #graphify/EXTRACTED #community/Inference_Diagnostics