---
source_file: "scripts/predict_real_traffic_v2.py"
type: "rationale"
community: "Scaling & Clipping Utils"
location: "L142"
tags:
  - graphify/rationale
  - graphify/EXTRACTED
  - community/Scaling__Clipping_Utils
---

# If time columns look like seconds (median < 1), convert to microseconds.

## Connections
- [[maybe_convert_time_units()_1]] - `rationale_for` [EXTRACTED]

#graphify/rationale #graphify/EXTRACTED #community/Scaling__Clipping_Utils