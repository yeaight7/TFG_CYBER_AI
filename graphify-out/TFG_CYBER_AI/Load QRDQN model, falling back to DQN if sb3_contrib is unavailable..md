---
source_file: "scripts/predict_real_traffic_v2.py"
type: "rationale"
community: "Scaling & Clipping Utils"
location: "L165"
tags:
  - graphify/rationale
  - graphify/EXTRACTED
  - community/Scaling__Clipping_Utils
---

# Load QRDQN model, falling back to DQN if sb3_contrib is unavailable.

## Connections
- [[load_model()_1]] - `rationale_for` [EXTRACTED]

#graphify/rationale #graphify/EXTRACTED #community/Scaling__Clipping_Utils