---
source_file: "scripts/predict_real_traffic_v2.py"
type: "code"
community: "Canonical Feature Schema"
location: "line 353"
tags:
  - graphify/code
  - graphify/EXTRACTED
  - community/Canonical_Feature_Schema
---

# predict_real_traffic_v2 main pipeline

## Connections
- [[apply_percentile_clipping()]] - `calls` [EXTRACTED]
- [[apply_z_clipping()]] - `calls` [EXTRACTED]
- [[batched_predict in v2]] - `calls` [EXTRACTED]
- [[compute_diagnostics z-score analysis]] - `calls` [EXTRACTED]
- [[compute_truth_metrics optional ground-truth eval]] - `calls` [EXTRACTED]
- [[deprecated_predict_real_traffic main pipeline]] - `conceptually_related_to` [EXTRACTED]
- [[load_model (QRDQNDQN fallback) in v2]] - `calls` [EXTRACTED]
- [[map_to_canonical function]] - `calls` [EXTRACTED]
- [[maybe_convert_time_units seconds-to-microseconds]] - `calls` [EXTRACTED]

#graphify/code #graphify/EXTRACTED #community/Canonical_Feature_Schema