---
type: "query"
date: "2026-05-15T10:47:10.341842+00:00"
question: "What is the Phase 2 real traffic inference pipeline and how does it handle domain shift?"
contributor: "graphify"
source_nodes: ["predict_real_traffic_v2 main pipeline", "Domain Shift Risk (Phase 2)", "scaling_utils_apply_percentile_clipping", "scaling_utils_apply_z_clipping", "compute_diagnostics z-score analysis", "maybe_convert_time_units seconds-to-microseconds"]
---

# Q: What is the Phase 2 real traffic inference pipeline and how does it handle domain shift?

## Answer

Phase 2 (scripts/predict_real_traffic_v2.py) is the production inference script for offline analysis of real network captures. Pipeline order: (1) maybe_convert_time_units() detects if CICFlowMeter exported duration in seconds (median < 1) and converts to microseconds to match training units. (2) map_to_canonical() applies FLOWMETER_PY_TO_CANON to produce the 76-feature canonical vector + missingness mask. (3) apply_percentile_clipping() clamps each feature to its training-data [p_low, p_high] percentile range (loaded from train_percentiles.npz). (4) apply_z_clipping() clamps scaled features to [-max_z, +max_z]. (5) batched_predict() runs the QRDQN model in batches to avoid OOM on large CSVs. (6) compute_diagnostics() computes per-feature z-scores to flag distributional drift. (7) compute_truth_metrics() runs if ground-truth labels are available. Domain shift risk (identified in docs/AGENT_CONTEXT.md and Research/CLAIMS_BANK.md) is mitigated by dual clipping fitted on training data - the graph correctly links Domain Shift Risk semantically_similar_to the research claim about cross-dataset generalization weakness (Layeghy2022, Cantone2024). v1 (deprecated) lacked clipping and diagnostics.

## Source Nodes

- predict_real_traffic_v2 main pipeline
- Domain Shift Risk (Phase 2)
- scaling_utils_apply_percentile_clipping
- scaling_utils_apply_z_clipping
- compute_diagnostics z-score analysis
- maybe_convert_time_units seconds-to-microseconds