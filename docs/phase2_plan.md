# Phase 2 Plan

This document describes the intended operational workflow for Phase 2: evaluating the trained defender on flow features extracted from traffic captured in a private lab.

## Objective

Move from offline dataset evaluation to a realistic, but still controlled, traffic-evaluation workflow:

1. generate labelled traffic in an isolated lab
2. capture PCAPs
3. extract flow features
4. map flows to the canonical schema
5. run offline inference
6. review metrics and diagnostics

## Current Baseline

Phase 2 is currently **offline inference only**.

The maintained inference entry point is:

- `scripts/predict_real_traffic_v2.py`

This baseline does **not** yet include real-time active blocking.

## Inputs and Outputs

### Inputs

- captured PCAPs or already extracted flow CSVs
- trained QRDQN model
- persisted scaler from the training run
- optional training percentiles for raw-feature clipping

### Outputs

- predictions per flow
- summary metrics such as block rate / allow rate
- optional diagnostics highlighting feature-distribution shift

## Execution Steps

### 1. Prepare the Private Lab

Use the isolated topology described in [gcp_lab.md](gcp_lab.md).

At minimum:

- one attacker VM
- one defender VM
- only controlled SSH access
- no external scanning

### 2. Generate Labelled Traffic

Traffic categories should include:

- benign HTTP
- benign SSH or file transfer
- scans
- DoS / DDoS bursts
- web attacks if suitable targets are present

Ground-truth must be logged outside the model itself.

### 3. Capture Traffic

Typical commands:

```bash
sudo tcpdump -i eth0 -w /data/captures/session_$(date +%Y%m%d_%H%M%S).pcap
```

or:

```bash
tshark -i eth0 -b duration:300 -w /data/captures/session.pcap
```

### 4. Extract Flow Features

Use CICFlowMeter or a compatible extractor.

Example:

```bash
java -jar CICFlowMeter.jar -i /data/captures/ -o /data/flows/
```

### 5. Map to the Canonical Schema

Use the mapping implemented in:

- `src/canonical_schema.py`
- `scripts/predict_real_traffic_v2.py` via `FLOWMETER_PY_TO_CANON`

Expected observation shape:

- 76 canonical feature values
- 76 missingness-mask values
- total `152`

### 6. Run Robust Offline Inference

Example:

El modelo de referencia actual es el run MAIN (entrenado con el conjunto completo de 2 264 594 filas, 3 000 000 pasos, completado el 2026-06-10):

```bash
python scripts/predict_real_traffic_v2.py \
  --flows pcaps/synthetic_real_traffic.csv \
  --model models/MAIN_qrdqn_cicids2017_canonical_full_random_20260609_193655.zip \
  --scaler runs/cicids2017/MAIN_qrdqn_cicids2017_canonical_full_random_20260609_193655/scaler.joblib \
  --percentiles runs/cicids2017/MAIN_qrdqn_cicids2017_canonical_full_random_20260609_193655/train_percentiles.npz \
  --clip-z 10.0 \
  --export-diagnostics
```

(El run `P2v2_pred_20260610_161231` ya utilizó este modelo.)

### 7. Store Run Artifacts

Every Phase 2 run should write to:

```text
runs/phase2/<RUN_ID>/
├── config.json
├── metrics.json
├── predictions.csv.gz          # comprimido; head en predictions_head_10000.csv
├── predictions_head_10000.csv  # primeras 10 000 filas sin comprimir
└── diagnostics.json   # opcional
```

Nota: runs anteriores a P2v2_pred_20260610_161231 contienen `predictions.csv` sin comprimir.

### 8. Review the Results

Review:

- block rate
- allow rate
- z-score diagnostics
- suspicious features with large z-scores
- consistency across benign-only, attack-only, and mixed captures

## What Good Phase 2 Evidence Looks Like

- inference runs complete without errors
- artifact folders are reproducible and self-describing
- metrics are tied to specific run IDs
- benign traffic and attack traffic are evaluated separately
- diagnostics explain extreme behaviour instead of hiding it

## Known Risks

| Risk | Why it matters | Current mitigation |
|------|----------------|--------------------|
| Domain shift | CICIDS2017 and lab traffic differ | scaler persistence, percentile clipping, z clipping, diagnostics |
| Feature mismatch | extractor naming may differ | explicit flowmeter-to-canonical mapping |
| Overclaiming results | Phase 2 behaviour changes across runs | tie every claim to a concrete run artifact |

## Non-Goals for the Current Baseline

- active inline blocking
- adversarial attacker training
- production deployment
- schema redesign

## Next Useful Milestones

1. Re-run benign, scan, and mixed traffic with the current v2 defaults and compare artifacts.
2. Decide whether lab-specific calibration is needed.
3. Only after stable offline evidence, consider a controlled active-blocking prototype.
