# Phase 2 Plan — Simulated Environment with Real Traffic

This document describes the step-by-step plan for **Phase 2** of the TFG, where the RL defender agent transitions from offline dataset evaluation to a simulated lab environment with generated network traffic.

---

## Overview

| Item | Detail |
|------|--------|
| **Goal** | Evaluate the trained QRDQN agent on live-captured traffic in a controlled lab |
| **Input** | PCAPs from a private virtual network (GCP or local VMs) |
| **Output** | Agent decisions (PERMIT / BLOCK) evaluated against ground-truth labels |
| **Primary dataset** | CICIDS2017 canonical schema (76 features + 76 missingness mask = 152 dims) |
| **Feature extractor** | CICFlowMeter (preferred) or Zeek |

---

## Step-by-Step Plan

### Step 1 — Set Up the Lab Environment

Deploy a minimal 2-VM private network (see [`docs/gcp_lab.md`](gcp_lab.md)):

| VM | Role | OS |
|----|------|----|
| **attacker** | Generate benign + malicious traffic | Kali Linux |
| **defender** | Run target services, capture traffic, run the agent | Ubuntu 22.04 |

Safety guardrails:
- Private VPC only, no external connectivity except SSH from a single IP.
- No scanning or exploitation of resources outside the lab.

### Step 2 — Generate Traffic

Produce labelled traffic with known ground-truth:

| Traffic Type | Tool / Method | Label |
|-------------|---------------|-------|
| Benign web | `curl`, `wget`, browser scripts | 0 (BENIGN) |
| Benign SSH | `ssh` commands, `scp` transfers | 0 |
| DDoS / DoS | `hping3`, `slowloris`, LOIC | 1 (ATTACK) |
| Port scan | `nmap -sS`, `nmap -sV` | 1 |
| Brute force | `hydra`, `medusa` | 1 |
| Web attacks | `sqlmap`, `nikto`, manual payloads | 1 |

Each traffic session is logged with start/end timestamps and a label file so that ground-truth can be joined to flows afterwards.

### Step 3 — Capture Traffic (PCAP)

On the **defender** VM:

```bash
sudo tcpdump -i eth0 -w /data/captures/session_YYYYMMDD_HHMMSS.pcap
```

Alternatively use `tshark` for live rotation:

```bash
tshark -i eth0 -b duration:300 -w /data/captures/session.pcap
```

### Step 4 — Extract Flow Features

Convert raw PCAPs to flow-level feature vectors using **CICFlowMeter** (or Zeek + custom post-processing):

```bash
# CICFlowMeter CLI (Java)
java -jar CICFlowMeter.jar -i /data/captures/ -o /data/flows/
```

Output: one CSV per PCAP with ~80 columns matching the CICIDS2017 schema.

### Step 5 — Map to Canonical Schema

Use the existing `canonical_schema.py` adapter to convert flow CSVs into the 152-dim observation vector:

```python
from canonical_schema import FEATURES_CANON, CICIDS2017_TO_CANON, map_to_canonical
import pandas as pd

df = pd.read_csv("/data/flows/session_flows.csv")
result = map_to_canonical(df, CICIDS2017_TO_CANON)
X = result.combined  # shape (n_flows, 152)
```

Apply the same `StandardScaler` that was fitted during training (saved alongside the model) to ensure consistent scaling.

### Step 6 — Inference Loop

Load the trained QRDQN model and predict on each flow:

```python
from sb3_contrib import QRDQN
import numpy as np

model = QRDQN.load("models/C01_qrdqn_cicids2017_canonical_full_20260212_200218.zip")

actions = []
for i in range(len(X)):
    action, _ = model.predict(X[i], deterministic=True)
    actions.append(int(action))  # 0 = PERMIT, 1 = BLOCK
```

### Step 7 — Evaluate Against Ground-Truth

Join agent decisions with ground-truth labels from the traffic generation log:

```python
from sklearn.metrics import classification_report, confusion_matrix

y_true = ground_truth_labels   # from step 2 log
y_pred = np.array(actions)

print(confusion_matrix(y_true, y_pred))
print(classification_report(y_true, y_pred, target_names=["BENIGN", "ATTACK"]))
```

Save results following the run-tracking convention:

```
runs/phase2/<RUN_ID>/
├── config.json
├── metrics.json
├── flows.csv           # extracted features
├── predictions.csv     # flow_id, y_true, y_pred
└── capture_metadata.json
```

### Step 8 — (Optional) Active Blocking

Once inference accuracy is validated, optionally integrate with `iptables` / `nftables` for real-time blocking:

```bash
# Example: block an IP that the agent flags
sudo iptables -A INPUT -s <attacker_ip> -j DROP
```

This step is **not required** for the TFG evaluation and should only be attempted after inference-only evaluation is complete.

---

## Success Criteria

| Criterion | Threshold |
|-----------|-----------|
| The agent runs inference without errors on lab-captured flows | Must pass |
| Accuracy on lab test set | ≥ 0.80 |
| Recall (attack) on lab test set | ≥ 0.70 |
| No data leakage (Check B equivalent on lab data) | Shuffled accuracy ≈ chance |
| Reproducible run with `RUN_ID`, `config.json`, `metrics.json` | Must pass |

---

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| CICFlowMeter output columns differ from CICIDS2017 version | Map columns via `CICIDS2017_TO_CANON`; test with a small capture first |
| Feature distribution shift (lab ≠ dataset) | Fine-tune model on a small labelled lab subset if needed |
| Missing features in extractor output | Handled by missingness mask (`m_i = 0`) |
| GPU not available in lab VM | Agent inference is CPU-friendly; training can stay on local GPU |

---

## Timeline (Estimated)

| Week | Milestone |
|------|-----------|
| 1 | Lab setup (VMs, VPC, services) |
| 2 | Traffic generation scripts + PCAP capture |
| 3 | Feature extraction pipeline + canonical mapping validation |
| 4 | Inference loop + evaluation + documentation |
| 5 | (Optional) Active blocking experiment |
