# Agent Context — Phase 2 Scope

This document provides additional context for coding agents and contributors working on Phase 2 of the TFG. It complements `.github/AGENT_CONTEXT.md` (project-wide source of truth) with Phase 2-specific scope, non-goals, guardrails, and run-tracking conventions.

---

## Phase 2 Scope

Phase 2 transitions the RL defender from offline dataset evaluation to a **simulated lab environment** with real (generated) traffic. Detailed steps are in [`docs/phase2_plan.md`](phase2_plan.md).

### In Scope

- Set up a private lab (GCP or local VMs) with attacker + defender topology.
- Generate labelled traffic (benign + attacks) with ground-truth logs.
- Capture PCAPs and extract flow features with CICFlowMeter / Zeek.
- Map extracted features to the **canonical schema** (76 features + 76 mask = 152 dims).
- Run inference with the trained QRDQN model and evaluate against ground-truth.
- Save all results following `runs/phase2/<RUN_ID>/` convention.

### Non-Goals (Phase 2)

The following are explicitly **out of scope** for Phase 2:

- **Real-time packet-level blocking** — Phase 2 is inference-only; active `iptables` integration is optional.
- **Adversarial RL agent** — training an attacker agent is reserved for Phase 3.
- **Multi-agent or distributed defence** — single-agent only.
- **Re-training in the lab** — use the model trained on CICIDS2017; fine-tuning is optional.
- **Production deployment** — no containerization, CI/CD, or monitoring stack.
- **Changing the canonical schema** — the 76-feature schema is frozen for Phase 2.
- **Changing model hyperparameters** — use the existing QRDQN configuration.

---

## Guardrails

### Lab Safety

| Rule | Detail |
|------|--------|
| Private VPC only | No public IPs except a single SSH bastion with allowlisted source IP |
| No external scanning | All attack traffic must stay inside the private network |
| Ephemeral VMs | Destroy lab VMs when not in use to avoid costs and exposure |
| No credentials in code | Use environment variables or GCP metadata for secrets |

### Code & Data Safety

| Rule | Detail |
|------|--------|
| Never commit `venv/` | Already in `.gitignore`; double-check before pushing |
| Never commit datasets | Large CSVs / PCAPs stay in `datasets/` (gitignored) or cloud storage |
| Never commit model weights > 50 MB | Use Git LFS or external storage if needed |
| No absolute paths in code | Use `pathlib.Path` relative to repo root |
| No data leakage features | IPs, timestamps, Flow IDs are prohibited in the observation vector |

---

## Run-Tracking Conventions

Every experiment or evaluation **must** produce a run folder:

```
runs/<category>/<RUN_ID>/
├── config.json            # full configuration (hyperparams, dataset, seed, device, …)
├── metrics.json           # final evaluation metrics
├── validation_results.json  # (if validation check)
└── …                       # TensorBoard logs, predictions, etc.
```

### RUN_ID Format

```
<PREFIX>_<descriptor>_<YYYYMMDD_HHMMSS>
```

Examples:
- `C01_qrdqn_cicids2017_canonical_full_20260212_200218`
- `VAL_checks_A_20260213_085434`
- `P2_lab_eval_20260301_143000`

### Categories

| Prefix | Directory | Purpose |
|--------|-----------|---------|
| `C*` | `runs/cicids2017/` | CICIDS2017 training runs |
| `E*` | `runs/nslkdd/` | NSL-KDD Phase 1 experiments |
| `VAL_*` | `runs/validation/` | Validation checks (A, B, C) |
| `P2_*` | `runs/phase2/` | Phase 2 lab evaluation runs |
| `study_*` | `runs/optuna/` | Hyperparameter tuning studies |

### JSON Schema (minimum fields)

**config.json**:
```json
{
  "run_id": "...",
  "algorithm": "QRDQN",
  "dataset": "CICIDS2017",
  "use_canonical": true,
  "seed": 42,
  "device": "cuda",
  "reward_config": { "tp": 1.5, "fp": -1.0, "fn": -5.0, "omission": 0.0 }
}
```

**metrics.json**:
```json
{
  "accuracy": 0.0,
  "precision_attack": 0.0,
  "recall_attack": 0.0,
  "f1_attack": 0.0
}
```

---

## Current Best Model

| Field | Value |
|-------|-------|
| RUN_ID | `C01_qrdqn_cicids2017_canonical_full_20260212_200218` |
| Accuracy | 0.9962 |
| F1 (attack) | 0.9963 |
| Recall (attack) | 0.9998 |
| Model file | `models/C01_qrdqn_cicids2017_canonical_full_20260212_200218.zip` |

Validation checks A and B confirmed: no leakage, metrics reproducible via direct prediction.

---

## Key References

- [`.github/AGENT_CONTEXT.md`](../.github/AGENT_CONTEXT.md) — project-wide source of truth
- [`docs/phase2_plan.md`](phase2_plan.md) — step-by-step Phase 2 execution plan
- [`docs/gcp_lab.md`](gcp_lab.md) — lab setup instructions
- [`docs/results.md`](results.md) — consolidated experiment results
- [`src/canonical_schema.py`](../src/canonical_schema.py) — canonical feature definitions
