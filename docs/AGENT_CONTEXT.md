# Phase 2 Context

This document covers **Phase 2 only**: the transition from offline dataset work to offline inference on traffic captured in a private lab.

Project-wide architecture, invariants, datasets, and current capabilities live in [../.github/AGENT_CONTEXT.md](../.github/AGENT_CONTEXT.md).

## Scope

Phase 2 is currently scoped to:

- capturing traffic in a private lab
- extracting flow-level features
- mapping extracted flows to the canonical schema
- running offline inference with the trained RL model
- storing reproducible run artifacts under `runs/phase2/`

## Out of Scope

The following are not part of the current Phase 2 baseline:

- active packet or flow blocking in production
- public internet exposure of the lab
- adversarial multi-agent RL
- large-scale operational deployment
- schema redesign

## Operational Guardrails

### Lab Safety

- keep the lab private and isolated
- only allow SSH access from controlled sources
- do not scan targets outside the lab
- avoid committing credentials or environment secrets

### Data and Repo Safety

- do not commit datasets or PCAPs
- do not commit generated large artifacts by accident
- do not document unverifiable claims as facts

## Run Tracking

Every Phase 2 inference run should create:

```text
runs/phase2/<RUN_ID>/
├── config.json
├── metrics.json
├── predictions.csv
└── diagnostics.json   # optional, when exported
```

## Maintained Entry Point

Use:

- `scripts/predict_real_traffic_v2.py`

and do not document `predict_real_traffic.py` as the recommended path except when discussing legacy behaviour.

## Current Open Risk

Phase 2 still shows sensitivity to distribution shift between CICIDS2017 and real lab traffic. When documenting or analysing Phase 2 performance, always tie the claim to the exact run artifact because behaviour changed across committed runs.
