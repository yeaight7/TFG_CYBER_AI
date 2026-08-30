# Phase 2 Context (could be obsolete or misaligned with new project direction)

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

- retain practical reproducibility and traceability evidence under the repository's
  current artifact policy
- use normal Git for appropriate lightweight evidence and Git LFS for approved
  large/binary evidence covered by `.gitattributes`
- respect `.gitignore` for intentionally excluded heavyweight, rebuildable, or
  sensitive generated exports
- treat `.gitattributes`, `.gitignore`, and
  [reproducibility.md](../reproducibility.md#experimental-artifact-versioning) as
  the authoritative Git/LFS policy; do not add arbitrary large files outside it
- do not document unverifiable claims as facts

## Run Tracking

Every Phase 2 inference run should create:

```text
runs/phase2/<RUN_ID>/
├── config.json
├── metrics.json
├── predictions.csv                  # commit-safe prediction columns only
├── predictions_sensitive_local.csv  # optional local-only metadata export
└── diagnostics.json                 # optional, when exported
```

By default, `predictions.csv` must not include IPs, ports, timestamps, or label columns. Full metadata exports require `--include-sensitive-metadata` and are written to the ignored local-only file `predictions_sensitive_local.csv`.

## Maintained Entry Point

Use:

- `scripts/predict_real_traffic_v2.py`

and do not document `scripts/archive/deprecated_predict_real_traffic.py` (the legacy script, formerly `predict_real_traffic.py`; archived under `scripts/archive/` in Phase 5) as the recommended path except when discussing legacy behaviour.

## Current Open Risk

Phase 2 still shows sensitivity to distribution shift between CICIDS2017 and real lab traffic. When documenting or analysing Phase 2 performance, always tie the claim to the exact run artifact because behaviour changed across committed runs.
