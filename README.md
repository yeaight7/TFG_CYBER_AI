# TFG_CYBER_AI

RL-based cybersecurity defender for binary `PERMIT` / `BLOCK` decisions on network flows.

The project is organised in two phases:

- **Phase 1**: offline training and validation on historical datasets.
- **Phase 2**: offline inference on flow features extracted from traffic captured in a private lab.

The current repository uses **CICIDS2017** as the main dataset, a **fixed canonical schema** of 76 flow features, and a **152-dimensional observation vector** once the missingness mask is appended.

## Current Status

| Area | Status |
|------|--------|
| Canonical schema | Implemented and frozen at 76 features |
| CICIDS2017 adapter | Implemented |
| NSL-KDD adapter | Implemented for historical Phase 1 benchmarking |
| RL algorithm | QRDQN |
| Validation suite | Checks A, B, C + leave-one-exact-CSV-out script |
| Phase 2 inference | Robust offline pipeline available (`predict_real_traffic_v2.py`) |
| Active blocking | Not implemented |

## Documentation Map

- [docs/README.md](docs/README.md): documentation index and document roles
- [.github/AGENT_CONTEXT.md](.github/AGENT_CONTEXT.md): project-wide technical source of truth
- [docs/results.md](docs/results.md): artifact-backed results snapshot
- [docs/AGENT_CONTEXT.md](docs/AGENT_CONTEXT.md): Phase 2 scope and guardrails
- [docs/phase2_plan.md](docs/phase2_plan.md): execution plan for the lab workflow
- [docs/gcp_lab.md](docs/gcp_lab.md): private lab deployment guide
- [experiments/README.md](experiments/README.md): experiment archive index
- [experiments/cicids2017_qrdqn_experiments.md](experiments/cicids2017_qrdqn_experiments.md): maintained CICIDS2017 + QRDQN run history
- [docs/DEFENSA_TFG_PROGRESO.md](docs/DEFENSA_TFG_PROGRESO.md): Spanish defense notes
- [docs/DEFENSA_TFG_SCRIPT.md](docs/DEFENSA_TFG_SCRIPT.md): Spanish defense script
- [report/report.tex](report/report.tex): thesis report source draft

## Repository Structure

```text
TFG_CYBER_AI/
├── .github/                 # Agent guidance and project-wide source of truth
├── datasets/                # Local datasets (not tracked in git)
├── docs/                    # Documentation, results, Phase 2 guides, defense material
├── experiments/             # Experiment archive notes: historical and maintained timelines
├── lab/                     # Lab-related assets
├── models/                  # Trained model files
├── pcaps/                   # Extracted flows and captures used for Phase 2 work
├── report/                  # Thesis report sources
├── runs/                    # Run artifacts: config.json, metrics.json, validation_results.json, etc.
├── scripts/                 # Phase 2 and utility scripts
└── src/                     # Training, validation, adapters, environment, utilities
```

## Core Technical Invariants

- `FEATURES_CANON` contains **76 flow-based features**.
- The observation vector is always **152 dimensions**:
  - 76 canonical feature values
  - 76 missingness-mask values
- The missingness mask uses:
  - `1` for present/valid features
  - `0` for imputed or unavailable features
- Labels are binary:
  - `0 = BENIGN`
  - `1 = ATTACK`
- Leakage-prone fields must not enter the model:
  - IP addresses
  - absolute timestamps
  - Flow IDs or unique identifiers
  - ports used directly as label proxies

## Quickstart

Install dependencies:

```bash
pip install -r requirements.txt
```

Train the RL model on CICIDS2017:

```bash
python src/train_rl_defender.py --smoke
python src/train_rl_defender.py --preset full
python src/train_rl_defender.py --split-mode day
```

Run the validation suite:

```bash
python src/validate_checks.py --model models/<MODEL>.zip --checks A B C
```

Run leave-one-exact-CSV-out validation:

```bash
python src/validate_leave_one_csv_out.py --timesteps 30000
python src/validate_leave_one_csv_out.py --timesteps 5000 --max-rows-per-csv 10000
```

Run robust Phase 2 offline inference:

```bash
python scripts/predict_real_traffic_v2.py \
  --flows pcaps/flows.csv \
  --model models/C03_qrdqn_cicids2017_canonical_full_random_20260223_232439.zip \
  --scaler runs/cicids2017/C03_qrdqn_cicids2017_canonical_full_random_20260223_232439/scaler.joblib \
  --percentiles runs/cicids2017/C03_qrdqn_cicids2017_canonical_full_random_20260223_232439/train_percentiles.npz \
  --clip-z 10.0 \
  --export-diagnostics
```

## Validation Overview

The repository currently includes four validation workflows:

| Validation | Purpose |
|------------|---------|
| Check A | Direct prediction on `X_test` vs `y_test` without relying on the environment |
| Check B | Shuffled-label anti-leakage test |
| Check C | Hard CSV/day split generalisation test |
| Leave-one-exact-CSV-out | One held-out CICIDS2017 CSV per fold, train on the remaining seven |

The leave-one-exact-CSV-out workflow is implemented in code, but this repository does not currently contain a committed full run artifact for it under `runs/validation/`.

## Results Snapshot

Artifact-backed historical results are summarised in [docs/results.md](docs/results.md). Highlights:

- Best committed CICIDS2017 run:
  - `C03_qrdqn_cicids2017_canonical_full_random_20260223_232439`
  - accuracy `0.99859`
  - attack recall `0.99945`
  - attack F1 `0.99876`
- Validation Check C historical artifact:
  - accuracy `0.84135`
  - train on Monday–Wednesday patterns
  - test on Thursday–Friday patterns
- Phase 2:
  - robust offline inference pipeline exists
  - latest committed benign-only v2 artifact shows that behaviour changed over time, so Phase 2 claims must always be tied to the exact run artifact

The longer experiment-by-experiment narrative now lives in [experiments/cicids2017_qrdqn_experiments.md](experiments/cicids2017_qrdqn_experiments.md) for CICIDS2017 and [experiments/nslkdd_experiments.md](experiments/nslkdd_experiments.md) for the older NSL-KDD branch.

## Notes for Submission and Defense

- English is the default language for repository documentation.
- The two defense-support documents remain in Spanish by design:
  - [docs/DEFENSA_TFG_PROGRESO.md](docs/DEFENSA_TFG_PROGRESO.md)
  - [docs/DEFENSA_TFG_SCRIPT.md](docs/DEFENSA_TFG_SCRIPT.md)
- Historical results are preserved, but they must not be confused with the **current code defaults**.

## Safety and Reproducibility

- Do not commit datasets, PCAPs, credentials, or large generated artifacts.
- Every training or evaluation workflow should persist a `RUN_ID` and write artifacts under `runs/<category>/<RUN_ID>/`.
- If documentation describes a result, it should reference an artifact that exists in `runs/` or be clearly marked as planned or historical.
