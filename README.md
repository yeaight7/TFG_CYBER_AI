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
| NSL-KDD adapter | Legacy — data/model dropped from the repo (see [experiments/nslkdd_experiments.md](experiments/nslkdd_experiments.md)); code kept for history only |
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
- [docs/runpod_main_experiment.md](docs/runpod_main_experiment.md): single main RunPod training run guide
- [docs/reproducibility.md](docs/reproducibility.md): recorded environment for the main QRDQN run and dependency-file strategy
- [experiments/README.md](experiments/README.md): experiment archive index
- [experiments/cicids2017_qrdqn_experiments.md](experiments/cicids2017_qrdqn_experiments.md): maintained CICIDS2017 + QRDQN run history
- [docs/DEFENSA_TFG_PROGRESO.md](docs/DEFENSA_TFG_PROGRESO.md): Spanish defense notes
- [docs/DEFENSA_TFG_SCRIPT.md](docs/DEFENSA_TFG_SCRIPT.md): Spanish defense script
- [memoria/memoria.tex](memoria/memoria.tex): **canonical thesis (Spanish)** — official source
- [report/report.tex](report/report.tex): English thesis draft — parked (may lag new sections)
- [docs/audits/](docs/audits/): dated read-only repository audits

## Repository Structure

```text
TFG_CYBER_AI/
├── .codex/                    # (empty — reserved for knowledge graph hooks; hooks.json not yet populated)
├── .github/                   # Agent guidance and coding/review agent instructions
├── datasets/                  # Local datasets (also tracked via git lfs)
├── docs/                      # Documentation, results, Phase 2 guides, defense material
|   ├── Personal Research/     # Deep-research notes (Markdown only; research, not a source of truth)
|   ├── audits/                # Dated read-only repository audits
|   └── archive/               # Obsolete docs kept for history (e.g. informe.* draft)
├── experiments/               # Experiment archive notes: historical and maintained timelines
├── lab/                       # Lab-related assets
├── memoria/                   # Canonical thesis (Spanish) — official source
├── models/                    # Trained model files (tracked)
├── pcaps/                     # Extracted flows and captures used for Phase 2 work (tracked)
├── report/                    # English thesis draft — parked (may lag new sections)
├── runs/                      # Run artifacts: config.json, metrics.json, validation_results.json, etc. (tracked)
├── scripts/                   # Phase 2 and utility scripts
└── src/                       # Training, validation, adapters, environment, utilities
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

## Dataset Versions (CICIDS2017)

Two versions of the CICIDS2017 data exist locally:

| Version | Path | Tracked | Description |
|---------|------|---------|-------------|
| Curated | `datasets/CICIDS2017/*.csv` | Yes | Leakage-prone and redundant columns removed pre-ingestion. What the adapter loads. |
| Raw | `datasets/CICIDS2017/Raw_dataset/` | No (gitignored) | Original CICFlowMeter CSV exports. All columns preserved. Local reference only. |

The adapter (`src/load_cicids2017.py`) applies further cleaning at load time regardless of which version is used. The anti-leakage policy in code is the authoritative gate.

### Provenance and integrity

- **Upstream source**: CICIDS2017, Canadian Institute for Cybersecurity (CIC), University of New Brunswick — <https://www.unb.ca/cic/datasets/ids-2017.html> (the labelled CICFlowMeter flow CSVs, one per capture day).
- **Redistribution terms**: CICIDS2017 is distributed by UNB/CIC for research use and requires citation/attribution; redistribution is not explicitly granted. The curated copy here is a research convenience for reproducibility — prefer obtaining the data from the official link above, and cite CIC/UNB in any derived work. The legacy NSL-KDD dataset has been **removed** from this repository (`datasets/nsl_kdd/`, `models/rf_nslkdd.joblib` are no longer tracked).
- The hashes below are SHA-256 of the **curated** CSVs actually used by the adapter (`datasets/CICIDS2017/*.csv`), i.e. this repository's derivative after pre-ingestion column removal — **not** the upstream file hashes. They let you confirm you are working from the exact curated copy used for the committed results. The `Raw_dataset/` originals are gitignored and not hashed here.

| File | Bytes | SHA-256 |
|------|-------|---------|
| `Monday-WorkingHours.pcap_ISCX.csv` | 176,927,918 | `852c4beb34eda186f32561fa79df7a0747e92e1a6535b01270820dd9ffe17f34` |
| `Tuesday-WorkingHours.pcap_ISCX.csv` | 135,078,995 | `52b8692ae8c7d2ed04671fe2b98335693c0a92c7ab157d8c8b534d6523080851` |
| `Wednesday-workingHours.pcap_ISCX.csv` | 225,166,395 | `893c27dc968bf7a8adef1689f90be55ca4a4dc3088fb63d6ff247ac56856df2a` |
| `Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv` | 52,023,263 | `d67066211fb1689c78406f1506f4c44704ecb92088353d5c96d96d6474eb819d` |
| `Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv` | 83,102,436 | `6bcda3857c2504676034e3ea57762d38393cc734cb377a726bd5cb153961b1b5` |
| `Friday-WorkingHours-Morning.pcap_ISCX.csv` | 58,316,725 | `53a41c24d570ea83b7ac55b2e94df94e7a8216aeb80a2af0246b6bc8bb543000` |
| `Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv` | 76,906,168 | `ca1824c51bfbb7b3c72290a11be04366ba8815878c6a1cc5c44cb1cee269e99b` |
| `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv` | 77,123,859 | `6ff1580f5f81c0ae28a26f7631721018577f5f7c5e0feac28b795fcfe7b411ee` |

Verify locally with `sha256sum datasets/CICIDS2017/*.csv` (or `Get-FileHash -Algorithm SHA256`).

## Quickstart

Install dependencies:

```bash
pip install -r requirements.txt
```

For the pinned RunPod CUDA 13.0 environment used by the main QRDQN run, use:

```bash
pip install -r requirements-runpod-cu130.txt
```

See [docs/reproducibility.md](docs/reproducibility.md) for the exact recorded environment.

Train the RL model on CICIDS2017:

```bash
python src/train_rl_defender.py --smoke
python src/train_rl_defender.py --preset full
python src/train_rl_defender.py --split-mode day
```

Run the validation suite:

```bash
python src/validate_checks.py --run-dir runs/cicids2017/<RUN_ID> --checks A B C
```

Run leave-one-exact-CSV-out validation:

```bash
python src/validate_leave_one_csv_out.py --timesteps 30000
python src/validate_leave_one_csv_out.py --timesteps 5000 --max-rows-per-csv 10000
```

Run robust Phase 2 offline inference:

```bash
python scripts/predict_real_traffic_v2.py \
  --flows pcaps/lab_capture_traffic.csv \
  --run-dir runs/cicids2017/MAIN_qrdqn_cicids2017_canonical_full_random_20260609_193655 \
  --percentiles runs/cicids2017/MAIN_qrdqn_cicids2017_canonical_full_random_20260609_193655/train_percentiles.npz \
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

- Official run (MAIN) — full dataset, fixed test partition:
  - `MAIN_qrdqn_cicids2017_canonical_full_random_20260609_193655`
  - train_shape [2264594, 152], test_shape [566149, 152]
  - accuracy `0.9938055`
  - attack recall `0.9953555`
  - attack F1 `0.9844499`
  - Artifact: `runs/cicids2017/MAIN_qrdqn_cicids2017_canonical_full_random_20260609_193655/metrics.json`
- Best historical pre-design probe (not official, not directly comparable to MAIN's fixed test partition):
  - `C03_qrdqn_cicids2017_canonical_full_random_20260223_232439` (max_rows=500k; 100k-row test set with distorted class mix)
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

- Every training or evaluation workflow should persist a `RUN_ID` and write artifacts under `runs/<category>/<RUN_ID>/`.
- If documentation describes a result, it should reference an artifact that exists in `runs/` or be clearly marked as planned or historical.
