# TFG_CYBER_AI

## Project Overview

**TFG_CYBER_AI** is a reinforcement learning (RL) based cybersecurity defender prototype for making binary `PERMIT` / `BLOCK` decisions on network flows. 
The project is built using Python and leverages libraries such as PyTorch, Stable Baselines3, sb3-contrib (for QRDQN), Gymnasium, scikit-learn, and pandas.

The architecture revolves around two main phases:
- **Phase 1:** Offline training and validation on historical datasets (primarily CICIDS2017). NSL-KDD was used historically but CICIDS2017 is the current baseline.
- **Phase 2:** Offline inference on flow features extracted from real traffic captured in a private lab.

**Core Technical Invariants:**
- **Features:** A fixed canonical schema (`FEATURES_CANON`) containing 76 flow-based features.
- **Observation Space:** A 152-dimensional observation vector (76 canonical feature values + 76 missingness-mask values).
- **Missingness Mask:** `1` represents a present/valid feature; `0` represents an imputed or unavailable feature.
- **Labels:** Binary classification where `0 = BENIGN` and `1 = ATTACK`.
- **Anti-leakage Policy:** Leakage-prone fields (e.g., IP addresses, absolute timestamps, Flow IDs, ports used as proxies) are strictly excluded. If a new dataset is added, leakage-prone fields must be removed *before* canonical mapping.

## Building and Running

**Dependencies:**
The project dependencies are managed via `requirements.txt` and `pyproject.toml`.
```bash
pip install -r requirements.txt
# Alternatively, if using a virtual environment and standard packaging:
# pip install -e .[dev]
```

**Training:**
Train the RL model (QRDQN) on the CICIDS2017 dataset. Do not fabricate results; leave heavy training to the user unless explicitly required.
```bash
# Smoke test (quick run)
python src/train_rl_defender.py --smoke

# Full training preset
python src/train_rl_defender.py --preset full

# Day split mode
python src/train_rl_defender.py --split-mode day
```

**Validation & Testing:**
The project relies on a Multi-Stage Validation Ladder to ensure generalization and guard against data leakage.
```bash
# Run validation checks A, B, and C
python src/validate_checks.py --model models/<MODEL>.zip --checks A B C

# Run leave-one-exact-CSV-out validation
python src/validate_leave_one_csv_out.py --timesteps 30000
```

**Inference (Phase 2):**
Run offline inference on captured PCAP/flow data using a trained model.
```bash
python scripts/predict_real_traffic_v2.py \
  --flows pcaps/lab_capture_traffic.csv \
  --model models/MAIN_qrdqn_cicids2017_canonical_full_random_20260609_193655.zip \
  --scaler runs/cicids2017/MAIN_qrdqn_cicids2017_canonical_full_random_20260609_193655/scaler.joblib \
  --percentiles runs/cicids2017/MAIN_qrdqn_cicids2017_canonical_full_random_20260609_193655/train_percentiles.npz \
  --clip-z 10.0 \
  --export-diagnostics
```

## Agent Workflows and Context Hierarchy

When investigating issues or gathering context, consult sources in this strict order:
1. `.github/AGENT_CONTEXT.md` (project-wide technical source of truth)
2. `docs/README.md` (documentation map)
3. `docs/AGENT_CONTEXT.md` (Phase 2 scope and guardrails)
4. `docs/results.md` (artifact-backed results snapshot)
5. **Code:** `src/canonical_schema.py`, `src/load_cicids2017.py`, `src/train_rl_defender.py`, `src/validate_checks.py`, `src/validate_leave_one_csv_out.py`, `scripts/predict_real_traffic_v2.py`.

*Note: If documentation and code disagree, prefer the **current code plus run artifacts**, and update the documentation accordingly.*

## Knowledge Graph (`graphify`)

The project maintains a local knowledge graph (`graphify-out/`). It is **untracked / gitignored** — a generated artifact, so a fresh clone has none and an existing one may be stale; regenerate locally with `graphify .`. When present, use it to accelerate repo orientation:
- **Start multi-file tasks** by reading `graphify-out/GRAPH_REPORT.md`.
- **Query the graph** directly (`graphify query "<question>"`, `graphify path "<node A>" "<node B>"`) instead of blindly reading raw files.
- **Hypotheses vs Facts:** Treat `INFERRED` edges (e.g., `semantically_similar_to`) as hypotheses to be verified in code.
- **Key Navigation Hubs:** `Canonical Flow Schema`, `CICIDSLoadConfig`, `RLDatasetDefenderEnv`, `Phase 2 Offline Inference`, `Robust v2 Inference Pipeline`.
- If structural code changes are made, git hooks will update the graph. For broad semantic or documentation changes, manually run `graphify .`.
- Small edits such as comments, docstrings, formatting, reward-value tweaks, and run artifacts under `runs/` do not trigger an automatic rebuild.
- If `graphify-out/needs_update` exists, semantic sources changed and the graph may be stale. Run `graphify .` for a full refresh before relying on the graph for architecture, documentation, or review work.
- Keep `docs/Personal Research/` and `.github/skills/` outside the maintained Graphify corpus to avoid mixing personal notes and skill metadata with project architecture signals.
- The Obsidian export at `graphify-out/obsidian/` (note-first or canvas view) is not generated by default; consult the graphify documentation for how to produce it if needed.

## Development Conventions

- **Artifact Persistence:** Every training or evaluation workflow MUST persist a `RUN_ID` and write all artifacts (`config.json`, `metrics.json`, etc.) to `runs/<category>/<RUN_ID>/`.
- **Reproducibility:** Documentation claiming results must reference an exact artifact located in the `runs/` directory. If preprocessing, clipping, scaling, or split logic changes, document the expected impact.
- **Language:** English is the default language for codebase documentation and comments. `docs/DEFENSA_*` files remain in Spanish.
- **Data Policy:** The project differentiates between "Curated" (leakage-prone columns removed) and "Raw" data. The adapter (`src/load_cicids2017.py`) acts as the authoritative gate enforcing the anti-leakage policy at load time.