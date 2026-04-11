# AGENTS.md — TFG_CYBER_AI (instructions for Codex)

## Project context (read before touching code)
This repository implements an RL-based cybersecurity defender agent that decides PERMIT(0)/BLOCK(1) on network traffic represented as flow-based features.

**Sources of truth (recommended order):**
1) `.github/AGENT_CONTEXT.md` — objective, phases, critical decisions (especially the canonical schema). This file was made for the Github Agent, but it could be useful.
2) `docs/AGENT_CONTEXT.md` — scope and guardrails for Phase 2 (lab / real traffic). [Current implementation]
3) `docs/results.md` — consolidated results and open issues (domain shift, Phase 2 runs).
4) Code: `src/canonical_schema.py`, `src/load_cicids2017.py`, `src/train_rl_defender.py`, `scripts/predict_real_traffic_v2.py`.

If anything contradicts your “general intuition” about ML/RL, prompt/ask the user preferably. 
There's a possibility some of these files (`.github/AGENT_CONTEXT.md` or `docs/results.md`) are obsolete or misdocumented.

## Code exploration rules
- Before proposing changes, locate the single source of truth:
  - Feature schema: `src/canonical_schema.py`
  - CICIDS2017 loading/preprocessing: `src/load_cicids2017.py`
  - RL environment: `src/rl_defender_env.py`
  - QRDQN training: `src/train_rl_defender.py`
  - Validation: `src/validate_checks.py`
  - Phase 2 inference: `scripts/predict_real_traffic_v2.py`
- Keep imports and paths reproducible. Avoid hardcoded absolute paths; use `Path(repo_root)/...` whenever possible.

## Project invariants (DO NOT break)
### Fixed canonical schema
- `FEATURES_CANON` contains 76 flow-based features.
- Final observation is ALWAYS 152 dims: `[x_1..x_76, m_1..m_76]`.
- The mask `m_i` is 1 if the feature is present/valid and 0 if it is imputed.

### Anti-leakage (critical)
Do not introduce the following as features:
- IPs, absolute timestamps, Flow IDs, or unique identifiers.
- Ports as a direct feature if they act as a label proxy.

If you add/adapt datasets, apply these rules before mapping to the canonical schema.

### Multi-dataset through adapters
Each dataset must have its own loader/adapter and return a unified format:
`(X_train, y_train, X_test, y_test, scaler, feature_names)` with `X=float32`, `y=int64`, labels `0=BENIGN 1=ATTACK`.

## Experiments and reproducibility
- Every experiment/evaluation must create a unique `RUN_ID` and save artifacts under `runs/<category>/<RUN_ID>/`.
- Save at minimum:
  - `config.json` with hyperparameters, seed, split, reward_config, and shapes.
  - `metrics.json` with accuracy/precision/recall/F1 (at least for attack).
- If you change preprocessing (scaling/clipping/mapping), document the assumption and the expected impact.
- Do not delete old runs without an explicit reason.

## How to handle changes in training, datasets, and evaluation
- Training (CICIDS2017):
  - Use `src/train_rl_defender.py` and its flags (`--preset`, `--split-mode`, `--max-rows`, `--seed`).
  - If you change reward_config/hyperparameters, make sure `config.json` reflects what was actually run.
- Validation:
  - After relevant changes, run `src/validate_checks.py` (A/B/C depending on cost).
  - If Check B (shuffled labels) does not drop to ~baseline: suspect leakage/artifacts and stop.
- Phase 2 (real traffic):
  - Use `scripts/predict_real_traffic_v2.py` (not v1).
  - Preserve: unit harmonization (seconds vs microseconds), clipping, and z-diagnostics.
  - When there is an extreme block-rate on real benign traffic, prioritize diagnosis (z-scores, top features) before “touching” the model.

## Documentation of assumptions
Before closing a change, leave a record (in `docs/` or in the PR/issue) of:
- Which assumption you changed (e.g. scaling, clipping, mapping, split).
- What evidence you used (logs/metrics/run_id).
- What you could not verify (e.g. dataset unavailable in your environment).

## Avoid dangerous or poorly verifiable changes
- DO NOT introduce credentials (Kaggle/GCP/API keys) into the repo.
- DO NOT modify the lab to expose the Internet or public ports; the lab must remain private/isolated.
- DO NOT add large datasets/PCAPs to the repository (they must stay out of git).
- If you cannot run CICIDS2017 in your environment, do not invent results: limit the change to verifiable refactoring and suggest how to validate it locally.

## If context is missing
If you are missing any of these elements for validation:
- CICIDS2017 dataset available,
- Phase 2 PCAP/flows.csv,
- GPU access,
then:
1) run synthetic tests or static checks (shapes, invariants, basic lint),
2) clearly document the limitation,
3) propose a reproducible command so the user can validate it on their machine.

## Running and training the model
Try to leave training to the user, since training process is very resource heavy. When in doubt, prompt the user for clarification.