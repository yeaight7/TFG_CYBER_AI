# AGENTS.md — TFG_CYBER_AI instructions for Codex

## Read This Before Making Changes

Use this order when you need project context:

1. [.github/AGENT_CONTEXT.md](.github/AGENT_CONTEXT.md) — project-wide technical source of truth
2. [docs/README.md](docs/README.md) — documentation map
3. [docs/AGENT_CONTEXT.md](docs/AGENT_CONTEXT.md) — Phase 2 scope and guardrails
4. [docs/results.md](docs/results.md) — artifact-backed results snapshot
5. Code: `src/canonical_schema.py`, `src/load_cicids2017.py`, `src/train_rl_defender.py`, `src/validate_checks.py`, `src/validate_leave_one_csv_out.py`, `scripts/predict_real_traffic_v2.py`

If documentation and code disagree, prefer the **current code plus run artifacts**, then update the documentation accordingly.

## Project Invariants

- `FEATURES_CANON` contains 76 canonical flow features.
- Final observation size is always 152:
  - 76 canonical values
  - 76 missingness-mask values
- Missingness-mask semantics:
  - `1 = present / valid`
  - `0 = missing / imputed`
- Dataset adapters must return:
  - `(X_train, y_train, X_test, y_test, scaler, feature_names)`
- Data types:
  - `X = float32`
  - `y = int64`
- Labels:
  - `0 = BENIGN`
  - `1 = ATTACK`

## Anti-Leakage Rules

Do not introduce any of the following as model features:

- IP addresses
- absolute timestamps
- Flow IDs or unique identifiers
- ports used directly as label proxies

If a new dataset is added, leakage-prone fields must be removed before canonical mapping.

## Training and Validation Rules

- Prefer leaving heavy training to the user unless the task explicitly requires running it.
- CICIDS2017 training entry point:
  - `src/train_rl_defender.py`
- Validation entry points:
  - `src/validate_checks.py`
  - `src/validate_leave_one_csv_out.py`
- Phase 2 offline inference entry point:
  - `scripts/predict_real_traffic_v2.py`

If you cannot run heavy training locally, do not fabricate results. Limit yourself to static checks, shape/invariant validation, or reproducible commands for the user.

## Documentation Rules

- English is the default language for repo documentation.
- Exception:
  - `docs/DEFENSA_*` stays in Spanish.
- Do not edit `docs/discusion_con_llm.md` if it appears in the future; treat it as a historical log.
- When a documented claim is historical, label it clearly as historical.
- When a documented claim reflects the current implementation, it must match the current codebase.

## Reproducibility Expectations

Every meaningful training or evaluation run should persist:

- `config.json`
- `metrics.json` or `validation_results.json`
- the exact `RUN_ID`

If preprocessing, clipping, scaling, reward values, or split logic changes, document the change and its expected impact.

## graphify

This project has a graphify knowledge graph at graphify-out/.

Rules:
- Before answering architecture or codebase questions, read graphify-out/GRAPH_REPORT.md for god nodes and community structure
- If graphify-out/wiki/index.md exists, navigate it instead of reading raw files
- After modifying code files in this session, run `python3 -c "from graphify.watch import _rebuild_code; from pathlib import Path; _rebuild_code(Path('.'))"` to keep the graph current
