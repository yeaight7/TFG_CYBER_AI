# GitHub Copilot Instructions — TFG_CYBER_AI

This file contains repository-level guidance for GitHub Copilot and related coding agents.

## Project Summary

`TFG_CYBER_AI` is a thesis project focused on an RL-based defender that classifies network flows and decides whether to permit or block them.

The repository currently centres on:

- CICIDS2017 as the main modern dataset
- a 76-feature canonical schema
- a 152-dimensional observation vector after adding the missingness mask
- QRDQN as the main RL algorithm

## Read First

Before making changes, read:

1. [.github/AGENT_CONTEXT.md](AGENT_CONTEXT.md)
2. [../AGENTS.md](../AGENTS.md)

## Repository Conventions

### Code Style

- Python 3.10+
- Type hints for non-trivial functions
- `snake_case` for functions and variables
- `PascalCase` for classes
- `UPPER_SNAKE_CASE` for constants
- prefer `pathlib.Path` over string paths

### Reproducibility

- Use seeded workflows where possible (`42` by default in current scripts)
- Keep run artifacts under `runs/<category>/<RUN_ID>/`
- Training and evaluation documentation should cite the exact run artifact when describing measured results

### Scripts and Entry Points

| Purpose | File |
|---------|------|
| Train CICIDS2017 QRDQN | `src/train_rl_defender.py` |
| Validation checks A/B/C | `src/validate_checks.py` |
| Leave-one-exact-CSV-out validation | `src/validate_leave_one_csv_out.py` |
| Robust Phase 2 inference | `scripts/predict_real_traffic_v2.py` |

## Design Rules

### Canonical Schema

- The canonical schema is fixed.
- Do not train models with variable-length or semantically inconsistent feature vectors.
- Missing canonical features must be handled through imputation plus the missingness mask.

### Anti-Leakage

Never introduce:

- IP addresses
- absolute timestamps
- Flow IDs
- port fields as direct label proxies

### Multi-Dataset Support

Each dataset should have its own adapter and map into the shared canonical schema.

## Documentation Rules

- English is the default for repository documentation.
- `docs/DEFENSA_*` stays in Spanish.
- Historical run-specific settings must not be presented as current defaults.
- Avoid stale references to scripts or flags that no longer exist.

## What Not to Do

- Do not modify `docs/discusion_con_llm.md` if it appears later.
- Do not hardcode absolute local paths.
- Do not delete historical runs without explicit reason.
- Do not claim success for validations you did not actually run.
