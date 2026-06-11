# GitHub Copilot Instructions — TFG_CYBER_AI

This file contains the repository-level guidance for GitHub Copilot when editing or explaining code.

Review-specific guidance lives in `.github/instructions/copilot-review.instructions.md`.

## Read First

Before changing code or documentation, use this order:

1. `AGENTS.md`
2. `.github/AGENT_CONTEXT.md`
3. `docs/AGENT_CONTEXT.md`
4. `docs/results.md`

If `graphify-out/GRAPH_REPORT.md` exists and the task is architectural, cross-file, or review-oriented, read it first and use Graphify context before broad raw-file exploration.

## Current Baseline

- CICIDS2017 is the main modern dataset.
- NSL-KDD is historical benchmark material unless the task explicitly targets the historical branch.
- The canonical schema is fixed at 76 flow features.
- The final observation size is 152: 76 feature values plus a 76-value missingness mask.
- Missingness-mask semantics are `1 = present / valid` and `0 = missing / imputed`.
- The maintained Phase 2 inference entry point is `scripts/predict_real_traffic_v2.py`.

## Coding Expectations

- Prefer Python type hints for non-trivial functions.
- Prefer `pathlib.Path` over hardcoded path strings.
- Keep changes local, reversible, and aligned with current code plus committed run artifacts.
- Use the narrowest meaningful validation first and do not claim validation you did not run.

## Data and Modeling Guardrails

- Never introduce IP addresses, absolute timestamps, Flow IDs, unique identifiers, or direct port proxies as model features.
- Dataset adapters should preserve the shared return contract: `(X_train, y_train, X_test, y_test, scaler, feature_names)`. The higher-level `load_cicids2017_split` wrapper extends this to `(X_train, y_train, X_test, y_test, scaler, feature_names, metadata)` where `metadata` is a JSON-serialisable dict containing split info and SHA-256 hashes of the partitions.
- Preserve `X` as `float32`, `y` as `int64`, and labels as `0 = BENIGN`, `1 = ATTACK`.
- New datasets require an adapter that maps into the shared canonical schema.

## Documentation and Reproducibility

- English is the default documentation language; `docs/DEFENSA_*` stays in Spanish.
- Historical settings or results must be labeled clearly as historical.
- Meaningful training and evaluation changes should keep artifacts under `runs/<category>/<RUN_ID>/`.
- If behavior changes, keep documentation and artifact references aligned with the current implementation.

## Graphify

- If `graphify-out/` is present, use it for repo orientation and architectural context.
- Start with `graphify-out/GRAPH_REPORT.md`.
- Treat `INFERRED` graph edges as hints that must be checked in code or maintained docs before relying on them.
