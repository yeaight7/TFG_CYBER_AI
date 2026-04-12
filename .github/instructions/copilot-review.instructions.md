---
applyTo: "**"
excludeAgent: "cloud-agent"
---

When performing a pull request review in this repository, prioritize bugs, silent regressions, data leakage risk, and missing validation over style nits.

Use `AGENTS.md`, `.github/AGENT_CONTEXT.md`, `docs/AGENT_CONTEXT.md`, and `docs/results.md` as the main review context when the diff touches architecture, datasets, validation, Phase 2 inference, or reported results.

Treat the current project baseline as:
- CICIDS2017 as the main modern dataset
- a fixed canonical schema of 76 flow features
- a 152-dimensional observation vector made of 76 values plus a 76-value missingness mask
- missingness-mask semantics of `1 = present / valid` and `0 = missing / imputed`
- `scripts/predict_real_traffic_v2.py` as the maintained Phase 2 offline inference entry point

Treat NSL-KDD as historical benchmark material unless the pull request explicitly works on historical experiments.

Flag any change that breaks or risks these invariants:
- dataset adapters must return `(X_train, y_train, X_test, y_test, scaler, feature_names)`
- `X` should remain `float32`
- `y` should remain `int64`
- labels remain `0 = BENIGN`, `1 = ATTACK`

Flag any attempt to use IP addresses, absolute timestamps, Flow IDs, unique identifiers, or direct port proxies as model features.

For changes to preprocessing, clipping, scaling, reward values, split logic, canonical mapping, evaluation methodology, or Phase 2 inference behavior, check that the pull request also updates documentation and reproducibility expectations where needed.

If a pull request claims improved metrics or changed behavior, ask for the exact `RUN_ID` and committed artifact path. Do not accept unverifiable training or evaluation claims.

Prefer comments about missing validation or missing artifact updates over comments about formatting. Heavy retraining is not required for every change, but targeted checks should exist when behavior changes.

Be skeptical of commits that add datasets, PCAPs, credentials, secrets, or large generated artifacts. Treat `graphify-out/cache` and temporary Graphify files as generated state, not source of truth.

If `graphify-out/` is present in the repository, use it as context for architectural understanding, but do not treat `INFERRED` graph edges as authoritative without checking code or maintained docs.

Avoid low-value style comments unless they hide maintainability or correctness risk.
