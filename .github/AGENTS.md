# GitHub Coding Agent Notes

This file is a repository-specific guide for autonomous coding agents working through GitHub-oriented workflows.

## Mandatory Reading Order

Before changing code or documentation, read:

1. [.github/AGENT_CONTEXT.md](AGENT_CONTEXT.md)
2. [.github/copilot-instructions.md](copilot-instructions.md)
3. [../AGENTS.md](../AGENTS.md)

If those sources disagree with older narrative documents, prefer:

- current code
- committed run artifacts
- `.github/AGENT_CONTEXT.md`

## Required Technical Assumptions

- The canonical schema is fixed at 76 features.
- Final observation size is always 152 dimensions.
- Missingness-mask semantics are `1 = present`, `0 = missing`.
- Adapters must preserve the shared output contract:
  - `(X_train, y_train, X_test, y_test, scaler, feature_names)`
- Leakage-prone columns must be removed before training or inference.

## Dataset Rules

- CICIDS2017 is the main dataset for the modern training pipeline.
- NSL-KDD is historical benchmark material only.
- New datasets require new adapters and must map into the canonical schema.

## Validation Rules

When a change affects training, adapters, evaluation, or preprocessing, validate with the narrowest meaningful workflow first:

- direct static checks or shape checks
- `validate_checks.py` when feasible
- `validate_leave_one_csv_out.py` when the change impacts split logic or generalisation reporting

Do not invent training results if you cannot run the required environment locally.

## Documentation Rules

- English is the default documentation language.
- `docs/DEFENSA_*` remains in Spanish on purpose.
- Historical results must be labelled clearly as historical.
- If a documentation statement claims to reflect the current implementation, it must match current code.

## Things to Avoid

- Do not add credentials, keys, or large datasets to the repository.
- Do not expose the private lab publicly.
- Do not change the canonical schema casually.
- Do not treat old run-specific reward values as if they were still the current defaults.
