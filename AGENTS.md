# AGENTS.md — TFG_CYBER_AI instructions for coding agents

Use this order when you need project context:

1. [.github/AGENT_CONTEXT.md](.github/AGENT_CONTEXT.md) — project-wide technical source of truth (invariants, anti-leakage rules, entry points, reproducibility rules)
2. [docs/README.md](docs/README.md) — documentation map and language policy
3. [docs/AGENT_CONTEXT.md](docs/AGENT_CONTEXT.md) — Phase 2 scope and guardrails
4. [docs/results.md](docs/results.md) — artifact-backed results snapshot

Rules:

- If documentation and code disagree, prefer the **current code plus run artifacts**, then update the documentation accordingly.
- Do not duplicate the invariants here — they live in `.github/AGENT_CONTEXT.md`.
- Leave heavy training to the user unless the task explicitly requires running it. If you cannot run heavy training, do not fabricate results — limit yourself to static checks, shape/invariant validation, or reproducible commands for the user.
- Every meaningful training or evaluation run must persist `config.json`, `metrics.json` (or `validation_results.json`), and its exact `RUN_ID` under `runs/`.
