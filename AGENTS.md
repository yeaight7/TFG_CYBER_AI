# AGENTS.md — TFG_CYBER_AI instructions for coding agents

Use this order when you need project context:

1. [.github/AGENT_CONTEXT.md](.github/AGENT_CONTEXT.md) — project-wide technical source of truth (invariants, anti-leakage rules, entry points, reproducibility rules)
2. [docs/README.md](docs/README.md) — documentation map and language policy
3. [docs/phase 2/AGENT_CONTEXT.md](docs/phase%202/AGENT_CONTEXT.md) — Phase 2 scope and guardrails
4. [docs/results.md](docs/results.md) — artifact-backed results snapshot

Rules:

- If documentation and code disagree, prefer the **current code plus run artifacts**, then update the documentation accordingly.
- Do not duplicate the invariants here — they live in `.github/AGENT_CONTEXT.md`.
- Leave heavy training to the user unless the task explicitly requires running it. If you cannot run heavy training, do not fabricate results — limit yourself to static checks, shape/invariant validation, or reproducible commands for the user.
- Every meaningful training or evaluation run must persist `config.json`, `metrics.json` (or `validation_results.json`), and its exact `RUN_ID` under `runs/`.
## Subagent model and reasoning routing

Choose each delegated task's model and reasoning effort separately. Configure both when they should differ from the parent; omitted settings can inherit parent values. Use only models and effort levels available for the selected agent in the current runtime.

- **GPT-6 Luna**: efficient choice for clear, bounded, repeatable, or high-volume work, including focused inspection, extraction, well-specified edits, straightforward implementation, tests, narrow debugging, and bounded review. For difficult work that stays well scoped, consider increasing Luna's effort before changing model.
- **GPT-6.1 Terra**: efficient choice for bounded work that requires more context, reasoning, or synthesis than Luna can provide, including cross-file implementation, integration, debugging, correctness review, and synthesis across a bounded subsystem.
- **GPT-6.2 Sol**: normal choice for substantial supporting engineering: meaningful exploration, cross-file implementation, integration, debugging, correctness review, and synthesis across a bounded subsystem.
- **GPT-6.2 Astra**: reserve for difficult ambiguity, architecture, cross-system root causes, subtle security or data-integrity reasoning, complex business logic, conflicting findings, or high-stakes review.

Model tier and effort are separate choices, not a fixed ranking. Luna supports Light (low) through Max (max); Sol and Astra support Light (low) through Max (max), plus Ultra (ultra). Confirm the selected model supports the chosen level. Use Light (low) for straightforward work, Medium (medium) for normal work, High (high) for careful logic and edge cases, Extra High (xhigh) for deep reasoning, and Max or Ultra only when extra deliberation materially helps.

Escalate the specific task: raise Luna's effort if the work remains bounded; use Sol when scope, context, autonomy, or engineering judgment expands; use Astra for genuinely difficult ambiguity or cross-system, security, or correctness reasoning. Start at Sol or Astra when needs are clear. An Astra parent can delegate routine work to Luna or Sol; do not raise model and effort automatically. The lead remains responsible for synthesis and verifying consequential findings.
