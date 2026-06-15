# Documentation Index

This directory contains the maintained project documentation used for development, submission preparation, and defense preparation.

## Document Roles

| Document | Audience | Purpose | Status |
|----------|----------|---------|--------|
| [../README.md](../README.md) | Evaluators, contributors, reviewers | Public project overview and quickstart | Maintained |
| [../.github/AGENT_CONTEXT.md](../.github/AGENT_CONTEXT.md) | Contributors and agents | Project-wide technical source of truth | Maintained |
| [AGENT_CONTEXT.md](AGENT_CONTEXT.md) | Contributors working on Phase 2 | Phase 2 scope, guardrails, and run conventions | Maintained |
| [results.md](results.md) | Evaluators, author, reviewers | Artifact-backed results snapshot | Maintained |
| [phase2_plan.md](phase2_plan.md) | Author, contributors | Execution plan for the Phase 2 workflow | Maintained |
| [gcp_lab.md](gcp_lab.md) | Author, contributors | Private lab setup guide | Maintained |
| [runpod_main_experiment.md](runpod_main_experiment.md) | Author, contributors | Single main RunPod training run guide | Maintained |
| [reproducibility.md](reproducibility.md) | Author, contributors | Dependency and environment reproduction notes | Maintained |
| [Personal Research/deep-defense-research/README.md](Personal%20Research/deep-defense-research/README.md) | Author | Deep multi-file thesis/defense research pack (Spanish) | Maintained |
| [../experiments/README.md](../experiments/README.md) | Author, reviewers | Historical experiment archive index | Maintained |
| [DEFENSA_TFG_PROGRESO.md](DEFENSA_TFG_PROGRESO.md) | Author | Spanish defense preparation notes | Maintained, Spanish by design |
| [DEFENSA_TFG_SCRIPT.md](DEFENSA_TFG_SCRIPT.md) | Author | Spanish defense script | Maintained, Spanish by design |
| [informe.tex](informe.tex) / [informe.pdf](informe.pdf) | Author, evaluators | LaTeX source and compiled thesis submission report | Maintained |

## Reading Order

If you are new to the repository, read in this order:

1. [../README.md](../README.md)
2. [../.github/AGENT_CONTEXT.md](../.github/AGENT_CONTEXT.md)
3. [results.md](results.md)
4. [AGENT_CONTEXT.md](AGENT_CONTEXT.md)
5. [phase2_plan.md](phase2_plan.md)
6. [reproducibility.md](reproducibility.md)

## Language Policy

- English is the default language for technical repository documentation such as `README.md`, implementation notes, contributor guidance, and reproducibility guides.
- Spanish is the default language for thesis, memoria, and defense-facing material.
- Spanish is intentionally preserved and maintained for:
  - [DEFENSA_TFG_PROGRESO.md](DEFENSA_TFG_PROGRESO.md)
  - [DEFENSA_TFG_SCRIPT.md](DEFENSA_TFG_SCRIPT.md)
  - [Personal Research/deep-defense-research/](Personal%20Research/deep-defense-research/)

## Notes

- Historical material should be clearly marked as historical.
- Results should only be presented as authoritative if backed by artifacts under `runs/`.
- If code and documentation disagree, update the documentation to match the current code and artifact state.
