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
| [research/personal/deep-defense-research/README.md](research/personal/deep-defense-research/README.md) | Author | Deep multi-file thesis/defense research notes (Spanish, Markdown) — research only, **not a strong source of truth** | Reference |
| [../experiments/README.md](../experiments/README.md) | Author, reviewers | Historical experiment archive index | Maintained |
| [defensa/README.md](defensa/README.md) | Author | Index of Spanish defense-prep material | Maintained, Spanish by design |
| [defensa/DEFENSA_TFG_PROGRESO.md](defensa/DEFENSA_TFG_PROGRESO.md) | Author | Spanish defense preparation notes | Maintained, Spanish by design |
| [defensa/DEFENSA_TFG_SCRIPT.md](defensa/DEFENSA_TFG_SCRIPT.md) | Author | Spanish defense script | Maintained, Spanish by design |
| [defensa/GUIA_TECNICA_TUTOR.md](defensa/GUIA_TECNICA_TUTOR.md) | Author | Tutor's technical guide: script + 50+10 answer bank + glossary | Reference, Spanish by design |
| [defensa/Preguntas-del-Tutor-para-verificar-comprension-del-alumno.md](defensa/Preguntas-del-Tutor-para-verificar-comprension-del-alumno.md) | Author | Tutor question bank to verify understanding | Reference, Spanish by design |
| [defensa/chatgpt_project_study_packet.md](defensa/chatgpt_project_study_packet.md) | Author | Project study packet (EN/ES) with tutor prompt | Reference |
| [research/README.md](research/README.md) | Author | Consolidated research notes index — not a source of truth | Reference |
| [../memoria/memoria.tex](../memoria/memoria.tex) | Author, evaluators, tribunal | Canonical thesis (Spanish) — official source | Maintained |
| [../report/report.tex](../report/report.tex) | Author | English thesis draft — parked (may lag new sections) | Historical |
| [audits/](audits/) | Author | Dated read-only repository audits (historical snapshots) | Reference |

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
  - [defensa/](defensa/) (all defense-prep material)
  - [research/personal/deep-defense-research/](research/personal/deep-defense-research/)

## Notes

- Historical material should be clearly marked as historical.
- Results should only be presented as authoritative if backed by artifacts under `runs/`.
- If code and documentation disagree, update the documentation to match the current code and artifact state.
- Obsolete drafts live in [archive/](archive/) (e.g. the superseded `informe.tex`/`.pdf`, an early draft of the Spanish thesis `memoria/`).
