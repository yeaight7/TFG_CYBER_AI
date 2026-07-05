# Codex Handoff for Thesis Research Drafting

Use this file when a future agent drafts or edits thesis material from the research base. This is a handoff, not a chapter draft.

## Required Reading

Read in this order:

1. `Research/README.md`
2. `Research/RESEARCH_INDEX.md`
3. `Research/CLAIMS_BANK.md`
4. `Research/CITATION_PLAN.md`
5. `.github/AGENT_CONTEXT.md`
6. `docs/results.md`
7. Focused raw dossier only for the section being drafted.

Use `Research/report-source-map.md` as the raw backbone. Use focused raw dossiers for depth:

- CICIDS2017: `Research/report-deep-dive.md`
- Dataset comparison: `Research/report-NIDS-datasets.md`
- Classification-as-RL: `Research/report-classification-dossier.md`
- QR-DQN: `Research/report-qrdqn-deep-distributional-rl.md`
- Reward design: `Research/report-reward-and-cost-sensitive-design-dossier.md`
- RL/DRL IDS closest works: `Research/Research2.md`

## Project Facts to Preserve

- The thesis/repo documentation baseline is English.
- The current project is an RL-based network-flow defender with binary actions:
  - `0 = PERMIT`
  - `1 = BLOCK`
- CICIDS2017 is the primary public dataset.
- The canonical schema has 76 flow features.
- The final observation has 152 values: 76 canonical values plus 76 missingness-mask values.
- Missingness mask semantics:
  - `1 = present / valid`
  - `0 = missing / imputed`
- The main RL algorithm is QR-DQN.
- Phase 1 is offline training and validation on datasets.
- Phase 2 is offline inference on extracted flow CSVs from private lab traffic.
- Active inline blocking is not implemented.
- Leave-one-exact-CSV-out validation exists in code, but no committed full artifact is currently reported in `docs/results.md`.
- Random Forest baseline protocol exists, but baseline metrics are still placeholders in `docs/results.md`.
- Phase 2 behavior must be tied to exact run artifacts because committed benign-only runs differ.

## Allowed Claims

Use claims from `CLAIMS_BANK.md` first. Good thesis posture:

- The project is a reproducible experimental prototype.
- CICIDS2017 is a useful benchmark but not enough for deployment conclusions.
- QR-DQN is a defensible distributional RL candidate, not proven best for NIDS.
- Cost-sensitive rewards are motivated by asymmetric FP/FN risk, but reward weights are scenario assumptions.
- Phase 2 is a preliminary external-distribution check under controlled offline conditions.
- Strict evaluation is necessary because NIDS literature often suffers from optimistic metrics, leakage risk, and poor cross-domain generalization.

## Forbidden Claims

Do not write that:

- The model works in the real world.
- The system is production-ready.
- CICIDS2017 fully represents modern traffic.
- RL for IDS has never been studied.
- QR-DQN is proven superior for NIDS.
- External validation is complete without committed supporting artifacts.
- Active real-time blocking is implemented.
- High random-split performance proves deployment robustness.
- The project is the first IDS to use a dataset-as-environment formulation.

## Citation Discipline

- Cite underlying papers, official dataset/tool docs, repo docs, code, and run artifacts.
- Do not cite raw research files in the thesis.
- Do not invent missing authors, years, titles, venues, DOIs, or pages.
- If a raw source has unstable metadata, either verify it or omit it.
- When citing project results, cite exact run IDs and artifact paths.
- When citing current implementation behavior, verify against code and maintained docs.
- When citing literature metrics, confirm the original split/protocol first.

## Writing Boundaries

- Do not draft final State of the Art text unless explicitly asked.
- Do not translate the repo or normalized research layer to Spanish.
- Do not edit `docs/DEFENSA_*`.
- Do not edit `report/report.tex` unless explicitly asked.
- Do not present raw Spanish dossier paragraphs as final English thesis prose.
- Do not run training or experiments during writing tasks unless explicitly requested.

## Suggested Drafting Structure for a Future Agent

When drafting later, use this order:

1. NIDS and flow-based detection background.
2. Public datasets and why CICIDS2017 is selected.
3. Supervised ML/DL baselines and why RF matters.
4. RL and DRL foundations, including DQN and QR-DQN.
5. RL/DRL for IDS and classification-as-RL positioning.
6. Cost-sensitive reward design.
7. Evaluation methodology, leakage, and cross-domain limitations.
8. Project positioning: offline QR-DQN PERMIT/BLOCK prototype with CICIDS2017 benchmark and offline lab-flow inference.

Keep the final positioning modest: methodological and experimental contribution, not operational deployment.

## Pre-Draft Checklist

- Verify current code/doc facts against `.github/AGENT_CONTEXT.md`, `docs/AGENT_CONTEXT.md`, and `docs/results.md`.
- Check whether `docs/results.md` has changed since this handoff.
- Audit citation metadata for sources selected from `CITATION_PLAN.md`.
- Confirm whether new validation artifacts exist under `runs/`.
- Confirm whether RF baseline metrics have been populated.
- Confirm whether Phase 2 lab data has labels or remains benign-only / unlabeled.
