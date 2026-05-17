# Research Folder Index

## Purpose

This folder contains the raw and normalized research base for the thesis literature review. The raw files come from Perplexity / Deep Research style exploration and are useful, but they are not safe to cite directly without verification. The normalized files added here provide a writing-ready index, claims bank, citation plan, and Codex handoff without drafting the final State of the Art chapter.

Raw research may contain duplicated sources, uncertain citation metadata, weak web sources, future-year items, and older Spanish drafting assumptions. Treat it as source material, not final thesis text.

## Recommended Reading Order

1. `RESEARCH_INDEX.md` for the canonical structured map.
2. `CLAIMS_BANK.md` for allowed, risky, and forbidden claims.
3. `CITATION_PLAN.md` for citation priority and verification status.
4. `STATE_OF_ART_HANDOFF.md` before asking an agent to draft the State of the Art chapter.
5. `METHODOLOGY_HANDOFF.md` before writing methodology/design sections or planning experiments.
6. `RESEARCH_GAPS_AND_TODOS.md` to see unresolved work before drafting.
7. `CODEX_HANDOFF.md` for general agent guardrails.
8. `report-source-map.md` as the raw backbone behind the normalized layer.
9. Focused dossiers as needed: CICIDS2017, datasets, QR-DQN, reward design, and classification-as-RL.
10. Earlier raw research files only for extra context or source discovery.

## File Inventory

| File | Type | Main topic | Usefulness | Status | Notes |
|---|---|---|---|---|---|
| `README.md` | Normalized index | Folder guide | Start here to understand the folder structure. | Processed | This file. |
| `RESEARCH_INDEX.md` | Normalized index | Canonical research map | Main navigation layer for thesis research areas, source clusters, and gaps. | Processed | Uses `report-source-map.md` as backbone. |
| `CLAIMS_BANK.md` | Normalized claims bank | Safe, risky, and forbidden claims | Reusable claim wording for thesis drafting. | Processed | Does not draft final prose. |
| `CITATION_PLAN.md` | Normalized citation plan | Must-cite, secondary, weak, and verify-first sources | Guides which sources should support formal thesis claims. | Processed | Exact BibTeX still needs final audit for some entries. |
| `STATE_OF_ART_HANDOFF.md` | Normalized handoff | Future State of the Art drafting | Precise section-by-section plan with claims, sources, caveats, and repo links. | Processed | This is the main handoff for later chapter drafting. |
| `METHODOLOGY_HANDOFF.md` | Normalized handoff | Methodology and experimental design | Actionable notes for benchmark design, baselines, metrics, leakage controls, and external validation. | Processed | Includes Plan A/B/C for lab validation. |
| `RESEARCH_GAPS_AND_TODOS.md` | Normalized TODO list | Remaining research and experiment gaps | Tracks missing sources, uncertain claims, experiments, writing decisions, and advisor questions. | Processed | Keep updated before final drafting. |
| `CODEX_HANDOFF.md` | Normalized handoff | Instructions for future writing agents | Prevents unsafe claims and stale project assumptions during drafting. | Processed | English-only, no chapter draft. |
| `report-source-map.md` | Raw master dossier | Full taxonomy, source matrix, claims bank, citation groups | Best raw backbone for the whole literature review. | Raw / needs verification | Contains useful source IDs S1-S38, but some entries need bibliographic audit. |
| `report-deep-dive.md` | Raw focused dossier | CICIDS2017 | Useful for dataset description, CICFlowMeter, limitations, and evaluation protocol. | Raw / needs verification | Contains thesis-ready Spanish wording that should not be copied into English docs. |
| `report-NIDS-datasets.md` | Raw focused dossier | NIDS dataset comparison | Useful for comparing CICIDS2017, NSL-KDD, UNSW-NB15, Bot-IoT, ToN-IoT, and related datasets. | Raw / needs verification | Good for dataset-selection rationale and dangerous-claim warnings. |
| `report-classification-dossier.md` | Raw focused dossier | Classification-as-RL | Useful for defending dataset-as-environment framing and its limitations. | Raw / needs verification | Contains many broad web links; filter before formal citation. |
| `report-qrdqn-deep-distributional-rl.md` | Raw focused dossier | DQN, distributional RL, QR-DQN | Useful for algorithm background and QR-DQN motivation. | Raw / needs verification | Strong for foundational RL keys, weaker for cybersecurity-specific QR-DQN claims. |
| `report-reward-and-cost-sensitive-design-dossier.md` | Raw focused dossier | Reward and cost-sensitive IDS design | Useful for FP/FN trade-offs and reward-design justification. | Raw / needs verification | Includes suggested reward variants; do not treat as implemented experiments. |
| `Research1.md` | Raw broad dossier | Source map, datasets, ML/DL, DRL, evaluation gaps | Useful for additional sources and methodological warnings. | Raw / needs verification | Contains older Spanish outline assumptions and many source links. |
| `Research2.md` | Raw focused dossier | RL/DRL for IDS | Useful for closest-work comparison, reward patterns, baselines, and weak areas. | Raw / needs verification | Good for identifying what not to overclaim about RL novelty. |
| `Research3.md` | Raw focused dossier | Datasets, CICIDS2017, leakage, external validation | Useful for dataset limitations and evaluation protocol. | Raw / needs verification | Includes Spanish drafting handoff, not final text. |
| `deep-research-report1.md` | Raw synthesis dossier | State-of-the-art structure, source base, safe/avoid claims | Useful for high-level positioning and BibTeX candidates. | Raw / needs verification | Strong warning source, but written around Spanish chapter drafting. |
| `deep-research-report2.md` | Raw positioning dossier | Defensible research gap | Useful for narrowing thesis contribution and avoiding novelty overclaims. | Raw / needs verification | Contains suggested Spanish paragraphs; do not copy as final chapter text. |

## Safety Notes

- Do not cite raw research files as sources in the thesis. Cite the underlying papers, standards, dataset pages, documentation, or repo artifacts.
- Do not use literature metrics as proof that this project performs in deployment.
- Do not describe Phase 2 as active blocking. Current maintained scope is offline inference on extracted flow CSVs.
- Do not treat CICIDS2017 as fully representative of current network traffic.
- Do not use weak or opaque sources for formal claims unless they are corroborated by stronger primary or survey sources.
