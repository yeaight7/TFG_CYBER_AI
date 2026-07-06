# Research Gaps and TODOs

This file lists remaining gaps after consolidating the raw research folder. It is intentionally skeptical.

## Missing Sources

| Gap | Why it matters | Next action |
|---|---|---|
| Stable IDS/IDPS background source | Needed for NIDS definition and distinction from prevention systems. | Choose NIST SP 800-94 or another stable standard if used. |
| Final authoritative CICIDS2017 citation | Needed for dataset description. | Verify official paper, official page, and local dataset variant. |
| Final CICFlowMeter citation | Needed for flow feature extraction. | Verify paper plus tool documentation/version. |
| Strong supervised tabular baseline source | Needed to justify Random Forest baseline. | Audit CICIDS2017 RF baseline papers and keep only clear protocols. |
| Peer-reviewed DRL-for-IDS survey | Needed for RL/DRL literature positioning. | Verify `DRLNIDSSurvey2024` publication status. |
| Classification-as-RL prior work | Needed to contextualize dataset-as-environment design. | Audit AE-RL / AE-SAC and related sources from `Research2.md`. |
| Cost-sensitive IDS source with modern dataset | Needed for reward design. | Verify S25-S29 and prefer modern NIDS datasets over NSL-KDD-only work. |
| Cyber range / lab validation source | Needed for external validation section. | Use UNSW-NB15 plus one or two verified cyber-range/testbed papers. |

## Uncertain Citations

| Citation / source group | Issue | Treatment |
|---|---|---|
| DatasetSurvey2025 | Authors/venue/DOI not fully audited. | Keep as needs verification. |
| DLNIDS-Survey1 / DLNIDSSurvey2024 | Raw files contain future-year and title variations. | Audit before must-cite use. |
| DRL-NIDS survey entries | May be arXiv-only or unstable. | Use qualitatively unless peer-reviewed. |
| Evaluation-SLR-NIDS | Metadata incomplete in raw source map. | Verify original paper. |
| Cost-sensitive sources S25-S29 | Some are older, thesis-like, or NSL-KDD-specific. | Use for concepts, not modern performance claims. |
| Future-year papers | Some may be unstable or hallucinated by raw research tools. | Do not cite until verified externally. |
| Scribd / Studocu / LinkedIn / blog links | Weak provenance. | Avoid for formal claims. |
| Duplicated Sharafaldin entries | Raw files use multiple related keys. | Normalize to separate dataset-generation and analysis keys. |

## Claims Needing Verification

| Claim | Required verification |
|---|---|
| CICIDS2017 local feature count and row count | Inspect local curated CSVs and loader output. |
| Exact mapping from CICFlowMeter features to 76 canonical features | Verify against `src/canonical_schema.py` and loader mapping. |
| Current reward defaults | Compare `.github/AGENT_CONTEXT.md`, `docs/results.md`, and code; note any disagreement. |
| RF baseline performance | Run baseline protocol and add artifact before citing metrics. |
| QRDQN vs baseline comparison | Same-split, same-preprocessing comparison with RF and optionally DQN. |
| leave-one-CSV-out performance | Run and commit/artifact full validation before reporting metrics. |
| Phase 2 external validation | Needs lab protocol, labels or clear benign-only scope, and exact run artifacts. |
| Data-efficiency | Needs controlled runs where only data size changes. |
| Attack-family error analysis | Needs preserved attack-family labels in evaluation artifacts. |

## Experiments Needed

| Experiment | Priority | Notes |
|---|---|---|
| Random Forest full sweep | High | Already listed as pending in `docs/results.md`. |
| leave-one-CSV-out validation | High | Code exists; full artifact missing. |
| Multi-seed QRDQN runs | Medium / high | Needed for robust comparative claims. |
| Data-efficiency curve | Medium | Only valid if protocol is controlled. |
| Reward sensitivity variants | Medium | Needed if reward design becomes a major thesis claim. |
| Attack-family error analysis | Medium | Depends on labels and artifacts. |
| Phase 2 lab validation | High if thesis needs external validation | Plan A/B/C in `METHODOLOGY_HANDOFF.md`. |
| DQN vs QRDQN comparison | Optional | Useful if claiming distributional RL benefit. |

## Writing Decisions Needed

| Decision | Options | Recommendation |
|---|---|---|
| How strong to make the external-validation claim | Completed / preliminary / planned | Use preliminary or planned unless final artifact exists. |
| Whether to include "classification-as-RL" as its own section | Separate section / methodology subsection | Use a short separate section in State of the Art, then expand in methodology. |
| How much RL theory to include | Short foundations / long tutorial | Keep short and tied to DQN/QRDQN. |
| Whether to discuss autonomous cyber defense | Full section / context paragraph | Keep as context and future work; not implemented scope. |
| Whether to include adversarial robustness | Limitations / main chapter section | Put in limitations unless experiments exist. |
| Language of final chapter | English / Spanish | Current repo baseline says English. Do not use Spanish raw paragraphs directly. |

## Advisor / Tutor Questions

- Is the final thesis expected to include an external lab validation result, or is a planned validation protocol acceptable?
- Is a Random Forest baseline sufficient, or should DQN/PPO/MLP baselines also be required?
- How many random seeds are expected for final reported RL results?
- Should the State of the Art include autonomous cyber defense as context, or keep it brief to avoid scope creep?
- Should QRDQN be framed as the main contribution or as one algorithmic choice inside a broader evaluation protocol?
- What level of detail is expected for CICIDS2017 preprocessing and feature mapping?
- If Phase 2 remains benign-only, is that acceptable as a false-positive/domain-shift check?
- Should attack-family analysis be mandatory even though the main RL task is binary?

## Explicit TODO List

- Verify must-cite metadata and build final BibTeX only for complete sources.
- Resolve reward-default documentation mismatch before final methodology writing.
- Confirm whether new validation artifacts exist before drafting results-linked claims.
- Decide final external-validation claim strength.
- Replace any weak raw sources with primary sources or remove them.
- Keep all overclaim warnings visible in drafting handoffs.
