# State of the Art Draft TODOs

This file tracks citation, verification, and writing risks for `state_of_the_art.md`.

## Missing Citations

- Stable NIDS/IDPS standard for the basic NIDS definition, preferably NIST SP 800-94 or another authoritative source.
- Classical NIDS taxonomy source covering signature-based, anomaly-based, specification-based, and hybrid approaches.
- Official / audited CSE-CIC-IDS2018 source.
- Primary Bot-IoT source.
- Primary ToN-IoT source.
- Audited ML/DL NIDS survey for model-family overview.
- Audited classification-as-RL critique or prior-work source.
- Audited AE-RL / AE-SAC or dataset-as-environment IDS sources from `Research2.md`.
- Audited autonomous cyber-defense source for broader ACD context.
- Audited QRDQN-in-cybersecurity source, if one is used at all.

## Suspicious or Incomplete Citation Keys

- `DatasetSurvey2025NIDS`: metadata incomplete; verify authors, title, venue, DOI.
- `DLNIDSSurvey2024`: raw research contains future-year/title variants; verify before final use.
- `DRLNIDSSurvey2024`: may be preprint or unstable; verify publication status.
- `Gueriani2024DRLIoTNIDS`: likely preprint; label accordingly if cited.
- `Ring2017FlowBasedIDS`: raw files may mix author/title details; verify exact metadata.
- `CostSensitiveIDSModel`: old or technical-report-like source; use for concept only until audited.
- `CSEIDS2021CostSensitive`: verify authors, venue, and experimental protocol.
- `Tavallaee2009NSLKDD`: appears in citation placement plan but not full must-cite table; add audited BibTeX.
- Any 2025/2026 source from raw research should be treated as needs verification.

## Claims Requiring Verification

- Exact local CICIDS2017 row counts, class counts, and attack-family distribution.
- Exact local feature list after curation, loader cleaning, and canonical mapping.
- Exact mapping from CICFlowMeter feature names to the 76 canonical features.
- Current reward defaults: both `.github/AGENT_CONTEXT.md` and `docs/results.md` agree on `fp=-2.0` (tp=1.5, fn=-5.0, omission=0.0); this matches `src/train_rl_defender.py` and `src/rl_defender_env.py`. No disagreement exists. Historical runs C01/C02 used `fp=-1.0`; that is historical record, not the current default.
- Whether the final thesis will include Random Forest baseline artifacts.
- Whether the final thesis will include a committed full leave-one-CSV-out validation artifact.
- Whether final Phase 2 lab traffic includes attack labels, benign-only traffic, or unlabeled traffic.
- Whether attack-family error analysis is possible with preserved labels.
- Whether data-efficiency experiments are controlled enough to call them a data-efficiency curve.

## Sources Needing BibTeX Cleanup

- `Sharafaldin2018CICIDS2017`
- `Sharafaldin2018CICIDSAnalysis`
- `Lashkari2017CICFlowMeter`
- `Ring2017FlowBasedIDS`
- `DatasetSurvey2025NIDS`
- `CrossDomain2023NIDS`
- `EvalLongTerm2022NIDS`
- `DLNIDSSurvey2024`
- `DRLNIDSSurvey2024`
- `Gueriani2024DRLIoTNIDS`
- `CostSensitiveIDSModel`
- `CSEIDS2021CostSensitive`
- Any source currently represented only by a raw URL, mirror, Scribd/Studocu page, GitHub repo, blog, or LinkedIn post.

## Places Where the Draft May Be Too Generic

- Section 1 needs a stable standard citation and could be tightened once the preferred NIDS/IDPS source is selected.
- Section 3 needs better audited detail on CSE-CIC-IDS2018, Bot-IoT, and ToN-IoT.
- Section 8 needs concrete named prior RL/DRL IDS works after citation audit.
- Section 9 needs one or two audited sources on classification-as-RL or adversarial-environment RL for IDS.
- Section 11 needs final lab validation details once the actual thesis evidence is known.

## Deeper Mathematical Explanation to Add Later

- Q-value definition and Bellman update.
- DQN target network and replay-buffer mechanics.
- Double DQN target calculation.
- Distributional Bellman operator at a high level.
- QRDQN quantile regression loss.
- How asymmetric reward weights affect learned policy incentives.
- Why a binary action space changes the interpretation of value estimates.

Keep those details for algorithm/design chapters unless the final State of the Art needs a short technical bridge.

## Repository Facts to Cross-Check Later

- `src/canonical_schema.py`: 76 canonical features and observation feature names.
- `src/load_cicids2017.py`: leakage drops, numeric cleaning, split behavior, exact adapter return contract.
- `src/rl_defender_env.py`: action semantics and reward defaults.
- `src/train_rl_defender.py`: QRDQN configuration and current reward config.
- `src/validate_checks.py`: Check A/B/C definitions and current defaults.
- `src/validate_leave_one_csv_out.py`: exact-file validation behavior.
- `scripts/predict_real_traffic_v2.py`: Phase 2 preprocessing, clipping, scaler loading, diagnostics, and batch prediction.
- `docs/results.md`: whether RF baseline and leave-one-CSV-out metrics remain pending.

## Overclaim Checks for Next Pass

- RL novelty is not claimed as first-ever.
- QRDQN is not claimed as superior for NIDS.
- CICIDS2017 is not described as production traffic.
- High random-split accuracy is not treated as deployment evidence.
- Phase 2 is offline inference only.
- External validation remains planned/preferred/preliminary unless exact final artifacts support stronger wording.

## Suggested Next Writing Pass

Next task:

```txt
Audit the citation keys in report/drafts/state_of_the_art.md against docs/research/state-of-the-art/nightly/CITATION_PLAN.md and verified source metadata, then replace [CITATION NEEDED] / [VERIFY: ...] markers with stable keys or explicit TODOs.
```

After that:

```txt
Build the experimental design chapter from docs/research/state-of-the-art/nightly/METHODOLOGY_HANDOFF.md
```
