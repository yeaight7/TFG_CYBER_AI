# Claims Bank

This file provides thesis-ready claim material. It is not a chapter draft. Use recommended wording as building blocks only after verifying final citations.

## Safe Claims

### Claim 1: Flow-based NIDS is an established representation

- **Claim:** Flow-based intrusion detection represents network activity through aggregated bidirectional flow features rather than raw packet payloads.
- **Evidence strength:** Strong.
- **Supporting sources / citation keys:** Ring2017-FlowSurvey (S5); Lashkari2017-CICFlowMeter (S3); Claise/IPFIX material from raw source map.
- **Where to use it:** Background and feature-representation sections.
- **Caveat:** Flow features reduce payload visibility and may lose semantic detail.
- **Recommended wording:** "Flow-based NIDS methods model traffic using aggregated statistics over network flows, which makes them practical for scalable traffic analysis but less expressive than payload-level inspection."

### Claim 2: CICIDS2017 is a reasonable primary benchmark

- **Claim:** CICIDS2017 is a widely used modern benchmark for flow-based NIDS research and is more suitable than legacy KDD-style datasets for this project.
- **Evidence strength:** Strong.
- **Supporting sources / citation keys:** Sharafaldin2018-ICISSP (S1); Sharafaldin2018-Analysis (S2); DatasetSurvey2025 (S8); `docs/results.md`.
- **Where to use it:** Dataset selection.
- **Caveat:** It remains a controlled lab dataset with known limitations and does not represent all current traffic.
- **Recommended wording:** "CICIDS2017 is used as the main public benchmark because it provides labeled flow-based traffic and modern attack scenarios, while still requiring cautious interpretation because it is a controlled benchmark dataset."

### Claim 3: The project uses a fixed canonical flow schema

- **Claim:** The implemented project maps CICIDS2017 and Phase 2 flow inputs into 76 canonical flow features and appends a 76-value missingness mask, producing 152-dimensional observations.
- **Evidence strength:** Strong.
- **Supporting sources / citation keys:** `.github/AGENT_CONTEXT.md`; `AGENTS.md`; `src/canonical_schema.py`.
- **Where to use it:** Methodology and implementation.
- **Caveat:** Do not present the schema as a universal standard; it is this project's canonical schema.
- **Recommended wording:** "In this implementation, each observation contains 76 canonical flow features and a 76-value missingness mask, yielding a 152-dimensional input vector."

### Claim 4: Supervised baselines are necessary

- **Claim:** Any RL-based IDS formulation should be compared against strong supervised baselines, especially Random Forest on tabular flow features.
- **Evidence strength:** Strong.
- **Supporting sources / citation keys:** Sharafaldin2018-Analysis (S2); DLNIDS-SLR2024 (S13); `docs/results.md`.
- **Where to use it:** Methodology and limitations.
- **Caveat:** The repo currently marks RF baseline metrics as pending.
- **Recommended wording:** "Because tabular flow-based NIDS often performs strongly with supervised models, the RL formulation should be interpreted alongside supervised baselines rather than in isolation."

### Claim 5: Accuracy alone is insufficient

- **Claim:** NIDS evaluation should report per-class precision, recall, F1, false positives, and false negatives, not only accuracy.
- **Evidence strength:** Strong.
- **Supporting sources / citation keys:** DatasetSurvey2025 (S8); DLNIDS surveys (S12-S13); `docs/results.md`.
- **Where to use it:** Evaluation methodology.
- **Caveat:** Add AUROC/AUPRC only if the implementation exports suitable scores.
- **Recommended wording:** "For imbalanced intrusion-detection data, aggregate accuracy can hide missed attacks or excessive false alarms, so the evaluation reports per-class precision, recall, F1, and confusion-matrix counts."

### Claim 6: Random row splits can be optimistic

- **Claim:** Random row-wise splits can overestimate NIDS performance when related flows, time periods, or scenario artifacts appear in both train and test data.
- **Evidence strength:** Strong.
- **Supporting sources / citation keys:** CrossDomain2023 (S9); EvalLongTerm2022 (S10); Evaluation-SLR-NIDS (S38); Arp2020.
- **Where to use it:** Evaluation critique and split design.
- **Caveat:** Random splits are not invalid for all purposes, but they should not be treated as deployment evidence.
- **Recommended wording:** "Random splits are useful for internal checks, but stricter temporal, scenario, or file-based splits are needed to test whether performance survives distribution changes."

### Claim 7: Cross-domain generalization is a known weakness

- **Claim:** ML-based NIDS models can perform well on one public dataset and degrade substantially when evaluated on another dataset or later traffic distribution.
- **Evidence strength:** Strong.
- **Supporting sources / citation keys:** CrossDomain2023 (S9); EvalLongTerm2022 (S10); DatasetSurvey2025 (S8).
- **Where to use it:** Limitations and Phase 2 motivation.
- **Caveat:** Generalization depends on feature mapping, labels, traffic domain, and adaptation strategy.
- **Recommended wording:** "Prior cross-domain evaluations show that high in-dataset scores do not guarantee stable performance under a different traffic distribution."

### Claim 8: RL for IDS exists but is methodologically heterogeneous

- **Claim:** DRL-based IDS has been studied, but the literature is heterogeneous and often limited by dataset choice, unclear protocols, or weak external validation.
- **Evidence strength:** Moderate.
- **Supporting sources / citation keys:** DRL-NIDS-Survey1 (S14); DRL-IoT-NIDS (S15); Research2 close-work matrix.
- **Where to use it:** Related work on RL/DRL for NIDS.
- **Caveat:** Verify original papers before using their performance numbers.
- **Recommended wording:** "DRL-based IDS is an active research direction, but reported results vary across datasets and protocols, and many works still need stronger reproducibility and external-validation evidence."

### Claim 9: Dataset-as-environment is an adaptation

- **Claim:** This project formulates flow classification as an RL environment where the action is a binary PERMIT/BLOCK decision and reward is derived from the labeled sample.
- **Evidence strength:** Moderate.
- **Supporting sources / citation keys:** `src/rl_defender_env.py`; Gymnasium (S32); `report-classification-dossier.md`; offline RL dataset analogies (S33-S34).
- **Where to use it:** Methodology.
- **Caveat:** This formulation is not equivalent to a rich online control environment unless temporal/adversarial dynamics are modeled.
- **Recommended wording:** "The environment adapts a labeled flow dataset into an RL interface: the agent observes one flow representation, chooses PERMIT or BLOCK, and receives a reward based on the ground-truth label."

### Claim 10: QR-DQN is justified as an experimental distributional RL choice

- **Claim:** QR-DQN is a defensible algorithmic choice because it extends value-based RL by estimating a return distribution through quantile regression.
- **Evidence strength:** Strong for RL theory, moderate for NIDS-specific advantage.
- **Supporting sources / citation keys:** Mnih2015-DQN; Bellemare2017-Distributional; Dabney2018-QRDQN; SB3-Contrib-QRDQN (S35).
- **Where to use it:** Algorithm background and implementation.
- **Caveat:** Do not claim QR-DQN is superior for NIDS unless supported by this project's comparisons.
- **Recommended wording:** "QR-DQN is used as an experimental value-based RL algorithm that estimates a distribution over returns, which is relevant when decisions have asymmetric costs."

### Claim 11: Cost-sensitive rewards match IDS risk structure

- **Claim:** Penalizing false negatives more heavily than false positives is consistent with cost-sensitive IDS literature, provided the chosen weights are described as scenario assumptions.
- **Evidence strength:** Moderate.
- **Supporting sources / citation keys:** FP-FN-CostModel2014 (S26); CostSensitiveModeling-IDS (S28); CSE-IDS-CostSensitive (S25); `report-reward-and-cost-sensitive-design-dossier.md`.
- **Where to use it:** Reward design.
- **Caveat:** Real operational costs are context-specific; reward weights require sensitivity analysis if claimed as robust.
- **Recommended wording:** "The reward function encodes an asymmetric cost assumption, assigning higher penalty to missed attacks while still penalizing false alarms."

### Claim 12: Phase 2 is offline inference, not active blocking

- **Claim:** The implemented Phase 2 pipeline performs offline inference on extracted flow CSVs from private lab traffic.
- **Evidence strength:** Strong.
- **Supporting sources / citation keys:** `.github/AGENT_CONTEXT.md`; `docs/AGENT_CONTEXT.md`; `scripts/predict_real_traffic_v2.py`; `docs/results.md`.
- **Where to use it:** Methodology, results, limitations.
- **Caveat:** Do not describe it as inline prevention or active response.
- **Recommended wording:** "Phase 2 evaluates the trained model through offline inference over extracted lab-flow CSVs; it does not implement inline packet or flow blocking."

### Claim 13: Phase 2 behavior is artifact-specific

- **Claim:** Phase 2 benign-only results changed across committed runs, so each behavioral statement must cite the exact run artifact.
- **Evidence strength:** Strong.
- **Supporting sources / citation keys:** `docs/results.md`; `runs/phase2/P2v2_pred_20260224_004121/`; `runs/phase2/P2v2_pred_20260408_230318/`.
- **Where to use it:** Results and limitations.
- **Caveat:** This supports caution, not external deployment validity.
- **Recommended wording:** "Phase 2 behavior is reported per run artifact because committed benign-only runs show different block/allow behavior under different conditions."

## Risky Claims

### Risky claim 1

- **Risky claim:** The project demonstrates performance on real networks.
- **Why risky:** Current Phase 2 is offline and controlled; behavior varies across artifacts.
- **Safer alternative:** "The project includes offline inference on private lab traffic as an initial distribution-shift check."
- **Verification needed:** Stable lab dataset description, labels, run artifacts, and protocol.

### Risky claim 2

- **Risky claim:** CICIDS2017 performance is enough to establish general robustness.
- **Why risky:** Cross-domain literature shows in-dataset metrics may not transfer.
- **Safer alternative:** "CICIDS2017 provides a reproducible benchmark, but robustness must be evaluated with stricter splits and external traffic."
- **Verification needed:** Check C, leave-one-exact-CSV-out artifact, and Phase 2 runs.

### Risky claim 3

- **Risky claim:** QR-DQN is better than DQN, RF, or other baselines for NIDS.
- **Why risky:** Foundational QR-DQN evidence is not NIDS-specific; project comparisons are incomplete.
- **Safer alternative:** "QR-DQN is evaluated as a distributional RL candidate for the binary flow-defense formulation."
- **Verification needed:** Same-split comparisons against DQN and Random Forest.

### Risky claim 4

- **Risky claim:** The reward function captures operational security cost.
- **Why risky:** Cost ratios are scenario assumptions, not measured business/security impact.
- **Safer alternative:** "The reward function approximates an asymmetric cost preference between missed attacks and false alarms."
- **Verification needed:** Reward-sensitivity experiments or an explicit cost model.

### Risky claim 5

- **Risky claim:** Shuffled-label validation proves there is no leakage.
- **Why risky:** It is one useful check, not exhaustive proof.
- **Safer alternative:** "The shuffled-label artifact reduces the likelihood of obvious leakage in that run."
- **Verification needed:** Feature audit, split audit, and additional leakage probes.

### Risky claim 6

- **Risky claim:** The lab traffic is external validation in the strong deployment sense.
- **Why risky:** The lab is private, controlled, and artifact-specific.
- **Safer alternative:** "The lab traffic provides a preliminary external-distribution check under controlled conditions."
- **Verification needed:** Lab topology, capture process, labels, and reproducible run folders.

## Forbidden Claims

Do not make these claims in the thesis or repo documentation:

- The model works in the real world.
- The system is production-ready.
- CICIDS2017 fully represents modern traffic.
- RL for IDS has never been studied.
- QR-DQN is proven superior for NIDS.
- External validation is complete if not backed by artifacts.
- Active real-time blocking is implemented.
- The project is the first dataset-as-environment IDS.
- High random-split accuracy proves deployment robustness.
- NSL-KDD-only results prove modern NIDS performance.
- Phase 2 lab inference is equivalent to operational cyber defense.
- A single benign-only lab run proves benign traffic safety.
