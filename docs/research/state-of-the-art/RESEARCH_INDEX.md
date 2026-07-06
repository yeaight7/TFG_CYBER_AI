# Research Index

This is the canonical structured index for the thesis evidence base. It consolidates the raw research files into research areas, source clusters, claim status, missing areas, and follow-up prompts.

Evidence labels:

- **Strong source:** peer-reviewed paper, official dataset/tool documentation, maintained project artifact, or widely accepted textbook/standard.
- **Plausible but needs verification:** useful raw finding, survey statement, preprint, or incomplete bibliographic entry that needs source audit before final thesis use.
- **Weak / avoid:** blog, opaque repository, LinkedIn, Scribd, Studocu, vague citation, future-year item without stable metadata, or source used only as discovery material.

## 1. Core Thesis Research Areas

| Area | Role in thesis | Current evidence status | Primary raw files |
|---|---|---|---|
| Flow-based NIDS | Background for using flow features instead of packet payloads. | Strong, with flow-survey and CICFlowMeter sources. | `report-source-map.md`, `Research1.md` |
| Public NIDS datasets | Justifies CICIDS2017 and frames NSL-KDD, UNSW-NB15, CSE-CIC-IDS2018, Bot-IoT, ToN-IoT. | Strong for core datasets, moderate for some newer datasets. | `report-NIDS-datasets.md`, `Research3.md` |
| CICIDS2017 | Primary project dataset and source of canonical feature design. | Strong for official facts; limitations must be stated. | `report-deep-dive.md`, `report-source-map.md` |
| Supervised ML/DL baselines | Context for Random Forest, tabular flow baselines, and why RL needs comparison. | Strong that RF is a serious baseline; local RF metrics still pending. | `report-source-map.md`, `Research1.md` |
| RL/DRL for IDS | Places the project among dataset-as-environment and autonomous cyber-defense work. | Moderate; heterogeneous literature and many weak protocols. | `Research2.md`, `deep-research-report2.md` |
| Classification-as-RL | Methodological defense of treating labeled flow classification as an RL environment. | Plausible, but must be framed as an adaptation rather than a canonical standard. | `report-classification-dossier.md` |
| QR-DQN / distributional RL | Algorithm rationale. | Strong for RL fundamentals; moderate for IDS-specific QR-DQN evidence. | `report-qrdqn-deep-distributional-rl.md` |
| Cost-sensitive rewards | Justifies asymmetric FP/FN rewards. | Moderate to strong for cost-sensitive IDS; reward ratios remain project assumptions. | `report-reward-and-cost-sensitive-design-dossier.md` |
| Evaluation methodology | Supports strict splits, leakage controls, and external validation caution. | Strong; central to thesis positioning. | `report-source-map.md`, `Research3.md` |
| Phase 2 offline inference | Connects literature to repo artifacts and lab traffic. | Strong for repo implementation status, weak for broad deployment claims. | `.github/AGENT_CONTEXT.md`, `docs/results.md` |

## 2. Source Clusters

### NIDS and Flow-Based Detection

- **Verified from strong source:** Flow-based IDS is an established approach for representing network activity through bidirectional flow statistics rather than raw payloads. Use Ring2017-FlowSurvey (S5), CICFlowMeter/Lashkari2017 (S3), and standards/background sources from `report-source-map.md`.
- **Plausible but needs verification:** Claims about encrypted-traffic advantages should be tied to flow-level observability and not generalized beyond what sources support.
- **Weak / avoid:** Do not use implementation blogs or GitHub READMEs for formal definitions unless they point to primary documentation.

### Public Datasets

- **Verified from strong source:** NSL-KDD is historical, UNSW-NB15 is a cyber-range dataset, and CICIDS2017/CSE-CIC-IDS2018 are modern CIC-family datasets used heavily in NIDS work. Use Tavallaee2009, Moustafa2015-UNSWNB15, Sharafaldin2018-ICISSP, and DatasetSurvey2025.
- **Plausible but needs verification:** Newer IoT/IIoT dataset claims from Bot-IoT, ToN-IoT, Edge-IIoTset, and CIC IoT datasets need exact dataset-page and paper checks before final writing.
- **Weak / avoid:** Dataset mirrors from Kaggle, Hugging Face, Scribd, Studocu, or market listings should not be treated as canonical if an official source exists.

### CICIDS2017

- **Verified from strong source:** CICIDS2017 was generated to improve on older IDS datasets and provides labeled benign and attack traffic represented as CICFlowMeter flow features.
- **Verified from repo artifact:** This project uses CICIDS2017 as the primary dataset and maps it into 76 canonical flow features plus 76 missingness-mask values, producing 152-dimensional observations.
- **Plausible but needs verification:** Exact flow counts, class counts, and feature counts must match the specific local curated CSVs before final thesis writing.
- **Weak / avoid:** Claims that CICIDS2017 alone demonstrates deployment performance.

### Supervised ML/DL Baselines

- **Verified from strong source:** Random Forest and other tabular supervised models are strong NIDS baselines on CICIDS2017-like flow features.
- **Verified from repo artifact:** The repository includes a Random Forest baseline protocol, but current `docs/results.md` still marks RF metrics as pending.
- **Plausible but needs verification:** Literature numbers for high CICIDS2017 accuracy need protocol audit before comparison.
- **Weak / avoid:** Near-perfect accuracy claims without split, preprocessing, or class-imbalance details.

### RL/DRL for IDS

- **Plausible but needs verification:** DRL for IDS is an active but heterogeneous area. Sources report DQN, SAC, actor-critic, adversarial-environment, and hybrid models across NSL-KDD, CICIDS2017, CSE-CIC-IDS2018, and IoT datasets.
- **Verified from strong source where available:** Foundational RL claims should use Sutton and Barto, Mnih2015, vanHasselt2016, Bellemare2017, and Dabney2018 rather than IDS-specific survey summaries.
- **Weak / avoid:** "No RL-based NIDS exists" and any first-ever claim.

### Classification-as-RL

- **Plausible but needs verification:** A labeled dataset can be exposed as an RL environment where the agent observes one sample, chooses a class/action, and receives reward from the label.
- **Caveat:** This is closer to reward-engineered classification than rich sequential network control unless state transitions encode temporal or adversarial dynamics.
- **Where it fits:** Methodology and limitations, not as proof of autonomous cyber defense.

### QR-DQN / Distributional RL

- **Verified from strong source:** DQN, Double DQN, distributional RL, and QR-DQN have strong foundational sources. QR-DQN models the return distribution through quantile regression.
- **Plausible but needs verification:** Distributional RL can support risk-aware interpretations, but this project must define the actual risk objective before claiming risk-sensitive behavior.
- **Weak / avoid:** Claims that QR-DQN is proven superior for NIDS.

### Cost-Sensitive Reward Design

- **Verified from strong or moderate source:** IDS evaluation has asymmetric costs: false negatives can represent missed attacks, while false positives can harm availability and analyst workload.
- **Verified from repo artifact:** The current project uses asymmetric reward values in the RL environment and validation scripts, but documentation contains a tension between current code defaults and historical run metadata.
- **Plausible but needs verification:** Specific reward variants from the raw dossier are design ideas, not completed experiments.

### Metrics

- **Verified from strong source:** Accuracy alone is insufficient for imbalanced NIDS. Use precision, recall, F1, false-positive rate, false-negative rate, confusion matrix, and per-class metrics.
- **Verified from repo artifact:** `docs/results.md` reports artifact-backed accuracy, precision, recall, F1, TP/FP/FN/TN for committed runs.
- **Plausible but needs verification:** AUROC/AUPRC discussion should be added only if the project actually exports scores or probabilities.

### Leakage and Evaluation Methodology

- **Verified from strong source:** Row-wise random splits can inflate NIDS performance when related flows or scenario artifacts leak across train/test.
- **Verified from repo artifact:** The project explicitly bans IP addresses, absolute timestamps, Flow IDs, unique identifiers, and direct port-proxy leakage features.
- **Verified from repo artifact:** Check B shuffled-label validation is artifact-backed and supports the current no-obvious-leakage claim for that historical run.
- **Weak / avoid:** Treating one anti-leakage check as exhaustive proof.

### External Validation / Lab Traffic

- **Verified from strong source:** Cross-dataset and long-term evaluation studies show poor generalization can occur when ML NIDS models move across domains.
- **Verified from repo artifact:** Phase 2 is offline inference on flow CSVs extracted from private lab traffic, not active blocking.
- **Verified from repo artifact:** Phase 2 behavior changed across committed benign-only artifacts, so claims must cite exact run IDs.
- **Weak / avoid:** Claiming external validation is complete without a committed artifact and clear traffic description.

### Autonomous Cyber Defense

- **Plausible but needs verification:** Autonomous cyber defense literature studies defender agents in simulated or abstracted environments, often beyond flow-level classification.
- **Where it fits:** Broader context and future work.
- **Weak / avoid:** Presenting this project as a fully autonomous defense system.

## 3. Must-Cite Sources

Use these as first-choice sources if their bibliographic details pass final audit:

- Sharafaldin2018-ICISSP / CICIDS2017 dataset design (S1).
- Sharafaldin2018-Analysis / CICIDS2017 detailed analysis and RF baseline (S2).
- Lashkari2017-CICFlowMeter / CICFlowMeter flow features (S3).
- Moustafa2015-UNSWNB15 / UNSW-NB15 cyber-range dataset (S4).
- Ring2017-FlowSurvey / flow-based IDS survey (S5).
- Sharafaldin2016-EvalDataset and Sharafaldin2017-ReliableDataset / IDS dataset evaluation criteria (S6-S7).
- DatasetSurvey2025 / modern NIDS dataset limitations (S8), if bibliographic metadata is stable.
- CrossDomain2023 and EvalLongTerm2022 / generalization risk (S9-S10).
- DLNIDS surveys (S12-S13), after exact venue/year audit.
- DRL-NIDS survey and DRL-IoT-NIDS survey (S14-S15), after audit.
- SuttonBarto2018, Mnih2015-DQN, vanHasselt2016-DoubleDQN, Wang2016-Dueling, Bellemare2017-Distributional, Dabney2018-QRDQN.
- Gymnasium and SB3-Contrib-QRDQN documentation for implementation framing, with version pinned from the repo.
- Cost-sensitive IDS sources S25-S29 for reward rationale, with caveats for older datasets.
- Arp2020 or similar ML-in-security methodology critique from `deep-research-report1.md`.

## 4. Useful but Secondary Sources

- Bot-IoT, ToN-IoT, Edge-IIoTset, LUFlow, OpTC, ARCS, and other dataset/testbed sources for context and future work.
- Mininet-IDS and reproducible ML-IDS sources for reproducibility discussion.
- Offline RL dataset-as-environment benchmarks such as DSRL and TriFinger for methodology analogy.
- Autonomous cyber defense surveys and method papers for broader context.
- Recent adversarial NIDS papers for limitations and future work.

## 5. Weak / Risky / Suspicious Sources

Use only for discovery or implementation hints unless corroborated:

- Blogs, Medium posts, vendor explainers, LinkedIn posts, and generic tutorial pages.
- Scribd, Studocu, opaque document mirrors, and noncanonical PDFs.
- GitHub repositories without associated paper or clear protocol.
- Future-year citations without stable publication metadata.
- Entries with placeholder bibliographic fields, such as "First Author and Others" or fake-looking DOI placeholders.
- Sources reporting extremely high metrics without preprocessing, split, class balance, or baseline details.
- NSL-KDD-only studies used to support modern deployment conclusions.

## 6. Missing Research Areas

- Final audited BibTeX for all must-cite sources.
- Exact local CICIDS2017 row counts, class counts, and post-cleaning feature list.
- Current package versions for Gymnasium, SB3, and SB3-Contrib from the actual environment or lockfile.
- Full committed leave-one-CSV-out validation artifact.
- Completed Random Forest baseline metrics for random, Check C, and leave-one-out splits.
- Clear final description of lab traffic: topology, capture period, traffic generation, labels, and whether any attack traffic exists.
- Reward sensitivity experiments, if the thesis wants to compare reward variants.
- Any probability-score metrics if AUROC/AUPRC will be discussed.

## 7. Recommended Next Research Prompts

- "Audit the must-cite bibliography from `Research/CITATION_PLAN.md` and produce verified BibTeX only for sources with stable DOI, venue, or official URL."
- "Compare the exact local CICIDS2017 curated CSV schema against the official CICIDS2017/CICFlowMeter feature descriptions."
- "Summarize cross-dataset NIDS generalization studies into 5 safe English claims with citations and caveats."
- "Audit the DRL-for-IDS sources and separate peer-reviewed works from arXiv-only or opaque web sources."
- "Prepare a source-backed methodology note on dataset-as-environment classification, explicitly stating limits versus sequential cyber-defense control."
- "Extract project-local package versions and run-artifact IDs needed for final thesis reproducibility tables."
