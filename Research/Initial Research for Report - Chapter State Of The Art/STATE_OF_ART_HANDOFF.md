# State of the Art Handoff

This is the handoff for a future Codex task that drafts the State of the Art chapter. It is not the chapter draft.

## Recommended Chapter Structure

Use English section titles:

1. Network Intrusion Detection Systems
2. Flow-based Traffic Representation
3. Public Datasets for NIDS
4. CICIDS2017 as the Main Internal Benchmark
5. Supervised ML and DL for NIDS
6. Reinforcement Learning Background
7. DQN and Distributional RL / QRDQN
8. RL/DRL for Intrusion Detection
9. Classification-as-RL Formulation
10. Evaluation Pitfalls in ML-based NIDS
11. External Validation and Lab Traffic
12. Positioning of This Thesis

## Section Plans

| Section | Purpose | Key claims | Sources / citation keys | Caveats | Forbidden overclaims | Repo implementation connection |
|---|---|---|---|---|---|---|
| Network Intrusion Detection Systems | Introduce NIDS and why ML methods are used. | NIDS monitor network activity for malicious behavior; anomaly and ML methods complement, not replace, classical approaches. | IDPS standard if used; `Ring2017FlowBasedIDS`; dataset surveys. | Avoid long IPS detail; this project is not an active IPS. | Do not claim active real-time blocking. | Binary action vocabulary is `PERMIT` / `BLOCK`, but current execution is offline. |
| Flow-based Traffic Representation | Explain flows and why flow features fit this project. | Flow-based NIDS uses aggregate statistics over bidirectional traffic flows. | `Lashkari2017CICFlowMeter`; `Ring2017FlowBasedIDS`; `ProjectAgentContext`. | Flow features lose payload semantics and can still encode dataset artifacts. | Do not claim flow features fully describe attacks. | Project maps flow inputs to 76 canonical features plus mask. |
| Public Datasets for NIDS | Situate CICIDS2017 among public benchmarks. | Public datasets support reproducibility but are controlled approximations. | `Sharafaldin2016DatasetFramework`; `Sharafaldin2017ReliableDataset`; `DatasetSurvey2025NIDS`; `MoustafaSlay2015UNSWNB15`; audited NSL-KDD source. | Dataset metadata differs across mirrors and releases. | Do not claim any public dataset represents all modern traffic. | NSL-KDD is historical in this repo; CICIDS2017 is primary. |
| CICIDS2017 as the Main Internal Benchmark | Justify main dataset choice. | CICIDS2017 is a modern flow-based benchmark with labeled benign and attack traffic. | `Sharafaldin2018CICIDS2017`; `Sharafaldin2018CICIDSAnalysis`; official CIC/UNB page. | Verify exact local row counts and class counts. | Do not claim CICIDS2017 performance proves deployment robustness. | Current adapter supports random split, CSV/day split, exact-file split. |
| Supervised ML and DL for NIDS | Establish baseline context. | Supervised tabular models, especially RF, are strong baselines for flow data. | `Sharafaldin2018CICIDSAnalysis`; `DLNIDSSurvey2024`; audited baseline papers. | Local RF metrics are currently placeholders unless generated later. | Do not compare QRDQN against missing RF results. | Repo has RF baseline protocol, but `docs/results.md` marks metrics pending. |
| Reinforcement Learning Background | Define RL vocabulary. | RL studies actions, rewards, policies, value functions, and interaction with an environment. | `SuttonBarto2018RL`; `Mnih2015DQN`. | Keep this short; do not write a full RL tutorial. | Do not imply all classification tasks are naturally sequential control. | Repo environment presents flow observations and binary actions. |
| DQN and Distributional RL / QRDQN | Explain algorithm choice. | DQN is foundational value-based deep RL; QRDQN estimates return distributions via quantile regression. | `Mnih2015DQN`; `VanHasselt2016DoubleDQN`; `Bellemare2017DistributionalRL`; `Dabney2018QRDQN`; `SB3ContribQRDQNDocs`. | QRDQN evidence is mostly from RL benchmarks, not NIDS. | Do not claim QRDQN is proven superior for NIDS. | Main project algorithm is QRDQN via SB3-Contrib style tooling. |
| RL/DRL for Intrusion Detection | Place the project in existing RL-IDS work. | RL for IDS already exists, but protocols and datasets vary widely. | `DRLNIDSSurvey2024`; `Gueriani2024DRLIoTNIDS`; audited closest works from `Research2.md`. | Audit original papers before numeric claims. | Do not claim first-ever RL IDS or no prior work. | This thesis is closer to flow-level binary decision/classification-style RL than autonomous cyber defense. |
| Classification-as-RL Formulation | Defend the dataset-as-environment design. | A labeled dataset can be exposed through an RL interface where reward is computed from classification correctness and cost. | `GymnasiumDocs`; DSRL/offline RL analogies; `report-classification-dossier.md`; repo code. | This is an adaptation and may be closer to reward-engineered classification than rich sequential control. | Do not claim it reproduces real network dynamics. | `RLDatasetDefenderEnv` turns labeled samples into PERMIT/BLOCK reward events. |
| Evaluation Pitfalls in ML-based NIDS | Make the critical methodology section strong. | Random splits, leakage, imbalance, unclear preprocessing, and poor cross-domain generalization can inflate metrics. | `Arp2020DosDontsMLSecurity`; `CrossDomain2023NIDS`; `EvalLongTerm2022NIDS`; `DatasetSurvey2025NIDS`; `ProjectResultsSnapshot`. | Do not accuse specific papers unless audited. | Do not use "state of the art" based on raw accuracy tables. | Repo has anti-leakage policy, shuffled-label check, Check C, and exact-file validation code. |
| External Validation and Lab Traffic | Justify lab traffic carefully. | External validation is preferred because public-dataset performance may not transfer. | `CrossDomain2023NIDS`; `MoustafaSlay2015UNSWNB15`; cyber-range sources after audit; `ProjectResultsSnapshot`. | Current Phase 2 artifacts are run-specific and preliminary. | Do not say external validation is complete unless exact artifacts and protocol support it. | Phase 2 is offline inference on extracted flow CSVs, not blocking. |
| Positioning of This Thesis | State the contribution modestly. | The contribution is a reproducible scoped prototype and evaluation protocol, not novelty by absence of prior work. | All above plus `ProjectAgentContext` and `ProjectResultsSnapshot`. | Match claims to current artifacts. | Do not say production-ready, first ever, or real-world proven. | QRDQN, 76-feature schema, 152 observation, internal CICIDS2017 benchmark, planned/preferred external lab validation. |

## Safe Thesis Positioning

RL-based intrusion detection has already been studied, and this thesis should not claim to introduce the first RL IDS. Public-dataset NIDS benchmarks such as CICIDS2017 are common and useful for reproducible experimentation, but the literature also shows that these benchmarks are methodologically fragile when evaluation relies on favorable splits, incomplete preprocessing detail, or no external validation. This thesis studies a binary flow-level `PERMIT`/`BLOCK` formulation using QRDQN over a fixed canonical flow schema. Its contribution is a reproducible, carefully scoped experimental prototype with internal benchmarking, supervised baseline comparison, data-efficiency analysis, leakage-aware evaluation, and planned or preferred external validation with lab-captured traffic. The work should be positioned as an experimental and methodological contribution, not as a production-ready defender or a proof that QRDQN is superior for NIDS.

## Drafting Guardrails

- Keep all section titles in English.
- Use cautious language: "evaluates", "studies", "positions", "provides evidence", "under this protocol".
- Cite project results only with exact run IDs and artifact paths.
- If final results are missing, write "planned", "pending", or omit the claim.
- Keep Spanish raw prose out of the English chapter.
- Do not edit `report/report.tex` until explicitly asked.
