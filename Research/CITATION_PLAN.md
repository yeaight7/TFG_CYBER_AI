# Citation Plan

This is a bibliography planning file, not a final `references.bib`. Do not invent missing metadata. If a source has incomplete authors, year, venue, DOI, or URL, keep it as a TODO.

## 1. Citation Key Convention

Use stable, readable keys:

```text
AuthorYearShortTopic
```

Rules:

- First author surname + year + short topic: `Sharafaldin2018CICIDS2017`.
- For official docs: organization + year + topic, or `OrgNoDateTopic` if no date is available.
- For books: author pair + edition year, e.g. `SuttonBarto2018RL`.
- For project artifacts: `RepoRunId` or `ProjectDocName`, not literature-style keys.
- Keep raw source-map IDs such as `S1` only as internal planning labels, not final citation keys.
- If duplicate keys appear, suffix by topic, not letters: `Sharafaldin2018Dataset`, `Sharafaldin2018Analysis`.

## 2. Must-Cite Source Table

| Citation key | Title | Authors if available | Year | Source type | Topic | Thesis section | Evidence quality | Metadata status | Notes |
|---|---|---|---:|---|---|---|---|---|---|
| `Sharafaldin2018CICIDS2017` | Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization | Iman Sharafaldin, Arash Habibi Lashkari, Ali A. Ghorbani | 2018 | Peer-reviewed paper / dataset paper | CICIDS2017 | Public datasets; CICIDS2017 benchmark | Strong | Partial | Verify final venue/pages from primary source. Raw ID: S1. |
| `Sharafaldin2018CICIDSAnalysis` | A Detailed Analysis of the CICIDS2017 Data Set | Sharafaldin et al. | 2018 | Peer-reviewed chapter/paper | CICIDS2017 analysis, RF baseline | CICIDS2017; supervised baseline | Strong | Partial | Verify exact authors, venue, DOI. Raw ID: S2. |
| `Lashkari2017CICFlowMeter` | Characterization of Tor Traffic Using Time Based Features | Lashkari et al. | 2017 | Peer-reviewed paper / tool-related source | CICFlowMeter and flow features | Flow-based traffic representation | Strong | Partial | Pair with official CICFlowMeter docs if needed. Raw ID: S3. |
| `MoustafaSlay2015UNSWNB15` | UNSW-NB15: A Comprehensive Data Set for Network Intrusion Detection Systems | Nour Moustafa, Jill Slay | 2015 | Peer-reviewed dataset paper | UNSW-NB15 | Public datasets; external validation context | Strong | Partial | Verify DOI/venue details. Raw ID: S4. |
| `Ring2017FlowBasedIDS` | Flow-based intrusion detection: Techniques and challenges | Ring / Sperotto source as listed in raw map | 2017 | Survey | Flow-based NIDS | NIDS; flow-based representation | Strong | Needs verification | Raw map may mix author/title details. Audit before BibTeX. Raw ID: S5. |
| `Sharafaldin2016DatasetFramework` | An Evaluation Framework for Intrusion Detection Dataset | Sharafaldin et al. | 2016 | Methodology paper | Dataset evaluation criteria | Public datasets; evaluation risks | Strong | Partial | Use for dataset-quality criteria. Raw ID: S6. |
| `Sharafaldin2017ReliableDataset` | Towards a Reliable Intrusion Detection Benchmark Dataset | Sharafaldin et al. | 2017 | Methodology paper | IDS benchmark design | Public datasets; evaluation risks | Strong | Partial | Verify journal and DOI. Raw ID: S7. |
| `DatasetSurvey2025NIDS` | Network intrusion datasets: A survey, limitations, and related topic | Unknown in raw research | 2025 | Survey | NIDS dataset taxonomy and limitations | Public datasets | Strong if verified | Needs verification | Keep only if stable publication metadata is confirmed. Raw ID: S8. |
| `CrossDomain2023NIDS` | Explainable Cross-domain Evaluation of ML-based Network Intrusion Detection Systems | Unknown in raw research | 2023 | Methodology / benchmark paper | Cross-domain generalization | Leakage/evaluation risks; external validation | Strong | Needs verification | Essential for distribution-shift claims. Raw ID: S9. |
| `EvalLongTerm2022NIDS` | Evaluation of Machine Learning Algorithms in Network Intrusion Detection Systems | Unknown in raw research | 2022 | Methodology paper / possible arXiv | Long-term performance and overfitting | Leakage/evaluation risks | Moderate | Needs verification | Check if peer-reviewed. Raw ID: S10. |
| `DLNIDSSurvey2024` | Deep Learning-Based Network Intrusion Detection Systems: A Systematic Literature Review | Unknown in raw research | 2024 | Systematic review | DL for NIDS | Supervised ML/DL for NIDS | Strong if verified | Needs verification | Verify exact title, venue, and year. Raw ID: S13. |
| `DRLNIDSSurvey2024` | A Survey for Deep Reinforcement Learning Based Network Intrusion Detection | Yang et al. in raw research | 2024 | Survey / possible preprint | DRL for NIDS | RL/DRL for IDS | Moderate | Needs verification | Do not use as sole evidence for numeric performance. Raw ID: S14. |
| `Gueriani2024DRLIoTNIDS` | Deep Reinforcement Learning for Intrusion Detection in IoT: A Survey | Gueriani et al. in raw research | 2024 | Preprint / survey | DRL for IoT NIDS | RL/DRL for IDS | Moderate | Needs verification | Label as preprint if used. Raw ID: S15. |
| `SuttonBarto2018RL` | Reinforcement Learning: An Introduction | Richard S. Sutton, Andrew G. Barto | 2018 | Textbook | RL fundamentals | Reinforcement Learning background | Strong | Complete | Use for state/action/reward/policy/value definitions. |
| `Mnih2015DQN` | Human-level control through deep reinforcement learning | Volodymyr Mnih et al. | 2015 | Peer-reviewed paper | DQN | RL fundamentals; DQN and QRDQN | Strong | Complete | DOI present in raw research. |
| `VanHasselt2016DoubleDQN` | Deep Reinforcement Learning with Double Q-learning | Hado van Hasselt, Arthur Guez, David Silver | 2016 | Peer-reviewed paper | Double DQN | DQN and QRDQN | Strong | Partial | Verify proceedings details. |
| `Wang2016DuelingDQN` | Dueling Network Architectures for Deep Reinforcement Learning | Ziyu Wang et al. | 2016 | Peer-reviewed paper | Dueling DQN | DQN and QRDQN | Strong | Partial | Use as optional background if dueling is mentioned. |
| `Bellemare2017DistributionalRL` | A Distributional Perspective on Reinforcement Learning | Marc G. Bellemare, Will Dabney, Remi Munos | 2017 | Peer-reviewed paper | Distributional RL | DQN and QRDQN | Strong | Partial | Verify final BibTeX. |
| `Dabney2018QRDQN` | Distributional Reinforcement Learning with Quantile Regression | Will Dabney, Mark Rowland, Marc G. Bellemare, Remi Munos | 2018 | Peer-reviewed paper | QR-DQN | DQN and QRDQN | Strong | Partial | Main QR-DQN source. |
| `GymnasiumDocs` | Gymnasium Documentation | Farama Foundation | No date / versioned docs | Official documentation | RL environment API | Methodology; classification-as-RL | Strong implementation source | Needs verification | Pin version if cited. |
| `SB3ContribQRDQNDocs` | SB3-Contrib QR-DQN documentation | Stable-Baselines3 contributors | No date / versioned docs | Official documentation | QR-DQN implementation | Methodology; DQN and QRDQN | Strong implementation source | Needs verification | Pin installed package version. |
| `CostSensitiveIDSModel` | Toward Cost-Sensitive Modeling for Intrusion Detection and Response | Lee et al. in raw research | Unknown | Technical report / methodology | IDS FP/FN costs | Cost-sensitive reward | Moderate | Needs verification | Use for concept, not modern benchmark performance. Raw ID: S28. |
| `CSEIDS2021CostSensitive` | CSE-IDS: Using cost-sensitive deep learning and ensemble algorithms to handle class imbalance in network-based IDS | Unknown in raw research | 2021 | Peer-reviewed paper | Cost-sensitive NIDS | Cost-sensitive reward; metrics | Moderate | Needs verification | Verify protocol before quantitative use. Raw ID: S25. |
| `Arp2020DosDontsMLSecurity` | Dos and Don'ts of Machine Learning in Computer Security | Daniel Arp et al. | 2020 | Preprint / methodology paper | ML security methodology | Evaluation pitfalls | Strong | Partial | Good support for leakage and evaluation caution. |
| `ProjectAgentContext` | AGENT_CONTEXT - TFG_CYBER_AI | This repository | Current | Maintained repo doc | Implementation facts | Positioning; methodology | Strong for project facts | Complete locally | Use for 76 features, 152 observations, scope, and non-goals. |
| `ProjectResultsSnapshot` | Consolidated Results Snapshot | This repository | Current | Maintained repo doc | Artifact-backed results | Methodology; results framing | Strong for repo artifacts | Complete locally | Use only with exact run IDs. |

## 3. Citation Placement Plan

| Future chapter/subsection | Citation keys to use | Notes |
|---|---|---|
| NIDS and flow-based detection | `Ring2017FlowBasedIDS`, `Lashkari2017CICFlowMeter`, optional IDPS standard | Define NIDS and flow-level representation. Do not over-describe IPS/active blocking. |
| Public datasets | `Sharafaldin2016DatasetFramework`, `Sharafaldin2017ReliableDataset`, `DatasetSurvey2025NIDS`, `MoustafaSlay2015UNSWNB15`, `Tavallaee2009NSLKDD` after audit | Emphasize benchmark utility plus limitations. |
| CICIDS2017 | `Sharafaldin2018CICIDS2017`, `Sharafaldin2018CICIDSAnalysis`, official CIC/UNB page | Match final text to local curated CSV schema. |
| Supervised ML/DL for NIDS | `Sharafaldin2018CICIDSAnalysis`, `DLNIDSSurvey2024`, audited RF / tabular baseline sources | State that RF is a necessary baseline; local RF metrics pending unless produced later. |
| RL fundamentals | `SuttonBarto2018RL`, `Mnih2015DQN` | Define state/action/reward/policy/value function. |
| DQN and QRDQN | `Mnih2015DQN`, `VanHasselt2016DoubleDQN`, `Bellemare2017DistributionalRL`, `Dabney2018QRDQN`, `SB3ContribQRDQNDocs` | QR-DQN rationale only; no superiority claim. |
| RL/DRL for IDS | `DRLNIDSSurvey2024`, `Gueriani2024DRLIoTNIDS`, audited closest works from `Research2.md` | Separate IDS classification-style RL from autonomous cyber defense. |
| Cost-sensitive reward | `CostSensitiveIDSModel`, `CSEIDS2021CostSensitive`, FP/FN cost sources from S26-S29 | Tie reward to asymmetric FP/FN assumptions. |
| Metrics | Dataset/DL surveys, `ProjectResultsSnapshot` | Accuracy, precision, recall, F1, FPR/FNR, confusion matrix. Add AUROC/AUPRC only if generated. |
| Leakage/evaluation risks | `Arp2020DosDontsMLSecurity`, `CrossDomain2023NIDS`, `EvalLongTerm2022NIDS`, `DatasetSurvey2025NIDS` | Use for random split caution, leakage controls, and external validity. |
| External validation | `CrossDomain2023NIDS`, `MoustafaSlay2015UNSWNB15`, `ProjectAgentContext`, `ProjectResultsSnapshot` | Describe lab traffic as planned/preferred external-distribution check unless current artifact supports a specific claim. |
| Autonomous cyber defense | audited ACD survey / Feng2025 if verified | Broader context and future work only. Do not imply implemented active defense. |

## 4. BibTeX TODOs

### Sources needing DOI

- `Ring2017FlowBasedIDS`: confirm exact title/authors/DOI because raw files may mix flow-survey metadata.
- `Sharafaldin2018CICIDSAnalysis`: verify DOI and chapter/proceedings metadata.
- `DatasetSurvey2025NIDS`: verify DOI, authors, and final publication venue.
- `CrossDomain2023NIDS`: verify DOI and exact title.
- `EvalLongTerm2022NIDS`: verify DOI or arXiv ID and publication status.
- DRL closest-work papers in `Research2.md`: add DOI only after original paper audit.

### Sources needing author/year verification

- `DLNIDSSurvey2024` and any raw "DLNIDS-Survey1" entry with 2025/2026 year.
- `DRLNIDSSurvey2024` if the final source differs from raw notes.
- `Gueriani2024DRLIoTNIDS` if cited beyond a preprint mention.
- Cost-sensitive IDS papers S25-S29.
- Autonomous cyber defense sources, especially any 2025/2026 items.

### Suspicious or incomplete sources

- Any entry with "First Author and Others".
- Any DOI placeholder like `10.1007/978-3-XXX`.
- Future-year survey entries without stable official publication pages.
- Scribd, Studocu, LinkedIn, Medium, tutorial pages, and opaque repository links.
- Papers reporting near-perfect accuracy without split methodology, preprocessing, class distribution, and baselines.

### Duplicated citation keys

- Raw files use multiple Sharafaldin/CICIDS keys. Normalize to:
  - `Sharafaldin2018CICIDS2017` for dataset generation.
  - `Sharafaldin2018CICIDSAnalysis` for detailed analysis/baselines.
- Raw files use multiple QR-DQN keys. Normalize to:
  - `Dabney2018QRDQN`.
- Raw files use multiple DRL-NIDS survey labels. Keep only after metadata audit.
- Raw source-map IDs `S1`, `S2`, etc. are planning IDs only. Do not use them in final BibTeX.
