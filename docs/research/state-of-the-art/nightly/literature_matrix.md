# Literature Matrix
## Second revised pass — 2026-05-16

This matrix maps relevant literature to the thesis by topic cluster. It is a working planning
document, not a chapter draft. Each row uses the following tier classification:

| Tier | Meaning |
|---|---|
| **Core** | Must be in final bibliography; thesis makes a direct claim requiring this work |
| **Supporting** | Should be cited if the corresponding section is included in the final chapter |
| **Context** | Background framing; cite only if the relevant topic is introduced |
| **Future** | Mention only in future work section; not required for the main argument |

**bib status:** Confirmed = key exists in `report/references.bib`; CANDIDATE = DOI available in
`Research/Research2.md`, needs bib addition after verification; VERIFY = not yet confirmed, must
not be cited until checked.

---

## Cluster 1 — Flow-Based Traffic Representation

| Work | Key contribution | Tier | bib status | Action / note |
|---|---|---|---|---|
| Sperotto et al. (2010), "An Overview of IP Flow-Based Intrusion Detection," IEEE Comm. Surveys | Defines flow-based IDS; distinguishing properties of flow-level detection | Core | Confirmed `Sperotto2010FlowIDS` | Use as replacement for broken `Ring2017FlowBasedIDS` key in draft |
| Lashkari et al. (2017), CICFlowMeter tool paper | Describes CICFlowMeter feature extraction pipeline; CICIDS2017 relies on this | Core | Confirmed `Lashkari2017CICFlowMeter` | Also locate official CICFlowMeter docs and pin version |
| Ring et al. (2019), "A Survey of Network-Based Intrusion Detection Data Sets," Computers & Security | Surveys NIDS datasets including feature diversity | Context | Confirmed `Ring2019DatasetSurvey` | Use for dataset-comparison content, not flow-methodology claims |
| Sarhan et al. (approx. 2022), "Towards a Standard Feature Set for Network Intrusion Detection System Datasets" | Proposes CICFlowMeter → NetFlow standard mapping; motivates cross-dataset comparability | Supporting | VERIFY `Sarhan2022StandardFeatureSet` | Verify venue, authors, DOI before adding to bib |

---

## Cluster 2 — Public NIDS Datasets

| Work | Key contribution | Tier | bib status | Action / note |
|---|---|---|---|---|
| KDD Cup 1999 dataset | Origin of IDS benchmarking; widely criticised for synthetic nature | Context | Confirmed `KDDCup1999` | Historical context only; do not use for modern performance claims |
| Tavallaee et al. (2009), "A Detailed Analysis of the KDD CUP 99 Data Set" | Describes NSL-KDD improvement over KDDCup99 | Context | Confirmed `Tavallaee2009NSLKDD` | Historical context; thesis treats NSL-KDD as legacy |
| Moustafa & Slay (2015), "UNSW-NB15: A Comprehensive Data Set" | Cyber-range dataset with hybrid attacks; more recent than NSL-KDD | Context | Confirmed `MoustafaSlay2015UNSWNB15` | Cite as alternative benchmark and cyber-range analogy for Phase 2 |
| Sharafaldin et al. (2018), "Toward Generating a New Intrusion Detection Dataset...," ICISSP | Describes CICIDS2017 design, attack scenarios, CICFlowMeter pipeline | Core | Confirmed `Sharafaldin2018CICIDS2017` | Primary dataset citation |
| Sharafaldin et al. (2018), "A Detailed Analysis of the CICIDS2017 Data Set" | CICIDS2017 feature analysis; Random Forest baseline performance | Core | Confirmed partial `Sharafaldin2018CICIDSAnalysis` | Verify full bib entry exists; needed for RF baseline justification |
| Koroniotis et al. (2019), Bot-IoT dataset | IoT-focused NIDS dataset with botnet attacks | Future | Confirmed `Koroniotis2019BotIoT` | Future work context only |
| Alsaedi et al. (2020), ToN-IoT dataset | Multi-device IoT/IIoT dataset | Future | Confirmed `Alsaedi2020ToNIoT` | Future work context only |
| Unknown (approx. 2025), NIDS dataset survey | Surveys 89+ public NIDS datasets and limitations | Supporting | VERIFY `DatasetSurvey2025NIDS` | Not in bib; verify authors, venue, DOI; remove from draft if unverifiable |

---

## Cluster 3 — CICIDS2017 Quality Concerns

| Work | Key contribution | Tier | bib status | Action / note |
|---|---|---|---|---|
| Engelen et al. (approx. 2021), "Troubleshooting an Intrusion Detection Dataset: The CICIDS2017 Case Study" | Reconstructed CICIDS2017 traces; found mislabelled flows, incorrect boundaries, ~25%+ artefact flows | Core | VERIFY `Engelen2021CICIDSIssues` | Authors likely Engelen, Rimmer, Latré; venue likely 2021 security conference; verify DOI |
| Lanvin et al. (approx. 2023), "Faulty use of the CIC-IDS 2017 Dataset in Information Security Research" | Argues summarised CICIDS2017 is used beyond its representativeness | Supporting | VERIFY `Lanvin2023CICIDSFaulty` | Verify authors, venue (likely 2023 security conf.), DOI before using |

---

## Cluster 4 — Supervised ML and DL for NIDS

| Work | Key contribution | Tier | bib status | Action / note |
|---|---|---|---|---|
| Scarfone & Mell (2007), NIST SP 800-94 | Authoritative NIDS/HIDS/IPS taxonomy | Core | Confirmed `ScarfoneMell2007` | Use for Section 1 NIDS definition |
| Liu & Lang (2019), "Machine Learning and Deep Learning Methods for IDS: A Survey" | General ML/DL IDS survey | Context | Confirmed `LiuLang2019IDSSurvey` | Use if no more recent confirmed survey is available |
| Unknown (approx. 2024), "Deep Learning-Based NIDS: A Systematic Literature Review" | Surveys DL architectures for NIDS | Supporting | VERIFY `DLNIDSSurvey2024` | Not in bib; verify before use; interim: use `LiuLang2019IDSSurvey` |
| Arp et al. (2020), "Dos and Don'ts of Machine Learning in Computer Security" | 10 evaluation pitfalls: leakage, split design, deployment gap | Core | Confirmed `Arp2020DosDontsMLSecurity` | Cite throughout evaluation methodology sections |
| Axelsson (2000), "The Base-Rate Fallacy and the Difficulty of Intrusion Detection," ACM TISSEC | Shows high-accuracy classifiers produce many FP at realistic attack base rates | Core | VERIFY `Axelsson1999BaseRate` | Likely DOI 10.1145/357802.357804; confirm and add to bib |

---

## Cluster 5 — Reinforcement Learning Foundations

| Work | Key contribution | Tier | bib status | Action / note |
|---|---|---|---|---|
| Sutton & Barto (2018), "Reinforcement Learning: An Introduction," MIT Press | Canonical RL definitions | Core | Confirmed `SuttonBarto2018RL` | First citation for any RL concept |
| Mnih et al. (2015), "Human-Level Control through Deep Reinforcement Learning," Nature | Introduces experience replay and target networks; DQN | Core | Confirmed `Mnih2015DQN` | Algorithm background |
| van Hasselt, Guez & Silver (2016), "Deep RL with Double Q-learning," AAAI | Addresses DQN overestimation bias | Core | Confirmed `VanHasselt2016DoubleDQN` | Algorithm background |
| Wang et al. (2016), "Dueling Network Architectures for Deep RL," ICML | Separates state-value and advantage estimation | Context | Confirmed `Wang2016DuelingDQN` | Include only if dueling is used in implementation |
| Bellemare, Dabney & Munos (2017), "A Distributional Perspective on RL," ICML | Introduces distributional RL and C51 | Core | Confirmed `Bellemare2017DistributionalRL` | Theoretical ancestor of QRDQN |
| Dabney et al. (2018), "Distributional RL with Quantile Regression," AAAI | Introduces QR-DQN | Core | Confirmed `Dabney2018QRDQN` | Primary algorithm paper |
| Farama Foundation, Gymnasium documentation (versioned) | RL environment API | Core | Confirmed `GymnasiumDocs` | Pin version from `uv.lock` or `requirements.txt` |
| SB3-Contrib contributors, QRDQN documentation (versioned) | Implementation reference | Core | Confirmed `SB3ContribQRDQNDocs` | Pin version from lockfile |

---

## Cluster 6 — RL and DRL for Intrusion Detection

| Work | Key contribution | Tier | bib status | Action / note |
|---|---|---|---|---|
| Yang et al. (2024), "A Survey for Deep RL Based Network Intrusion Detection," arXiv | Comprehensive DRL-IDS survey; identifies common patterns and open problems | Core | Confirmed `Yang2024DRLNIDSSurvey` (arXiv) | Cite to show field exists; audit publication venue |
| Gueriani et al. (2024), "Deep RL for Intrusion Detection in IoT: A Survey," arXiv | DRL-IDS survey focused on IoT | Context | Confirmed `Gueriani2024DRLIoTIDSSurvey` (arXiv) | Use for broader RL-IDS scope; note IoT focus |
| López-Martín et al. (2020), "Application of Deep RL to Intrusion Detection for Supervised Problems," Expert Systems with Applications | Explicitly reformulates IDS as RL over labelled records; compares DQN, Double DQN, PG, AC on NSL-KDD/AWID | Core | CANDIDATE `LopezMartin2020DRLIDS` — DOI 10.1016/j.eswa.2019.112963 | Closest methodological antecedent; verify DOI and add to bib |
| López-Martín et al. (2021), "Network Intrusion Classification Using Deep Learning and RL," IEEE Access | Extends to offline RL with RBF network; evaluates on CICIDS2017 and UNSW-NB15 | Core | CANDIDATE `LopezMartin2021RBFOfflineRL` — DOI 10.1109/ACCESS.2021.3127689 | Uses CICIDS2017 under similar formulation; verify DOI |
| Ren et al. (2022), "Intrusion Detection Based on Deep RL," Scientific Reports | DRL for joint feature selection and classification on CSE-CIC-IDS2018 | Supporting | CANDIDATE `Ren2022IDRDRL` — DOI 10.1038/s41598-022-19366-3 | Different framing (feature selection); verify DOI |
| Alavizadeh et al. (approx. 2022), "Deep Q-Learning Based RL Approach for Network Intrusion Detection" | Applies DQN to CICIDS2017; relevant direct precedent | Context | VERIFY `Alavizadeh2022DQLearningNIDS` | Verify authors, venue, DOI; only cite if confirmed |
| He et al. (approx. 2024), "Reinforcement Learning Meets Network Intrusion Detection" | RL adapted to novel attack patterns in NIDS | Context | VERIFY `He2024RLmeetsNIDS` | Verify stability before citing |
| Alam et al. (approx. 2025), "Adaptive Defense: Zero-Day Attack Detection in NIDS with DRL" | DRL for zero-day detection | Future | VERIFY `Alam2025ZeroDayDRL` | Very recent; verify peer-review status; use only if confirmed |

---

## Cluster 7 — Cost-Sensitive Evaluation

| Work | Key contribution | Tier | bib status | Action / note |
|---|---|---|---|---|
| Lee et al. (2002), "Toward Cost-Sensitive Modeling for Intrusion Detection and Response," Journal of Computer Security | Formalises asymmetric IDS cost structure; FN costs differ from FP costs | Core | Confirmed `Lee2002CostSensitiveIDS` | Note 2002 vintage; use for concept only, not modern benchmarks |
| Axelsson (2000), ACM TISSEC | See Cluster 4 | Core | VERIFY `Axelsson1999BaseRate` | Also relevant here; motivates FP cost concern |
| Unknown (approx. 2021), "CSE-IDS: Using Cost-Sensitive Deep Learning..." | Cost-sensitive deep learning for NIDS | Context | VERIFY `CSEIDS2021CostSensitive` | Verify before citing; if unverifiable, omit |

---

## Cluster 8 — Evaluation Methodology and Pitfalls

| Work | Key contribution | Tier | bib status | Action / note |
|---|---|---|---|---|
| Arp et al. (2020), "Dos and Don'ts of ML in Computer Security" | See Cluster 4 | Core | Confirmed `Arp2020DosDontsMLSecurity` | Central methodology citation |
| Sommer & Paxson (2010), "Outside the Closed World: On Using ML for Network Intrusion Detection," IEEE S&P | Foundational argument that NIDS is harder than benchmarks suggest; base-rate and distribution-shift | Core | VERIFY `SommerPaxson2010ClosedWorld` | Likely DOI 10.1109/SP.2010.25; confirm and add to bib |

---

## Cluster 9 — Cross-Domain Generalisation

| Work | Key contribution | Tier | bib status | Action / note |
|---|---|---|---|---|
| Apruzzese et al. (2022), "The Cross-Evaluation of ML-based NIDS," IEEE Transactions | Shows cross-dataset evaluation is underused; demonstrates hidden risks in intra-dataset performance | Core | Confirmed `Apruzzese2022CrossEvaluationNIDS` | Use as replacement for broken `CrossDomain2023NIDS` key |
| Layeghy & Portmann (2023), "Explainable Cross-Domain Evaluation of ML-Based NIDS," Computers and Electrical Engineering | Shows cross-domain performance degrades; explains via feature importance | Core | Confirmed `Layeghy2023CrossDomainNIDS` | Use as replacement for broken `EvalLongTerm2022NIDS` key |
| Cantone et al. (approx. 2024), "On the Cross-Dataset Generalisation of ML for NIDS" | Near-perfect intra-dataset accuracy degrades toward chance across datasets | Supporting | VERIFY `Cantone2024CrossDataset` | If unverifiable, Apruzzese and Layeghy support the same claim |

---

## Cluster 10 — Data Efficiency

| Work | Key contribution | Tier | bib status | Action / note |
|---|---|---|---|---|
| Di Monda et al. (approx. 2024), "Few-Shot Class-Incremental Learning for NIDS" | Addresses learning new attack families from minimal examples | Context | VERIFY `DiMonda2024FewShotNIDS` | Provides a hook for data-efficiency framing; verify before citing |

---

## Cluster 11 — Repository Artifacts

| Work | Key contribution | Tier | bib status | Action / note |
|---|---|---|---|---|
| This repository, AGENT_CONTEXT.md | 76-feature schema, 152-observation, PERMIT/BLOCK space, anti-leakage policy, Phase 2 scope | Core | Confirmed `ProjectAgentContext` | Cite for all project-specific factual claims |
| This repository, docs/results.md | Artifact-backed results: QRDQN runs, Check A/B/C, Phase 2 artifacts | Core | Confirmed `ProjectResultsSnapshot` | Always cite with exact run ID when reporting metrics |

---

## Priority Summary

| Priority | Item |
|---|---|
| **Blocking** | Fix 7 citation key mismatches in draft (see `state_of_art_expansion.md` Administrative section) |
| **High — add to bib** | `LopezMartin2020DRLIDS` (DOI available), `LopezMartin2021RBFOfflineRL`, `Ren2022IDRDRL` |
| **High — verify and add** | `SommerPaxson2010ClosedWorld`, `Axelsson1999BaseRate`, `Engelen2021CICIDSIssues`, `Lanvin2023CICIDSFaulty` |
| **High — confirm bib entry** | `Sharafaldin2018CICIDSAnalysis` (verify full entry exists, not just citation plan entry) |
| **Medium — verify** | `Sarhan2022StandardFeatureSet`, `DiMonda2024FewShotNIDS`, `Cantone2024CrossDataset`, `DatasetSurvey2025NIDS`, `DLNIDSSurvey2024` |
| **Low — verify or omit** | `Alavizadeh2022DQLearningNIDS`, `He2024RLmeetsNIDS`, `Alam2025ZeroDayDRL`, `CSEIDS2021CostSensitive` |
