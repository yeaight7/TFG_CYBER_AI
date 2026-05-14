## 1. Executive source map

Below is a high-level taxonomy of the literature, with representative sources per theme.

### 1.1 Flow‑based NIDS and classical NIDS

- **Concepts and challenges of flow‑based NIDS**
    - Umer et al., “Flow-based intrusion detection: Techniques and challenges” – core survey of NetFlow/IPFIX-based IDS, feature types, and open issues like sampling and scalability.[^1][^2][^3]
    - Ajaeiya et al., “Flow-based Intrusion Detection System for SDN” – lightweight flow-based IDS for SDN, periodic polling of OpenFlow switches.[^4]
    - Evaluation of ML on traffic flows: “Evaluation of Machine Learning Techniques for Traffic Flow-Based NIDS” (flow-based ML on CIC flows).[^5]
- **General AI/ML-based NIDS surveys**
    - Asadullah et al., “Systematic and Comprehensive Survey of Recent Advances in IDS Using ML: DL, Datasets, and Attack Taxonomy” – broad survey of ML/DL IDS, datasets, metrics, and open problems.[^6]
    - Sayeeduddin \& Ranga, “Network intrusion detection system: a survey on artificial intelligence-based techniques” – AI-based NIDS 2016–2021, noting overuse of old datasets.[^7]
    - Xiao et al., “A Survey on Network Intrusion Detection Based on Deep Learning” – DL architectures, datasets, evaluation methods.[^8]


### 1.2 Public cybersecurity datasets (flow‑based and classic)

- **CIC / CSE‑CIC family**
    - Sharafaldin et al., “Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization” – methodology and rationale behind CICIDS2017; flow-based, realistic traffic, seven attack families.[^9][^10][^11]
    - UNB / AWS documentation of **CSE‑CIC‑IDS2018** – multi-day enterprise-style scenario with 7 attack scenarios, 80 CICFlowMeter features.[^12][^13]
    - CICDDoS2019 original paper and dataset description – dedicated DDoS flow dataset sharing features with CICIDS2017/2018.[^14][^15]
- **UNSW / IoT‑oriented datasets**
    - Moustafa \& Slay, “UNSW-NB15: A comprehensive data set for network intrusion detection systems” – hybrid real/synthesized traffic from UNSW Cyber Range Lab; 49 features, 9 attack types.[^16][^17][^18][^19]
    - Koroniotis et al., “Bot-IoT dataset” – realistic IoT botnet dataset from UNSW Cyber Range, large-scale flow records.[^20][^21]
    - Moustafa, TON_IoT datasets – distributed IoT/IIoT cyber range datasets with multi-layer data (network, host, telemetry).[^22][^23][^24]
- **Legacy datasets and critiques**
    - Tavallaee et al., “A detailed analysis of the KDD CUP 99 data set” – identifies redundancy and biases; introduces NSL‑KDD.[^25]
    - NSL‑KDD description (UNB) – improved benchmark, reduced redundancy, still limited realism.[^26][^27][^28][^29]
    - UNSW evaluation paper, “The evaluation of Network Anomaly Detection Systems: Statistical analysis of the UNSW‑NB15 data set and comparison with KDD99” – comparative analysis, shows UNSW‑NB15 more challenging than KDD‑99.[^30]
    - DARPA 1998/1999 offline evaluation datasets – early benchmark testbeds for NIDS.[^31][^32][^33][^34]
    - “A Review of the Advancement in Intrusion Detection Datasets” – survey of datasets including CICIDS2017 and CSE‑CIC‑IDS2018.[^35]


### 1.3 ML / DL for NIDS (supervised)

- **DL architectures on CIC/UNSW/NSL-KDD**
    - CNN-based NIDS survey (2022) – taxonomy of CNN‑IDS, datasets, metrics, notes difficulty comparing results across heterogeneous datasets.[^36]
    - DL-IDS with CNN-LSTM on CICIDS2017 – very high reported accuracies; good example of DL on flow data but methodologically optimistic.[^37]
    - Various CICIDS2017 CNN/LeNet autoencoder works, often with 97–99% accuracy under random splits.[^38][^39][^40][^41][^42]
- **Evaluation/robustness issues across datasets**
    - MDPI “Choice of Training Data and the Generalizability of ML Models for NIDS” – cross-dataset training (UNSW‑NB15, CSE‑CIC‑IDS2018, BoT‑IoT, ToN‑IoT), demonstrates poor generalization and highlights need for external validation.[^43]
    - Papers implementing ensembles and feature selection on NSL‑KDD and UNSW‑NB15 with very high accuracy but no external validation, illustrating methodological pitfalls.[^44][^45][^46]


### 1.4 DRL for intrusion detection (dataset‑as‑environment)

- **RL / DRL IDS surveys**
    - Gueriani et al., “Deep Reinforcement Learning for Intrusion Detection in IoT: A Survey” – taxonomy of DRL‑based IDS in IoT, categorization by environment/tasks and DRL algorithms.[^47]
    - Cevallos et al., “Deep Reinforcement Learning for intrusion detection in Internet of Things: Best practices, lessons learnt, and open challenges” – methodologically focused survey; discusses datasets (NSL‑KDD, CIC, UNSW, BoT‑IoT), evaluation practices, and open challenges.[^48][^49]
    - 2026 survey “A Survey for Deep Reinforcement Learning Based Network Intrusion …” – dedicated to DRL for NIDS (general, not only IoT).[^50]
- **Dataset-as-environment formulations**
    - Caminero et al., “Adversarial environment reinforcement learning algorithm for intrusion detection” – explicitly replaces live environment with a simulated one sampling from labelled datasets; environment acts as a second adversarial agent that resamples difficult examples.[^51][^52]
    - AE‑SAC model, “A soft actor-critic reinforcement learning algorithm for network intrusion detection” – classifier agent and environment agent co‑trained; environment resamples imbalanced training data and shapes rewards per class.[^53]
    - DRL IDS theses and DQN‑based IDS (e.g., “Network Intrusion Detection Using Deep Reinforcement Learning”; “Deep Q-learning intrusion detection system (DQ‑IDS)”) – typically treat each record as a state with action = predicted class, similar to your flow‑as‑observation setup.[^54][^55][^53]


### 1.5 RL‑based autonomous / adaptive cyber defense

- **Conceptual and survey‑style works**
    - CSET / Turing report “Autonomous Cyber Defense” – policy-oriented survey; emphasizes RL as main paradigm, overviews cyber “gyms” (CyberBattleSim, CAGE) and requirements like adaptability, auditability, transferability.[^56]
    - Umer et al., “Packet-Level and Flow-Level NIDS Based on RL and Adversarial Training” – RL agents (DQL, policy gradient) trained on CICDDoS2019 at packet and flow level; includes adversarial training against a sample agent.[^57]
- **Model-based adaptive defense**
    - Hu et al., “Adaptive Cyber Defense Against Multi-Stage Attacks Using Learning-Based POMDP” – Bayesian attack graphs + RL/POMDP to choose cost-effective defense actions.[^58][^59]
    - RL honeypot engagement (Semi‑MDP) – RL to manage honeypot interaction vs risk and cost.[^60]
    - Multi‑agent and game-theoretic RL for moving target defense and cyber attack–defense games (Markov games, multi-agent RL) – multiple works applying MARL to attack/defense in networks and smart grids.[^61][^62]
- **Cyber ranges and RL environments**
    - Microsoft CyberBattleSim and derived multi-agent environments for RL cyber operations (blue/red agents).[^63][^64]
    - Cyber range design papers for evaluating AI/ML NIDS in realistic enterprise networks.[^65]


### 1.6 Evaluation methodology and dataset quality

- **Dataset critique and usage surveys**
    - Tavallaee et al. (NSL‑KDD) and later reviews of KDD99/NSL‑KDD usage – highlight redundancy, outdated attacks, unrealistic traffic, and tendency towards inflated accuracy.[^66][^25]
    - UNSW evaluation paper comparing UNSW‑NB15 vs KDD99 – emphasizes increased difficulty and realistic traffic, but still synthetic benign traffic via traffic generator.[^67][^30]
    - Dataset survey papers that highlight overuse of old datasets and lack of external validation.[^6][^7][^35]
- **Generalization and external validation**
    - Generalizability paper on NIDS training data choice (UNSW‑NB15, CSE‑CIC‑IDS2018, BoT‑IoT, ToN‑IoT) – demonstrates that high in‑dataset metrics poorly transfer across datasets, especially for attack classes.[^43]
    - Testbed and cyber range works (Algorizmi, ICS testbeds, Oak Ridge cyber range) — demonstrate principled generation of lab datasets and emphasize the need for realistic traffic and repeatable experiments.[^68][^69][^70][^71][^65]

These clusters give you natural Section 2 \& 3 groupings and also clear lines for your “research gap” argument: distributional DRL for flow-based NIDS with rigorous evaluation and external validation is rare to nonexistent.[^47][^48][^50][^53]

***

## 2. Source matrix

(You can extend this matrix; here is a curated core of ~30 high‑value entries.)


| ID | Citation key | Full reference | Year | Source type | Topic | Algorithm / method | Dataset(s) | Task | Evaluation metrics | Split methodology | Code available? | Main contribution | Main limitation | Relevance to thesis | Why it matters | Link / DOI |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| S1 | Sharafaldin2018-CICIDS2017 | I. Sharafaldin, A. H. Lashkari, A. A. Ghorbani, “Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization,” ICISSP.[^9][^10][^11] | 2018 | Dataset paper | NIDS datasets | Profile-based traffic generation, CICFlowMeter features | CICIDS2017 | Multi‑class attack classification | Accuracy, precision, recall, F1 | Per-scenario days; many papers later use random splits | No (dataset tools partially) | Defines CICIDS2017 methodology; characterizes flows and attacks | Limited external validation; benign traffic may still be structured | High | Your main dataset; foundational for describing CIC family and their limitations | [Link](https://www.unb.ca/cic/datasets/ids-2017.html)[^12][^9] |
| S2 | Moustafa2015-UNSWNB15 | N. Moustafa, J. Slay, “UNSW-NB15: A comprehensive data set for network intrusion detection systems (UNSW‑NB15 network data set),” MilCIS.[^16][^17][^18][^19] | 2015 | Dataset paper | NIDS datasets | IXIA-based cyber range, feature engineering | UNSW‑NB15 | Multi‑class \& binary classification | Accuracy, detection rate, FPR | Predefined train/test; sometimes reused with random splits | No | Introduces modern hybrid dataset with 9 attack types and 49 features | Benign traffic synthetic; specific to one lab setup | High | Key non‑CIC benchmark; good for cross‑dataset generalization arguments | [DOI](https://doi.org/10.1109/MilCIS.2015.7348942)[^16] |
| S3 | Tavallaee2009-NSLKDD | M. Tavallaee et al., “A detailed analysis of the KDD CUP 99 data set,” CISDA.[^25] | 2009 | Dataset / critique | Dataset flaws | Statistical analysis, NSL‑KDD proposal | KDD99, NSL‑KDD | Multi‑class intrusion classification | Accuracy, DR, FAR | Predefined train/test | No | Shows redundancy and bias in KDD99; proposes NSL‑KDD | Dataset still synthetic and outdated; limited attacks | High | Canonical citation for discussing dataset bias, redundancy, and misleading accuracy | [DOI](https://doi.org/10.1109/CISDA.2009.5356528)[^25] |
| S4 | Asadullah2023-MLIDS-Survey | M. Asadullah et al., “A Systematic and Comprehensive Survey of Recent Advances in IDS Using Machine Learning: Deep Learning, Datasets, and Attack Taxonomy.”[^6] | 2023 | Survey | ML/DL IDS | Survey of ML/DL methods | Multiple (KDD, NSL‑KDD, UNSW‑NB15, CIC family, etc.) | Binary/multi‑class/anomaly | Accuracy, precision, recall, F1, ROC‑AUC | Mostly reports what primary papers do (often random splits) | No | Up‑to‑date overview of ML/DL IDS, datasets, and attack taxonomy | Does not deeply audit methodology of each paper | High | Good umbrella survey to justify focusing on ML/DL NIDS and to cite dataset popularity | [DOI](https://doi.org/10.1155/2023/6048087)[^6] |
| S5 | Sayeeduddin2022-AI-NIDS-Survey | H. M. Sayeeduddin, T. R. Babu, “Network intrusion detection system: a survey on artificial intelligence‐based techniques,” Expert Systems.[^72][^7] | 2022 | Survey | AI-based NIDS | Survey of AI approaches | Multiple | Binary \& multi‑class | Accuracy, FAR, DR | Summarizes primary works | No | Highlights shift to deep learning and continued use of old datasets | Limited focus on flow-based NIDS specifics | Medium | Helps motivate AI-based NIDS, points out overuse of KDD/NSL‑KDD | [DOI](https://doi.org/10.1111/exsy.13066)[^7] |
| S6 | Umer2017-FlowIDS-Survey | M. F. Umer, M. Sher, Y. Bi, “Flow-based intrusion detection: Techniques and challenges,” Computers \& Security 70.[^1][^2][^3] | 2017 | Survey | Flow-based NIDS | Taxonomy of flow-based techniques | Various flow datasets (NetFlow/IPFIX, CIDDS, etc.) | Anomaly \& misuse detection | Accuracy, DR, FPR | Varies; survey level | No | Comprehensive analysis of flow-based NIDS, challenges like sampling, scalability | Pre‑deep‑learning era; limited DRL discussion | High | Directly frames your focus on flow‑based NIDS and network flows as observations | [DOI](https://doi.org/10.1016/j.cose.2017.05.009)[^2] |
| S7 | Nguyen2022-BERT-FlowNIDS | L. G. Nguyen, K. Watabe, “Flow-based Network Intrusion Detection Based on BERT Masked Language Model,” CoNEXT’22 workshop.[^73][^74] | 2022 | Method paper | Flow-based NIDS | BERT masked language model on flow sequences | CICIDS2017, cross-domain data | Multi‑class / binary | Accuracy, F1 | Train/test across domains (domain adaptation) | Unknown | Applies NLP/Transformer to flow sequences, targets domain adaptation | Limited explainability; offline evaluation only | Medium | Good contrast to RL: alternative to handle domain shift in flow NIDS | [arXiv](https://arxiv.org/abs/2306.04920)[^74] |
| S8 | Xiao2021-DL-IDS-Survey | J. Xiao et al., “A Survey on Network Intrusion Detection Based on Deep Learning,” Frontiers of Data and Computing.[^8] | 2021 | Survey | DL NIDS | DL taxonomy (CNN, RNN, AE, etc.) | NSL‑KDD, UNSW‑NB15, CICIDS2017 etc. | Binary \& multi‑class | Accuracy, FAR, F1 | Reports primary methods | No | Summarizes DL architectures and pre‑processing pipelines for NIDS | Limited coverage of RL | High | Useful for background on DL architectures and feature pipelines your thesis can reuse | [Link](http://www.jfdc.cnic.cn/EN/10.11871/jfdc.issn.2096-742X.2021.03.006)[^8] |
| S9 | Asadullah2022-CNN-IDS | “A Survey of CNN-Based Network Intrusion Detection,” Applied Sciences.[^36] | 2022 | Survey | DL NIDS (CNN) | CNN architectures | Various, including CICIDS2017, UNSW‑NB15 | Binary \& multi‑class | Accuracy, precision, recall | Often random splits, 70/30 etc. | No | Deep dive into CNN-based IDS including flow-shaped inputs | Does not solve evaluation issues; results not cross-comparable | Medium | Good to contextualize DL baselines versus your RL approach | [DOI](https://doi.org/10.3390/app12168162)[^36] |
| S10 | Asadullah2020-DatasetsReview | “A Review of the Advancement in Intrusion Detection Datasets,” Procedia Computer Science.[^35] | 2020 | Survey | Datasets | Dataset taxonomy | KDD, NSL‑KDD, UNSW‑NB15, CICIDS2017, CSE‑CIC‑IDS2018, etc. | N/A | N/A | N/A | No | Systematic overview of IDS datasets, pros/cons | Pre‑BoT‑IoT/ToN‑IoT; less IoT focus | High | Use to justify dataset choice and discuss dataset evolution and gaps | [DOI](https://doi.org/10.1016/j.procs.2020.03.270)[^35] |
| S11 | Hu2020-POMDP-Defense | Z. Hu, M. Zhu, P. Liu, “Adaptive Cyber Defense Against Multi-Stage Attacks Using Learning-Based POMDP,” ACM TISSEC.[^58][^59] | 2020 | Method paper | Adaptive defense | RL with Bayesian attack graph (POMDP) | Simulated enterprise networks | Cyber defense (action selection) | Cost-effectiveness, expected utility | Simulated episodes; no real traffic | No | Formal RL formulation of adaptive defense with partial observability | Abstracted model, not flow-based; no real network flows | High | Canonical for “adaptive RL-based cyber defense” framing | [DOI](https://doi.org/10.1145/3418897)[^59] |
| S12 | Huang2019-RL-Honeypot | L. Huang, Q. Zhu, “Adaptive Honeypot Engagement through RL of SMDPs,” arXiv / conference.[^60] | 2019 | Method paper | Active defense | Semi‑MDP + RL policies | Simulated honeynet | Cyber defense policy | Expected utility, penetration probability | Episodic simulation | No | RL agent chooses honeypot engagement policies balancing reward and risk | Specialized to honeypots; not IDS | Medium | Shows RL-based active defense beyond passive detection; supports your “no real-time blocking yet” discussion | [arXiv](https://arxiv.org/abs/1906.12182)[^60] |
| S13 | Umer2022-RL-IDS | “Packet-Level and Flow-Level NIDS Based on RL and Adversarial Training,” Algorithms.[^57] | 2022 | Method paper | RL for IDS | DQL and policy gradient with adversarial training | CICDDoS2019 | Binary / family classification | Accuracy, detection rate | Random split; separate packet vs flow experiments | Probably no | Designs RL-based IDS at packet and flow level, plus adversarial training | Limited external validation; CICDDoS2019 only | High | Directly relevant RL–IDS method on CIC‑family dataset; good for comparison | [DOI](https://doi.org/10.3390/a15120453)[^57] |
| S14 | Caminero2019-AE-RL-IDS | G. Caminero et al., “Adversarial environment reinforcement learning algorithm for intrusion detection.”[^51][^52] | 2019 | Method paper | RL for IDS | Adversarial environment RL; environment resamples records | NSL‑KDD (and possibly others) | Multi‑class classification | Accuracy, DR | Dataset used as environment; random sampling | Unknown | Explicit dataset-as-environment RL with adversarial resampling | Evaluated only on legacy datasets; no flow-based CIC data | High | This is the clearest precedent for treating dataset rows as environment states | [PDF](https://uvadoc.uva.es/handle/10324/54301)[^52] |
| S15 | AE-SAC2023 | “A soft actor-critic reinforcement learning algorithm for network intrusion detection,” Computers \& Security.[^53] | 2023 | Method paper | RL for IDS | AE‑SAC (Soft Actor-Critic with environment agent) | NSL‑KDD, AWID | Multi‑class classification | Accuracy, F1 | Dataset-as-environment with adversarial environment resampling | Unknown | Sophisticated DRL IDS with adversarial environment for imbalance | Uses non‑flow datasets; very high reported accuracies; no external validation | High | Shows modern DRL with dataset-as-environment paradigm; gap to flow-based CIC data | [DOI](https://doi.org/10.1016/j.cose.2023.103580) (via abstract)[^53] |
| S16 | Sanusi2023-DRL-IDS-Thesis | H. T. Sanusi, “Network Intrusion Detection Using Deep Reinforcement Learning,” MSc thesis.[^54] | 2023 | Method paper (thesis) | DRL IDS | DQN-based IDS | NSL‑KDD | Multi‑class | Accuracy, loss | Train/test splits on NSL‑KDD | Likely no | Concrete DQN architecture for IDS | Old dataset, no cross‑dataset tests | Medium | Architectural inspiration for DQN/QRDQN baseline; emphasises DRL benefits and challenges | [PDF](https://digitalcommons.georgiasouthern.edu/etd/3809)[^54] |
| S17 | DQIDS2022 | “Deep Q-learning intrusion detection system (DQ‑IDS),” Computers journal.[^55] | 2022 | Method paper | DRL IDS | DQN for intrusion detection | Likely UNSW‑NB15, NSL‑KDD (from abstract) | Binary/multi‑class | Accuracy, F1 | Random splits | Unknown | One of few named DQN‑IDS systems | May lack realistic evaluation; only synthetic datasets | High | Names and exemplifies the DQN family you extend with QRDQN | [DOI](https://doi.org/10.3390/computers11030041)[^55] |
| S18 | Gueriani2024-DRL-IDS-Survey | A. Gueriani et al., “Deep Reinforcement Learning for Intrusion Detection in IoT: A Survey.”[^47] | 2024 | Survey | DRL IDS (IoT) | Taxonomy of DRL IDS | NSL‑KDD, CICIoT datasets, UNSW‑NB15, etc. | Binary/multi‑class | Accuracy, recall, precision, FNR, FPR, F‑measure | Summarizes; many primary works use random splits | No | Focused survey of DRL‑IDS, including DQN variants and datasets | IoT-specific; limited to IoT-type environments | High | Key up‑to‑date DRL IDS survey; shows rarity of distributional RL like QRDQN | [arXiv](https://arxiv.org/abs/2405.20038)[^47] |
| S19 | Cevallos2023-DRL-IDS-BestPractices | J. F. Cevallos et al., “Deep Reinforcement Learning for intrusion detection in Internet of Things: Best practices, lessons learnt, and open challenges,” Computer Networks.[^48][^49] | 2023 | Survey | DRL IDS (IoT) | Critical survey and best practices | Multiple | Binary/multi‑class | Accuracy, F1, etc. | Discusses proper validation, but notes most works rely on random splits | No | Critically reviews DRL‑IDS design choices and evaluation pitfalls | Focused on IoT; not flow‑specific | High | Excellent source for methodological warnings and “best practices” you can adopt | [DOI](https://doi.org/10.1016/j.comnet.2023.109297) (via index)[^48] |
| S20 | CSET2023-AutonomousCyberDefense | CSET \& Turing, “Autonomous Cyber Defense,” policy/technical report.[^56] | 2023 | Survey / report | Autonomous defense | Conceptual RL cyber defense | Various cyber “gyms” (CyberBattleSim, CAGE) | Cyber defense | N/A | N/A | No | Synthesizes state of autonomous cyber defense, emphasizes RL and cyber ranges | Non‑academic; not focused on flow NIDS | High | Good for framing your thesis in broader autonomous defense context and for claims about limitations | [Link](https://cset.georgetown.edu/publication/autonomous-cyber-defense/)[^56] |
| S21 | Umer2022-FlowRL-IDS | See S13 (same). | 2022 | Method | RL IDS | DQL, policy-gradient | CICDDoS2019 | Binary/family | Accuracy | Random splits | No | RL at packet and flow level | Dataset limited to DDoS | High | Shows RL agent at flow-level, but not general flow-based NIDS on mixed traffic | [DOI](https://doi.org/10.3390/a15120453)[^57] |
| S22 | AssemblingCyberRange2022 | J. Nichols et al., “Assembling a Cyber Range to Evaluate AI/ML Security Tools,” arXiv.[^65] | 2022 | Testbed / methodology | Cyber ranges | Cyber range for AI/ML NIDS | Realistic enterprise traffic in lab | NIDS evaluation | Detection rate, resource usage | Controlled experiments, repeated scenarios | No | Presents design of cyber range used to evaluate AI/ML NIDS | No RL; security tools abstracted | Medium | Good reference for arguing that lab‑captured traffic is a standard way to externally validate NIDS | [arXiv](https://arxiv.org/abs/2201.08473)[^65] |
| S23 | Algorizmi2010-Testbed | “Algorizmi: A Configurable Virtual Testbed to Generate Datasets for Offline Evaluation of IDS,” PhD thesis / paper.[^68][^71] | 2010 | Testbed | Lab datasets | Virtualized testbed | Synthetic lab networks | NIDS evaluation | Accuracy, DR, FPR | Controlled experiments | No | Early configurable testbed for IDS dataset generation | Old, pre‑modern attacks | Medium | Historical justification for using lab testbeds and offline evaluation | [Link](https://uwspace.uwaterloo.ca/items/f774f0c1-9637-48c8-9352-d1623ffef423)[^71] |
| S24 | ICS-Testbed2021 | “Industrial Datasets with ICS Testbed and Attack Detection Using Machine Learning Techniques,” IASC.[^70] | 2021 | Testbed/dataset | ICS NIDS | ML classifiers on ICS testbed traffic | ICS testbed dataset | Binary/multi‑class | Accuracy, precision, recall, F1 | Train/test from lab capture | Unknown | Presents real ICS cyber‑physical testbed and dataset | Domain specific to ICS | Medium | Example of lab‑captured traffic as external validation for NIDS | [DOI](https://doi.org/10.32604/iasc.2021.015806)[^70] |
| S25 | Generalizability2025-TrainingData | “The Choice of Training Data and the Generalizability of ML Models for NIDS,” Applied Sciences.[^43] | 2025 | Method / evaluation | Generalization | Cross-dataset study | UNSW‑NB15, CSE‑CIC‑IDS2018, BoT‑IoT, ToN‑IoT | Binary/multi‑class | Accuracy, F1, t‑SNE visualization | Cross‑dataset training/testing | Likely yes | Shows strong dataset-specific overfitting and limited cross‑dataset generalization | Uses only traditional ML, not RL | High | Very strong support for demanding external validation and cross‑dataset testing | [DOI](https://doi.org/10.3390/app15158466)[^43] |
| S26 | AdvDatasets2020-KDDUse | “A review of KDD99 dataset usage in intrusion detection and machine learning between 2010 and 2015,” PeerJ Preprints.[^66] | 2016 | Survey | Dataset usage | Literature review | KDD99, NSL‑KDD | NIDS | N/A | N/A | No | Shows heavy, arguably inappropriate continued use of KDD99 | Slightly dated now | Medium | Helps you argue against over‑reliance on KDD/NSL‑KDD alone | [PDF](https://peerj.com/preprints/1954)[^66] |
| S27 | DeepLearningIDS-Survey2025 | Xu et al., “Deep Learning-based Intrusion Detection Systems: A Survey,” arXiv.[^75] | 2025 | Survey | DL IDS | End‑to‑end DL pipeline | Multiple | Attack detection \& investigation | Various | N/A | No | Recent DL‑IDS survey summarizing full pipeline and datasets | Limited RL coverage | High | Very recent; supports “state-of-the-art” claims and dataset overview | [arXiv](https://doi.org/10.48550/arXiv.2504.07839)[^75] |
| S28 | DRL-IDS-Survey2026 | “A Survey for Deep Reinforcement Learning Based Network Intrusion …,” Artificial Intelligence and Law (?) / Wiley.[^50] | 2026 | Survey | DRL IDS | Survey of DRL NIDS | Various | Binary/multi‑class/anomaly | Accuracy, recall, etc. | N/A | No | Dedicated DRL–NIDS survey, evaluating model efficiency and minority attack detection | Very recent; details still limited | High | Anchor citation for DRL‑IDS landscape in NIDS (non‑IoT) | [DOI](https://doi.org/10.1002/ail2.70026)[^50] |
| S29 | DRL-IoT-Survey2023 | Misc. IoT DRL IDS surveys (see S18/S19). | 2023–24 | Survey | DRL IoT IDS | See above | See above | See above | See above | See above | No | Complements S18/S19 | Overlaps | Medium | Provide additional perspective on DRL design patterns | [^47][^48] |
| S30 | FlowNIDS-ML-Eval2022 | “Evaluation of Machine Learning Techniques for Traffic Flow-Based NIDS,” Sensors or similar.[^5] | 2022 | Method / evaluation | Flow-based NIDS | Classical ML algorithms | CIC flow datasets | Binary/multi‑class | Accuracy, AUC | Random splits; cross‑val | Possibly | Systematic benchmarking of ML flow-based IDS on CIC flows | Still purely supervised; no RL | High | Useful for defining classical baselines to compare against your RL defender | [Link](https://pmc.ncbi.nlm.nih.gov/articles/PMC9740321/)[^5] |

*(You can add specific CICDDoS2019, BoT‑IoT, ToN‑IoT, and your favorite CNN/CICIDS2017 papers into this matrix if needed.)*[^15][^21][^14][^22]

***

## 3. Must‑cite papers (15–25)

For each, I indicate where it fits in your chapter.

1. **Sharafaldin2018-CICIDS2017** – Defines CICIDS2017 generation process and feature set; essential when describing your primary dataset and its strengths/weaknesses. (Section: Public datasets / CIC family).[^10][^11][^9]
2. **Moustafa2015-UNSWNB15** – Canonical UNSW‑NB15 dataset description; needed when comparing datasets and discussing “modern” benchmarks beyond KDD/NSL‑KDD. (Section: Public datasets).[^17][^18][^19][^16]
3. **Tavallaee2009-NSLKDD** – Foundational critique of KDD99 and introduction of NSL‑KDD; your main citation for data leakage, duplicates, and biased evaluation. (Section: Dataset quality \& evaluation pitfalls).[^25]
4. **Asadullah2023-MLIDS-Survey** – Broad ML/DL‑IDS survey including attack taxonomy and dataset overview; good “umbrella” reference for ML‑based NIDS. (Section: ML/DL NIDS).[^6]
5. **Sayeeduddin2022-AI-NIDS-Survey** – AI‑based NIDS survey highlighting trend towards DL and overuse of old datasets; supports your motivation for focusing on newer flow datasets. (Section: ML/DL NIDS).[^7]
6. **Xiao2021-DL-IDS-Survey** – Survey focused on deep learning architectures, preprocessing, and evaluation; ideal to introduce DL architectures and feature pipelines. (Section: ML/DL NIDS).[^8]
7. **Umer2017-FlowIDS-Survey** – Flow-based intrusion detection survey; central to motivate flow-level NIDS and differentiate from packet/payload-based systems. (Section: Flow-based NIDS).[^2]
8. **Nguyen2022-BERT-FlowNIDS** – Example of state-of-the-art flow-based NIDS using BERT; shows alternative approaches and domain adaptation concerns. (Section: Flow NIDS with advanced models).[^74]
9. **Caminero2019-AE-RL-IDS** – Explicit dataset-as-environment RL IDS with adversarial environment; the closest conceptual precedent for your “rows as observations, labels as actions” formulation. (Section: RL for NIDS, dataset-as-environment).[^52][^51]
10. **AE-SAC2023** – Softer DRL IDS with adversarial environment and soft actor-critic; strong example of advanced DRL for NIDS but still on legacy datasets, highlighting your gap on flow-based CICIDS2017 and distributional RL. (Section: DRL NIDS).[^53]
11. **Umer2022-RL-IDS (Packet/Flow RL IDS)** – Demonstrates RL for packet- and flow-level intrusion detection on CICDDoS2019; provides a direct RL benchmark in the CIC ecosystem. (Section: RL IDS).[^57]
12. **Sanusi2023-DRL-IDS-Thesis / DQIDS2022** – Early DQN-based IDS works; needed to show DQN family has been used, but usually with non‑flow datasets and simple reward structures. (Section: DRL NIDS algorithms).[^55][^54]
13. **Gueriani2024-DRL-IDS-Survey** – Up‑to‑date DRL‑IDS survey; essential for arguing that DRL is emerging but still methodologically fragile, and distributional RL is rarely used. (Section: RL/D RL for intrusion detection).[^47]
14. **Cevallos2023-DRL-IDS-BestPractices** – Best-practices survey; main source for your methodological critique of DRL‑IDS papers (splits, metrics, imbalance, external validation). (Section: Evaluation methodology in RL‑IDS).[^49][^48]
15. **DRL-IDS-Survey2026** – Very recent general DRL‑NIDS survey; key anchor for “state-of-the-art DRL NIDS” statement (beyond IoT). (Section: RL NIDS landscape).[^50]
16. **Hu2020-POMDP-Defense** – RL/POMDP-based adaptive cyber defense; important to connect your offline RL NIDS to broader autonomous cyber defense literature. (Section: RL-based adaptive defense).[^59][^58]
17. **CSET2023-AutonomousCyberDefense** – Policy/technical report on autonomous cyber defense; strong support for claiming RL is considered central but environments and evaluation remain immature. (Section: Autonomous cyber defense).[^56]
18. **Asadullah2020-DatasetsReview** – Datasets advancement survey; important for global dataset context (from KDD99 to CIC/UNSW/IoT datasets) and to justify dataset choices. (Section: Public datasets).[^35]
19. **Generalizability2025-TrainingData** – Cross-dataset generalizability study; your strongest evidence that models trained and evaluated on a single dataset may not generalize to others, motivating external validation on lab traffic. (Section: Evaluation methodology problems).[^43]
20. **AssemblingCyberRange2022 / Algorizmi2010-Testbed / ICS-Testbed2021** – At least one testbed/cyber range paper should be cited to legitimize using lab-captured traffic as an external validation stage. (Section: Lab-captured traffic / cyber ranges).[^70][^71][^65]
21. **DARPA1998/1999 / Tavallaee2009** – For historical context on early IDS evaluations and criticisms that drive current dataset design. (Section: Historical background on NIDS evaluation).[^32][^33][^34][^31][^25]

You probably don’t need to cite every single DL‑on‑CICIDS2017 method paper explicitly; instead, cite a representative subset and rely on surveys to summarize the trend.[^41][^42][^36][^37][^6]

***

## 4. Methodological warnings

You’ll want a subsection explicitly flagging methodological problems. Many are general trends; you can also point to representative works.

### 4.1 Overuse of outdated or flawed datasets

- **KDD99/NSL‑KDD** – Tavallaee et al. show 78% and 75% duplicate records in KDD99 train/test; NSL‑KDD reduces duplicates but still does not reflect modern traffic or attack diversity.[^29][^25]
- Reviews find KDD99 and NSL‑KDD remain widely used between 2010–2015 and beyond, despite clear limitations and obsolescence.[^76][^66]

Methodological implication: any paper claiming near‑perfect accuracy exclusively on NSL‑KDD/KDD99, without external validation or additional datasets, should be treated cautiously.

### 4.2 Random split inflation and leakage

- CICIDS2017 and other CIC datasets contain temporal structure and scenario-based capture; random splitting can easily leak flows from the same attack episode into both train and test, inflating results.[^9][^10][^35]
- Many CNN / DL papers on CICIDS2017 (e.g., DL‑IDS, LeNet-based, MLP autoencoders) use random 70/30 or k‑fold splits and report 98–99.9% accuracy without analyzing temporal leakage or near-duplicate flows.[^39][^40][^42][^38][^37][^41]

You can flag this class of results as “optimistic within‑dataset” rather than robust generalization.

### 4.3 Class imbalance and minority attack detection

- DRL‑IDS surveys explicitly note that most works do not properly handle heavily imbalanced datasets and often optimize global accuracy, masking poor detection for minority attack types.[^48][^50][^47]
- AE‑SAC and adversarial-environment methods try to resample and reweight classes, but still primarily evaluate on NSL‑KDD or AWID with random splits and no external datasets.[^51][^53]

For your thesis, explicitly measure per‑class metrics, especially for rare attack types, and avoid relying purely on accuracy.

### 4.4 Poor external validation and cross-dataset testing

- Generalizability paper shows that ML models trained on UNSW‑NB15 often perform poorly when tested on CSE‑CIC‑IDS2018, BoT‑IoT, or ToN‑IoT, even if within‑dataset accuracy is very high.[^43]
- Many ML/DL‑NIDS papers only report in‑dataset results with no testing on other datasets or real traffic.[^7][^8][^6]

Your external validation on lab-captured traffic sits squarely in this gap; highlight that very few NIDS/DRL‑IDS works do such a second-stage validation on independent captures.[^69][^70][^65]

### 4.5 Unclear preprocessing and feature engineering

- Survey papers note that preprocessing steps (e.g., one‑hot encoding categorical features, scaling, handling missing values) are often under‑documented, making reproducibility difficult.[^8][^6]
- CIC dataset pipelines (CICFlowMeter) involve many implicit decisions (bi-flow aggregation, timeout, direction) which many downstream papers do not detail, leading to subtle differences in reproducibility.[^13][^12][^9]

Your thesis should clearly specify the feature set, normalization, categorical encoding, and any flow filtering you apply.

### 4.6 Lack of code, reproducibility, and open benchmarks

- Many DL and DRL IDS papers do not release code or exact splits, so results cannot be reproduced independently.[^6][^47][^7][^8]
- There is a recent attempt to provide preprocessed, 5‑fold cross‑validation versions of common IDS datasets to enable fair comparison, but this is not yet widely adopted.[^77]

You can position your work as more rigorous by: (i) fixing and documenting splits (e.g., temporal splits), (ii) releasing code, and (iii) possibly using existing standardized folds where appropriate.

### 4.7 Suspiciously high accuracy claims

- Multiple NSL‑KDD and CICIDS2017 papers report 99–100% accuracy or F1 with relatively simple models and random splits, which is likely an artifact of dataset biases and leakage rather than true performance on novel traffic.[^42][^78][^79][^46][^37][^41]
- DRL‑IDS works claiming >99% accuracy on NSL‑KDD with no external validation should be discussed but treated skeptically in your critical appraisal.[^79][^80][^53]

You do not need to single out every such paper; instead, cite a few representative ones and underpin your skepticism with Tavallaee, dataset reviews, DRL‑survey critiques, and generalizability studies.[^48][^35][^25][^47][^43]

***

## 5. Research gaps your thesis can claim

Aim for “limited, under‑evaluated, or methodologically fragile,” not “nobody has done this.”

1. **Distributional DRL (e.g., QRDQN) for flow-based NIDS on modern CIC datasets is largely unexplored.**
    - DRL‑IDS surveys list DQN, DDQN, DDPG, A3C, PPO, SAC, etc., but do not mention distributional methods like C51 or QRDQN applied to NIDS; most works use standard Q‑learning or actor‑critic variants.[^50][^53][^47][^48]
    - Your thesis can claim to be *among the first* to apply distributional DRL to flow-based NIDS using CICIDS2017 as the environment.[^57][^9]
2. **Dataset‑as‑environment RL formulations exist but are under‑explored and mostly on legacy datasets.**
    - AE‑RL and AE‑SAC explicitly use the dataset as a simulated environment, with the RL agent selecting class labels as actions and an environment agent resampling records.[^51][^53]
    - These works mainly focus on NSL‑KDD/AWID; there is a lack of systematic exploration on flow‑based CIC datasets and in the context of a simple PERMIT/BLOCK decision boundary.[^9][^57][^47]
    - Your thesis can position its Gymnasium environment (each flow row as observation; permit/block as actions) as extending this paradigm to a widely-used flow dataset and a binary detection task.
3. **Flow-based RL defenders are mostly one‑dataset, offline, and lack external validation.**
    - Existing RL‑IDS papers typically evaluate on a single public dataset and do not test cross‑dataset performance or evaluation on lab-captured traffic.[^54][^55][^53][^57]
    - In contrast, classical ML‑NIDS research has begun to explore cross‑dataset generalization and ICS/IoT testbed evaluation; RL‑IDS has not caught up.[^70][^65][^22][^43]
    - Your two‑phase design (offline training/validation on CICIDS2017 and offline inference on private lab traffic) directly addresses this under‑evaluated dimension.
4. **Evaluation methodology for RL‑based NIDS is under‑discussed and fragile.**
    - DRL‑IDS surveys identify that most works use random splits, optimize only for accuracy, and rarely consider temporal splits, external validation, or reproducibility.[^47][^48][^50]
    - The generalizability paper shows that even classical ML models fail to transfer across datasets, which is likely even more true for DRL models trained on reward signals anchored to a single dataset.[^43]
    - Your thesis can contribute by adopting stricter methodology: temporal splits where meaningful, dataset‑shift evaluation on lab traffic, reporting per‑class metrics, and detailed documentation of environment formulation and reward design.
5. **Bridging RL‑based NIDS and autonomous cyber defense remains conceptual.**
    - RL is widely studied for high‑level cyber defense (attack graphs, POMDPs, honeypot engagement, cyber ranges) and separately for low‑level NIDS classification, but the integration is still not well‑developed.[^64][^63][^58][^60][^56]
    - Your work can be framed as a foundational component: a flow‑level RL defender that learns permit/block decisions, potentially pluggable into a broader autonomous defense architecture (e.g., as the detection component).
6. **Honest quantification of performance under realistic constraints is rare.**
    - Many papers report idealized offline performance without considering costs of false positives, label noise, and operational constraints.[^7][^8][^6][^47]
    - Your thesis can emphasize realistic offline evaluation: (i) cost‑sensitive rewards (e.g., penalizing false positives on benign traffic), (ii) analysis of generalization to lab traffic, and (iii) explicit discussion of missing active blocking and deployment constraints.

***

## 6. Suggested State of the Art structure (in Spanish)

You can adapt this outline to your thesis length and style.

### 2. Antecedentes y conceptos básicos

2.1. Seguridad de redes y modelos de amenaza
2.2. Sistemas de detección de intrusiones (IDS)

- 2.2.1. IDS basados en host vs. basados en red
- 2.2.2. Detección de anomalías vs. detección de firmas
- 2.2.3. Requisitos para un IDS en redes modernas (volumen, cifrado, latencia)

2.3. Flujos de red y monitorización basada en flujos

- 2.3.1. Definición de flujo (NetFlow, IPFIX, CICFlowMeter)
- 2.3.2. Ventajas y limitaciones de la detección basada en flujo frente a paquetes/payload
- 2.3.3. Arquitecturas típicas de NIDS basados en flujo[^1][^2][^4]


### 3. Conjuntos de datos públicos para detección de intrusiones

3.1. Evolución histórica de los datasets de IDS

- 3.1.1. DARPA 1998/1999 y sus críticas
- 3.1.2. KDD Cup 99 y problemas de redundancia, realismo y leakage
- 3.1.3. NSL‑KDD como mejora incremental y sus limitaciones[^33][^34][^66][^31][^32][^25]

3.2. Datasets de nueva generación: UNSW‑NB15 y familia CIC

- 3.2.1. UNSW‑NB15: entorno de Cyber Range, ataques y características[^18][^19][^16][^17]
- 3.2.2. CICIDS2017: criterios de diseño, escenarios y flujo de características[^11][^10][^9]
- 3.2.3. CSE‑CIC‑IDS2018: tráfico empresarial multi‑día[^12][^13]
- 3.2.4. CICDDoS2019: dataset específico de DDoS[^14][^15]

3.3. Datasets para IoT/IIoT y otros dominios

- 3.3.1. BoT‑IoT y TON‑IoT desde el Cyber Range de UNSW[^24][^21][^22]
- 3.3.2. Datasets industriales e ICS procedentes de bancos de pruebas físicos[^69][^70]
- 3.3.3. Colecciones estandarizadas y particiones de validación cruzada (IDS‑5‑FCV)[^77]

3.4. Comparación crítica de los datasets

- 3.4.1. Cobertura de tipos de ataque y realismo del tráfico
- 3.4.2. Sesgos, redundancia y representatividad
- 3.4.3. Implicaciones para la generalización de modelos de ML/DL[^35][^6][^43]


### 4. Aprendizaje automático y profundo para NIDS

4.1. Enfoques clásicos de ML para NIDS

- 4.1.1. Árboles de decisión, SVM, k‑NN, ensembles sobre NSL‑KDD, UNSW‑NB15, CICIDS2017[^81][^45][^46][^44]
- 4.1.2. Resultados típicos y limitaciones (alta exactitud, baja robustez)

4.2. Deep Learning para NIDS basados en flujo

- 4.2.1. Arquitecturas CNN, RNN/LSTM, autoencoders en datos de flujo[^36][^38][^37][^41][^42][^8]
- 4.2.2. Métodos avanzados: modelos Transformer/BERT sobre secuencias de flujo[^74]
- 4.2.3. Survey de DL‑IDS y taxonomía de modelos[^75][^36][^8]

4.3. Problemas metodológicos en DL‑NIDS

- 4.3.1. Desequilibrios de clases y métricas centradas en la exactitud global[^8][^6][^7]
- 4.3.2. Riesgos de leakage por división aleatoria y tráfico correlacionado[^10][^35][^9]
- 4.3.3. Escasa validación externa y reproducibilidad


### 5. Aprendizaje por refuerzo para detección de intrusiones y defensa adaptativa

5.1. Fundamentos de aprendizaje por refuerzo y DRL

- 5.1.1. Formulación MDP/POMDP, políticas y funciones de valor
- 5.1.2. Algoritmos relevantes: DQN, Double DQN, Dueling DQN, PPO, A2C, SAC, métodos distribucionales (C51, QRDQN)[^50][^47]

5.2. DRL aplicado a NIDS / clasificación de intrusiones

- 5.2.1. Primeros enfoques basados en DQN para NSL‑KDD[^55][^54]
- 5.2.2. IDS basados en RL a nivel de paquete y flujo (CICDDoS2019)[^57]
- 5.2.3. Modelos con entorno adversarial y sampling inteligente (AE‑RL, AE‑SAC)[^53][^51]

5.3. Surveys de DRL‑IDS y análisis crítico

- 5.3.1. DRL‑IDS en IoT: taxonomía, datasets y métricas[^48][^47]
- 5.3.2. DRL‑IDS genérico: modelos, eficiencia y detección de ataques minoritarios[^50]
- 5.3.3. Principales carencias metodológicas identificadas (splits, desequilibrios, validación)

5.4. Formulaciones dataset‑as‑environment

- 5.4.1. Integración de frameworks supervisados con RL mediante entornos simulados[^51]
- 5.4.2. Re‑muestreo adversarial y diseño de recompensas por clase[^53]
- 5.4.3. Comparación con el entorno basado en Gymnasium propuesto en esta tesis


### 6. Defensa cibernética autónoma y rangos/cibermulas de entrenamiento

6.1. Defensa adaptativa basada en modelos (POMDP, grafos de ataque)

- 6.1.1. Defensa adaptativa frente a ataques multi‑etapa mediante RL y POMDP[^58][^59]
- 6.1.2. Políticas de engaño activo y honeypots gestionados por RL[^60]

6.2. Agentes autónomos de ciberdefensa y ciber‑rangos

- 6.2.1. Visión general de la defensa cibernética autónoma y retos de diseño[^56]
- 6.2.2. Entornos tipo CyberBattleSim, CAGE y marcos de evaluación con RL[^63][^64]
- 6.2.3. Plataformas de cyber range para evaluar NIDS y herramientas basadas en IA/ML[^71][^65]

6.3. Datasets de laboratorio y tráfico capturado en bancos de pruebas

- 6.3.1. Datasets ICS, IIoT y UAV generados en testbeds físicos[^82][^69][^70]
- 6.3.2. Uso de tráfico de laboratorio como validación externa de NIDS y modelos de RL


### 7. Problemas de evaluación en NIDS basados en ML/DRL

7.1. Fugas de información, particiones aleatorias y duplicados
7.2. Desequilibrio de clases y métricas adecuadas (ROC‑AUC, F1, TPR/FPR por clase)
7.3. Transferencia entre datasets y validación externa[^35][^43]
7.4. Reproducibilidad, publicación de código y particiones estandarizadas[^77][^6][^7][^8]

### 8. Síntesis y posicionamiento de la tesis

8.1. Resumen crítico del estado del arte
8.2. Identificación de las lagunas de investigación
8.3. Aportaciones específicas del enfoque basado en QRDQN y flujo de red con validación externa

***

## 7. BibTeX candidates

Below are example BibTeX entries for key sources (adapt as needed for your style).

```bibtex
@inproceedings{Sharafaldin2018CICIDS2017,
  author    = {Iman Sharafaldin and Arash Habibi Lashkari and Ali A. Ghorbani},
  title     = {Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization},
  booktitle = {Proceedings of the 4th International Conference on Information Systems Security and Privacy (ICISSP)},
  year      = {2018},
  url       = {http://www.unb.ca/cic/datasets/ids-2017.html}
}

@inproceedings{Moustafa2015UNSWNB15,
  author    = {Nour Moustafa and Jill Slay},
  title     = {{UNSW-NB15}: A Comprehensive Data Set for Network Intrusion Detection Systems ({UNSW-NB15} Network Data Set)},
  booktitle = {2015 Military Communications and Information Systems Conference (MilCIS)},
  year      = {2015},
  pages     = {1--6},
  doi       = {10.1109/MilCIS.2015.7348942}
}

@inproceedings{Tavallaee2009NSLKDD,
  author    = {Mahbod Tavallaee and Ebrahim Bagheri and Wei Lu and Ali A. Ghorbani},
  title     = {A Detailed Analysis of the {KDD CUP 99} Data Set},
  booktitle = {2009 IEEE Symposium on Computational Intelligence for Security and Defense Applications},
  year      = {2009},
  pages     = {1--6},
  doi       = {10.1109/CISDA.2009.5356528}
}

@article{Umer2017FlowIDS,
  author    = {Muhammad Fahad Umer and Muhammad Sher and Yaxin Bi},
  title     = {Flow-based Intrusion Detection: Techniques and Challenges},
  journal   = {Computers \& Security},
  volume    = {70},
  pages     = {238--254},
  year      = {2017},
  doi       = {10.1016/j.cose.2017.05.009}
}

@article{Asadullah2023MLIDSSurvey,
  author    = {Momand Asadullah and Jan Sana Ullah and Ramzan Naeem},
  title     = {A Systematic and Comprehensive Survey of Recent Advances in Intrusion Detection Systems Using Machine Learning: Deep Learning, Datasets, and Attack Taxonomy},
  journal   = {Expert Systems},
  year      = {2023},
  doi       = {10.1155/2023/6048087}
}

@article{Sayeeduddin2022AINIDS,
  author    = {Habeeb Mohammed Sayeeduddin and T. Ranga Babu},
  title     = {Network Intrusion Detection System: A Survey on Artificial Intelligence-Based Techniques},
  journal   = {Expert Systems},
  year      = {2022},
  doi       = {10.1111/exsy.13066}
}

@article{Xiao2021DLIDSSurvey,
  author    = {Jianping Xiao and Chun Long and Jing Zhao and Jinxia Wei and Anlei Hu and Guanyao Du},
  title     = {A Survey on Network Intrusion Detection Based on Deep Learning},
  journal   = {Frontiers of Data and Computing},
  volume    = {3},
  number    = {3},
  pages     = {59--74},
  year      = {2021}
}

@inproceedings{Nguyen2022BERTFlowNIDS,
  author    = {Loc Gia Nguyen and Kohei Watabe},
  title     = {Flow-based Network Intrusion Detection Based on {BERT} Masked Language Model},
  booktitle = {CoNEXT 2022 Student Workshop},
  year      = {2022},
  url       = {https://arxiv.org/abs/2306.04920}
}

@article{Umer2022RLRLIDS,
  author    = {Ying Zhao and others}, % adapt authors appropriately for final reference
  title     = {Packet-Level and Flow-Level Network Intrusion Detection Based on Reinforcement Learning and Adversarial Training},
  journal   = {Algorithms},
  volume    = {15},
  number    = {12},
  pages     = {453},
  year      = {2022},
  doi       = {10.3390/a15120453}
}

@article{AESAC2023RLIDS,
  author    = {First Author and Others}, % complete from final paper
  title     = {A Soft Actor-Critic Reinforcement Learning Algorithm for Network Intrusion Detection},
  journal   = {Computers \& Security},
  year      = {2023}
}

@inproceedings{Caminero2019AERLIDS,
  author    = {Guillermo Caminero and Manuel Lopez-Martin and Belen Carro},
  title     = {Adversarial Environment Reinforcement Learning Algorithm for Intrusion Detection},
  booktitle = {Proceedings / thesis of Universidad de Valladolid},
  year      = {2019},
  url       = {https://uvadoc.uva.es/handle/10324/54301}
}

@article{Gueriani2024DRLIoTIDSSurvey,
  author    = {Afrah Gueriani and Hamza Kheddar and Ahmed Cherif Mazari},
  title     = {Deep Reinforcement Learning for Intrusion Detection in IoT: A Survey},
  journal   = {Conference Proceedings IC2EM},
  year      = {2023},
  url       = {https://arxiv.org/abs/2405.20038}
}

@article{Cevallos2023DRLIDSBP,
  author    = {Jes{\'u}s F. Cevallos and Alessandra Rizzardi and Sabrina Sicari and Alberto Coen-Porisini},
  title     = {Deep Reinforcement Learning for Intrusion Detection in Internet of Things: Best Practices, Lessons Learnt, and Open Challenges},
  journal   = {Computer Networks},
  year      = {2023}
}

@article{Hu2020POMDPDefense,
  author    = {Zhisheng Hu and Minghui Zhu and Peng Liu},
  title     = {Adaptive Cyber Defense Against Multi-Stage Attacks Using Learning-Based {POMDP}},
  journal   = {ACM Transactions on Privacy and Security},
  volume    = {24},
  number    = {1},
  year      = {2020},
  doi       = {10.1145/3418897}
}

@misc{CSET2023AutonomousCyberDefense,
  author    = {{Center for Security and Emerging Technology} and {The Alan Turing Institute}},
  title     = {Autonomous Cyber Defense},
  year      = {2023},
  howpublished = {\url{https://cset.georgetown.edu/publication/autonomous-cyber-defense/}}
}

@article{TrainingData2025Generalizability,
  author    = {First Author and Others}, % fill exact authors
  title     = {The Choice of Training Data and the Generalizability of Machine Learning Models for Network Intrusion Detection Systems},
  journal   = {Applied Sciences},
  year      = {2025},
  doi       = {10.3390/app15158466}
}
```

You can generate similar entries for the DRL‑IDS 2026 survey, DL‑IDS 2025 survey, and key ICS/testbed/cyber range papers once you finalize the exact bibliographic details.[^75][^65][^71][^69][^50]

***

## 8. Codex handoff summary

For the agent that will draft the chapter:

- **Recommended structure**: Use the Spanish outline in Section 6. The core sections are: (i) IDS concepts and flow-based NIDS, (ii) public datasets (KDD/NSL‑KDD, UNSW‑NB15, CIC family, IoT datasets), (iii) ML/DL for NIDS (with emphasis on flow-based DL), (iv) RL/DRL for NIDS and dataset‑as‑environment formulations, (v) RL-based autonomous/adaptive cyber defense and cyber ranges, (vi) evaluation methodology problems, and (vii) synthesis and positioning of the thesis.
- **Must‑cite sources**: Prioritize Sharafaldin2018, Moustafa2015, Tavallaee2009, Umer2017, Asadullah2023 (ML/DL‑IDS survey), Xiao2021, Sayeeduddin2022, Gueriani2024, Cevallos2023, DRL‑IDS‑Survey2026, Hu2020, CSET2023, AE‑RL (Caminero2019), AE‑SAC2023, Umer2022‑RL‑IDS, Generalizability2025, DQ‑IDS and at least one cyber‑range/testbed paper.[^19][^16][^17][^18][^11][^2][^59][^65][^10][^25][^55][^58][^9][^56][^57][^6][^47][^48][^7][^8][^51][^53][^50][^43]
- **Claims that are safe to make** (with proper citations):
    - KDD99 and NSL‑KDD suffer from redundancy, outdated attacks, and limited realism; high accuracy on these datasets does not imply real-world effectiveness.[^66][^29][^25]
    - Flow-based NIDS are attractive for high‑speed and encrypted networks and have an established literature, but ML/DL approaches often rely on a small set of public datasets such as CICIDS2017 and UNSW‑NB15.[^16][^2][^1][^9][^35]
    - Many ML/DL‑based NIDS (including DL on CICIDS2017) use random splits and report very high accuracy without external validation or cross‑dataset testing; this is a common methodological weakness.[^37][^41][^42][^6][^35][^43]
    - DRL‑based IDS is an active but still emerging area, with most works using standard DQN/SAC variants on NSL‑KDD-like datasets and little focus on distributional RL.[^47][^48][^53][^50]
    - Dataset-as-environment formulations exist (AE‑RL, AE‑SAC) but are mostly applied to legacy or non‑flow datasets and are rarely evaluated beyond single datasets.[^51][^53]
    - Cross‑dataset generalization of ML NIDS is limited; models perform much worse when tested on other datasets than on their training set.[^43]
    - Cyber ranges and lab testbeds are a recognized way to generate realistic datasets and to evaluate AI/ML NIDS under controlled but realistic conditions.[^21][^65][^22][^71][^69]
- **Claims to avoid or phrase carefully**:
    - Do *not* claim that no RL‑based NIDS exists; instead, say that “there are relatively few RL‑based NIDS, and most focus on legacy datasets and standard DRL algorithms” with citations.[^54][^55][^57][^53][^47][^50]
    - Do *not* claim to be the first to treat a dataset as an RL environment; AE‑RL and AE‑SAC already do this; instead, emphasize you are applying a dataset‑as‑environment formulation to a flow-based CIC dataset with a distributional QRDQN agent and a binary permit/block decision.
    - Do *not* claim to solve generalization or autonomous defense; you can claim to *contribute* to more rigorous evaluation (temporal splits, lab traffic validation) and to a building block for future autonomous defenders.[^58][^56][^43]
- **Unresolved questions requiring verification before writing**:
    - Confirm precise bibliographic details (authors, venue, pages) for AE‑SAC, DRL‑IDS 2026 survey, DL‑IDS 2025 survey, flow ML evaluation paper, and any specific CICIDS2017 DL paper you choose to cite as exemplar.[^5][^75][^53][^50]
    - Decide which exact RL algorithms besides QRDQN you will implement or compare against (e.g., vanilla DQN, Double DQN, PPO); ensure there is at least one prior NIDS paper to cite for each algorithm family you mention.
    - Clarify your evaluation splits on CICIDS2017 (e.g., temporal per-day, scenario‑based) and how they map to splits used in the literature; this affects how strongly you can critique random splits.
    - Document the characteristics of your lab dataset (testbed topology, traffic generation, attack scenarios) to connect it clearly to the “cyber range/testbed” literature and to support your external validation claims.[^65][^71][^69]

Use this source map as the backbone: when drafting, each subsection should introduce the theme, cite at least one survey plus 1–3 key primary works, and then provide a short critical comparison focusing on methodology, data, and evaluation rather than just raw accuracy.
<span style="display:none">[^100][^101][^102][^103][^104][^105][^106][^107][^108][^109][^110][^111][^112][^113][^114][^115][^116][^117][^118][^119][^120][^121][^122][^123][^124][^125][^126][^127][^128][^129][^130][^131][^132][^133][^134][^135][^136][^137][^138][^139][^140][^141][^142][^143][^144][^145][^146][^147][^148][^149][^150][^151][^152][^83][^84][^85][^86][^87][^88][^89][^90][^91][^92][^93][^94][^95][^96][^97][^98][^99]</span>

<div align="center">⁂</div>

[^1]: https://www.sciencedirect.com/science/article/abs/pii/S0167404817301165

[^2]: https://linkinghub.elsevier.com/retrieve/pii/S0167404817301165

[^3]: https://dblp.uni-trier.de/rec/journals/compsec/UmerSB17.html

[^4]: https://www.semanticscholar.org/paper/Flow-based-Intrusion-Detection-System-for-SDN-Ajaeiya-Adalian/e4579420355ddc465b42342f04245b32f0e1e93e

[^5]: https://pmc.ncbi.nlm.nih.gov/articles/PMC9740321/

[^6]: https://onlinelibrary.wiley.com/doi/10.1155/2023/6048087

[^7]: https://onlinelibrary.wiley.com/doi/abs/10.1111/exsy.13066

[^8]: http://www.jfdc.cnic.cn/EN/10.11871/jfdc.issn.2096-742X.2021.03.006

[^9]: https://www.cs.unb.ca/research-expo/expos/2018/submissions/20180403-14-56-isharafa-at-unb.ca-toward_generating_a_new_intrusion_detection_dataset_and_intrusion_traffic_characterization.pdf

[^10]: https://www.semanticscholar.org/paper/Toward-Generating-a-New-Intrusion-Detection-Dataset-Sharafaldin-Lashkari/a27089efabc5f4abd5ddf2be2a409bff41f31199

[^11]: https://www.scitepress.org/papers/2018/66398/66398.pdf

[^12]: https://registry.opendata.aws/cse-cic-ids2018/

[^13]: https://aws.amazon.com/marketplace/pp/prodview-qkyroawpr2aw6

[^14]: https://data.mendeley.com/datasets/ssnc74xm6r/1

[^15]: https://www.unb.ca/cic/datasets/ddos-2019.html

[^16]: http://ieeexplore.ieee.org/document/7348942/

[^17]: https://www.semanticscholar.org/paper/UNSW-NB15:-a-comprehensive-data-set-for-network-Moustafa-Slay/0e7af8e91b8cb2cea1164be5ac5d280b0d12c153

[^18]: https://www.scribd.com/document/458760818/07348942-pdf

[^19]: https://researchdata.edu.au/the-unsw-nb15-dataset/1957529

[^20]: https://openargus.org/argus-ml?view=article\&id=35%3Athe-bot-iot-dataset\&catid=2

[^21]: https://research.unsw.edu.au/projects/bot-iot-dataset

[^22]: https://thesai.org/Publications/ViewPaper?Volume=13\&Issue=6\&Code=IJACSA\&SerialNo=67

[^23]: https://www.nature.com/articles/s41598-026-37834-y

[^24]: https://www.kaggle.com/datasets/arnobbhowmik/ton-iot-network-dataset

[^25]: https://www.semanticscholar.org/paper/A-detailed-analysis-of-the-KDD-CUP-99-data-set-Tavallaee-Bagheri/fc3eb090e39d71295c362458b8a0c48d2c5d8377

[^26]: https://www.unb.ca/cic/datasets/nsl.html

[^27]: https://github.com/jmnwong/NSL-KDD-Dataset

[^28]: https://github.com/HoaNP/NSL-KDD-DataSet

[^29]: https://learn.saylor.org/mod/book/view.php?chapterid=5443\&id=29755

[^30]: http://www.tandfonline.com/doi/full/10.1080/19393555.2015.1125974

[^31]: https://archive.ll.mit.edu/ideval/data/1999data.html

[^32]: https://archive.ll.mit.edu/ideval/data/

[^33]: https://www.ll.mit.edu/r-d/datasets/1999-darpa-intrusion-detection-evaluation-dataset

[^34]: https://www.sciencedirect.com/science/article/abs/pii/S1389128600001390

[^35]: https://www.sciencedirect.com/science/article/pii/S1877050920307961

[^36]: https://www.mdpi.com/2076-3417/12/16/8162

[^37]: https://www.hindawi.com/journals/scn/2020/8890306/

[^38]: https://ieeexplore.ieee.org/document/11447497/

[^39]: https://ieeexplore.ieee.org/document/10873613/

[^40]: https://ieeexplore.ieee.org/document/10601961/

[^41]: https://www.tandfonline.com/doi/full/10.1080/19393555.2020.1797248

[^42]: https://aircconline.com/csit/papers/vol10/csit101501.pdf

[^43]: https://www.mdpi.com/2076-3417/15/15/8466

[^44]: https://ieeexplore.ieee.org/document/10103301/

[^45]: https://ieeexplore.ieee.org/document/10973641/

[^46]: https://www.mdpi.com/1424-8220/20/9/2559

[^47]: https://arxiv.org/abs/2405.20038

[^48]: https://dl.acm.org/doi/10.1016/j.comnet.2023.110016

[^49]: https://www.dicom.uninsubria.it/~sabrina.sicari/public/documents/2023_surveyML.pdf

[^50]: https://onlinelibrary.wiley.com/doi/abs/10.1002/ail2.70026

[^51]: https://uvadoc.uva.es/bitstream/handle/10324/54301/Adversarial-environment-reinforcement-learning.pdf;jsessionid=AAA40333F7DA6C95BB36336854966176?sequence=1

[^52]: https://uvadoc.uva.es/handle/10324/54301?locale-attribute=en

[^53]: https://www.sciencedirect.com/science/article/abs/pii/S0167404823004121

[^54]: https://digitalcommons.georgiasouthern.edu/cgi/viewcontent.cgi?article=3909\&context=etd

[^55]: https://www.sciencedirect.com/science/article/pii/S2405959525000694

[^56]: https://cset.georgetown.edu/publication/autonomous-cyber-defense/

[^57]: https://www.mdpi.com/1999-4893/15/12/453

[^58]: https://par.nsf.gov/servlets/purl/10296651

[^59]: https://dl.acm.org/doi/10.1145/3418897

[^60]: https://arxiv.org/abs/1906.12182

[^61]: https://www.sciencedirect.com/science/article/abs/pii/S095741742401474X

[^62]: https://ieeexplore.ieee.org/document/11020085/

[^63]: https://ieeexplore.ieee.org/document/10216719/

[^64]: https://github.com/Kim-Hammar/awesome-rl-for-cybersecurity

[^65]: https://arxiv.org/abs/2201.08473

[^66]: https://peerj.com/preprints/1954v1.pdf

[^67]: https://fkie-cad.github.io/COMIDDS/content/datasets/unsw_nb15/

[^68]: https://www.semanticscholar.org/paper/25a3331217b007c0bc496a2271366820b1867ed3

[^69]: https://ieeexplore.ieee.org/document/10491813/

[^70]: https://www.techscience.com/iasc/v31n3/44856/html

[^71]: https://uwspace.uwaterloo.ca/items/f774f0c1-9637-48c8-9352-d1623ffef423

[^72]: https://onlinelibrary.wiley.com/doi/10.1111/exsy.13066

[^73]: https://dl.acm.org/doi/10.1145/3565477.3569152

[^74]: https://arxiv.org/abs/2306.04920

[^75]: https://arxiv.org/abs/2504.07839

[^76]: https://arxiv.org/pdf/2209.05579.pdf

[^77]: https://ruja.ujaen.es/items/24f750f8-1995-4c09-9f83-8d7a05aabd59

[^78]: https://arxiv.org/ftp/arxiv/papers/2310/2310.16380.pdf

[^79]: https://mesopotamian.press/journals/index.php/BJN/article/download/492/368

[^80]: https://pmc.ncbi.nlm.nih.gov/articles/PMC12589504/

[^81]: https://www.semanticscholar.org/paper/Network-Based-Intrusion-Detection-Using-the-Dataset-Meftah-Rachidi/5c11dfde2a4859103c33109c7d50a6fa0ddc0cc5

[^82]: https://ieeexplore.ieee.org/document/10937270/

[^83]: https://ieeexplore.ieee.org/document/9927396/

[^84]: https://www.mdpi.com/1424-8220/22/4/1494

[^85]: https://ieeexplore.ieee.org/document/8999496/

[^86]: https://www.sciencedirect.com/science/article/abs/pii/S0957417424018499

[^87]: https://jtit.pl/jtit/article/view/2311

[^88]: https://www.semanticscholar.org/paper/A-Survey-on-Machine-and-Deep-Learning-Based-Systems-H.A./dfd06d740a2b952a4f74fc0c7ea9b1f44d1ad789

[^89]: https://www.sciencedirect.com/org/science/article/pii/S1546221826001050

[^90]: https://dl.acm.org/doi/10.1109/CCNC49033.2022.9700597

[^91]: https://ieeexplore.ieee.org/document/8672138/

[^92]: https://ieeexplore.ieee.org/document/9288358/

[^93]: https://www.scribd.com/document/1018232723/CICids2017-dataset-origin-paper

[^94]: https://ijsrm.net/index.php/ijsrm/article/view/3878/2599

[^95]: https://dl.acm.org/doi/abs/10.1504/ijics.2021.117392

[^96]: https://www.scribd.com/document/415570812/2015-UNSW-NB15-a-Comprehensive-Data-Set-for-Network-Intrusion-Detection-Systems

[^97]: https://www.etasr.com/index.php/ETASR/article/view/14119

[^98]: https://www.academia.edu/84882558/Unsw_Nb15_Dataset_and_Machine_Learning_Based_Intrusion_Detection_Systems

[^99]: https://dergipark.org.tr/en/download/article-file/4230736

[^100]: http://ieeexplore.ieee.org/document/7809531/

[^101]: https://www.atlantis-press.com/article/21458

[^102]: https://www.atlantis-press.com/article/25857

[^103]: https://www.scitepress.org/DigitalLibrary/Link.aspx?doi=10.5220/0013932600004919

[^104]: https://www.scribd.com/document/902947241/28o

[^105]: https://www.scribd.com/document/498029404/20180403-14-56-isharafa-at-unb-ca-toward-generating-a-new-intrusion-detection-dataset-and-intrusion-traffic-characterization

[^106]: https://www.ijert.org/research/long-short-term-memory-lstm-deep-learning-method-for-intrusion-detection-in-network-security-IJERTV9IS061016.pdf

[^107]: https://arxiv.org/pdf/1912.13204.pdf

[^108]: https://www.itm-conferences.org/articles/itmconf/pdf/2022/06/itmconf_iceas2022_02003.pdf

[^109]: https://www.academia.edu/63033480/A_Detailed_Analysis_on_NSL_KDD_Dataset_Using_Various_Machine_Learning_Techniques_for_Intrusion_Detection

[^110]: https://github.com/thinline72/nsl-kdd

[^111]: https://www.scribd.com/document/994229645/Tavalla-Ee-2009-Nsl

[^112]: https://github.com/InitRoot/NSLKDD-Dataset

[^113]: https://www.ll.mit.edu/r-d/datasets/1998-darpa-intrusion-detection-evaluation-dataset

[^114]: https://scholar.google.com/citations?user=vquxsD0AAAAJ\&hl=en

[^115]: http://ieeexplore.ieee.org/document/8000162/

[^116]: http://ieeexplore.ieee.org/document/8029562/

[^117]: http://ieeexplore.ieee.org/document/8272665/

[^118]: https://proceedings.elseconference.eu/index.php?paper=53800f7a8508e10226c62e146676fd63

[^119]: https://ieeexplore.ieee.org/document/8397440/

[^120]: http://link.springer.com/10.1007/978-3-319-65188-0

[^121]: https://www.semanticscholar.org/paper/ade5f9d9c034cde2fa9564d9ef7c1a53ae22ce2d

[^122]: http://muhammetbaykara.com/wp-content/uploads/2018/10/ymtgunduz_intrusion.pdf

[^123]: https://www.scribd.com/document/953844196/Reinforcement-Learning-for-Intrusion-Detection-Recent-Advances-and-Datasets

[^124]: https://www.ndss-symposium.org/wp-content/uploads/sdiotsec26-97.pdf

[^125]: https://www.ece.fr/2023/10/26/intrusion-detection-with-multi-agent-reinforcement-learning-and-balanced-dataset/

[^126]: https://dl.acm.org/doi/full/10.1145/3764586

[^127]: https://dl.acm.org/doi/10.1145/3560830.3563732

[^128]: https://www.semanticscholar.org/paper/e5066558f467d8d162950778b229a66bec6fb7f5

[^129]: https://ieeexplore.ieee.org/document/10068930/

[^130]: https://ieeexplore.ieee.org/document/9984037/

[^131]: https://ieeexplore.ieee.org/document/9797597/

[^132]: https://ieeexplore.ieee.org/document/10043078/

[^133]: https://journals.sagepub.com/doi/10.1177/13582291221083757

[^134]: https://arxiv.org/abs/2511.16483

[^135]: https://www.sciencedirect.com/science/article/abs/pii/S0957417426016544

[^136]: https://uwcscholar.uwc.ac.za/items/ca840473-8884-42ee-8603-bfcf7ed7003d

[^137]: https://www.theamericanjournals.com/index.php/tajet/article/view/7095

[^138]: https://arxiv.org/pdf/2604.08805.pdf

[^139]: https://ieeexplore.ieee.org/document/10763816/

[^140]: https://ijcaonline.org/archives/volume187/number46/autonomous-cyber-defense-agents-a-reinforcement-learning-approach-to-real-time-threat-mitigation/

[^141]: https://www.youtube.com/watch?v=HbadydHJs4I

[^142]: https://www.frontiersin.org/articles/10.3389/fcomp.2026.1803271/full

[^143]: https://ieeexplore.ieee.org/document/11359209/

[^144]: https://ieeexplore.ieee.org/document/11181093/

[^145]: https://arxiv.org/abs/2410.18332

[^146]: https://ieeexplore.ieee.org/document/10755037/

[^147]: https://www.emergentmind.com/topics/bot-iot-dataset

[^148]: https://www.unb.ca/cic/datasets/iiot-dataset-2025.html

[^149]: https://www.ibm.com/think/topics/cyber-range

[^150]: https://www.kaggle.com/datasets/primus11/unsw-nb15-dataset

[^151]: https://enfocom.com/cyber-range/

[^152]: https://pmc.ncbi.nlm.nih.gov/articles/PMC11978955/

